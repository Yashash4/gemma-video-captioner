"""Track-2 container entrypoint: /input/tasks.json -> /output/results.json.

Pure I/O plumbing around pipeline.caption_video. This file's whole job is to be
UNKILLABLE around the pipeline:
  * every input task -> exactly one output entry (a missing task scores 0),
  * every output entry -> exactly that task's requested styles, all non-empty
    (a missing style scores 0),
  * one bad clip never sinks the run (per-task try/except -> grounded fallback),
  * results.json is PRE-SEEDED with a complete grounded fallback for every task
    BEFORE any model call, then each clip atomically UPGRADES its entry as it
    finishes. So a COMPLETE, valid file is on disk from t=0 -- immune to a kill /
    timeout / hang mid-run (which would otherwise leave no output = "could not be
    scored"). A global time budget stops starting new clips before the 10-min cap.

The fallback text is reused from pipeline._fallback (the same grounded, per-style,
never-a-fragment captions the pipeline itself emits) -- never re-implemented here.

Optional behavior telemetry: if the TELEMETRY_URL env is set, run/clip events
(task_id, timing, fallback count, our captions -- NO video content) are POSTed
fire-and-forget so it can NEVER slow or break the run, and it is completely inert
when the env is unset. Diagnostics only.
"""
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pipeline

# defaults match the grader's mount points; env-overridable so we can test locally.
INPUT_DIR = os.environ.get("INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
# stop STARTING model work at this elapsed second, leaving tail room under the 10-min
# (600s) hard cap for in-flight clips to finish + the file write. Checked before a clip
# STARTS (not preemptive). With the pre-seed, even a hard kill leaves a complete file.
# 500 (not 570): extra tail room for up to MAX_CLIPS clips still in flight under congestion.
TIME_BUDGET_SEC = float(os.environ.get("TIME_BUDGET_SEC", "500"))
# clip-level concurrency. 2 clips x up to 4 concurrent style calls = ~8 concurrent Ollama
# calls (gemma has bounded retry/backoff). Do NOT raise: 3+ storms 429s.
MAX_CLIPS = int(os.environ.get("MAX_CLIPS", "2"))
# Optional, inert unless set. Behavior/diagnostics sink; never affects the run.
_TELEMETRY_URL = os.environ.get("TELEMETRY_URL", "").strip()


def _telemetry(event, **data):
    """Fire-and-forget POST of a behavior event. No-op when TELEMETRY_URL is unset;
    swallows every error and runs on a daemon thread so it can never slow/break the run."""
    if not _TELEMETRY_URL:
        return
    payload = json.dumps({"event": event, **data}).encode("utf-8")

    def _post():
        try:
            req = urllib.request.Request(_TELEMETRY_URL, data=payload,
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=3).read()
        except Exception:
            pass  # telemetry must never affect the run

    threading.Thread(target=_post, daemon=True).start()


def load_tasks(input_dir):
    """Read tasks.json; accept a bare list [...] or a wrapped {"tasks":[...]}.

    utf-8-sig strips a leading BOM (which would make json.load raise -> empty results.json
    -> every task scores 0). Styles are normalized once here: a bare string becomes [string]
    and non-string/empty elements are dropped, so downstream never char-splats a string
    (for s in "formal" -> 'f','o',...) or crashes on a non-string style."""
    with open(os.path.join(input_dir, "tasks.json"), encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("tasks", [])
    if not isinstance(data, list):
        return []
    tasks = []
    for t in data:
        if not isinstance(t, dict):
            continue
        styles = t.get("styles")
        if isinstance(styles, str):
            styles = [styles]
        elif not isinstance(styles, list):
            styles = []
        t["styles"] = [s for s in styles if isinstance(s, str) and s.strip()]
        tasks.append(t)
    return tasks


def fallback_captions(styles):
    """Fastest possible complete answer: pipeline's grounded per-style fallback with no
    facts (no API call). Reused, not duplicated -- the exact text the pipeline emits when
    the model never returned a valid caption for a style."""
    return {s: pipeline._fallback(s, {}) for s in styles}


def _seed(task):
    """A COMPLETE fallback entry for a task (no model call) -- what sits on disk until the
    real caption lands, and what stays if the clip fails or the budget is spent."""
    return {"task_id": task.get("task_id"), "captions": fallback_captions(list(task.get("styles") or []))}


def caption_task(task):
    """One task -> {"task_id", "captions"} with EXACTLY task['styles'], all non-empty.
    Wraps the pipeline; any failure degrades to the grounded fallback for that clip."""
    styles = list(task.get("styles") or [])
    task_id = task.get("task_id")
    try:
        result = pipeline.caption_video(task)
        caps = result.get("captions") or {}
        # mirror the requested styles dynamically; belt-and-suspenders fill any the
        # pipeline somehow left empty (it shouldn't) so no style is ever missing/blank.
        caps = {s: (caps.get(s) or pipeline._fallback(s, {})) for s in styles}
        out = {"task_id": result.get("task_id", task_id), "captions": caps}
        if result.get("_diag") is not None:
            out["_diag"] = result["_diag"]  # per-clip diagnostics (concurrency-safe, unlike last_stats); stripped before write
        return out
    except Exception as e:  # one clip must never sink the run
        print(f"[agent] task {task_id!r} failed ({e!r}); using fallback", file=sys.stderr)
        return {"task_id": task_id, "captions": fallback_captions(styles)}


def write_results(output_dir, results):
    """Atomic-ish write: full JSON to a temp file, fsync, then os.replace into place so
    /output/results.json is never seen half-written and always parses."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def run(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, budget=TIME_BUDGET_SEC, max_clips=MAX_CLIPS):
    """Caption every task, up to `max_clips` clips CONCURRENTLY (so a 15-clip grader run fits
    under the 10-min cap even when Ollama is congested), always producing a complete
    /output/results.json. Pre-seeded with a grounded fallback for every task before any model
    call, then each clip atomically upgrades its entry UNDER A LOCK -- so the file is complete
    at all times and a kill/timeout/hang can never leave 'no output'. A per-clip budget check
    skips STARTING a clip once the budget is spent (it keeps its pre-seeded fallback). The
    finally-write is a final belt-and-suspenders."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    try:
        tasks = load_tasks(input_dir)
        results = [_seed(t) for t in tasks]        # complete-from-first-write
        write_results(output_dir, results)          # every task_id valid on disk BEFORE any model call
        _telemetry("run_start", n_tasks=len(tasks))
        start = time.monotonic()
        lock = threading.Lock()                      # guards the shared results[] + results.json write (same tmp path)

        def work(i, task):
            if time.monotonic() - start >= budget:
                # budget spent before this clip STARTED: keep its pre-seeded fallback, no model call.
                _telemetry("budget_spent", at_clip=i, n_tasks=len(tasks),
                           elapsed=round(time.monotonic() - start, 1))
                return
            t0 = time.monotonic()
            entry = caption_task(task)               # calls pipeline.caption_video; degrades to fallback
            diag = entry.pop("_diag", None)          # strip diagnostics: results.json schema stays {task_id, captions}
            with lock:                               # atomic: upgrade this clip's entry + flush the file
                results[i] = entry
                write_results(output_dir, results)
            d = diag or {}                           # _diag missing on the fallback path -> log what we have
            _telemetry("clip_done", task_id=task.get("task_id"), i=i, n=len(tasks),
                       elapsed=round(time.monotonic() - t0, 1),
                       fallbacks=d.get("fallbacks"), calls=d.get("calls"),
                       n_frames=d.get("n_frames"), facts=d.get("facts"),
                       captions=entry.get("captions"))

        with ThreadPoolExecutor(max_workers=max(1, max_clips)) as ex:
            list(ex.map(work, range(len(tasks)), tasks))  # list() forces completion + surfaces any error
        _telemetry("run_end", n=len(results), total=round(time.monotonic() - start, 1))
    except Exception as e:
        _telemetry("run_error", error=str(e)[:300])
        raise
    finally:
        path = write_results(output_dir, results)    # always leave a complete file
    return results, path


if __name__ == "__main__":
    # boot ping FIRST: proves the container started and shows its environment. If this
    # never arrives, the container didn't start (or outbound is blocked). main_done vs
    # main_crash vs (nothing after run_start) tells us exactly where a run dies.
    _telemetry("boot",
               version=os.environ.get("APP_VERSION", "?"),  # baked at build (--build-arg APP_VERSION=v8);
               provider=os.environ.get("PROVIDER", ""),      # tells us WHICH version the grader actually ran
               has_ollama_key=bool(os.environ.get("OLLAMA_API_KEY")),
               has_google_key=bool(os.environ.get("GOOGLE_AI_STUDIO_API_KEY")),
               input_exists=os.path.exists(os.path.join(INPUT_DIR, "tasks.json")),
               output_dir=OUTPUT_DIR, python=sys.version.split()[0])
    try:
        results, path = run()
        print(f"[agent] wrote {len(results)} result(s) -> {path}")
        _telemetry("main_done", n=len(results), path=path)
    except Exception as e:
        _telemetry("main_crash", error=repr(e)[:300])
        raise
