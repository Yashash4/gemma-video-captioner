"""Track-2 container entrypoint: /input/tasks.json -> /output/results.json.

Pure I/O plumbing around pipeline.caption_video (T3). The pipeline does the smart
captioning per clip; this file's whole job is to be UNKILLABLE around it:
  * every input task -> exactly one output entry (a missing task scores 0),
  * every output entry -> exactly that task's requested styles, all non-empty
    (a missing style scores 0),
  * one bad clip never sinks the run (per-task try/except -> grounded fallback),
  * a global time budget guarantees results.json is COMPLETE and on disk before
    the 10-min hard cap (partial-but-complete beats timed-out-and-empty).

Clips are processed CONCURRENTLY (MAX_CLIPS at a time) so a late clip lands seconds
into the run, not minutes — the latency gate measures each clip's completion from run
START, so serial processing would push tail clips past ~30s -> graded as the placeholder.
results.json is PRE-SEEDED with a complete fallback for every task and rewritten
atomically each time a clip upgrades its entry, so the file is always complete and every
task_id already has a valid caption on disk before its real one lands.

The fallback text is reused from pipeline._fallback (the same grounded, per-style,
never-a-fragment captions the pipeline itself emits) — never re-implemented here.
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pipeline

# defaults match the grader's mount points; env-overridable so we can test locally.
INPUT_DIR = os.environ.get("INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
# stop STARTING model work at this elapsed second, leaving tail room under the 10-min
# (600s) hard cap for the in-flight clip to finish + the file write. Checked before each
# task STARTS (not preemptive) — ponytail: a running caption_video can't be interrupted, so
# keep the budget comfortably below 600 (default 570); tune per measured per-clip time.
TIME_BUDGET_SEC = float(os.environ.get("TIME_BUDGET_SEC", "570"))
# Clips processed at once. MEASURED on 8 real clips (see report): pool=2 is the sweet spot.
#   serial : 231s total, 0x429, first clip @29s     (tail clip lands minutes in)
#   pool=2 : 144s total, ~1x429, first wave @27-30s  <-- beats the gate, 1.6x faster than serial
#   pool=3 : ~100s total, 3-13x429, first wave @35-43s (15 concurrent Ollama calls storm 429s
#            AND contend for CPU/net, pushing even the HEAD clips past 30s -> worse for the gate)
# So 2 is the largest pool whose first wave still lands < 30s. Env-overridable to retune.
MAX_CLIPS = int(os.environ.get("MAX_CLIPS", "2"))


def load_tasks(input_dir):
    """Read tasks.json; accept a bare list [...] or a wrapped {"tasks":[...]}."""
    with open(os.path.join(input_dir, "tasks.json"), encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("tasks", [])
    return [t for t in data if isinstance(t, dict)] if isinstance(data, list) else []


def fallback_captions(styles):
    """Fastest possible complete answer: pipeline's grounded per-style fallback with no
    facts (no API call). Reused, not duplicated — this is the exact text the pipeline
    emits when the model never returned a valid caption for a style."""
    return {s: pipeline._fallback(s, {}) for s in styles}


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
        return {"task_id": result.get("task_id", task_id), "captions": caps}
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


def _seed(task):
    """A COMPLETE fallback entry for a task (no model call) — what sits on disk until the
    real caption lands, and what stays if the clip fails or the budget is spent."""
    return {"task_id": task.get("task_id"), "captions": fallback_captions(list(task.get("styles") or []))}


def run(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, budget=TIME_BUDGET_SEC, max_clips=MAX_CLIPS):
    """Caption every task CONCURRENTLY, always producing a complete /output/results.json.

    results.json is pre-seeded with a fallback for every task and rewritten atomically as
    each clip upgrades its entry, so it is complete at all times. Clips run max_clips at a
    time. The finally-write guarantees the file exists even if loading/looping throws.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    try:
        tasks = load_tasks(input_dir)
        results = [_seed(t) for t in tasks]          # complete-from-first-write
        lock = threading.Lock()

        def flush():
            with lock:
                write_results(output_dir, results)   # atomic; serialized across worker threads

        flush()                                       # every task_id valid on disk before any model call
        start = time.monotonic()

        def work(i_task):
            i, task = i_task
            # budget check happens as this clip is about to START (not preemptive); if spent,
            # leave the pre-seeded fallback and never call the model.
            if time.monotonic() - start >= budget:
                return
            entry = caption_task(task)                # calls pipeline.caption_video; degrades to fallback
            results[i] = entry
            flush()                                   # upgrade this clip's entry as soon as it's ready

        if tasks:
            with ThreadPoolExecutor(max_workers=max(1, min(max_clips, len(tasks)))) as ex:
                list(ex.map(work, enumerate(tasks)))
    finally:
        path = write_results(output_dir, results)     # always leave a complete file
    return results, path


if __name__ == "__main__":
    results, path = run()
    print(f"[agent] wrote {len(results)} result(s) -> {path}")
