"""Track-2 container entrypoint: /input/tasks.json -> /output/results.json.

Pure I/O plumbing around pipeline.caption_video (T3). The pipeline does the smart
captioning per clip; this file's whole job is to be UNKILLABLE around it:
  * every input task -> exactly one output entry (a missing task scores 0),
  * every output entry -> exactly that task's requested styles, all non-empty
    (a missing style scores 0),
  * one bad clip never sinks the run (per-task try/except -> grounded fallback),
  * a global time budget guarantees results.json is COMPLETE and on disk before
    the 10-min hard cap (partial-but-complete beats timed-out-and-empty).

The fallback text is reused from pipeline._fallback (the same grounded, per-style,
never-a-fragment captions the pipeline itself emits) — never re-implemented here.
"""
import json
import os
import sys
import time

import pipeline

# defaults match the grader's mount points; env-overridable so we can test locally.
INPUT_DIR = os.environ.get("INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
# stop STARTING model work at this elapsed second, leaving tail room under the 10-min
# (600s) hard cap for the in-flight clip to finish + the file write. Checked between
# tasks (not preemptive) — ponytail: a running caption_video can't be interrupted, so
# keep the budget comfortably below 600 (default 570); tune per measured per-clip time.
TIME_BUDGET_SEC = float(os.environ.get("TIME_BUDGET_SEC", "570"))


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


def run(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, budget=TIME_BUDGET_SEC):
    """Caption every task, always producing a complete /output/results.json.

    Serial by design (free-tier rate limits — see HANDOFF 'serial Track-2'). The
    finally-write guarantees the file exists even if loading/looping throws.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    try:
        tasks = load_tasks(input_dir)
        start = time.monotonic()
        for task in tasks:
            if time.monotonic() - start >= budget:
                # budget spent: fill the rest with the fastest fallback, no model calls,
                # so the file is COMPLETE (every task, every style) before the hard cap.
                styles = list(task.get("styles") or [])
                results.append({"task_id": task.get("task_id"),
                                "captions": fallback_captions(styles)})
            else:
                results.append(caption_task(task))
    finally:
        path = write_results(output_dir, results)
    return results, path


if __name__ == "__main__":
    results, path = run()
    print(f"[agent] wrote {len(results)} result(s) -> {path}")
