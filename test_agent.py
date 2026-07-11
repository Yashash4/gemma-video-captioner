"""Offline self-check for agent.py — validates the whole I/O contract in seconds with
a MOCKED pipeline (no real Gemma API, no rate-limit contention).

Proves the four things T3 exists to guarantee:
  1. results.json is written, is a list, one entry per input task.
  2. every entry has task_id + a captions dict whose keys are EXACTLY that task's styles.
  3. a clip whose pipeline call RAISES is still present with all styles non-empty (fallback).
  4. when the time budget is exhausted, output is still complete and the model is NOT called.

Run:  python test_agent.py   (from the code/ directory)
"""
import json
import os
import tempfile

import agent
import pipeline

# Tasks with DELIBERATELY different style sets to prove styles are mirrored dynamically
# (never the hardcoded 4): A has all four, B has two, C (which will fail) has three.
TASKS = [
    {"task_id": "A", "video_url": "x", "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]},
    {"task_id": "B", "video_url": "x", "styles": ["formal", "sarcastic"]},
    {"task_id": "C", "video_url": "x", "styles": ["formal", "humorous_tech", "humorous_non_tech"]},
]


def _write_input(input_dir):
    with open(os.path.join(input_dir, "tasks.json"), "w", encoding="utf-8") as f:
        json.dump(TASKS, f)


def fake_caption(task):
    """Canned success for A/B; RAISES for C to exercise the per-task fallback path."""
    if task["task_id"] == "C":
        raise RuntimeError("simulated per-clip failure")
    return {"task_id": task["task_id"],
            "captions": {s: f"canned {s} caption describing the {task['task_id']} scene"
                         for s in task["styles"]}}


def must_not_call(task):
    raise AssertionError("pipeline.caption_video was called after the budget was exhausted")


def _assert_complete(results, tasks):
    """Shared checks: one entry per task, exact-style mirror, every caption non-empty."""
    assert isinstance(results, list), "results.json is not a JSON list"
    assert len(results) == len(tasks), f"expected {len(tasks)} entries, got {len(results)}"
    by_id = {r["task_id"]: r for r in results}
    for t in tasks:
        r = by_id[t["task_id"]]
        assert "captions" in r and isinstance(r["captions"], dict), f"{t['task_id']}: bad captions"
        assert set(r["captions"]) == set(t["styles"]), (
            f"{t['task_id']}: styles {set(r['captions'])} != requested {set(t['styles'])}")
        for s, cap in r["captions"].items():
            assert isinstance(cap, str) and cap.strip(), f"{t['task_id']}/{s}: empty caption"


def main():
    orig = pipeline.caption_video
    try:
        # --- normal + per-task-failure path (assertions 1, 2, 3) ---
        pipeline.caption_video = fake_caption
        with tempfile.TemporaryDirectory() as d:
            in_dir, out_dir = os.path.join(d, "in"), os.path.join(d, "out")
            os.makedirs(in_dir)
            _write_input(in_dir)
            agent.run(in_dir, out_dir, budget=570)

            # assertion 1: file written and is a list of the right length
            with open(os.path.join(out_dir, "results.json"), encoding="utf-8") as f:
                results = json.load(f)
            _assert_complete(results, TASKS)  # covers 1 (list+count) and 2 (exact styles)

            # assertion 3: the clip that raised (C) is present with all styles non-empty
            c = next(r for r in results if r["task_id"] == "C")
            assert set(c["captions"]) == set(TASKS[2]["styles"])
            assert all(v.strip() for v in c["captions"].values())
            # and it's the real pipeline fallback text, not the canned string
            assert not any("canned" in v for v in c["captions"].values())
            print("PASS 1-3: complete output, exact-style mirror, per-task fallback fired")

        # --- budget-exhausted path (assertion 4) ---
        pipeline.caption_video = must_not_call  # would raise if the guard let it through
        with tempfile.TemporaryDirectory() as d:
            in_dir, out_dir = os.path.join(d, "in"), os.path.join(d, "out")
            os.makedirs(in_dir)
            _write_input(in_dir)
            agent.run(in_dir, out_dir, budget=0.0)  # zero budget => everything falls back

            with open(os.path.join(out_dir, "results.json"), encoding="utf-8") as f:
                results = json.load(f)
            _assert_complete(results, TASKS)  # still complete without any model call
            print("PASS 4: budget-exhausted output is complete and the model was not called")
    finally:
        pipeline.caption_video = orig

    print("\nall self-check assertions passed.")
    # show a small real sample of the emitted schema (from the fallback path)
    print("\nsample results.json entry (fallback for task C):")
    print(json.dumps(c, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
