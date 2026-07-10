"""T1 smoke test / driver — de-risks the #1 unknown: can the free-tier Gemma path
sustain 1 vision + 4 concurrent text calls per clip inside the 30s/clip budget?

For each sample task: local cached clip -> 10 frames -> Stage A grounding vision call
-> Stage B 4 styles (concurrent). Measures wall-time splits + throttle retries, and
compares concurrent vs serial Stage B on one clip. Writes scene JSON + captions to
spike_out/. __main__ asserts every clip produced a non-empty scene JSON and 4
non-empty, mutually-distinct captions.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import gemma
from frames import extract_frames, ffmpeg_source

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TASKS = os.path.join(ROOT, "samples", "input", "tasks.json")
CLIPS = os.path.join(ROOT, "samples", "clips")
OUT = os.path.join(HERE, "spike_out")

# generous per-call timeout so we MEASURE real (possibly throttled ~45s) latency
# instead of cutting slow-but-successful calls short. Container tuning happens later.
VISION_TIMEOUT = 90
TEXT_TIMEOUT = 60

# Stage A grounding prompt (51 §5.1) — structured + hedged, forbids guessing.
GROUNDING_PROMPT = """You see FRAMES sampled from ONE video. Describe ONLY what is actually visible.
If unsure, mark it uncertain — never guess. Return JSON only, no prose:
{
  "subjects": [],        // people/animals/objects clearly present
  "actions": [],         // what they do, visible motion
  "setting": "",         // place/environment/time of day if visible
  "on_screen_text": [],  // legible text/logos, else []
  "audio_summary": "no clear speech",
  "mood": "",            // overall vibe
  "uncertain": []        // things hinted but not confirmable
}
Do NOT infer brand names, locations, or identities that aren't clearly shown.
Output ONLY the JSON object: start with { and end with }. No prose, no markdown, no reasoning."""

# Basic style prompts for the spike (tuned templates are T2). Shared preamble protects accuracy.
# NOTE: gemma-4-31b-it loves to "think out loud" (drafts, word counts, "Final Selection:") —
# the last two lines below fight that; clean() extracts the final line as a backstop.
PREAMBLE = ("You write ONE English caption for a video, given a verified JSON description of what it "
            "contains. Use ONLY facts in the JSON; do not invent anything; if something is marked "
            "uncertain, do not assert it. Every claim must stay literally TRUE — the style is in tone "
            "and framing, never in fabricated events. 1-2 sentences, ~12-40 words. No hashtags, emojis, "
            "quotes, or preamble.\n"
            "CRITICAL: Do NOT show reasoning, drafts, alternatives, bullet points, or word counts. "
            "Output only the final caption, on a single line, prefixed exactly with 'CAPTION: '.")
STYLES = {
    "formal": "STYLE = FORMAL. Neutral, precise, documentary register, third person, no contractions, no jokes.",
    "sarcastic": "STYLE = SARCASTIC. Dry, ironic, deadpan, mock-unimpressed. Facts stay true; sarcasm is in attitude, not inverting what happened.",
    "humorous_tech": "STYLE = HUMOROUS (TECH). Witty using software/engineering metaphors (APIs, latency, bugs, deploys, caching) regardless of subject. Facts stay true.",
    "humorous_non_tech": "STYLE = HUMOROUS (EVERYDAY). Warm, punny, relatable, like a funny friend. NO tech/software jargon at all. Facts stay true.",
}


def style_prompt(style, scene_str):
    return f"{PREAMBLE}\n{STYLES[style]}\nJSON: {scene_str}"


def parse_scene(text):
    """Tolerant JSON extraction: first '{' .. last '}'. Falls back to wrapping raw text."""
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(text[s:e + 1])
        except Exception:
            pass
    return {"raw": text.strip()}


_FINAL_MARKERS = ("final result:", "final selection:", "final polish:", "final answer:", "final:")


def clean(cap):
    """Extract the caption. Primary: text after the last 'CAPTION:' sentinel (deterministic).
    Backstops for when gemma ignores the sentinel and dumps reasoning. ponytail: the real
    fix (few-shot / firmer templates so it never reasons) is T2's job."""
    cap = cap.strip()
    low = cap.lower()
    k = low.rfind("caption:")
    if k != -1:
        cap = cap[k + len("caption:"):]                  # deterministic: after the last sentinel
    else:
        idx = max((low.rfind(m) for m in _FINAL_MARKERS), default=-1)
        if idx != -1:
            cap = cap[idx:].split(":", 1)[1]             # after the last "Final...:" marker
        elif "* " in cap or "Draft" in cap or len(cap) > 400:
            chunks = [c.strip() for c in cap.replace("\n", " ").split(". ") if c.strip()]
            cap = chunks[-1] if chunks else cap          # last sentence of the dump
    cap = cap.split("\n", 1)[0]                           # caption is the first line; drop trailing reasoning
    for tell in (" check ", " let's ", " wait", " word count", " - "):  # ponytail: T2 kills this at the source
        j = cap.lower().find(tell)
        if j > 20:
            cap = cap[:j]
    cap = " ".join(cap.split())
    return cap.strip().strip('"').strip("*").strip()


def stage_b(scene_str, styles, concurrent=True):
    if concurrent:
        out = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(gemma.call_text, style_prompt(s, scene_str), TEXT_TIMEOUT): s for s in styles}
            for f in as_completed(futs):
                s = futs[f]
                try:
                    out[s] = clean(f.result())
                except Exception as ex_:
                    out[s] = f"[ERROR: {ex_}]"
        return out
    return {s: clean(gemma.call_text(style_prompt(s, scene_str), TEXT_TIMEOUT)) for s in styles}


def run_clip(task):
    tid = task["task_id"]
    styles = task["styles"]
    clip = os.path.join(CLIPS, f"{tid}.mp4")
    print(f"\n=== {tid} ===")

    t0 = time.perf_counter()
    frames = extract_frames(clip, n=10)
    t_frames = time.perf_counter() - t0
    print(f"  frames: {len(frames)} in {t_frames:.1f}s ({ffmpeg_source()})")

    r0 = gemma.RETRY_STATS["total_retries"]
    t0 = time.perf_counter()
    try:
        scene_raw = gemma.call_vision(GROUNDING_PROMPT, frames, timeout=VISION_TIMEOUT)
        scene = parse_scene(scene_raw)
    except Exception as e:  # never let one clip's throttle kill the batch (partial > zero)
        scene = {"error": f"vision failed after retries: {e}"}
    t_a = time.perf_counter() - t0
    scene_str = json.dumps(scene, ensure_ascii=False)
    print(f"  stage A (vision): {t_a:.1f}s, {len(scene_str)} chars JSON")

    t0 = time.perf_counter()
    caps = stage_b(scene_str, styles, concurrent=True)
    t_b = time.perf_counter() - t0
    retries = gemma.RETRY_STATS["total_retries"] - r0
    print(f"  stage B (4 concurrent): {t_b:.1f}s | throttle-retries this clip: {retries}")

    json.dump({"task_id": tid, "scene": scene, "captions": caps},
              open(os.path.join(OUT, f"{tid}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return {"tid": tid, "t_frames": t_frames, "t_a": t_a, "t_b": t_b,
            "retries": retries, "scene": scene, "caps": caps, "scene_str": scene_str}


def main():
    os.makedirs(OUT, exist_ok=True)
    tasks = json.load(open(TASKS, encoding="utf-8"))
    print(f"ffmpeg backend: {ffmpeg_source()} | tasks: {len(tasks)}")

    # --- download-path check: pull ONE clip from its real video_url (not the cache) ---
    print("\n[download check] extracting 2 frames straight from v1's video_url ...")
    t0 = time.perf_counter()
    dl = extract_frames(tasks[0]["video_url"], n=2)
    print(f"  download+extract OK: {len(dl)} frames in {time.perf_counter() - t0:.1f}s")
    assert dl and all(dl), "download path failed — got no frames from video_url"

    gemma.reset_stats()
    results = [run_clip(t) for t in tasks]

    # --- concurrency vs serial baseline on ONE clip (v1) ---
    v1 = results[0]
    print("\n[concurrency test] re-running v1 Stage B serially for a baseline ...")
    r0 = gemma.RETRY_STATS["total_retries"]
    t0 = time.perf_counter()
    try:
        stage_b(v1["scene_str"], tasks[0]["styles"], concurrent=False)
        t_serial = time.perf_counter() - t0
        serial_ok = True
    except Exception as e:
        t_serial = time.perf_counter() - t0
        serial_ok = False
        print(f"  serial baseline failed (throttle/500): {e}")
    serial_retries = gemma.RETRY_STATS["total_retries"] - r0
    speedup = (t_serial / v1["t_b"]) if (v1["t_b"] and serial_ok) else float("nan")

    # --- report ---
    total = sum(r["t_frames"] + r["t_a"] + r["t_b"] for r in results)
    print("\n" + "=" * 60 + "\nREPORT\n" + "=" * 60)
    print(f"{'clip':6} {'frames':>7} {'stageA':>7} {'stageB':>7} {'total':>7} {'retries':>8}")
    for r in results:
        print(f"{r['tid']:6} {r['t_frames']:7.1f} {r['t_a']:7.1f} {r['t_b']:7.1f} "
              f"{r['t_frames'] + r['t_a'] + r['t_b']:7.1f} {r['retries']:8d}")
    print(f"\n3-clip wall-time (sequential): {total:.1f}s | per-clip avg: {total / len(results):.1f}s")
    print(f"throttle retries total: {gemma.RETRY_STATS}")
    if serial_ok:
        print(f"\nv1 Stage B — concurrent: {v1['t_b']:.1f}s ({v1['retries']} retries) | "
              f"serial: {t_serial:.1f}s ({serial_retries} retries) | serial/concurrent = {speedup:.2f}x")
        concl = ("concurrency HELPED, no extra throttling" if speedup > 1.3 and v1["retries"] <= serial_retries
                 else "concurrency gave little/no speedup — free tier is serializing us (throttle-bound)")
        print(f"verdict: {concl}")
    else:
        print(f"\nv1 Stage B — concurrent: {v1['t_b']:.1f}s ({v1['retries']} retries) | "
              f"serial baseline FAILED after {t_serial:.1f}s ({serial_retries} retries) — "
              f"free tier too throttled to complete 4 serial calls back-to-back")
    # 12-clip projection (sequential per-clip; orchestrator can overlap clips later)
    proj = total / len(results) * 12
    print(f"\n12-clip projection @ this per-clip rate: {proj:.0f}s total "
          f"({'WITHIN' if proj < 600 else 'OVER'} 10min budget), "
          f"per-clip {total / len(results):.1f}s ({'WITHIN' if total / len(results) < 30 else 'OVER'} 30s/clip)")

    # --- self-check ---
    for r in results:
        assert "error" not in r["scene"], f"{r['tid']}: Stage A failed -> {r['scene'].get('error')}"
        assert r["scene"] and json.dumps(r["scene"]).strip("{}\" "), f"{r['tid']}: empty scene JSON"
        vals = list(r["caps"].values())
        assert len(vals) == 4 and all(v.strip() for v in vals), f"{r['tid']}: missing/empty caption"
        assert all(not v.startswith("[ERROR") for v in vals), f"{r['tid']}: a style call errored"
        assert len(set(vals)) == 4, f"{r['tid']}: captions not mutually distinct"
    print("\nself-check PASSED: every clip has a non-empty scene JSON + 4 distinct non-empty captions")


if __name__ == "__main__":
    main()
