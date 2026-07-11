"""Local eval harness for the Track-2 captioning agent — scores our captions the way
the leaderboard's LLM-judge does, so we can tune toward >=0.91 without burning submissions.

SPLIT CROSS-FAMILY JUDGE (never judge Gemma with Gemma — self-preference inflates scores,
docs/52 §5). Each score goes to the judge best suited to it:
  - accuracy   (0-1): MULTIMODAL judge that SEES the same frames the captioner saw, so
       accuracy is grounded not blind. Primary = Gemini via GOOGLE_AI_STUDIO_API_KEY
       (gemini-2.5-flash; true cross-family). Fallback = gemma3:27b on Ollama (multimodal;
       Gemma-3 is a DIFFERENT generation from our gemma4 captioner -> low but non-zero
       self-bias, flagged in the header).
  - style_match(0-1) + separation: TEXT-ONLY, where cross-family matters most. Judge =
       gpt-oss:120b on Ollama (genuinely cross-family). Anchored with 1 exemplar/style.
  - separation (0-1): hand the text judge the 4 captions UNLABELED -> it assigns each to a
       style -> fraction correct. Bias-resistant; catches humorous_tech<->non_tech collapse.

The tier-locked Ollama judges (gemini-3-flash-preview, glm-5.2, deepseek-v4-pro, kimi-k2.6)
all 403 "requires subscription" on this key, hence this accessible split.

Re-runnable: fixed frames (n=10, same as the pipeline) + temp-0 judges, averaged over
EVAL_JUDGE_RUNS runs to cut wobble. Tolerant JSON parsing reuses pipeline._parse_json.
Judge rate-limits are counted SEPARATELY from gemma.RETRY_STATS, so the latter measures
ONLY the Ollama captioning burst — the number we need for the graded-run decision.
"""
import json
import os
import random
import sys
import time

import requests

import gemma
import pipeline
from frames import extract_frames

STYLES4 = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
N_FRAMES = 10                                   # == pipeline default; fixed => comparable runs
JUDGE_RUNS = int(os.environ.get("EVAL_JUDGE_RUNS", "3"))  # docs/52 §5: run Nx temp 0, average
JUDGE_TIMEOUT = 120
_GEMINI_MODELS = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash"]
_OLLAMA_TEXT_MODELS = ["gpt-oss:120b", "minimax-m3", "nemotron-3-super"]  # cross-family, accessible
_OLLAMA_VISION_FALLBACK = "gemma3:27b"          # accessible multimodal fallback for accuracy
_OLLAMA_URL = "https://ollama.com/api/chat"

# judge-side rate-limit counters, kept SEPARATE from gemma.RETRY_STATS.
JUDGE_STATS = {"429": 0, "5xx": 0, "network": 0}

# short, judge-side style anchors (independent of the captioner's exemplars on purpose:
# a judge that rewards echoing the captioner's own exemplar is miscalibrated).
_STYLE_ANCHORS = {
    "formal": 'neutral documentary register, third person, no jokes/opinion. '
              'e.g. "A kayaker paddles through choppy rapids as spray breaks over the bow."',
    "sarcastic": 'dry, deadpan, mock-unimpressed — facts stay true, attitude is ironic. '
                 'e.g. "A cat knocks a cup off a table. Riveting cinema, truly."',
    "humorous_tech": 'playful humor built from SOFTWARE/ENGINEERING metaphors (APIs, latency, '
                     'bugs, deploys, caching). e.g. "The dog fetches the ball with sub-100ms '
                     'latency and zero dropped packets."',
    "humorous_non_tech": 'playful, warm, everyday-life humor — ABSOLUTELY NO tech/software jargon. '
                         'e.g. "This dog treats fetch like an Olympic sport nobody told it about."',
}


# ---------------------------------------------------------------------------
# raw judge transports (Gemini via Google key; Ollama /api/chat)
# ---------------------------------------------------------------------------
def _gemini_call(model, prompt, frames_b64, temperature):
    """One Gemini generateContent call, retry/backoff on 429/5xx/network. Raises on exhaust."""
    key = gemma._env("GOOGLE_AI_STUDIO_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    parts = [{"text": prompt}] + [{"inlineData": {"mimeType": "image/jpeg", "data": b}}
                                  for b in (frames_b64 or [])]
    body = {"contents": [{"parts": parts}], "generationConfig": {"temperature": temperature}}
    return _post_with_retry(url, body, None, lambda j: j["candidates"][0]["content"]["parts"][0]["text"])


def _ollama_call(model, prompt, frames_b64, temperature):
    """One Ollama /api/chat call (think:false), retry/backoff. Raises on exhaust."""
    msg = {"role": "user", "content": prompt}
    if frames_b64:
        msg["images"] = list(frames_b64)
    body = {"model": model, "messages": [msg], "stream": False, "think": False,
            "options": {"temperature": temperature}}
    hdr = {"Authorization": f"Bearer {gemma._env('OLLAMA_API_KEY')}"}
    return _post_with_retry(_OLLAMA_URL, body, hdr, lambda j: (j.get("message") or {}).get("content") or "")


def _post_with_retry(url, body, headers, extract):
    """Shared POST+retry for both judge transports. `extract(json)->text`. Counts JUDGE_STATS."""
    last = None
    for attempt in range(5):
        try:
            r = requests.post(url, json=body, headers=headers, timeout=JUDGE_TIMEOUT)
            if r.status_code == 200:
                txt = (extract(r.json()) or "").strip()
                if txt:
                    return txt
                last = RuntimeError("empty judge response")
            else:
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:150]}")
                if r.status_code == 429:
                    JUDGE_STATS["429"] += 1
                elif r.status_code >= 500:
                    JUDGE_STATS["5xx"] += 1
                else:
                    break  # 4xx (bad model/tier-locked) — retrying this model is pointless
        except requests.RequestException as e:
            last = e
            JUDGE_STATS["network"] += 1
        except (KeyError, IndexError, ValueError) as e:
            last = e  # malformed/blocked body — one retry may help
        time.sleep(min(20, 2 * (2 ** attempt)) + random.uniform(0, 1.5))
    raise last


# ---------------------------------------------------------------------------
# judge selection (probe once, cache) + dispatch
# ---------------------------------------------------------------------------
_VISION = None   # ("gemini", model, cross_family_label) | ("ollama", "gemma3:27b", label)
_TEXT = None     # ("ollama", model, label)


def _select_vision_judge():
    global _VISION
    if _VISION:
        return _VISION
    # EVAL_VISION=ollama forces the sustainable gemma3:27b judge (Gemini free-tier quota is tiny)
    if os.environ.get("EVAL_VISION", "").lower() != "ollama" and gemma._env("GOOGLE_AI_STUDIO_API_KEY"):
        for m in _GEMINI_MODELS:
            try:
                _gemini_call(m, "Reply with exactly: OK", None, 0)
                _VISION = ("gemini", m, "CROSS-FAMILY")
                return _VISION
            except Exception:
                continue
    _VISION = ("ollama", _OLLAMA_VISION_FALLBACK, "Gemma-3 (diff generation from gemma4 — low self-bias)")
    return _VISION


def _select_text_judge():
    global _TEXT
    if _TEXT:
        return _TEXT
    for m in _OLLAMA_TEXT_MODELS:
        try:
            _ollama_call(m, "Reply with exactly: OK", None, 0)
            _TEXT = ("ollama", m, "CROSS-FAMILY")
            return _TEXT
        except Exception:
            continue
    raise RuntimeError("no accessible text judge among " + ", ".join(_OLLAMA_TEXT_MODELS))


def _vision_judge(prompt, frames_b64, temperature=0):
    global _VISION
    backend, model, _ = _select_vision_judge()
    try:
        return (_gemini_call if backend == "gemini" else _ollama_call)(model, prompt, frames_b64, temperature)
    except Exception:
        if backend == "gemini":  # quota exhausted mid-run -> switch to gemma3:27b for the rest, don't crash
            _VISION = ("ollama", _OLLAMA_VISION_FALLBACK, "gemma3:27b fallback (gemini exhausted)")
            return _ollama_call(_OLLAMA_VISION_FALLBACK, prompt, frames_b64, temperature)
        raise


def _text_judge(prompt, temperature=0):
    _, model, _ = _select_text_judge()
    return _ollama_call(model, prompt, None, temperature)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def _clip01(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _accuracy_prompt(caps):
    listed = "\n".join(f'{s}: "{c}"' for s, c in caps.items())
    keys = ", ".join(f'"{s}":{{"accuracy":<0-1>,"reason":"<short>"}}' for s in caps)
    return f"""You are a strict, fair judge scoring AI-generated video captions for ACCURACY only.
You are shown {N_FRAMES} frames sampled from ONE video, then captions written about it.
For EACH caption, score accuracy from 0.0 to 1.0: does it correctly describe what is
ACTUALLY VISIBLE in the frames? Penalize hallucinated subjects/actions/objects/on-screen
text, wrong subjects, and invented brands/places/numbers/purposes. Vague-but-not-wrong
scores in the middle; asserting things not supported by the frames scores low. Ignore
tone/humor/style — judge ONLY factual grounding.

CAPTIONS:
{listed}

Return ONLY one JSON object, no prose: {{{keys}}}"""


def _style_prompt(caps):
    anchors = "\n".join(f"- {s}: {_STYLE_ANCHORS.get(s, s)}" for s in caps)
    listed = "\n".join(f'{s}: "{c}"' for s, c in caps.items())
    keys = ", ".join(f'"{s}":{{"style_match":<0-1>,"reason":"<short>"}}' for s in caps)
    return f"""You are a strict, fair judge scoring AI-generated captions for STYLE MATCH only.
Each caption below is labeled with the TARGET STYLE it was written in. For EACH, score
style_match from 0.0 to 1.0: does it read as a native, well-executed example of that style
per the anchors? A flat/generic caption in a style that should be sarcastic or humorous
scores low even if factually fine. humorous_tech and humorous_non_tech MUST be clearly
distinct — if a "humorous_non_tech" caption uses software/engineering jargon, score it low;
if a "humorous_tech" caption has no tech metaphor, score it low. Ignore factual accuracy here.

STYLE ANCHORS:
{anchors}

CAPTIONS (labeled with their target style):
{listed}

Return ONLY one JSON object, no prose: {{{keys}}}"""


def score_accuracy(frames_b64, caps):
    """MULTIMODAL judge sees frames. Returns {style:{accuracy, reason}}, averaged over runs."""
    acc = {s: [] for s in caps}
    reason = {s: "" for s in caps}
    prompt = _accuracy_prompt(caps)
    for run in range(JUDGE_RUNS):
        res = pipeline._parse_json(_vision_judge(prompt, frames_b64, 0)) or {}
        for s in caps:
            v = res.get(s) or {}
            if "accuracy" in v:
                acc[s].append(_clip01(v.get("accuracy")))
            if run == 0 and v.get("reason"):
                reason[s] = str(v["reason"])[:200]
    return {s: {"accuracy": _mean(acc[s]), "reason": reason[s]} for s in caps}


def score_style(caps):
    """TEXT judge (cross-family). Returns {style:{style_match, reason}}, averaged over runs."""
    sm = {s: [] for s in caps}
    reason = {s: "" for s in caps}
    prompt = _style_prompt(caps)
    for run in range(JUDGE_RUNS):
        res = pipeline._parse_json(_text_judge(prompt, 0)) or {}
        for s in caps:
            v = res.get(s) or {}
            if "style_match" in v:
                sm[s].append(_clip01(v.get("style_match")))
            if run == 0 and v.get("reason"):
                reason[s] = str(v["reason"])[:200]
    return {s: {"style_match": _mean(sm[s]), "reason": reason[s]} for s in caps}


def separation(caps):
    """Bias-resistant: TEXT judge gets the captions UNLABELED (deterministically shuffled)
    and assigns each to a style. Returns (fraction_correct, [(true, guess, ok), ...])."""
    items = list(caps.items())
    seed = int("".join(filter(str.isdigit, "".join(caps.values()))) or 0) & 0xffff
    order = items[:]
    random.Random(seed).shuffle(order)
    listing = "\n".join(f'Caption {i+1}: "{c}"' for i, (_, c) in enumerate(order))
    prompt = f"""Here are {len(order)} captions for the SAME video, each written in a DIFFERENT one
of these styles, each style used EXACTLY once:
- formal: neutral documentary register, no jokes.
- sarcastic: dry, deadpan, ironic attitude.
- humorous_tech: humor from software/engineering metaphors (APIs, latency, bugs, deploys).
- humorous_non_tech: playful everyday-life humor with NO tech/software jargon.

Assign each caption to the style it best matches. Use each style EXACTLY once.
{listing}

Return ONLY JSON mapping caption number to style: {{{", ".join(f'"{i+1}":"<style>"' for i in range(len(order)))}}}"""
    res = pipeline._parse_json(_text_judge(prompt, 0)) or {}
    correct, assigned = 0, []
    for i, (true_style, _) in enumerate(order):
        guess = str(res.get(str(i + 1), "")).strip().lower()
        ok = guess == true_style
        correct += ok
        assigned.append((true_style, guess, ok))
    return correct / max(len(order), 1), assigned


# ---------------------------------------------------------------------------
# per-clip eval
# ---------------------------------------------------------------------------
def eval_clip(clip_id, path, styles):
    frames = extract_frames(path, n=N_FRAMES)          # fixed frames => re-runnable
    r0 = gemma.RETRY_STATS["total_retries"]
    t0 = time.perf_counter()
    caps = pipeline.caption_video({"task_id": clip_id, "styles": styles, "video_url": path},
                                  frames_b64=frames)["captions"]
    cap_dt = time.perf_counter() - t0
    accs = score_accuracy(frames, caps)
    stys = score_style(caps)
    sep, assigned = separation(caps)
    scores = {s: {"accuracy": accs[s]["accuracy"], "style_match": stys[s]["style_match"],
                  "acc_reason": accs[s]["reason"], "style_reason": stys[s]["reason"]} for s in styles}
    return {
        "clip": clip_id, "captions": caps, "scores": scores, "separation": sep,
        "assigned": assigned, "cap_time": cap_dt,
        "pipe_retries": gemma.RETRY_STATS["total_retries"] - r0,
        "acc": _mean([scores[s]["accuracy"] for s in styles]),
        "sm": _mean([scores[s]["style_match"] for s in styles]),
    }


def _fmt_row(rec):
    return (f"  {rec['clip']:<14} acc={rec['acc']:.2f}  style={rec['sm']:.2f}  "
            f"sep={rec['separation']:.2f}  proxy={(rec['acc']+rec['sm'])/2:.2f}  "
            f"| cap={rec['cap_time']:4.0f}s pipe_retries={rec['pipe_retries']}")


def run(clips):
    gemma.reset_stats()
    vb, vm, vlabel = _select_vision_judge()
    tb, tm, tlabel = _select_text_judge()
    print(f"ACCURACY judge : {vb}/{vm}  [{vlabel}]  (sees frames)")
    print(f"STYLE+SEP judge: {tb}/{tm}  [{tlabel}]  (text-only)")
    print(f"JUDGE_RUNS={JUDGE_RUNS}  N_FRAMES={N_FRAMES}\n")
    recs, weakest = [], []
    t0 = time.perf_counter()
    for clip_id, path in clips:
        rec = eval_clip(clip_id, path, STYLES4)
        recs.append(rec)
        print(_fmt_row(rec))
        for s in STYLES4:
            sc = rec["scores"][s]
            combined = (sc["accuracy"] + sc["style_match"]) / 2
            why = sc["acc_reason"] if sc["accuracy"] <= sc["style_match"] else sc["style_reason"]
            weakest.append((combined, rec["clip"], s, sc["accuracy"], sc["style_match"], why))
    wall = time.perf_counter() - t0

    macc = _mean([r["acc"] for r in recs])
    msm = _mean([r["sm"] for r in recs])
    msep = _mean([r["separation"] for r in recs])
    proxy = (macc + msm) / 2

    print("\n" + "=" * 74)
    print(f"OVERALL  ({len(recs)} clips)   mean_accuracy={macc:.3f}   mean_style_match={msm:.3f}")
    print(f"         separation_accuracy={msep:.3f}   LEADERBOARD_PROXY (mean acc,style)={proxy:.3f}")
    verdict = "ABOVE" if proxy >= 0.90 else "BELOW"
    print(f"         --> {verdict} the 0.90 board wall ({'>=0.91 crown' if proxy >= 0.91 else 'under 0.91 crown'})")

    print("\nWEAKEST (clip, style) pairs — tune these:")
    for combined, clip, s, a, m, why in sorted(weakest)[:3]:
        print(f"  [{clip} / {s}]  acc={a:.2f} style={m:.2f}  proxy={combined:.2f}  :: {why or '(no reason)'}")

    print(f"\nWALL TIME: {wall:.0f}s total  ({wall/max(len(recs),1):.0f}s/clip)")
    print(f"CAPTIONING (Ollama gemma4) retries during run: {gemma.RETRY_STATS}")
    print(f"JUDGE rate-limit hits (Gemini + Ollama judges): {JUDGE_STATS}")
    return recs


# ---------------------------------------------------------------------------
# self-check: the judge must discriminate — a wrong caption scores accuracy < 0.3
# ---------------------------------------------------------------------------
def self_check(frames_b64):
    wrong = {"formal": "A rocket launches into space, leaving a trail of fire against a starry sky."}
    a = score_accuracy(frames_b64, wrong)["formal"]["accuracy"]
    ok = a < 0.30
    print(f"self-check: deliberately-wrong caption scored accuracy={a:.2f} "
          f"(must be < 0.30) -> {'PASS' if ok else 'FAIL — judge miscalibrated'}")
    assert ok, f"judge failed to reject a wrong caption (accuracy={a:.2f}); prompt is miscalibrated"
    return a


BUCKET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples", "bucket_clips")
_ALL = {  # bucket_clips filenames keyed by Pexels id
    "12471596": "12471596-uhd_2560_1440_30fps.mp4", "34796505": "34796505-hd_1920_1080_60fps.mp4",
    "31948459": "31948459-hd_1920_1080_24fps.mp4", "2697636": "2697636-uhd_1920_1440_30fps.mp4",
    "5404543": "5404543-hd_1920_1080_24fps.mp4", "12122308": "12122308-uhd_2560_1440_24fps.mp4",
    "7963468": "7963468-uhd_2560_1440_25fps.mp4", "8533913": "8533913-uhd_2560_1440_25fps.mp4",
    "5054248": "5054248-uhd_2560_1440_25fps.mp4", "6023186": "6023186-hd_1920_1080_30fps.mp4",
    "3121327": "3121327-uhd_2560_1440_24fps.mp4", "11785757": "11785757-hd_1920_1080_30fps.mp4",
}
# 5 DIVERSE validation clips (big UHD / 60fps / tiny / mid) — T5 spec.
_DIVERSE = ["12471596", "34796505", "31948459", "2697636", "5404543"]


if __name__ == "__main__":
    ids = sys.argv[1:] or _DIVERSE
    if ids == ["all"]:
        ids = list(_ALL)
    clips = [(cid, os.path.join(BUCKET, _ALL[cid])) for cid in ids]

    _select_vision_judge()
    _select_text_judge()
    # self-check first, on the smallest clip's frames (fast decode)
    self_check(extract_frames(os.path.join(BUCKET, _ALL["31948459"]), n=N_FRAMES))
    print()
    run(clips)
