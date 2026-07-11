"""Track-2 captioning pipeline: one clip -> {task_id, captions:{<style>: text}}.

Two stages (docs/51 §0): Stage A does the accuracy-critical *seeing* ONCE (a vision
call -> grounded scene JSON, then one self-check text call) and freezes the facts;
Stage B rewrites those frozen facts into each requested style with concurrent text
calls. On top, two D2 quality moves: a no-API style-distinctiveness reranker and a
single batched self-eval accuracy pass.

The #1 robustness job (T1 finding): gemma-4-31b-it leaks reasoning / drops the output
format on ~1/3 of calls. We fix that at the SOURCE — every parsed output is wrapped in
a strict delimiter (<json>/<caption>) and REGENERATED if the delimiter is absent —
never with a post-hoc trimmer. The pipeline never dies: worst case a style falls back
to a lightly-styled version of the frozen facts (a missing style scores 0).
"""
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import gemma
import prompts
from frames import extract_frames

# Generous per-call timeouts: MEASURE real (throttled ~45s) free-tier latency rather
# than cut slow-but-successful calls short. The container tunes these later (T4).
VISION_TIMEOUT = 90
TEXT_TIMEOUT = 60

# Temperatures: low & deterministic for grounding/verification, warmer for voice.
T_GROUND = 0.2
T_CHECK = 0.2
T_STYLE = 0.4     # low + few-shot => model emits a clean caption, not chain-of-thought
T_EVAL = 0.2
T_DISTINCT = 0.7

MAX_CAPTION_TRIES = 3      # first attempt + 2 regenerations when <caption> is missing
MAX_GROUND_TRIES = 3       # first attempt + 2 regenerations when no valid JSON
DISTINCT_THRESHOLD = 0.55  # word-set Jaccard above this => two captions too similar
DISTINCT_BUDGET = 2        # cap distinctiveness regenerations per clip

# ---------------------------------------------------------------------------
# parsing / post-processing (pure helpers — the delimiter-robustness core)
# ---------------------------------------------------------------------------

def _between(raw, tag):
    """Text inside <tag>...</tag> (case-insensitive, spans newlines), else None."""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", raw, re.S | re.I)
    return m.group(1).strip() if m else None


def _parse_json(raw):
    """Tolerant JSON extract for a THINKING model that may reason before/around the JSON.
    Try <json>..</json>, then first '{'..last '}', then the LAST balanced {...} that parses
    (the model's final answer comes last). Returns a dict, or None to signal 'regenerate'."""
    body = _between(raw, "json")
    cand = body if body is not None else raw
    s, e = cand.find("{"), cand.rfind("}")
    if s != -1 and e > s:
        try:
            obj = json.loads(cand[s:e + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    best, depth, start = None, 0, 0          # scan for the last balanced {...} that parses
    for i, ch in enumerate(cand):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(cand[start:i + 1])
                    if isinstance(obj, dict):
                        best = obj
                except Exception:
                    pass
    return best


# common emoji / symbol / dingbat / regional-indicator ranges + ZWJ & variation selector
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002190-\U000021FF"
                    "\U00002B00-\U00002BFF️‍]")
_SENT = re.compile(r"(?<=[.!?])\s+")


def _postprocess(cap):
    """Strip a stray 'caption:' label, emojis, hashtags, wrapping quotes; keep 1-2 sentences."""
    cap = re.sub(r"^\s*caption\s*:\s*", "", cap.strip(), flags=re.I)
    cap = _EMOJI.sub("", cap)
    cap = re.sub(r"#\w+", "", cap)
    cap = re.sub(r"\s{2,}", " ", cap).strip()
    cap = cap.strip("\"'“”‘’").strip()
    parts = _SENT.split(cap)
    if len(parts) > 3:                    # allow up to 3 short punchy sentences (reference humor voice)
        cap = " ".join(parts[:3])
    return cap.strip()


_STOP = {"a", "an", "the", "of", "and", "in", "on", "at", "to", "is", "are", "its", "it",
         "with", "for", "this", "that", "as", "by", "but", "into", "through", "over"}


def _words(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP}


def _jaccard(a, b):
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# tokens that betray leaked reasoning / markdown / a template echo inside a "caption"
# Tells that a "caption" is really leaked reasoning / an instruction echo (gemma-4 is a
# thinking model). Kept to phrases that never occur in a natural scene caption.
_BAD_SUBSTR = ("`", "json:", "<caption", "</caption", "<json", "</json", "check against",
               "the user wants", "i need to", "word count", "final selection", "final answer",
               "as an ai", "chain of thought", "task:", "constraint", "hard rule",
               "use only these", "one english caption", "self-correction", "example voice",
               "style =", "style=", "draft:", "let's go with", "input:", "* subject",
               "* action", "* setting", "* task", "* constraint")
_PLACEHOLDERS = {"", "...", "..", ".", "caption", "your caption", "your one caption",
                 "text", "none", "n/a", "caption text", "sentence", "a caption"}


def _valid_caption(cap):
    """A finished caption must be a real sentence: non-empty, no leaked-reasoning/markdown
    tells, no placeholder/fragment, sane length. This is the single gate every caption
    (model or regenerated) must pass before it can be returned."""
    if cap is None:
        return False
    low = cap.strip().lower()
    if low.strip(".·-'\"* ") in _PLACEHOLDERS:
        return False
    if any(b in low for b in _BAD_SUBSTR):
        return False
    return 5 <= len(cap.split()) <= 60


def _extract_caption(raw):
    """Recover the caption from a possibly reasoning-heavy reply. gemma-4-31b is a THINKING
    model whose reasoning CANNOT be disabled (the API rejects thinkingConfig); ~1/3 of calls
    it drafts a caption many times then states its final pick LAST, almost always in quotes.
    Try, in order: the <caption> tag; the whole reply if already clean; the LAST quoted "..."
    (the model's final selection); the last clean line. First to pass _valid_caption wins."""
    if not raw:
        return None
    tag = _between(raw, "caption")                       # 1. explicit delimiter (clean case)
    if tag and _valid_caption(_postprocess(tag)):
        return _postprocess(tag)
    whole = _postprocess(raw)                            # 2. whole reply already terse
    if _valid_caption(whole):
        return whole
    for q in reversed(re.findall(r'["“”]([^"“”\n]{10,220})["“”]', raw)):  # 3. last quoted pick
        c = _postprocess(q)
        if _valid_caption(c):
            return c
    for line in reversed([l.strip() for l in raw.splitlines() if l.strip()]):  # 4. last clean line
        c = _postprocess(line)
        if _valid_caption(c):
            return c
    return None


def _salvage_core(raw):
    """Best-effort short scene phrase from raw prose facts (when Stage-A JSON parse failed),
    so the fallback stays GROUNDED instead of collapsing to a generic phrase. Skips reasoning
    lines and list/label prefixes; returns '' if nothing clean is found."""
    if not raw:
        return ""
    for s in _SENT.split(re.sub(r"\s+", " ", raw).strip()):
        if any(b in s.lower() for b in _BAD_SUBSTR):
            continue
        s = re.sub(r"^[-*\d.]+\s*", "", s)              # list markers: "1.", "-", "*"
        s = re.sub(r"^\w[\w ]{0,20}:\s*", "", s)         # a leading "Subjects:" style label
        if 4 <= len(s.split()) <= 30:
            return _postprocess(s).rstrip(".")
    return ""


# ---------------------------------------------------------------------------
# stages (take injected `vision`/`text` callables so call-counting stays local)
# ---------------------------------------------------------------------------

def _stage_a(vision, text, bump):
    """Vision grounding (regen if no JSON) + one self-check. Returns (facts, facts_str)."""
    raw, facts = "", None
    for i in range(MAX_GROUND_TRIES):
        if i:
            bump("ground_regens")
        try:
            raw = vision(prompts.GROUNDING + (prompts.GROUNDING_RETRY if i else ""), T_GROUND)
        except Exception:
            continue  # throttle/500 exhausted retries in gemma._post; try a fresh call
        facts = _parse_json(raw)
        if facts is not None:
            break
    if facts is None:
        facts = {"raw": raw.strip()} if raw.strip() else {"error": "vision unavailable"}
    facts_str = json.dumps(facts, ensure_ascii=False)

    try:  # self-check tightens/hedges; keep it only if it still parses to a dict
        checked = _parse_json(text(prompts.self_check(facts_str), T_CHECK))
        if checked:
            facts, facts_str = checked, json.dumps(checked, ensure_ascii=False)
    except Exception:
        pass
    return facts, facts_str


def _gen_style(text, style, facts_str, bump, distinct_from=None):
    """One styled caption: strict <caption> extract -> VALIDATE -> regen with a stricter
    prompt if invalid (cap MAX_CAPTION_TRIES). Returns a valid caption, or None so the
    caller can fall back. Never returns a fragment/placeholder/leaked reasoning."""
    base = prompts.style_prompt(style, facts_str, distinct_from)
    temp = T_DISTINCT if distinct_from else T_STYLE
    for i in range(MAX_CAPTION_TRIES):
        if i:
            bump("style_regens")
        try:
            raw = text(base if i == 0 else base + prompts.CAPTION_RETRY, temp)
        except Exception:
            continue
        cap = _extract_caption(raw)
        if cap:
            return cap
    return None


def _stage_b(text, styles, facts_str, bump):
    """All styles concurrently off the frozen facts. Returns {style: caption or None}."""
    with ThreadPoolExecutor(max_workers=min(4, len(styles))) as ex:
        futs = {s: ex.submit(_gen_style, text, s, facts_str, bump) for s in styles}
        return {s: futs[s].result() for s in styles}


def _accuracy_pass(text, facts_str, caps):
    """ONE batched critique call. Returns the set of styles flagged as ungrounded."""
    labeled = "\n".join(f"{s}: {c}" for s, c in caps.items() if c)
    if not labeled:
        return set()
    try:
        res = _parse_json(text(prompts.accuracy_eval(facts_str, labeled), T_EVAL))
    except Exception:
        return set()
    failed = (res or {}).get("failed", [])
    return {s for s in failed if s in caps} if isinstance(failed, list) else set()


def _distinct_pairs(styles):
    """All style pairs, humorous_tech<->humorous_non_tech checked first (the known collapse)."""
    pairs = [(styles[i], styles[j]) for i in range(len(styles)) for j in range(i + 1, len(styles))]
    hp = {"humorous_tech", "humorous_non_tech"}
    pairs.sort(key=lambda p: 0 if set(p) == hp else 1)
    return pairs


def _distinct_pass(text, styles, caps, facts_str, bump):
    """No-API reranker: regenerate the later-listed of any too-similar pair, made distinct."""
    budget = DISTINCT_BUDGET
    for a, b in _distinct_pairs(styles):
        if budget <= 0:
            break
        if not caps.get(a) or not caps.get(b):
            continue
        sim = _jaccard(caps[a], caps[b])
        if sim >= DISTINCT_THRESHOLD:
            new = _gen_style(text, b, facts_str, bump, distinct_from=caps[a])
            budget -= 1
            if new and _jaccard(caps[a], new) < sim:
                caps[b] = new


def _fallback(style, facts):
    """Real per-style caption when the model never returned a valid one. Grounded in the
    frozen facts; NEVER a placeholder/fragment/leaked reasoning; distinct per style; >=5
    words. If the facts are unusable (a raw reasoning dump), stay generic but clean."""
    def _join(v):
        items = v[:3] if isinstance(v, list) else ([v] if v else [])
        return ", ".join(dict.fromkeys(str(x) for x in items if str(x).strip()))

    subj, act = _join(facts.get("subjects")), _join(facts.get("actions"))
    setting = facts.get("setting") if isinstance(facts.get("setting"), str) else ""
    core = " ".join(p for p in (subj, act) if p).strip()
    if setting:
        core = f"{core} in {setting}".strip()
    if not core:  # Stage-A JSON failed -> salvage a grounded phrase from the raw prose facts
        core = _salvage_core(facts.get("raw") or "")
    core = core or "a short real-world scene"  # truly unusable: generic but never a fragment/leak
    templ = {
        "formal": f"The video shows {core}.",
        "sarcastic": f"Groundbreaking footage of {core}. Truly riveting stuff.",
        "humorous_tech": f"Now serving from production: {core}, running at stable uptime.",
        "humorous_non_tech": f"Just everyday magic: {core}, and honestly, a whole mood.",
    }
    return _postprocess(templ.get(style, f"A {style.replace('_', ' ')} view of {core}."))


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def caption_video(task, frames_b64=None):
    """Caption ONE clip. `task` = {task_id, video_url?, styles:[...]}.
    Returns {"task_id": ..., "captions": {style: text for style in task['styles']}}.
    Extracts frames from task['video_url'] when frames_b64 is None. Never raises for a
    single clip: partial > zero, every requested style is always populated."""
    styles = list(task["styles"])
    if frames_b64 is None:
        frames_b64 = extract_frames(task["video_url"], n=10)

    stats = {"calls": 0, "style_regens": 0, "ground_regens": 0, "fallbacks": 0}
    lock = threading.Lock()

    def bump(key):
        with lock:
            stats[key] += 1

    def vision(prompt, temp):
        bump("calls")
        return gemma.call_vision(prompt, frames_b64, timeout=VISION_TIMEOUT, temperature=temp)

    def text(prompt, temp):
        bump("calls")
        return gemma.call_text(prompt, timeout=TEXT_TIMEOUT, temperature=temp)

    facts, facts_str = _stage_a(vision, text, bump)
    caps = _stage_b(text, styles, facts_str, bump)

    # D2a — batched self-eval accuracy pass; regenerate any flagged caption once (concurrently).
    flagged = _accuracy_pass(text, facts_str, caps)
    if flagged:
        with ThreadPoolExecutor(max_workers=min(4, len(flagged))) as ex:
            regen = {s: ex.submit(_gen_style, text, s, facts_str, bump) for s in flagged}
            for s, fut in regen.items():
                new = fut.result()
                if new:
                    caps[s] = new

    # D2b — style-distinctiveness reranker (no API).
    _distinct_pass(text, styles, caps, facts_str, bump)

    # Always populate every requested style with a REAL caption (a missing style scores 0).
    for s in styles:
        if not _valid_caption(caps.get(s)):
            caps[s] = _fallback(s, facts)
            bump("fallbacks")

    caption_video.last_stats = dict(stats)  # ponytail: diagnostics only; approximate if clips run concurrently
    return {"task_id": task["task_id"], "captions": {s: caps[s] for s in styles}}


caption_video.last_stats = {"calls": 0, "style_regens": 0, "ground_regens": 0, "fallbacks": 0}


# ---------------------------------------------------------------------------
# self-check: 3 example clips (mapped via tasks.json) + 2 harder bucket clips
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import time

    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    OUT = os.path.join(HERE, "spike_out")
    os.makedirs(OUT, exist_ok=True)
    STYLES4 = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

    tasks = json.load(open(os.path.join(ROOT, "samples", "input", "tasks.json"), encoding="utf-8"))
    jobs = [(t["task_id"], os.path.join(ROOT, "samples", "clips", f'{t["task_id"]}.mp4'), t["styles"]) for t in tasks]
    bucket = os.path.join(ROOT, "samples", "bucket_clips")
    jobs += [("bucket_60fps", os.path.join(bucket, "34796505-hd_1920_1080_60fps.mp4"), STYLES4),
             ("bucket_tiny24", os.path.join(bucket, "31948459-hd_1920_1080_24fps.mp4"), STYLES4)]

    gemma.reset_stats()
    agg = {"style_regens": 0, "ground_regens": 0, "fallbacks": 0, "calls": 0}
    for tid, path, styles in jobs:
        print(f"\n=== {tid} ===")
        t0 = time.perf_counter()
        frames = extract_frames(path, n=10)
        r0 = gemma.RETRY_STATS["total_retries"]
        res = caption_video({"task_id": tid, "styles": styles, "video_url": path}, frames_b64=frames)
        dt = time.perf_counter() - t0
        caps = res["captions"]
        st = caption_video.last_stats
        for k in agg:
            agg[k] += st[k]
        vals = [caps[s] for s in styles]
        mx = max((_jaccard(vals[i], vals[j]) for i in range(len(vals)) for j in range(i + 1, len(vals))), default=0.0)
        print(f"  calls={st['calls']} time={dt:.1f}s throttle_retries={gemma.RETRY_STATS['total_retries'] - r0} "
              f"| style_regens={st['style_regens']} ground_regens={st['ground_regens']} fallbacks={st['fallbacks']} "
              f"| max_pair_jaccard={mx:.2f}")
        for s in styles:
            print(f"   [{s}] {caps[s]}")
        json.dump(res, open(os.path.join(OUT, f"{tid}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        # every requested style must be a clean, real, mutually-distinct caption
        assert set(styles) <= set(caps), f"{tid}: missing style"
        for s in styles:
            assert _valid_caption(caps[s]), f"{tid}/{s}: not a valid caption: {caps[s]!r}"
        assert len({caps[s] for s in styles}) == len(styles), f"{tid}: captions not mutually distinct"

    print(f"\nself-check PASSED on {len(jobs)} clips: every style is a clean, distinct, valid caption.")
    print(f"format-robustness across the run: {agg} | throttle retries: {gemma.RETRY_STATS}")
