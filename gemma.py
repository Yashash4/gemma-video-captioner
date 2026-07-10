"""Gemma-4-31B client for the AMD Track-2 captioning agent.

One vision call + text calls through a shared _post() with retry+backoff+jitter
on the free tier's 503/429/500 throttling. Key comes from the environment first
(baked into the Docker image), falling back to a local ../.env.local for dev.
"""
import json
import os
import random
import time

import requests

MODEL = "gemma-4-31b-it"  # 26b-a4b returns HTTP 500 — do not use.
_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
_HDR = {"Content-Type": "application/json"}

# retry policy: 6 tries, 2s base, exp backoff capped at 30s + up to 2s jitter.
# (free tier throws sporadic 500s that clear on retry — a 6th shot buys margin.)
_MAX_TRIES = 6
_BASE = 2.0
_CAP = 30.0
_JITTER = 2.0
_RETRYABLE = (429, 500, 502, 503)

# thread-safe-enough counters for the spike report (GIL-protected int += is fine here).
RETRY_STATS = {"429": 0, "500": 0, "502": 0, "503": 0, "network": 0, "total_retries": 0}


def reset_stats():
    for k in RETRY_STATS:
        RETRY_STATS[k] = 0


def _bump(key):
    RETRY_STATS[key] += 1
    RETRY_STATS["total_retries"] += 1


_key_cache = None


def _get_key():
    global _key_cache
    if _key_cache:
        return _key_cache
    k = os.environ.get("GOOGLE_AI_STUDIO_API_KEY")
    if not k:
        # dev fallback: walk up from this file looking for .env.local (lives in repo root, outside code/)
        d = os.path.dirname(os.path.abspath(__file__))
        for _ in range(6):
            p = os.path.join(d, ".env.local")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GOOGLE_AI_STUDIO_API_KEY="):
                            k = line.split("=", 1)[1].strip()
                            break
            if k or d == os.path.dirname(d):
                break
            d = os.path.dirname(d)
    if not k:
        raise RuntimeError("GOOGLE_AI_STUDIO_API_KEY not set and no .env.local found")
    _key_cache = k
    return k


def _post(parts, timeout=30):
    """POST one generateContent call; retry retryable statuses / network errors.

    Raises the last error after _MAX_TRIES (caller decides whether to degrade).
    """
    url = f"{_ENDPOINT}?key={_get_key()}"
    body = {"contents": [{"parts": parts}]}
    last = None
    for attempt in range(_MAX_TRIES):
        retryable = False
        try:
            r = requests.post(url, json=body, headers=_HDR, timeout=timeout)
            if r.status_code == 200:
                cands = r.json().get("candidates") or []
                if not cands:
                    # safety block / empty completion — surface it, don't loop.
                    raise RuntimeError("no candidates (safety block/empty): " + json.dumps(r.json())[:300])
                return cands[0]["content"]["parts"][0]["text"]
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            if r.status_code in _RETRYABLE:
                _bump(str(r.status_code) if str(r.status_code) in RETRY_STATS else "500")
                retryable = True
        except requests.RequestException as e:
            last = e
            _bump("network")
            retryable = True
        if not retryable or attempt == _MAX_TRIES - 1:
            break
        time.sleep(min(_CAP, _BASE * (2 ** attempt)) + random.uniform(0, _JITTER))
    raise last


def call_vision(prompt, frames_b64, timeout=30):
    """One multimodal call: text prompt + N JPEG frames (base64). Returns text."""
    parts = [{"text": prompt}]
    parts += [{"inlineData": {"mimeType": "image/jpeg", "data": b}} for b in frames_b64]
    return _post(parts, timeout=timeout)


def call_text(prompt, timeout=30):
    """Text-only call. Returns text."""
    return _post([{"text": prompt}], timeout=timeout)


if __name__ == "__main__":
    # self-check: the key resolves and a trivial text call comes back non-empty.
    out = call_text("Reply with exactly the word: OK")
    print("text reply:", repr(out.strip()[:80]))
    assert out.strip(), "empty response"
    print("gemma.py self-check passed; retries:", RETRY_STATS)
