"""Gemma-4-31B client for the AMD Track-2 captioning agent.

Two providers behind ONE interface (call_vision / call_text), chosen by env PROVIDER
(or auto: ollama if OLLAMA_API_KEY is present, else google):
  - "ollama"  -> Ollama Cloud gemma4:31b-cloud. FAST (~1-4s) and `think:false` returns
                CLEAN output (no reasoning dump) — the dev + graded endpoint.
  - "google"  -> AI Studio generateContent (free tier; slow + always-thinking). Fallback.
All calls go through a shared retry+backoff+jitter wrapper for 429/5xx/network throttling.
Keys come from the environment first (baked into the image), then a dev ../.env.local.
"""
import json
import os
import random
import time

import requests

# ---- retry policy (shared) -------------------------------------------------
_MAX_TRIES = 6
_BASE = 2.0
_CAP = 30.0
_JITTER = 2.0
_RETRYABLE = (408, 409, 425, 429, 500, 502, 503, 504)
RETRY_STATS = {"429": 0, "500": 0, "502": 0, "503": 0, "network": 0, "total_retries": 0}


def reset_stats():
    for k in RETRY_STATS:
        RETRY_STATS[k] = 0


def _bump(key):
    RETRY_STATS[key if key in RETRY_STATS else "500"] += 1
    RETRY_STATS["total_retries"] += 1


# ---- env / key resolution (env first, then repo-root ../.env.local) --------
_env_cache = {}


def _env(name):
    if name in _env_cache:
        return _env_cache[name]
    v = os.environ.get(name)
    if not v:
        d = os.path.dirname(os.path.abspath(__file__))
        for _ in range(6):
            p = os.path.join(d, ".env.local")
            if os.path.exists(p):
                for line in open(p, encoding="utf-8"):
                    line = line.strip()
                    if line.startswith(name + "="):
                        v = line.split("=", 1)[1].strip()
                        break
            if v or d == os.path.dirname(d):
                break
            d = os.path.dirname(d)
    _env_cache[name] = v
    return v


def _provider():
    p = _env("PROVIDER")
    if p:
        return p.strip().lower()
    return "ollama" if _env("OLLAMA_API_KEY") else "google"


# ---- shared retry wrapper --------------------------------------------------
def _with_retry(do_request, timeout):
    """do_request() -> (status_or_None, text_or_None, exc_or_None). Retries retryable
    HTTP statuses and network errors; returns the parsed text or raises the last error."""
    last = None
    for attempt in range(_MAX_TRIES):
        retryable = False
        try:
            status, text, err = do_request()
            if err is None and status == 200:
                return text
            if err is not None:
                raise err
            last = RuntimeError(f"HTTP {status}: {str(text)[:200]}")
            if status in _RETRYABLE:
                _bump(str(status))
                retryable = True
        except requests.RequestException as e:
            last = e
            _bump("network")
            retryable = True
        if not retryable or attempt == _MAX_TRIES - 1:
            break
        time.sleep(min(_CAP, _BASE * (2 ** attempt)) + random.uniform(0, _JITTER))
    raise last


# ---- Ollama Cloud provider -------------------------------------------------
_OLLAMA_URL = "https://ollama.com/api/chat"


def _ollama(prompt, images_b64, timeout, temperature):
    model = _env("OLLAMA_MODEL") or "gemma4:31b-cloud"
    msg = {"role": "user", "content": prompt}
    if images_b64:
        msg["images"] = list(images_b64)
    body = {"model": model, "messages": [msg], "stream": False, "think": False}
    if temperature is not None:
        body["options"] = {"temperature": temperature}
    hdr = {"Authorization": f"Bearer {_env('OLLAMA_API_KEY')}", "Content-Type": "application/json"}

    def do():
        r = requests.post(_OLLAMA_URL, json=body, headers=hdr, timeout=timeout)
        if r.status_code != 200:
            return r.status_code, r.text, None
        content = ((r.json().get("message") or {}).get("content") or "").strip()
        if not content:
            return None, None, RuntimeError("empty ollama content: " + json.dumps(r.json())[:300])
        return 200, content, None

    return _with_retry(do, timeout)


# ---- Google AI Studio provider (fallback) ----------------------------------
_G_MODEL = "gemma-4-31b-it"  # 26b-a4b returns HTTP 500 — do not use.
_G_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{_G_MODEL}:generateContent"


def _google(prompt, images_b64, timeout, temperature):
    parts = [{"text": prompt}] + [{"inlineData": {"mimeType": "image/jpeg", "data": b}} for b in (images_b64 or [])]
    body = {"contents": [{"parts": parts}]}
    if temperature is not None:
        body["generationConfig"] = {"temperature": temperature}
    url = f"{_G_URL}?key={_env('GOOGLE_AI_STUDIO_API_KEY')}"

    def do():
        r = requests.post(url, json=body, headers={"Content-Type": "application/json"}, timeout=timeout)
        if r.status_code != 200:
            return r.status_code, r.text, None
        cands = r.json().get("candidates") or []
        if not cands:
            return None, None, RuntimeError("no candidates (safety block/empty): " + json.dumps(r.json())[:300])
        return 200, cands[0]["content"]["parts"][0]["text"], None

    return _with_retry(do, timeout)


def _dispatch(prompt, images_b64, timeout, temperature):
    return (_ollama if _provider() == "ollama" else _google)(prompt, images_b64, timeout, temperature)


# ---- public interface ------------------------------------------------------
def call_vision(prompt, frames_b64, timeout=60, temperature=None):
    """One multimodal call: text prompt + N JPEG frames (base64). Returns text."""
    return _dispatch(prompt, frames_b64, timeout, temperature)


def call_text(prompt, timeout=60, temperature=None):
    """Text-only call. Returns text."""
    return _dispatch(prompt, None, timeout, temperature)


if __name__ == "__main__":
    print("provider:", _provider())
    out = call_text("Reply with exactly the word: OK")
    print("text reply:", repr(out.strip()[:80]))
    assert out.strip(), "empty response"
    print("gemma.py self-check passed; retries:", RETRY_STATS)
