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


def _ollama(prompt, images_b64, timeout, temperature, system=None):
    model = _env("OLLAMA_MODEL") or "gemma4:31b-cloud"
    msg = {"role": "user", "content": prompt}
    if images_b64:
        msg["images"] = list(images_b64)
    messages = [{"role": "system", "content": system}, msg] if system else [msg]
    # Gemma-4's calibrated sampling (top_k 64 / top_p 0.95) — sent ALWAYS, else Ollama
    # silently uses its own 40/0.9. num_ctx 8192: 10 frames (~2800 img tokens) + the long
    # prompt exceed Ollama's default 4096, which would SILENTLY truncate frames.
    options = {"top_k": 64, "top_p": 0.95, "num_ctx": 8192}
    if temperature is not None:
        options["temperature"] = temperature
    body = {"model": model, "messages": messages, "stream": False, "think": False, "options": options}
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


def _google(prompt, images_b64, timeout, temperature, system=None):
    # AI Studio's Gemma endpoint rejects systemInstruction, so fold the persistent rules
    # into the prompt text (fallback path only; Ollama is what ships).
    if system:
        prompt = system + "\n\n" + prompt
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


# ---- Fireworks AI provider (dedicated on-demand; no rate limits) -----------
# OpenAI-compatible endpoint. Used when PROVIDER=fireworks. FIREWORKS_MODEL is the
# deployment-specific model string (baked at build once the on-demand deployment exists):
#   accounts/fireworks/models/gemma-4-31b-it-nvfp4#accounts/<acct>/deployments/<id>
_FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"


def _fireworks(prompt, images_b64, timeout, temperature, system=None):
    model = _env("FIREWORKS_MODEL") or "accounts/fireworks/models/gemma-4-31b-it-nvfp4"
    content = [{"type": "text", "text": prompt}]
    for b in (images_b64 or []):
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    # Disable Gemma-4's thinking (the Fireworks analog of Ollama's think:false). Without it the
    # model dumps chain-of-thought into reasoning_content and returns content=null on harder
    # prompts -> "empty content" -> fallback. enable_thinking=False keeps content populated;
    # reasoning_effort=none is belt-and-suspenders in case the template kwarg is ignored.
    body = {"model": model, "messages": messages, "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}
    if temperature is not None:
        body["temperature"] = temperature
    hdr = {"Authorization": f"Bearer {_env('FIREWORKS_API_KEY')}", "Content-Type": "application/json"}

    def do():
        r = requests.post(_FIREWORKS_URL, json=body, headers=hdr, timeout=timeout)
        if r.status_code != 200:
            return r.status_code, r.text, None
        choices = r.json().get("choices") or []
        if not choices:
            return None, None, RuntimeError("no choices: " + json.dumps(r.json())[:300])
        text = ((choices[0].get("message") or {}).get("content") or "").strip()
        if not text:
            return None, None, RuntimeError("empty fireworks content: " + json.dumps(r.json())[:300])
        return 200, text, None

    return _with_retry(do, timeout)


_PROVIDERS = {"ollama": _ollama, "google": _google, "fireworks": _fireworks}


def _dispatch(prompt, images_b64, timeout, temperature, system=None):
    return _PROVIDERS.get(_provider(), _google)(prompt, images_b64, timeout, temperature, system)


# ---- public interface ------------------------------------------------------
def call_vision(prompt, frames_b64, timeout=60, temperature=None, system=None):
    """One multimodal call: text prompt + N JPEG frames (base64). Returns text.
    `system` (Gemma-4 native system role) carries the persistent rules."""
    return _dispatch(prompt, frames_b64, timeout, temperature, system)


def call_text(prompt, timeout=60, temperature=None, system=None):
    """Text-only call. Returns text. `system` = persistent-rules system message."""
    return _dispatch(prompt, None, timeout, temperature, system)


if __name__ == "__main__":
    print("provider:", _provider())
    out = call_text("Reply with exactly the word: OK")
    print("text reply:", repr(out.strip()[:80]))
    assert out.strip(), "empty response"
    # system role must be accepted (Gemma-4 native system support)
    sysout = call_text("What word did your instructions tell you to reply with?",
                       system="Always reply with exactly one word: PONG")
    print("system reply:", repr(sysout.strip()[:80]))
    assert sysout.strip(), "empty response with system message"
    print("gemma.py self-check passed; retries:", RETRY_STATS)
