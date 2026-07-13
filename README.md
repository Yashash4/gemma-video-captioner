# Gemma Video Captioner

A Google Gemma-4-31B agent that captions any video in four styles (`formal`, `sarcastic`, `humorous_tech`, `humorous_non_tech`), grounded so the humor stays true to what is on screen.

## See once, style four times

The core idea is that the accuracy-critical work of *seeing* happens exactly once. A single vision pass reads the frames and produces a frozen set of scene facts. The four style passes then rewrite those same frozen facts, each in its own voice. Because all four captions are built from one shared visual read, they cannot contradict each other, and a joke can only reframe a real detail, never invent one.

## How it works (pipeline)

The pipeline (`pipeline.py`) turns one clip into `{task_id, captions:{style: text}}` in two stages plus two quality passes.

**Stage A, see it once.** One Gemma-4-31B vision call runs over about ten sampled frames and returns a grounded scene-fact JSON: subjects (with colors, counts, and distinctive features), actions, setting, any clearly legible on-screen text read verbatim, spatial layout, mood, and an `uncertain` list for anything blurry, distant, tiny, or guessed. Anything not confirmable is pushed into `uncertain` rather than asserted. If the model fails to return parseable JSON, the vision call is regenerated with a stricter instruction (up to three attempts). The resulting facts are then frozen and reused for everything downstream.

**Stage B, style it four times.** Four Gemma-4-31B text calls run concurrently, each rewriting the same frozen facts into one requested style. Each caption is pulled out of a strict `<caption>` delimiter and validated (real sentence, no leaked reasoning, no placeholder, sane length); an invalid reply is regenerated with a stricter prompt, up to three attempts.

**Self-eval accuracy pass.** One batched Gemma-4-31B call reviews all four captions against the frozen facts and flags any caption that asserts a literal detail not in the facts (jokes, sarcasm, and clearly figurative metaphors are explicitly not flagged). Each flagged style is regenerated, grounded and concurrently.

**Distinctiveness reranker (no API for detection).** A word-set Jaccard similarity check finds any two captions that are too similar. The similarity detection uses no model call. When a too-similar pair is found, the later-listed caption is regenerated with a Gemma text call that asks for a different vocabulary, so the four voices stay distinct.

Every requested style is always populated. When a call never returns a valid caption, that style degrades to a grounded per-style fallback built from the frozen facts, never a fragment, a placeholder, or a zero.

## Model and providers

Gemma-4-31B is load-bearing: every scene fact and every caption comes from Gemma-4-31B. There is no second model doing the real work.

`gemma.py` exposes one interface (`call_vision` / `call_text`) over a pluggable provider chosen by the `PROVIDER` environment variable. The submitted image sets `PROVIDER=fireworks` and calls Gemma-4-31B served on Fireworks AI, a dedicated on-demand deployment reached through an OpenAI-compatible chat completions endpoint that accepts image inputs as base64 data URLs. Two alternate providers are implemented in the same file: Google AI Studio (`google`, its `generateContent` endpoint) and Ollama Cloud (`ollama`, `gemma4:31b-cloud`). The `ollama` provider additionally sends Gemma-4's calibrated sampling values (`top_k=64`, `top_p=0.95`); the shipped Fireworks path does not set those and relies on temperature alone. All providers that support it use Gemma-4's native `system` role for the persistent grounding rules; the Google path folds those rules into the prompt text because its Gemma endpoint rejects a system instruction.

## Frame sampling

`frames.py` turns a clip into base64 JPEG frames with ffmpeg (the system binary when present, otherwise the bundled `imageio-ffmpeg` binary, so a missing system install never blocks the run).

Each clip is downloaded once, in full, to a temporary file with a browser `User-Agent` header (some CDNs reject the default Python user agent), and then ffmpeg seeks the local file. This is a full download followed by local seeking, not streaming.

The extractor pulls about ten frames at timestamps that form a superset of the leaderboard judge's six accuracy sample points at `dur*(j+0.5)/6`. Those six midpoints are kept unconditionally; a few start, end, and gap-filler frames are added around them and dropped only if they land within half a second of a kept stamp. Because our sample set is a superset of the judge's, our evidence is always at least what the judge sees. Frames are scaled to 768px on the long side with the aspect ratio preserved (never squished) and encoded at JPEG quality 3, so each frame stays small and the batch stays well under the per-call size cap. A timestamp ffmpeg cannot seek is skipped rather than failing the whole clip.

## Robustness

The agent (`agent.py`) is built to always leave a complete, valid `/output/results.json` on disk.

- **Pre-seeded output.** Before any model call, `results.json` is written with a complete grounded fallback entry for every task. Each clip then atomically upgrades its own entry as it finishes, under a lock, writing to a temp file and `os.replace`-ing it into place. A complete file is on disk from the first write, so a kill, timeout, or hang mid-run can never leave "no output" (which would score zero).
- **Time-budget guard.** A global elapsed-time check stops starting new clips once the budget is spent (default 500 seconds, leaving tail room under the grader's 600 second cap). A clip that never starts keeps its pre-seeded fallback.
- **Retry with backoff.** Every provider call goes through a shared wrapper that retries throttling and 5xx responses (408, 409, 425, 429, 500, 502, 503, 504) and network errors with exponential backoff and jitter.
- **Per-clip concurrency.** Clips are captioned concurrently with a `ThreadPoolExecutor` sized by `MAX_CLIPS` (default 2), so a multi-clip run fits under the cap even when the provider is congested.
- **One bad clip never sinks the run.** Each task is wrapped in its own try/except that degrades to the grounded fallback for that clip.

## Build

Keys are never committed. Because the grader runs the image headless with no environment flags, the provider key and settings are baked at build time through `--build-arg`:

```
docker buildx build --platform linux/amd64 \
  --build-arg PROVIDER=fireworks \
  --build-arg FIREWORKS_API_KEY=<your_key> \
  --build-arg FIREWORKS_MODEL=accounts/<acct>/deployments/<id> \
  --build-arg APP_VERSION=v13 \
  -t ghcr.io/<you>/gemma-video-captioner:v13 --push .
```

The `<your_key>`, `<acct>`, and `<id>` above are placeholders. Substitute your own values at build time. No key or token is stored in this repository.

## Run

```
docker run --rm -v /abs/in:/input -v /abs/out:/output ghcr.io/<you>/gemma-video-captioner:v13
```

The container reads `/input/tasks.json` and writes `/output/results.json`.

**Input** is a JSON array of task objects (a top-level `{"tasks":[...]}` wrapper is also accepted):

```json
[
  {
    "task_id": "clip_001",
    "video_url": "https://example.com/clip_001.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

**Output** is a JSON array with one entry per task, each carrying exactly that task's requested styles:

```json
[
  {
    "task_id": "clip_001",
    "captions": {
      "formal": "A cyclist rides along a coastal road at sunset.",
      "sarcastic": "A cyclist braves an entire coastal road at sunset, truly the stuff of legend.",
      "humorous_tech": "A cyclist ships himself down the coast road at sunset, one smooth deploy per pedal stroke.",
      "humorous_non_tech": "A cyclist glides down the coast at sunset, pedaling like the ice cream truck is three streets ahead."
    }
  }
]
```

## Team

Team tripod. AMD Developer Hackathon (ACT II), Track 2, Google Gemma prize.
