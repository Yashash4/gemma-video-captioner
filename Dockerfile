# Track-2 video-captioning agent (team tripod). linux/amd64, well under the 10GB cap.
# Grader runs it headless: `docker run <image>` with /input + /output mounted, no env flags
# (FAQ #21) — so the model key is baked at BUILD time via --build-arg (never committed).
FROM python:3.11-slim

# ffmpeg = frame extraction (frames.py prefers the system binary; imageio-ffmpeg is the fallback)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py ./

# Keys/provider baked at build time (grader runs headless with no env):
#   docker buildx build --platform linux/amd64 \
#     --build-arg PROVIDER=ollama --build-arg OLLAMA_API_KEY=xxx -t <repo> --push .
ARG OLLAMA_API_KEY=""
ARG GOOGLE_AI_STUDIO_API_KEY=""
ARG FIREWORKS_API_KEY=""
# Deployment-specific model string for the on-demand (dedicated) deployment, e.g.
# accounts/<acct>/deployments/<id>. Without it, gemma._fireworks falls back to the base
# model path which has no serverless mode -> 404. Baked at build time.
ARG FIREWORKS_MODEL=""
ARG PROVIDER=""
# Optional diagnostics sink (agent._telemetry). Inert unless set. No secret; passed at build
# time so no URL is committed to source.
ARG TELEMETRY_URL=""
# Version stamp (--build-arg APP_VERSION=v8). Logged in the boot telemetry so we know
# WHICH version the grader actually ran when a score lands on the shared :latest tag.
ARG APP_VERSION=""
ENV OLLAMA_API_KEY=$OLLAMA_API_KEY
ENV GOOGLE_AI_STUDIO_API_KEY=$GOOGLE_AI_STUDIO_API_KEY
ENV FIREWORKS_API_KEY=$FIREWORKS_API_KEY
ENV FIREWORKS_MODEL=$FIREWORKS_MODEL
ENV PROVIDER=$PROVIDER
ENV TELEMETRY_URL=$TELEMETRY_URL
ENV APP_VERSION=$APP_VERSION
ENV PYTHONUNBUFFERED=1

# agent.py: reads /input/tasks.json -> writes /output/results.json
ENTRYPOINT ["python", "agent.py"]
