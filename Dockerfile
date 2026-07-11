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

# Keys/provider baked at build time:
#   docker buildx build --platform linux/amd64 \
#     --build-arg PROVIDER=fireworks --build-arg FIREWORKS_API_KEY=xxx -t <repo> --push .
ARG GOOGLE_AI_STUDIO_API_KEY=""
ARG FIREWORKS_API_KEY=""
ARG PROVIDER=""
ENV GOOGLE_AI_STUDIO_API_KEY=$GOOGLE_AI_STUDIO_API_KEY \
    FIREWORKS_API_KEY=$FIREWORKS_API_KEY \
    PROVIDER=$PROVIDER \
    PYTHONUNBUFFERED=1

# agent.py: reads /input/tasks.json -> writes /output/results.json
ENTRYPOINT ["python", "agent.py"]
