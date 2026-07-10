"""Video -> uniformly-sampled JPEG frames (base64), via ffmpeg.

Prefers system ffmpeg on PATH; falls back to the imageio-ffmpeg bundled binary so
we're never blocked on a system install. Duration is read from ffmpeg's own stderr
(no ffprobe needed -> works with the bundled binary too).
"""
import base64
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request

_FFMPEG = None
_SRC = None


def _ffmpeg():
    global _FFMPEG, _SRC
    if _FFMPEG:
        return _FFMPEG
    sysbin = shutil.which("ffmpeg")
    if sysbin:
        _FFMPEG, _SRC = sysbin, "system"
    else:
        import imageio_ffmpeg
        _FFMPEG, _SRC = imageio_ffmpeg.get_ffmpeg_exe(), "imageio-ffmpeg (bundled)"
    return _FFMPEG


def ffmpeg_source():
    _ffmpeg()
    return _SRC


def _duration(path):
    # ffmpeg prints "Duration: HH:MM:SS.ss" to stderr; parse it (portable, no ffprobe).
    err = subprocess.run([_ffmpeg(), "-hide_banner", "-i", path],
                         capture_output=True, text=True).stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", err)
    if not m:
        return 0.0
    h, mm, ss = m.groups()
    return int(h) * 3600 + int(mm) * 60 + float(ss)


def extract_frames(video_path_or_url, n=10, size=512):
    """Return n base64 JPEGs uniformly spaced in [5%,95%] of the video.

    URL inputs are downloaded to a temp file first. Frames scaled to `size`px wide,
    JPEG q~80 (-q:v 3) -> ~30-80KB each, so n frames stay well under the 10MB/call cap.
    Unseekable timestamps are skipped rather than aborting the whole clip.
    """
    tmp = None
    path = video_path_or_url
    if path.startswith(("http://", "https://")):
        fd, tmp = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        urllib.request.urlretrieve(path, tmp)  # ponytail: trusted hackathon URLs; add size cap if inputs widen
        path = tmp
    try:
        dur = _duration(path)
        if dur > 0 and n > 1:
            lo, hi = dur * 0.05, dur * 0.95
            stamps = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        elif dur > 0:
            stamps = [dur * 0.5]
        else:
            stamps = [float(i) for i in range(n)]  # duration unknown: 1s apart
        frames = []
        for t in stamps:
            fd, jpg = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            try:
                subprocess.run(
                    [_ffmpeg(), "-y", "-ss", f"{t:.3f}", "-i", path,
                     "-frames:v", "1", "-vf", f"scale={size}:-2", "-q:v", "3", jpg],
                    capture_output=True, check=True)
                with open(jpg, "rb") as f:
                    data = f.read()
                if data:
                    frames.append(base64.b64encode(data).decode())
            except subprocess.CalledProcessError:
                pass  # ponytail: skip a frame ffmpeg can't seek, keep the rest
            finally:
                if os.path.exists(jpg):
                    os.remove(jpg)
        return frames
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    # self-check: extract from the local v1 clip and confirm we got real JPEG bytes.
    here = os.path.dirname(os.path.abspath(__file__))
    clip = os.path.join(os.path.dirname(here), "samples", "clips", "v1.mp4")
    fr = extract_frames(clip, n=10)
    print("ffmpeg:", ffmpeg_source(), "| frames:", len(fr),
          "| avg KB:", round(sum(len(x) for x in fr) / max(len(fr), 1) * 3 / 4 / 1024, 1))
    assert len(fr) >= 8 and all(fr), "expected >=8 non-empty frames"
    print("frames.py self-check passed")
