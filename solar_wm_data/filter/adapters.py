"""Real adapters around the vendored quality/flow tools.

* VMAF motion  -> static-ffmpeg ``vmafmotion`` filter (single input, no reference)
* DOVER        -> evaluate_one_video.py fused [0,1] score
* UniMatch     -> GMFlow mean optical-flow magnitude (see _unimatch.py)
* scene cuts   -> PySceneDetect

Paths resolve under ``SOLAR_WM_ROOT`` (the deployed repo dir). These are the
real-execution paths used when ``dry_run`` is false; nothing is faked.
"""

from __future__ import annotations

import os
import re
import subprocess

_ROOT = os.environ.get(
    "SOLAR_WM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_FFMPEG = os.path.join(_ROOT, "bin", "ffmpeg")
_DOVER = os.path.join(_ROOT, "third_party", "DOVER")


class AdapterError(RuntimeError):
    pass


def _run(cmd, cwd=None, env=None, timeout=600):
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=timeout)
    return proc


def ffmpeg_vmaf(video_path: str, cfg: dict) -> float:
    """Mean VMAF motion via static ffmpeg's vmafmotion filter (self-referential)."""
    ff = _FFMPEG if os.path.exists(_FFMPEG) else "ffmpeg"
    proc = _run([ff, "-hide_banner", "-i", video_path, "-vf", "vmafmotion",
                 "-f", "null", "-"])
    text = proc.stderr + proc.stdout
    m = re.search(r"VMAF Motion avg:\s*([0-9.]+)", text)
    if not m:
        raise AdapterError(f"vmafmotion avg not found:\n{text[-500:]}")
    return float(m.group(1))


def unimatch_flow(video_path: str, cfg: dict) -> float:
    """Mean GMFlow optical-flow magnitude over frame pairs sampled every 0.5 s across
    the first 60 s. _unimatch pairs consecutive frames, so sampling at
    a 0.5 s wall-clock stride makes every pair exactly 0.5 s apart — independent of clip
    length, so the UniMatch ranges (calibrated for 0.5 s pairs) stay valid."""
    from ..pose.adapters import read_frames_timed
    from . import _unimatch
    frames = read_frames_timed(
        video_path, cfg.get("flow_interval_s", 0.5), cfg.get("flow_max_s", 60.0)
    )
    # Flow magnitude is in PIXELS, so the metric is only comparable across the corpus
    # at a bounded working resolution; 4K inputs also OOM GMFlow's global attention
    # (the silent unimatch=null path on dl3dv). Downscale anything above 1080p.
    max_h = cfg.get("flow_max_h", 1080)
    if frames.shape[1] > max_h:
        import cv2
        import numpy as np
        s = max_h / frames.shape[1]
        w = int(round(frames.shape[2] * s))
        frames = np.stack(
            [cv2.resize(f, (w, max_h), interpolation=cv2.INTER_AREA) for f in frames])
    return _unimatch.mean_flow_magnitude(frames)


def _dover_one(video_path: str, cfg: dict) -> float:
    """Fused DOVER quality in [0,1] for a single (sub)clip."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    # DOVER fetches its ConvNeXt backbone via torch.hub on first use; if you are
    # behind a proxy set HTTP_PROXY (or pass cfg["proxy"]). Cached after first run.
    proxy = cfg.get("proxy", os.environ.get("HTTP_PROXY", ""))
    if proxy:
        env.setdefault("http_proxy", proxy)
        env.setdefault("https_proxy", proxy)
    env.pop("HF_HUB_OFFLINE", None)  # torch.hub uses urllib, not HF; don't block
    proc = _run(["python3", "evaluate_one_video.py", "-v", video_path, "-f"],
                cwd=_DOVER, env=env)
    text = proc.stdout + proc.stderr
    m = re.search(r"overall score \(scale in \[0,1\]\):\s*([0-9.]+)", text)
    if not m:
        m = re.search(r"fused[^0-9]*([0-9.]+)", text)
    if not m:
        raise AdapterError(f"DOVER score not found:\n{text[-600:]}")
    return float(m.group(1))


def _cut_segment(video_path: str, start_s: float, dur_s: float) -> str:
    """Cut [start_s, start_s+dur_s) into a temp mp4; caller deletes it."""
    import tempfile
    ff = _FFMPEG if os.path.exists(_FFMPEG) else "ffmpeg"
    fd, out = tempfile.mkstemp(suffix=".mp4", prefix="doverchunk_")
    os.close(fd)
    _run([ff, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start_s:.3f}",
          "-i", video_path, "-t", f"{dur_s:.3f}", "-an",
          "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", out])
    return out


def _dover_many(paths: list[str], cfg: dict) -> dict:
    """Score many clips with a SINGLE DOVER model load (scripts/dover_score_many.py).

    Returns ``{path: fused_score}``; clips DOVER fails on are simply absent. A shared
    model load keeps 5 s chunk scoring efficient."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    proxy = cfg.get("proxy", os.environ.get("HTTP_PROXY", ""))
    if proxy:
        env.setdefault("http_proxy", proxy)
        env.setdefault("https_proxy", proxy)
    env.pop("HF_HUB_OFFLINE", None)
    script = os.path.join(_ROOT, "scripts", "dover_score_many.py")
    proc = _run(["python3", script, "dover.yml", *paths], cwd=_DOVER, env=env,
                timeout=max(600, 120 * len(paths)))
    out = {}
    for m in re.finditer(r"DOVER_SCORE (\S+) ([0-9.]+)", proc.stdout + proc.stderr):
        out[m.group(1)] = float(m.group(2))
    return out


def dover(video_path: str, cfg: dict) -> tuple[float, float]:
    """DOVER fused quality in [0,1], AVERAGED over non-overlapping 5 s chunks across the
    first 60 s — the span the DOVER ranges were calibrated on;
    whole-clip DOVER on a minute-scale video dilutes brief low-quality sections. All
    chunks are scored in one model load (`_dover_many`), so chunking runs by default.
    Returned as (avg, avg). Set ``dover_chunk_s: 0`` to fall back to whole-clip scoring."""
    from ..pose.adapters import _video_fps
    chunk_s = float(cfg.get("dover_chunk_s", os.environ.get("SOLAR_WM_DOVER_CHUNK_S", "5")))
    max_s = float(cfg.get("dover_max_s", 60.0))
    total, fps = _video_fps(video_path)
    dur = (total / fps) if fps > 0 else 0.0
    n = 0 if chunk_s <= 0 else min(int(dur // chunk_s), int(max_s // chunk_s))
    if n <= 1:  # short clip (<=1 chunk) or chunking disabled -> single whole-clip score
        s = _dover_one(video_path, cfg)
        return s, s
    segs = [_cut_segment(video_path, k * chunk_s, chunk_s) for k in range(n)]
    try:
        scored = _dover_many(segs, cfg)
    finally:
        for seg in segs:
            try:
                os.remove(seg)
            except OSError:
                pass
    scores = [scored[s] for s in segs if s in scored]
    if not scores:
        raise AdapterError("DOVER produced no chunk scores")
    avg = sum(scores) / len(scores)
    return avg, avg


def pyscenedetect_cuts(video_path: str) -> int:
    """Number of scene cuts via PySceneDetect content detector.

    The detector threshold is configurable because fast motion and soft transitions can
    otherwise be classified as cuts. When unset, the library default is used."""
    from scenedetect import detect, ContentDetector  # type: ignore
    th = os.environ.get("SOLAR_WM_SCENECUT_THRESHOLD")
    det = ContentDetector(threshold=float(th)) if th else ContentDetector()
    scenes = detect(video_path, det)
    return max(0, len(scenes) - 1)
