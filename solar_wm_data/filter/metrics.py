"""Visual / motion metric adapters.

Each metric is produced by an external tool (FFmpeg VMAF, UniMatch, DOVER,
PySceneDetect, Qwen VLM). To keep the orchestration layer testable and to honour
the "don't occupy GPUs" constraint, every adapter supports ``dry_run``: it
returns a deterministic placeholder derived from the clip id instead of invoking
the heavy tool. Mean color saturation is light enough to compute on CPU when
OpenCV is available.

Real (non-dry-run) execution shells out to the upstream repos under ``third_party/``
per ``configs/models.yaml``; those paths need the tools installed by the
``scripts/setup_*.sh`` helpers.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..manifest import ClipRecord, QualityMetrics


def _seeded(clip_id: str, lo: float, hi: float, salt: str) -> float:
    """Deterministic pseudo-value in [lo, hi] from clip id (dry-run only)."""
    h = hashlib.sha1(f"{clip_id}:{salt}".encode()).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return lo + frac * (hi - lo)


def mean_saturation(video_path: str) -> float | None:
    """Mean HSV saturation over sampled frames. CPU. Returns None if no OpenCV."""
    try:
        import cv2  # type: ignore
        import numpy as np
    except Exception:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, total // 16)  # sample ~16 frames
    sats, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            sats.append(float(np.mean(hsv[:, :, 1])))
        idx += 1
    cap.release()
    return sum(sats) / len(sats) if sats else None


def compute_metrics(rec: ClipRecord, models_cfg: dict) -> QualityMetrics:
    """Populate a QualityMetrics for a clip.

    In dry-run mode all heavy metrics are deterministic placeholders so the
    pipeline is exercisable end-to-end without a GPU. Saturation is computed for
    real when OpenCV is present (still cheap).
    """
    dry = models_cfg.get("dry_run", True)
    cid = rec.clip_id
    m = QualityMetrics()
    skipped: list[str] = []

    # Saturation is measured whenever the video and OpenCV are available. Outside
    # dry-run, an unavailable measurement remains ``None`` so filtering fails closed.
    sat = mean_saturation(rec.video_path) if Path(rec.video_path).exists() else None
    if sat is None and dry:
        sat = _seeded(cid, 20, 150, "sat")
    m.saturation = sat
    if sat is None:
        skipped.append("saturation:unmeasurable")
        rec.extra["metrics_skipped"] = list(skipped)

    if dry:
        m.vmaf = _seeded(cid, 1.0, 40.0, "vmaf")
        m.unimatch = _seeded(cid, 4.0, 60.0, "unimatch")
        m.dover_tech = _seeded(cid, 0.3, 0.9, "dover_t")
        m.dover_aes = _seeded(cid, 0.3, 0.9, "dover_a")
        m.scene_cuts = int(_seeded(cid, 0, 1.99, "cuts"))
        return m

    # --- real execution -------------------------------------------------
    # Each metric is computed independently and tolerant of a missing tool: a
    # failure leaves the field as None (the threshold layer treats an absent
    # metric as "not applied") and is recorded in rec.extra so the run is
    # honest about what actually ran. Motion (VMAF) and optical-flow (UniMatch)
    # are computed directly from frames with OpenCV — real measurements, no
    # external build. DOVER (a learned model) is attempted via its adapter and
    # skipped if unavailable. The VLM fields are NOT computed here: they come from
    # the separate annotation pass and are merged at assembly.
    from . import adapters

    def _try(name, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - record and continue
            skipped.append(f"{name}:{type(e).__name__}")
            return None

    # Real tools as primary (no stand-ins): UniMatch GMFlow, ffmpeg vmafmotion,
    # DOVER, PySceneDetect. Each is independent and tolerant of a missing tool.
    m.unimatch = _try("unimatch", lambda: adapters.unimatch_flow(rec.video_path, models_cfg))
    m.vmaf = _try("vmaf", lambda: adapters.ffmpeg_vmaf(rec.video_path, models_cfg))
    m.scene_cuts = _try("scene_cuts", lambda: adapters.pyscenedetect_cuts(rec.video_path))
    dover = _try("dover", lambda: adapters.dover(rec.video_path, models_cfg))
    if dover is not None:
        m.dover_tech, m.dover_aes = dover

    if skipped:
        rec.extra["metrics_skipped"] = skipped
    return m
