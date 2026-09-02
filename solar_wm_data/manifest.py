"""Clip manifest schema and JSONL I/O for the SolarWM annotation pipeline.

A *manifest* is a JSON Lines file: one :class:`ClipRecord` per line. Every stage
of the pipeline reads a manifest, augments each record with the fields it owns,
and writes the manifest back out. Records are append-only and stages are
idempotent, so a partially-processed manifest can always be resumed.

A record carries everything a clip is judged and trained on: a metric-scale 6-DoF
pose track, per-frame intrinsics, the visual and motion quality metrics, the
camera-specific filter quantities, a scene-static caption, and the verdict with its
ordered reasons.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class CameraMetrics:
    """Camera-specific filter quantities.

    The gates run PER FRAME — a clip with one bad frame is rejected — so the extremes
    are what a later re-judgement needs. The medians describe the clip but cannot prove
    that every frame is inside a bound. Re-judgement therefore uses extremes when present
    and explicitly reports any fallback to medians.
    """

    fov_x: float | None = None  # horizontal field of view, degrees (median)
    fov_y: float | None = None  # vertical field of view, degrees (median)
    focal_div: float | None = None  # |fx-fy| / ((fx+fy)/2) (median)
    scale_cov: float | None = None  # std(s_t) / (mean(s_t)+eps) over per-frame scales

    # The values the gates actually compared against.
    fov_x_min: float | None = None
    fov_x_max: float | None = None
    fov_y_min: float | None = None
    fov_y_max: float | None = None
    focal_div_max: float | None = None


@dataclass
class QualityMetrics:
    """Visual / motion quality metrics."""

    saturation: float | None = None  # mean color saturation
    vmaf: float | None = None  # FFmpeg VMAF motion score
    unimatch: float | None = None  # UniMatch optical-flow magnitude
    dover_tech: float | None = None  # DOVER technical quality
    dover_aes: float | None = None  # DOVER aesthetic quality
    scene_cuts: int | None = None  # PySceneDetect scene-cut count
    # --- VLM annotation ----------------------------------------------------------
    # These do NOT come from the filter stage. They arrive from the annotation pass as a
    # create-only overlay and are merged at assembly, which is why they are None on a
    # freshly produced clip. The public contract uses a single 1-5 quality scale.
    vlm_entity_density: int | None = None      # 1 none | 2 sparse | 3 dense (max simultaneous)
    vlm_quality: float | None = None           # 1.0 .. 5.0, whole steps
    vlm_reject_flags: list[str] | None = None  # subset of caption.REJECT_FLAGS
    vlm_scene_type: str | None = None          # one of caption.SCENE_TYPES
    vlm_scene_transition: dict | None = None   # {label, count, timestamps_sec, evidence}


@dataclass
class ClipRecord:
    """One clip flowing through the pipeline.

    Fields are grouped by the stage that owns them. ``None`` means "not yet
    computed by the responsible stage".
    """

    # --- ingest ---
    clip_id: str
    source: str  # a registered source, e.g. "dl3dv", "omniworld"
    video_path: str
    mode: str = "default"  # pose annotation mode: default | gt_depth | gt_pose
    fps: float | None = None
    num_frames: int | None = None
    width: int | None = None
    height: int | None = None

    # --- pose ---
    pose_path: str | None = None  # .npy, shape (N, 4, 4) camera-to-world (c2w, OpenCV); camera center = M[:3,3]
    intrinsics_path: str | None = None  # .npy, shape (N, 4) per-frame (fx,fy,cx,cy)
    scale_factors: list[float] | None = None  # per-frame metric scale s_t
    pose_mode: str | None = None  # actual mode used by the pose engine
    #: Units of the translation column of poses.npy. "metric" means metres. Recorded
    #: rather than assumed, because a consumer cannot tell by looking: a trajectory in
    #: COLMAP units has the same shape, the same dtype and the same plausible-looking
    #: numbers as one in metres.
    pose_units: str | None = None

    # --- filter ---
    metrics: QualityMetrics = field(default_factory=QualityMetrics)
    camera: CameraMetrics = field(default_factory=CameraMetrics)
    kept: bool | None = None
    # Three disjoint physical labels: "xhigh" | "high" | None. None means rejected, and
    # `kept` is the boolean view of the same decision -- they are never set apart.
    kept_tier: str | None = None
    # ORDERED: the order records which class of gate fired first and is stable run to run.
    reject_reasons: list[str] = field(default_factory=list)

    # --- caption ---
    caption: str | None = None

    # arbitrary stage extras without schema churn
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ClipRecord":
        d = dict(d)
        metrics = d.pop("metrics", None) or {}
        camera = d.pop("camera", None) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        extra = d.pop("extra", None) or {}
        # tolerate unknown top-level keys by folding them into extra
        for k in list(d.keys()):
            if k not in known:
                extra[k] = d.pop(k)
        rec = cls(**d)
        rec.metrics = QualityMetrics(**_only_known(metrics, QualityMetrics))
        rec.camera = CameraMetrics(**_only_known(camera, CameraMetrics))
        rec.extra = extra
        return rec


def _only_known(d: dict[str, Any], cls: type) -> dict[str, Any]:
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in d.items() if k in names}


def read_manifest(path: str | Path) -> list[ClipRecord]:
    """Load all records from a JSONL manifest."""
    records: list[ClipRecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(ClipRecord.from_dict(json.loads(line)))
    return records


def iter_manifest(path: str | Path) -> Iterator[ClipRecord]:
    """Stream records from a JSONL manifest (memory-friendly for large runs)."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield ClipRecord.from_dict(json.loads(line))


def write_manifest(path: str | Path, records: Iterable[ClipRecord]) -> None:
    """Write records to a JSONL manifest atomically (write to .tmp then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    tmp.replace(path)
