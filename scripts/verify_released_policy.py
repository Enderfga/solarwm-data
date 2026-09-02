#!/usr/bin/env python3
"""Check that `configs/filters_released.yaml` still reproduces the corpus it describes.

Every record in a corpus `meta.jsonl` already carries the `kept_tier` this policy gave it,
so the file does not have to be taken on trust: re-judge those records and compare.
Agreement is evidence that the published policy and the published corpus are the same
thing. A single disagreement is a bug — in the policy file, in the judging code, or in
the claim that the two match.

    python3 scripts/verify_released_policy.py --meta-dir <dir of <owner>.jsonl>
    python3 scripts/verify_released_policy.py --s3 s3://<bucket>/corpus/<prefix> --limit 2000

WHAT THIS DOES AND DOES NOT COVER. It judges from the metrics each record stores, which is
exactly what the frozen policy consumed. The camera gate is evaluated from the record's
stored `camera` block (median fov_x / fov_y / focal_div / scale_cov) rather than per-frame
from intrinsics.npy, because the per-frame arrays live in the shards, not the metadata.
Per-frame evaluation is therefore stricter; a clip that passes this metadata-only check
can still fail on one frame. Inspect any mismatch in that direction.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solar_wm_data.config import load_config  # noqa: E402
from solar_wm_data.manifest import CameraMetrics, ClipRecord, QualityMetrics  # noqa: E402
from solar_wm_data.filter.thresholds import judge_clip  # noqa: E402

# Corpus owner directory -> policy key. The two DL3DV temporal views share one rule.
OWNER_TO_POLICY = {
    "abot": "abot",
    "dl3dv-10s": "dl3dv", "dl3dv-60s": "dl3dv",
    "mind": "mind",
    "miradata": "miradata", "miradata-clean": "miradata_clean",
    "multicamvideo": "multicamvideo",
    "omniworld": "omniworld",
    "realcam_vid": "realcam_vid",
    "sekai_game": "sekai_game",
    "sekai_walking": "sekai_walking", "sekai_walking-clean": "sekai_walking_clean",
    "spatialvid": "spatialvid", "spatialvid-clean": "spatialvid_clean",
}


def record_from_meta(d: dict) -> ClipRecord:
    """Rebuild the judged object from a stored corpus record."""
    m = d.get("metrics") or {}
    c = d.get("camera") or {}
    rec = ClipRecord(
        clip_id=d.get("clip_id", ""), source=d.get("source", ""),
        video_path="", fps=d.get("fps"), num_frames=d.get("num_frames"),
        width=d.get("width"), height=d.get("height"),
        caption=d.get("caption"),
    )
    rec.metrics = QualityMetrics(
        saturation=m.get("saturation"), vmaf=m.get("vmaf"), unimatch=m.get("unimatch"),
        dover_tech=m.get("dover_tech"), dover_aes=m.get("dover_aes"),
        scene_cuts=m.get("scene_cuts"),
        vlm_entity_density=m.get("vlm_entity_density"), vlm_quality=m.get("vlm_quality"),
        vlm_reject_flags=m.get("vlm_reject_flags"), vlm_scene_type=m.get("vlm_scene_type"),
        vlm_scene_transition=m.get("vlm_scene_transition"),
    )
    rec.camera = CameraMetrics(
        fov_x=c.get("fov_x"), fov_y=c.get("fov_y"),
        focal_div=c.get("focal_div"), scale_cov=c.get("scale_cov"),
    )
    return rec


def camera_reasons(rec: ClipRecord, cam_cfg: dict) -> list[str]:
    """The camera gate, from the record's stored medians (see the module docstring)."""
    out: list[str] = []
    lo, hi = cam_cfg["fov_deg"]
    for name, v in (("fov_x", rec.camera.fov_x), ("fov_y", rec.camera.fov_y)):
        if v is None:
            out.append(f"{name}=None")
        elif not (lo <= v <= hi):
            out.append(f"{name}={v:.2f} outside [{lo},{hi}]")
    fd = rec.camera.focal_div
    if fd is not None and fd > cam_cfg["focal_div_max"]:
        out.append(f"focal_div={fd:.3f} > {cam_cfg['focal_div_max']}")
    sc = rec.camera.scale_cov
    if sc is not None and sc > cam_cfg["scale_cov_max"]:
        out.append(f"scale_cov={sc:.3f} > {cam_cfg['scale_cov_max']}")
    return out


def judge(d: dict, cfg: dict, policy_key: str) -> str | None:
    ds = cfg["datasets"][policy_key]
    rec = record_from_meta(d)
    cam_cfg = {**cfg["camera"], **(ds.get("camera") or {})}
    # A source whose rule names no camera gate is not camera-gated at all: the Sekai-Game
    # row deliberately omits C120, and applying it anyway would reject clips the frozen
    # policy kept.
    if "camera" in ds or _row_uses_camera(policy_key):
        if camera_reasons(rec, cam_cfg):
            return None
    tier, _ = judge_clip(rec, ds)
    return tier


# Which rows carry a camera gate.
#
# Ten rows name C120 or C125. Sekai-Game names no geometry gate. The three Clean owners
# use `G`, which includes the camera scalars, so they remain camera-gated here.
_NO_CAMERA_ROWS = {"sekai_game"}


def _row_uses_camera(policy_key: str) -> bool:
    return policy_key not in _NO_CAMERA_ROWS


#: Rejection reasons this file structurally cannot reach, each one-sided (the policy file
#: is the permissive side) and each enforced elsewhere in the pipeline.
_OUT_OF_SCOPE = (
    # The full gate evaluates each frame; the stored record carries only medians, so a
    # clip whose median field of view sits just inside the bound can still have frames
    # outside it — invisible from here. Disappears once per-frame intrinsics are available.
    "recipe_camera_diagnostics",
    # "Exact kept-source lineage": a clean clip whose SOURCE the policy rejected inherits
    # the rejection. That needs the source owner's whole kept set, not a range over this
    # clip's own metrics, so it lives in `clean_plate.kept_source_reasons` instead.
    "recipe_lineage_",
)


def _explained(d: dict, stored: str | None, judged: str | None) -> bool:
    """Is this disagreement a known limitation rather than a transcription error?

    Only when the record was rejected for something this file cannot express, and only in
    the permissive direction — we judged kept, the corpus stored rejected. A disagreement
    the other way means the rule as written here is STRICTER than the frozen one, which no
    limitation below can produce, so it is a real failure and must not be absorbed.
    """
    if not (stored is None and judged is not None):
        return False
    reasons = d.get("reject_reasons") or []
    return any(r.startswith(_OUT_OF_SCOPE) for r in reasons)


def iter_records(path: Path, limit: int):
    n = 0
    with open(path, errors="ignore") as fh:
        for line in fh:
            if limit and n >= limit:
                return
            try:
                yield json.loads(line)
            except Exception:
                continue          # a truncated tail line is not a policy failure
            n += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", required=True,
                    help="directory of <owner>.jsonl corpus metadata samples")
    ap.add_argument("--filters", default="filters_released")
    ap.add_argument("--limit", type=int, default=0, help="records per owner (0 = all)")
    a = ap.parse_args()

    cfg = load_config(a.filters)
    total = agree = 0
    worst: list[tuple[str, int, int, dict]] = []

    for f in sorted(Path(a.meta_dir).glob("*.jsonl")):
        owner = f.stem
        key = OWNER_TO_POLICY.get(owner)
        if key is None:
            print(f"  {owner:22s} SKIP (no policy row)")
            continue
        n = ok = expl = 0
        cm: collections.Counter = collections.Counter()
        for d in iter_records(f, a.limit):
            got, want = judge(d, cfg, key), d.get("kept_tier")
            n += 1
            if got == want:
                ok += 1
            elif _explained(d, want, got):
                expl += 1
            else:
                cm[(want, got)] += 1
        if not n:
            continue
        bad = n - ok - expl
        total += n
        agree += ok + expl
        note = f"   (+{expl} explained)" if expl else ""
        flag = "   <-- UNEXPLAINED" if bad else ""
        print(f"  {owner:22s} {ok:6d}/{n:<6d} = {100 * ok / n:6.2f}%{note}{flag}")
        if bad:
            worst.append((owner, n - bad, n, dict(cm)))

    print(f"\n  TOTAL {agree}/{total} = {100 * agree / max(total, 1):.3f}% "
          f"(agreement + explained)")
    for owner, ok, n, cm in worst:
        print(f"\n  {owner}: {n - ok} disagreements (stored -> judged)")
        for (w, g), v in sorted(cm.items(), key=lambda x: -x[1])[:6]:
            print(f"    {str(w):6s} -> {str(g):6s}  {v}")
    return 0 if agree == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
