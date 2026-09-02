#!/usr/bin/env python3
"""Extract OmniWorld-Game GT camera poses + intrinsics from a scene's annotation tar,
so the solar_wm gt_pose path uses the REAL GT trajectory instead of the fabricated
``1.7 * pred_pos`` proxy (audit finding: the acquire only assembled RGB frames and never
extracted the GT that OmniWorld ships, so gt_pose fell into its no-GT proxy branch).

Format + math mirror the validated downstream WebDataset packer:
  annotations/OmniWorld-Game/<uid>/<uid>_others.tar.gz
    split_info.json                 -> {"split": [[global frame ids], ...]}
    camera/split_<i>.json           -> {"quats":[S,4] wxyz, "trans":[S,3], "focals":[S],
                                        "cx":float, "cy":float}   (LOCAL-indexed per split)
  c2w = inv(w2c=[R(quat)|t]);  per-UID metric_scale scales c2w translation to meters.
Camera arrays are local to a split; split_info["split"][i] gives their GLOBAL frame ids,
which match the color/<g>.png frames the acquire assembles (sorted) into video.mp4.

Writes <clip_dir>/poses.npy (N,4,4 c2w, float64) + intrinsics.npy (N,4 fx,fy,cx,cy) frame-
ordered by global id. Then ingest picks up poses.npy (gt_positions_path) and the gt_pose
branch keeps this REAL trajectory + Umeyama metric scale.

Before running this over a new release of the source, confirm on one scene that (a) the
global-id ordering matches the assembled mp4 frame order, (b) every assembled frame has a
camera entry (subsample if not), and (c) the metric_scale is the one you expect. The
split-concat is re-derived here rather than shared with the packer.
"""
from __future__ import annotations

import json
import re
import sys
import tarfile
from pathlib import Path

import numpy as np


def _quat_wxyz_to_R(q_wxyz: np.ndarray) -> np.ndarray:
    """(...,4) wxyz unit quaternion -> (...,3,3) rotation (matches scipy from_quat(xyzw))."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    n = np.where(n > 0, n, 1.0)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def quats_trans_to_c2w(quats_wxyz, trans, metric_scale=None) -> np.ndarray:
    """OmniWorld camera json -> c2w (F,4,4). Mirrors packer quats_trans_to_c2w."""
    R = _quat_wxyz_to_R(np.asarray(quats_wxyz, dtype=np.float64))   # (F,3,3)
    t = np.asarray(trans, dtype=np.float64)                         # (F,3)
    w2c = np.repeat(np.eye(4)[None], len(R), axis=0)
    w2c[:, :3, :3] = R
    w2c[:, :3, 3] = t
    c2w = np.linalg.inv(w2c)
    if metric_scale is not None:
        c2w[:, :3, 3] *= float(metric_scale)
    return c2w.astype(np.float64)


def extract_gt(others_tar: str, metric_scale: float | None = None):
    """Return (c2w (N,4,4), intrinsics (N,4) fx,fy,cx,cy, global_ids (N,)) frame-ordered."""
    split_info = None
    cams: dict[int, dict] = {}
    with tarfile.open(others_tar, "r:gz") as tf:
        for m in tf.getmembers():
            if m.name.endswith("split_info.json"):
                split_info = json.load(tf.extractfile(m))
            else:
                mo = re.search(r"camera/split_(\d+)\.json$", m.name)
                if mo:
                    cams[int(mo.group(1))] = json.load(tf.extractfile(m))
    if split_info is None:
        raise ValueError(f"{others_tar}: no split_info.json")

    # global frame id -> (quat_wxyz, trans, focal, cx, cy) using each split's local order
    perframe: dict[int, tuple] = {}
    for i, frames in enumerate(split_info["split"]):
        c = cams.get(i)
        if c is None:
            continue
        q, tr, fo = c["quats"], c["trans"], c["focals"]
        cx, cy = float(c["cx"]), float(c["cy"])
        for local, g in enumerate(frames):
            if local < len(q):
                perframe[int(g)] = (q[local], tr[local], float(fo[local]), cx, cy)

    gids = sorted(perframe)
    if not gids:
        raise ValueError(f"{others_tar}: no camera entries")
    quats = [perframe[g][0] for g in gids]
    trans = [perframe[g][1] for g in gids]
    c2w = quats_trans_to_c2w(quats, trans, metric_scale)
    intr = np.array([[perframe[g][2], perframe[g][2], perframe[g][3], perframe[g][4]]
                     for g in gids], dtype=np.float64)
    return c2w, intr, np.asarray(gids, dtype=np.int64)


def main():
    if len(sys.argv) < 3:
        print("usage: extract_omniworld_gt.py <others.tar.gz> <clip_dir> [metric_scale]")
        raise SystemExit(64)
    others_tar, clip_dir = sys.argv[1], sys.argv[2]
    ms = float(sys.argv[3]) if len(sys.argv) > 3 else None
    c2w, intr, gids = extract_gt(others_tar, ms)
    out = Path(clip_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "poses.npy", c2w)
    np.save(out / "intrinsics.npy", intr)
    print(f"OMNIWORLD_GT_DONE {clip_dir} poses={c2w.shape} intr={intr.shape} "
          f"frames[{gids[0]}..{gids[-1]}] metric_scale={ms}")


if __name__ == "__main__":
    main()
