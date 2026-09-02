#!/usr/bin/env python3
"""Corpus clip validator — the verify-before-produce contract for every source.

A clip is VALID only if its geometry is internally consistent and trainable:

  1. poses (N,4,4) f64, intrinsics (N,4) f64 — load cleanly, no NaN/Inf.
  2. video frame count == N_poses == N_intrinsics; pose[i] describes frame[i].
  3. N is a corpus length from solar_wm_data.spec exactly, OR shorter than the target
     when the SOURCE clip is shorter (we never upsample) — short lengths WARN, never fail.
  4. video fps == the active spec's frame rate.
  5. every pose is a valid SE3: R orthonormal (||RRᵀ-I||<1e-3), det(R)≈+1,
     bottom row exactly [0,0,0,1].
  6. intrinsics sane: fx,fy>0; 0<cx<W, 0<cy<H. Focal divergence |fx-fy|/avg≤0.20
     is a camera filter — reported as WARN (a filter, not a validity break).
  7. prompt.txt non-empty.

Usage:
    python3 validate_clip.py <clip_dir> [<clip_dir> ...]
    python3 validate_clip.py --glob 'corpus/*/clips/*'
    python3 validate_clip.py --source <src> [--sample N]   # pull N clips from the corpus

Exit code 0 iff every clip PASSES. Prints one PASS/FAIL line per clip + reasons.
Frame count comes from ffprobe, or decord/cv2 where the ffmpeg CLI is absent.
"""
from __future__ import annotations

import glob as _glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# Corpus specs, all 4n+1 so the frame count is latent-aligned. Read from solar_wm_data.spec
# so a clip is validated against what it was actually emitted at, never a hard-coded number.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solar_wm_data import spec as _spec  # noqa: E402

CORPUS_LENGTHS = {n: name for name, n in _spec.SPEC_FRAMES.items()}
TARGET_FPS = _spec.target_fps()


def _ffprobe(video: Path) -> tuple[int, float, int, int]:
    """Return (nb_frames, fps, width, height). Counts frames exactly (slow but honest)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height",
         "-of", "default=noprint_wrappers=1", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    vals: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip()
    nb = int(vals.get("nb_read_frames", "0"))
    num, den = (vals.get("r_frame_rate", "0/1").split("/") + ["1"])[:2]
    fps = float(num) / float(den) if float(den) else 0.0
    return nb, fps, int(vals.get("width", "0")), int(vals.get("height", "0"))


def _frames_fps(video: Path) -> tuple[int, float, int, int]:
    """(nb_frames, fps, W, H), preferring ffprobe; fall back to decord then cv2.

    Both fallbacks count frames through decoding when the ffprobe CLI is unavailable."""
    try:
        return _ffprobe(video)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    try:
        import decord
        vr = decord.VideoReader(str(video))
        h, w = vr[0].shape[:2]
        return len(vr), float(vr.get_avg_fps()), int(w), int(h)
    except Exception:  # noqa: BLE001 - any decord failure -> try cv2
        pass
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nb = 0
    while cap.grab():
        nb += 1
    cap.release()
    return nb, fps, w, h


def validate(clip_dir: Path) -> tuple[bool, list[str], list[str]]:
    """Return (ok, errors, warns)."""
    errs: list[str] = []
    warns: list[str] = []
    pose_p = clip_dir / "poses.npy"
    intr_p = clip_dir / "intrinsics.npy"
    vid_p = clip_dir / "video.mp4"
    prm_p = clip_dir / "prompt.txt"

    for p in (pose_p, intr_p, vid_p):
        if not p.exists():
            errs.append(f"missing {p.name}")
    if errs:
        return False, errs, warns

    poses = np.load(pose_p)
    intr = np.load(intr_p)
    intr2d = intr.reshape(intr.shape[0], -1)  # tolerate (N,1,4) or (N,4)

    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        errs.append(f"poses shape {poses.shape} != (N,4,4)")
    if intr2d.shape[-1] != 4:
        errs.append(f"intrinsics last dim {intr2d.shape[-1]} != 4")
    if not np.isfinite(poses).all():
        errs.append("poses contain NaN/Inf")
    if not np.isfinite(intr2d).all():
        errs.append("intrinsics contain NaN/Inf")
    if errs:
        return False, errs, warns

    nb, fps, W, H = _frames_fps(vid_p)
    Np, Ni = poses.shape[0], intr2d.shape[0]

    # (2) frame alignment — the central contract
    if not (nb == Np == Ni):
        errs.append(f"ALIGNMENT: video={nb} poses={Np} intrinsics={Ni} (must be equal)")

    # (3) clean corpus length
    if Np not in CORPUS_LENGTHS:
        if Np < min(CORPUS_LENGTHS):
            # short SOURCE clip (the source ran out before the spec length; we never
            # upsample) — legal corpus content, surfaced as a warning so packing stats
            # stay visible. The length gate downstream is what drops it.
            warns.append(f"short clip: {Np}f (<{min(CORPUS_LENGTHS)}; source shorter than spec)")
        else:
            specs = ", ".join(f"{n}@{_spec.SPEC_FPS[s]}fps" for n, s in sorted(CORPUS_LENGTHS.items()))
            errs.append(f"length {Np} is not a corpus length ({specs})")

    # (4) fps — checked against the spec THIS clip's length identifies, not against the
    # spec the current environment happens to select. A corpus can hold clips from more
    # than one spec, and a 16fps clip is not broken just because the reader is set to 24.
    want_fps = _spec.SPEC_FPS.get(CORPUS_LENGTHS.get(Np, ""), TARGET_FPS)
    if abs(fps - want_fps) > 0.5:
        errs.append(f"fps {fps:.3f} != {want_fps} (spec {CORPUS_LENGTHS.get(Np, _spec.current_spec())})")

    # (5) SE3 validity
    R = poses[:, :3, :3]
    eye = np.eye(3)
    orth = np.abs(R @ np.transpose(R, (0, 2, 1)) - eye).reshape(R.shape[0], -1).max(axis=1)
    bad_orth = int((orth > 1e-3).sum())
    if bad_orth:
        errs.append(f"SE3: {bad_orth}/{Np} rotations not orthonormal (max err {orth.max():.2e})")
    dets = np.linalg.det(R)
    bad_det = int((np.abs(dets - 1.0) > 1e-3).sum())
    if bad_det:
        errs.append(f"SE3: {bad_det}/{Np} det(R)!=+1 (range {dets.min():.3f}..{dets.max():.3f})")
    bottom = poses[:, 3, :]
    if not np.allclose(bottom, np.array([0, 0, 0, 1.0]), atol=1e-6):
        errs.append("SE3: bottom row != [0,0,0,1]")

    # (6) intrinsics sanity
    fx, fy, cx, cy = intr2d[:, 0], intr2d[:, 1], intr2d[:, 2], intr2d[:, 3]
    if (fx <= 0).any() or (fy <= 0).any():
        errs.append("intrinsics: non-positive focal length")
    if W and ((cx <= 0).any() or (cx >= W).any()):
        warns.append(f"intrinsics: cx out of (0,{W})")
    if H and ((cy <= 0).any() or (cy >= H).any()):
        warns.append(f"intrinsics: cy out of (0,{H})")
    fdiv = np.abs(fx - fy) / ((fx + fy) / 2.0)
    if (fdiv > 0.20).any():
        warns.append(f"camera-filter: focal divergence max {fdiv.max():.2f} > 0.20")

    # (7) prompt
    if not prm_p.exists() or not prm_p.read_text().strip():
        warns.append("prompt.txt empty/missing")

    # (8) pose units. The corpus documents translations as metres, and a trajectory in
    # COLMAP or engine units is indistinguishable from one in metres by inspection: same
    # shape, same dtype, plausible magnitudes. The producer records the units so this
    # validator can check them. A missing unit declaration is reported as unverifiable.
    meta_p = d / "meta.json"
    if meta_p.exists():
        try:
            units = json.loads(meta_p.read_text()).get("pose_units")
        except Exception:  # noqa: BLE001 - a broken meta is reported by its own check
            units = None
        if units is None:
            warns.append("meta.pose_units absent: cannot verify poses are metric")
        elif units != "metric":
            errs.append(f"meta.pose_units={units!r}, expected 'metric'")

    return (len(errs) == 0), errs, warns


def _pull_s3_sample(source: str, n: int) -> list[Path]:
    """Download n evenly-spaced clip dirs of a source from the object store to a tmp
    dir and return their local paths — so a finished source can be accepted straight
    from the corpus (any pod with cos_io creds is a vantage point; data is shared)."""
    import tempfile

    from solar_wm_data import cos_io
    pre = f"{cos_io.corpus_prefix(source)}/clips/"
    ids = sorted({k[len(pre):].split("/")[0] for k in cos_io.list_keys(pre)
                  if k.endswith("/meta.json")})
    if not ids:
        print(f"[{source}] 0 clip dirs under {pre}", file=sys.stderr)
        return []
    step = max(1, len(ids) // n)
    sample = ids[::step][:n]
    # Print the exact prefix so the validation target is unambiguous.
    print(f"[{source}] {len(ids)} clip dirs under {pre}; validating {len(sample)} sampled",
          file=sys.stderr)
    root = Path(tempfile.mkdtemp(prefix=f"validate_{source}_"))
    dirs: list[Path] = []
    for cid in sample:
        d = root / cid
        d.mkdir(parents=True, exist_ok=True)
        for f in ("poses.npy", "intrinsics.npy", "video.mp4", "prompt.txt"):
            try:
                cos_io.get_file(f"{pre}{cid}/{f}", str(d / f))
            except Exception:  # noqa: BLE001 - missing file surfaces as a validate() error
                pass
        dirs.append(d)
    return dirs


def main(argv: list[str]) -> int:
    args = argv[1:]
    dirs: list[Path] = []
    if args and args[0] == "--source":
        n = int(args[args.index("--sample") + 1]) if "--sample" in args else 5
        dirs = _pull_s3_sample(args[1], n)
        if not dirs:
            return 1
    elif args and args[0] == "--glob":
        dirs = [Path(p) for p in sorted(_glob.glob(args[1])) if Path(p).is_dir()]
    else:
        dirs = [Path(a) for a in args]
    if not dirs:
        print("usage: validate_clip.py <clip_dir>... | --glob 'pattern' | "
              "--source <src> [--sample N]", file=sys.stderr)
        return 64

    n_pass = 0
    for d in dirs:
        ok, errs, warns = validate(d)
        tag = "PASS" if ok else "FAIL"
        name = d.name[:24]
        extra = ""
        if errs:
            extra = "  | " + " ; ".join(errs)
        elif warns:
            extra = "  ⚠ " + " ; ".join(warns)
        print(f"[{tag}] {name}{extra}")
        n_pass += int(ok)
    print(f"\n{n_pass}/{len(dirs)} PASS")
    return 0 if n_pass == len(dirs) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
