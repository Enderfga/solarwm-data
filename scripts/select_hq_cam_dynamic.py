#!/usr/bin/env python3
"""Select the high-quality, genuinely camera-dynamic training subset from the
SolarWM corpus.

This streams every clip of the requested sources from object storage, reads
``meta.json`` (quality + camera metrics) and ``poses.npy`` (camera-to-world SE3),
computes a small set of trajectory-motion features from the camera centers, then
applies a quality gate AND a camera-dynamic gate. The kept clips are written to
``hq_train_list.jsonl`` and per-source kept counts + total are printed.

WHY A SEPARATE SELECTOR (vs. the pipeline ``kept`` flag / assemble_corpus.py):
the per-clip ``kept`` verdict and the assembler both gate on *quality* metrics
(DOVER / unimatch / scene-cuts / camera-intrinsic sanity). NEITHER looks at the
actual camera trajectory. A clip can pass every quality gate while the camera
barely translates (near-static large scene). For world-model training we want the
genuinely camera-DYNAMIC subset, so this adds trajectory gates on top.

POSE-SCALE CAVEAT (important):
The trajectory features (``avg_motion_m``, ``span_m``, ``path_length``) are in the
clip's own pose coordinate units, and those units are NOT comparable across sources:
  - gt_pose sources (dl3dv, omniworld, realcam_vid, multicamvideo,
    sekai_game, zod) carry metric / COLMAP-scaled poses -> spans of ~1..500.
  - default-pose sources (spatialvid, miradata, ditto, vidgen, openvid) carry
    monocular-SLAM poses normalised near unit scale -> spans of ~0.01..10.
Because of this the camera-dynamic thresholds are applied PER POSE MODE; tortuosity
(unit-free) is the one cross-source-comparable shape feature.

Usage (with S3 credentials in the environment):
    SOLAR_WM_STORAGE=s3 SOLAR_WM_S3_BUCKET=<your-bucket> \\
    SOLAR_WM_CORPUS_PREFIX=corpus PYTHONPATH=. \\
        python3 scripts/select_hq_cam_dynamic.py --out /tmp/hq [--sources dl3dv,omniworld]
        [--sample 500]                  # per-source cap (audit)
        [--shard-index 0 --shard-count 8]  # trivial sharding by clip-hash
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

WM = os.environ.get("SOLAR_WM_ROOT") or str(Path(__file__).resolve().parents[1])
sys.path.insert(0, WM)

from solar_wm_data import cos_io  # noqa: E402

# Pose mode per source (gt_pose = metric/COLMAP-scaled poses; default = monocular
# SLAM, near-unit-normalised). Mirrors solar_wm_data.ingest.SOURCE_MODE but kept
# local so the selector runs even if that import path drifts; falls back to the
# real table when available.
_FALLBACK_MODE = {
    "dl3dv": "gt_pose", "omniworld": "gt_pose",
    "realcam_vid": "gt_pose", "multicamvideo": "gt_pose", "sekai_game": "gt_pose",
    "zod": "gt_pose", "sekai_walking": "gt_pose",
    "spatialvid": "default", "miradata": "default", "ditto": "default",
    "vidgen": "default", "openvid": "default",
}
try:
    from solar_wm_data.ingest import SOURCE_MODE as _SM  # noqa: E402
    SOURCE_MODE = {**_FALLBACK_MODE, **dict(_SM)}
except Exception:  # noqa: BLE001
    SOURCE_MODE = dict(_FALLBACK_MODE)

# Default source set: gt_pose primary + best default-pose; ditto/vidgen included but
# strictly gated (only their genuinely dynamic, high-quality tail survives).
DEFAULT_SOURCES = [
    "dl3dv", "omniworld", "realcam_vid", "multicamvideo",
    "sekai_game", "zod", "sekai_walking",              # gt_pose
    "spatialvid", "miradata",                            # best default-pose
    "ditto", "vidgen", "openvid",                        # strictly filtered
]


def pose_mode(src: str) -> str:
    return SOURCE_MODE.get(src, "default")


def traj_metrics(poses: np.ndarray) -> dict | None:
    """Camera-trajectory motion features from the c2w centers t = poses[:, :3, 3]."""
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or poses.shape[0] < 2:
        return None
    t = poses[:, :3, 3].astype(np.float64)
    if not np.isfinite(t).all():
        return None
    seg = np.linalg.norm(np.diff(t, axis=0), axis=1)
    path_length = float(seg.sum())
    span_m = float(np.linalg.norm(t.max(axis=0) - t.min(axis=0)))  # bbox diagonal
    n = t.shape[0]
    avg_motion_m = path_length / (n - 1)
    tortuosity = path_length / (span_m + 1e-6)
    # Real-cut detector from the trajectory itself. A genuine scene cut teleports the
    # camera -> one inter-frame segment dwarfs the rest. PySceneDetect's scene_cuts is
    # unreliable on fast continuous capture/render (false positives on orbit motion);
    # the pose jump is fps- and source-agnostic. median(seg) guards against the static
    # case where a tiny absolute jump would still ratio-explode.
    med = float(np.median(seg)) if seg.size else 0.0
    max_seg = float(seg.max()) if seg.size else 0.0
    jump_ratio = max_seg / (med + 1e-6)
    # Per-frame camera ANGULAR velocity (deg between consecutive EMITTED frames at
    # 16fps), from the c2w rotation blocks. This is the LEARNABILITY axis the
    # translation/pixel-flow features miss: a camera that rotates a large angle per
    # frame yields near-decorrelated consecutive frames the model cannot learn
    # (timelapse / genuinely-too-fast capture). Unlike span_m it is scale-free
    # (degrees), so it is the one motion bound directly comparable across pose modes.
    R = poses[:, :3, :3].astype(np.float64)
    Rrel = np.matmul(np.transpose(R[:-1], (0, 2, 1)), R[1:])   # R_i^T R_{i+1}
    cos_a = np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos_a))                         # deg/frame, length n-1
    ang_med = float(np.median(ang)) if ang.size else 0.0
    ang_p95 = float(np.percentile(ang, 95)) if ang.size else 0.0
    ang_max = float(ang.max()) if ang.size else 0.0
    return dict(span_m=span_m, path_length=path_length,
                avg_motion_m=avg_motion_m, tortuosity=tortuosity, n_frames=n,
                max_seg=max_seg, jump_ratio=jump_ratio,
                ang_med=ang_med, ang_p95=ang_p95, ang_max=ang_max)


def quality_ok(meta: dict, a, mode: str):
    """Visual-quality gate. Returns (ok, reason)."""
    m = (meta.get("metrics") or {})
    cam = (meta.get("camera") or {})
    dt, da = m.get("dover_tech"), m.get("dover_aes")
    if dt is None or da is None:
        return False, "dover_missing"
    # DOVER distributions differ by pose mode: geometric gt_pose footage (real-world
    # COLMAP captures / synthetic renders) scores lower on the aesthetic head than
    # curated T2V clips, yet its camera poses are exactly what we want -> a lower floor
    # for gt_pose, a strict floor for the default-pose internet-video sources.
    dover_floor = a.dover_min_gt if mode == "gt_pose" else a.dover_min_default
    if (dt + da) / 2.0 < dover_floor:
        return False, "dover_low"
    # unimatch (timed 0.5s optical flow) is used ONLY as a low anti-frozen-IMAGE floor
    # here (duplicate/near-static frames); the camera-MOTION floor lives in dynamic_ok
    # (rotation OR translation), and the too-fast CEILING is the angular gate. The 2026-06
    # forensic confirmed the stored value is NOT fps-confounded (that was avg_motion_m).
    uni = m.get("unimatch")
    if uni is None or uni < a.unimatch_min:
        return False, "frozen_img"
    # scale_cov gates only the noisy default-pose VIPE intrinsics (zoom/focus drift
    # corrupts the pose<->motion coupling). gt_pose is EXEMPT: its scale is a single
    # Umeyama/COLMAP scalar broadcast (cov 0 by construction), and a synthetic source
    # whose Pi3 scale recovery returned all-zeros would otherwise fail-close here.
    if mode != "gt_pose":
        sco = cam.get("scale_cov")
        if sco is not None and sco >= a.scale_cov_max:
            return False, "scale_cov"
    return True, "ok"


def dynamic_ok(tm: dict, mode: str, a):
    """Camera-motion gate -> (ok, reason). Combines a NOT-FROZEN floor (rotation OR
    translation), a NOT-TOO-FAST ceiling (per-frame angular velocity + translational/
    rotational cut), and a length floor. Rotation (deg/frame) is scale-free and uniform
    across pose modes; translation span is per-mode (gt_pose metric ~1..500, default
    SLAM-normalised ~0.4..80). Calibrated on the 2026-06-20 corpus scan."""
    # Length floor: a clip shorter than the training sequence window cannot fill a
    # training sample (e.g. a 63-frame vidgen scene-split vs an 81-frame window).
    if tm["n_frames"] < a.min_frames:
        return False, "short"
    # NOT-FROZEN floor: perceptible ROTATION (scale-free deg/frame) OR TRANSLATION
    # (per-mode span). OR-combined so a rotation-dominant orbit (multicam, tiny span)
    # AND a translation-dominant dolly (zod, ~0 rotation) both pass, while a static
    # camera with only object motion fails BOTH and is dropped. This replaces the old
    # span-only gate that wrongly nuked smooth-slow gt_pose (multicam/sekai_game -> 0%).
    span_min = a.gt_span_min if mode == "gt_pose" else a.def_span_min
    if tm["ang_med"] < a.rot_floor and tm["span_m"] < span_min:
        return False, "static_cam"
    # Real-cut gate (translational teleport): one inter-frame segment dominates the span.
    if tm["span_m"] > 0 and (tm["max_seg"] / tm["span_m"]) > a.jump_frac_max:
        return False, "jump_cut"
    # (tortuosity gate removed: path/span dips below 1 as a numeric artifact on
    #  small-motion clips and wrongly cut premium multicam clips that pass the motion
    #  floor; "frozen" is already caught by the rotation-or-span floor above.)
    # NOT-TOO-FAST ceiling (the learnability axis span_m/pixel-flow miss; deg/frame at
    # 16fps, scale-free, uniform across pose modes):
    #   ang_med caps a UNIFORMLY-too-fast/timelapse clip (dl3dv ~3.9 deg/frame, ~10x the
    #     smooth sources -> cut to its slow ~35%, clean sources stay ~100%).
    #   ang_max caps a SINGLE-frame rotational teleport (cut / pose glitch) the
    #     translation-only jump_frac misses (miradata 105 deg, rendered orbits 180 deg).
    if tm["ang_med"] > a.ang_med_max:
        return False, "too_fast"
    if tm["ang_max"] > a.ang_max_max:
        return False, "rot_cut"
    return True, "ok"


def clip_ids(src: str, sample: int, shard_index: int, shard_count: int):
    pre = f"{cos_io.corpus_prefix(src)}/clips/"
    seen, ids = set(), []
    for k in cos_io.list_keys(pre):
        cid = k[len(pre):].split("/")[0]
        if not cid or cid in seen:
            continue
        seen.add(cid)
        if shard_count > 1:
            h = int(hashlib.md5(cid.encode()).hexdigest(), 16)
            if h % shard_count != shard_index:
                continue
        ids.append(cid)
        if sample and len(ids) >= sample:
            break
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=".", help="dir for hq_train_list.jsonl")
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    ap.add_argument("--sample", type=int, default=0, help="per-source clip cap (audit)")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    # ---- quality gate (DOVER floor split by pose mode; see quality_ok) ----
    ap.add_argument("--dover_min_gt", type=float, default=0.25,
                    help="(dover_tech+dover_aes)/2 floor for gt_pose sources (synthetic/"
                         "geometric footage under-scores DOVER's aesthetic head)")
    ap.add_argument("--dover_min_default", type=float, default=0.50,
                    help="(dover_tech+dover_aes)/2 floor for default-pose sources")
    ap.add_argument("--unimatch_min", type=float, default=5.0,
                    help="anti-frozen-IMAGE floor only (duplicate/near-static frames); the "
                         "camera-MOTION floor is rotation-or-span in dynamic_ok")
    ap.add_argument("--scale_cov_max", type=float, default=0.10,
                    help="camera-intrinsic-stability ceiling; trims default-pose sources "
                         "with unstable intrinsics (zoom/focus drift corrupts pose<->motion). "
                         "gt_pose reports ~0 so always passes.")
    # ---- camera-dynamic gate (calibrated against the 2026-06-18 corpus scan) ----
    # span_m is the primary gate (fps-independent); avg_motion is only a frozen floor.
    ap.add_argument("--gt_span_min", type=float, default=1.0,
                    help="metric-unit min camera TRANSLATION for gt_pose; OR-ed with the "
                         "scale-free rotation floor so rotation-dominant orbits still pass")
    ap.add_argument("--def_span_min", type=float, default=0.5,
                    help="SLAM-normalised min camera TRANSLATION for default-pose; OR-ed "
                         "with the rotation floor")
    ap.add_argument("--rot_floor", type=float, default=0.1,
                    help="min MEDIAN per-frame camera rotation (deg/frame) to count as "
                         "camera-dynamic when translation span is below span_min; a clip "
                         "below BOTH is a static/frozen camera (object-motion-only) -> drop")
    ap.add_argument("--motion_floor", type=float, default=0.001,
                    help="reject only frozen cameras; kept low to protect slow GS renders")
    ap.add_argument("--jump_frac_max", type=float, default=0.5,
                    help="max single-frame segment / span; >this = real scene cut, reject")
    ap.add_argument("--tortuosity_min", type=float, default=1.0,
                    help="reject degenerate zero-length / pure-jitter paths (straight "
                         "forward dollies like zod have tort~1.0 and are kept)")
    # ---- per-frame angular-velocity gate (the learnability / too-fast axis; deg/frame
    #      at 16fps, scale-free so uniform across pose modes; 2026-06-20 scan calibration) ----
    ap.add_argument("--ang_med_max", type=float, default=3.0,
                    help="max MEDIAN per-frame camera rotation (deg/frame); caps a "
                         "uniformly-too-fast/timelapse clip (dl3dv median ~3.9 -> ~37%% kept; "
                         "clean gt_pose sources stay ~100%%).")
    ap.add_argument("--ang_max_max", type=float, default=25.0,
                    help="max SINGLE-frame camera rotation (deg); a rotational teleport = "
                         "real cut / pose glitch the translation jump_frac misses.")
    ap.add_argument("--min_frames", type=int, default=81,
                    help="reject clips shorter than the training sequence window; a "
                         "63-frame scene-split cannot fill an 81-frame training sample.")
    # ---- per-source cap (keeps the huge default-pose sources from dominating; lands
    #      the total in the target band, honours the gt_pose-primary priority) ----
    ap.add_argument("--max_per_source", type=int, default=0,
                    help="cap kept clips per source (0 = NO cap; process all eligible). "
                         "Default 0: emit the full Priority-1 trainable set, no balancing.")
    ap.add_argument("--no_quality_gate", action="store_true",
                    help="skip the DOVER/unimatch/scale_cov quality gate entirely and keep only the "
                         "camera-DYNAMIC gate. Use when the input is ALREADY quality-curated upstream "
                         "and the local store carries no quality metrics (a reproduce corpus: "
                         "the assembled kept set, reproduced WITHOUT re-running filters, "
                         "so meta.metrics is null by design — re-gating here would reject everything).")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = [s.strip() for s in a.sources.split(",") if s.strip()]
    c, b = cos_io.client()
    pool = ThreadPoolExecutor(max_workers=a.threads)

    out_path = out / "hq_train_list.jsonl"
    fout = open(out_path, "w")
    per_src, total, reject_report = {}, 0, {}

    def fetch(cid, _pre):
        base = f"{_pre}{cid}/"
        try:
            # backend-agnostic read (works for cos / s3 / local); the S3-only
            # client.get_object(Bucket,Key) interface is absent on the local fs backend.
            meta = json.loads(cos_io.get_bytes(base + "meta.json"))
            pb = cos_io.get_bytes(base + "poses.npy")
            poses = np.load(io.BytesIO(pb))
            return cid, meta, poses
        except Exception:  # noqa: BLE001 - mid-upload / missing object; counted, not fatal
            return cid, None, None

    import itertools

    def _chunks(it, n):
        it = iter(it)
        while True:
            c = list(itertools.islice(it, n))
            if not c:
                return
            yield c

    for src in sources:
        mode = pose_mode(src)
        ids = clip_ids(src, a.sample, a.shard_index, a.shard_count)
        pre = f"{cos_io.corpus_prefix(src)}/clips/"
        kept = n_err = n_seen = 0
        cap_hit = False
        rej = collections.Counter()
        # Read clips lazily in chunks and STOP a source as soon as its cap is
        # reached -- otherwise a 715K-clip source is fully read just to keep
        # ~3500 (the dominant cost). Over-reads at most one chunk past cap.
        for chunk in _chunks(ids, 4000):
            if cap_hit:
                break
            for cid, meta, poses in pool.map(lambda cid, _p=pre: fetch(cid, _p), chunk):
                n_seen += 1
                if meta is None or poses is None:
                    n_err += 1
                    continue
                tm = traj_metrics(poses)
                if tm is None:
                    rej["bad_poses"] += 1
                    continue
                if a.no_quality_gate:
                    qok, qwhy = True, "ok"   # quality curated upstream; local metrics are null
                else:
                    qok, qwhy = quality_ok(meta, a, mode)
                if not qok:
                    rej[qwhy] += 1
                    continue
                dok, dwhy = dynamic_ok(tm, mode, a)
                if not dok:
                    rej[dwhy] += 1
                    continue
                if a.max_per_source and kept >= a.max_per_source:
                    cap_hit = True
                    break
                m = meta.get("metrics") or {}
                # null-safe: on a reproduce corpus DOVER/unimatch can be null (quality
                # was gated upstream, not recomputed) -> record 0.0 rather than crash.
                dover = ((m.get("dover_tech") or 0.0) + (m.get("dover_aes") or 0.0)) / 2.0
                rec = {
                    "src": src,
                    "id": cid,
                    "s3_prefix": f"{pre}{cid}/",
                    "pose_mode": mode,
                    "n_frames": tm["n_frames"],
                    "width": meta.get("width"),
                    "height": meta.get("height"),
                    "avg_motion_m": round(tm["avg_motion_m"], 6),
                    "span_m": round(tm["span_m"], 4),
                    "tortuosity": round(tm["tortuosity"], 4),
                    "jump_frac": round(tm["max_seg"] / (tm["span_m"] + 1e-6), 4),
                    "ang_med": round(tm["ang_med"], 3),
                    "ang_max": round(tm["ang_max"], 2),
                    "dover": round(dover, 4),
                    "unimatch": round(float(m.get("unimatch") or 0.0), 3),
                }
                fout.write(json.dumps(rec) + "\n")
                kept += 1
                total += 1
        per_src[src] = kept
        reject_report[src] = {"mode": mode, "seen": n_seen, "kept": kept, "err": n_err,
                              "rejects": dict(rej.most_common())}
        fout.flush()
        print(f"[{time.strftime('%H:%M:%S')}] {src:14s} mode={mode:8s} "
              f"seen={n_seen} kept={kept} err={n_err} rej={rej.most_common(3)}", flush=True)

    fout.close()
    # Reject reason -> recovery priority. P2 = recoverable by engineering (re-window the
    # too-fast/cut clips at native step; re-cut the short scenes). Everything else = P3
    # discard (low quality / static camera / degenerate / frozen image).
    P2_REASONS = {"too_fast", "rot_cut", "jump_cut", "short"}
    p2 = sum(v for r in reject_report.values() for k, v in r["rejects"].items() if k in P2_REASONS)
    p3 = sum(v for r in reject_report.values() for k, v in r["rejects"].items() if k not in P2_REASONS)
    cfg_keys = ("dover_min_gt", "dover_min_default", "unimatch_min", "scale_cov_max",
                "gt_span_min", "def_span_min", "rot_floor", "jump_frac_max",
                "tortuosity_min", "ang_med_max", "ang_max_max", "min_frames", "max_per_source",
                "no_quality_gate")
    json.dump({"config": {k: getattr(a, k) for k in cfg_keys},
               "total_kept_p1": total, "p2_recoverable": p2, "p3_discard": p3,
               "per_source": reject_report},
              open(out / "select_report.json", "w"), indent=1)
    print("\n# per-source kept (P1, no cap):")
    for s in sources:
        print(f"  {s:14s} {per_src.get(s, 0)}")
    print(f"# TOTAL P1 trainable = {total}  | P2 recoverable = {p2}  | P3 discard = {p3}")
    print(f"# list -> {out_path}   report -> {out/'select_report.json'}")


if __name__ == "__main__":
    main()
