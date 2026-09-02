"""Per-source ingest: normalise raw inputs into ClipRecords.

Each source has its own access layout, but they all reduce to the
same ClipRecord plus a per-source pose-annotation *mode*. The mode mapping below is
the authoritative source->mode table used by the rest of the pipeline.
"""

from __future__ import annotations

from pathlib import Path

from ..manifest import ClipRecord

# Source -> pose annotation mode: where each source's camera trajectory comes from.
# NOT every entry here can be acquired by this repository. SOURCE_MODE answers "where do
# this owner's camera poses come from", which every owner of the released corpus needs --
# including ones whose acquire is not wired here, because their already-produced clips
# still have to be judged, assembled and read. `ACQUIRE` in scripts/run_solarwm_fleet.py
# is the strictly smaller set that can be fetched from raw.
#: For a `gt_pose` source, whether the trajectory it ships is already in METRES.
#:
#: The recipe this engine follows takes metric scale from exactly two places: MoGe-2
#: metric depth (the `default` and `gt_depth` paths), or the ground-truth trajectory
#: itself (`gt_pose`). In `gt_pose` the Umeyama fit brings Pi3's structure INTO the
#: trajectory's gauge -- the trajectory is the reference, not the thing being corrected.
#: That is only sound when the trajectory is metric to begin with.
#:
#: A source whose ground truth is COLMAP-scale has no metric reference anywhere on that
#: path, so its poses come out in arbitrary units. Declaring it here makes the pose stage
#: recover a real scale from MoGe-2 and apply it, instead of emitting arbitrary units
#: under a field documented as metres.
GT_POSE_METRIC = {
    "omniworld": True,        # ships metric GT
    "multicamvideo": True,    # UE5 extrinsics, cm converted to m at ingest
    "realcam_vid": True,      # per-clip metric w2c
    "zod": True,              # GPS/IMU ego-pose, metres by construction
    # The released Sekai trajectories are authoritative c2w, but their translation is
    # normalized per clip rather than expressed in metres. Preserve their rotations and
    # trajectory shape, then recover one metric translation scale from Pi3 + MoGe-2.
    "sekai_game": False,
    "abot": False,            # COLMAP text model, no points3D, undeclared units
}


def gt_pose_is_metric(source: str) -> bool:
    """Whether `source`'s ground-truth trajectory is already in metres.

    Raises for an undeclared `gt_pose` source. There is deliberately no default: the two
    possible defaults are both wrong. Assuming metric emits arbitrary units under a field
    documented as metres, which is invisible downstream; assuming non-metric silently
    rescales a trajectory that was already correct. A new source has to say which it is.
    """
    if SOURCE_MODE.get(source) != "gt_pose":
        raise ValueError(f"{source!r} is not a gt_pose source")
    if source not in GT_POSE_METRIC:
        raise KeyError(
            f"gt_pose source {source!r} does not declare whether its ground-truth "
            f"trajectory is metric. Add it to GT_POSE_METRIC: True if the source ships "
            f"metres, False if its poses are COLMAP-scale or in undeclared units (the "
            f"pose stage will then recover a real scale from MoGe-2 and apply it).")
    return GT_POSE_METRIC[source]


SOURCE_MODE = {
    "spatialvid": "default",     # internet video, VIPE default
    # dl3dv was gt_pose until 2026-08-07. Its COLMAP GT exists only at ~4-5 Hz (an even
    # extraction over the whole source video), so pose-indexed clips came out as 3-7x
    # timelapses and the source had to be exempted from every motion gate to survive.
    # Interpolating that GT up to 24fps would fabricate 4 of every 5 poses, so we
    # estimate from the video instead and cut
    # contiguous native-step windows. DL3DV — static, textured, smoothly orbited — is
    # the best case for SLAM. Nothing consumes DL3DV's ColmapCache any more.
    "dl3dv": "default",          # VIPE-estimated camera (was gt_pose; see above)
    "omniworld": "gt_pose",      # OmniWorld provides metric GT camera poses; we use
                                 # them directly (higher quality + reuses the validated
                                 # gt_pose path) rather than VIPE + GT depth.
    "sekai_game": "gt_pose",     # normalized GT poses + Pi3/MoGe metric bridge
    "sekai_walking": "default",  # real walking video, VIPE default
    "miradata": "default",       # long real video, VIPE default
    # MIND ships COLMAP poses for its TEST subsets only. Using them would make pose
    # provenance vary between one owner's own splits, with nothing in a clip to say
    # which it got, so every MIND clip is estimated instead.
    "mind": "default",

    # --- sources beyond the original seven, classified by the GT each ships, by the
    # priority pose > depth > nothing. Action-annotation datasets (keyboard/mouse or
    # controller traces) are deliberately excluded: an action is not a camera pose, so
    # they carry no camera-control signal.
    "multicamvideo": "gt_pose",  # ReCamMaster MultiCamVideo: UE5 GT extrinsics + intrinsics
    "realcam_vid": "gt_pose",    # RealCam-Vid: per-clip metric w2c + intrinsics in npz (MonST3R-derived)
    "zod": "gt_pose",            # Zenseact: GPS/IMU ego-pose + camera calib -> metric camera trajectory
    "openvid": "default",        # OpenVid-1M: T2V, video-only -> VIPE
    "vidgen": "default",         # VidGen-1M: T2V, video-only -> VIPE
    "ditto": "default",          # Ditto-1M: edit triplets, video-only -> VIPE

    # ABot-World-Explorer-500h. 30,969 UE-rendered exploration episodes,
    # 1080p/30fps/60s. Ships a PER-FRAME COLMAP text model (every frame posed, not
    # keyframes) -> gt_pose. Its COLMAP carries no points3D and undeclared translation
    # units, so the trajectory is GT in shape but arbitrary in scale; annotate_pose's
    # Pi3 + MoGe-2 bridge is what makes it metric, exactly as for sekai_game.
    "abot": "gt_pose",
}


def mode_for(source: str) -> str:
    if source not in SOURCE_MODE:
        raise KeyError(f"unknown source '{source}'; known: {sorted(SOURCE_MODE)}")
    return SOURCE_MODE[source]


def _probe_video(path: Path) -> dict:
    """Best-effort (fps, num_frames, w, h) probe. Tries decord then OpenCV."""
    try:
        import decord  # type: ignore
        vr = decord.VideoReader(str(path))
        f0 = vr[0].asnumpy()
        return {"fps": float(vr.get_avg_fps()) or None, "num_frames": len(vr),
                "height": int(f0.shape[0]), "width": int(f0.shape[1])}
    except Exception:
        pass
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return {}
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS) or None,
            "num_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None,
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None,
        }
        cap.release()
        return info
    except Exception:
        return {}


def ingest_clip_dir(clip_dir: str | Path, source: str, clip_id: str | None = None,
                    fallback: dict | None = None) -> ClipRecord:
    """Ingest a single clip directory laid out as ``{video.mp4, [poses.npy], ...}``.

    Matches the converted per-clip layout the training reader expects,
    so the pipeline can run directly over already-converted clips.
    """
    clip_dir = Path(clip_dir)
    video = clip_dir / "video.mp4"
    if not video.exists():  # tolerate alternate names
        cands = list(clip_dir.glob("*.mp4"))
        video = cands[0] if cands else video
    cid = clip_id or clip_dir.name
    rec = ClipRecord(
        clip_id=cid, source=source, video_path=str(video.resolve()),
        mode=mode_for(source),
    )
    info = _probe_video(video) or (fallback or {})
    for k, v in info.items():
        if v:
            setattr(rec, k, v)
    # carry GT pose/depth hints for the pose stage if present
    if (clip_dir / "poses.npy").exists():
        rec.extra["gt_positions_path"] = str((clip_dir / "poses.npy").resolve())
    if (clip_dir / "intrinsics.npy").exists():
        rec.extra["gt_intrinsics_path"] = str((clip_dir / "intrinsics.npy").resolve())
    if (clip_dir / "gt_depth.npz").exists():
        rec.extra["gt_depth_path"] = str((clip_dir / "gt_depth.npz").resolve())
    # Optional companions, carried through untouched when the SOURCE has them and simply
    # absent when it does not. We never synthesise either: a fabricated audio track or an
    # invented action stream is worse than the field being missing, because a consumer
    # cannot tell the difference.
    if (clip_dir / "audio.m4a").exists():
        rec.extra["audio_path"] = str((clip_dir / "audio.m4a").resolve())
    if (clip_dir / "action.npy").exists():
        rec.extra["action_path"] = str((clip_dir / "action.npy").resolve())
    return rec


def ingest_source(root: str | Path, source: str, fallback: dict | None = None) -> list[ClipRecord]:
    """Ingest every clip subdirectory under ``root`` for one source."""
    root = Path(root)
    records = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        records.append(ingest_clip_dir(d, source, fallback=fallback))
    return records
