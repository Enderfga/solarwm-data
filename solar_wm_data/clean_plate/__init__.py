"""LTX Clean Plate — the frozen contract and the deterministic task rules.

Dynamic people and vehicles are a standing ambiguity for a camera-controlled world model:
they add motion the camera does not explain and the user cannot control. This stage
removes them while preserving the source camera path, timing, caption, camera arrays and
optional audio window, producing additional dataset owners — a clean clip is a SIBLING of
its source, never a silent substitution for it.

Network-free and GPU-free on purpose: the planning rules and the lineage check live here
so they are testable without weights, and `scripts/clean_plate_*.py` do the rest.

RE-MEASURE, DO NOT INHERIT. A clean output does not inherit its source's quality record.
Removal can introduce texture artifacts, temporal discontinuities and altered motion
statistics, so caption, visual metrics, semantic metrics and camera diagnostics are ALL
recomputed on every output window. Metric scale is restored from deterministic
source-frame correspondences; an ambiguous reconstruction fails closed rather than
falling back to the source's scale.

WHAT IT IS NOT GOOD AT. Clean Plate is a generative reconstruction. It is most reliable
when the hidden background appears elsewhere in the clip; large persistent occluders, fine
text, reflections and repeated textures still need a human to look.
"""

from __future__ import annotations

LTX_SOURCE_REVISION = "9377758131b1ffde4b7f766804590a6617bf2ab9"

# Frozen generation contract. Changing any of it makes a NEW owner, not an update.
DENOISE_STEPS = 8
STRENGTH = 1.0
SEED = 42
MODEL_SIZE = (1248, 704)     # the model path runs here ...
MODEL_FPS = 24               # ... at 24 fps ...
OUTPUT_SIZE = (1280, 720)    # ... and the committed clip is Lanczos-converted to here ...
OUTPUT_FPS = 16              # ... at 16 fps.

#: Committed output length -> model frame count. The 4n+1 convention lives HERE, on the
#: model side (121 = 4*30+1, 241 = 4*60+1), which is why the committed 160-frame clip is
#: not 4n+1 and is not meant to be: it is 10 s of real time at 16 fps.
MODEL_FRAMES = {81: 121, 160: 241}

#: Source requirements. A source that fails any of these is rejected at manifest time,
#: before a GPU is touched.
SOURCE_FPS = 16
MIN_SOURCE_SIZE = (1280, 720)

#: Clean owners are independent recipe owners, each a sibling of its source.
CLEAN_OF = {
    "spatialvid": "spatialvid_clean",
    "miradata": "miradata_clean",
    "sekai_walking": "sekai_walking_clean",
}

#: Files a source clip must have before it can be planned.
CORE_FILES = ("video.mp4", "poses.npy", "intrinsics.npy", "meta.json", "prompt.txt")


def slice_plan(num_frames: int) -> tuple[int, int] | None:
    """Which source frames to clean: ``(start, length)``, or None to reject.

    Deterministic, so a manifest built twice is the same manifest:

        >= 160 frames  ->  the FIRST 160
        81..159        ->  the CENTERED 81
        < 81           ->  rejected

    The head is taken for long clips and the centre for short ones on purpose. A long clip
    has plenty of usable span either way, so the head keeps the choice reproducible without
    depending on the exact length; a short clip is likelier to have its content in the
    middle, and taking its head would systematically favour whatever the source happened to
    open on.
    """
    if num_frames >= 160:
        return 0, 160
    if num_frames >= 81:
        return (num_frames - 81) // 2, 81
    return None


def model_frames_for(target_frames: int) -> int:
    """Model frame count for a committed output length; raises on an unsupported length."""
    try:
        return MODEL_FRAMES[target_frames]
    except KeyError:
        raise ValueError(
            f"no model frame count for a {target_frames}-frame output; "
            f"supported: {sorted(MODEL_FRAMES)}") from None


def source_rejections(fps: float | None, width: int | None, height: int | None,
                      num_frames: int | None, pose_frames: int | None,
                      intr_frames: int | None) -> list[str]:
    """Why this source clip cannot be cleaned; empty list means it can.

    Frame-count agreement is checked here rather than trusted, because the whole point of
    the stage is to slice video, poses, intrinsics and audio from the SAME interval. A
    source whose arrays already disagree cannot be sliced coherently, and the error would
    otherwise surface as a clean clip whose poses belong to a neighbouring interval —
    which looks completely normal and is silently wrong.
    """
    out: list[str] = []
    if fps is None or abs(fps - SOURCE_FPS) > 0.01:
        out.append(f"fps={fps} != {SOURCE_FPS}")
    if width is None or height is None:
        out.append("resolution unknown")
    else:
        if width < height:
            out.append(f"not landscape ({width}x{height})")
        if width < MIN_SOURCE_SIZE[0] or height < MIN_SOURCE_SIZE[1]:
            out.append(f"{width}x{height} below {MIN_SOURCE_SIZE[0]}x{MIN_SOURCE_SIZE[1]}")
    if num_frames is None:
        out.append("num_frames unknown")
    elif slice_plan(num_frames) is None:
        out.append(f"num_frames={num_frames} < 81")
    for name, n in (("poses", pose_frames), ("intrinsics", intr_frames)):
        if n is not None and num_frames is not None and n != num_frames:
            out.append(f"{name} has {n} frames, video has {num_frames}")
    return out


def kept_source_reasons(clean_meta: dict, kept_source_ids: "set[str] | frozenset[str]",
                        source_owner: str) -> list[str]:
    """Why this clean clip's SOURCE disqualifies it; empty list means the lineage holds.

    Separate from `check_lineage` because it is a different question. That one asks whether
    the clean clip really is the interval it claims to be; this one asks whether that
    interval was itself accepted — a clean clip whose source the policy rejected inherits
    the rejection, however good the clean output measures.

    It cannot live in `configs/filters_released.yaml` with the other gates: those are metric
    ranges over one clip's own record, and this needs the source owner's whole kept set.
    Leaving it out is not harmless — it silently promotes clips the frozen policy rejected
    (measured: 39 of 961 sampled miradata-clean rows, ~4%, all of them scoring well enough
    to pass every metric gate).

    The reason string matches the one the corpus records, so a rebuild's verdicts can be
    diffed against the published ones without a translation table.
    """
    src_id = clean_meta.get("source_clip_id")
    if not src_id:
        return ["clean meta records no source_clip_id"]
    if src_id not in kept_source_ids:
        return [f"recipe_lineage_not_in_{source_owner}_kept_source"]
    return []


def check_lineage(source_meta: dict, clean_meta: dict) -> list[str]:
    """Why a clean output's lineage does not hold; empty list means it does.

    Cheap, GPU-free, and the check that stops a plausible-looking but misaligned clean clip
    from reaching the corpus. Run it on the whole delivery before publishing, not on a
    sample: a misaligned clip is invisible to every quality metric.
    """
    out: list[str] = []
    src_id = clean_meta.get("source_clip_id")
    if not src_id:
        out.append("clean meta records no source_clip_id")
    elif source_meta.get("clip_id") and src_id != source_meta["clip_id"]:
        out.append(f"source_clip_id={src_id} != {source_meta['clip_id']}")

    start, length = clean_meta.get("source_start_frame"), clean_meta.get("num_frames")
    src_n = source_meta.get("num_frames")
    if start is None or length is None:
        out.append("clean meta records no source interval")
    elif src_n is not None:
        if start < 0 or start + length > src_n:
            out.append(f"interval [{start},{start + length}) outside source 0..{src_n}")
        expect = slice_plan(src_n)
        if expect is not None and (start, length) != expect:
            out.append(f"interval {(start, length)} is not the planned {expect}")

    if clean_meta.get("fps") not in (None, OUTPUT_FPS):
        out.append(f"fps={clean_meta['fps']} != {OUTPUT_FPS}")
    w, h = clean_meta.get("width"), clean_meta.get("height")
    if (w, h) != (None, None) and (w, h) != OUTPUT_SIZE:
        out.append(f"{w}x{h} != {OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}")
    return out
