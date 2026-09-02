"""Corpus output spec — the single source of truth for clip length AND frame rate.

Five specs ship, in two lineages. Each one fixes BOTH numbers, because a length without
its frame rate is not a spec: the same 81 frames is a 5.06 s clip at 16 fps and a 3.38 s
clip at 24 fps.

    spec      frames   fps   seconds   lineage
    5s          121     24    5.0417   default — the 24 fps rebuild
    60s        1437     24   59.875    the 24 fps rebuild, minute-scale
    81f          81     16    5.0625   the published corpus, short
    160f        160     16   10.0      the published corpus, ~10 s
    960f        960     16   60.0      the published corpus, minute-scale

CANONICAL LENGTH vs MODEL WINDOW — these are different things and the corpus reports them
separately. The numbers above are CANONICAL lengths: what a run emits per clip, measured
against the released corpus (dl3dv-10s and most of spatialvid are 160; abot, dl3dv-60s,
miradata, sekai_walking and mind are 960; multicamvideo and most of omniworld are 81).
A source that runs out early yields a shorter clip and that is legal — we never upsample.

The 4n+1 lengths a reader will also see quoted (81, 153, 957) are MODEL WINDOWS cut from
those clips downstream, which is why the corpus's own frame buckets are <81, 81-152,
153-956 and >=957: each boundary is "which window can this clip still serve". Only 81 is
both. Express a window inline when you need one — ``153@16``, ``957@16``.

The 24 fps lineage is 4n+1 at the clip level because it was cut that way; the 16 fps
canonical lengths are round seconds (160 = 10 s, 960 = 60 s) and are NOT 4n+1.

BOTH RATES ARE SUPPORTED. Neither lineage is deprecated; `5s` is only the DEFAULT. The
24 fps rebuild is what fixed DL3DV, whose COLMAP ground truth exists only at ~4-5 Hz:
indexing clips by that GT produced 3-7x timelapses that every motion gate rejected
wholesale, so the rebuild cuts contiguous native-step windows and estimates the camera from
the video instead. The 16 fps specs reproduce the published corpus with the same code.

The one asymmetry worth knowing: `configs/filters.yaml` is calibrated against the 24 fps
measured distributions, so a 16 fps run reproduces the geometry but not the published
selection policy — that needs its own calibrated threshold set.

WHY 160 AND 960 ARE NOT 4n+1, AND WHY THAT IS FINE. They are canonical clip lengths, not
model windows — 10 s and 60 s of real time at 16 fps. Clean Plate makes the same split
visible from the other side: its MODEL frame counts are 4n+1 (121 for an 81-frame output,
241 for a 160-frame one, at 24 fps) while the committed clip is the 16 fps result.
``is_latent_aligned`` reports the fact without judging it.

ONE fleet run emits ONE spec, selected by ``SOLAR_WM_SPEC``. Runs are kept apart by
``SOLAR_WM_RUN_ID``, which namespaces the corpus prefix AND the done-markers
(``cos_io.corpus_prefix``), so clip ids carry no spec suffix and runs cannot collide.

This module exists so the producer (``run_solarwm_fleet``) and the validator
(``validate_clip``) read the SAME numbers. A spec duplicated across call sites
is exactly the "stale default silently re-applies" failure mode: one site gets updated,
the other keeps emitting off-spec clips, and nothing errors.
"""

from __future__ import annotations

import os

# 60s is 1437, NOT the 1441 that "just under a minute, 4n+1" first suggests.
# A window is cut from target_seconds() of SOURCE time, and 1441/24 = 60.0417 s is LONGER
# than a minute — while the sources that ship "60 second" clips ship exactly 60.000 s:
# an ABot episode measures 1800 frames at 30 fps, a Sekai-Walking segment the same. At 1441
# the required source span is 1801 frames, one more than exists, so every one of those clips
# yields ZERO windows — abot by raising (its item then never gets a done-marker and retries
# forever), the default-mode sources by silently emitting nothing. The 60s corpus would have
# come out nearly empty, and nothing in the run would have said why.
# 1437 = 4*359+1 is equally latent-aligned and spans 59.875 s, so it fits inside a 60.000 s
# source at any source rate >= 24 fps with integer-exact resampling and no fabricated frames.
SPEC_FRAMES = {"5s": 121, "60s": 1437, "81f": 81, "160f": 160, "960f": 960}

# The frame rate belongs to the spec so frame count and duration cannot drift apart.
# ``SOLAR_WM_TARGET_FPS`` remains an explicit experimental override.
SPEC_FPS = {"5s": 24, "60s": 24, "81f": 16, "160f": 16, "960f": 16}

DEFAULT_SPEC = "5s"


def parse_spec(name: str) -> tuple[int, int]:
    """Resolve a spec name to (frames, fps).

    Accepts a name from SPEC_FRAMES, or an inline ``<frames>@<fps>`` (e.g. ``241@24``)
    so a run is not limited to the catalogue. Both fps families are first-class: nothing
    here prefers 24 over 16, only DEFAULT_SPEC does.
    """
    name = (name or "").strip()
    if name in SPEC_FRAMES:
        return SPEC_FRAMES[name], SPEC_FPS[name]
    if "@" in name:
        head, _, tail = name.partition("@")
        try:
            frames, fps = int(head), int(tail)
        except ValueError:
            raise ValueError(f"inline spec must be <frames>@<fps>, got {name!r}") from None
        if frames < 1 or fps < 1:
            raise ValueError(f"inline spec needs positive frames and fps, got {name!r}")
        return frames, fps
    raise ValueError(
        f"unknown spec {name!r}: use one of {sorted(SPEC_FRAMES)} or an inline <frames>@<fps>"
    )


def is_latent_aligned(frames: int) -> bool:
    """Is this length 4n+1, the shape a latent-aligned model window takes?

    NOT every catalogue spec is: the 24 fps lineage is 4n+1 because it was cut that way,
    while 160f and 960f are round seconds at 16 fps and are not. That is a property to
    report, not a rule to enforce — a hard check here would reject lengths the corpus
    genuinely contains. ``spec show`` and the fleet's startup banner report it.
    """
    return frames >= 1 and (frames - 1) % 4 == 0


def current_spec() -> str:
    """The active spec NAME (catalogue entry or inline). Validated on read."""
    name = os.environ.get("SOLAR_WM_SPEC", DEFAULT_SPEC).strip() or DEFAULT_SPEC
    parse_spec(name)          # raises on anything unusable
    return name


def target_frames() -> int:
    return parse_spec(current_spec())[0]


def target_fps() -> int:
    """Frame rate of the active spec. SOLAR_WM_TARGET_FPS overrides it deliberately."""
    return int(os.environ.get("SOLAR_WM_TARGET_FPS") or parse_spec(current_spec())[1])


def target_seconds() -> float:
    """Real duration one spec clip spans. Source windows are cut to THIS many seconds of
    source time, then resampled to ``target_frames()`` — duration in == duration out.
    Sizing a source window by anything else changes playback speed."""
    return target_frames() / target_fps()


def spec_of_frames(n: int) -> str | None:
    """Which spec a clip of ``n`` frames belongs to, or None if it is not a spec length.

    Lengths are unique across all five specs, so a stored clip identifies its own spec
    without needing the environment that produced it. That is what lets a validator check
    a clip against the fps it was actually emitted at rather than the fps of whatever run
    happens to be reading it.
    """
    for name, frames in SPEC_FRAMES.items():
        if frames == n:
            return name
    return None
