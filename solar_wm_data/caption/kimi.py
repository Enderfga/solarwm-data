"""Kimi-K2.6 annotation contract — the schema, and the validator that enforces it.

This module is deliberately network-free: it holds the frozen contract and the code that
decides whether a response conforms. `scripts/kimi_caption.py` does the sampling and the
requests and imports everything here, so there is ONE definition of the schema and the
validator can be tested without an endpoint.

The pass produces the corpus's captions and semantic scores. Only ``dense_caption``
becomes a text condition; the other five fields are stored as annotations, and
``reject_flags`` / ``scene_transition`` never modify ``kept`` or ``reject_reasons`` on
their own.

RUNTIME IDENTITY IS RECORDED, NOT PRESCRIBED. Whatever model revision and server build a
run uses travels with each record, so a re-run under a different build is visibly a
different generation rather than an invisible drift. ``RELEASE_MODEL_REVISION`` and
``RELEASE_RUNTIME`` are what produced the released corpus — the defaults, and the values
to match if you are reproducing it, not a restriction on what you may run. The prompt is part of the schema the same way: edit one byte of
``configs/kimi_prompt.txt`` and every record already annotated is provenance-stale, which
is why the runner verifies its sha256 before processing anything.

VISUAL INPUT. Sample the ENTIRE clip at 1 fps, at most 64 frames, ffmpeg round-up; max
image edge 768 px, JPEG qscale 3, no audio. Send them as separate image parts — some
gateways accept a single video part with a 200 and silently drop it, leaving prompt_tokens
at the text-only floor while the model invents a scene. Every caption still differs, so
dedup-based detection cannot see it; check prompt_tokens on the first response instead.
Clips longer than MAX_FRAMES seconds must be split before annotation: the runner raises
rather than truncating, because a caption written from the first 64 s of a longer clip
describes something the clip is not.
"""

from __future__ import annotations

import math
import re
from typing import Any

#: The model revision and server build that annotated the released corpus. Defaults for a
#: reproduction run; override both when you annotate with your own deployment.
RELEASE_MODEL_REVISION = "7eb5002f6aadc958aed6a9177b7ed26bb94011bb"
RELEASE_RUNTIME = "vllm-0.26.0+cu129"
DELIVERY_SCHEMA_VERSION = 4

# Sampling contract.
SAMPLE_FPS = 1
MAX_FRAMES = 64
MAX_IMAGE_EDGE = 768
JPEG_QSCALE = 3

# Decoding contract.
TEMPERATURE = 0
SEED = 0
THINKING = False
MAX_COMPLETION_TOKENS = 4096
ATTEMPTS = 2

TOP_KEYS = {
    "vlm_entity_density", "vlm_quality", "reject_flags",
    "scene_type", "scene_transition", "dense_caption",
}
TRANSITION_KEYS = {"label", "count", "timestamps_sec", "evidence"}
REJECT_FLAGS = {
    "text_heavy", "watermark", "ui_overlay", "blurry",
    "near_static", "low_light", "nsfw", "single_color",
}
SCENE_TYPES = {"real_world", "rendered", "game", "animation", "mixed"}
TRANSITION_LABELS = {"none", "possible", "definite"}
QUALITY_VALUES = {1.0, 2.0, 3.0, 4.0, 5.0}
CAPTION_WORDS = (60, 150)

# Wording the caption contract forbids, checked after the model answers. The prompt asks
# for all of this; the check is what makes the ask enforceable. Three groups: media and
# render vocabulary (a rendered world is described as the world it depicts), dynamic
# subjects and their actions (they are not persistent scene content), and camera or
# viewpoint language (the text condition must not leak the control signal). Speculative
# filler is last — it pads to the word count without adding visible evidence.
FORBIDDEN_CAPTION_PATTERNS = (
    ("video_game", r"\bvideo game\b"), ("game", r"\bgame\b"),
    ("render", r"\brender(?:ed)?\b"), ("cgi", r"\bcgi\b"),
    ("animation", r"\banimat(?:ion|ed)\b"), ("screenshot", r"\bscreenshot\b"),
    ("hud", r"\bhud\b"), ("third_person_view", r"\bthird-person view\b"),
    ("video", r"\bvideo\b"), ("image", r"\bimage\b"),
    ("clip", r"\bclip\b"), ("footage", r"\bfootage\b"),
    ("person_people",
     r"\b(?:person|people|human|humans|man|men|woman|women|boy|boys|girl|girls)\b"),
    ("crowd", r"\bcrowds?\b"), ("clothing", r"\bclothing\b"),
    ("animal", r"\banimals?\b"), ("bird", r"\bbirds?\b"), ("insect", r"\binsects?\b"),
    ("driving", r"\bdriv(?:e|es|ing|en)\b"), ("passing", r"\bpass(?:es|ing|ed)?\b"),
    ("turning", r"\bturn(?:s|ing|ed)?\b"), ("arriving", r"\barriv(?:e|es|ing|ed)\b"),
    ("departing", r"\bdepart(?:s|ing|ed)?\b"), ("traffic_flow", r"\btraffic flow\b"),
    ("camera", r"\bcamera\b"), ("pan", r"\bpan(?:s|ning|ned)?\b"),
    ("tilt", r"\btilt(?:s|ing|ed)?\b"), ("zoom", r"\bzoom(?:s|ing|ed)?\b"),
    ("dolly", r"\bdolly\b"), ("orbit", r"\borbit(?:s|ing|ed)?\b"),
    ("tracking", r"\btracking\b"), ("moving_forward", r"\bmoving forward\b"),
    ("moving_backward", r"\bmoving backward\b"), ("viewpoint", r"\bviewpoints?\b"),
    ("framing", r"\bframing\b"), ("perspective", r"\bperspective changes?\b"),
    ("shot_type", r"\bshot types?\b"),
    ("likely", r"\blikely\b"), ("perhaps", r"\bperhaps\b"),
    ("suggests", r"\bsuggests\b"), ("appears_to_be", r"\bappears to be\b"),
    ("seems", r"\bseems\b"), ("ideal_for", r"\bideal for\b"),
    ("overall_appearance", r"\boverall appearance\b"),
)

#: Where each accepted field lands on the clip record.
METRIC_MAP = {
    "vlm_entity_density": "vlm_entity_density",
    "vlm_quality": "vlm_quality",
    "reject_flags": "vlm_reject_flags",
    "scene_type": "vlm_scene_type",
    "scene_transition": "vlm_scene_transition",
}


def sample_timestamps(duration_sec: float) -> list[float]:
    """The frame timestamps to send: 1 fps over the WHOLE clip, ffmpeg round-up.

    Raises when the clip is longer than the frame budget rather than silently sampling
    its head — see the module docstring.
    """
    n = math.ceil(duration_sec * SAMPLE_FPS)
    if n > MAX_FRAMES:
        raise ValueError(
            f"clip is {duration_sec:.1f}s -> {n} frames at {SAMPLE_FPS} fps, over the "
            f"{MAX_FRAMES}-frame budget; split it before annotating")
    return [float(i) for i in range(max(n, 1))]


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text))


def validate_response(value: Any, duration_sec: float) -> tuple[list[str], dict | None]:
    """Return ``(errors, normalized)``. A non-empty error list means REJECT the response.

    Nothing is repaired and nothing is defaulted: a partially-valid annotation is
    indistinguishable from a real one once it is in the store, so the only two outcomes
    are a conforming record or a rejection with reasons.
    """
    errors: list[str] = []
    if not isinstance(value, dict):
        return ["response is not an object"], None
    if set(value) != TOP_KEYS:
        errors.append("top-level keys differ from the schema")

    density = value.get("vlm_entity_density")
    if isinstance(density, bool) or density not in {1, 2, 3}:
        errors.append("vlm_entity_density must be 1, 2, or 3")

    quality = value.get("vlm_quality")
    if (isinstance(quality, bool) or not isinstance(quality, (int, float))
            or float(quality) not in QUALITY_VALUES):
        errors.append("vlm_quality is invalid")

    flags = value.get("reject_flags")
    if (not isinstance(flags, list) or len(flags) != len(set(flags or []))
            or not set(flags or []).issubset(REJECT_FLAGS)):
        errors.append("reject_flags is invalid")

    if value.get("scene_type") not in SCENE_TYPES:
        errors.append("scene_type is invalid")

    errors += _transition_errors(value.get("scene_transition"), duration_sec)
    errors += _caption_errors(value.get("dense_caption"))

    if errors:
        return errors, None

    out = dict(value)
    out["vlm_quality"] = float(out["vlm_quality"])
    out["scene_transition"] = dict(out["scene_transition"])
    out["scene_transition"]["timestamps_sec"] = [
        float(x) for x in out["scene_transition"]["timestamps_sec"]]
    return [], out


def _transition_errors(tr: Any, duration_sec: float) -> list[str]:
    if not isinstance(tr, dict) or set(tr) != TRANSITION_KEYS:
        return ["scene_transition is invalid"]
    errs: list[str] = []
    label, count = tr.get("label"), tr.get("count")
    stamps, evidence = tr.get("timestamps_sec"), tr.get("evidence")
    if label not in TRANSITION_LABELS:
        errs.append("scene_transition.label is invalid")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        errs.append("scene_transition.count is invalid")
    if (not isinstance(stamps, list)
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in stamps)):
        errs.append("scene_transition.timestamps_sec is invalid")
    else:
        nums = [float(x) for x in stamps]
        if nums != sorted(nums) or any(
                not math.isfinite(x) or x < 0 or x > duration_sec + 1 for x in nums):
            errs.append("scene transition timestamps are invalid")
        if isinstance(count, int) and not isinstance(count, bool) and len(stamps) != count:
            errs.append("scene transition count/timestamp mismatch")
    if not isinstance(evidence, str):
        errs.append("scene_transition.evidence is invalid")
    # The label and its evidence have to agree, or "definite" becomes a free-floating
    # opinion with nothing behind it.
    if label == "none" and (count != 0 or stamps != [] or evidence != ""):
        errs.append("a none transition must have count=0, timestamps=[], evidence=''")
    if label in {"possible", "definite"} and (
            not isinstance(count, int) or count < 1 or not evidence):
        errs.append("a possible/definite transition needs a positive count and evidence")
    return errs


def _caption_errors(caption: Any) -> list[str]:
    if not isinstance(caption, str):
        return ["dense_caption must be a string"]
    errs: list[str] = []
    n = word_count(caption)
    lo, hi = CAPTION_WORDS
    if not lo <= n <= hi:
        errs.append(f"dense_caption word count is {n}, expected {lo}..{hi}")
    hits = [name for name, pat in FORBIDDEN_CAPTION_PATTERNS
            if re.search(pat, caption, re.IGNORECASE)]
    if hits:
        errs.append("dense_caption contains forbidden wording: " + ",".join(hits))
    return errs
