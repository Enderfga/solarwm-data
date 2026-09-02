"""Scene-static captioning + content gate (SolarWM Section 4.2, following LingBot-World).

Captioning for a *camera-controllable* world model has one non-obvious rule that drives
the whole design: when each clip already carries a metric camera trajectory (the action
condition), the caption must describe **what is in the scene** — objects, layout,
materials, lighting, scene-internal motion — while **omitting all camera ego-motion
language** ("pan left", "move forward", "orbit", ...). Encoding camera motion in *both*
the text and the pose branch makes the two fight at train time and leaks trajectory
supervision into the language path; the SolarWM recipe (and Uni3C / AC3D) keep camera
control exclusively in the pose branch. Camera motion, if noted at all, goes into a
separate ``_camera_note`` field used only for QC — never into the training caption.

The caption rubric applies these rules:
  * **One call does both the content gate and the caption.** A VLM that already watched
    the clip can score quality and describe content in a single pass.
  * **Describe content present throughout the clip**, not transient things the camera
    sweeps past for a moment — those confuse a world model that must hold the scene.
  * **Render faithfully.** A 3DGS / synthetic / game-engine clip is described as the
    world it depicts ("a stone courtyard"), never as "a video game / screenshot / HUD /
    third-person view", so the caption never invites UI artifacts.
  * **Dense, Wan-style ~60-150 words** — terse captions underspecify the scene.

Output schema (one JSON object per clip)::

    {
      "dense_caption": "...",             # scene-static, 60-150 words, NO camera motion
      "vlm_entity_density": 1 | 2 | 3,    # max SIMULTANEOUS presence, not a count
      "vlm_quality": 1.0-5.0,             # 5 = clean & richly describable, 1 = unusable
      "reject_flags": [ ... ],            # hard problems, see REJECT_FLAGS
      "scene_type": "real_world" | "rendered" | "game" | "animation" | "mixed",
      "scene_transition": {"label": "none" | "possible" | "definite",
                           "count": N, "timestamps_sec": [...], "evidence": "..."}
    }

Only ``dense_caption`` becomes a text condition. The other five are stored as annotations;
``reject_flags`` and ``scene_transition`` never modify ``kept`` or ``reject_reasons`` on
their own.

Backend note: some hosted VLM gateways
**cache responses by prompt text and ignore the video**. Because the rubric is identical
for every clip, such a gateway returns the *same* caption for different videos — silently
producing garbage. Before trusting any gateway at scale: (a) sample N produced captions
and check they are nearly all distinct; (b) send the same rubric with two different videos
and confirm the captions differ. If a gateway caches, append a per-clip nonce (the clip
id) to the prompt to bust the cache, or switch gateways. (Separately, reasoning-model
backends spend tokens on hidden thinking — give them a generous output budget or the
content field comes back empty.)

The Qwen VLM call is an adapter (``qwen_runner``); the dry-run path returns a
deterministic scene-static caption so the pipeline is testable without weights.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

# Canonical vocabularies for the content gate (used by downstream validation/filtering).
REJECT_FLAGS = (
    "text_heavy", "watermark", "ui_overlay", "blurry", "near_static",
    "transition_heavy", "low_light", "nsfw", "single_color",
)
SCENE_TYPES = ("real_world", "rendered", "game", "animation", "mixed")

# Phrases that describe camera motion and must NOT appear in scene-static captions.
CAMERA_MOTION_PATTERNS = [
    r"\bpan(s|ning|ned)?\b", r"\btilt(s|ing|ed)?\b", r"\bzoom(s|ing|ed)?\b",
    r"\bdolly(ing)?\b", r"\btrack(ing|s|ed)?\s+shot\b", r"\borbit(s|ing|ed)?\b",
    r"\bcrane\s+shot\b", r"\bpush(es|ing)?\s+in\b", r"\bpull(s|ing)?\s+out\b",
    r"\bmove(s|ing)?\s+(forward|backward|left|right|up|down)\b",
    r"\bcamera\s+(moves?|pans?|tilts?|rotat\w+|fl(y|ies|ying)|trucks?|pushes?)\b",
    r"\b(fly(s|ing)?\s+through|flythrough)\b", r"\brotat(e|es|ing)\s+around\b",
    r"\bfollow(s|ing)?\s+shot\b", r"\bhandheld\b", r"\bsteadicam\b",
]
_CAMERA_RE = re.compile("|".join(CAMERA_MOTION_PATTERNS), re.IGNORECASE)

# Caption-only prompt (dense, render-faithful, scene-static). Used by the two-stage path.
SCENE_STATIC_PROMPT = (
    "Describe what is in this scene: the objects present throughout the clip, their "
    "spatial layout, materials, colors, lighting, and any scene-internal motion "
    "(people walking, water flowing, foliage moving). Focus on what is stably present, "
    "not on things the camera only sweeps past for a moment. If the clip is a 3D render, "
    "game, or synthetic scene, describe the depicted world faithfully (e.g. 'a stone "
    "courtyard with carved archways') and do NOT call it a video game, screenshot, HUD, "
    "or third-person view. Do NOT mention any camera movement, viewpoint change, or shot "
    "type (no 'pan', 'zoom', 'move forward', 'orbit', etc.). Write one dense descriptive "
    "paragraph of about 60 to 150 words."
)

def contains_camera_motion(text: str) -> bool:
    """True if the caption mentions camera motion / shot type."""
    return bool(_CAMERA_RE.search(text or ""))


def strip_camera_motion(text: str) -> str:
    """Remove camera-motion clauses; collapse leftover whitespace/punctuation."""
    cleaned = _CAMERA_RE.sub("", text or "")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;])", r"\1", cleaned)
    cleaned = re.sub(r"([,;]\s*){2,}", ", ", cleaned)
    return cleaned.strip(" ,;.").strip()


def caption_clip(rec, models_cfg: dict) -> str:
    """Produce a scene-static caption for a clip (dry-run = deterministic).

    A source-provided scene-static caption (for example ABot's episode caption) is used
    when available because it can cover more of the clip than sampled VLM frames. The
    acquisition step writes it to ``prompt.txt``. Disable this behavior with
    ``SOLAR_WM_NATIVE_CAPTION=0``.
    """
    if os.environ.get("SOLAR_WM_NATIVE_CAPTION", "1") == "1" and rec.video_path:
        try:
            p = Path(rec.video_path).parent / "prompt.txt"
            if p.exists():
                txt = p.read_text(encoding="utf-8").strip()
                # Any non-empty native caption is accepted; length can be evaluated later
                # as a selection signal.
                if txt:
                    return strip_camera_motion(txt) if contains_camera_motion(txt) else txt
        except Exception:
            pass   # fall through to the normal captioning path

    if models_cfg.get("dry_run", True):
        h = hashlib.sha1(rec.clip_id.encode()).hexdigest()
        subjects = ["an indoor living room", "a forest clearing", "a city street",
                    "a stone courtyard", "a kitchen interior", "a mountain trail"]
        details = ["with wooden furniture and soft daylight",
                   "with dense green foliage and dappled light",
                   "with parked cars and brick facades",
                   "with weathered walls and scattered plants",
                   "with tiled counters and hanging utensils",
                   "with rocky ground and distant peaks"]
        cap = (f"{subjects[int(h[:2], 16) % len(subjects)]} "
               f"{details[int(h[2:4], 16) % len(details)]}.")
    else:
        try:
            from .qwen_runner import run_caption  # vendored Qwen VLM adapter
            cap = run_caption(rec.video_path, models_cfg, prompt=SCENE_STATIC_PROMPT)
        except ImportError as e:
            # No VLM on this fleet. A missing caption must never cost us the clip: the
            # video, poses and intrinsics are the expensive, irreplaceable products and
            # captions are routinely (re)generated downstream. Emit an empty caption and
            # let the assembler treat it as a missing field.
            print(f"[warn] captioning unavailable ({e}); emitting empty caption for "
                  f"{rec.clip_id}", flush=True)
            cap = ""

    # enforce the no-camera-motion invariant regardless of source
    if contains_camera_motion(cap):
        cap = strip_camera_motion(cap)
    return cap


def run_caption_stage(records, models_cfg: dict):
    for r in records:
        r.caption = caption_clip(r, models_cfg)
    return records
