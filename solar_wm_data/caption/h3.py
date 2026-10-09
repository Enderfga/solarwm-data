"""Final H3 caption request and output contract from the 2026-10-05 production run.

The validator and serializer live in ``_h3_contract``.
H3 is a caption-only pass: it never computes or replaces VLM scores.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from . import _h3_contract as contract
from ._h3_retry_feedback import feedback_messages

PROMPT_FILES = {
    "main": "kimi_prompt_h3.txt",
    "system": "kimi_prompt_h3_system.txt",
    "post_images": "kimi_prompt_h3_post_images.txt",
}


def load_prompt_bundle(configs: Path) -> dict[str, Any]:
    """Read the current instruction files; edits take effect on the next run."""
    texts = {}
    for role, name in PROMPT_FILES.items():
        text = (configs / name).read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"{name} must not be empty")
        texts[role] = text.strip() if role != "main" else text
    return texts


def parse_response(text: str) -> tuple[dict[str, Any], str]:
    import json

    normalized, audit = contract.normalize_json_envelope(text)
    value = json.loads(normalized)
    if not isinstance(value, dict):
        raise ValueError("H3 response must be a JSON object")
    return value, audit["envelope"]


def validate_response(raw: dict[str, Any], facts: dict[str, Any],
                      frame_count: int) -> list[str]:
    errors = contract.validate_facts(facts)
    if raw["choices"][0].get("finish_reason") != "stop":
        errors.append("completion did not stop normally")
    tokens = (raw.get("usage") or {}).get("prompt_tokens")
    if type(tokens) is not int or tokens < max(2000, 200 * frame_count):
        errors.append("multimodal token floor not met; image transport requires review")
    return errors


def make_overlay(source: dict[str, Any], facts: dict[str, Any],
                 provenance: dict[str, Any]) -> dict[str, Any]:
    # Earlier authorized backfills may already have archived caption under this name.
    legacy = copy.deepcopy(source)
    if "caption" not in legacy and isinstance(legacy.get("static_scene_description"), str):
        legacy["caption"] = legacy["static_scene_description"]
    return contract.make_test_copy(legacy, facts, provenance)
