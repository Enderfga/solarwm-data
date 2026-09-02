#!/usr/bin/env python3
"""Materialize accepted Kimi captions into a create-only metadata overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


METRIC_MAP = {
    "vlm_entity_density": "vlm_entity_density",
    "vlm_quality": "vlm_quality",
    "reject_flags": "vlm_reject_flags",
    "scene_type": "vlm_scene_type",
    "scene_transition": "vlm_scene_transition",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_bytes_create(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_create(path: Path, value: Any) -> None:
    write_bytes_create(path, (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row.get("sample_id") or "") for row in rows]
    if not rows or any(not sample_id for sample_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("manifest sample IDs must be nonempty and unique")
    return rows


def resolve(path: str, manifest: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (manifest.parent / value).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--caption-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    args.output_root.mkdir(parents=True, exist_ok=False)
    receipts = []
    for row in rows:
        sample_id = str(row["sample_id"])
        meta_path = resolve(str(row.get("meta_path") or ""), args.manifest)
        if not meta_path.is_file():
            raise FileNotFoundError(f"{sample_id}: meta_path is required and must exist")
        record_path = args.caption_run / "records" / f"{sample_id}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "success" or not isinstance(record.get("response"), dict):
            raise RuntimeError(f"{sample_id}: caption is not accepted")
        response = record["response"]
        caption = response["dense_caption"]
        source_bytes = meta_path.read_bytes()
        meta = json.loads(source_bytes)
        meta["caption"] = caption
        metrics = dict(meta.get("metrics") or {})
        for source_key, target_key in METRIC_MAP.items():
            metrics[target_key] = response[source_key]
        meta["metrics"] = metrics
        extra = dict(meta.get("extra") or {})
        extra["kimi_caption_delivery"] = {
            "schema_version": 4,
            "sample_id": sample_id,
            "caption_sha256": sha256_bytes(caption.encode()),
            "response_sha256": sha256_bytes(
                json.dumps(response, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            ),
            "prompt_sha256": record["prompt_sha256"],
            "model": record["model"],
            "source_video_sha256": record["input"]["video_sha256"],
            "source_meta_sha256": sha256_bytes(source_bytes),
            "selected_attempt": next(
                item["attempt"] for item in record["attempts"] if not item.get("validation_errors") and not item.get("error")
            ),
            "transition_status": "unverified",
            "reject_flags_are_audit_only": True,
        }
        meta["extra"] = extra
        relative = Path(str(row.get("output_relpath") or sample_id))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{sample_id}: output_relpath must stay below output-root")
        destination = args.output_root / relative
        meta_bytes = (json.dumps(meta, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        prompt_bytes = (caption + "\n").encode()
        write_bytes_create(destination / "meta.json", meta_bytes)
        write_bytes_create(destination / "prompt.txt", prompt_bytes)
        receipts.append(
            {
                "sample_id": sample_id,
                "source_meta_sha256": sha256_bytes(source_bytes),
                "meta_sha256": sha256_bytes(meta_bytes),
                "prompt_sha256": sha256_bytes(prompt_bytes),
                "output_relpath": str(relative),
            }
        )
    manifest_bytes = "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in receipts).encode()
    write_bytes_create(args.output_root / "delivery_manifest.jsonl", manifest_bytes)
    write_json_create(
        args.output_root / "COMPLETE.json",
        {
            "schema_version": "solar_kimi_caption_delivery_v1",
            "samples": len(receipts),
            "delivery_manifest_sha256": sha256_bytes(manifest_bytes),
            "prompt_is_caption_plus_newline": True,
            "source_metadata_mutated": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
