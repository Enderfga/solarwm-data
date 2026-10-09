#!/usr/bin/env python3
"""Materialize accepted Kimi captions into a create-only metadata overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solar_wm_data.caption import h3  # noqa: E402


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_h3(row: dict[str, Any], manifest: Path,
               caption_run: Path) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    sample_id = str(row["sample_id"])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", sample_id):
        raise ValueError("invalid sample_id")
    source_path = resolve(str(row.get("meta_path") or ""), manifest)
    source_bytes = source_path.read_bytes()
    record_bytes = (caption_run / "records" / f"{sample_id}.json").read_bytes()
    record = json.loads(record_bytes)
    if (record.get("sample_id") != sample_id or record.get("caption_format") != "h3"
            or record.get("status") != "success"):
        raise ValueError(f"{sample_id}: accepted H3 record identity/status mismatch")
    if sha256_bytes(source_bytes) != record["input"].get("source_meta_sha256"):
        raise ValueError(f"{sample_id}: source metadata changed after annotation")
    video_path = resolve(str(row.get("video_path") or ""), manifest)
    if (video_path.resolve() != Path(record["input"]["video_path"]).resolve()
            or sha256_file(video_path) != record["input"]["video_sha256"]):
        raise ValueError(f"{sample_id}: source video differs from annotated video")
    selected_attempt = next(item["attempt"] for item in record["attempts"]
                            if not item.get("validation_errors") and not item.get("error"))
    source = json.loads(source_bytes)
    meta = h3.make_overlay(source, record["response"], {
        "sample_id": sample_id,
        "model": record["model"],
        "prompt_bundle": record.get("prompt_bundle"),
        "source_video_path": str(video_path.resolve()),
        "source_meta_path": str(source_path.resolve()),
        "caption_record_sha256": sha256_bytes(record_bytes),
        "source_meta_sha256": sha256_bytes(source_bytes),
        "source_video_sha256": record["input"]["video_sha256"],
        "frame_policy": record["frame_policy"],
        "selected_attempt": selected_attempt,
    })
    # Preserve legacy NaN metadata values, just as the production codec did.
    encoded = (json.dumps(meta, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    relative = Path(str(row.get("output_relpath") or sample_id))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{sample_id}: output_relpath must stay below output-root")
    receipt = {"sample_id": sample_id, "source_meta_sha256": sha256_bytes(source_bytes),
               "caption_record_sha256": sha256_bytes(record_bytes),
               "meta_sha256": sha256_bytes(encoded), "output_relpath": str(relative)}
    return meta, encoded, receipt


def materialize_h3(rows: list[dict[str, Any]], manifest: Path,
                   caption_run: Path, output_root: Path) -> int:
    relative_paths = [str(Path(str(row.get("output_relpath") or row["sample_id"]))) for row in rows]
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("duplicate output_relpath")
    receipts = []
    staging = output_root / "meta.jsonl.tmp"
    with staging.open("xb") as handle:
        for row in rows:
            _, encoded, receipt = prepare_h3(row, manifest, caption_run)
            handle.write(encoded)
            receipts.append(receipt)
        handle.flush()
        os.fsync(handle.fileno())
    root_sha = sha256_file(staging)
    staging.rename(output_root / "meta.jsonl")
    write_json_create(output_root / "META_JSONL_READY.json", {
        "schema_version": "solar_kimi_h3_meta_jsonl_ready_v1",
        "samples": len(rows), "meta_jsonl_sha256": root_sha,
        "required_prompt_field": "h3_prompt",
        "per_clip_files_required_before_encoding": False,
        "source_metadata_mutated": False, "VLM_metrics_recomputed": False,
    })
    # Only hashes/IDs are retained in memory; do not cache the full corpus's metadata.
    for row, frozen in zip(rows, receipts, strict=True):
        meta, encoded, receipt = prepare_h3(row, manifest, caption_run)
        if receipt != frozen:
            raise ValueError(f"{row['sample_id']}: source/response changed after JSONL publication")
        prompt = (meta["h3_prompt"] + "\n").encode()
        destination = output_root / receipt["output_relpath"]
        write_bytes_create(destination / "meta.json", encoded)
        write_bytes_create(destination / "prompt.txt", prompt)
        if ((destination / "meta.json").read_bytes() != encoded
                or (destination / "prompt.txt").read_bytes() != prompt):
            raise ValueError("H3 clip full-readback mismatch")
        receipt.update(prompt_sha256=sha256_bytes(prompt), source_metadata_differences=0,
                       actual_meta_prompt_readback_verified=True)
    raw = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in receipts).encode()
    write_bytes_create(output_root / "delivery_manifest.jsonl", raw)
    write_json_create(output_root / "COMPLETE.json", {
        "schema_version": "solar_kimi_h3_caption_delivery_v1", "samples": len(receipts),
        "delivery_manifest_sha256": sha256_bytes(raw), "meta_jsonl_sha256": root_sha,
        "prompt_is_h3_prompt_plus_newline": True, "source_metadata_mutated": False,
        "VLM_metrics_recomputed": False, "source_metadata_differences": 0,
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--caption-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--caption-format", choices=("static", "h3"), default="static")
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    args.output_root.mkdir(parents=True, exist_ok=False)
    if args.caption_format == "h3":
        return materialize_h3(rows, args.manifest, args.caption_run, args.output_root)
    receipts = []
    for row in rows:
        sample_id = str(row["sample_id"])
        meta_path = resolve(str(row.get("meta_path") or ""), args.manifest)
        if not meta_path.is_file():
            raise FileNotFoundError(f"{sample_id}: meta_path is required and must exist")
        record_path = args.caption_run / "records" / f"{sample_id}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("caption_format") == "h3":
            raise RuntimeError(f"{sample_id}: select --caption-format h3 for this record")
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
            "prompt_text": record.get("prompt_text"),
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
