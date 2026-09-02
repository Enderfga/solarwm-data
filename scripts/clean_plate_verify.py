#!/usr/bin/env python3
"""Verify a completed SOLAR Clean Plate delivery against its frozen manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


LTX_SOURCE_REVISION = "9377758131b1ffde4b7f766804590a6617bf2ab9"
PROMPT_FILE_SHA256 = "4c6f7fc9a5ec19a980afb1a3fd151852de3266c27e7484e7ec7d5bb6863dca42"
MODELS_MANIFEST_SHA256 = "7f4bc948c5ab19b9df42b801c5bd56798aa30896dcc12d4f768c48c4387b2e68"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_component(value: str) -> bool:
    return bool(value) and value not in {".", ".."} and Path(value).name == value


def video_info(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": int(numerator) / int(denominator),
        "frames": int(stream["nb_read_frames"]),
    }


def audio_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return float(result.stdout.strip())


def verify_task(task: dict[str, Any], output_root: Path, receipt: dict[str, Any]) -> None:
    clip_id = str(task["output_clip_id"])
    if not safe_component(str(task["dataset"])) or not safe_component(clip_id):
        raise ValueError("dataset and output_clip_id must be safe path components")
    destination = output_root / str(task["dataset"]) / "clips" / clip_id
    target = int(task["target_frames"])
    start = int(task["source_start_frame"])
    expected_files = {"video.mp4", "poses.npy", "intrinsics.npy", "prompt.txt", "meta.json"}
    if task.get("audio_path"):
        expected_files.add("audio.m4a")
    actual_files = {path.name for path in destination.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ValueError(f"{clip_id}: payload files mismatch: {sorted(actual_files)}")
    if any((destination / name).stat().st_size == 0 for name in expected_files):
        raise ValueError(f"{clip_id}: empty payload file")

    expected_video = {"width": 1280, "height": 720, "fps": 16.0, "frames": target}
    if video_info(destination / "video.mp4") != expected_video:
        raise ValueError(f"{clip_id}: video contract mismatch")
    source_poses = np.load(task["poses_path"], mmap_mode="r")[start : start + target]
    source_intrinsics = np.load(task["intrinsics_path"], mmap_mode="r")[start : start + target]
    if not np.array_equal(np.load(destination / "poses.npy"), source_poses):
        raise ValueError(f"{clip_id}: poses are not the exact source slice")
    if not np.array_equal(np.load(destination / "intrinsics.npy"), source_intrinsics):
        raise ValueError(f"{clip_id}: intrinsics are not the exact source slice")
    if (destination / "prompt.txt").read_bytes() != Path(task["prompt_path"]).read_bytes():
        raise ValueError(f"{clip_id}: prompt bytes changed")
    if task.get("audio_path") and abs(audio_duration(destination / "audio.m4a") - target / 16.0) > 0.15:
        raise ValueError(f"{clip_id}: audio duration mismatch")

    meta_path = destination / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_meta = {
        "clip_id": clip_id,
        "video_path": "video.mp4",
        "pose_path": "poses.npy",
        "intrinsics_path": "intrinsics.npy",
        "num_frames": target,
        "fps": 16.0,
        "width": 1280,
        "height": 720,
    }
    if any(meta.get(key) != value for key, value in expected_meta.items()):
        raise ValueError(f"{clip_id}: metadata contract mismatch")
    clean_plate = (meta.get("extra") or {}).get("clean_plate") or {}
    expected_provenance = {
        "schema_version": 2,
        "status": "complete",
        "method": "LTX-2.3 Clean Plate IC-LoRA",
        "source_clip_id": task["source_clip_id"],
        "source_num_frames": int(task["source_num_frames"]),
        "source_start_frame": start,
        "source_end_frame_exclusive": start + target,
        "target_frames": target,
        "model_frames_24fps": {81: 121, 160: 241}[target],
        "chunk_index": int(task["chunk_index"]),
        "source_scale_factors_count": None,
        "denoise_steps": 8,
        "seed": 42,
        "conditioning_strength": 1.0,
        "metrics_note": "Source visual metrics are retained and were not recomputed after Clean Plate.",
    }
    source_meta = json.loads(Path(task["meta_path"]).read_text(encoding="utf-8"))
    if isinstance(source_meta.get("scale_factors"), list):
        expected_provenance["source_scale_factors_count"] = len(source_meta["scale_factors"])
    timings = clean_plate.pop("timings_seconds", None)
    if (
        not isinstance(timings, dict)
        or set(timings) != {"prompt_seconds", "reference_encode_seconds", "denoise_seconds", "decode_encode_seconds"}
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 for value in timings.values())
    ):
        raise ValueError(f"{clip_id}: inference timings are invalid")
    if clean_plate != expected_provenance:
        raise ValueError(f"{clip_id}: Clean Plate provenance mismatch")
    payload_mtimes = [(destination / name).stat().st_mtime_ns for name in expected_files - {"meta.json"}]
    if meta_path.stat().st_mtime_ns < max(payload_mtimes):
        raise ValueError(f"{clip_id}: meta.json was not committed last")
    if receipt.get("sample_id") != clip_id:
        raise ValueError(f"{clip_id}: receipt ID mismatch")
    if receipt.get("video_sha256") != sha256_file(destination / "video.mp4"):
        raise ValueError(f"{clip_id}: video receipt mismatch")
    if receipt.get("meta_sha256") != sha256_file(meta_path):
        raise ValueError(f"{clip_id}: metadata receipt mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    tasks = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    receipt_path = args.output_root / "receipts.jsonl"
    receipt_bytes = receipt_path.read_bytes()
    receipts = [json.loads(line) for line in receipt_bytes.splitlines() if line.strip()]
    marker = json.loads((args.output_root / "COMPLETE.json").read_text(encoding="utf-8"))
    if marker.get("schema_version") != "solar_clean_plate_complete_v1":
        raise ValueError("invalid completion schema")
    if marker.get("samples") != len(tasks) or len(receipts) != len(tasks):
        raise ValueError("task/receipt/completion counts differ")
    if marker.get("manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("completion manifest SHA-256 mismatch")
    if marker.get("receipts_sha256") != hashlib.sha256(receipt_bytes).hexdigest():
        raise ValueError("completion receipt SHA-256 mismatch")
    if marker.get("prompt_sha256") != PROMPT_FILE_SHA256:
        raise ValueError("completion prompt SHA-256 mismatch")
    if marker.get("models_manifest_sha256") != MODELS_MANIFEST_SHA256:
        raise ValueError("completion model manifest SHA-256 mismatch")
    if marker.get("ltx_source_revision") != LTX_SOURCE_REVISION:
        raise ValueError("completion LTX source revision mismatch")
    if [row["sample_id"] for row in receipts] != [str(task["output_clip_id"]) for task in tasks]:
        raise ValueError("receipt order does not match the manifest")
    for task, receipt in zip(tasks, receipts, strict=True):
        verify_task(task, args.output_root, receipt)
    print(json.dumps({"status": "verified", "samples": len(tasks)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
