#!/usr/bin/env python3
"""Run the final LTX-2.3 Clean Plate method on a frozen local task manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solar_wm_data.clean_plate import (  # noqa: E402
    LTX_SOURCE_REVISION, MODEL_FRAMES, slice_plan,  # noqa: F401
)
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


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def probe(path: Path) -> dict[str, Any]:
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


def prepare_reference(task: dict[str, Any], output: Path) -> None:
    start = int(task["source_start_frame"])
    target = int(task["target_frames"])
    model_frames = MODEL_FRAMES[target]
    run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-v",
            "error",
            "-i",
            task["video_path"],
            "-vf",
            f"trim=start_frame={start}:end_frame={start + target},setpts=PTS-STARTPTS,fps=24,tpad=stop_mode=clone:stop_duration=1",
            "-frames:v",
            str(model_frames),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
    )
    info = probe(output)
    if info["frames"] != model_frames or abs(info["fps"] - 24.0) > 1e-3:
        raise RuntimeError("model reference frame/fps mismatch")


def finalize_video(source: Path, target_frames: int, output: Path) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            "scale=1280:720:flags=lanczos,fps=16,tpad=stop_mode=clone:stop_duration=1",
            "-frames:v",
            str(target_frames),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
    )
    info = probe(output)
    if info != {"width": 1280, "height": 720, "fps": 16.0, "frames": target_frames}:
        raise RuntimeError(f"final video contract mismatch: {info}")


def slice_payload(task: dict[str, Any], output_dir: Path, timings: dict[str, float]) -> dict[str, Any]:
    start = int(task["source_start_frame"])
    target = int(task["target_frames"])
    end = start + target
    meta_path = Path(task["meta_path"])
    source_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    poses = np.load(task["poses_path"])
    intrinsics = np.load(task["intrinsics_path"])
    source_frames = int(task["source_num_frames"])
    if int(source_meta["num_frames"]) != source_frames or poses.shape[0] != source_frames or intrinsics.shape[0] != source_frames:
        raise RuntimeError("source payload frame mismatch")
    pose_slice, intrinsic_slice = poses[start:end], intrinsics[start:end]
    if pose_slice.shape[0] != target or intrinsic_slice.shape[0] != target:
        raise RuntimeError("camera slice length mismatch")
    if not np.isfinite(pose_slice).all() or not np.isfinite(intrinsic_slice).all():
        raise RuntimeError("camera slice contains non-finite values")
    np.save(output_dir / "poses.npy", pose_slice)
    np.save(output_dir / "intrinsics.npy", intrinsic_slice)
    (output_dir / "prompt.txt").write_bytes(Path(task["prompt_path"]).read_bytes())
    metadata = dict(source_meta)
    metadata.update(
        {
            "clip_id": task["output_clip_id"],
            "video_path": "video.mp4",
            "pose_path": "poses.npy",
            "intrinsics_path": "intrinsics.npy",
            "num_frames": target,
            "fps": 16.0,
            "width": 1280,
            "height": 720,
        }
    )
    if isinstance(metadata.get("audio"), dict):
        metadata["audio"] = dict(metadata["audio"])
        metadata["audio"]["duration_s"] = target / 16.0
    source_scale_factors_count = None
    if isinstance(metadata.get("scale_factors"), list):
        source_scale_factors_count = len(metadata["scale_factors"])
        if len(metadata["scale_factors"]) == source_frames:
            metadata["scale_factors"] = metadata["scale_factors"][start:end]
        else:
            metadata.pop("scale_factors")
    extra = dict(metadata.get("extra") or {})
    extra["clean_plate"] = {
        "schema_version": 2,
        "status": "complete",
        "method": "LTX-2.3 Clean Plate IC-LoRA",
        "source_clip_id": task["source_clip_id"],
        "source_num_frames": source_frames,
        "source_start_frame": start,
        "source_end_frame_exclusive": end,
        "target_frames": target,
        "model_frames_24fps": MODEL_FRAMES[target],
        "chunk_index": int(task["chunk_index"]),
        "source_scale_factors_count": source_scale_factors_count,
        "denoise_steps": 8,
        "seed": 42,
        "conditioning_strength": 1.0,
        "timings_seconds": {key: round(value, 3) for key, value in timings.items()},
        "metrics_note": "Source visual metrics are retained and were not recomputed after Clean Plate.",
    }
    metadata["extra"] = extra
    return metadata


def slice_audio(task: dict[str, Any], output_dir: Path) -> None:
    value = task.get("audio_path")
    if not value:
        return
    start = int(task["source_start_frame"]) / 16.0
    duration = int(task["target_frames"]) / 16.0
    run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            str(start),
            "-i",
            str(value),
            "-t",
            str(duration),
            "-vn",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(output_dir / "audio.m4a"),
        ]
    )


def validate_task(task: dict[str, Any]) -> None:
    required = {
        "dataset",
        "source_clip_id",
        "output_clip_id",
        "video_path",
        "poses_path",
        "intrinsics_path",
        "meta_path",
        "prompt_path",
        "source_num_frames",
        "source_start_frame",
        "target_frames",
        "chunk_index",
    }
    if not required.issubset(task):
        raise ValueError(f"task is missing fields: {sorted(required - set(task))}")
    if not safe_component(str(task["dataset"])) or not safe_component(str(task["output_clip_id"])):
        raise ValueError("dataset and output_clip_id must be safe path components")
    if int(task["target_frames"]) not in MODEL_FRAMES:
        raise ValueError("target_frames must be 81 or 160")
    if int(task["source_start_frame"]) < 0 or int(task["source_start_frame"]) + int(task["target_frames"]) > int(task["source_num_frames"]):
        raise ValueError("source window is outside the source clip")
    for key in ("video_path", "poses_path", "intrinsics_path", "meta_path", "prompt_path"):
        if not Path(task[key]).is_file():
            raise FileNotFoundError(task[key])


def process_task(
    task: dict[str, Any],
    output_root: Path,
    work_root: Path,
    pipeline: Any,
    prompt: str,
) -> dict[str, Any]:
    import clean_plate_ltx_runner as ltx_runner

    validate_task(task)
    dataset_dir = output_root / str(task["dataset"]) / "clips"
    destination = dataset_dir / str(task["output_clip_id"])
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    stage = dataset_dir / f".{task['output_clip_id']}.partial-{uuid.uuid4().hex}"
    stage.mkdir()
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="solar-clean-plate-", dir=work_root) as temporary:
            temporary_path = Path(temporary)
            reference = temporary_path / "reference.mp4"
            generated = temporary_path / "generated.mp4"
            prepare_reference(task, reference)
            infer_timings = ltx_runner.run_stage1(
                pipeline,
                str(reference),
                str(generated),
                prompt,
                MODEL_FRAMES[int(task["target_frames"])],
                1248,
                704,
                42,
                1.0,
                decode=True,
                denoise_steps=8,
            )
            finalize_video(generated, int(task["target_frames"]), stage / "video.mp4")
        metadata = slice_payload(task, stage, infer_timings)
        slice_audio(task, stage)
        (stage / "meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        )
        meta_path = stage / "meta.json"
        if not meta_path.is_file():
            raise RuntimeError("meta.json must be written last")
        os.rename(stage, destination)
        return {
            "sample_id": task["output_clip_id"],
            "status": "complete",
            "wall_sec": time.monotonic() - started,
            "video_sha256": sha256_file(destination / "video.mp4"),
            "meta_sha256": sha256_file(destination / "meta.json"),
        }
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    import torch

    import clean_plate_ltx_runner as ltx_runner

    if args.output_root.exists():
        raise SystemExit("--output-root must not exist; publication is create-only")
    args.work_root.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    ids = [str(task.get("output_clip_id") or "") for task in tasks]
    if not tasks or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise SystemExit("manifest must contain unique output_clip_id values")
    for task in tasks:
        validate_task(task)
    root = Path(__file__).resolve().parents[1] / "configs"
    prompt_bytes = (root / "clean_plate_prompt.txt").read_bytes()
    if hashlib.sha256(prompt_bytes).hexdigest() != PROMPT_FILE_SHA256:
        raise SystemExit("clean_plate_prompt.txt SHA-256 mismatch")
    if sha256_file(root / "clean_plate_models.json") != MODELS_MANIFEST_SHA256:
        raise SystemExit("clean_plate_models.json SHA-256 mismatch")
    prompt = prompt_bytes.decode("utf-8").strip()
    model_paths = ltx_runner.resolve_model_paths(args.model_root, root / "clean_plate_models.json")
    torch.cuda.set_device(0)
    pipeline = ltx_runner.build_pipeline(model_paths)
    args.output_root.mkdir(parents=True, exist_ok=False)
    receipts = []
    for task in tasks:
        receipt = process_task(task, args.output_root, args.work_root, pipeline, prompt)
        receipts.append(receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
    receipt_bytes = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in receipts).encode()
    (args.output_root / "receipts.jsonl").write_bytes(receipt_bytes)
    marker = {
        "schema_version": "solar_clean_plate_complete_v1",
        "samples": len(receipts),
        "manifest_sha256": sha256_file(args.manifest),
        "receipts_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "models_manifest_sha256": sha256_file(root / "clean_plate_models.json"),
        "runner_sha256": sha256_file(Path(__file__).with_name("clean_plate_ltx_runner.py")),
        "ltx_source_revision": LTX_SOURCE_REVISION,
        "parameters": {
            "denoise_steps": 8,
            "seed": 42,
            "conditioning_strength": 1.0,
            "model_resolution": [1248, 704],
            "output_resolution": [1280, 720],
            "output_fps": 16,
        },
    }
    (args.output_root / "COMPLETE.json").write_text(json.dumps(marker, sort_keys=True, indent=2) + "\n")
    print(json.dumps(marker, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
