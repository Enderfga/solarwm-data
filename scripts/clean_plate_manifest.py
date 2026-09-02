#!/usr/bin/env python3
"""Build the final local Clean Plate task manifest from a SOLAR corpus dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solar_wm_data.clean_plate import (  # noqa: E402
    CORE_FILES, MIN_SOURCE_SIZE, SOURCE_FPS, kept_source_reasons, slice_plan,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_component(value: str) -> bool:
    return bool(value) and value not in {".", ".."} and Path(value).name == value


def write_bytes_create(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


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
        "num_frames": int(stream["nb_read_frames"]),
    }


def load_kept_ids(path: Path) -> set[str]:
    """The source owner's kept clip ids, from a JSONL of records or a plain id per line."""
    ids: set[str] = set()
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kept_tier") is None and "kept_tier" in rec:
                continue                      # judged and rejected: not a kept source
            cid = rec.get("clip_id")
            if cid:
                ids.add(str(cid))
        else:
            ids.add(line)
    if not ids:
        raise SystemExit(f"--kept-sources {path} yielded no clip ids")
    return ids


def audit_clip(dataset: str, clip_dir: Path,
               kept_ids: "set[str] | None" = None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    clip_id = clip_dir.name
    # Cheapest possible rejection, and the one that saves the most: a clean clip whose
    # source the policy rejected inherits that rejection, so planning the work at all
    # would spend the GPU pass to produce something inadmissible on arrival.
    if kept_ids is not None:
        why = kept_source_reasons({"source_clip_id": clip_id}, kept_ids, dataset)
        if why:
            return None, {"clip_id": clip_id, "reason": why[0]}
    missing = [name for name in CORE_FILES if not (clip_dir / name).is_file() or (clip_dir / name).stat().st_size == 0]
    if missing:
        return None, {"clip_id": clip_id, "reason": "missing_or_empty_core_files", "files": missing}
    try:
        meta = json.loads((clip_dir / "meta.json").read_text(encoding="utf-8"))
        probe = video_info(clip_dir / "video.mp4")
        poses = np.load(clip_dir / "poses.npy", mmap_mode="r")
        intrinsics = np.load(clip_dir / "intrinsics.npy", mmap_mode="r")
        expected = int(meta["num_frames"])
        if str(meta.get("clip_id")) != clip_id:
            raise ValueError("clip_id mismatch")
        if expected != probe["num_frames"] or poses.shape[0] != expected or intrinsics.shape[0] != expected:
            raise ValueError("video/camera frame-count mismatch")
        if poses.shape[1:] != (4, 4) or intrinsics.shape[1:] not in {(3, 3), (4,)}:
            raise ValueError("unsupported camera shape")
        if not np.isfinite(poses).all() or not np.isfinite(intrinsics).all():
            raise ValueError("camera arrays contain non-finite values")
        if abs(float(meta["fps"]) - SOURCE_FPS) > 1e-6 or abs(probe["fps"] - SOURCE_FPS) > 1e-3:
            raise ValueError("source fps is not 16")
        width, height = int(meta["width"]), int(meta["height"])
        if (width, height) != (probe["width"], probe["height"]):
            raise ValueError("metadata/video resolution mismatch")
        if width < MIN_SOURCE_SIZE[0] or height < MIN_SOURCE_SIZE[1] or width <= height:
            return None, {"clip_id": clip_id, "reason": "source_is_not_720p_landscape", **probe}
        # ONE definition of the slicing rule, shared with the library and its tests, so a
        # manifest built here and a lineage check made later cannot disagree about which
        # source frames an output was supposed to come from.
        plan = slice_plan(expected)
        if plan is None:
            return None, {"clip_id": clip_id, "reason": "shorter_than_81_frames", **probe}
        start, target = plan
        task = {
            "dataset": dataset,
            "source_clip_id": clip_id,
            "output_clip_id": f"{clip_id}__{target}f",
            "video_path": str((clip_dir / "video.mp4").resolve()),
            "poses_path": str((clip_dir / "poses.npy").resolve()),
            "intrinsics_path": str((clip_dir / "intrinsics.npy").resolve()),
            "meta_path": str((clip_dir / "meta.json").resolve()),
            "prompt_path": str((clip_dir / "prompt.txt").resolve()),
            "source_num_frames": expected,
            "source_start_frame": start,
            "target_frames": target,
            "chunk_index": 0,
        }
        audio = clip_dir / "audio.m4a"
        if audio.is_file() and audio.stat().st_size > 0:
            task["audio_path"] = str(audio.resolve())
        return task, None
    except Exception as error:
        return None, {"clip_id": clip_id, "reason": "source_contract_mismatch", "error": str(error)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--corpus-root", type=Path, required=True, help="Dataset root containing clips/<clip_id>")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--kept-sources", type=Path,
                        help="the source owner's kept rows (meta JSONL or one clip id per "
                             "line). Without it every clip in --corpus-root is planned, "
                             "including ones the policy rejected")
    args = parser.parse_args()
    if not safe_component(args.dataset):
        raise SystemExit("--dataset must be one safe path component")
    clip_root = args.corpus_root / "clips"
    clip_dirs = sorted(path for path in clip_root.iterdir() if path.is_dir())
    if args.limit is not None:
        clip_dirs = clip_dirs[: args.limit]
    if not clip_dirs:
        raise SystemExit("no clip directories found")
    kept_ids = load_kept_ids(args.kept_sources) if args.kept_sources else None
    if kept_ids is None:
        print("WARNING: no --kept-sources given; planning Clean Plate work for every clip "
              "in the corpus root, including ones the policy rejected", file=sys.stderr)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tasks, rejected = [], []
    for clip_dir in clip_dirs:
        task, reject = audit_clip(args.dataset, clip_dir, kept_ids)
        if task is not None:
            tasks.append(task)
        else:
            rejected.append(reject)
    task_bytes = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in tasks).encode()
    reject_bytes = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rejected).encode()
    write_bytes_create(args.output_dir / "tasks.jsonl", task_bytes)
    write_bytes_create(args.output_dir / "rejected.jsonl", reject_bytes)
    marker = {
        "schema_version": "solar_clean_plate_plan_v1",
        "dataset": args.dataset,
        "source_clips": len(clip_dirs),
        "kept_source_filter": str(args.kept_sources) if args.kept_sources else None,
        "tasks": len(tasks),
        "tasks_81f": sum(row["target_frames"] == 81 for row in tasks),
        "tasks_160f": sum(row["target_frames"] == 160 for row in tasks),
        "rejected": len(rejected),
        "frame_rule": "n>=160:first 160; 81<=n<160:centered 81; n<81:reject",
        "tasks_sha256": sha256_bytes(task_bytes),
        "rejected_sha256": sha256_bytes(reject_bytes),
    }
    write_bytes_create(
        args.output_dir / "PLAN_COMPLETE.json",
        (json.dumps(marker, sort_keys=True, indent=2) + "\n").encode(),
    )
    print(json.dumps(marker, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
