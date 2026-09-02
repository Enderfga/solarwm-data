#!/usr/bin/env python3
"""Caption videos with a local OpenAI-compatible Kimi-K2.6 endpoint."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import ipaddress
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROMPT_SHA256 = "acd355fa3213334c45e2c2624f4e40d3fb61d4f4d48f69b3e47ba6fb88dc8dbb"
# The schema, the frozen runtime identities and the validator live in the package so
# there is ONE definition of the contract. Editing them here instead would let the
# runner and the library disagree about what a valid annotation is.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solar_wm_data.caption.kimi import (  # noqa: E402
    RELEASE_MODEL_REVISION, RELEASE_RUNTIME, TOP_KEYS, TRANSITION_KEYS, REJECT_FLAGS, SCENE_TYPES,
    TRANSITION_LABELS, QUALITY_VALUES, FORBIDDEN_CAPTION_PATTERNS,
    word_count, validate_response,
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_create(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be a JSON object")
        sample_id = str(row.get("sample_id") or "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", sample_id):
            raise ValueError(f"{path}:{line_number}: invalid sample_id")
        video_path = Path(str(row.get("video_path") or ""))
        if not video_path.is_absolute():
            video_path = (path.parent / video_path).resolve()
        row = dict(row)
        row["sample_id"] = sample_id
        row["video_path"] = str(video_path)
        row["manifest_line"] = line_number
        rows.append(row)
    ids = [row["sample_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("manifest must contain at least one row and unique sample_id values")
    return rows


def checked_endpoint(value: str) -> str:
    """Validate the inference endpoint. Any HTTP(S) host is allowed.

    Credentials, queries and fragments are rejected because this URL is logged with every
    request: a key embedded in it would end up in the run's own output. Pass the key
    through the environment instead.
    """
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("endpoint must be an HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("endpoint must not contain credentials, query, or fragment — "
                         "they would be written into the run's logs")
    return value.rstrip("/")


def probe_video(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    value = json.loads(result.stdout)
    if len(value.get("streams") or []) != 1:
        raise RuntimeError(f"expected one video stream: {path}")
    duration = float((value.get("format") or {}).get("duration") or 0.0)
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"invalid duration: {path}")
    return value


def timestamps(duration_sec: float) -> list[float]:
    return [float(index) for index in range(max(1, int(math.ceil(duration_sec - 1e-9))))]


def extract_frames(video_path: Path, work_dir: Path, duration_sec: float) -> list[dict[str, Any]]:
    stamps = timestamps(duration_sec)
    if len(stamps) > 64:
        raise ValueError(f"1-fps sample has {len(stamps)} frames; split the clip instead of truncating it")
    frame_dir = work_dir / "frames"
    frame_dir.mkdir()
    scale = "scale=w='min(768,iw)':h='min(768,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"setpts=PTS-STARTPTS,fps=fps=1:start_time=0:round=up,{scale}",
            "-frames:v",
            str(len(stamps)),
            "-q:v",
            "3",
            "-fps_mode",
            "passthrough",
            str(frame_dir / "frame_%06d.jpg"),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    paths = sorted(frame_dir.glob("frame_*.jpg"))
    if len(paths) != len(stamps):
        raise RuntimeError(f"frame extraction mismatch: expected {len(stamps)}, got {len(paths)}")
    return [
        {
            "index": index,
            "timestamp_sec": stamp,
            "path": path,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for index, (path, stamp) in enumerate(zip(paths, stamps, strict=True), 1)
    ]


def response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(TOP_KEYS),
        "properties": {
            "vlm_entity_density": {"type": "integer", "enum": [1, 2, 3]},
            "vlm_quality": {"type": "number", "enum": sorted(QUALITY_VALUES)},
            "reject_flags": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "enum": sorted(REJECT_FLAGS)},
            },
            "scene_type": {"type": "string", "enum": sorted(SCENE_TYPES)},
            "scene_transition": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(TRANSITION_KEYS),
                "properties": {
                    "label": {"type": "string", "enum": sorted(TRANSITION_LABELS)},
                    "count": {"type": "integer", "minimum": 0},
                    "timestamps_sec": {"type": "array", "items": {"type": "number", "minimum": 0}},
                    "evidence": {"type": "string"},
                },
            },
            "dense_caption": {"type": "string"},
        },
    }


def multimodal_content(prompt: str, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame in frames:
        label = f"Frame {frame['index']:04d}/{len(frames):04d}; timestamp_sec={frame['timestamp_sec']:.6f}"
        encoded = base64.b64encode(frame["path"].read_bytes()).decode("ascii")
        content.extend(
            [
                {"type": "text", "text": label},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
            ]
        )
    return content


def call_model(endpoint: str, model: str, content: list[dict[str, Any]], timeout: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "seed": 0,
        "max_completion_tokens": 4096,
        "stream": False,
        "chat_template_kwargs": {"thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "solar_kimi_caption_v1", "strict": True, "schema": response_schema()},
        },
    }
    request = urllib.request.Request(
        f"{endpoint}/chat/completions",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = json.load(response)
    if not isinstance(raw, dict):
        raise ValueError("endpoint response must be a JSON object")
    return raw


def response_text(raw: dict[str, Any]) -> str:
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        )
    raise ValueError("unsupported response content")


def parse_json_response(text: str) -> tuple[dict[str, Any], str]:
    try:
        value = json.loads(text)
        mode = "native_json"
    except json.JSONDecodeError as direct_error:
        match = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(\{.*\})\r?\n```[ \t]*", text, re.DOTALL | re.IGNORECASE)
        if not match:
            raise direct_error
        value = json.loads(match.group(1))
        mode = "exact_json_fence"
    if not isinstance(value, dict):
        raise ValueError("caption response is not a JSON object")
    return value, mode


def process_sample(
    row: dict[str, Any],
    output_dir: Path,
    prompt: str,
    endpoint: str,
    model: str,
    model_revision: str,
    runtime: str,
    workers: int,
    timeout: int,
) -> dict[str, Any]:
    sample_id = row["sample_id"]
    record_path = output_dir / "records" / f"{sample_id}.json"
    attempt_dir = output_dir / "attempts" / sample_id
    started = time.monotonic()
    attempts: list[dict[str, Any]] = []
    try:
        video_path = Path(row["video_path"])
        probe = probe_video(video_path)
        duration = float(probe["format"]["duration"])
        with tempfile.TemporaryDirectory(prefix=f"solar-kimi-{sample_id}-") as temporary:
            frames = extract_frames(video_path, Path(temporary), duration)
            content = multimodal_content(prompt, frames)
            final_response = None
            status = "terminal_invalid"
            for attempt in (1, 2):
                attempt_started = time.monotonic()
                envelope: dict[str, Any] = {"sample_id": sample_id, "attempt": attempt, "created_at": utc_now()}
                try:
                    raw = call_model(endpoint, model, content, timeout)
                    text = response_text(raw)
                    parsed, normalization = parse_json_response(text)
                    validation_errors, normalized = validate_response(parsed, duration)
                    envelope.update(
                        {
                            "api_response": raw,
                            "transport_normalization": normalization,
                            "validation_errors": validation_errors,
                            "elapsed_sec": time.monotonic() - attempt_started,
                        }
                    )
                    write_json_create(attempt_dir / f"attempt_{attempt:02d}.json", envelope)
                    attempts.append(
                        {
                            "attempt": attempt,
                            "validation_errors": validation_errors,
                            "elapsed_sec": envelope["elapsed_sec"],
                        }
                    )
                    if validation_errors:
                        continue
                    final_response = normalized
                    status = "success"
                    break
                except Exception as error:
                    envelope.update(
                        {
                            "error": {"type": type(error).__name__, "message": str(error)},
                            "elapsed_sec": time.monotonic() - attempt_started,
                        }
                    )
                    write_json_create(attempt_dir / f"attempt_{attempt:02d}.json", envelope)
                    attempts.append({"attempt": attempt, "error": envelope["error"], "elapsed_sec": envelope["elapsed_sec"]})
            record = {
                "schema_version": "solar_kimi_caption_record_v1",
                "sample_id": sample_id,
                "status": status,
                "manifest_line": row["manifest_line"],
                "input": {
                    "video_path": str(video_path),
                    "video_bytes": video_path.stat().st_size,
                    "video_sha256": sha256_file(video_path),
                },
                "video_probe": probe,
                "frame_policy": {
                    "sample_fps": 1,
                    "max_frames": 64,
                    "max_edge": 768,
                    "jpeg_qscale": 3,
                    "audio": "not_read",
                    "frames": [
                        {key: frame[key] for key in ("index", "timestamp_sec", "bytes", "sha256")}
                        for frame in frames
                    ],
                },
                "model": {"name": model, "revision": model_revision, "runtime": runtime},
                "generation": {
                    "workers": workers,
                    "temperature": 0,
                    "seed": 0,
                    "max_attempts": 2,
                    "thinking": False,
                },
                "prompt_sha256": PROMPT_SHA256,
                "attempts": attempts,
                "response": final_response,
                "wall_sec": time.monotonic() - started,
            }
    except Exception as error:
        record = {
            "schema_version": "solar_kimi_caption_record_v1",
            "sample_id": sample_id,
            "status": "failed",
            "manifest_line": row["manifest_line"],
            "prompt_sha256": PROMPT_SHA256,
            "error": {"type": type(error).__name__, "message": str(error)},
            "attempts": attempts,
            "wall_sec": time.monotonic() - started,
        }
    write_json_create(record_path, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    # Model, revision and runtime are RECORDED with every response, not restricted to one
    # value: the point is that a record says which build produced it. The defaults are what
    # annotated the released corpus, so a reproduction run needs no flags.
    parser.add_argument("--model", default="moonshotai/Kimi-K2.6")
    parser.add_argument("--model-revision", default=RELEASE_MODEL_REVISION)
    parser.add_argument("--runtime", default=RELEASE_RUNTIME,
                        help="free-form identifier for the serving build, stored with each record")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id", action="append")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers must be in 1..4")
    if args.timeout < 1 or args.limit is not None and args.limit < 1:
        raise SystemExit("timeout and limit must be positive")
    endpoint = checked_endpoint(args.endpoint)
    # The frozen prompt is part of the schema: its sha256 travels with every record,
    # so a changed prompt is a new generation, not an edit. Verify before any work.
    prompt_path = Path(__file__).resolve().parents[1] / "configs" / "kimi_prompt.txt"
    prompt_bytes = prompt_path.read_bytes()
    if sha256_bytes(prompt_bytes) != PROMPT_SHA256:
        raise SystemExit(f"{prompt_path} SHA256 mismatch — this is a NEW prompt version")
    rows = read_manifest(args.manifest)
    if args.sample_id:
        requested = set(args.sample_id)
        rows = [row for row in rows if row["sample_id"] in requested]
        if {row["sample_id"] for row in rows} != requested:
            raise SystemExit("one or more requested sample IDs are absent")
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no samples selected")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json_create(
        args.output_dir / "run_manifest.json",
        {
            "schema_version": "solar_kimi_caption_run_v1",
            "created_at": utc_now(),
            "input_manifest": str(args.manifest.resolve()),
            "input_manifest_sha256": sha256_file(args.manifest),
            "sample_ids_sha256": sha256_bytes("\n".join(row["sample_id"] for row in rows).encode()),
            "samples": len(rows),
            "prompt_sha256": PROMPT_SHA256,
            "endpoint": endpoint,
            "model": args.model,
            "model_revision": args.model_revision,
            "runtime": args.runtime,
            "workers": args.workers,
            "frame_policy": {"sample_fps": 1, "max_frames": 64, "max_edge": 768, "jpeg_qscale": 3},
            "generation": {"thinking": False, "temperature": 0, "seed": 0, "attempts": 2},
        },
    )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_sample,
                row,
                args.output_dir,
                prompt_bytes.decode("utf-8"),
                endpoint,
                args.model,
                args.model_revision,
                args.runtime,
                args.workers,
                args.timeout,
            ): row["sample_id"]
            for row in rows
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"sample_id": result["sample_id"], "status": result["status"]}), flush=True)
    statuses = {status: sum(row["status"] == status for row in results) for status in ("success", "terminal_invalid", "failed")}
    summary = {
        "schema_version": "solar_kimi_caption_summary_v1",
        "samples": len(results),
        "status_counts": statuses,
        "complete_with_rejects": statuses["failed"] == 0,
    }
    write_json_create(args.output_dir / "SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if statuses["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
