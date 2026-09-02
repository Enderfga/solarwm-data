"""Qwen2.5-VL backend for scene-static captioning — the LOCAL fallback captioner.

Loaded lazily and cached. Reads a clip's video, samples frames, and asks the model for a
scene-static caption. Weights come from the repo ``weights/`` dir if present, else HF.

Released captions and semantic scores come from the annotation pass
(``solar_wm_data.caption.kimi`` / ``scripts/vlm_annotate.py``) and are merged as an
overlay. This local backend provides captioning only.

A small CLI (``python -m solar_wm_data.caption.qwen_runner --video V``) is provided so the
stage can run in a separate process if desired.
"""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path

_WEIGHTS = Path(os.environ.get(
    "SOLAR_WM_WEIGHTS", str(Path(__file__).resolve().parents[2] / "weights")))


def _model_src() -> str:
    local = _WEIGHTS / "qwen25vl7b"
    return str(local) if local.exists() else "Qwen/Qwen2.5-VL-7B-Instruct"


@lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    src = _model_src()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        src, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(src)
    return model, processor


def _clamp_nframes(video_path: str, nframes: int) -> int:
    """Qwen's fetch_video requires nframes in [2, total_frames] (and a multiple of 2).
    A degenerate ultra-short source clip (seen in the wild: a 7-frame video) crashed
    captioning forever; clamp to the video's real frame count instead."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or nframes
        cap.release()
        return max(2, min(nframes, (total // 2) * 2))
    except Exception:  # noqa: BLE001 - probing must never break captioning
        return nframes


def _ask(video_path: str, prompt: str, nframes: int = 8, max_new_tokens: int = 160) -> str:
    nframes = _clamp_nframes(video_path, nframes)
    from qwen_vl_utils import process_vision_info
    model, processor = _load()
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": f"file://{Path(video_path).resolve()}",
             "nframes": nframes, "max_pixels": 360 * 420},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True)
    # newer transformers validate `fps` as a scalar; process_vision_info returns
    # it as a per-video list — unwrap to a scalar before handing to the processor.
    fps = video_kwargs.pop("fps", None)
    if isinstance(fps, (list, tuple)):
        fps = fps[0] if fps else None
    proc_kwargs = dict(video_kwargs)
    if fps is not None:
        proc_kwargs["fps"] = fps
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt", **proc_kwargs)
    inputs = inputs.to(model.device)
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    out = processor.batch_decode(trimmed, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
    return out[0].strip()


def run_caption(video_path: str, models_cfg: dict, prompt: str) -> str:
    """Scene-static caption for a clip via Qwen2.5-VL."""
    return _ask(video_path, prompt, nframes=models_cfg.get("caption_nframes", 8))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()
    from . import SCENE_STATIC_PROMPT
    print(json.dumps({"caption": run_caption(args.video, {}, SCENE_STATIC_PROMPT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
