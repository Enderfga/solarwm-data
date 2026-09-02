#!/usr/bin/env python3
"""Exact single-GPU LTX-2.3 Clean Plate Stage-1 inference path."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.conditioning import ConditioningItemAttentionStrengthWrapper, VideoConditionByReferenceLatent
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization.fp8_cast import build_policy as build_fp8_cast_policy
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.iclora_utils import temporal_subsample
from ltx_pipelines.utils.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import combined_image_conditionings
from ltx_pipelines.utils.media_io import decode_video_by_frame, encode_video, normalize_images, resize_and_center_crop
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode


FPS = 24


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_file(path: Path, spec: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(spec["size"]):
        raise ValueError(f"model size mismatch: {path}")
    if sha256_file(path) != spec["sha256"]:
        raise ValueError(f"model SHA-256 mismatch: {path}")


class CleanPlatePipeline(ICLoraPipeline):
    """IC-LoRA pipeline with tiled, CPU-preprocessed reference VAE input."""

    reference_tiling_config = TilingConfig.default()

    def _create_conditionings(
        self,
        images,
        video_conditioning,
        height,
        width,
        num_frames,
        video_encoder,
        conditioning_attention_strength=1.0,
        conditioning_attention_mask=None,
    ):
        if conditioning_attention_mask is not None:
            raise NotImplementedError("the final Clean Plate method is mask-free")
        conditionings = combined_image_conditionings(
            images=images,
            height=height,
            width=width,
            video_encoder=video_encoder,
            dtype=self.dtype,
            device=self.device,
        )
        scale = self.reference_downscale_factor
        ref_height, ref_width = height // scale, width // scale
        for video_path, strength in video_conditioning:
            prepared = []
            for frame in decode_video_by_frame(path=video_path, frame_cap=num_frames, device=torch.device("cpu")):
                frame = resize_and_center_crop(frame.to(torch.float32), ref_height, ref_width)
                prepared.append(normalize_images(frame, torch.device("cpu"), self.dtype))
            if len(prepared) != num_frames:
                raise ValueError(f"decoded {len(prepared)} reference frames, expected {num_frames}")
            video = torch.cat(prepared, dim=2)
            del prepared
            if self.reference_temporal_scale_factor > 1:
                video = temporal_subsample(video, self.reference_temporal_scale_factor)
            encoded = video_encoder.tiled_encode(video, self.reference_tiling_config)
            del video
            condition = VideoConditionByReferenceLatent(
                latent=encoded,
                downscale_factor=scale,
                temporal_scale_factor=self.reference_temporal_scale_factor,
                strength=strength,
            )
            if conditioning_attention_strength < 1.0:
                condition = ConditioningItemAttentionStrengthWrapper(
                    condition,
                    attention_mask=conditioning_attention_strength,
                )
            conditionings.append(condition)
        return conditionings


def resolve_model_paths(model_root: Path, config_path: Path) -> dict[str, Path]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = {
        "checkpoint": model_root / config["ltx_checkpoint"]["path"],
        "upsampler": model_root / config["spatial_upsampler"]["path"],
        "lora": model_root / config["clean_plate_lora"]["path"],
        "gemma": model_root / config["gemma"]["path"],
    }
    verify_model_file(paths["checkpoint"], config["ltx_checkpoint"])
    verify_model_file(paths["upsampler"], config["spatial_upsampler"])
    verify_model_file(paths["lora"], config["clean_plate_lora"])
    if not (paths["gemma"] / "config.json").is_file():
        raise FileNotFoundError(paths["gemma"] / "config.json")
    for spec in config["gemma"]["files"]:
        verify_model_file(paths["gemma"] / spec["path"], spec)
    return paths


def build_pipeline(paths: dict[str, Path]) -> CleanPlatePipeline:
    return CleanPlatePipeline(
        distilled_checkpoint_path=str(paths["checkpoint"]),
        spatial_upsampler_path=str(paths["upsampler"]),
        gemma_root=str(paths["gemma"]),
        loras=(
            LoraPathStrengthAndSDOps(
                str(paths["lora"]),
                1.0,
                LTXV_LORA_COMFY_RENAMING_MAP,
            ),
        ),
        device=torch.device("cuda", torch.cuda.current_device()),
        quantization=build_fp8_cast_policy(str(paths["checkpoint"])),
        offload_mode=OffloadMode.NONE,
        alloc_trim_strategy=AllocatorTrimStrategy.TRIM,
    )


@torch.inference_mode()
def run_stage1(
    pipeline: CleanPlatePipeline,
    reference: str,
    output: str,
    prompt: str,
    frames: int,
    width: int,
    height: int,
    seed: int,
    conditioning_strength: float,
    decode: bool,
    denoise_steps: int,
) -> dict[str, float]:
    timings: dict[str, float] = {}
    started = time.monotonic()
    (prompt_context,) = pipeline.prompt_encoder(
        [prompt],
        enhance_first_prompt=False,
        enhance_prompt_image=None,
        enhance_prompt_seed=seed,
    )
    timings["prompt_seconds"] = time.monotonic() - started
    video_context = prompt_context.video_encoding
    audio_context = prompt_context.audio_encoding

    started = time.monotonic()
    conditionings = pipeline.image_conditioner(
        lambda encoder: pipeline._create_conditionings(
            images=[],
            video_conditioning=[(reference, 1.0)],
            height=height,
            width=width,
            video_encoder=encoder,
            num_frames=frames,
            conditioning_attention_strength=conditioning_strength,
            conditioning_attention_mask=None,
        )
    )
    timings["reference_encode_seconds"] = time.monotonic() - started

    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator)
    started = time.monotonic()
    with pipeline.stage_1.model_context() as transformer:
        video_state, _ = pipeline.stage_1.run(
            transformer=transformer,
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=DISTILLED_SIGMAS[: denoise_steps + 1].to(dtype=torch.float32, device=pipeline.device),
            noiser=noiser,
            width=width,
            height=height,
            frames=frames,
            fps=FPS,
            video=ModalitySpec(context=video_context, conditionings=conditionings),
            audio=ModalitySpec(context=audio_context),
        )
    timings["denoise_seconds"] = time.monotonic() - started
    if video_state is None:
        raise RuntimeError("Stage 1 returned no video state")
    if decode:
        started = time.monotonic()
        video = pipeline.video_decoder(video_state.latent, TilingConfig.default(), generator)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        encode_video(
            video=video,
            fps=FPS,
            audio=None,
            output_path=output,
            video_chunks_number=get_video_chunks_number(frames, TilingConfig.default()),
        )
        timings["decode_encode_seconds"] = time.monotonic() - started
    return timings
