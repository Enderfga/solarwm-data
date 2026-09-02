#!/usr/bin/env python3
"""Score many videos with DOVER, loading the model ONCE.

``evaluate_one_video.py`` reloads the DOVER ConvNeXt backbone on every call, so the
paper-faithful 5 s-chunk averaging (≈12 chunks for a 60 s clip) paid a ~12× model-load
tax — which is why chunk averaging shipped disabled. This driver loads the model and the
temporal samplers once, then scores every input clip, so chunk averaging is cheap enough
to run by default.

Run with cwd = the DOVER repo (so ``dover.yml`` and the ``dover`` package resolve), e.g.
    cd third_party/DOVER && python3 <repo>/scripts/dover_score_many.py dover.yml a.mp4 b.mp4
Prints one line per input: ``DOVER_SCORE <path> <fused[0,1]>`` (or ``DOVER_FAIL <path> ...``).
Logic mirrors evaluate_one_video.py exactly (mean/std, view normalisation, fuse_results).
"""
import os
import sys

import numpy as np
import torch
import yaml

# Run with cwd = the DOVER repo. Python puts THIS script's dir (scripts/) on sys.path,
# not the cwd, so the `dover` package next to dover.yml isn't importable without this.
sys.path.insert(0, os.getcwd())

from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
from dover.models import DOVER

mean, std = (
    torch.FloatTensor([123.675, 116.28, 103.53]),
    torch.FloatTensor([58.395, 57.12, 57.375]),
)


def fuse_results(results):
    # identical to evaluate_one_video.py: score-level fusion -> sigmoid -> [0,1]
    x = (results[0] - 0.1107) / 0.07355 * 0.6104 + (results[1] + 0.08285) / 0.03774 * 0.3896
    return float(1 / (1 + np.exp(-x)))


def main():
    opt_path = sys.argv[1]
    videos = sys.argv[2:]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(opt_path, "r") as f:
        opt = yaml.safe_load(f)

    evaluator = DOVER(**opt["model"]["args"]).to(device)
    evaluator.load_state_dict(torch.load(opt["test_load_path"], map_location=device))
    evaluator.eval()

    dopt = opt["data"]["val-l1080p"]["args"]
    temporal_samplers = {}
    for stype, sopt in dopt["sample_types"].items():
        if "t_frag" not in sopt:
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
            )
        else:
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"] // sopt["t_frag"],
                sopt["t_frag"],
                sopt["frame_interval"],
                sopt["num_clips"],
            )

    for vp in videos:
        try:
            views, _ = spatial_temporal_view_decomposition(
                vp, dopt["sample_types"], temporal_samplers
            )
            for k, v in views.items():
                num_clips = dopt["sample_types"][k].get("num_clips", 1)
                views[k] = (
                    ((v.permute(1, 2, 3, 0) - mean) / std)
                    .permute(3, 0, 1, 2)
                    .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                    .transpose(0, 1)
                    .to(device)
                )
            with torch.no_grad():
                results = [r.mean().item() for r in evaluator(views)]
            print(f"DOVER_SCORE {vp} {fuse_results(results):.6f}", flush=True)
        except Exception as e:  # one bad chunk must not kill the batch
            print(f"DOVER_FAIL {vp} {type(e).__name__}:{e}", flush=True)


if __name__ == "__main__":
    main()
