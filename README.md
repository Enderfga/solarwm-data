<div align="center">

# SolarWM Data Engine

The open data pipeline for
**SolarWM: Open Data and Scalable Training for Long-Horizon Video World Models**

[Project page](https://junchao-cs.github.io/SolarWM-Web/) ·
[Dataset](https://huggingface.co/datasets/junchaoh-cs/SolarWM-Data) ·
[Model weights](https://huggingface.co/collections/junchaoh-cs/solarwm) ·
[Paper](https://arxiv.org/pdf/2609.02886)

</div>

The engine turns raw video and multi-view scenes into camera-controllable
world-model training data. Each clip carries a metric-scale 6-DoF camera
trajectory, per-frame intrinsics, a scene-static caption, quality metadata,
selection decisions, and provenance aligned to the same frame indices.

This is the code that produced the released SolarWM corpus: **1,425,694 canonical clips
from 14 datasets, totaling 25.85 TB across 29,272 shards.** 876,593 clips pass the released
selection policy — 471,798 in the `high` tier and 404,795 in `xhigh` — and the remaining
549,101 are published as well, fully annotated, each with the machine-readable reasons it
was turned down.

This standalone repository is also the data-engine submodule of SolarWM. The
main project provides training and inference; the
[SolarWM-Data](https://huggingface.co/datasets/junchaoh-cs/SolarWM-Data)
repository provides release controls and the annotation reconstruction package.

## An engine, not a list of clips

Publishing a corpus as a fixed list of accepted clips throws away the expensive part.
Camera estimation, captioning and the quality models cost a GPU pass per clip; selection is
only metadata computed from what they produced. So this engine annotates **everything
first and decides afterwards** — nothing is discarded for failing a threshold.

That inverts what a change costs. Raising a quality floor, dropping a metric you do not
trust, rebalancing the mixture, holding out a different split, or cutting a different
window length for a different backbone is an index pass over metadata, not another run
across the fleet. The released `high` and `xhigh` tiers are one convenient policy over the
corpus, not a definition of which clips are valid.

Four layers stay separable, and changing one never rebuilds the one below it:

| Layer | What it fixes | Cost to change |
|---|---|---|
| Physical corpus | canonical clips, annotations, metrics, verdicts | a GPU pass per clip |
| Logical recipe | split membership, tiers, source weights, repeats | an index |
| Model view | the fixed-length windows a backbone reads | an index |
| Packed shards | WebDataset tars, partitioned by tier | a copy |

## What the corpus contains

Fourteen independently addressable dataset owners, spanning real-world, synthetic and game
environments. All 1,425,694 clips were captioned in one pass by Kimi-K2.6 under a single
frozen prompt and response contract, so caption style does not vary by source.

| Dataset owner | All | High | xhigh | Rejected |
|---|---:|---:|---:|---:|
| ABOT | 30,966 | 127 | 30,715 | 124 |
| DL3DV-10s | 120,924 | 54,528 | 60,396 | 6,000 |
| DL3DV-60s | 10,077 | 3,578 | 6,065 | 434 |
| MIND | 533 | 117 | 402 | 14 |
| MiraData | 140,877 | 3,683 | 17,806 | 119,388 |
| MiraData-Clean | 135,224 | 12,865 | 5,740 | 116,619 |
| MultiCamVideo | 123,117 | 89,587 | 5,369 | 28,161 |
| OmniWorld | 19,632 | 3,773 | 13,552 | 2,307 |
| RealCam | 45,697 | 16,154 | 16,074 | 13,469 |
| Sekai-Game | 2,550 | 537 | 1,410 | 603 |
| Sekai-Walking | 22,990 | 6,054 | 12,976 | 3,960 |
| Sekai-Walking-Clean | 109,248 | 46,831 | 33,660 | 28,757 |
| SpatialVID | 365,345 | 100,815 | 127,180 | 137,350 |
| SpatialVID-Clean | 298,514 | 133,149 | 73,450 | 91,915 |
| **Total** | **1,425,694** | **471,798** | **404,795** | **549,101** |

Three of those owners are **Clean Plate** derivatives — 542,986 clips in which people and
vehicles are removed while scene layout and camera motion are retained. Dynamic actors add
motion the camera does not explain and the user cannot control, so a clean sibling is a
different kind of supervision, not a better version of its source: each is its own recipe
owner, scored from scratch rather than inheriting its source's metrics.

## What one clip looks like

```
<corpus>/<source>/<clip_id>/
├── video.mp4         # the clip, at the spec's frame rate
├── poses.npy         # (N, 4, 4) float64 — camera-to-world, METRIC scale
├── intrinsics.npy    # (N, 4) float64 — per-frame (fx, fy, cx, cy) in pixels
├── prompt.txt        # scene-static dense caption
├── meta.json         # source, metrics, verdict + reasons, provenance
└── gt_depth.npz      # only where the source ships metric GT depth
```

Three properties hold or the clip is not usable, and none of them are visible by watching
it: `frame_count == len(poses) == len(intrinsics)` with index `i` meaning the same instant
in each; `poses.npy` is camera-to-world, so the camera centre is the translation column
`M[:3,3]` **directly**, not `-Rᵀt`; and translations are metric, recovered per clip.
`scripts/validate_clip.py` checks the first and is worth running on any source whose
extraction or pose code you touch.

## Install

```bash
pip install -e ".[full]"            # from this directory
solarwm-pipeline spec list          # the output specs, and which one is active
python3 scripts/validate_dryrun.py  # the whole pipeline on synthetic input, no GPU
```

That much is pure Python and needs no GPU. Producing real annotations additionally needs a
CUDA PyTorch environment, the external tools fetched by `scripts/setup_*.sh`, and
`dry_run: false` in `configs/models.yaml` — the shipped default emits placeholders that
pass every structural check. Copy `.env.example`, fill it in, and source it; no credential
is baked into the code.

## Documentation

The detailed pipeline guide is
[`skills/solarwm-data-pipeline/SKILL.md`](skills/solarwm-data-pipeline/SKILL.md).
It covers:

- adding your own raw source, from probing its layout to wiring it in
- running the fleet across a worker pool and keeping it busy
- the output specs, the shipped sources, and how pose modes are chosen
- calibrating quality thresholds against **your** data rather than ours
- verifying that what came out is structurally, geometrically and visually correct
- assembling recipes, model views and packed shards
- the invariants that must not change, and what to check when something is wrong

## Citation

**If you use SolarWM-Data, the data engine, or the released models in your research,
please cite our paper.**

Paper: https://arxiv.org/abs/2609.02886

```bibtex
@misc{huang2026solarwmopendatascalable,
      title={SolarWM: Open Data and Scalable Training for Long-Horizon Video World Models}, 
      author={Junchao Huang and Guian Fang and Shengju Qian and Xianghao Kong and Zhuoran Zhao and Wei Huang and Yihua Du and Zixin Zhang and Justin Cui and Yuchao Gu and Yukang Chen and Xinting Hu and Tianyu He and Shaoshuai Shi and Zhuotao Tian and Xin Wang and Mike Zheng Shou and Li Jiang},
      year={2026},
      eprint={2609.02886},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.02886}, 
}
```

## Acknowledgements

The camera path combines Pi3X structure, MoGe-2 metric depth, and a VIPE SLAM
backend with per-frame intrinsics. We thank those projects and SANA-WM for the
work this implementation builds on. See `NOTICE` for complete attribution and
license information.

## Licence

Apache-2.0 — see `LICENSE`. The external models this engine drives, and the datasets it
annotates, each keep their own terms; several are non-commercial. See `NOTICE`.
