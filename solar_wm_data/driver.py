"""Pipeline driver: chain ingest -> pose -> filter -> caption -> package.

Each stage is resumable: a record already carrying a stage's output fields is
skipped, so a partially-processed manifest can be re-run cheaply.

CLI:
    solarwm-pipeline run    --manifest M --out OUT [--stage ...]
    solarwm-pipeline ingest --root R --source S --manifest M
    solarwm-pipeline spec   list | show <name>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from . import spec as _spec
from .config import load_config
from .manifest import ClipRecord, read_manifest, write_manifest
from .ingest import ingest_source
from .pose.stage import annotate_pose
from .filter.stage import filter_clip
from .caption import caption_clip

STAGES = ["pose", "filter", "caption", "package"]


# --------------------------------------------------------------------------- #
# package stage
# --------------------------------------------------------------------------- #
def store_all_default() -> bool:
    """Whether rejected clips are packaged too. ON unless explicitly turned off.

    The engine's premise is that every clip is annotated before any selection is applied,
    so a clip the current thresholds reject is kept with its annotations and reasons and
    can be re-judged without recomputing annotations.
    """
    return os.environ.get("SOLAR_WM_STORE_ALL", "1") != "0"


def package_clip(rec: ClipRecord, out_root: Path, store_all: bool = True) -> ClipRecord:
    """Emit final per-clip dir {video.mp4, poses.npy, intrinsics.npy, prompt.txt, meta.json}.

    Mirrors the converted per-clip layout the training reader expects.
    Every clip is packaged by default. With ``store_all=False``, only kept clips are
    packaged. The per-clip ``meta.json`` records the verdict, reasons, and metrics so
    filtering can also be deferred to a downstream selection step.
    """
    if not rec.kept and not store_all:
        return rec
    d = out_root / rec.source / rec.clip_id
    d.mkdir(parents=True, exist_ok=True)
    if rec.video_path and Path(rec.video_path).exists():
        shutil.copy(rec.video_path, d / "video.mp4")
    if rec.pose_path and Path(rec.pose_path).exists():
        shutil.copy(rec.pose_path, d / "poses.npy")
    if rec.intrinsics_path and Path(rec.intrinsics_path).exists():
        shutil.copy(rec.intrinsics_path, d / "intrinsics.npy")
    # Optional GT depth (OmniWorld): float16 metric depth (invalid=-1) keyed "depth"
    # in a compressed npz, frame-aligned to video.mp4. A bonus annotation for depth
    # supervision; the camera trajectory still comes from the exact GT poses.
    gt_depth = rec.extra.get("gt_depth_path")
    if gt_depth and Path(gt_depth).exists():
        shutil.copy(gt_depth, d / "gt_depth.npz")
    # Optional companions: present iff the source had them (see ingest_clip_dir).
    for key, name in (("audio_path", "audio.m4a"), ("action_path", "action.npy")):
        src = rec.extra.get(key)
        if src and Path(src).exists():
            shutil.copy(src, d / name)
    (d / "prompt.txt").write_text(rec.caption or "", encoding="utf-8")
    # Per-clip verdict + metrics — lets downstream filtering select by kept / any
    # metric threshold without recomputing anything (meta.json["kept"] is the
    # verdict under the active thresholds; reject_reasons + metrics carry the raw values).
    (d / "meta.json").write_text(json.dumps(rec.to_dict(), ensure_ascii=False), encoding="utf-8")
    rec.extra["packaged_dir"] = str(d)
    return rec


# --------------------------------------------------------------------------- #
# stage dispatch (resumable)
# --------------------------------------------------------------------------- #
def _needs(rec: ClipRecord, stage: str) -> bool:
    if stage == "pose":
        return rec.pose_path is None
    if stage == "filter":
        return rec.kept is None
    if stage == "caption":
        return rec.caption is None
    if stage == "package":
        # Judged (kept True or False) and not yet packaged. Gating on `kept is True` here
        # would skip rejected clips regardless of store_all and quietly drop them.
        return rec.kept is not None and "packaged_dir" not in rec.extra
    return True


def run_pipeline(
    records: list[ClipRecord], out: Path, stages: list[str],
    filters_cfg: dict, models_cfg: dict, store_all: bool | None = None,
) -> list[ClipRecord]:
    work = out / "work"
    if store_all is None:
        store_all = store_all_default()
    if models_cfg.get("dry_run", True):
        # Loud, because the placeholders are plausible: synthetic forward-motion poses,
        # metrics seeded from the clip id, captions drawn from a handful of templates.
        # A dry run that is mistaken for a real one produces a corpus that validates.
        print("WARNING: dry_run is enabled — poses, metrics and captions are PLACEHOLDERS, "
              "not measurements. Set dry_run: false in the models config for real output.")
    for rec in records:
        if "pose" in stages and _needs(rec, "pose"):
            annotate_pose(rec, work / "pose", models_cfg)
        if "filter" in stages and _needs(rec, "filter"):
            filter_clip(rec, filters_cfg, models_cfg)
        if "caption" in stages and _needs(rec, "caption"):
            rec.caption = caption_clip(rec, models_cfg)
        if "package" in stages and _needs(rec, "package"):
            package_clip(rec, out / "corpus", store_all=store_all)
    return records


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="solarwm-pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="ingest a source into a manifest")
    pi.add_argument("--root", required=True)
    pi.add_argument("--source", required=True)
    pi.add_argument("--manifest", required=True)

    pr = sub.add_parser("run", help="run pose/filter/caption/package over a manifest")
    pr.add_argument("--manifest", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--stage", action="append", choices=STAGES, default=None)
    pr.add_argument("--filters-config", default="filters")
    pr.add_argument("--models-config", default="models")

    ps = sub.add_parser("spec", help="list or inspect the output specs")
    ps.add_argument("action", choices=["list", "show"])
    ps.add_argument("name", nargs="?", help="spec name, or an inline <frames>@<fps>")

    for sp in (pi, pr):
        sp.add_argument("--spec", default=None,
                        help="output spec: a catalogue name or an inline <frames>@<fps>. "
                             "Sets SOLAR_WM_SPEC for every stage in this run.")

    args = p.parse_args(argv)

    # One spec per run, resolved ONCE and pushed into the environment, because every
    # stage reads it through solar_wm_data.spec. Passing it per-stage is how a run ends
    # up cutting windows at one length and validating them against another.
    if getattr(args, "spec", None):
        _spec.parse_spec(args.spec)          # fail here, not three stages in
        os.environ["SOLAR_WM_SPEC"] = args.spec

    if args.cmd == "spec":
        if args.action == "list":
            active = _spec.current_spec()
            print(f"{'spec':>8}  {'frames':>6}  {'fps':>3}  {'seconds':>8}  {'4n+1':>5}")
            for name in _spec.SPEC_FRAMES:
                f, r = _spec.parse_spec(name)
                mark = " *" if name == active else ""
                print(f"{name:>8}  {f:6d}  {r:3d}  {f / r:8.4f}  {str(_spec.is_latent_aligned(f)):>5}{mark}")
            print("\n* = active. An inline <frames>@<fps> (e.g. 241@24) works anywhere a name does.")
            return 0
        name = args.name or _spec.current_spec()
        f, r = _spec.parse_spec(name)
        print(f"spec            {name}")
        print(f"frames          {f}")
        print(f"fps             {r}")
        print(f"seconds         {f / r:.4f}")
        print(f"latent-aligned  {_spec.is_latent_aligned(f)} (4n+1)")
        if not _spec.is_latent_aligned(f):
            print("                NOTE: not 4n+1, so it is not a latent-aligned model "
                  "window. Canonical clip lengths need not be.")
        return 0

    if args.cmd == "ingest":
        recs = ingest_source(args.root, args.source)
        write_manifest(args.manifest, recs)
        print(f"ingested {len(recs)} clips -> {args.manifest}")
        return 0

    if args.cmd == "run":
        records = read_manifest(args.manifest)
        filters_cfg = load_config(args.filters_config)
        models_cfg = load_config(args.models_config)
        stages = args.stage or STAGES
        run_pipeline(records, Path(args.out), stages, filters_cfg, models_cfg)
        write_manifest(args.manifest, records)
        kept = sum(1 for r in records if r.kept)
        print(f"ran {stages} on {len(records)} clips; kept {kept} -> {args.manifest}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
