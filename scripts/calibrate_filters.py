#!/usr/bin/env python3
"""Emit per-source threshold configs from measured corpus distributions.

WHY PERCENTILES AND NOT NUMBERS. Thresholds copied between corpora do not mean what they
meant at home. A DOVER floor of 0.40 set on one-minute clips is far harsher on 5.04 s ones,
because DOVER averages over 5 s chunks: a 60 s clip averages twelve of them and a 5 s clip
has exactly one, so the same numeric floor sits at a different place in the distribution.
Optical-flow ceilings move with fps and clip length for the same reason. What DOES transfer
is the intent — "cut the fastest tenth", "keep the top half by quality" — so this script
expresses that intent as percentiles of the selected corpus.

Two tiers, differing only in how much of each tail they take:

  main   flow [p02,p90]  dover >=p25  vmaf <=p97  sat <=p97   quality >=4
  elite  flow [p05,p75]  dover >=p50  vmaf <=p90  sat <=p90   quality  =5

Camera-motion gate (from scripts/traj_stats.py, the axis no pixel metric covers): reject the
least-moving tail of each source — main drops the bottom 2% of trajectory length, elite the
bottom 15%. It is a per-source percentile rather than an absolute distance ON PURPOSE: metric
scale is recovered per source, and one absolute floor would silently delete whole sources
whose recovered scale runs small.

Flag policy is per source and derived from measured rates: a flag that appears on more than
`--flag-tolerate` of a source's clips is treated as characteristic of the source (dropping it
would not clean the source, it would delete it) and is tolerated in `main`; elite tolerates
none. low_light is always tolerated — night is a legitimate part of the distribution, and
dropping it biases the corpus's lighting rather than cleaning it.

    python3 scripts/calibrate_filters.py --traj <dir> \
        --out-main  configs/filters_calibrated.yaml \
        --out-elite configs/filters_calibrated_xhigh.yaml [--sample 8000]

Both outputs are WRITTEN by this script; neither needs to exist first. Review the
result and copy it over configs/filters.yaml deliberately — do not point --out-main
straight at the live config, or a bad calibration run silently becomes policy.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WM = os.environ.get("SOLAR_WM_ROOT") or str(Path(__file__).resolve().parents[1])
sys.path.insert(0, WM)

from solar_wm_data import cos_io  # noqa: E402
from solar_wm_data.config import load_config  # noqa: E402

# Sources whose footage is continuous by construction (engine render, single capture, sim):
# a "scene transition" there is a false positive of fast motion, not an edit. Measured: the
# VLM and PySceneDetect disagree almost completely on these (abot 61 vs 19 flagged, 3 in
# common; a simulator source's detector called 38% of its continuous trajectories cuts).
CONTINUOUS = {"abot", "dl3dv", "omniworld", "sekai_game",
              "multicamvideo"}
ALWAYS_TOLERATED = {"low_light"}
FLAGS = ("watermark", "ui_overlay", "low_light", "blurry", "near_static", "text_heavy",
         "single_color")


def pct(vals: list[float], p: float) -> float | None:
    v = sorted(x for x in vals if x is not None)
    return v[min(len(v) - 1, int(p / 100 * len(v)))] if v else None


def r2(x: float | None, nd: int = 2) -> float | None:
    return None if x is None else round(float(x), nd)


def measure(src: str, sample: int, threads: int, traj_dir: Path | None) -> dict | None:
    pre = f"{cos_io.corpus_prefix(src)}/clips/"
    ids = sorted({k[len(pre):].split("/")[0] for k in cos_io.list_keys(pre)
                  if k.endswith("/meta.json")})
    if not ids:
        return None
    if sample and len(ids) > sample:
        ids = ids[::max(1, len(ids) // sample)][:sample]
    vroot = Path(os.environ.get("SOLAR_WM_LOCAL_ROOT", "")) / "vlm_anno" / \
        f"{src}-{os.environ.get('SOLAR_WM_RUN_ID', '').strip()}"

    def one(cid):
        try:
            m = json.loads(cos_io.get_bytes(f"{pre}{cid}/meta.json")).get("metrics") or {}
        except Exception:  # noqa: BLE001
            return None
        try:
            v = json.loads((vroot / f"{cid}.json").read_text()).get("response") or {}
        except Exception:  # noqa: BLE001
            v = {}
        dt, da = m.get("dover_tech"), m.get("dover_aes")
        return {"flow": m.get("unimatch"), "vmaf": m.get("vmaf"), "sat": m.get("saturation"),
                "dover": (dt + da) / 2 if dt is not None and da is not None else None,
                "flags": set(v.get("reject_flags") or [])}
    with ThreadPoolExecutor(threads) as ex:
        rs = [r for r in ex.map(one, ids) if r]
    if not rs:
        return None
    n = len(rs)
    out = {"n": n,
           "flow": [pct([r["flow"] for r in rs], p) for p in (2, 5, 75, 90)],
           "dover": [pct([r["dover"] for r in rs], p) for p in (25, 50)],
           "vmaf": [pct([r["vmaf"] for r in rs], p) for p in (90, 97)],
           "sat": [pct([r["sat"] for r in rs], p) for p in (90, 97)],
           "flag_rate": {f: sum(1 for r in rs if f in r["flags"]) / n for f in FLAGS}}
    tf = (traj_dir / f"{src}.jsonl") if traj_dir else None
    if tf and tf.exists():
        paths = [json.loads(line).get("path") for line in open(tf)]
        out["path_p02"] = pct(paths, 2)
        out["path_p15"] = pct(paths, 15)
        out["path_p50"] = pct(paths, 50)
    return out


def rows(src: str, m: dict, tol: float, elite: bool, path_max: float | None = None) -> dict:
    flow = [m["flow"][1], m["flow"][2]] if elite else [m["flow"][0], m["flow"][3]]
    d = {
        "vmaf": [0.5, r2(m["vmaf"][0] if elite else m["vmaf"][1], 1)],
        "unimatch": [r2(flow[0], 1), r2(flow[1], 1)],
        "dover": [r2(m["dover"][1] if elite else m["dover"][0], 3), 1.0],
        "color_sat": [0, r2(m["sat"][0] if elite else m["sat"][1], 1)],
        "scene_cuts": None if src in CONTINUOUS else (0 if elite else 1),
        "vlm_entity_density": None,
        "vlm_quality": None,
    }
    tolerated = set() if elite else {
        f for f, r in m["flag_rate"].items() if r > tol} | ALWAYS_TOLERATED
    if elite:
        tolerated = ALWAYS_TOLERATED
    sem = {"quality_min": 5 if elite else 4,
           "flags_allowed": sorted(tolerated)}
    if src not in CONTINUOUS:
        sem["transition_max"] = 0
    key = "path_p15" if elite else "path_p02"
    if m.get(key) is not None:
        sem["path_min_m"] = r2(m[key], 3)
    if path_max is not None:
        sem["path_max_m"] = r2(path_max, 1)
    d["semantic"] = sem
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-main", required=True)
    ap.add_argument("--out-elite", required=True)
    ap.add_argument("--traj", default="")
    ap.add_argument("--sources", default="abot,dl3dv,miradata,spatialvid,sekai_walking,"
                                         "realcam_vid,omniworld,sekai_game")
    ap.add_argument("--sample", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--flag-tolerate", type=float, default=0.10)
    ap.add_argument("--max-speed-mps", type=float, default=100.0,
                    help="physical sanity ceiling on mean camera speed (100 m/s = 360 km/h)")
    a = ap.parse_args()
    traj = Path(a.traj) if a.traj else None
    base = load_config("filters")
    # One ABSOLUTE ceiling, shared by every source and both tiers: no camera in this corpus
    # travels 360 km/h, so anything above it is a diverged metric scale rather than fast
    # motion. Measured maxima before this gate: 1.27e9 m (miradata), 3.1e8 (spatialvid),
    # 8.9e6 (dl3dv) — all inside one 5.04 s clip.
    from solar_wm_data import spec as _spec
    path_max = a.max_speed_mps * _spec.target_seconds()
    head = ("# GENERATED by scripts/calibrate_filters.py from the corpus's own measured\n"
            "# distributions — every bound is a percentile of THIS corpus, not a number\n"
            "# carried over from a differently-specced one. Regenerate rather than hand-edit\n"
            "# the bounds; hand-edit the intent (the percentiles) in the script.\n")
    cfgs = {a.out_main: {}, a.out_elite: {}}
    for src in [s.strip() for s in a.sources.split(",") if s.strip()]:
        m = measure(src, a.sample, a.threads, traj)
        if not m:
            print(f"{src}: no clips, skipped", flush=True)
            continue
        print(f"{src}: n={m['n']} flow p02/p90={m['flow'][0]:.0f}/{m['flow'][3]:.0f} "
              f"p05/p75={m['flow'][1]:.0f}/{m['flow'][2]:.0f} "
              f"dover p25/p50={m['dover'][0]:.3f}/{m['dover'][1]:.3f} "
              f"path p02/p15={m.get('path_p02')}/{m.get('path_p15')} "
              f"flags>{a.flag_tolerate:.0%}: "
              f"{sorted(f for f, r in m['flag_rate'].items() if r > a.flag_tolerate)}",
              flush=True)
        cfgs[a.out_main][src] = rows(src, m, a.flag_tolerate, False, path_max)
        cfgs[a.out_elite][src] = rows(src, m, a.flag_tolerate, True, path_max)
    import yaml
    for path, ds in cfgs.items():
        with open(path, "w") as fh:
            fh.write(head)
            yaml.safe_dump({"camera": base["camera"], "datasets": ds}, fh,
                           sort_keys=False, default_flow_style=None)
        print(f"wrote {path} ({len(ds)} sources)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
