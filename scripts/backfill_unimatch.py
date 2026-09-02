#!/usr/bin/env python3
"""Fill missing ``metrics.unimatch`` values in stored clip metadata.

The command computes GMFlow from each stored ``video.mp4`` and updates only
``meta.json``. Assembly can then re-evaluate the clip without reprocessing it.

Run with ``third_party/unimatch``, its weights, a GPU, and object-store access:
    LOGDIR=/path/to/logs; mkdir -p "$LOGDIR"
    for g in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$g python3 scripts/backfill_unimatch.py \
        --source dl3dv --shard $g --world 8 >"$LOGDIR"/backfill_uni_$g.log 2>&1 &
    done

Clips whose metadata already has a non-null UniMatch value are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

WM = os.environ.get("SOLAR_WM_ROOT") or str(Path(__file__).resolve().parents[1])
sys.path.insert(0, WM)

from solar_wm_data import cos_io  # noqa: E402
from solar_wm_data.config import load_config  # noqa: E402
from solar_wm_data.filter.adapters import unimatch_flow  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    models_cfg = load_config("models")
    pre = f"{cos_io.corpus_prefix(a.source)}/clips/"
    seen, cids = set(), []
    for k in cos_io.list_keys(pre):
        cid = k[len(pre):].split("/")[0]
        if cid and cid not in seen:
            seen.add(cid)
            cids.append(cid)
    cids = sorted(cids)[a.shard::a.world]
    if a.limit:
        cids = cids[:a.limit]
    scratch = os.environ.get("SOLAR_WM_SCRATCH", "/tmp") + f"/backfill_uni_{a.shard}"
    os.makedirs(scratch, exist_ok=True)
    n_done = n_skip = n_err = 0
    print(f"[{time.strftime('%H:%M:%S')}] shard {a.shard}/{a.world}: {len(cids)} clips", flush=True)
    for i, cid in enumerate(cids):
        mkey = f"{pre}{cid}/meta.json"
        try:
            mloc = f"{scratch}/meta.json"
            cos_io.get_file(mkey, mloc, skip_if_exists=False)
            meta = json.load(open(mloc))
            if (meta.get("metrics") or {}).get("unimatch") is not None:
                n_skip += 1
                continue
            vid = f"{scratch}/video.mp4"
            cos_io.get_file(f"{pre}{cid}/video.mp4", vid, skip_if_exists=False)
            val = unimatch_flow(vid, models_cfg)
            meta.setdefault("metrics", {})["unimatch"] = val
            cos_io.put_bytes(json.dumps(meta).encode(), mkey)
            os.remove(vid)
            n_done += 1
        except Exception as e:  # noqa: BLE001 - count and continue; rerun resumes
            n_err += 1
            print(f"  ERR {cid}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] {i + 1}/{len(cids)} "
                  f"(done {n_done}, skip {n_skip}, err {n_err})", flush=True)
    print(f"FINISHED shard {a.shard}: done {n_done}, skip {n_skip}, err {n_err}", flush=True)


if __name__ == "__main__":
    main()
