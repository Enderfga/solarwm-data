#!/usr/bin/env python3
"""Export a minimal pointer-only recipe for rebuilding selected clips from public
source media without transferring derived annotations.

A recipe line is:
    {"source": "...", "item": "<public-source item id>", "kept_clips": [clip_id, ...]}

It contains no poses, captions, metrics, or video; it records only the released
selection decision. The reproduce worker (run_solarwm_fleet with
SOLAR_WM_REPRODUCE=<recipe>) then, per source: processes only items that appear here,
runs pose+caption only on the listed clips, and trusts the keep verdict (no filter
re-run). The ~75% rejected clips never touch a GPU.

Inputs:
  * train_list.jsonl  — the authoritative kept set (from assemble_corpus.py): the
    keep decision is taken from HERE (current filters.yaml + overrides), not from the
    per-item manifests' original verdicts.
  * per-item manifests on S3 (corpus/<source>/manifest/<item>.jsonl) — used ONLY to
    map clip_id -> item (which public-source item each kept clip came from).

Every kept clip MUST map to an item; unmapped clips are reported loudly (a source with
unmapped clips falls back to no item-filter — see `--report`). Run with S3 creds:
    AWS_PROFILE=... SOLAR_WM_S3_BUCKET=<bucket> \
      python3 scripts/export_recipe.py --train-list <assembly-dir>/train_list.jsonl \
        --out /tmp/recipe.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bucket", default=os.environ.get("SOLAR_WM_S3_BUCKET"))
    ap.add_argument("--prefix", default=os.environ.get("SOLAR_WM_CORPUS_PREFIX", "corpus"))
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    if not a.bucket:
        sys.exit("set --bucket or SOLAR_WM_S3_BUCKET")
    import boto3
    s3 = boto3.client("s3")

    # 1. authoritative kept clip_ids per source (from the train list)
    kept = defaultdict(set)
    with open(a.train_list) as f:
        for line in f:
            r = json.loads(line)
            kept[r["source"]].add(r["clip_id"])
    print(f"train_list: {sum(len(v) for v in kept.values())} kept clips across {len(kept)} sources",
          flush=True)

    rows = []          # (source, item, [kept_clips])
    report = {}
    for source, kset in sorted(kept.items()):
        man_pre = f"{a.prefix}/{source}/manifest/"
        man_keys = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=a.bucket, Prefix=man_pre):
            man_keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".jsonl")]

        def scan(key):
            item = key[len(man_pre):-len(".jsonl")]
            body = s3.get_object(Bucket=a.bucket, Key=key)["Body"].read().decode()
            hits = []
            for ln in body.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    cid = json.loads(ln).get("clip_id")
                except Exception:
                    continue
                if cid in kset:
                    hits.append(cid)
            return (item, hits) if hits else None

        mapped = set()
        with ThreadPoolExecutor(max_workers=a.threads) as pool:
            for res in pool.map(scan, man_keys):
                if res:
                    item, hits = res
                    rows.append((source, item, hits))
                    mapped.update(hits)
        unmapped = kset - mapped
        report[source] = {"kept": len(kset), "mapped": len(mapped), "unmapped": len(unmapped),
                          "items": sum(1 for r in rows if r[0] == source)}
        flag = "" if not unmapped else f"  ⚠ {len(unmapped)} UNMAPPED (will need no-item-filter)"
        print(f"  {source}: {len(kset)} kept -> {report[source]['items']} items, "
              f"{len(mapped)} mapped{flag}", flush=True)

    with open(a.out, "w") as f:
        for source, item, clips in rows:
            f.write(json.dumps({"source": source, "item": item, "kept_clips": sorted(clips)}) + "\n")
    with open(a.out + ".report.json", "w") as f:
        json.dump(report, f, indent=1)
    total_unmapped = sum(r["unmapped"] for r in report.values())
    print(f"\nDONE: {len(rows)} recipe rows -> {a.out}", flush=True)
    if total_unmapped:
        print(f"⚠ {total_unmapped} kept clips have no manifest item — those sources need "
              f"item-filter OFF (reproduce still finds them via clip-filter, just re-acquires "
              f"all items). See {a.out}.report.json", flush=True)


if __name__ == "__main__":
    main()
