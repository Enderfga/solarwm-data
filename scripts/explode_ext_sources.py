#!/usr/bin/env python3
"""Prepare unshardable extension-source archives for parallel processing (CPU-only).

Streams the raw S3 archives and writes a per-scene or per-sequence layout that can be
sharded at item granularity:

  multicam:  raw/multicamvideo/MultiCamVideo-Dataset.part{aa..ap}  (ONE 333GB gz stream
             split into 16 parts — gzip can't seek, so a single sequential pass)
        ->   raw/multicamvideo/exploded/train/<fdir>/<sceneN>/{cameras/...json, videos/camNN.mp4}

  zod:       raw/zod/downloads/sequences/{infos.tar.gz_*, images_blur_<a>_<b>.tar.gz_*}
             (3 independent image tars -> parallelisable via --only)
        ->   raw/zod/exploded/sequences/<id>/{calibration.json, ego_motion.json,
             camera_front_blur/<name>.jpg}

Resume-safe: a per-scene/per-seq done-marker under <exploded>/.done/ skips re-upload
(the gz stream must still be read through — only uploads are saved on resume).

Usage (with AWS credentials in the environment):
  python3 explode_ext_sources.py multicam
  python3 explode_ext_sources.py zod [--only images_blur_000000_000490]   # one tar per worker
  python3 explode_ext_sources.py zod --only infos
"""
from __future__ import annotations

import argparse
import os
import io
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

import boto3

BKT = os.environ.get("SOLAR_WM_S3_BUCKET") or ""
if not BKT:
    raise SystemExit("SOLAR_WM_S3_BUCKET not set (the raw/corpus bucket)")
S3 = boto3.client("s3")
UP = ThreadPoolExecutor(max_workers=16)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def exists(key: str) -> bool:
    try:
        S3.head_object(Bucket=BKT, Key=key)
        return True
    except Exception:  # noqa: BLE001
        return False


def put(key: str, data: bytes):
    S3.put_object(Bucket=BKT, Key=key, Body=data)


class ChainStream(io.RawIOBase):
    """Sequential read()-only stream over a list of S3 keys (for split gz parts)."""

    def __init__(self, keys):
        self.keys = list(keys)
        self.body = None

    def readable(self):  # noqa: D102
        return True

    def read(self, n=-1):  # noqa: D102
        while True:
            if self.body is None:
                if not self.keys:
                    return b""
                k = self.keys.pop(0)
                log(f"  stream part: {k}")
                self.body = S3.get_object(Bucket=BKT, Key=k)["Body"]
            chunk = self.body.read(n if n and n > 0 else 1 << 20)
            if chunk:
                return chunk
            self.body = None  # part exhausted -> next

    def readinto(self, b):  # BufferedReader calls THIS (RawIOBase.read defers to it)
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def explode_multicam():
    pre = "raw/multicamvideo/"
    # exactly the 16 top-level split parts — ".cache/" holds a duplicate HF-download copy
    parts = sorted(
        o["Key"] for p in S3.get_paginator("list_objects_v2").paginate(Bucket=BKT, Prefix=pre)
        for o in p.get("Contents", [])
        if o["Key"].startswith(pre + "MultiCamVideo-Dataset.part"))
    log(f"multicam: {len(parts)} parts")
    out_pre = pre + "exploded/"
    tf = tarfile.open(fileobj=io.BufferedReader(ChainStream(parts), 1 << 24), mode="r|gz")
    n_up = n_skip = n_done = 0
    # A scene's members are NOT contiguous in the tar (verified: json+5 videos in one
    # run, the other 5 videos much later) — so completion is judged by CONTENT, never
    # by order: the marker is written only once all 11 expected files are uploaded.
    expected = {"cameras/camera_extrinsics.json"} | {f"videos/cam{i:02d}.mp4" for i in range(1, 11)}
    got: dict = {}        # scene -> set(subpaths uploaded)
    futs_of: dict = {}    # scene -> [futures]
    skip_cache: dict = {} # scene -> marker-exists (one HEAD per scene)
    for m in tf:
        if not m.isfile():
            continue
        # member: MultiCamVideo-Dataset/train/<fdir>/<scene>/...  -> exploded/train/...
        rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
        scene = "/".join(rel.split("/")[:3])          # train/<fdir>/<scene>
        done = f"{out_pre}.done/{scene.replace('/', '_')}.done"
        if scene not in skip_cache:
            skip_cache[scene] = exists(done)
        if skip_cache[scene]:
            n_skip += 1
            continue
        data = tf.extractfile(m).read()
        futs_of.setdefault(scene, []).append(UP.submit(put, out_pre + rel, data))
        n_up += 1
        got.setdefault(scene, set()).add(rel[len(scene) + 1:])
        if expected <= got[scene]:                    # all 11 present -> safe to mark
            for f in futs_of.pop(scene):
                f.result()
            put(done, b"ok")
            skip_cache[scene] = True                  # later duplicate members skip
            del got[scene]
            n_done += 1
        if n_up % 500 == 0:
            log(f"  uploaded {n_up} files (scenes done {n_done}, skip {n_skip})")
    for fs in futs_of.values():
        for f in fs:
            f.result()
    if got:                                           # incomplete scenes get NO marker
        log(f"  WARNING: {len(got)} scenes ended INCOMPLETE (no marker; rerun resumes them): "
            f"{list(got)[:5]}")
    log(f"multicam done: {n_up} files, {n_done} scenes marked, {n_skip} member-skips")


def explode_zod(only: str | None):
    pre = "raw/zod/downloads/sequences/"
    keys = [o["Key"] for p in S3.get_paginator("list_objects_v2").paginate(Bucket=BKT, Prefix=pre)
            for o in p.get("Contents", [])]
    tars = [k for k in keys if ("infos.tar.gz" in k or "images_blur" in k)]
    if only:
        tars = [k for k in tars if only in k]
    out_pre = "raw/zod/exploded/"
    log(f"zod: {len(tars)} tar(s): {[k.split('/')[-1] for k in tars]}")
    for tk in tars:
        body = S3.get_object(Bucket=BKT, Key=tk)["Body"]
        tf = tarfile.open(fileobj=io.BufferedReader(body, 1 << 24), mode="r|gz")  # type: ignore[arg-type]
        n_up = n_skip = 0
        futs = []
        for m in tf:
            if not m.isfile():
                continue
            rel = m.name                              # sequences/<id>/...
            keep = (rel.endswith(("calibration.json", "ego_motion.json"))
                    or "/camera_front_blur/" in rel)
            if not keep:
                continue
            okey = out_pre + rel
            if exists(okey):                          # cheap per-file resume (jsons tiny)
                n_skip += 1
                continue
            data = tf.extractfile(m).read()
            futs.append(UP.submit(put, okey, data))
            n_up += 1
            if n_up % 1000 == 0:
                for f in futs:
                    f.result()
                futs.clear()
                log(f"  {tk.split('/')[-1]}: {n_up} uploaded (skip {n_skip})")
        for f in futs:
            f.result()
        log(f"zod {tk.split('/')[-1]} done: {n_up} uploaded, {n_skip} skipped")


def explode_ditto():
    """Ditto-1M `videos/source/` ONLY (the real source videos — the edited subsets
    re-render the SAME camera trajectories with stylised appearance, no new camera
    signal). 10-part split gz stream -> loose mp4s under raw/ditto-1m/exploded/source/."""
    pre = "raw/ditto-1m/videos/source/"
    parts = sorted(
        o["Key"] for p in S3.get_paginator("list_objects_v2").paginate(Bucket=BKT, Prefix=pre)
        for o in p.get("Contents", []) if ".tar.gz." in o["Key"])
    log(f"ditto source: {len(parts)} parts")
    out_pre = "raw/ditto-1m/exploded/source/"
    have = set()
    for p in S3.get_paginator("list_objects_v2").paginate(Bucket=BKT, Prefix=out_pre):
        have.update(o["Key"] for o in p.get("Contents", []))
    log(f"  resume: {len(have)} mp4s already exploded")
    tf = tarfile.open(fileobj=io.BufferedReader(ChainStream(parts), 1 << 24), mode="r|gz")
    n_up = n_skip = 0
    futs = []
    for m in tf:
        if not (m.isfile() and m.name.lower().endswith(".mp4")):
            continue
        okey = out_pre + m.name.split("/")[-1]
        if okey in have:
            n_skip += 1
            continue
        futs.append(UP.submit(put, okey, tf.extractfile(m).read()))
        n_up += 1
        if n_up % 1000 == 0:
            for f in futs:
                f.result()
            futs.clear()
            log(f"  uploaded {n_up} mp4s (skip {n_skip})")
    for f in futs:
        f.result()
    log(f"ditto done: {n_up} mp4s uploaded, {n_skip} skipped")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["multicam", "zod", "ditto"])
    ap.add_argument("--only", default=None, help="zod: substring to select one tar")
    a = ap.parse_args()
    if a.which == "multicam":
        explode_multicam()
    elif a.which == "ditto":
        explode_ditto()
    else:
        explode_zod(a.only)
    sys.exit(0)
