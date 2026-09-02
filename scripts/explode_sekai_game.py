#!/usr/bin/env python3
"""One-time prep: explode the monolithic Sekai-Game zips into per-clip files in S3.

Sekai-Game ships only 2 archives (sekai-game-drone.zip 8.9GB, sekai-game-walking.zip
~90GB split). With item = subset, the fleet gives sekai_game only 2 work-items -> only
2 of N workers do anything and each re-encodes ~900 clips serially. This script downloads
each zip ONCE, extracts, and uploads each clip's <id>.mp4 + <id>.npz to

    s3://<bucket>/<RAW_PREFIX>/sekai-game/exploded/<subset>/<id>.{mp4,npz}

After this, `_list_sekai_game` enumerates per-clip items and every worker fetches only
its own clips (KB-MB), so sekai_game shards across the whole fleet.

Idempotent: a clip already present in S3 is skipped (resume-safe). Run once per subset:
    SUBSET=drone   python3 scripts/explode_sekai_game.py
    SUBSET=walking python3 scripts/explode_sekai_game.py
Reads raw from the S3 mirror (same env the fleet uses: SOLAR_WM_STORAGE=s3, bucket, RAW_PREFIX).
"""
from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path

WM = os.environ.get("SOLAR_WM_ROOT") or str(Path(__file__).resolve().parents[1])
sys.path.insert(0, WM)

from solar_wm_data import cos_io  # noqa: E402

RAW_PREFIX = os.environ.get("SOLAR_WM_RAW_PREFIX", "raw").strip("/")
SCRATCH = os.environ.get("SOLAR_WM_SCRATCH", "/tmp/solarwm")
EXPLODED = "sekai-game/exploded"


def _log(m: str) -> None:
    print(m, flush=True)


def _download(key: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    return Path(cos_io.get_file(key, str(dst)))


def explode(subset: str) -> None:
    root = Path(SCRATCH) / "explode_sekai" / subset
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    dl = root / "dl"
    dl.mkdir(parents=True, exist_ok=True)

    if subset == "drone":
        zp = _download(f"{RAW_PREFIX}/sekai-game/sekai-game-drone.zip", dl / "drone.zip")
    elif subset == "walking":
        pa = _download(f"{RAW_PREFIX}/sekai-game/sekai-game-walking.zip.part_aa", dl / "p_aa")
        pb = _download(f"{RAW_PREFIX}/sekai-game/sekai-game-walking.zip.part_ab", dl / "p_ab")
        zp = str(dl / "walking.zip")
        with open(zp, "wb") as out:
            for p in (pa, pb):
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out)
                os.remove(p)
    else:
        raise SystemExit(f"subset must be drone|walking, got {subset!r}")
    _log(f"[{subset}] downloaded zip, extracting (stdlib zipfile, Zip64-safe)...")

    ex = root / "ex"
    ex.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(ex)
    os.remove(zp)

    mp4s = sorted(ex.rglob("*.mp4"))
    _log(f"[{subset}] {len(mp4s)} clips extracted; uploading per-clip mp4+npz to S3...")
    up = skip = nonpair = 0
    for i, mp4 in enumerate(mp4s):
        npz = mp4.with_suffix(".npz")
        if not npz.exists():
            nonpair += 1
            continue
        cid = mp4.stem
        kmp4 = f"{RAW_PREFIX}/{EXPLODED}/{subset}/{cid}.mp4"
        knpz = f"{RAW_PREFIX}/{EXPLODED}/{subset}/{cid}.npz"
        if cos_io.exists(kmp4) and cos_io.exists(knpz):
            skip += 1
            continue
        cos_io.put_file(str(mp4), kmp4, skip_if_exists=True)
        cos_io.put_file(str(npz), knpz, skip_if_exists=True)
        up += 1
        if (i + 1) % 100 == 0:
            _log(f"[{subset}] {i + 1}/{len(mp4s)}  uploaded={up} skipped={skip}")
    shutil.rmtree(root, ignore_errors=True)
    _log(f"[{subset}] DONE  uploaded={up} skipped={skip} no-npz={nonpair} total={len(mp4s)}")


if __name__ == "__main__":
    subsets = sys.argv[1:] or [os.environ.get("SUBSET", "drone")]
    for s in subsets:
        explode(s)
