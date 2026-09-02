#!/usr/bin/env python3
"""Run Qwen2.5-VL captioning over an existing manifest and update each
packaged clip's ``prompt.txt``.

    SOLAR_WM_WEIGHTS=<weights> python3 scripts/caption_only.py <manifest.jsonl>
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solar_wm_data.manifest import read_manifest, write_manifest
from solar_wm_data.caption import caption_clip


def _log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    manifest = sys.argv[1]
    records = read_manifest(manifest)
    models_cfg = {"dry_run": False, "caption_nframes": 8}
    for r in records:
        _log(f"caption {r.clip_id} (Qwen2.5-VL) ...")
        r.caption = caption_clip(r, models_cfg)
        _log(f"  -> {r.caption!r}")
        pkg = r.extra.get("packaged_dir")
        if pkg:
            (Path(pkg) / "prompt.txt").write_text(r.caption or "", encoding="utf-8")
    write_manifest(manifest, records)
    _log("CAPTION_ONLY_DONE")


if __name__ == "__main__":
    main()
