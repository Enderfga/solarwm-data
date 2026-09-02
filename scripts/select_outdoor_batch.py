#!/usr/bin/env python3
"""Cut an OUTDOOR + REAL batch out of an assembled train list.

Four filters, applied in order, each one reported so the cost of each is visible:

  1. TECHNICAL — already applied by assemble_corpus.py, whose train_list.jsonl is this
     script's input.
  2. REAL — `vlm.scene_type == "real_world"`, from the per-clip annotation.
  3. OUTDOOR — scripts/outdoor_rule.py, read from the caption.
  4. CAMERA MOTION — the axis no pixel metric covers, from scripts/traj_stats.py output.
     Judged on NET DISPLACEMENT, not path length: path is a sum of per-frame steps, so
     handheld jitter can inflate it even when scale is valid.

Optional semantic gates (scene transitions, watermark, ui_overlay, VLM quality) are PRICED
and NOT applied — which of them binds is a policy call, and the point of printing the
table is to make that call on numbers rather than on taste.

    select_outdoor_batch.py --list <train_list.jsonl> --anno <vlm_anno_root>
                            --traj <traj_stats_dir> --out <batch.jsonl> [--run-id 60s]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from outdoor_rule import classify  # noqa: E402

# A 60 s clip whose camera really travelled 2 km moved at 120 km/h; this corpus has no
# such footage, so that is a scale that ran away rather than a fast vehicle. PATH_MAX is
# the same judgement one order of magnitude looser, to catch runaways whose net
# displacement happens to stay small.
DISP_MAX, PATH_MAX, PATH_MIN = 2000.0, 5000.0, 1.0


def quant(vals, p):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(len(vals) * p))] if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="assemble_corpus.py train_list.jsonl")
    ap.add_argument("--anno", required=True, help="dir holding <source>-<run_id>/<clip>.json")
    ap.add_argument("--traj", default="", help="dir of traj_stats.py output (optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default="60s")
    ap.add_argument("--threads", type=int, default=48)
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.list).read_text().splitlines() if l.strip()]
    by_src = Counter(r["source"] for r in rows)
    print(f"technical pass (assembled): {len(rows)}")

    real = [r for r in rows if (r.get("vlm") or {}).get("scene_type") == "real_world"]
    print(f"scene_type == real_world:   {len(real)}")
    for s in sorted(by_src):
        n = sum(1 for r in real if r["source"] == s)
        print(f"    {s:<16} {n:>7} / {by_src[s]:<7} ({100 * n / by_src[s]:.0f}%)")

    anno_root = Path(a.anno)

    def load(r):
        p = anno_root / f"{r['source']}-{a.run_id}" / f"{r['clip_id']}.json"
        try:
            return r, json.loads(p.read_text())["response"]
        except Exception:
            return r, None

    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        loaded = list(ex.map(load, real))

    verdict, gates, kept = Counter(), defaultdict(int), []
    for r, ann in loaded:
        if ann is None:
            verdict["no_annotation"] += 1
            continue
        v = classify(ann.get("dense_caption", ""))
        verdict[v] += 1
        if v != "outdoor":
            continue
        kept.append(r)
        st = ann.get("scene_transition") or {}
        if (st.get("count") or 0) > 0:
            gates["scene_transition>0"] += 1
        for f in (ann.get("reject_flags") or []):
            gates[f"flag:{f}"] += 1
        if float(ann.get("vlm_quality") or 0) < 4:
            gates["vlm_quality<4"] += 1
    print(f"\ncaption verdict: {dict(verdict)}")
    print(f"outdoor + real:  {len(kept)}")

    if a.traj:
        traj = {}
        for f in Path(a.traj).glob("*.jsonl"):
            for line in f.read_text().splitlines():
                t = json.loads(line)
                if t.get("ok"):
                    traj[(f.stem, t["clip_id"])] = t
        joined, no_traj = [], 0
        for r in kept:
            t = traj.get((r["source"], r["clip_id"]))
            if t is None:
                no_traj += 1
                continue
            joined.append(dict(r, traj={k: t[k] for k in
                                        ("path", "disp", "diag", "tort", "straight",
                                         "rot_deg", "vel_med", "vel_p95") if k in t}))
        print(f"\ntrajectory joined: {len(joined)} (no record: {no_traj})")
        for k in ("path", "disp", "tort"):
            v = [r["traj"][k] for r in joined if r["traj"].get(k) is not None]
            print(f"  {k:<6} p50={quant(v, .5):9.2f} p95={quant(v, .95):9.2f} "
                  f"p99={quant(v, .99):9.2f} max={max(v) if v else float('nan'):12.2f}")
        motion, final = Counter(), []
        for r in joined:
            t = r["traj"]
            why = []
            if t.get("path", 0) < PATH_MIN:
                why.append("static")
            if t.get("disp", 0) > DISP_MAX or t.get("path", 0) > PATH_MAX:
                why.append("scale_diverged")
            motion["+".join(why) or "ok"] += 1
            if not why:
                final.append(r)
        print(f"motion verdict: {dict(motion)}")
        kept = final

    # Best first. Whatever consumes this list may be stopped early — by a disk budget, by
    # a deadline — and an early stop should leave the best clips produced, not an arbitrary
    # prefix. Flags first (a watermark or an overlay is a defect, not a matter of degree),
    # then the VLM's quality score, then the annotator's own transition count.
    def rank(r):
        v = r.get("vlm") or {}
        return (len(v.get("flags") or []), -(v.get("quality") or 0), v.get("cuts") or 0)

    kept.sort(key=rank)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in kept))
    print(f"\nFINAL: {len(kept)} -> {a.out}")
    for s, n in Counter(r["source"] for r in kept).most_common():
        print(f"    {s:<16} {n:>7}")
    print("\noptional semantic gates, priced against the outdoor set (NOT applied):")
    for k, v in sorted(gates.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<24} would drop {v:>7} ({100 * v / max(len(kept), 1):.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
