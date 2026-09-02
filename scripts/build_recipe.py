#!/usr/bin/env python3
"""Build a logical recipe — train / test membership, tier policy and source balancing.

A recipe REFERENCES rows; it never copies them. Changing a mixture therefore costs a small
index, not a re-pack of the corpus, and the same physical clip can appear in several
recipes without being duplicated on disk.

    python3 scripts/build_recipe.py --meta-dir <dir of <owner>.jsonl> --out recipes/short
    ... --tier xhigh,high --test-per-owner 100 --repeat abot=6,miradata=6,sekai_game=6
    ... --owners abot,miradata,sekai_game,...        # a recipe may span fewer owners

Outputs
    <out>/train.jsonl     one line per PHYSICAL training row (never repeated on disk)
    <out>/test.jsonl      the held-out view
    <out>/report.json     counts, virtual occurrences, and the overlap check

SPLIT BY STABLE ID, NOT BY SHUFFLE. Test membership is decided by a hash of the sample id
alone, so it depends on neither file order nor a seed anyone has to remember: the same
inputs always produce the same split.

    Membership is NOT preserved when an owner grows. The rule takes the N lowest-ranked
    ids, so a clip added later that ranks lower evicts one that was held out before. That
    is the price of an exact per-owner count. If you need membership frozen across corpus
    growth, hold the split file rather than rebuilding it, or switch to a hash-threshold
    rule (every id below a cut) and accept a count that drifts.

    A rebuilt recipe follows the rules selected on this command and need not be
    bit-identical to a released recipe. Use the released index for exact release
    membership.

Repeat factors are recorded, never materialised: `train.jsonl` holds each physical row
once with its `repeat`, and the report states the virtual per-epoch total. Writing a row
six times would inflate the corpus on disk and make a later dedup pass look like data loss.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solar_wm_data.split import split_rank      # noqa: E402  the one split rule

TIERS = ("xhigh", "high")


def load_owner(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tier", default="xhigh,high",
                    help="comma-separated tiers eligible for the recipe")
    ap.add_argument("--owners", default="",
                    help="comma-separated owners to include; default every *.jsonl found. "
                         "A recipe is allowed to span fewer owners than the corpus")
    ap.add_argument("--test-per-owner", type=int, default=100)
    ap.add_argument("--repeat", default="",
                    help="owner=factor,... source balancing; unlisted owners repeat once")
    a = ap.parse_args()

    tiers = {t.strip() for t in a.tier.split(",") if t.strip()}
    unknown = tiers - set(TIERS)
    if unknown:
        raise SystemExit(f"unknown tier(s) {sorted(unknown)}; valid: {list(TIERS)}")
    repeats = {}
    for part in filter(None, (p.strip() for p in a.repeat.split(","))):
        owner, _, factor = part.partition("=")
        repeats[owner] = int(factor)
    wanted = {o.strip() for o in a.owners.split(",") if o.strip()}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    report: dict = {"tiers": sorted(tiers), "repeat": repeats, "owners": {}}

    files = sorted(Path(a.meta_dir).glob("*.jsonl"))
    if wanted:
        # Named but absent is an error, not an empty contribution: a recipe that silently
        # drops an owner because of a typo still builds, still reports a plausible total,
        # and trains on a mixture nobody chose.
        missing = wanted - {f.stem for f in files}
        if missing:
            raise SystemExit(f"--owners names {sorted(missing)}, not present in "
                             f"{a.meta_dir}")
        files = [f for f in files if f.stem in wanted]

    for f in files:
        owner = f.stem
        eligible = [r for r in load_owner(f) if r.get("kept_tier") in tiers]
        # Deterministic: rank by id hash, the lowest N are the test view. Ties are
        # impossible in practice and would be broken by the id itself.
        eligible.sort(key=lambda r: (split_rank(str(r.get("clip_id", ""))),
                                     str(r.get("clip_id", ""))))
        n_test = min(a.test_per_owner, len(eligible))
        rep = repeats.get(owner, 1)
        for i, r in enumerate(eligible):
            row = {"owner": owner, "clip_id": r.get("clip_id"),
                   "kept_tier": r.get("kept_tier")}
            if i < n_test:
                test_rows.append(row)
            else:
                train_rows.append({**row, "repeat": rep})
        report["owners"][owner] = {
            "eligible": len(eligible), "test": n_test,
            "train": len(eligible) - n_test, "repeat": rep,
            "virtual": (len(eligible) - n_test) * rep,
        }

    # The check that matters. A recipe that leaks its test set is not detectably wrong from
    # any training curve, so it is asserted here rather than trusted from the construction.
    stray = set(repeats) - set(report["owners"])
    if stray:
        raise SystemExit(f"--repeat names {sorted(stray)}, which contribute no rows to "
                         f"this recipe; the balancing would silently do nothing")

    train_ids = {(r["owner"], r["clip_id"]) for r in train_rows}
    test_ids = {(r["owner"], r["clip_id"]) for r in test_rows}
    overlap = train_ids & test_ids
    if overlap:
        raise SystemExit(f"train/test overlap on {len(overlap)} ids, e.g. {sorted(overlap)[:3]}")
    if len(train_ids) != len(train_rows) or len(test_ids) != len(test_rows):
        raise SystemExit("duplicate (owner, clip_id) within a split")

    (out / "train.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train_rows), encoding="utf-8")
    (out / "test.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in test_rows), encoding="utf-8")

    report["physical_train_rows"] = len(train_rows)
    report["virtual_train_occurrences"] = sum(r["repeat"] for r in train_rows)
    report["test_rows"] = len(test_rows)
    report["train_test_overlap"] = 0
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"  {len(report['owners'])} owners")
    print(f"  train {report['physical_train_rows']} physical rows "
          f"-> {report['virtual_train_occurrences']} virtual occurrences per epoch")
    print(f"  test  {report['test_rows']} rows, zero overlap")
    by_rep = collections.Counter(r["repeat"] for r in train_rows)
    if len(by_rep) > 1:
        print(f"  repeat factors in use: {dict(sorted(by_rep.items()))}")
    print(f"  {out}/train.jsonl  {out}/test.jsonl  {out}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
