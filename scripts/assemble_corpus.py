#!/usr/bin/env python3
"""Build a uniformly judged and deduplicated training list from stored manifests.

  1. Re-evaluate every clip with the selected filter configuration and camera gates,
     assigning each clip its ``kept_tier``. ``ASSEMBLY_OVERRIDES`` can evaluate a policy
     variant without changing the base configuration.
  2. DEDUP across sources by clip-stem collision. Source-level dedup already removed
     the known overlaps (RealCam-Vid is RealEstate10K-only); this catches residual
     collisions (e.g. two internet-T2V curations drawing from the same pool) and
     keeps the copy from the higher-priority source (GT-pose recipes first). A source
     whose clips are novel renders rather than copies of another source's video can be
     listed in DEDUP_EXEMPT.
  3. Emit <out>/train_list.jsonl (one record per kept clip: store path, source, tier,
     pose_mode, n_frames, metrics), <out>/meta/<owner>.jsonl (every judged clip with its
     kept_tier and reasons — the --meta-dir the recipe and window-view tools read), and
     <out>/assembly_report.json (per-source totals, reject histogram, dedup collisions,
     threshold provenance).

Run from any environment with access to the configured object store:
    python3 scripts/assemble_corpus.py --out /tmp/corpus_v1 [--sources zod,dl3dv]
        [--sample 200]      # per-source cap, for a quick dry-run audit
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

WM = os.environ.get("SOLAR_WM_ROOT") or str(Path(__file__).resolve().parents[1])
sys.path.insert(0, WM)

from solar_wm_data import cos_io  # noqa: E402
from solar_wm_data.config import load_config  # noqa: E402
from solar_wm_data.filter.thresholds import judge_clip  # noqa: E402
from solar_wm_data.manifest import QualityMetrics  # noqa: E402
from solar_wm_data.ingest import SOURCE_MODE  # noqa: E402
from solar_wm_data import spec as _spec  # noqa: E402

# The only clip lengths this corpus ships. Anything else is structurally off-spec.
VALID_LEN = frozenset(_spec.SPEC_FRAMES.values())

# Optional assembly-time overrides on top of configs/filters.yaml.
ASSEMBLY_OVERRIDES: dict[str, dict] = {}

# Assembly-time METRIC corrections — replace a stored measurement known to be wrong for a
# whole source, WITHOUT moving the threshold. The case this exists for: a detector that
# systematically misreads one source (a cut detector over-segmenting a fast continuous
# render, say) stores false values that an otherwise correct gate then acts on. Correcting
# the metric keeps the gate honest; loosening the gate would not. Empty by default.
METRIC_CORRECTIONS: dict[str, dict] = {}

# Dedup priority: on a stem collision keep the copy from the EARLIEST source here.
# GT trajectories (most exact recipe first) beat estimated ones. A source whose clips are
# novel renders rather than copies of another source's video belongs in DEDUP_EXEMPT.
DEDUP_PRIORITY = ["dl3dv", "omniworld", "sekai_game", "multicamvideo", "zod",
                  "realcam_vid", "spatialvid", "sekai_walking", "miradata",
                  "openvid", "vidgen", "ditto"]
DEDUP_EXEMPT: set[str] = set()

# Take every field the judge can read directly from the schema so new metrics cannot
# be silently omitted from assembly.
_METRIC_KEYS = tuple(f.name for f in dataclasses.fields(QualityMetrics))


def _metrics_ns(meta: dict, corrections: dict | None = None) -> SimpleNamespace:
    m = dict(meta.get("metrics") or {})
    if corrections:
        m.update(corrections)
    return SimpleNamespace(**{k: m.get(k) for k in _METRIC_KEYS})


def _outside(value, lo, hi) -> bool:
    """True when `value` is not inside [lo, hi] — INCLUDING when it is NaN.

    Written as a positive containment test rather than `value > hi`, because every
    comparison against NaN is False and the `>` form therefore lets a NaN through as if
    it had passed. That is the opposite of this corpus's policy, and invisible in the
    output: the clip simply appears kept.
    """
    return not (lo <= value <= hi)


def _camera_ok(meta: dict, cam_cfg: dict) -> tuple[bool, list[str]]:
    """The uniform camera gates, re-applied to a stored record.

    Per-frame gates use the stored extremes. Records that contain only medians are judged
    on those medians and report that limitation in the reason.
    """
    reasons = []
    cam = meta.get("camera") or {}
    lo, hi = cam_cfg["fov_deg"]
    per_frame = cam.get("fov_x_min") is not None

    for ax in ("fov_x", "fov_y"):
        if per_frame:
            vmin, vmax = cam.get(f"{ax}_min"), cam.get(f"{ax}_max")
            if vmin is None or vmax is None:
                reasons.append(f"{ax} extremes missing")
            elif _outside(vmin, lo, hi) or _outside(vmax, lo, hi):
                reasons.append(f"{ax} in [{vmin:.1f},{vmax:.1f}] outside {cam_cfg['fov_deg']}")
        else:
            v = cam.get(ax)
            if v is None:
                reasons.append(f"{ax}=None (not computed)")
            elif _outside(v, lo, hi):
                reasons.append(f"{ax}={v:.1f} outside {cam_cfg['fov_deg']} (median only)")

    fd = cam.get("focal_div_max") if per_frame else cam.get("focal_div")
    if fd is None:
        reasons.append("focal_div=None")
    elif _outside(fd, 0.0, cam_cfg["focal_div_max"]):
        suffix = "" if per_frame else " (median only)"
        reasons.append(f"focal_div={fd:.3f} > {cam_cfg['focal_div_max']}{suffix}")

    sf = meta.get("scale_factors") or []
    if not sf:
        reasons.append("scale_cov=inf (no recovered scales)")  # fail-closed
    else:
        import statistics as st
        mean = st.fmean(sf)
        cov = (st.pstdev(sf) / abs(mean)) if mean else float("inf")
        if _outside(cov, 0.0, cam_cfg["scale_cov_max"]):
            reasons.append(f"scale_cov={cov:.3f} > {cam_cfg['scale_cov_max']}")
    return (not reasons), reasons


def _judge(source: str, meta: dict, filters_cfg: dict) -> tuple[str | None, list[str]]:
    """Re-judge one stored clip. Returns ``(kept_tier, reasons)``.

    The tier — not just kept/rejected — is what the recipe, window-view and packing
    layers consume, so the assembler is where it is decided: it is the one place that
    sees the current thresholds and every stored clip at once.
    """
    ds = filters_cfg["datasets"].get(source)
    if ds is None:
        raise SystemExit(
            f"no thresholds for source '{source}' in this filters config. A source gets a "
            f"row calibrated from ITS OWN measured distribution — copying the nearest row "
            f"or running it ungated were both tried and both were wrong. Restrict the run "
            f"with --sources, use configs/filters_released.yaml for the released policy, or "
            f"calibrate this source with scripts/calibrate_filters.py first.")
    ds = dict(ds)
    ds.update(ASSEMBLY_OVERRIDES.get(source, {}))
    rec = SimpleNamespace(metrics=_metrics_ns(meta, METRIC_CORRECTIONS.get(source)),
                          source=source, num_frames=meta.get("num_frames"),
                          width=meta.get("width"), height=meta.get("height"),
                          caption=meta.get("caption"))
    tier, why_q = judge_clip(rec, ds)
    ok_c, why_c = _camera_ok(meta, {**filters_cfg["camera"], **(ds.get("camera") or {})})
    if tier is None or not ok_c:
        # A camera failure rejects outright: no xhigh extra relaxes a camera gate.
        return None, why_q + why_c
    return tier, []


# Files a clip must actually have on the store to be trainable. meta.json alone is NOT
# proof: a worker killed mid-clip leaves meta + intrinsics with no video and no poses, and
# such a clip reads as produced at every directory-level count.
REQUIRED_PAYLOAD = frozenset({"video.mp4", "poses.npy", "intrinsics.npy", "meta.json"})


# ---------------------------------------------------------------------------------
# Semantic stage. filters.yaml removes clips that are technically BROKEN; these rules
# judge whether a technically-fine clip is good TRAINING material, from the VLM pass
# (scripts/vlm_annotate.py) whose records live beside the corpus in vlm_anno/.
#
# Every rule is an audit by default: with --semantic off (the default) nothing is dropped
# and the report states how many clips each rule would drop per source. Users can then
# select an explicit semantic policy.
SEMANTIC_RULES = {
    # A cut inside a clip breaks the pose-to-frame correspondence a camera-control model
    # learns from: the trajectory is continuous, the pixels are not.
    "scene_transition": lambda v: ((v.get("scene_transition") or {}).get("count") or 0) > 0,
    "near_static":      lambda v: "near_static" in (v.get("reject_flags") or []),
    "blurry":           lambda v: "blurry" in (v.get("reject_flags") or []),
    "single_color":     lambda v: "single_color" in (v.get("reject_flags") or []),
    # Burned-in 2D overlays do not move with the camera, so they actively contradict the
    # geometry the model is being taught. Expensive to drop, though — see the measurements.
    "watermark":        lambda v: "watermark" in (v.get("reject_flags") or []),
    "ui_overlay":       lambda v: "ui_overlay" in (v.get("reject_flags") or []),
    "text_heavy":       lambda v: "text_heavy" in (v.get("reject_flags") or []),
    # NOT in any preset: night footage is a legitimate part of the distribution (16% of
    # sekai_game is night by nature), and dropping it biases the corpus's lighting.
    "low_light":        lambda v: "low_light" in (v.get("reject_flags") or []),
    "quality_lt3":      lambda v: (v.get("vlm_quality") or 0) < 3,
    "quality_lt4":      lambda v: (v.get("vlm_quality") or 0) < 4,
}

SEMANTIC_PRESETS = {
    "off": (),
    "wide": ("scene_transition", "near_static", "blurry", "single_color", "quality_lt3"),
    "elite": ("scene_transition", "near_static", "blurry", "single_color", "quality_lt4",
              "watermark", "ui_overlay", "text_heavy"),
}


SEMANTIC_KEYS = frozenset({"quality_min", "quality_max", "transition_max", "flags_allowed",
                           "flags_forbidden", "caption_required", "path_min_m",
                           "path_max_m"})


def _semantic_config_reject(vlm: dict | None, sem: dict | None,
                            traj_path: float | None = None) -> list[str]:
    """Apply a dataset's `semantic:` block from the filters config. Returns reject reasons.

    The policy is PER SOURCE and lives in the config, not in code: real policies are not
    uniform (a corpus can reasonably tolerate a watermark on walking footage while refusing
    one on a re-annotated real-estate clip), and every such choice belongs in a yaml the
    corpus owner edits rather than hardcoded here. Keys, all optional:
      quality_min / quality_max   vlm_quality bounds
      transition_max              max allowed scene_transition count
      flags_allowed               ONLY these flags may appear ([] = none may)
      flags_forbidden             none of these may appear
      caption_required            reject an empty caption
    A clip with no annotation record fails any block that is present: an unmeasured clip is
    not a clip that passed (a null metric is a missing measurement, not a verdict).
    """
    if not sem:
        return []
    # A key nobody reads is a gate that silently does nothing — the failure mode where a
    # policy looks applied and is not. Fail loudly instead.
    unknown = set(sem) - SEMANTIC_KEYS
    if unknown:
        raise KeyError(f"unknown semantic keys {sorted(unknown)}; known: {sorted(SEMANTIC_KEYS)}")
    out = []
    if sem.get("path_min_m") is not None or sem.get("path_max_m") is not None:
        # Camera motion, the axis no pixel metric covers. Missing = unmeasured, and an
        # unmeasured clip has not passed.
        if traj_path is None:
            out.append("path_unmeasured")
        else:
            if sem.get("path_min_m") is not None and traj_path < sem["path_min_m"]:
                out.append("path_min")          # camera never moved: no signal to learn
            if sem.get("path_max_m") is not None and traj_path > sem["path_max_m"]:
                # Scale divergence, not fast motion: measured maxima run to 1.27e9 metres
                # in 5 s. scale_cov cannot see it — a scale that is uniformly wrong by six
                # orders of magnitude is perfectly self-consistent.
                out.append("path_max")
    if vlm is None:
        return out + ["vlm_missing"]
    q = vlm.get("vlm_quality")
    if sem.get("quality_min") is not None and (q is None or q < sem["quality_min"]):
        out.append("q_min")
    if sem.get("quality_max") is not None and (q is None or q > sem["quality_max"]):
        out.append("q_max")
    if sem.get("transition_max") is not None:
        if ((vlm.get("scene_transition") or {}).get("count") or 0) > sem["transition_max"]:
            out.append("transition")
    flags = set(vlm.get("reject_flags") or [])
    if sem.get("flags_allowed") is not None and (flags - set(sem["flags_allowed"])):
        out.append("flags")
    if sem.get("flags_forbidden") and (flags & set(sem["flags_forbidden"])):
        out.append("flags")
    if sem.get("caption_required") and not (vlm.get("dense_caption") or "").strip():
        out.append("caption_empty")
    return out


def _vlm_root(source: str) -> Path:
    """Where the annotation pass writes: <local root>/vlm_anno/<source>-<run id>/."""
    run = os.environ.get("SOLAR_WM_RUN_ID", "").strip()
    root = Path(os.environ.get("SOLAR_WM_LOCAL_ROOT", "")) / "vlm_anno"
    return root / (f"{source}-{run}" if run else source)


def _vlm_load(cid: str, vroot: Path) -> dict | None:
    try:
        return json.loads((vroot / f"{cid}.json").read_text()).get("response") or {}
    except Exception:  # noqa: BLE001 - not annotated yet / unreadable; counted, not fatal
        return None


def _clip_ids(source: str, sample: int) -> tuple[list[str], dict[str, set[str]]]:
    """Complete clip enumeration from clips/ listing — the authoritative produced set.
    (Per-item manifests are NOT complete: written per item and overwritten each run, so an
    item processed across resumes keeps only the LAST run's clips; the rest stay in clips/
    but vanish from the manifest. Enumerate from clips/, judge from each clip's meta.json.)

    Also returns clip id -> the file names present for it. The listing already walks every
    key under clips/, so recording which payloads exist costs nothing beyond the dict."""
    pre = f"{cos_io.corpus_prefix(source)}/clips/"
    seen, ids = set(), []
    have: dict[str, set[str]] = {}
    for k in cos_io.list_keys(pre):
        rest = k[len(pre):]
        cid, _, name = rest.partition("/")
        if not cid:
            continue
        if cid not in seen:
            # Break on the FIRST key of the (sample+1)-th clip, so every clip we keep has
            # had all of its own keys walked — otherwise a sampled run would report the
            # last clip as missing payloads it simply never listed.
            if sample and len(ids) >= sample:
                break
            seen.add(cid)
            ids.append(cid)
        if name:
            have.setdefault(cid, set()).add(name)
    return ids, have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default=",".join(sorted(SOURCE_MODE)))
    ap.add_argument("--sample", type=int, default=0, help="per-source clip cap (audit runs)")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--semantic", default="off",
                    choices=sorted(SEMANTIC_PRESETS) + ["config"],
                    help="semantic policy to APPLY: a preset, or 'config' to use each "
                         "dataset's `semantic:` block. Every rule is audited regardless")
    ap.add_argument("--traj", default="",
                    help="dir of scripts/traj_stats.py output; enables the path_min_m gate")
    ap.add_argument("--filters", default="filters",
                    help="filters config to judge with (name under configs/ or a path) — "
                         "use an alternative file to evaluate a different policy")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    filters_cfg = load_config(a.filters)
    sources = [s.strip() for s in a.sources.split(",") if s.strip()]

    # Per-source assembly exclusions (configs/exclusions/<source>.txt, one clip id per
    # line). These source-policy exclusions are omitted from the training list:
    #   realcam_vid    723 non-RealEstate10K-subset clips (DL3DV-10K/MiraData9K subsets
    #                  overlap scenes covered by dl3dv/miradata; stem dedup
    #                  can't catch them — ids don't textually match)
    #   sekai_walking  4,782 non-Walking-HQ clips (full uncurated real-walking extracted
    #                  before the HQ-csv filter landed; paper trains on Walking-HQ only)
    excl_dir = Path(__file__).resolve().parents[1] / "configs" / "exclusions"

    stem_owner: dict = {}            # stem -> (priority, source, cid)
    rows, report = [], {}
    # Per-owner judged metadata, written to <out>/meta/<owner>.jsonl. This is the input
    # the recipe, window-view and released-policy tools read: they select on `kept_tier`,
    # and the assembler is the only stage that knows it, because it is the only stage that
    # applies the CURRENT thresholds to every stored clip. Rejected clips are recorded too
    # — with their reasons — so a later policy change can be priced from this file alone.
    meta_out: dict[str, list[dict]] = {}
    pool = ThreadPoolExecutor(max_workers=a.threads)
    for src in sources:
        meta_rows: list[dict] = meta_out.setdefault(src, [])
        ids, have = _clip_ids(src, a.sample)
        # Drop clips whose payload is incomplete BEFORE judging. The judge reads meta.json,
        # which such a clip still has, so without this it would pass every quality gate and
        # enter train_list pointing at a video that does not exist — a defect that surfaces
        # only when training reads it. Seen for real: one abot clip out of 235,347 left with
        # meta + intrinsics and no video/poses after a worker died mid-upload.
        broken = [c for c in ids if not REQUIRED_PAYLOAD <= have.get(c, set())]
        if broken:
            miss = {c: sorted(REQUIRED_PAYLOAD - have.get(c, set())) for c in broken[:5]}
            print(f"{src}: DROPPED {len(broken)} clips with incomplete payload, e.g. {miss}",
                  flush=True)
            ids = [c for c in ids if c not in set(broken)]
        excl_path = excl_dir / f"{src}.txt"
        if excl_path.exists():
            excl = set(excl_path.read_text().split())
            before = len(ids)
            ids = [c for c in ids if c not in excl]
            print(f"{src}: excluded {before - len(ids)} clips via configs/exclusions/{src}.txt",
                  flush=True)
        pre = f"{cos_io.corpus_prefix(src)}/clips/"

        vroot = _vlm_root(src)

        def fetch(cid, _pre=pre, _vroot=vroot):
            # in-memory meta read (no /tmp spill, no skip_if_exists HEAD) — see cos_io.get_bytes
            # The annotation record is read in the SAME pass: it lives outside the corpus
            # prefix, and a second walk over half a million clips is the expensive part.
            try:
                meta = json.loads(cos_io.get_bytes(f"{_pre}{cid}/meta.json"))
            except Exception:  # noqa: BLE001 - clip mid-upload etc.; counted, not fatal
                return cid, None, None
            return cid, meta, _vlm_load(cid, _vroot)

        kept = 0
        rej_hist: dict = {}
        sem_hist: dict = {k: 0 for k in SEMANTIC_RULES}
        n_err = n_offspec = n_novlm = n_sem_dropped = 0
        applied = SEMANTIC_PRESETS.get(a.semantic, ())
        sem_cfg = (filters_cfg["datasets"].get(src) or {}).get("semantic") \
            if a.semantic == "config" else None
        # traj_of[cid] = path in metres; cid present with value None = measured and
        # UNUSABLE (non-finite poses). Absent = never measured.
        traj_of: dict[str, float | None] = {}
        traj_full: dict[str, dict] = {}
        if a.traj and (Path(a.traj) / f"{src}.jsonl").exists():
            for line in open(Path(a.traj) / f"{src}.jsonl"):
                r = json.loads(line)
                ok = r.get("ok", True)
                traj_of[r["clip_id"]] = r.get("path") if ok else None
                if ok:
                    traj_full[r["clip_id"]] = r
        for cid, meta, vlm in pool.map(fetch, ids):
            if meta is None:
                n_err += 1
                continue
            # Off-spec LENGTH is a structural defect, rejected for the same reason as a
            # missing payload: no quality metric can see it. The acquire trims a source to
            # the spec and lets a source shorter than one window come out short, on the
            # assumption that such a clip "fails the length gate downstream" — but no such
            # gate existed, so 26% of spatialvid (50-120 frames against a 121-frame
            # contract) would have entered the training list unchallenged. The corpus
            # ships exactly the lengths in spec.SPEC_FRAMES and nothing else.
            # An unknown length is not a valid one. `nf is not None and ...` let a record
            # with no num_frames through the gate whose whole purpose is to keep off-spec
            # clips out of the training list.
            nf = meta.get("num_frames")
            if nf not in VALID_LEN:
                n_offspec += 1
                continue
            tier, why = _judge(src, meta, filters_cfg)
            meta_rows.append({"clip_id": cid, "source": src, "kept_tier": tier,
                              "num_frames": meta.get("num_frames"),
                              "reject_reasons": why})
            if tier is None:
                for w in why:
                    rej_hist[w.split("=")[0]] = rej_hist.get(w.split("=")[0], 0) + 1
                continue
            # Semantic audit runs on clips that PASSED the technical gate, so each count is
            # "how much would this rule cost on top of what is already rejected" — the only
            # form in which the numbers are decision-grade.
            if vlm is None:
                n_novlm += 1
                hits = ()
            else:
                hits = tuple(k for k, f in SEMANTIC_RULES.items() if f(vlm))
                for k in hits:
                    sem_hist[k] += 1
            if applied and any(k in applied for k in hits):
                n_sem_dropped += 1
                continue
            # Poses that are not finite are a defect no pixel metric can see: the file has
            # the right shape and size, so payload checks and directory counts all pass, and
            # only training would ever read the NaNs. Reject whenever the trajectory pass
            # measured this clip and found it unusable — independent of any semantic policy.
            if a.traj and cid in traj_of and traj_of[cid] is None:
                rej_hist["poses_not_finite"] = rej_hist.get("poses_not_finite", 0) + 1
                continue
            if sem_cfg is not None:
                why_s = _semantic_config_reject(vlm, sem_cfg, traj_of.get(cid))
                if why_s:
                    n_sem_dropped += 1
                    for w in why_s:
                        rej_hist["sem_" + w] = rej_hist.get("sem_" + w, 0) + 1
                    continue
            rows.append({"source": src, "clip_id": cid, "kept_tier": tier,
                         "store_path": f"{pre}{cid}/",
                         "pose_mode": meta.get("pose_mode"), "n_frames": meta.get("num_frames"),
                         "metrics": meta.get("metrics"),
                         # Length of the caption that ships WITH the clip. prompt.txt in the
                         # store already holds the annotated caption (verified identical on 900
                         # sampled clips across three sources), and that file is what the
                         # packer reads — meta's own `caption` predates the annotation pass.
                         "caption_len": len((vlm or {}).get("dense_caption") or ""),
                         "vlm": None if vlm is None else {
                             "quality": vlm.get("vlm_quality"),
                             "flags": vlm.get("reject_flags") or [],
                             "cuts": (vlm.get("scene_transition") or {}).get("count") or 0,
                             "scene_type": vlm.get("scene_type")},
                         # Camera motion travels WITH the row so the training side can weight
                         # or bucket by it (metres; tort ~1 = straight) without re-reading
                         # half a million pose files.
                         "traj": None if traj_full.get(cid) is None else {
                             k: round(traj_full[cid][k], 4)
                             for k in ("path", "disp", "tort", "rot_deg")
                             if traj_full[cid].get(k) is not None}})
            kept += 1
        report[src] = {"clips_seen": len(ids), "kept": kept, "meta_errors": n_err,
                       "incomplete_payload": len(broken), "incomplete_ids": broken[:50],
                       "offspec_length": n_offspec,
                       "reject_histogram": dict(sorted(rej_hist.items(), key=lambda x: -x[1])),
                       "semantic_preset": a.semantic,
                       "semantic_dropped": n_sem_dropped,
                       "semantic_unannotated": n_novlm,
                       "semantic_audit": dict(sorted(sem_hist.items(), key=lambda x: -x[1]))}
        top_sem = {k: v for k, v in report[src]["semantic_audit"].items() if v}
        print(f"[{time.strftime('%H:%M:%S')}] {src}: {kept}/{len(ids)} kept "
              f"(err {n_err}) {report[src]['reject_histogram']}"
              + (f" | semantic would drop {top_sem}" if top_sem else "")
              + (f" | {n_novlm} unannotated" if n_novlm else ""), flush=True)

    # cross-source stem dedup (priority wins)
    prio = {s: i for i, s in enumerate(DEDUP_PRIORITY)}
    collisions = []
    final = []
    for r in sorted(rows, key=lambda r: prio.get(r["source"], 99)):
        if r["source"] in DEDUP_EXEMPT:
            final.append(r)
            continue
        stem = r["clip_id"]
        if stem in stem_owner:
            collisions.append({"stem": stem, "kept": stem_owner[stem], "dropped": r["source"]})
            continue
        stem_owner[stem] = r["source"]
        final.append(r)

    meta_dir = out / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for src, mrows in meta_out.items():
        with open(meta_dir / f"{src}.jsonl", "w") as f:
            for r in mrows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out / "train_list.jsonl", "w") as f:
        for r in final:
            f.write(json.dumps(r) + "\n")
    json.dump({"sources": report, "total_kept": len(final),
               "dedup_collisions": len(collisions), "collision_samples": collisions[:50],
               "assembly_overrides": ASSEMBLY_OVERRIDES,
               "filters_yaml_snapshot": filters_cfg["datasets"]},
              open(out / "assembly_report.json", "w"), indent=1)
    n_meta = sum(len(v) for v in meta_out.values())
    print(f"DONE: {len(final)} clips -> {out}/train_list.jsonl "
          f"({len(collisions)} dedup collisions; report in assembly_report.json)")
    print(f"      {n_meta} judged clips -> {meta_dir}/<owner>.jsonl "
          f"(feed this directory to build_recipe.py / build_window_view.py as --meta-dir)")


if __name__ == "__main__":
    main()
