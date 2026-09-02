"""Per-dataset selection policy: judge a clip into ``xhigh`` / ``high`` / rejected.

A policy is TWO rules per dataset owner. The **kept rule** decides whether the clip is
usable at all; the **xhigh extra** is a stricter rule applied *in addition to* it, and a
clip satisfying both is promoted. Everything else is retained as rejected, with its
reasons, so a threshold change is a re-judgement rather than a reprocessing run.

    kept rule satisfied + xhigh extra satisfied  ->  "xhigh"
    kept rule satisfied                          ->  "high"
    otherwise                                    ->  None  (rejected, with reasons)

The three labels are disjoint and `None` is what a rejected clip stores in ``kept_tier``.

FAIL-CLOSED IS THE WHOLE POINT. A gate that is configured but whose metric is missing
rejects. An unmeasured clip has not passed; it is simply unjudged, and treating "we did
not measure" as "it is fine" is how a broken metric adapter silently promotes garbage.
A gate that is *not configured* (``null``) is skipped, which is a different thing.

Config shape — one block per dataset in ``configs/filters.yaml``::

    miradata:
      unimatch:  [3, 120]          # [min, max], inclusive
      dover:     [0.40, 1.0]       # technical+aesthetic averaged into [0,1]
      scene_cuts: 0                # max-only (<= N)
      min_frames: 81               # N >= 81
      resolution: [1280, 720]      # exact WxH
      require_caption: true        # non-empty caption
      vlm_reject_flags: []         # R must be a subset of this list ([] = must be empty)
      vlm_transition_max: 0        # scene_transition.count <= N
      vlm_quality: [4.0, 5.0]
      camera: {fov_deg: [25, 125]} # per-source override of the global camera block
      xhigh:                       # applied ON TOP of everything above
        vlm_quality: [5.0, 5.0]
        unimatch:  [3, 100]

The ``xhigh`` block is a partial override merged over the kept rule, so it can only
narrow. Omit it and nothing is promoted: every kept clip is ``high``.
"""

from __future__ import annotations

from typing import Any

from ..manifest import ClipRecord

#: Keys that are gates. Anything else in a dataset block (``xhigh``, ``camera``) is
#: structure, not a gate.
GATE_KEYS = frozenset({
    "vmaf", "unimatch", "dover", "color_sat", "scene_cuts",
    "min_frames", "resolution", "require_caption",
    "vlm_entity_density", "vlm_quality", "vlm_reject_flags",
    "vlm_reject_flags_forbidden", "vlm_transition_max",
})

#: Keys that are legal in a dataset block but are not gates.
STRUCTURE_KEYS = frozenset({"xhigh", "camera"})


def validate_policy(cfg: dict, where: str = "policy") -> None:
    """Raise if a dataset block names a key that is neither a gate nor structure.

    A misspelled gate is the worst kind of config bug: `min_frame` instead of
    `min_frames` reads fine, loads fine, and is simply never applied, so the corpus is
    selected by a policy that differs from the one written down and nothing says so.
    Unknown keys are therefore an error, not a warning — the same fail-closed reasoning
    that rejects a clip whose gated metric is missing.
    """
    for scope, block in (("", cfg), ("xhigh.", cfg.get("xhigh") or {})):
        unknown = sorted(set(block) - GATE_KEYS - STRUCTURE_KEYS)
        if unknown:
            raise ValueError(
                f"{where}: unknown key(s) {[scope + u for u in unknown]}. Valid gates: "
                f"{sorted(GATE_KEYS)}; structure: {sorted(STRUCTURE_KEYS)}. A misspelled "
                f"gate is silently never applied, so this is an error.")


def _in_range(value: float | None, rng: Any) -> bool:
    if rng is None:  # gate not configured for this dataset -> skip (correctly)
        return True
    if value is None:  # gate IS configured but the metric is missing -> FAIL-CLOSED.
        return False
    lo, hi = rng
    return lo <= value <= hi


def _kept_reasons(rec: ClipRecord, cfg: dict) -> list[str]:
    """Every reason this clip fails ``cfg``. Empty list means it satisfies the rule.

    Reasons are appended in a fixed order — cheap structural gates first, then measured
    metrics, then VLM annotation — so the ORDER carries which class of problem fired
    first and is stable across runs.
    """
    m = rec.metrics
    reasons: list[str] = []

    # --- structural: knowable without any model ------------------------------------
    min_frames = cfg.get("min_frames")
    if min_frames is not None:
        if rec.num_frames is None:
            reasons.append("num_frames=None (unknown)")
        elif rec.num_frames < min_frames:
            reasons.append(f"num_frames={rec.num_frames} < {min_frames}")

    res = cfg.get("resolution")
    if res is not None:
        if rec.width is None or rec.height is None:
            reasons.append("resolution=None (unknown)")
        elif [rec.width, rec.height] != list(res):
            reasons.append(f"resolution={rec.width}x{rec.height} != {res[0]}x{res[1]}")

    if cfg.get("require_caption") and not (rec.caption or "").strip():
        reasons.append("caption empty")

    # --- measured metrics ----------------------------------------------------------
    if not _in_range(m.vmaf, cfg.get("vmaf")):
        reasons.append(f"vmaf={m.vmaf} outside {cfg['vmaf']}")
    if not _in_range(m.unimatch, cfg.get("unimatch")):
        reasons.append(f"unimatch={m.unimatch} outside {cfg['unimatch']}")

    # DOVER is reported as technical+aesthetic; the gate is on their mean.
    dover_rng = cfg.get("dover")
    if dover_rng is not None:
        parts = [v for v in (m.dover_tech, m.dover_aes) if v is not None]
        if not parts:
            reasons.append("dover=None (not computed)")
        else:
            dover = sum(parts) / len(parts)
            if not (dover_rng[0] <= dover <= dover_rng[1]):
                reasons.append(f"dover={dover:.3f} outside {dover_rng}")

    if not _in_range(m.saturation, cfg.get("color_sat")):
        reasons.append(f"color_sat={m.saturation} outside {cfg['color_sat']}")

    # scene_cuts is a max-only bound (<= N).
    sc_max = cfg.get("scene_cuts")
    if sc_max is not None:
        if m.scene_cuts is None:
            reasons.append("scene_cuts=None (not computed)")
        elif m.scene_cuts > sc_max:
            reasons.append(f"scene_cuts={m.scene_cuts} > {sc_max}")

    # --- VLM annotation ------------------------------------------------------------
    # Produced by the annotation pass and merged in, not computed by the filter stage,
    # so on a freshly produced clip these are None and any configured gate rejects.
    if not _in_range(m.vlm_entity_density, cfg.get("vlm_entity_density")):
        reasons.append(
            f"vlm_entity_density={m.vlm_entity_density} outside {cfg['vlm_entity_density']}")
    if not _in_range(m.vlm_quality, cfg.get("vlm_quality")):
        reasons.append(f"vlm_quality={m.vlm_quality} outside {cfg['vlm_quality']}")

    allowed = cfg.get("vlm_reject_flags")
    if allowed is not None:
        if m.vlm_reject_flags is None:
            reasons.append("vlm_reject_flags=None (not annotated)")
        else:
            extra = sorted(set(m.vlm_reject_flags) - set(allowed))
            if extra:
                reasons.append(f"vlm_reject_flags {extra} not in allowed {list(allowed)}")

    forbidden = cfg.get("vlm_reject_flags_forbidden")
    if forbidden is not None:
        if m.vlm_reject_flags is None:
            reasons.append("vlm_reject_flags=None (not annotated)")
        else:
            hit = sorted(set(m.vlm_reject_flags) & set(forbidden))
            if hit:
                reasons.append(f"vlm_reject_flags {hit} forbidden")

    tmax = cfg.get("vlm_transition_max")
    if tmax is not None:
        tr = m.vlm_scene_transition
        count = tr.get("count") if isinstance(tr, dict) else None
        if count is None:
            reasons.append("vlm_scene_transition=None (not annotated)")
        elif count > tmax:
            reasons.append(f"vlm_scene_transition.count={count} > {tmax}")

    return reasons


def xhigh_cfg(cfg: dict) -> dict:
    """The xhigh rule: the kept rule with the ``xhigh`` block merged over it.

    Merging rather than replacing is what makes the extra *additional* — an xhigh block
    naming only ``vlm_quality`` still has to satisfy every gate the kept rule set.
    """
    extra = cfg.get("xhigh") or {}
    return {k: v for k, v in cfg.items() if k != "xhigh"} | dict(extra)


def judge_clip(rec: ClipRecord, cfg: dict) -> tuple[str | None, list[str]]:
    """Return ``(kept_tier, reject_reasons)``.

    ``kept_tier`` is ``"xhigh"``, ``"high"``, or ``None`` when the clip is rejected.
    ``reject_reasons`` is non-empty only for a rejected clip: failing the xhigh extra is
    not a rejection, it is simply not a promotion.
    """
    validate_policy(cfg, f"policy for {rec.source!r}")
    reasons = _kept_reasons(rec, cfg)
    if reasons:
        return None, reasons
    if not cfg.get("xhigh"):
        return "high", []
    return ("xhigh" if not _kept_reasons(rec, xhigh_cfg(cfg)) else "high"), []


def apply_quality_thresholds(rec: ClipRecord, cfg: dict) -> tuple[bool, list[str]]:
    """Boolean view of :func:`judge_clip`, for callers that only need kept/rejected."""
    tier, reasons = judge_clip(rec, cfg)
    return tier is not None, reasons
