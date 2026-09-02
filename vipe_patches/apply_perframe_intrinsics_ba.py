#!/usr/bin/env python3
"""VIPE modification #2: per-frame intrinsics in the bundle adjustment.

VIPE stores ONE intrinsics per view (`buffer.intrinsics` is (V,4)) and the generic
BA solver gathers it per-view (by rig index qi/qj). This patch makes the solver
optimise a separate per-FRAME intrinsics buffer `intrinsics_pf` (N,4): the dense
flow term gathers intrinsics by frame index (fi/fj, decoupled from the rig), and
scatters its Jacobian per frame. Use with `ba.fused=false` (the fused CUDA kernel
assumes shared intrinsics) and `optimize_intrinsics=true`. The per-view machinery
(depth filter, point cloud, output) is left untouched — only the solver BA and the
final dump read per-frame intrinsics.

Idempotent. Reports every edit's status; applies matches, skips already-applied.

    python3 apply_perframe_intrinsics_ba.py [VIPE_ROOT]
"""
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "SOLAR_WM_ROOT", ".") + "/third_party/vipe"
V = os.path.join(ROOT, "vipe")

GEOM = os.path.join(V, "slam/maths/geom.py")
TERMS = os.path.join(V, "slam/ba/terms.py")
BUFFER = os.path.join(V, "slam/components/buffer.py")
SYSTEM = os.path.join(V, "slam/system.py")

# (path, label, old, new)  — applied only if `old` present and `new` absent
EDITS = [
    # ---- geom.py: add fi/fj params, gather intrinsics per-frame ----
    (GEOM, "geom.signature", """    jacobian_p_d: bool,
    jacobian_f: bool,
    jacobian_r: bool,
):""", """    jacobian_p_d: bool,
    jacobian_f: bool,
    jacobian_r: bool,
    fi: torch.Tensor | None = None,
    fj: torch.Tensor | None = None,
):"""),
    (GEOM, "geom.fidef", """    jacobian_p_d = jacobian_p_d or jacobian_f or jacobian_r
""", """    jacobian_p_d = jacobian_p_d or jacobian_f or jacobian_r
    # Per-frame intrinsics: gather intrinsics by frame index (fi/fj) when
    # provided, decoupled from rig/view index (qi/qj). Default = per-view behaviour.
    if fi is None:
        fi = qi
    if fj is None:
        fj = qj
"""),
    (GEOM, "geom.iproj_qi", """        disps_uv,
        intrinsics[qi],
        camera_type,
        compute_jz=jacobian_p_d,""", """        disps_uv,
        intrinsics[fi],
        camera_type,
        compute_jz=jacobian_p_d,"""),
    (GEOM, "geom.proj_qj", "proj_points(X1, intrinsics[qj], camera_type",
     "proj_points(X1, intrinsics[fj], camera_type"),

    # ---- terms.py: pass fi/fj, relax assert, scatter intrinsics per-frame ----
    (TERMS, "terms.assert", "        assert intrinsics.shape[0] == rig.shape[0]\n",
     "        # per-frame intrinsics may differ in count from rig (views); see geom fi/fj\n"),
    (TERMS, "terms.geomcall", """            jacobian_r=jacobian and optimize_rig,
        )""", """            jacobian_r=jacobian and optimize_rig,
            fi=self.pose_i_inds,
            fj=self.pose_j_inds,
        )"""),
    (TERMS, "terms.scatter", """                J_dict["intrinsics"] = SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.rig_i_inds, self.rig_j_inds]),""",
     """                J_dict["intrinsics"] = SparseDenseBlockMatrix(
                    i_inds=torch.cat([term_inds, term_inds]),
                    j_inds=torch.cat([self.pose_i_inds, self.pose_j_inds]),"""),

    # ---- buffer.py: allocate intrinsics_pf, use it as the BA variable ----
    (BUFFER, "buffer.alloc", """        self.intrinsics = torch.zeros(
            self.n_views,
            self.camera_type.intrinsics_dim(),
            device=device,
            dtype=torch.float,
        )""", """        self.intrinsics = torch.zeros(
            self.n_views,
            self.camera_type.intrinsics_dim(),
            device=device,
            dtype=torch.float,
        )
        # Per-frame intrinsics: one (fx,fy,cx,cy) per frame,
        # optimised in the generic-solver BA. Initialised lazily in bundle_adjustment.
        self.intrinsics_pf = torch.zeros(
            buffer_size,
            self.camera_type.intrinsics_dim(),
            device=device,
            dtype=torch.float,
        )"""),
    (BUFFER, "buffer.var", """            "dense_disp": disps_flattened,
            "intrinsics": self.intrinsics,
            "rig": SE3(self.rig),""", """            "dense_disp": disps_flattened,
            "intrinsics": self.intrinsics_pf,
            "rig": SE3(self.rig),"""),
    (BUFFER, "buffer.fix", """        if not optimize_intrinsics:
            solver.set_fixed("intrinsics")
""", """        # init any uninitialised per-frame intrinsics from the per-view intrinsics
        _pf_uninit = self.intrinsics_pf[:, 0] == 0
        if _pf_uninit.any():
            self.intrinsics_pf[_pf_uninit] = self.intrinsics[0]
        if not optimize_intrinsics:
            solver.set_fixed("intrinsics")
        else:
            # per-frame: optimise intrinsics only for frames in the active window
            _all_pf = torch.arange(self.intrinsics_pf.shape[0], device=self.device)
            _active_pf = pi_unique[(pi_unique >= t0) & (pi_unique < t1)] if t0 < t1 else pi_unique
            solver.set_fixed("intrinsics", _all_pf[~torch.isin(_all_pf, _active_pf)])
"""),

    # ---- system.py: drop metric-depth assert; dump per-frame intrinsics ----
    (SYSTEM, "system.assert", "                assert not self.config.optimize_intrinsics\n",
     "                pass  # SolarWM: per-frame intrinsics optimised alongside metric depth\n"),
    (SYSTEM, "system.dump", """        original_intrinsics = torch.stack(
            [resizer.recover_intrinsics(self.buffer.intrinsics[v]) for v, resizer in enumerate(resizers)]
        )""", """        original_intrinsics = torch.stack(
            [resizer.recover_intrinsics(self.buffer.intrinsics[v]) for v, resizer in enumerate(resizers)]
        )
        # SolarWM: dump per-frame optimised intrinsics (recovered to original res)
        import os as _os
        _pf_dump = _os.environ.get("SOLAR_WM_PF_DUMP", "")
        if _pf_dump and getattr(self.buffer, "intrinsics_pf", None) is not None:
            import numpy as _np
            _pf = self.buffer.intrinsics_pf[: self.buffer.n_frames]
            _rec = torch.stack([resizers[0].recover_intrinsics(_pf[t]) for t in range(_pf.shape[0])])
            _np.save(_pf_dump, _rec.detach().cpu().numpy())
            print(f"SOLAR_WM_PF_DUMP wrote {_rec.shape} -> {_pf_dump}", flush=True)"""),
]

ok = miss = done = 0
for path, label, old, new in EDITS:
    if not os.path.exists(path):
        print(f"NOFILE {label}: {path}"); miss += 1; continue
    s = open(path).read()
    if new in s and old not in s:
        print(f"SKIP   {label} (already applied)"); done += 1; continue
    if old in s:
        s = s.replace(old, new, 1)
        open(path, "w").write(s)
        print(f"OK     {label}"); ok += 1
    else:
        print(f"MISS   {label}  <-- anchor not found in {os.path.basename(path)}"); miss += 1

print(f"\nsummary: applied={ok} already={done} MISSED={miss}")
sys.exit(1 if miss else 0)
