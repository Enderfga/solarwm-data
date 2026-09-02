#!/bin/bash
# Deploy the SolarWM modifications into the vendored VIPE tree.
# Mod #1: install the Pi3X+MoGe-2 fused depth backend and register it in make_depth_model.
# Mod #2: per-frame-intrinsics BA (apply_perframe_intrinsics_ba.py) — gather intrinsics by
#   frame, scatter Jacobian by frame, drop the metric-depth assert. Use with the solarwm
#   pipeline (optimize_intrinsics=true, ba.fused=false).
set -e
R="${SOLAR_WM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VIPE="$R/third_party/vipe/vipe"
DEPTH="$VIPE/priors/depth"

cp "$R/vipe_patches/pi3x_moge_depth.py" "$DEPTH/pi3x_moge.py"

# register a `pi3xmoge` branch in make_depth_model (idempotent)
python3 - "$DEPTH/__init__.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
if "pi3xmoge" not in s:
    needle = '    else:\n        raise ValueError(f"Unknown depth model: {model}")'
    branch = (
        '    elif model_name == "pi3xmoge":\n'
        '        from .pi3x_moge import Pi3xMogeModel\n\n'
        '        return Pi3xMogeModel()\n\n'
        + needle
    )
    s = s.replace(needle, branch)
    open(p, "w").write(s)
    print("registered pi3xmoge in make_depth_model")
else:
    print("pi3xmoge already registered")
PY

# Mod #2: per-frame intrinsics BA (idempotent; reports per-edit status)
python3 "$R/vipe_patches/apply_perframe_intrinsics_ba.py" "$R/third_party/vipe"

echo "APPLY_VIPE_PATCHES_DONE"
