#!/bin/bash
# Build the real VIPE pose engine (nv-tlabs/vipe) in its own venv.
# VIPE is a DROID-SLAM derivative that needs torch 2.7+ with a matching CUDA and
# compiles CUDA extensions, so it gets its own venv that INHERITS the host torch
# rather than installing one (Pi3/MoGe go in the same venv for the fused-depth backend).
# Editable clone so the per-frame-intrinsics BA modification can be patched in.
R="${SOLAR_WM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TP="$R/third_party"
VENV="${VIPE_VENV:-$R/.venv-vipe}"   # set to fast local storage if available
mkdir -p "$TP"

# The venv inherits the active PyTorch installation because VIPE compiles CUDA
# extensions against that exact runtime. Override the tested version when needed.
EXPECT_TORCH="${SOLAR_WM_EXPECT_TORCH:-2.8}"
python3 -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('$EXPECT_TORCH') else 1)" 2>/dev/null || {
  echo "PREFLIGHT FAIL: expected host torch $EXPECT_TORCH.x; got '$(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null || echo none)'. Set SOLAR_WM_EXPECT_TORCH to override." >&2; exit 30; }

[ -d "$TP/vipe" ] || git clone --depth 1 https://github.com/nv-tlabs/vipe.git "$TP/vipe" || exit 31

# Do not install a second PyTorch inside this venv. ``--system-site-packages`` keeps
# the Python package, CUDA runtime, and JIT-compiled extension on one version.
python3 -m venv --system-site-packages "$VENV" || exit 32
PIP="$VENV/bin/pip"
$PIP install -U pip wheel setuptools ninja cmake pybind11 || exit 33
"$VENV/bin/python" -c "import torch;print('venv torch',torch.__version__,'cuda',torch.version.cuda)" || exit 35

echo "=== build VIPE (editable CUDA extension) ==="
cd "$TP/vipe"
export TORCH_CUDA_ARCH_LIST="${SOLAR_WM_CUDA_ARCH_LIST:-9.0}"
export MAX_JOBS="${SOLAR_WM_BUILD_JOBS:-32}"
$PIP install -e . --no-build-isolation 2>&1 | tail -45 || exit 36

# Pin NumPy to the ABI line expected by the tested PyTorch environment after VIPE
# resolves its optional OpenCV dependencies.
export PIP_CONSTRAINT=
$PIP install "numpy<2" || exit 39

echo "=== install SolarWM pipeline profile (so 'vipe infer -p solarwm' resolves) ==="
cp "$R/vipe_patches/solarwm_pipeline.yaml" "$TP/vipe/configs/pipeline/solarwm.yaml" || exit 38

echo "=== verify import + torch<->numpy bridge ==="
"$VENV/bin/python" -c "import vipe; print('vipe import OK')" || exit 37
"$VENV/bin/python" -c "import torch,numpy as np; np.diag(torch.eye(3).numpy()); print('numpy bridge OK', np.__version__)" || exit 40

echo "VIPE_SETUP_DONE"
touch "$R/.vipe_setup_done"
