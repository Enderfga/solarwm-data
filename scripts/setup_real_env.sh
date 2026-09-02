#!/bin/bash
# Set up the real-inference environment on a single-GPU node.
# Installs deps, vendors Pi3 + MoGe, and predownloads
# model weights into the repo's weights/ dir. Idempotent; writes a DONE marker.
#
# The weight downloads are large — run it detached, e.g.
#   setsid nohup bash scripts/setup_real_env.sh > setup.log 2>&1 < /dev/null &

export PIP="python3 -m pip install --user"

ROOT="${SOLAR_WM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TP="$ROOT/third_party"
WT="$ROOT/weights"
export HF_HOME="$WT/hf"
mkdir -p "$TP" "$WT" "$HF_HOME"

echo "=== [0/6] system ffmpeg (libaom-av1 decoder + ffprobe) ==="
# Install ffmpeg with libaom-av1 and ffprobe support. The bundled imageio-ffmpeg
# binary remains a fallback for H.264-only inputs.
(apt-get update && apt-get install -y ffmpeg) >/tmp/apt_ffmpeg.log 2>&1 \
  && echo "system ffmpeg OK" || echo "WARN: apt ffmpeg failed (imageio-ffmpeg fallback covers h264 only)"

echo "=== [1/6] pip: filter + VLM deps ==="
# imageio-ffmpeg: bundled static ffmpeg fallback if system ffmpeg is unavailable
# (covers h264 trim/resample; AV1 needs the system libaom build above).
$PIP scenedetect decord transformers accelerate "qwen-vl-utils[decord]" huggingface_hub safetensors einops "opencv-python-headless<4.10" imageio-ffmpeg || exit 11

echo "=== [2/6] clone Pi3 ==="
[ -d "$TP/Pi3" ] || git clone --depth 1 https://github.com/yyfz/Pi3.git "$TP/Pi3" || exit 12
[ -f "$TP/Pi3/requirements.txt" ] && $PIP -r "$TP/Pi3/requirements.txt"

echo "=== [3/6] install MoGe ==="
$PIP "git+https://github.com/microsoft/MoGe.git" || exit 13

echo "=== [4/6] download Pi3 weights ==="
python3 -c "from huggingface_hub import snapshot_download as s; s('yyfz233/Pi3', local_dir='$WT/pi3')" || exit 14

echo "=== [5/6] download MoGe-2 weights ==="
python3 -c "from huggingface_hub import snapshot_download as s; s('Ruicheng/moge-2-vitl-normal', local_dir='$WT/moge2')" || exit 15

echo "=== [6/6] download Qwen2.5-VL-7B ==="
python3 -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='$WT/qwen25vl7b')" || exit 16

echo "=== [fix] pin numpy<2 ==="
# Keep NumPy on the ABI line expected by the tested PyTorch environment.
$PIP "numpy<2" || exit 17
python3 -c "import torch,numpy as np; print('numpy',np.__version__); torch.from_numpy(np.zeros((2,2)))" || exit 18

# NOTE: Pi3 is a cloned repo (not pip-installed) — callers must put it on
# PYTHONPATH (export PYTHONPATH=$ROOT/third_party/Pi3:$PYTHONPATH) before running
# precompute_fused_depth.py / the pose stage under system python3.

echo "REALSETUP_DONE"
touch "$ROOT/.realsetup_done"
