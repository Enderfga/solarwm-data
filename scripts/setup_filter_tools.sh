#!/bin/bash
# Set up the REAL quality/flow tools (no stand-ins): DOVER, UniMatch GMFlow, and
# a static ffmpeg with the vmafmotion filter. All sources are HTTP(S) direct
# (proxy-friendly): DOVER weights = GitHub release, UniMatch = AWS S3, ffmpeg =
# BtbN static build. Idempotent; writes a DONE marker.
R="${SOLAR_WM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TP="$R/third_party"; WT="$R/weights"; BIN="$R/bin"
mkdir -p "$TP" "$WT/unimatch" "$BIN"
# Install into the active runtime environment so the metric adapters import the same
# packages that this setup command verifies.
PIP="python3 -m pip install"

echo "=== DOVER ==="
[ -d "$TP/DOVER" ] || git clone --depth 1 https://github.com/QualityAssessment/DOVER.git "$TP/DOVER" || exit 21
$PIP einops timm pyyaml eva-decord fvcore scikit-video || exit 22
mkdir -p "$TP/DOVER/pretrained_weights"
[ -s "$TP/DOVER/pretrained_weights/DOVER.pth" ] || \
  wget -q -O "$TP/DOVER/pretrained_weights/DOVER.pth" \
    https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth || exit 23

echo "=== UniMatch GMFlow ==="
[ -d "$TP/unimatch" ] || git clone --depth 1 https://github.com/autonomousvision/unimatch.git "$TP/unimatch" || exit 24
[ -s "$WT/unimatch/gmflow-scale1.pth" ] || \
  wget -q -O "$WT/unimatch/gmflow-scale1.pth" \
    https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth || exit 25

echo "=== static ffmpeg (vmafmotion) ==="
if [ ! -x "$BIN/ffmpeg" ]; then
  cd /tmp
  wget -q -O ff.tar.xz https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz || exit 26
  tar xf ff.tar.xz
  cp ffmpeg-*/bin/ffmpeg "$BIN/ffmpeg" && chmod +x "$BIN/ffmpeg"
fi
"$BIN/ffmpeg" -hide_banner -filters 2>/dev/null | grep -q vmafmotion && echo "vmafmotion OK"

echo "FILTERSETUP_DONE"
touch "$R/.filtersetup_done"
