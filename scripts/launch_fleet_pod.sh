#!/bin/bash
# Launch eight SolarWM annotation workers on one node, one per local GPU.
# Done-markers make worker restarts resumable.
#
#   launch_fleet_pod.sh <source> <node_rank> [limit] [world]
#
# node_rank: group-relative node index (0-based).
# limit: maximum work items per worker (0 means all).
# world: total worker count participating in this source's sharding.
set -u
SRC="$1"; NR="$2"; LIMIT="${3:-0}"; WORLD_N="${4:-8}"
# Repository root defaults to the parent of this scripts directory.
D="${SOLAR_WM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Source an optional gitignored environment file for storage credentials and paths.
for _e in "${SOLAR_WM_COS_ENV:-}" "$D/.solarwm_cos.env" "$D/../../.solarwm_cos.env" "$HOME/.solarwm_cos.env"; do
  [ -n "$_e" ] && [ -f "$_e" ] && { source "$_e"; break; }
done
export SOLAR_WM_ROOT="$D" SOLAR_WM_WEIGHTS="$D/weights" HF_HOME="${HF_HOME:-$D/weights/hf}"
export WORLD="$WORLD_N" SOLAR_WM_LIMIT="$LIMIT"
# Use SpatialVID-HQ by default for the registered SpatialVID source.
if [ "$SRC" = "spatialvid" ]; then
  export SOLAR_WM_SPATIALVID_HQ="${SOLAR_WM_SPATIALVID_HQ:-1}"
fi

LOGD="${SOLAR_WM_LOGDIR:-$D/logs}/$SRC"
mkdir -p "$LOGD"
cd "$D" || exit 1

for lr in 0 1 2 3 4 5 6 7; do
  gr=$((NR * 8 + lr))
  # setsid fully detaches into a new session so workers survive their launching shell
  # closing. nohup alone is NOT enough under container exec: the runtime kills the
  # exec process group on exit, which orphans and kills every worker.
  CUDA_VISIBLE_DEVICES=$lr LOCAL_RANK=$lr NODE_RANK=$NR \
    setsid python3 scripts/run_solarwm_fleet.py "$SRC" > "$LOGD/r${gr}.log" 2>&1 < /dev/null &
done
echo "pod NODE_RANK=$NR launched 8 workers for '$SRC' (global ranks $((NR*8))-$((NR*8+7)), limit=$LIMIT)"
