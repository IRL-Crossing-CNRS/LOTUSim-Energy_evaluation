#!/usr/bin/env bash
# scaling_factor sweep: for each k, patch aerialWorld.world, fly the calib
# scenario, and report cruise tilt from the PX4 ULog. Restores the world at the end.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
STUDY="$(cd "$SP/.." && pwd)"
SCENARIOS="$STUDY/scenarios/wind_wake_examples"
RESULTS="$STUDY/results"
# The two LOTUSim workspaces. env.sh exports both; set them in the environment
# if it was not sourced.
CORE="${LOTUSIM_PATH:-${LOTUSIM_WS:-$HOME/lotusim_ws}/src/LOTUSim}"
SCEN="${LOTUSIM_SCENARIO_WS:-$HOME/Documents/workspace/lotusim/LOTUSim-generic-scenario}"
LAUNCH="$SCEN/src/simulation_run/executable/scenario_launch.sh"
WORLD="$CORE/assets/worlds/aerialWorld.world"
PX4="${PX4_AUTOPILOT_PATH:-$HOME/PX4-Autopilot}"
RUN_S="${RUN_S:-170}"
trap 'sed -i "s|<scaling_factor>[0-9.]*</scaling_factor>|<scaling_factor>1.0</scaling_factor>|" "$WORLD"' EXIT

teardown() {
  pkill -f scenario_launch.sh; pkill -f simulation_run; pkill -f "px4 -i"
  pkill -f "gz sim"; pkill -f gnome-terminal; sleep 5
}

printf '%-6s %6s %6s %6s %6s %6s %6s\n' k n mean p50 p95 max "%>44"
for k in 1.0 0.5 0.25 0.125; do
  teardown
  sed -i "s|<scaling_factor>[0-9.]*</scaling_factor>|<scaling_factor>${k}</scaling_factor>|" "$WORLD"
  rm -rf "$PX4"/build/px4_sitl_default/rootfs/0/log 2>/dev/null
  ( cd "$SCEN" && "$SP/run-clean.sh" "$LAUNCH" \
      --config "$SCENARIOS/calib_scaling.json" >/dev/null 2>&1 ) &
  sleep "$RUN_S"
  ULG=$(find "$PX4"/build/px4_sitl_default/rootfs/0 -name '*.ulg' \
        -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  if [ -n "$ULG" ]; then
    printf '%-6s %s\n' "$k" "$(python3 "$SP/tilt_stats.py" "$ULG" 2>/dev/null || echo ERR)"
  else
    printf '%-6s %s\n' "$k" "NO_ULOG"
  fi
done
teardown
echo "done; scaling_factor restored to 1.0"
