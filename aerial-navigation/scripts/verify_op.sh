#!/usr/bin/env bash
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
RUN_S="${RUN_S:-220}"
trap 'sed -i "s|<scaling_factor>[0-9.]*</scaling_factor>|<scaling_factor>1.0</scaling_factor>|" "$WORLD"' EXIT
teardown(){ for pat in "scenario_lau""nch.sh" "simulation""_run" "px""4 -i" "gz"" sim" "gnome-""terminal"; do pkill -f "$pat"; done
  for i in $(seq 1 30); do n=$(pgrep -fc "px""4 -i|gz"" sim|simulation""_run" 2>/dev/null); [ "${n:-0}" = "0" ] && break; sleep 2; done; sleep 8; }

for k in 0.25 0.35 0.5; do
  teardown
  sed -i "s|<scaling_factor>[0-9.]*</scaling_factor>|<scaling_factor>${k}</scaling_factor>|" "$WORLD"
  rm -rf "$PX4"/build/px4_sitl_default/rootfs/{0,1}/log 2>/dev/null
  LOGBEFORE=$(ls -dt "$SCEN"/scenario_logs/*/ 2>/dev/null | head -1)
  ( cd "$SCEN" && "$SP/run-clean.sh" "$LAUNCH" \
      --config "$SCENARIOS/verify_operating_point.json" >/dev/null 2>&1 ) &
  sleep "$RUN_S"
  LOG=$(ls -dt "$SCEN"/scenario_logs/*/ | head -1)
  # Assertions: probe the live topics before teardown (the region count is on a
  # latched topic that dies with the run), then judge the whole run afterwards.
  bash -c "source $STUDY/env.sh >/dev/null 2>&1; python3 $SP/probe_run.py '$LOG' --seconds 10" >/dev/null 2>&1
  echo "### k=$k"
  bash -c "source $STUDY/env.sh >/dev/null 2>&1; python3 $SP/check_run.py '$LOG' --collect" || echo "    (run rejected — excluded from analysis)"
  for d in 0 1; do
    lab=$([ "$d" = 0 ] && echo "IN-WAKE  " || echo "CLEAR-AIR")
    U=$(find "$PX4"/build/px4_sitl_default/rootfs/$d -name '*.ulg' -printf '%T@ %p\n' 2>/dev/null|sort -rn|head -1|cut -d' ' -f2-)
    [ -n "$U" ] && printf '    %s %s\n' "$lab" "$(python3 "$SP/tilt_stats.py" "$U" 2>/dev/null || echo ERR)" || echo "    $lab NO_ULOG"
  done
done
teardown
echo "done; scaling_factor restored"
