#!/usr/bin/env bash
# Smoke test for the turbulence path: does the gust actually reach the drone?
#
# Two arms of the SAME scenario geometry, run back to back:
#   OFF  verify_operating_point.json + stock world  -> deterministic field
#   ON   smoke_turbulence.json + world patched with the turbulence tags
#
# What makes it a test rather than a demo: the mean field is identical between
# the arms (same wind, same wake model, same waypoints), so any change in the
# attitude STANDARD DEVIATION can only have come from the gust. The mean tilt
# should stay put — the fluctuation is zero-mean — while sd rises, and rises
# more for the in-wake drone (sigma ~2.57 m/s) than the clear-air one (ambient
# sigma 0.80 m/s). Mean moving would mean the gust is NOT zero-mean, and the
# mean field the wake benchmark validated is no longer what the vehicle flies.
#
# Three independent pieces of evidence are collected, so a failure says where
# it broke rather than just that it broke:
#   world -> plugin   the plugin's own "ready" line echoes the parsed SDF
#   SDK   -> wire     probe_run.py records region_sigma_max off the live topic
#   plugin -> vehicle the ULog attitude statistics
#
# There is deliberately no fourth check on the sigma each LINK resolved: that
# is only recorded in the plugin's per-second Update summary, which is logged
# at debug, and launch/lotusim assigns LOTUSIM_SPDLOG_LEVEL="info"
# unconditionally, so exporting it here is overwritten. Getting it needs
# --debug through scenario_launch.sh, which also flips gz to -v4.
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
# Follow the env var rather than a hardcoded path: PX4-Autopilot was moved
# off the root filesystem (where its ULog writing filled a 46G partition and
# silently truncated 40+ runs) and is reached through a symlink.
PX4="${PX4_AUTOPILOT_PATH:-$HOME/PX4-Autopilot}"
RUN_S="${RUN_S:-220}"
AMBIENT_SIGMA="${AMBIENT_SIGMA:-0.8}"
TAU="${TAU:-1.0}"
SEED="${SEED:-1}"

# Requested, but note it does NOT take effect: launch/lotusim overwrites it
# (see the header). Left in so that if that assignment is ever made
# conditional, the debug summary starts appearing here with no other change.
export LOTUSIM_SPDLOG_LEVEL=debug

# Always leave the world as we found it, however this exits.
cp "$WORLD" "$WORLD.smoke_bak"
trap 'mv -f "$WORLD.smoke_bak" "$WORLD"' EXIT

teardown(){ for pat in "scenario_lau""nch.sh" "simulation""_run" "px""4 -i" "gz"" sim" "gnome-""terminal"; do pkill -f "$pat"; done
  for i in $(seq 1 30); do n=$(pgrep -fc "px""4 -i|gz"" sim|simulation""_run" 2>/dev/null); [ "${n:-0}" = "0" ] && break; sleep 2; done; sleep 8; }

world_off(){ cp -f "$WORLD.smoke_bak" "$WORLD"; }
world_on(){ world_off
  python3 - "$WORLD" "$AMBIENT_SIGMA" "$TAU" "$SEED" <<'PY'
import sys, pathlib
w, sigma, tau, seed = sys.argv[1:5]
p = pathlib.Path(w); s = p.read_text()
old = "    </plugin>\n  </world>"
new = (f"      <ambient_turbulence_sigma>{sigma}</ambient_turbulence_sigma>\n"
       f"      <turbulence_time_constant>{tau}</turbulence_time_constant>\n"
       f"      <turbulence_seed>{seed}</turbulence_seed>\n"
       "    </plugin>\n  </world>")
assert s.count(old) == 1, "wind_regions plugin block not found as expected"
p.write_text(s.replace(old, new))
PY
}

run_arm(){ # $1 = label, $2 = config
  teardown
  rm -rf "$PX4"/build/px4_sitl_default/rootfs/{0,1}/log 2>/dev/null
  ( cd "$SCEN" && "$SP/run-clean.sh" "$LAUNCH" \
      --config "$SCENARIOS/$2" >/dev/null 2>&1 ) &
  sleep "$RUN_S"
  LOG=$(ls -dt "$SCEN"/scenario_logs/*/ | head -1)
  bash -c "source $STUDY/env.sh >/dev/null 2>&1; python3 $SP/probe_run.py '$LOG' --seconds 10" >/dev/null 2>&1
  echo "### ARM $1"
  echo "    log: $LOG"

  PLOG=$(ls -t "$CORE"/lotus_logs/*/aerialWorld_wind_regions_plugin.txt 2>/dev/null | head -1)
  grep -h "WindRegionsPlugin ready" "$PLOG" 2>/dev/null | tail -1 | sed 's/^.*\[info\]: /    /'
  echo "    region_sigma_max on the wire  : $(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('region_sigma_max'))" "$LOG/run_assertions.json" 2>/dev/null)"

  bash -c "source $STUDY/env.sh >/dev/null 2>&1; python3 $SP/check_run.py '$LOG' --collect" || echo "    (run rejected)"
  printf '    %-9s %6s %6s %6s %6s %6s %6s %6s\n' "" n mean sd p50 p95 max "%>44"
  # Read the ULogs check_run.py just COLLECTED into this run's own log dir, not
  # PX4's rootfs. The rootfs holds whatever is newest, which during a two-arm
  # comparison is not necessarily this arm's — the trap README.md documents,
  # and one this script fell into: it reported a clear-air p95 of 18.96 vs the
  # collected copy's 12.20 for the very same flight.
  for d in 0 1; do
    lab=$([ "$d" = 0 ] && echo "IN-WAKE  " || echo "CLEAR-AIR")
    U="$LOG/ulogs/instance_$d.ulg"
    [ -e "$U" ] && printf '    %s %s\n' "$lab" "$(python3 "$SP/tilt_stats.py" "$U" 2>/dev/null || echo ERR)" || echo "    $lab NO_ULOG"
  done
  echo
}

world_off; run_arm "OFF (deterministic)" verify_operating_point.json
world_on;  run_arm "ON  (ambient sigma=$AMBIENT_SIGMA m/s, T=$TAU s, seed=$SEED)" smoke_turbulence.json
teardown
echo "done; world restored"
