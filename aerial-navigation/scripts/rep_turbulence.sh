#!/usr/bin/env bash
# Does the in-wake attitude-variance excess survive repetition?
#
# The smoke run showed the in-wake drone gaining ~25% attitude sd against the
# clear-air drone's ~5% when turbulence was switched on. That was n=1, one
# seed, so it could have been one lucky noise draw. This runs the turbulence-ON
# arm over SEEDS different gust realisations, everything else held fixed.
#
# The comparison is a difference-in-differences against the deterministic OFF
# baseline: per seed, how much sd the in-wake drone gained over its OWN
# turbulence-off value, minus the same for the clear-air drone. Comparing raw
# in-wake vs clear-air sd would confound the gust with the fact that the two
# drones fly different air to begin with — they sit at 19.5 deg and 12.8 deg
# mean tilt. Their OFF sds are near-identical (4.65 vs 4.84), which is exactly
# what makes the ON gap attributable to turbulence.
#
# Writes one manifest line per seed to rep_manifest.txt for the analysis step;
# ULogs live in each run's own scenario_logs/<stamp>/ulogs/.
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
SEEDS="${SEEDS:-1 2 3 4 5 6}"
MANIFEST="$RESULTS/manifests/rep_manifest.txt"
mkdir -p "$(dirname "$MANIFEST")"

cp "$WORLD" "$WORLD.rep_bak"
trap 'mv -f "$WORLD.rep_bak" "$WORLD"' EXIT

teardown(){ for pat in "scenario_lau""nch.sh" "simulation""_run" "px""4 -i" "gz"" sim" "gnome-""terminal"; do pkill -f "$pat"; done
  for i in $(seq 1 30); do n=$(pgrep -fc "px""4 -i|gz"" sim|simulation""_run" 2>/dev/null); [ "${n:-0}" = "0" ] && break; sleep 2; done; sleep 8; }

patch_world(){ cp -f "$WORLD.rep_bak" "$WORLD"
  python3 - "$WORLD" "$AMBIENT_SIGMA" "$TAU" "$1" <<'PY'
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

: > "$MANIFEST"
for seed in $SEEDS; do
  teardown
  patch_world "$seed"
  rm -rf "$PX4"/build/px4_sitl_default/rootfs/{0,1}/log 2>/dev/null
  ( cd "$SCEN" && "$SP/run-clean.sh" "$LAUNCH" \
      --config "$SCENARIOS/smoke_turbulence.json" >/dev/null 2>&1 ) &
  sleep "$RUN_S"
  LOG=$(ls -dt "$SCEN"/scenario_logs/*/ | head -1)
  bash -c "source $STUDY/env.sh >/dev/null 2>&1; python3 $SP/probe_run.py '$LOG' --seconds 10" >/dev/null 2>&1
  echo "### seed=$seed  $LOG"
  # --collect copies the ULogs into this run's own dir. Everything downstream
  # reads those, never PX4's rootfs, which during a multi-run campaign holds
  # whatever finished most recently rather than this run's.
  if bash -c "source $STUDY/env.sh >/dev/null 2>&1; python3 $SP/check_run.py '$LOG' --collect"; then
    echo "$seed $LOG" >> "$MANIFEST"
  else
    echo "    (run rejected - excluded from the analysis)"
  fi
  for d in 0 1; do
    lab=$([ "$d" = 0 ] && echo "IN-WAKE  " || echo "CLEAR-AIR")
    U="$LOG/ulogs/instance_$d.ulg"
    [ -e "$U" ] && printf '    %s %s\n' "$lab" "$(python3 "$SP/tilt_stats.py" "$U" 2>/dev/null || echo ERR)" || echo "    $lab NO_ULOG"
  done
  echo
done
teardown
echo "done; world restored; manifest at $MANIFEST"
