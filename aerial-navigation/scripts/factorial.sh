#!/usr/bin/env bash
# A mission as a 2x2 factorial: mean field x turbulence.
#
#   MISSION=m1_factorial bash factorial.sh          (default; 2 drones)
#   MISSION=m3_factorial INSTANCES="0 1 2 3" RUN_S=300 bash factorial.sh
#   MISSION=m2_factorial RUN_S=300 bash factorial.sh
#
# Sensitivity sweeps reuse the same driver with two cells rather than four:
#   CELLS="A D" KAPPA=0.25 TAG=kappa0.25 bash factorial.sh
#   CELLS="A D" TAU=0.5    TAG=tau0.5    bash factorial.sh
# A and D alone are enough to show whether the ordering survives a parameter
# change; the B/C decomposition does not need repeating at every value.
#
# Cells come from make_factorial_configs.py; this only drives them.
#
#            | ambient TI (0.8 m/s) | wake TI (2.1-2.6 m/s)
#   ---------+----------------------+-----------------------
#   uniform  |  A  baseline         |  B  turbulence only
#   wake     |  C  deficit only     |  D  the real wake
#
# Why four cells and not two. A wake changes the mean speed AND the turbulence
# at the same time, and the pilots showed those two push in OPPOSITE directions
# for a transiting multirotor: the deficit made M2 easier (-8.3% tilt) while
# turbulence makes flying harder. A wake-vs-clear-air comparison measures their
# sum, which is why M2 and M3 came out null. These four cells separate them:
#
#   B-A  what turbulence does with no deficit
#   C-A  what the deficit does with no added turbulence
#   D-A  the real wake, both together
#   D-B-C+A  the interaction: does a deficit make the vehicle MORE sensitive to
#            a given gust, or is the extra variance just the larger sigma?
#
# B is not a physical field. There is no place with freestream speed and
# wake-level turbulence; it is a manipulation to break the confound, and it
# must be reported as a synthetic control, not as a scenario. Its region
# geometry is identical to C and D (verified offline), so B/C/D differ in
# exactly one factor at a time.
#
# Ambient sigma is 0.8 m/s in EVERY cell, including A: the air outside the wake
# must be the same everywhere, or "no wake regions" would also mean "unnaturally
# still air" and the baseline would be measuring two things too.
#
# RUN_S is long enough for a complete out-and-back. At 220 s the mission
# captured the whole downwind leg but only 23-44 s of the upwind one, and how
# much varied with PX4 startup jitter — enough to decide the answer on its own
# (see rep_analysis.py). Both legs complete, or the leg comparison is not
# available.
#
# Ordered seed-major so that stopping early leaves complete 2x2 blocks rather
# than a few cells over-sampled.
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
RUN_S="${RUN_S:-470}"
AMBIENT_SIGMA="${AMBIENT_SIGMA:-0.8}"
TAU="${TAU:-1.0}"
# Wind-to-vehicle coupling. Lives in the world file, not the scenario config,
# so it is swept here rather than by generating more configs.
KAPPA="${KAPPA:-0.35}"
# Distinguishes manifests when the same mission is run at several parameter
# values; without it a kappa sweep would overwrite the baseline campaign's
# record of which log directory holds which cell and seed.
TAG="${TAG:-}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8}"
CELLS="${CELLS:-A B C D}"
MISSION="${MISSION:-m1_factorial}"
INSTANCES="${INSTANCES:-0 1}"
# One manifest per mission, so running M2 does not overwrite M1's record of
# which log directory holds which cell and seed.
MANIFEST="$RESULTS/manifests/factorial_manifest_${MISSION}${TAG:+_$TAG}.txt"
mkdir -p "$(dirname "$MANIFEST")"

cp "$WORLD" "$WORLD.fact_bak"
trap 'mv -f "$WORLD.fact_bak" "$WORLD"' EXIT

teardown(){ for pat in "scenario_lau""nch.sh" "simulation""_run" "px""4 -i" "gz"" sim" "gnome-""terminal"; do pkill -f "$pat"; done
  for i in $(seq 1 30); do n=$(pgrep -fc "px""4 -i|gz"" sim|simulation""_run" 2>/dev/null); [ "${n:-0}" = "0" ] && break; sleep 2; done; sleep 8; }

patch_world(){ cp -f "$WORLD.fact_bak" "$WORLD"
  python3 - "$WORLD" "$AMBIENT_SIGMA" "$TAU" "$1" "$KAPPA" <<'PY'
import re, sys, pathlib
w, sigma, tau, seed, kappa = sys.argv[1:6]
p = pathlib.Path(w); s = p.read_text()
s, n = re.subn(r"<scaling_factor>[0-9.]+</scaling_factor>",
               f"<scaling_factor>{kappa}</scaling_factor>", s)
assert n == 1, "scaling_factor not found exactly once"
old = "    </plugin>\n  </world>"
new = (f"      <ambient_turbulence_sigma>{sigma}</ambient_turbulence_sigma>\n"
       f"      <turbulence_time_constant>{tau}</turbulence_time_constant>\n"
       f"      <turbulence_seed>{seed}</turbulence_seed>\n"
       "    </plugin>\n  </world>")
assert s.count(old) == 1, "wind_regions plugin block not found as expected"
p.write_text(s.replace(old, new))
PY
}

# A full disk does not fail a run, it truncates its ULog: the vehicle flies
# normally and PX4 simply stops writing. That is indistinguishable from a short
# flight in the logs, so check before starting rather than discover it after.
free_mb=$(df -Pm "$PX4" | awk 'NR==2{print $4}')
if [ "${free_mb:-0}" -lt 5000 ]; then
  echo "ABORT: only ${free_mb}MB free where PX4 writes its logs ($PX4)."
  echo "A campaign needs several GB; a full disk truncates ULogs silently."
  exit 1
fi
echo "free space where PX4 logs: ${free_mb}MB"

: > "$MANIFEST"
t_start=$(date +%s)
total=$(( $(echo $SEEDS | wc -w) * $(echo $CELLS | wc -w) )); done_n=0
for seed in $SEEDS; do
  for cell in $CELLS; do
    teardown
    patch_world "$seed"
    # Clear every instance this mission uses, not a hardcoded pair: a stale
    # log under an instance we forgot to clear is exactly how a previous run's
    # ULog gets analysed as if it were this one.
    for d in $INSTANCES; do
      rm -rf "$PX4/build/px4_sitl_default/rootfs/$d/log" 2>/dev/null
    done
    ( cd "$SCEN" && "$SP/run-clean.sh" "$LAUNCH" \
        --config "$SCENARIOS/${MISSION}_${cell}.json" >/dev/null 2>&1 ) &
    sleep "$RUN_S"
    LOG=$(ls -dt "$SCEN"/scenario_logs/*/ | head -1)
    bash -c "source $STUDY/env.sh >/dev/null 2>&1; python3 $SP/probe_run.py '$LOG' --seconds 10" >/dev/null 2>&1
    done_n=$((done_n+1))
    el=$(( $(date +%s) - t_start ))
    printf '### %s%s cell=%s seed=%s   [%d/%d, %dm elapsed, ~%dm left]\n' \
      "$MISSION" "${TAG:+ $TAG}" "$cell" "$seed" "$done_n" "$total" "$((el/60))" "$(( (el/done_n*(total-done_n))/60 ))"
    echo "    $LOG"
    if bash -c "source $STUDY/env.sh >/dev/null 2>&1; python3 $SP/check_run.py '$LOG' --collect --instances $INSTANCES --run-seconds $RUN_S"; then
      echo "$cell $seed $LOG" >> "$MANIFEST"
    else
      echo "    (rejected - excluded)"
    fi
    for d in $INSTANCES; do
      U="$LOG/ulogs/instance_$d.ulg"
      [ -e "$U" ] && printf '    inst%-6s %s\n' "$d" "$(python3 "$SP/tilt_stats.py" "$U" 2>/dev/null || echo ERR)" || echo "    inst$d NO_ULOG"
    done
    echo
  done
done
teardown
echo "done; world restored; manifest at $MANIFEST"
