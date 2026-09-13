# Aerial navigation and energy inside the wake

The wake field is not only computed and published: it acts on the vehicle. This
is the campaign that measures that coupling, and the analysis that turns 160
flights into the reported numbers.

An X500 quadrotor under PX4 SITL flies a 16-turbine farm through the field
LOTUSim's `WindRegionsPlugin` publishes from the `BlendedWakeModel` of
[`../wake-models`](../wake-models) — graded across the wake cone and carrying a
turbulent gust, rather than a uniform block of moving air.

**The finding the design exists to produce: the wake acts through two
independent mechanisms.** The velocity deficit changes how hard the vehicle
works — mean tilt and power — and adds no disturbance; slower air is calmer air.
The turbulence does the reverse: it adds disturbance and leaves the operating
point alone. A wake-versus-clear-air comparison measures only their sum, which
is why the design below delivers them separately.

The consequence for anyone simulating a farm flight: a uniform wind field set to
the wake's mean speed reproduces the mission's energy cost and predicts zero
disturbance, where the real wake roughly triples attitude variance. Battery
sizing comes out right; station-keeping does not.

Every run is judged before it is analysed, and every reported contrast is stored
with the manifest and commit it came from.

## What this study needs

| | version |
|---|---|
| [LOTUSim](https://github.com/IRL-Crossing-CNRS/LOTUSim) | see [`../docs/upstream-requirements.md`](../docs/upstream-requirements.md) |
| [LOTUSim-generic-scenario](https://github.com/IRL-Crossing-CNRS/LOTUSim-generic-scenario) | idem |
| PX4-Autopilot | SITL, built for `px4_sitl_default` |

The wind field this campaign needs is graded across the wake and carries a
turbulent fluctuation. Both are recent additions to `WindRegion`, and a
workspace without them runs these scenarios **without error** while producing a
different field. Check for them before trusting a run — the three one-line
checks are in
[`../docs/upstream-requirements.md`](../docs/upstream-requirements.md).

## Setup

```bash
export LOTUSIM_WS=$HOME/lotusim_ws
export LOTUSIM_SCENARIO_WS=$HOME/path/to/LOTUSim-generic-scenario
source aerial-navigation/env.sh
```

`scenario_launch.sh` sets up its own environment, so launching a scenario needs
no sourcing; everything else below does. `env.sh` sources ROS 2, the core
workspace and the scenario workspace, and then forces the core to the front of
every resolution path — colcon's prepend-unique dedupes an entry that is already
present rather than moving it, so a second LOTUSim install left on `PYTHONPATH`
otherwise stays ahead. That failure is quiet: the Python module resolves from
one workspace and the C typesupport from the other, which imports cleanly and
misbehaves somewhere unrelated. `env.sh` warns when it sees two.

## The experiment

Four wind fields, identical in everything else:

| | ambient turbulence | wake turbulence |
|---|---|---|
| **uniform mean wind** | A — baseline | B — turbulence only |
| **wake-resolved mean wind** | C — deficit only | D — resolved wake |

B is not a physical field. It delivers the wake's turbulence at freestream speed
so the turbulent and mean-deficit contributions separate. A real wake changes
both at once, and they act on different quantities — the deficit on the
operating point, the turbulence on the disturbance about it — so a
wake-vs-clear-air comparison measures only their sum.

Ambient turbulence is present in every cell, A included: the air outside the
wake has to be the same everywhere, or the baseline would also be measuring
"unnaturally still air".

Three missions: `m1` transits a turbine column, `m2` orbits one rotor, `m3`
crosses the wake at x/D = 2, 4, 6, 8. The scenario configs are in
[`scenarios/wind_wake_examples/`](scenarios/wind_wake_examples/) and are passed
to `scenario_launch.sh` by absolute path, so nothing has to be copied into the
scenario workspace.

## Running a campaign

```bash
MISSION=m1_factorial INSTANCES="0 1"     bash aerial-navigation/scripts/factorial.sh
MISSION=m3_factorial INSTANCES="0 1 2 3" bash aerial-navigation/scripts/factorial.sh
MISSION=m2_factorial INSTANCES="0 1"     bash aerial-navigation/scripts/factorial.sh

CELLS="A D" KAPPA=0.25 TAG=kappa0.25 bash aerial-navigation/scripts/factorial.sh
CELLS="A D" TAU=0.5    TAG=tau0.5    bash aerial-navigation/scripts/factorial.sh
```

Eight seeds per cell, ~8.5 min per run. The full set is 160 runs, about 21 h.
Cells are ordered seed-major, so stopping early leaves complete 2×2 blocks
rather than a few over-sampled cells.

Each run is checked before it is kept. `check_run.py` asserts that the plugin
loaded, that the field was published with the region count the cell expects, and
that every vehicle actually flew; it rejects a flight whose ULog is shorter than
60 % of the run window, which is what a full disk produces — PX4 keeps flying and
silently stops writing, so a truncated log otherwise looks like a short flight.
`factorial.sh` refuses to start with under 5 GB free where PX4 writes.

PX4 writes its ULogs to its own rootfs rather than the run's log directory, so a
batch analysis would otherwise validate whichever run finished last;
`check_run.py --collect` copies them into `<log_dir>/ulogs/` per run and
everything downstream reads those.

Accepted runs are appended to
`results/manifests/factorial_manifest_<mission>[_<tag>].txt`, one line of
`cell seed log_directory`. The manifests are the index from a condition to the
flights behind it; the ULogs themselves live under the scenario workspace's
`scenario_logs/` and are not in this repository.

The manifests of the 160-run campaign are committed. Their run directories
carry the literal `<scenario_ws>` where the machine that flew them had its
workspace; the analysis scripts substitute `$LOTUSIM_SCENARIO_WS` for it, so
they work unchanged on any machine holding the same logs. They were
reconstructed from `results/campaign_2026-09-04.log`, which records every run's
condition, log directory and verdict — all 160 passed.

## Analysis

```bash
python3 aerial-navigation/scripts/factorial_analysis.py --metric resvar
python3 aerial-navigation/scripts/factorial_analysis.py \
        --manifest aerial-navigation/results/manifests/factorial_manifest_m3_factorial.txt \
        --instance 0 --metric energy
```

Metrics: `mean`, `sd`, `p95`, `var`, `resvar`, `sat`, `power`, `energy`, `rotor`.

The mission is an out-and-back along the wind, so its two legs sit at very
different tilts (~19° downwind, ~35–41° upwind) and are never pooled — how much
of each a run captures depends on PX4 startup jitter, which on an earlier
campaign flipped the answer outright (p = 0.58 pooled against p = 0.0156 per leg
on identical data). The analysis segments by leg and uses only complete blocks.

`resvar` is what is reported for variability. Raw variance over a leg counts both
how much the vehicle was disturbed and how much its steady working point changed
with position; a mean deficit changes the second without disturbing anything.
Removing the deterministic along-track profile first separates them. On M1's
downwind leg the deficit contributes +12.9 deg² raw and −0.95 residual: slower
air is calmer air, and the raw figure was reading its steady tilt profile as if
it were disturbance. Turbulence contributes +7.83 deg² residual, in 8 seeds
out of 8.

`energy` is Wh per leg or lap. On an orbit the mean-field effect on attitude
cancels across headings, but power goes as the cube of rotor speed and does not.

`rotor` is the 99th percentile of the busiest rotor as a fraction of its ceiling
— thrust margin rather than attitude margin.

## Reported numbers

```bash
python3 aerial-navigation/scripts/collect_results.py
```

Writes [`results/factorial_results.json`](results/factorial_results.json): every
manifest × instance × metric, each contrast with a bootstrap CI, seed count,
source manifest and the commit that produced it. Recomputing a reported value
from the ULogs on demand makes it track whatever the analysis code says that
day; this file is what the tables cite, so a number can be traced without
re-running anything.

Without the ULogs it collects nothing, and rather than write an empty file over
the stored one it says so and leaves the file alone.

`resvar` is deliberately absent from that file. It needs the profile averaged
across seeds and cannot be computed per run like the others; storing a subtly
different quantity under the same name would be worse than leaving it out.
Reproduce it with `factorial_analysis.py --metric resvar`.

[`results/campaign_2026-09-04.log`](results/campaign_2026-09-04.log) is the
console log of the 160-run campaign: every run's verdict and per-run statistics.
Absolute paths in it have been replaced by `<scenario_ws>` and `<results>`.

[`results/sweep_failure_record.json`](results/sweep_failure_record.json) records
what the disk-truncated sweeps showed before they were deleted.

## Other tools

- `wake_field_topview.py` — top view of the delivered field at hub height.
  Replays `regions_for_wind()` and the plugin's `ResolveWind`/`SampleRadial`, so
  it shows what the vehicle receives rather than what the model computes.
- `wake_replay.py` — recovers the wind and its lateral gradient along a recorded
  trajectory. Both are functions of position, so nothing is instrumented in the
  loop.
- `probe_run.py`, `tilt_stats.py`, `ulog_extract.py` — per-run probes and ULog
  extraction.
- `make_factorial_configs.py` — derives the four cells from any base scenario:
  `python3 scripts/make_factorial_configs.py <base.json> <prefix>`.
- `verify_op.sh`, `sweep_scaling.sh` — wind-to-vehicle coupling calibration and
  its verification at the operating point. Both patch `scaling_factor` in the
  core's `aerialWorld.world`, fly a short scenario per value, and restore the
  world on exit.
- `smoke_turbulence.sh` — turbulence smoke test: the same mission with the gust
  on and off. `rep_turbulence.sh` repeats the ON case over gust seeds, writing
  `results/manifests/rep_manifest.txt`.
- `rep_analysis.py` — reads that manifest against one deterministic OFF baseline
  and reports, per leg, how much more attitude variance the in-wake vehicle
  gains from turbulence than the clear-air one does:
  `python3 scripts/rep_analysis.py --off <log_dir>`. This is the smoke test's
  analysis; the factorial campaign uses `factorial_analysis.py` instead.

### Modules the above import

Not entry points — you do not run these directly.

- `metrics.py` — turns one replayed flight into the metric row every analysis
  reads: tilt statistics, actuator saturation, and rotor shaft power integrated
  to Wh. Aerial power is computed from the simulator's own rotor constants
  (`P = kQ·kT·ω³`), not from LOTUSim's marine energy plugin, whose polynomial is
  fitted to a thruster a PX4 quadrotor neither uses nor writes to.
- `run-clean.sh` — wrapper that strips a snap-confined editor's GTK and locale
  variables before launching. Without it, a scenario started from inside such an
  editor's terminal loads the snap's glibc against the system one and
  `gnome-terminal` dies with a symbol lookup error. The campaign drivers call it
  for you; it matters only if you launch a scenario by hand from that terminal.
