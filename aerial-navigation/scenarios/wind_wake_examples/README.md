# Campaign scenarios

Scenario configs for the aerial-navigation study. They are passed to
`scenario_launch.sh` by absolute path, so nothing has to be copied into the
scenario workspace:

```bash
"$LOTUSIM_SCENARIO_WS/src/simulation_run/executable/scenario_launch.sh" \
    --config "$PWD/aerial-navigation/scenarios/wind_wake_examples/m1_factorial_D.json"
```

All of them run on `energy.world`, with the aerial world started alongside it.
The smaller bring-up scenarios that exercise wind regions and PX4 SITL on their
own (`test_wind.json`, `x500_px4.json`, `px4_manual_wake_flying.json`,
`px4_offboard_patrol_test.json`, `wake_crossing_demo.json`) are not duplicated
here — they ship with LOTUSim-generic-scenario under the same directory name.

## The factorial cells

| File | Mean field | Turbulence |
|---|---|---|
| `m{1,2,3}_factorial_A.json` | uniform | ambient |
| `m{1,2,3}_factorial_B.json` | uniform | wake (synthetic control) |
| `m{1,2,3}_factorial_C.json` | wake-resolved | ambient |
| `m{1,2,3}_factorial_D.json` | wake-resolved | wake |

Only the `Wake` agent's `wind_regions` block differs between cells. The
turbines, the wind, the vehicles and their missions are copied through
untouched, so the four cells put the same vehicle on the same path and change
nothing but the field. Cell B's region geometry is identical to C and D — the
segment set is chosen on the true deficit before the published velocity is
decided — so no pair of cells differs in more than one factor.

Regenerate any set from its base scenario with
`scripts/make_factorial_configs.py <base.json> <prefix>`.

## The missions

| Mission | Configs | Farm | Vehicles |
|---|---|---|---|
| `m1` — transit along a turbine column | `m1_factorial_*.json` | 16 turbines, hub 52 m | 2 (in-wake, clear-air) |
| `m2` — orbit of one rotor | `m2_orbit.json` | 2 turbines | 2 |
| `m3` — wake crossing at x/D = 2, 4, 6, 8 | `m3_edge_traverse.json` | 1 turbine | 4, one per station |

`m2_orbit_E1.json` and `m3_edge_traverse_E1.json` are the uniform-wind controls
for their missions.

## Calibration and checks

| File | Purpose |
|---|---|
| `calib_scaling.json` | wind-to-vehicle coupling sweep, driven by `scripts/sweep_scaling.sh` |
| `verify_operating_point.json` | verification at the chosen coupling, driven by `scripts/verify_op.sh` |
| `verify_operating_point_render.json` | the same, with the Unity renderer on, for visual inspection |
| `smoke_turbulence.json` | turbulence smoke test and its seed repetition (`scripts/smoke_turbulence.sh`, `scripts/rep_turbulence.sh`) |
| `wake_crossing_demo_headless.json` | headless 16-turbine farm, no renderer; bring-up for the `m1` layout |

## Notes

PX4 agents (`"px4": true`) must spawn at `z: 0`. Turbine hub altitude is
per-file, and within a file `wake.turbines[].z` and the flight-path `z` values
must match.

The `wind` and `wake` JSON blocks are documented in LOTUSim-generic-scenario's
`doc/WRITE_SCENARIO.md`; the wake model itself in `doc/WAKE_EFFECT.md`.
