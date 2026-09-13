# What the upstream repositories have to provide

Two of the three studies run entirely on their own; one drives the simulator and
therefore pins what the simulator must be able to do.

| Study | Needs a simulator |
|---|---|
| [`../wake-models`](../wake-models) | no — pure Python: numpy, matplotlib, FLORIS |
| [`../aerial-navigation`](../aerial-navigation) | yes — LOTUSim + LOTUSim-generic-scenario + PX4 SITL |
| [`../inspection-perception`](../inspection-perception) | no — the evaluation is offline; the simulator produced the rendered imagery, which is committed |

## The three repositories

| Repository | Role here |
|---|---|
| [LOTUSim](https://github.com/IRL-Crossing-CNRS/LOTUSim) | simulation core: Gazebo worlds, the wind-region plugin, the message interfaces |
| [LOTUSim-generic-scenario](https://github.com/IRL-Crossing-CNRS/LOTUSim-generic-scenario) | scenario layer: the launcher, the agent SDK, the wake agent that publishes the field |
| [LOTUSim-Energy](https://github.com/IRL-Crossing-CNRS/LOTUSim-Energy) | Unity renderer for the offshore farm — it produced the rendered inspection imagery, and it is what you watch a flight in. Not needed for a headless campaign |

Install the first two as their own documentation describes —
`install_core_and_generic_scenario.sh` in LOTUSim-generic-scenario sets up both
— then point this repository at them:

```bash
export LOTUSIM_WS=$HOME/lotusim_ws                                  # the core workspace
export LOTUSIM_SCENARIO_WS=$HOME/path/to/LOTUSim-generic-scenario   # the scenario workspace
```

Those two variables are all this repository needs. `aerial-navigation/env.sh`
derives the rest (`LOTUSIM_PATH`, `LOTUSIM_MODELS_PATH`) from them.

## Required version

The aerial-navigation campaign needs a wind field that is **graded across the
wake** and carries a **turbulent fluctuation**. Both are recent additions.
A workspace without them runs the scenarios without error and produces a
different field — a top-hat carrying the centreline deficit, with no gust — so
check for them before trusting a run:

```bash
grep -c radial_speed      "$LOTUSIM_PATH/interfaces/lotusim_msgs/msg/WindRegion.msg"   # expect 1
grep -c turbulence_sigma  "$LOTUSIM_PATH/interfaces/lotusim_msgs/msg/WindRegion.msg"   # expect 1
grep -c radial_samples    "$LOTUSIM_SCENARIO_WS/src/lotusim_sdk/lotusim_sdk/agents/environment/wake/wake_regions.py"  # expect >= 1
```

The commits that add them:

**LOTUSim**
- `feat: turbulent gust on the vehicle wind path`
- `tune: set aerialWorld wind coupling to the calibrated operating point`
- `Deliver the wake's radial profile instead of its centreline`

**LOTUSim-generic-scenario**
- `feat(wake): publish a per-segment turbulence amplitude`
- `feat(wake): synthetic control cells for a 2x2 factorial`
- `Publish the wake's radial profile, not just its centreline`

## What they change

1. **A graded cross-wake profile.** A wind region carries one `linear_velocity`,
   so a producer publishing only that value has to pick a single radius to stand
   for the whole cross-section. Picking the axis hands every vehicle in the cone
   the centreline deficit, out to the radius where the true deficit has already
   decayed to the cut-off — about 2.1× the deficit the model predicts once
   integrated over the cross-section, and a top-hat with no interior gradient at
   all. `WindRegion.radial_speed` carries the profile instead, and the plugin
   interpolates it at the vehicle's radius.

2. **A turbulent fluctuation.** `WindRegion.turbulence_sigma` gives the standard
   deviation in m/s of a zero-mean fluctuation about `linear_velocity`, resolved
   from the same region lookup that decides a link's mean wind, so the gust
   changes as a vehicle crosses a wake edge. Sigma in m/s rather than a
   dimensionless intensity: an intensity is a ratio to *some* reference speed,
   and `linear_velocity` carries the deficit speed while Crespo-type wake-added
   turbulence intensity is defined against the freestream, so a consumer handed
   a ratio could not recover the right reference.

3. **Two synthetic-control options** on the producer, which the factorial cells
   need: publishing the freestream speed while keeping the wake's segment
   geometry and sigma (cell B), and forcing sigma to the ambient value inside
   the wake (cell C).

Both message fields are additive on the wire — `turbulence_sigma` defaults to
0.0 and `radial_speed` to empty, which is what every earlier publisher produces
— so a consumer that ignores them behaves exactly as before.

## Other requirements

- **PX4-Autopilot**, built for `px4_sitl_default`. The campaign drivers clear
  and read `build/px4_sitl_default/rootfs/<instance>/log`. Set
  `PX4_AUTOPILOT_PATH` if it is not at `$HOME/PX4-Autopilot`.
- **Disk.** PX4 writes ULogs to its own rootfs. A full disk does not fail a run,
  it truncates the log silently, so `factorial.sh` refuses to start with under
  5 GB free and `check_run.py` rejects a truncated log afterwards. A full
  campaign needs several GB.
