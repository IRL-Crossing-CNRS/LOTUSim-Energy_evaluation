# models/

Wake model implementations: the two production classes used by LOTUSim-Energy,
plus the full set of four models used for the benchmark.

## Files

| File | Contents |
|------|----------|
| `larsen.py` | `LarsenWakeModel` — the recommended **power** model |
| `blended.py` | `BlendedWakeModel` — the recommended **spatial-field** model |
| `extended_models.py` | All four benchmarked models (`JensenWakeModel`, `GaussianWakeModel`, `LarsenWakeModel`, `BlendedWakeModel`) plus `run_floris()` and the P2-1 / P2-3 / P2-4 comparison driver |
| `__init__.py` | Exports `LarsenWakeModel` and `BlendedWakeModel` |

> **`extended_models.py` is both a library and a script.** Importing it executes
> a module-level body that runs FLORIS, prints the P2-1, P2-3 and P2-4 tables,
> and writes its figures to `figures/supplementary/`. The figure windows now
> open only when the file is run directly (`python3 models/extended_models.py`);
> previously a bare module-level `plt.show()` blocked every import under an
> interactive matplotlib backend, which hung the whole benchmark.
> `benchmark/compute_time_benchmark.py` shows how to skip the script body
> entirely: it execs only the class definitions.

> **If you only need the two production models, import them directly** —
> `from models.larsen import LarsenWakeModel` and
> `from models.blended import BlendedWakeModel` have no script body and no
> FLORIS dependency.

## Which model to use

There is no single best model — that is the benchmark's actual finding, not a
gap. The two models answer different questions and are used side by side.

### `LarsenWakeModel` — power

Use for per-turbine wind speed and power, farm energy yield (AEP), LCOE, and
any quantitative power prediction.

Lowest power-prediction error of all models benchmarked: per-turbine RMSE
**0.212 MW** against CFD in Layout A (3-turbine row, 7D), 3.3–3.6× better than
FLORIS v4 (0.697 / 0.767 MW), with farm-total power within **0.3%** of CFD. It
remains the best at farm scale (Layout B, 16 turbines): RMSE 0.284 MW and a
farm efficiency of 43.6% against 43.0% for CFD, where FLORIS overstates farm
output by 40–43%.

Rotor inflow is rotor-disk averaged over 20 points across the full rotor
diameter, accounting for vertical wind shear (power law, `alpha = 0.12`).

### `BlendedWakeModel` — spatial velocity field

Use for the resolved hub-height velocity field across the farm, wake-overlap
and high-deficit zone identification, drone/UAV hazard mapping and path
planning, and spatial planning for maintenance vessels.

It blends a near-wake Larsen component with a calibrated far-wake Gaussian
component, weighted by `Ct`. It gives the lowest lateral-profile RMSE (0.0245)
and the most faithful wake-edge velocity gradient: **17.5%** mean absolute
error against the isolated-turbine CFD reference over 1D–14D, and **26.1%** on
an independent 3-turbine 5D-spacing hold-out never used in calibration
(Larsen 42.6%, Gaussian 70.1% on the same hold-out). Gradient error stays below
20% across 8–12 m/s, where the other models exceed 50%.

**It is not a power model.** Its wide calibrated lateral sigma — the same
property that reproduces the wake-edge gradient — over-spreads the deficit and
depresses the centreline velocity at a collinear downstream turbine, giving a
−22.8% error at T2 and a −6.4% farm total in P2-1. Use Larsen for power.

### Complementary by design

Larsen gives the numbers; Blended gives the picture. A typical workflow runs
Larsen for power output, then Blended for the spatial wake structure used in
operations planning. `simulator/WindFarmSimulator` wires both together.

## Constructor parameters

Shared: `diameter` (m, required), `ct` (required), `air_density`
(default 1.225 kg/m³), `cp`, `ambient_ti` (default 0.08), `cut_in_speed`
(default 3.0 m/s), `cut_out_speed` (default 25.0 m/s).

- `LarsenWakeModel`: `cp` defaults to **0.35**; also accepts `rated_power`.
- `BlendedWakeModel`: `cp` defaults to **0.498**; also accepts `eps`
  (Gaussian initial-width parameter, default 0.22).

The benchmark uses `diameter=126.0`, `ct=0.75`, `cp=0.498`, `ambient_ti=0.08`,
`air_density=1.225` for both.

## Key methods — `LarsenWakeModel`

| Method | Returns |
|--------|---------|
| `wind_speeds_full(turbines, wind_vector)` | rotor-averaged effective wind speed at each turbine |
| `power(wind_speed)` | turbine power output, W |
| `multi_speed(turbines, wind_vectors, times)` | farm energy yield over a time series |
| `single_speed(turbines, wind_vector, hours)` | farm energy for one condition |
| `velocity_at_point(U_inf, x, r)` | single-wake velocity at downstream `x`, lateral `r` |
| `farm_velocity_at_point(U_inf, x, y, turbines)` | superposed farm velocity at a point |
| `wake_radius(x)`, `local_ti(...)`, `rotor_averaged_speed(...)` | internals, exposed for inspection |

## Key methods — `BlendedWakeModel`

| Method | Returns |
|--------|---------|
| `velocity_at_point(U_inf, x, r)` | blended single-wake velocity |
| `velocity_gradient_at_point(U_inf, x, r)` | lateral gradient `dU/dr`, s⁻¹ — the P1-3 metric |
| `farm_velocity_field(U_inf, X, Y, turbines)` | vectorised 2D hub-height velocity field |
| `farm_hazard_field(U_inf, X, Y, turbines)` | hazard grid, 0 = safe, 1 = caution, 2 = restricted |
| `farm_velocity_at_point(U_inf, x, y, turbines)` | farm velocity at a single point |
| `wake_hazard_zone(U_inf, x, r)` | hazard class at a single point |
| `calibrated_sigma(x)`, `blend_weight(x, ct)` | the calibration internals |

Hazard-zone thresholds and the drone application are documented in
[../analysis/README.md](../analysis/README.md).

## Provenance of the numbers above

Every figure quoted here comes from the benchmark below and is reproducible from `benchmark/` — see
[../benchmark/README.md](../benchmark/README.md) for which script produces which
number, and [../docs/validation_summary.md](../docs/validation_summary.md) for
the tables. They are a snapshot: if the code in this directory changes, rerun
the benchmark before citing them.
