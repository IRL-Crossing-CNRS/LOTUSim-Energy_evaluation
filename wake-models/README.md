# Wake-resolved atmosphere: the models and their benchmark

![Python](https://img.shields.io/badge/python-3.10%2B-blue)

The environment a robot meets inside a wind farm, and the evidence that it is
computed accurately enough to be worth resolving. These are the engineering wake
models integrated into **LOTUSim-Energy**, the offshore wind farm simulator built
on LOTUSim: one drives the farm's energy estimate, the other supplies the
velocity field the inspection vehicles fly through.

Four models — **Jensen**, **Bastankhah–Porté-Agel Gaussian**, **Larsen**, and
**LOTUSim-Blended** (a near/far-wake blending formulation developed in this
work) — are benchmarked against an actuator-line CFD reference
(OpenFOAM v8 + turbinesFoam, k–ε RANS) and against **FLORIS v4** in both its
Gaussian and Jensen configurations, for the NREL 5 MW reference turbine, across
**seven protocols** in **two decoupled phases**.

---

## What is being evaluated, and why in two phases

A simulator that must both predict farm energy yield *and* fly autonomous
inspection drones through the farm imposes two different demands on a wake
model: an accurate **rotor-averaged inflow** for power, and an accurate
**resolved velocity field** — particularly its lateral gradient at the wake edge
— for navigation. A model can satisfy one and fail the other, because
compensating errors in the velocity field can cancel once that field is
integrated into a power.

The benchmark therefore keeps the two apart:

- **Phase P1 — wake field** (P1-1 … P1-3): centreline decay, lateral profile
  shape, and wake-edge velocity gradient. Metrics are dimensionless
  (`U/U_inf`, `|dU/dy|/U_inf`). No reference to power.
- **Phase P2 — power** (P2-1 … P2-4): per-turbine power, farm total, farm
  efficiency, and directional response. Metrics are in MW and %.

**P1 and P2 results are not interchangeable and must not be merged** — a low
wake-field RMSE says nothing about power accuracy, and vice versa.

### The finding: there is no single best model

This is the result, not an omission.

| Task | Recommended model | Why |
|------|-------------------|-----|
| Power, AEP, LCOE | **LOTUSim-Larsen** (`models/larsen.py`) | Lowest per-turbine RMSE (0.212 MW, 3.3–3.6× better than FLORIS); farm total within 0.3% of CFD |
| Spatial velocity field, drone hazard mapping, path planning | **LOTUSim-Blended** (`models/blended.py`) | Lowest lateral-profile RMSE; wake-edge gradient error below 20% across 8–12 m/s where the other models exceed 50% |

Both are integrated side by side in LOTUSim-Energy: Larsen drives the farm
energy estimate, Blended supplies the resolved field flown by the inspection
vehicles.

---

## Key results

**P2-1 — baseline power, Layout A (3-turbine row, 7D), `U_inf = 10 m/s`**

| Model | Per-turbine RMSE (MW) | Farm-total error | Time per evaluation |
|-------|----------------------|------------------|---------------------|
| LOTUSim-Larsen | **0.212** | +0.3% | 0.60 ms |
| LOTUSim-Blended | 0.251 | −6.4% | 0.25 ms |
| LOTUSim-Jensen | 0.309 | +0.3% | 0.053 ms |
| LOTUSim-Gaussian | 0.390 | +2.2% | 0.055 ms |
| FLORIS-Jensen | 0.697 | +11.4% | 2.8 ms |
| FLORIS-Gaussian | 0.767 | +9.4% | 6.6 ms |

Against the primary FLORIS-Gaussian configuration all four LOTUSim models run
**11–125× faster**; LOTUSim-Blended generates a full hub-height velocity field
over the 3-turbine farm in 4.5 ms at 100×100 points against ~150 ms for FLORIS
sampling. Timings were measured on a 12th-gen Intel Core i7-12650H and are
machine-dependent (`benchmark/compute_time_benchmark.py`).

**P1-3 — wake-edge gradient**, mean absolute error vs CFD at `r = 0.5D`,
1D–14D: Blended **17.5%** (isolated turbine) and **26.1%** on the independent
5D-spacing hold-out, against 42.6% (Larsen) and 70.1% (Gaussian).

Full per-protocol tables are in [docs/validation_summary.md](docs/validation_summary.md)
and in [benchmark/README.md](benchmark/README.md).

---

## Repository structure

```
wake-models/
├── models/          # LarsenWakeModel, BlendedWakeModel; extended_models.py = all 4 benchmark models
├── simulator/       # wind_farm.py — WindFarmSimulator, the LOTUSim-Energy
│                   integration layer (a library; API in simulator/README.md)
├── benchmark/       # The 7 validation protocols, figure generation, and data/ (CFD extracts)
├── analysis/        # Drone/UAV hazard-zone scripts built on BlendedWakeModel (supplementary)
├── plotting/        # wake_plot.py — shared plotting helpers (a library)
├── docs/            # cfd_setup.md, validation_summary.md, INTEGRATION.md
├── figures/         # main/ (primary figures, 600 DPI) and supplementary/
└── test_models.py   # quick sanity check of both production models
```

Start with **[benchmark/README.md](benchmark/README.md)** — it describes each
protocol, what it measures physically, which CFD case it scores against, which
script reproduces it, and which figure it produces.

---

## Installation

```bash
git clone https://github.com/IRL-Crossing-CNRS/LOTUSim-Energy_evaluation.git
cd LOTUSim-Energy_evaluation/wake-models
pip install -r requirements.txt
```

Python 3.10+, with numpy ≥ 1.24, matplotlib ≥ 3.7, floris ≥ 4.0, pyyaml ≥ 6.0,
pandas ≥ 2.0, Pillow ≥ 10.0. FLORIS is required for the benchmark (it is the
comparison baseline) but not for using the LOTUSim models themselves.

---

## Reproducing the benchmark

```bash
python3 test_models.py                          # 30-second sanity check

python3 benchmark/gen_all_figures.py            # 8 figures (7 reported) + the P1 and P2 tables
python3 benchmark/wind_speed_sweep.py           # P2-2 absolute-power figure
python3 benchmark/wind_direction_normalised.py  # P2-4 figure
python3 benchmark/compute_time_benchmark.py     # computational-cost table
python3 models/extended_models.py               # P2-1 / P2-3 / P2-4 numeric tables
```

Every path is resolved from the script's own location or the platform temp
directory — **no absolute paths, no dependence on the working directory**. Run
the scripts from anywhere. Primary figures go to `figures/main/` at 600 DPI,
everything else to `figures/supplementary/`. The CFD extracts the scripts read
live in `benchmark/data/`; the full OpenFOAM cases (several GB) are not in this
repository — see [docs/cfd_setup.md](docs/cfd_setup.md).

Verified on a clean run from a fresh shell: `gen_all_figures.py` reproduces all
seven of its committed primary figures **byte-for-byte**, `wind_speed_sweep.py`
reproduces the eighth, and together they reproduce every number in the
P1-1, P1-2, P1-3, P2-1, P2-2, P2-3 and P2-4 tables.

One figure needs data that is not shipped: `blended_multispeed_validation.png`
requires the 8 and 12 m/s single-turbine CFD cases, which are not shipped
here. The script reports which speeds are missing and leaves the
committed figure in place rather than overwriting it with a partial one; drop
the extracts into `benchmark/data/layout_single_8ms/` and
`layout_single_12ms/` and it regenerates with no code change.

See [benchmark/README.md](benchmark/README.md) §6 for the full list of caveats.

---

## Quick start

### Power prediction — `LarsenWakeModel`

```python
from models.larsen import LarsenWakeModel

model = LarsenWakeModel(diameter=126.0, ct=0.75, air_density=1.225,
                        cp=0.498, ambient_ti=0.08)

# 3-turbine aligned row, 7D spacing: (x_lateral, y_hub, z_downstream) in metres
turbines = [(0, 90.0, 0), (0, 90.0, 882), (0, 90.0, 1764)]

_, velocities, _ = model.wind_speeds_full(turbines, [0.0, 10.0])
print([round(v, 2) for v in velocities])              # [10.0, 7.71, 6.13] m/s
print([round(model.power(v) / 1e6, 3) for v in velocities])  # [3.724, 1.707, 0.858] MW
```

These are the Layout A / P2-1 numbers reported below.

### Spatial wake field — `BlendedWakeModel`

```python
import numpy as np
from models.blended import BlendedWakeModel

model = BlendedWakeModel(diameter=126.0, ct=0.75, air_density=1.225,
                         cp=0.498, ambient_ti=0.08)

turbines = [(0, 90.0, 0), (0, 90.0, 882), (0, 90.0, 1764)]
X, Y = np.meshgrid(np.linspace(0, 2200, 200), np.linspace(-300, 300, 200))

U_field = model.farm_velocity_field(10.0, X, Y, turbines)   # hub-height velocity
hazard  = model.farm_hazard_field(10.0, X, Y, turbines)     # 0=safe, 1=caution, 2=restricted
```

### Both together — `WindFarmSimulator`

See [simulator/README.md](simulator/README.md) and
[docs/INTEGRATION.md](docs/INTEGRATION.md) for the full integration reference:
all inputs, outputs, and access patterns.

---

## Benchmark configuration

**Turbine — NREL 5 MW reference**
rotor diameter 126 m, hub height 90 m, rated 5 MW,
cut-in / rated / cut-out 3 / 11.4 / 25 m/s, optimal TSR 7.55.

**Baseline flow**
`U_inf = 10 m/s` (near-peak `Ct = 0.75`, where the deficit and hence the
discrimination between models is largest), `TI = 8%`, shear exponent
`alpha = 0.12`, `rho = 1.225 kg/m3`, `z0 = 2e-4 m`, aligned inflow, neutral
stability, `Cp = 0.498`.

**Configurations**
Layout A — 3-turbine aligned row at 7D; Layout B — 16-turbine 4×4 grid, 7D
lateral × 9D streamwise; plus an isolated single-turbine case (the
uncontaminated wake-phase reference) and a 3-turbine 5D-spacing case used as an
independent hold-out.

**CFD reference**
OpenFOAM v8 + turbinesFoam actuator line, transient `pimpleFoam`, k–ε RANS,
neutral offshore ABL. Docker image `ellaj03/openfoam8-turbinesfoam`.
Steady RANS under-predicts far-wake recovery by roughly 10–15% relative to LES;
the comparison is internally consistent, but absolute far-wake behaviour should
be read with that bias in mind. Full setup in
[docs/cfd_setup.md](docs/cfd_setup.md).

---

## Data availability

The extracted CFD probe and turbine-power data used by every benchmark script
are in `benchmark/data/`. The full OpenFOAM case directories (mesh and field
files) and the 8 / 12 m/s single-turbine sensitivity cases are not shipped
here; open an issue on this repository to request them.
