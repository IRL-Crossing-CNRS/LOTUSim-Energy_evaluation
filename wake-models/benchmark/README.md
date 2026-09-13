# benchmark/ — wake-model validation protocols

This directory contains everything needed to reproduce the wake-model
benchmark: power and spatial-field accuracy against actuator-line CFD and
FLORIS.

Four engineering wake models — **LOTUSim-Jensen**, **LOTUSim-Gaussian**,
**LOTUSim-Larsen** and **LOTUSim-Blended** — are compared against an
actuator-line CFD reference (OpenFOAM v8 + turbinesFoam, k–ε RANS) and against
**FLORIS v4** in both its Gaussian and Jensen velocity-model configurations,
for the NREL 5 MW reference turbine.

---

## 1. What the benchmark actually measures

The benchmark is deliberately split into **two decoupled phases**. This split is
the point of the study, not a presentation choice:

| Phase | Question it answers | Quantity | Units |
|-------|--------------------|----------|-------|
| **P1 — wake field** | Does the model put the *velocity* in the right place? | resolved hub-height velocity field: centreline decay, lateral profile shape, lateral gradient at the wake edge | dimensionless (`U/U_inf`, `\|dU/dy\|/U_inf`) |
| **P2 — power** | Does the model predict the *energy* the farm produces? | per-turbine power, farm total, farm efficiency, directional response | MW and % |

A model can carry compensating errors in the velocity field that cancel once
the field is integrated into a power, so a power-only comparison flatters a
spatially inaccurate model — and vice versa. **P1 and P2 results must never be
merged or averaged**: a low `RMSE_wake` says nothing about power accuracy.

The headline finding is that **no single model wins both phases**:
LOTUSim-Larsen is the most accurate power model (P2), LOTUSim-Blended is the
most accurate spatial-field model (P1). Both are integrated side by side into
LOTUSim-Energy.

### Why the lateral gradient matters

`|dU/dy| / U_inf` evaluated at `r = 0.5D` (the rotor-edge radius) measures how
*sharp* the wake boundary is. It is the discriminating metric for flying an
inspection drone through a farm: the rate at which the wind changes across the
wake edge is what determines hazard-zone boundaries and whether a vehicle can
hold its trajectory. It is a higher-order quantity than the velocity itself, so
it separates models that a centreline plot makes look equivalent.

Both Jensen implementations (LOTUSim and FLORIS) are **structurally excluded**
from this metric: their top-hat profile is uniform inside the wake and equal to
freestream outside it, so `dU/dy` is a step discontinuity at the wake boundary
and identically zero everywhere else, including at `r = 0.5D`. This is a
property of the model, not a performance result.

---

## 2. Reference data (`data/`)

Only the **extracted CSV probe and turbine data** are in this repository. The
full OpenFOAM case directories (mesh + fields, several GB each) are not; see
[docs/cfd_setup.md](../docs/cfd_setup.md) for how the cases were built and run.

| Directory | CFD case | Role in the benchmark |
|-----------|----------|----------------------|
| `data/layout_single/` | isolated single NREL 5 MW rotor, `U_inf = 10 m/s` | **Primary wake-phase reference** (P1-1, P1-2, P1-3 left panel). Uncontaminated by downstream-rotor induction. |
| `data/layout_a/` | Layout A — 3-turbine aligned row, 7D spacing, `U_inf = 10 m/s`, aligned | Power baseline (P2-1) and the P1-3 *calibration* farm panel. P2-2 and P2-4 also use Layout A, but their off-baseline CFD points (6/8/12/14 m/s, 15° yaw) are separate runs not shipped here — see §6. |
| `data/layout_5D/` | 3-turbine aligned row, **5D** spacing | **Independent hold-out** for P1-3. Never used to calibrate any model. |
| `data/layout_b/` | Layout B — 16-turbine 4×4 grid, 7D lateral × 9D streamwise | Farm-scale power (P2-3) and the Layout B extensions of P1-1 / P1-2. |

Each case holds:

```
<case>/wakeProfiles/<timestep>/wake_{1D,3D,5D,7D,10D,14D}_U.csv   # streamwise velocity
                              wake_{...}_k.csv                    # turbulent kinetic energy
<case>/turbines/<timestep>/turbineN.csv                           # per-turbine power (layout_a, layout_b)
```

Profiles are hub-height lateral lines spanning ±2.5D. All scripts time-average
over the **statistically converged tail** (the last 20 dumps, or the second half
of the run if fewer are available) to exclude the start-up transient.

> **Why `layout_single` and not `layout_a` for P1-1/P1-2:** the two agree to
> within 0.001 at 1D–5D, but from 7D the Layout A probes sit inside the
> induction field of T2/T3 and read `U/U_inf = 0.34–0.56` against 0.71 for the
> isolated rotor. Scoring a single-turbine model against Layout A far probes
> would penalise it for physics it is not being asked to represent.

> **LOTUSim-Blended caveat:** its Gaussian sub-model is calibrated against
> `layout_single`, so any P1-1/P1-2 score against that case is an **in-sample
> fit**, reported for transparency and excluded from the rankings. Its genuinely
> independent tests are the Layout B extension and the `layout_5D` hold-out.

### Tracing the CFD power reference

The wake-phase scripts read the velocity profiles from `data/` directly. The
**power-phase CFD values are hardcoded literals** in `gen_all_figures.py` and
`models/extended_models.py` (`cfd_powers = [3.806, 1.904, 0.560]`,
`cfd_rows = [3.964, 2.046, 0.619, 0.190]`) rather than being recomputed from
`data/` at run time. They are nonetheless fully traceable to the shipped
turbine CSVs: each is `P = 0.5 · rho · A · <Cp> · U_inf^3` with `Cp` averaged
over the final 20 samples of the turbinesFoam output.

```python
import numpy as np, pandas as pd
rho, D, U = 1.225, 126.0, 10.0
A = np.pi * (D / 2) ** 2

for i in (1, 2, 3):                       # Layout A -> 3.806, 1.905, 0.560 MW
    cp = pd.read_csv(f'data/layout_a/turbines/0/turbine{i}.csv').tail(20)['cp'].mean()
    print(i, round(0.5 * rho * A * cp * U ** 3 / 1e6, 3))
```

The same computation over `data/layout_b/turbines/<t>/turbine1..16.csv` gives
row means 3.964 / 2.046 / 0.619 / 0.189 MW and a farm total of 27.273 MW,
matching the P2-3 table.

---

## 3. The seven protocols

All protocols use the baseline condition unless stated: NREL 5 MW
(`D = 126 m`, hub 90 m), `U_inf = 10 m/s`, `TI = 8%`, shear exponent
`alpha = 0.12`, `rho = 1.225 kg/m3`, `Ct = 0.75`, `Cp = 0.498`, aligned inflow.

### Wake-field phase (P1)

| ID | What is evaluated, physically | Metric | Reference case | Produced by |
|----|-------------------------------|--------|----------------|-------------|
| **P1-1** | Streamwise wake recovery along the hub-height centreline — how fast the wake refills with momentum, 1D→14D. Extended to Layout B to test decay under multi-row accumulation. | `U/U_inf` vs `x/D`; RMSE over the six probe stations | `layout_single` (+ `layout_b` for the extension) | `gen_all_figures.py` |
| **P1-2** | Lateral wake width and cross-sectional shape across the rotor plane — the quantity that governs how much of a downstream rotor a wake actually covers. | `RMSE_wake` of the lateral `U/U_inf` profile across ±2.5D, averaged over six downstream positions | `layout_single` (+ `layout_b`) | `gen_all_figures.py` |
| **P1-3** | Lateral velocity gradient at the wake edge — wake-boundary sharpness, i.e. hazard-zone delineation for UAV trajectory planning. Three panels: isolated turbine, 7D farm (calibration), 5D farm (hold-out). | `\|dU/dy\| / U_inf` at `r = 0.5D` vs `x/D`; mean absolute % error vs CFD | `layout_single`, `layout_a`, `layout_5D` | `gen_all_figures.py` |

P1-3 additionally reports a **multi-speed generalisation test** (8, 10,
12 m/s ⇒ `Ct = 0.80, 0.75, 0.53`) confirming that Blended's `Ct`-based
calibration is not fitted to one operating point. See §6 for the data caveat.

### Power phase (P2)

| ID | What is evaluated, physically | Metric | Layout | Produced by |
|----|-------------------------------|--------|--------|-------------|
| **P2-1** | Baseline power accuracy in a single aligned row: wake-deficit error isolated from turbine aerodynamics by holding `Cp` fixed. | per-turbine power (MW), `RMSE_abs` (MW), farm-total error (%) | A (3 turbines, 7D) | `gen_all_figures.py`; table also printed by `models/extended_models.py` |
| **P2-2** | Whether wake-interaction accuracy holds across the operational wind-speed range, with the real NREL 5 MW `Cp`/`Ct` curves (so `Ct` and hence deficit strength vary). | `P/P_T1`, `RMSE_norm` vs CFD | A; `U_inf` swept 5–15 m/s, CFD at 6, 8, 10, 12, 14 m/s | `wind_speed_sweep.py` (absolute); `gen_all_figures.py` (normalised) |
| **P2-3** | Farm-scale accuracy under deep multi-row wake compounding, where linear superposition is stressed hardest; the LCOE-relevant number. | row-averaged power (MW), farm total (MW), `RMSE_abs`, farm efficiency `eta` | B (16 turbines, 7D × 9D) | `gen_all_figures.py`; also `models/extended_models.py` |
| **P2-4** | Wake-overlap recovery under non-aligned inflow — exposes models with unrealistically sharp wake boundaries that snap to full recovery too early. | `P_T2 / P_T1,0deg` vs direction offset 0°–90° | A; CFD spot checks at 0° and 15° only | `models/extended_models.py` (computes the sweep); `wind_direction_normalised.py` (plots it) |

**Computational cost** is measured by
`compute_time_benchmark.py`: wall-clock ms per evaluation for the 3-turbine
power task and for full hub-height field generation, model construction
excluded, versus both FLORIS configurations.

> **Protocol numbering:** the in-code section headings now match the
> **P1-1..P1-3, P2-1..P2-4** labels (seven protocols). Earlier revisions
> used a P1-1..P1-4 development numbering in which the gradient protocol was
> "P1-4"; if you meet that label in an old figure or note, it means P1-3.

---

## 4. How to rerun

```bash
pip install -r ../requirements.txt      # numpy, matplotlib, floris>=4, pandas, pyyaml, Pillow
```

Every path used by every script — CFD input, figure output, scratch YAML for
FLORIS — is derived from the script's own location or from the platform temp
directory. There are no absolute paths and nothing depends on the working
directory, so the scripts can be run from anywhere. Figures always land in
`figures/main/` or `figures/supplementary/`, never in whatever directory you
happened to launch from.

FLORIS is imported at run time; its bundled NREL 5 MW turbine definition and
default input file are used as the FLORIS baseline.

### Master script — run this first

```bash
python3 benchmark/gen_all_figures.py
```

Writes 600 DPI PNGs into `../figures/main/` and prints, to stdout, the numbers
behind the P1-1, P1-2, P1-3 and Layout B centreline tables.

It also prints the P2-1, P2-3 and P2-4 tables before it starts, because
importing `models/extended_models.py` for the model classes executes that
file's script body. That is why the first few minutes of the run are FLORIS
solves for protocols this script does not itself plot — deliberate, since it
means one command produces every P1 and P2 number, but it is the bulk of the
runtime. Total is a few minutes.

Verified reproducible: a clean run from a fresh shell reproduces all seven
figures it writes **byte-for-byte**, plus every number in the
P1-1, P1-2, P1-3, P2-1, P2-3 and P2-4 tables. The eighth figure,
`blended_multispeed_validation.png`, needs CFD data that is not shipped — the
script says so and leaves the committed file alone (see §6).

| Output figure (`figures/main/`) | Protocol | Content |
|----------------------------------|----------|---------|
| `wake_centreline_decay.png` | P1-1 | Centreline `U/U_inf`, 1D–14D, isolated turbine |
| `wake_centreline_4x4.png` | P1-1 (Layout B ext.) | Centreline decay through rows R2–R4 of the 4×4 farm |
| `wake_lateral_profiles.png` | P1-2 | Lateral profiles at 1D, 3D, 5D, 7D, 10D, 14D |
| `drone_gradient_comparison.png` | P1-3 | 3-panel wake-edge gradient: single turbine / 7D calibration / 5D hold-out |
| `blended_multispeed_validation.png` | P1-3 | Blended gradient at 8, 10, 12 m/s vs CFD — **not reproducible from the shipped data, see §6** |
| `wake_comparison_power.png` | P2-1 | Per-turbine power and normalised wake-loss profile |
| `grid_4x4_comparison.png` | P2-3 | Row-averaged power and per-row error, 16-turbine farm |
| `wind_speed_sweep_normalised.png` | P2-2 | `P/P_T1` vs wind speed, T2 and T3 panels |

### Individual scripts

```bash
python3 benchmark/wind_speed_sweep.py            # P2-2, absolute power version
python3 benchmark/wind_direction_normalised.py   # P2-4 figure
python3 benchmark/compute_time_benchmark.py      # computational-cost table (stdout)
python3 models/extended_models.py                # P2-1/P2-3/P2-4 numeric tables (stdout)
```

- `wind_speed_sweep.py` writes the P2-2 figure straight to
  `figures/main/wind_speed_sweep.png`. Its second, normalised figure goes to
  `figures/supplementary/wind_speed_sweep_normalised_allmodels.png` — it
  previously shared a filename with a *different* normalised figure produced by
  `gen_all_figures.py`, so whichever ran last silently overwrote the other.
- `wind_direction_normalised.py` **replots the tabulated P2-4 model outputs**
  rather than recomputing them; the sweep itself is run by
  `models/extended_models.py`, which prints the same normalised
  `P_T2/P_T1,0deg` table. It writes a single figure to `figures/main/`.
- `compute_time_benchmark.py` reports the fastest of N evaluations
  (N = 2000 LOTUSim, N = 300 FLORIS for power; N = 200 / 40 for field
  generation), so results are machine-dependent — the published numbers were
  measured on a 12th-gen Intel Core i7-12650H.

### Figure ↔ file mapping

| Figure label | File | Source |
|-------------------|------|--------|
| `fig:wake_centreline` | `wake_centreline_decay.png` | `gen_all_figures.py` |
| `fig:4x4_wake_structure` | `wake_centreline_4x4.png` | `gen_all_figures.py` |
| `fig:wake_lateral` | `wake_lateral_profiles.png` | `gen_all_figures.py` |
| `fig:drone_gradient_comparison` | shipped as `p1-3_full.png` | `gen_all_figures.py` → `drone_gradient_comparison.png` |
| `fig:blended_multispeed` | `blended_multispeed_validation.png` | `gen_all_figures.py` |
| `fig:wake_comparison_power` | `wake_comparison_power.png` | `gen_all_figures.py` |
| `fig:wind_speed_sweep` | `wind_speed_sweep.png` | `wind_speed_sweep.py` |
| `fig:grid_4x4` | `grid_4x4_comparison.png` | `gen_all_figures.py` |
| `fig:wind_direction` | `wind_direction_normalised.png` | `wind_direction_normalised.py` |
| `fig:farm_layouts`, `fig:layout_p1-1` | `layout_A_3turbine.png`, `layout_B_4x4grid.png`, `centreline_single_turbine.png`, `centreline4x4_overlap.png` | hand-drawn schematics — **not** script-generated |
| `fig:wake_rating`, `fig:power_rating` | — | TikZ, drawn directly in the LaTeX source |

---

## 5. Headline results

As reported (see `../docs/validation_summary.md` for the
tables, and note that those numbers are a snapshot — rerun the scripts if the
model code changes).

**P1 — wake field**

- P1-1 centreline RMSE vs isolated-turbine CFD: Jensen **0.105**, Larsen 0.124,
  FLORIS-Jensen 0.133, FLORIS-Gaussian 0.246, Gaussian 0.353.
  (Blended 0.020 — in-sample, excluded from the ranking.)
- P1-2 average lateral-profile RMSE: Blended **0.0245** (in-sample), Larsen
  0.0353, Jensen 0.0506, FLORIS-Jensen 0.0867, FLORIS-Gaussian 0.0940,
  Gaussian 0.1640.
- P1-3 wake-edge gradient, mean absolute error vs CFD — isolated turbine:
  Blended **17.5%**, Larsen 43.1%, FLORIS-Gaussian 68.3%, Gaussian 76.5%.
  On the independent 5D hold-out: Blended **26.1%**, Larsen 42.6%,
  Gaussian 70.1%. Blended stays below 20% across 8–12 m/s where the other
  models exceed 50%.

**P2 — power**

- P2-1 per-turbine RMSE vs CFD (Layout A): Larsen **0.212 MW**, Blended
  0.251, Jensen 0.309, Gaussian 0.390, FLORIS-Jensen 0.697,
  FLORIS-Gaussian 0.767. Farm total: Larsen +0.3%, Jensen +0.3%,
  Gaussian +2.2%, Blended −6.4%, FLORIS +9.4 / +11.4%.
- P2-3 farm-scale RMSE (Layout B): Larsen **0.284 MW**, Jensen 0.547,
  Gaussian 0.823, FLORIS-Jensen 1.173, FLORIS-Gaussian 1.294. Farm efficiency
  `eta`: CFD 43.0%, Larsen 43.6%, Jensen 49.4%, Gaussian 55.8%, FLORIS 60–62%.
- Cost, 3-turbine power evaluation: LOTUSim-Jensen 0.053 ms, Gaussian 0.055 ms,
  Blended 0.25 ms, Larsen 0.60 ms, against FLORIS-Jensen 2.8 ms and
  FLORIS-Gaussian 6.6 ms — **11–125× faster** than FLORIS-Gaussian.

---

## 6. Known gaps and caveats

- **`blended_multispeed_validation.png` needs CFD data that is not shipped.**
  The 8 m/s and 12 m/s single-turbine sensitivity runs are not in this
  repository; only the 10 m/s case is, as `data/layout_single/`. The script
  detects this, prints which speeds are missing, and **leaves the committed
  figure untouched** rather than writing a partial one over it. To regenerate
  it in full, request those two cases and extract them to:

  ```
  benchmark/data/layout_single_8ms/wakeProfiles/<timestep>/wake_*_U.csv
  benchmark/data/layout_single_12ms/wakeProfiles/<timestep>/wake_*_U.csv
  ```

  No code change is needed — the paths are already wired up. The same missing
  cases underlie the multi-speed and hold-out rows of the P1-3 generalisation
  table.
- **The reference is steady k–ε RANS**, which is known to under-predict
  far-wake recovery by roughly 10–15% relative to LES. The CFD wake here is
  correspondingly persistent (`U/U_inf ≈ 0.71` still at 14D), and *every*
  engineering model over-predicts recovery against it. The comparison is
  internally consistent, but the absolute far-wake ranking should be read with
  that bias in mind.
- **P2-4 has CFD only at 0° and 15°.** Intermediate angles are compared against
  FLORIS, so the directional conclusions are correspondingly weaker.
- **Layout B shows a ~6% (LOTUSim) / ~14% (FLORIS) upstream-row discrepancy**
  before any wake interaction, larger than Layout A's 2.2%, likely from domain
  size and mutual blockage. It propagates into every downstream row and limits
  strict cross-layout comparison of absolute power.
- **`P2-3` evaluates LOTUSim-Blended for reference only**, using its
  spatial-field method without the local-TI rescaling and wake-meandering
  corrections the power candidates use. Its −20% farm total there demonstrates
  what omitting those corrections costs; it is not a claim about Blended as a
  power model.
- **The P2-2 and P2-4 CFD reference points are hardcoded** in the scripts
  (`cfd_data` at 6, 8, 10, 12, 14 m/s; the 0°/15° direction spot checks). Unlike
  the P2-1 and P2-3 values, the underlying CFD runs for the off-baseline speeds
  and the 15° yaw case are **not** in `data/`, so those two references cannot be
  re-derived from this repository.
- `wind_direction_normalised.py` plots tabulated P2-4 outputs rather than
  recomputing them. Rerun `models/extended_models.py` to regenerate the numbers
  themselves; the two agree to the printed precision.
- Single turbine, single turbulence intensity, neutral stability only.

---

## 7. Legacy / superseded scripts

Kept for provenance; **not** part of the current protocol set, and none of
their output is reported. They have been repointed at `data/` and
now run against this repository, writing to `figures/supplementary/`.

| Script | Reads | Writes (into `figures/supplementary/`) | Superseded by |
|--------|-------|----------------------------------------|---------------|
| `centreline_plot.py` | `data/layout_single` | `wake_centreline_decay_legacy.png` | P1-1 section of `gen_all_figures.py` |
| `wake_shape_comparison.py` | `data/layout_single` | `wake_shape_1t_*.png` | P1-2 section of `gen_all_figures.py` |
| `wake_shape_comparison_4x4.py` | `data/layout_b` | `wake_profiles_4x4.png`, `wake_shape_4x4_*.png` | — (P1-2 Layout B; figure not used in the reported results) |
| `plot_comparison.py` | `data/layout_a/turbines` | `wake_model_comparison.png` | P2-1 section of `gen_all_figures.py` |

`wake_shape_comparison.py` also carried a syntax error (a `suptitle` call whose
opening line alone had been commented out, leaving its arguments dangling), so
it had never been runnable; that is fixed.

Note that `wake_profiles_4x4.png` exists in **both** `figures/main/` and
`figures/supplementary/`, and the two are different renders. The
`supplementary/` one is what `wake_shape_comparison_4x4.py` produces today; the
`main/` one is an older copy, kept because it is the version the shortened
results section refers to. Neither appears in the current figure set.

All generated images now live under `figures/`: `figures/main/` for the
primary figures and `figures/supplementary/` for everything else. The loose
duplicates that used to sit at the top of `benchmark/` and in the repository
root — left over from when these scripts wrote into their working directory —
have been removed.
