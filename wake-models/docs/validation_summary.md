# Validation summary

Per-protocol results for the seven-protocol benchmark (P1-1..P1-3, P2-1..P2-4)
Protocol definitions, what each
one measures physically, and the script that reproduces it are in
[../benchmark/README.md](../benchmark/README.md).

**These numbers are a point-in-time snapshot.** They were last confirmed by a
clean run of `python3 benchmark/gen_all_figures.py` and
`python3 benchmark/wind_speed_sweep.py` against the CFD extracts in
`benchmark/data/`. If anything in `models/` or `benchmark/` changes, rerun
before citing them.

**Do not merge P1 and P2 results.** They use different metrics, different units,
and answer different questions.

## Benchmark setup

- Turbine: NREL 5 MW, D = 126 m, hub height 90 m, rated 5 MW
- Baseline: U_inf = 10 m/s, TI = 8%, alpha = 0.12, rho = 1.225 kg/m3,
  Ct = 0.75, Cp = 0.498, aligned inflow, neutral stability
- CFD reference: OpenFOAM v8 + turbinesFoam actuator line, k-epsilon RANS
- Engineering baseline: FLORIS v4, Gaussian and Jensen velocity models

---

## Phase P1 — wake field (dimensionless)

### P1-1 Centreline velocity decay — isolated single turbine (`data/layout_single`)

Normalised centreline velocity U/U_inf at hub height.

Position         | 1D    | 3D    | 5D    | 7D    | 10D   | 14D   | RMSE
-----------------|-------|-------|-------|-------|-------|-------|-------
CFD reference    | 0.735 | 0.708 | 0.707 | 0.708 | 0.711 | 0.716 | -
LOTUSim-Jensen   | 0.556 | 0.640 | 0.704 | 0.752 | 0.805 | 0.852 | 0.105 best
LOTUSim-Larsen   | 0.490 | 0.650 | 0.720 | 0.730 | 0.810 | 0.850 | 0.124
FLORIS-Jensen    | 0.558 | 0.684 | 0.762 | 0.815 | 0.866 | 0.907 | 0.133
FLORIS-Gaussian  | 0.258 | 0.429 | 0.610 | 0.738 | 0.834 | 0.896 | 0.246
LOTUSim-Gaussian | ~0*   | 0.321 | 0.624 | 0.747 | 0.840 | 0.900 | 0.353
LOTUSim-Blended  |       |       |       |       |       |       | 0.020 in-sample

\* Gaussian near-wake singularity: the eps = 0.22 initialisation exceeds the
validity bound of the Bastankhah deficit formula below ~2D.

The CFD wake is persistent (U/U_inf ~ 0.71 still at 14D). Every engineering
model over-predicts far-wake recovery against it, reaching 0.85-0.91 at 14D.
Blended is calibrated on this dataset, so its RMSE is an in-sample fit and is
excluded from the ranking.

### P1-1 Layout B extension — centreline through the 4x4 farm (`data/layout_b`)

Position | CFD  | Jensen      | Blended     | Larsen      | Gaussian     | FLORIS-G     | FLORIS-J
---------|------|-------------|-------------|-------------|--------------|--------------|-------------
R2 (9D)  | 0.60 | 0.40 (-34%) | 0.73 (+22%) | 0.83 (+38%) | 0.82 (+37%)  | 0.81 (+35%)  | 0.85 (+43%)
R3 (18D) | 0.37 | 0.35 (-5%)  | 0.55 (+49%) | 0.71 (+92%) | 0.76 (+105%) | 0.84 (+128%) | 0.84 (+127%)
R4 (27D) | 0.30 | 0.32 (+8%)  | 0.41 (+37%) | 0.62 (+107%)| 0.73 (+143%) | 0.84 (+182%) | 0.83 (+177%)

Mean absolute error: Jensen 15.7%, Blended 36%, Larsen 79%, Gaussian 95%,
FLORIS-G 115%, FLORIS-J 116%. This is an independent test for Blended.

### P1-2 Lateral profile RMSE — isolated single turbine

Average RMSE of the normalised lateral velocity profile across +/-2.5D,
over all six downstream positions (1D-14D).

Model            | Avg RMSE | Note
-----------------|----------|------------------------------------------
LOTUSim-Blended  | 0.0245   | lowest, but in-sample (calibration set)
LOTUSim-Larsen   | 0.0353   |
LOTUSim-Jensen   | 0.0506   |
FLORIS-Jensen    | 0.0867   |
FLORIS-Gaussian  | 0.0940   |
LOTUSim-Gaussian | 0.1640   | near-wake singularity inflates the error

### P1-3 Wake-edge lateral velocity gradient

Normalised lateral gradient abs(dU/dy) / U_inf at r = 0.5D; mean absolute
error versus CFD over 1D-14D.

Case                                   | Blended | Larsen | Gaussian | FLORIS-G
---------------------------------------|---------|--------|----------|---------
Isolated turbine (`layout_single`)      | 17.5%   | 43.1%  | 76.5%    | 68.3%
3-turbine 5D hold-out (`layout_5D`)     | 26.1%   | 42.6%  | 70.1%    | -

The 5D case is a genuine hold-out, never used to tune any model. That Blended's
hold-out error stays close to its single-turbine error, rather than degrading,
indicates the calibration generalises rather than over-fits.

Raw gradient values, isolated turbine (1D, 3D, 5D, 7D, 10D, 14D):

- CFD:              0.00343, 0.00374, 0.00373, 0.00370, 0.00367, 0.00369
- LOTUSim-Blended:  0.00338, 0.00316, 0.00305, 0.00294, 0.00282, 0.00272
- LOTUSim-Larsen:   0.00338, 0.00254, 0.00207, 0.00176, 0.00146, 0.00120
- LOTUSim-Gaussian: 0.00889, 0.00780, 0.00430, 0.00262, 0.00135, 0.00063
- FLORIS-Gaussian:  0.00842, 0.00659, 0.00440, 0.00271, 0.00142, 0.00067

The CFD gradient is near-constant across the domain; Blended is the only model
that reproduces that shape, modestly under-predicting the magnitude. Both
Jensen formulations are structurally excluded: a top-hat profile has no defined
gradient at a fixed radius.

### P1-3 generalisation tests (LOTUSim-Blended)

Test                                    | Condition                              | Max error
----------------------------------------|----------------------------------------|----------
Multi-speed validation                  | U_inf = 8 m/s (Ct = 0.80)              | 15.3%
Multi-speed validation                  | U_inf = 12 m/s (Ct = 0.53)             | 19.4%
Hold-out cross-validation               | sigma calibrated at 8 and 12, applied at 10 | 3.9%
Farm compounding, 5D, freestream superposition  | 7D-14D                         | ~65%
Farm compounding, 5D, sequential superposition  | 7D-14D                         | 14-15%

The superposition result matters for LCOE: computing each deficit against the
undisturbed inflow rather than the locally reduced inflow is a factor-of-four
error at farm scale, not an implementation detail.

**Reproduction caveat:** the 8 and 12 m/s single-turbine CFD cases are not in
this repository, so the multi-speed rows above cannot be regenerated from
`benchmark/data/`. The figure script detects this and leaves the committed
figure untouched rather than writing a partial one. Extracts dropped into
`benchmark/data/layout_single_8ms/` and `layout_single_12ms/` regenerate it with
no code change. See `benchmark/README.md` §6.

---

## Phase P2 — power (MW and %)

### P2-1 Baseline power — Layout A, 3-turbine row, 7D, U_inf = 10 m/s

Fixed Cp = 0.498. Farm total = sum over T1-T3; RMSE across T1-T3.

Model            | T1 MW | T2 MW | T3 MW | Farm total MW | RMSE MW
-----------------|-------|-------|-------|---------------|--------
CFD (OpenFOAM)   | 3.806 | 1.904 | 0.560 | 6.270         | -
LOTUSim-Larsen   | 3.724 | 1.707 | 0.858 | 6.289 (+0.3%) | 0.212 best
LOTUSim-Blended  | 3.803 | 1.470 | 0.595 | 5.868 (-6.4%) | 0.251
LOTUSim-Jensen   | 3.724 | 1.583 | 0.981 | 6.288 (+0.3%) | 0.309
LOTUSim-Gaussian | 3.724 | 1.552 | 1.130 | 6.406 (+2.2%) | 0.390
FLORIS Jensen    | 3.418 | 1.863 | 1.702 | 6.983 (+11.4%)| 0.697
FLORIS Gaussian  | 3.418 | 1.638 | 1.803 | 6.859 (+9.4%) | 0.767

T3 (two accumulated wakes) is the decisive comparison: CFD records an 85.3%
wake loss; both FLORIS configurations more than triple the CFD power there,
failing to compound the 7D and 14D wakes.

Blended's accurate T3 (+6.3%) is outweighed by a systematic T2 under-prediction
(-22.8%), because the wide calibrated lateral sigma that reproduces the P1-3
gradient also over-spreads the deficit at a collinear downstream turbine. T2
carries the largest recoverable power in an aligned farm, so its -6.4% farm
total rules Blended out for power prediction.

### P2-2 Wind-speed sweep — Layout A, real NREL 5 MW Cp/Ct curves

U_inf swept 5-15 m/s; CFD at 6, 8, 10, 12, 14 m/s. Normalised RMSE on P/P_T1
cancels absolute Cp discrepancies above rated to isolate wake interaction.

Values below are the "<= rated" column of `wind_speed_sweep.py` output
(5-11 m/s, where Ct is high and wake interaction dominates); the script also
prints an all-speeds column, which ranks slightly differently.

Model            | P2-2 normalised RMSE | P2-1 absolute RMSE (MW)
-----------------|----------------------|------------------------
LOTUSim-Larsen   | 0.1591 best          | 0.212
FLORIS Gaussian  | 0.1768               | 0.767
FLORIS Jensen    | 0.1952               | 0.697
LOTUSim-Gaussian | 0.2265               | 0.390
LOTUSim-Jensen   | 0.3299               | 0.309
LOTUSim-Blended  | 0.4076               | 0.251

The two columns rank differently, which is the point of reporting both: a
normalised wake-interaction metric and an absolute power metric are not
interchangeable.

### P2-3 Farm scale — Layout B, 16 turbines, 7D lateral x 9D row

Model            | R1 MW | R2 MW | R3 MW | R4 MW | Farm total MW | RMSE MW
-----------------|-------|-------|-------|-------|---------------|--------
CFD (OpenFOAM)   | 3.964 | 2.046 | 0.619 | 0.190 | 27.272        | -
LOTUSim-Larsen   | 3.724 | 1.767 | 0.918 | 0.503 | 27.645 (+1.4%)| 0.284 best
LOTUSim-Jensen   | 3.724 | 1.829 | 1.266 | 1.009 | 31.309        | 0.547
LOTUSim-Gaussian | 3.724 | 2.023 | 1.635 | 1.460 | 35.367        | 0.823
FLORIS Jensen    | 3.418 | 2.138 | 2.022 | 1.986 | 38.254        | 1.173
FLORIS Gaussian  | 3.418 | 2.004 | 2.164 | 2.191 | 39.108        | 1.294
LOTUSim-Blended* | 3.635 | 1.308 | 0.429 | 0.077 | 21.796 (-20%) | 0.419

\* Reference only: evaluated with the simplified sequential spatial-field
method, without the local-TI rescaling and wake-meandering corrections the
power candidates use. Included to show what omitting those corrections costs at
farm scale, not as a power-prediction candidate.

Farm efficiency eta = P_farm / (N_T * P_R1,CFD): CFD 43.0%, LOTUSim-Larsen
43.6%, LOTUSim-Jensen 49.4%, LOTUSim-Gaussian 55.8%, both FLORIS 60-62%.
FLORIS overstates farm output by 40-43%, which is unacceptable for LCOE.

Caveats: all models miss the CFD R1 by 6% (LOTUSim) / 14% (FLORIS) before any
wake interaction, larger than Layout A's 2.2%, likely from domain size and
mutual blockage; this propagates downstream. At R3-R4 every candidate
over-predicts sharply (Larsen +48% / +165%, FLORIS +250% / +1053%) as linear
superposition breaks down where U/U_inf falls to 0.30-0.37. Larsen's good farm
total is partly fortuitous cancellation of R1/R2 under- against R3/R4
over-prediction.

### P2-4 Wind-direction sweep — Layout A

Normalised T2 power P_T2 / P_T1,0deg. CFD spot checks at 0 and 15 deg only;
intermediate angles are compared against FLORIS, so conclusions here are
correspondingly weaker.

Offset | CFD   | Jensen | Gaussian | Larsen | Blended | FLORIS-G | FLORIS-J
-------|-------|--------|----------|--------|---------|----------|---------
0 deg  | 0.500 | 0.425  | 0.417    | 0.458  | 0.386   | 0.440    | 0.500
5 deg  | -     | 0.645  | 0.529    | 0.794  | 0.746   | 0.663    | 0.641
10 deg | -     | 1.000  | 0.771    | 1.000  | 0.992   | 0.889    | 0.918
15 deg | 0.900 | 1.000  | 0.941    | 1.000  | 1.000   | 0.917    | 0.918
20 deg | -     | 1.000  | 0.991    | 1.000  | 1.000   | 0.918    | 0.918
30 deg | -     | 1.000  | 1.000    | 1.000  | 1.000   | 0.918    | 0.918
45 deg | -     | 1.000  | 1.000    | 1.000  | 1.000   | 0.918    | 0.918

Jensen and Larsen (hard wake boundary) jump to full recovery at 10 deg, which
is unphysical for a 126 m rotor only partly clearing the wake, and over-predict
recovery by 11% at the 15 deg CFD check. Gaussian recovers smoothly and is
closest there (+4.6%). FLORIS plateaus at 0.918, an artefact of its overlap
term not decaying to zero. The best aligned-inflow model is therefore not the
best for directional response.

---

## Computational performance

Machine: 12th Gen Intel Core i7-12650H. Model / FlorisModel construction
excluded; value = fastest of N evaluations (N = 2000 LOTUSim, N = 300 FLORIS
for power; N = 200 Blended, N = 40 FLORIS for fields). Script:
`benchmark/compute_time_benchmark.py`. Results are machine-dependent.

### Power prediction, 3-turbine Layout A

Model            | time (ms) | x vs FLORIS-Gaussian | x vs FLORIS-Jensen
-----------------|-----------|----------------------|-------------------
FLORIS-Gaussian  | 6.6       | 1.0                  | 0.42
FLORIS-Jensen    | 2.8       | 2.4                  | 1.0
LOTUSim-Jensen   | 0.053     | 125                  | 53
LOTUSim-Gaussian | 0.055     | 120                  | 51
LOTUSim-Blended  | 0.25      | 26                   | 11
LOTUSim-Larsen   | 0.60      | 11                   | 4.7

FLORIS-Gaussian, the primary configuration, is ~2.4x slower than FLORIS-Jensen.

### Spatial-field generation (FLORIS-Jensen excluded)

Hub-height velocity field over the 3-turbine row and its wake (downstream
-100..2400 m, lateral -400..400 m). "NxN" is the query-point count, not metres.
FLORIS via `sample_flow_at_points`.

Grid    | points | Blended | FLORIS-Gaussian     | speedup
--------|--------|---------|---------------------|--------
100x100 | 10,000 | 4.5 ms  | ~150 ms             | ~33x
200x200 | 40,000 | 20 ms   | ~150 ms             | ~7.5x

FLORIS field cost is dominated by a fixed solve overhead (measured 100-230 ms)
and is therefore roughly independent of grid size. FLORIS-Jensen is excluded
because a top-hat profile has no defined lateral gradient.

> An earlier revision of this file quoted a single "FLORIS 5.043 ms" figure and
> LOTUSim timings of 0.042/0.049/0.360/0.770 ms. Those did not distinguish the
> two FLORIS configurations and have been superseded by the table above.

---

## Overall

- **LOTUSim-Larsen** for power, AEP and LCOE (P2-1, P2-2, P2-3).
- **LOTUSim-Blended** for the resolved velocity field and wake-edge gradient
  (P1-2, P1-3), including drone hazard mapping and path planning.

No single model wins both phases. That is the benchmark's finding, and both
models are integrated side by side into LOTUSim-Energy.

Bounding conditions: steady RANS reference (under-predicts far-wake recovery by
~10-15% relative to LES), a single turbine type, a single turbulence intensity,
and neutral stability only.
