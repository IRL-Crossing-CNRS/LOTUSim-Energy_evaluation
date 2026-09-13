# LOTUSim-Energy evaluation

A UAV inspecting an offshore wind turbine flies inside its neighbours' wakes,
and what that wake does to it is not what a uniform wind field predicts. Sea
trials for offshore robot teams are expensive, weather-limited and hard to
repeat, so simulation is the practical route to validating an autonomy stack
before it is trusted offshore — which only means something if the simulator
reproduces the disturbance the vehicles will actually meet.

**LOTUSim-Energy** extends LOTUSim with an atmosphere that has wake structure in
it, and an offshore O&M layer that works in it. These are its features and the
evidence each one has:

| Feature | What it is | Evaluated in |
|---|---|---|
| **Wake-resolved atmosphere** | four engineering wake models with shear, rotor-disk averaging and wake-added turbulence, plus one targeting point-wise velocity accuracy over rotor-averaged power | [`wake-models/`](wake-models/) — seven protocols against actuator-line CFD and FLORIS v4 |
| **Aerial navigation inside the wake** | the field acts on a PX4 multirotor, graded across the cone and carrying a turbulent gust | [`aerial-navigation/`](aerial-navigation/) — 160 paired flights, three missions, eight gust seeds |
| **Propulsion-coupled energy model** | shaft power from the vehicle's own rotor constants, integrated over a mission | [`aerial-navigation/`](aerial-navigation/) — reported per leg alongside the disturbance |
| **Corrosion and crack detection** | on turbine towers and blades, from the vehicle's camera stream over ROS 2 during a mission: corrosion by colour thresholding, cracks by a YOLO segmentation model | [`inspection-perception/`](inspection-perception/) — the crack detector scored on six real and rendered image sets; the corrosion threshold is implemented, not characterised |
| **Farm scene and mission primitives** | the wind-farm world, and a task library restricted to primitives the missions exercise | exercised by the campaign above; not separately evidenced |

---

## 1 · Wake-resolved atmosphere

→ [`wake-models/`](wake-models/)

**What it has to get right.** Two things at once, which one model does not do
equally well. Predicting farm yield needs an accurate rotor-averaged inflow.
Flying a robot through the farm needs an accurate *velocity field* — in
particular its lateral gradient at the wake edge, where the vehicle is pushed
sideways. Compensating errors in the field cancel once it is integrated into a
power, so a model can pass one test and fail the other. And both have to run
inside a live simulation, at a cost that leaves room for the rest of the stack.

**How it is tested.** Seven protocols in two phases kept deliberately apart: P1
scores the wake field in dimensionless terms, P2 scores power in MW. The
reference is an actuator-line CFD case (OpenFOAM v8 + turbinesFoam, k–ε RANS);
FLORIS v4 is the comparison baseline. NREL 5 MW turbine, a 3-turbine row at 7D,
a 16-turbine 4×4 grid, an isolated turbine, and a 5D-spacing case held out of
calibration.

**What came out.** No single model wins both jobs — the result, not an omission.

| Model | Per-turbine power RMSE | Farm total | Time per evaluation |
|---|---:|---:|---:|
| **Larsen** | **0.212 MW** | +0.3 % | 0.60 ms |
| Blended | 0.251 MW | −6.4 % | 0.25 ms |
| Jensen | 0.309 MW | +0.3 % | 0.053 ms |
| Gaussian | 0.390 MW | +2.2 % | 0.055 ms |
| FLORIS-Jensen | 0.697 MW | +11.4 % | 2.8 ms |
| FLORIS-Gaussian | 0.767 MW | +9.4 % | 6.6 ms |

At the wake edge (r = 0.5D, 1D–14D) the ranking inverts: **Blended** holds a
17.5 % mean gradient error on an isolated turbine and 26.1 % on the independent
5D hold-out, against 42.6 % for Larsen and 70.1 % for Gaussian.

So the simulator runs both: **Larsen drives the energy estimate, Blended supplies
the field the vehicles fly through** — 11–125× faster than FLORIS-Gaussian, and
a full 100×100 hub-height field in 4.5 ms, which is what makes it usable in the
loop rather than as a precomputed layer.

```bash
cd wake-models && pip install -r requirements.txt
python3 test_models.py               # ~30 s: reproduces the P2-1 baseline powers
python3 benchmark/gen_all_figures.py # the figures and the P1/P2 tables
```

---

## 2 · Aerial navigation and energy inside the wake

→ [`aerial-navigation/`](aerial-navigation/)

**What it has to get right.** That the field actually reaches the vehicle, and
that its effect is separable. A wake changes the mean wind *and* its turbulence
at once, and those act on different things — so a wake-versus-clear-air
comparison measures only their sum and can come out looking like nothing
happened. Resolving the wake is only worth the cost if the two mechanisms can be
told apart and behave differently.

**How it is tested.** An X500 quadrotor under PX4 SITL flies a 16-turbine farm
in a 2×2 factorial that breaks the confound: uniform or wake-resolved mean
field, crossed with ambient or wake turbulence. Three missions — a transit along
a turbine column, an orbit of one rotor, a crossing at x/D = 2…8 — eight gust
seeds per cell, **160 flights, all passing pre-analysis checks**. Contrasts are
paired within seed, so the gust realisation cancels. Variance is taken about the
deterministic tilt profile along track, so it counts disturbance rather than
steady drift with position.

**What came out. Two independent mechanisms.**

| | Δ variance (deg²) | | Δ mean tilt (deg) | | Δ power (W) | |
|---|---:|---:|---:|---:|---:|---:|
| | turb. | deficit | turb. | deficit | turb. | deficit |
| **M1 transit** | **+7.83** | −0.95 | +0.28 | **+7.31** | n.s. | **+9.31** |
| M3, x/D = 2 | +0.84 | n.s. | n.s. | −0.20 | n.s. | −1.29 |
| M3, x/D = 8 | +0.77 | n.s. | n.s. | −0.30 | n.s. | −1.37 |
| M2 orbit, R = 60 m | *unresolved* | | n.s. | −1.35 | n.s. | −6.38 |

**The deficit changes how hard the vehicle works; the turbulence changes how
much it is shaken.** On the transit the deficit adds mean tilt and power in all
eight seeds while variance *falls* — slower air is calmer air. Turbulence does
the reverse: variance in all eight seeds, power untouched. The four crossing
stations show the same split.

That is the case for resolving the wake at all. **A uniform field set to the
wake's mean speed reproduces the energy cost and predicts zero disturbance,
where the real wake roughly triples attitude variance** — battery sizing would
come out right, station-keeping would not.

Three consequences for planning:

- **Heading reverses the deficit's sign.** The wake slows the air either way, so
  it costs tilt and power flown downwind and *saves* them flown upwind
  (−4.54°, −35.9 W, all eight seeds). Turbulence has no such structure: it
  raises variance on every heading. A route can be traded against the mean
  field; the turbulent cost cannot be routed around.
- **Margin is spent upwind.** The upwind leg already runs at 42.2° against a 44°
  limit; wake turbulence holds it at that limit for 18.2 points more of the leg,
  and peak rotor demand rises 4× more than downwind (0.033 against 0.008).
  Flight direction becomes a scheduling decision.
- **The disturbance is a property of place.** Sorted by lateral distance from
  the wake axis it is ~8× the clear-air value inside the wake (4.9 against
  0.6 deg²) and back to baseline at 0.97 D, matching the model's wake edge — so
  roughly one rotor diameter is a usable planning standoff.

```bash
export LOTUSIM_WS=$HOME/lotusim_ws
export LOTUSIM_SCENARIO_WS=$HOME/path/to/LOTUSim-generic-scenario
source aerial-navigation/env.sh
python3 aerial-navigation/scripts/factorial_analysis.py --metric resvar
```

The 160 ULogs are not shipped, but every reported contrast is stored with its
bootstrap interval, seed count and source manifest in
[`factorial_results.json`](aerial-navigation/results/factorial_results.json), so
no number needs a re-flight to be read.

---

## 3 · Corrosion and crack detection on turbine structures

→ [`inspection-perception/`](inspection-perception/)

**What the layer is.** The farm scene, a task library restricted to primitives
the missions actually exercise, a propulsion-coupled energy model, and an
inspection pipeline that looks for the two defects an offshore turbine is
inspected for. As the vehicle flies a tower, its camera stream feeds two
detectors over ROS 2 for the duration of the mission — **corrosion**, by colour
thresholding in HSV and CIELab, and **cracks**, by a YOLO segmentation model —
each publishing labelled detections the mission can act on. Inspection is a
closed loop in the simulation, not a rendering exercise.

**What it has to get right, and what is evidenced.** Composing Gazebo, Unity,
xdyn, ROS 2 and a public detector checkpoint is engineering, and nothing
scientific is claimed for it. Neither detector is a contribution. What needed
measuring is whether a detector behaves on rendered imagery as it would on the
real thing — because if it does not, everything tuned in simulation has to be
re-tuned in the field, and the layer has not saved the work it exists to save.

So the crack detector is used here as a **fixed instrument for measuring the
rendered imagery**, not as a capability being claimed. The corrosion threshold
is reported as implemented and is not characterised; the mission primitives and
the energy model are exercised by the campaign in section 2 and not separately
evidenced.

**How it is tested.** The crack model — YOLO11n-seg, trained only on real
photographs — is scored on six sets: real and rendered, in air and underwater, each
underwater pair with and without CLAHE contrast correction. The rendered set is
read at the *same* confidence threshold as its real counterpart — re-tuning per
set would compare two operating points and hide the gap being measured.

**What came out.** The aerial rendering transfers; the underwater rendering
transfers only until you preprocess it.

| Comparison | Real | Rendered | Gap |
|---|---:|---:|---:|
| Aerial | 0.771 | 0.688 | **0.083** |
| Underwater, raw | 0.272 | 0.233 | **0.039** |
| Underwater, CLAHE | 0.619 | 0.264 | **0.355** |

CLAHE more than doubles AP50 on real underwater imagery (0.272 → 0.619, recall
0.222 → 0.519) and moves rendered underwater imagery almost not at all (0.233 →
0.264), while costing it precision. **The renderer degrades its underwater views,
but not by the physical mechanism a histogram equaliser inverts** — so an
underwater preprocessing chain tuned in simulation should not be trusted in the
water. The aerial case carries a good deal better.

```bash
cd inspection-perception
python3 precision_recall.py aerial   # seconds, no GPU, no model load
```

---

## Reproducing any of it

Features 1 and 3 are evaluated end-to-end on a laptop. Only the flight campaign
needs a simulator, and its results are committed so its numbers can be re-derived
without re-flying anything.

| You are running | From | It needs |
|---|---|---|
| `wake-models/*.py`, `wake-models/benchmark/*.py` | anywhere | numpy, matplotlib, FLORIS |
| `inspection-perception/*.py` | `inspection-perception/` | ultralytics + a GPU, except `precision_recall.py` |
| `aerial-navigation/scripts/*` | anywhere, after `source aerial-navigation/env.sh` | both simulator workspaces, PX4 SITL |

Nothing has to be copied anywhere: scripts resolve paths from their own
location, and flight scenarios are handed to the launcher by absolute path.
Each study's README says what every script does and which table its output
feeds.

## The simulator

This repository evaluates three open-source repositories; it does not contain
them.

| Repository | Role |
|---|---|
| [LOTUSim](https://github.com/IRL-Crossing-CNRS/LOTUSim) | simulation core: Gazebo worlds, plugins, message interfaces |
| [LOTUSim-generic-scenario](https://github.com/IRL-Crossing-CNRS/LOTUSim-generic-scenario) | scenario layer: launcher, agent SDK, mission primitives |
| [LOTUSim-Energy](https://github.com/IRL-Crossing-CNRS/LOTUSim-Energy) | Unity renderer for the offshore farm |

[`docs/upstream-requirements.md`](docs/upstream-requirements.md) says which of
them each evaluation needs, which version, and how to check a workspace has it
— worth reading before a campaign, because a workspace missing the graded wake
field runs the scenarios **without error** and produces a different one.

## Where the data comes from

- **CFD reference** (`wake-models/benchmark/data/`) — extracted from the
  OpenFOAM cases described in
  [`wake-models/docs/cfd_setup.md`](wake-models/docs/cfd_setup.md). The full case
  directories are several GB and are not shipped.
- **Flight logs** — not shipped. The manifests record which run directory held
  which condition and seed.
- **Crack imagery** — the real photographs are the public `crack-bphdr` dataset
  (Roboflow Universe, Public Domain); the rendered sets come from
  LOTUSim-Energy's Unity project, with the aerial set's per-frame capture
  settings in `capture_log.csv`.

About 560 MB checked out, no Git LFS: a plain `git clone` gets everything.

## Licence

[EPL-2.0](LICENSE), matching LOTUSim. The crack datasets keep their own
(Public Domain).
