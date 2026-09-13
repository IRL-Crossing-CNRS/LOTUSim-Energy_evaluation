# CFD Reference Simulations

## Overview

High-fidelity CFD simulations were used as the validation reference for all
engineering wake models. Simulations were run using OpenFOAM v8 with the
turbinesFoam actuator line method.

## Software

- OpenFOAM v8 (openfoam.org)
- turbinesFoam (actuator line turbine modelling)
- Docker image: `ellaj03/openfoam8-turbinesfoam` (pre-built, includes turbinesFoam)

## Turbine

- NREL 5MW reference turbine
- Rotor diameter: 126 m, Hub height: 90 m
- Ct = 0.75, Cp = 0.498 at U = 10 m/s

## Simulation Cases

Four CFD cases underpin the benchmark. Protocol IDs below follow the benchmark
numbering (P1-1..P1-3, P2-1..P2-4); see `benchmark/README.md`.

### Isolated single turbine
- 1 x NREL 5MW turbine, U = 10 m/s
- The **primary wake-phase reference**: free of the downstream-rotor induction
  that contaminates Layout A's far probes (at 7D-14D the two differ by
  U/U_inf = 0.71 against 0.34-0.56)
- Used for P1-1, P1-2, and the single-turbine panel of P1-3
- Extract: `benchmark/data/layout_single/`
- Sensitivity runs at 8 and 12 m/s exist but are **not** included in this
  repository (see "CFD Data Availability" below). The benchmark scripts already
  look for them at `benchmark/data/layout_single_8ms/` and
  `benchmark/data/layout_single_12ms/`; drop the extracts there and the P1-3
  multi-speed figure regenerates with no code change.

### Layout A - 3-turbine inline row
- 3 x NREL 5MW turbines
- Spacing: 7D = 882 m hub-to-hub
- Used for P2-1, P2-2, P2-4 (power) and the 7D calibration panel of P1-3
- Case directory: `offshore3turbine/nrel5mw_3turbines/`
- Extract: `benchmark/data/layout_a/`

### Layout A at 5D spacing - independent hold-out
- 3 x NREL 5MW turbines, 5D = 630 m spacing
- Never used to calibrate any model; the hold-out panel of P1-3
- Extract: `benchmark/data/layout_5D/`

### Layout B - 4x4 grid
- 16 x NREL 5MW turbines
- Spacing: 7D = 882 m hub-to-hub laterally (between columns), 9D = 1134 m
  hub-to-hub longitudinally (between rows, downstream direction)
- Row spacing was increased from 7D to 9D after preliminary runs at uniform
  7D produced severe cumulative velocity deficits in the downstream rows and
  unstable actuator-line simulations; lateral spacing was kept at 7D,
  matching Layout A. A 5D lateral buffer from the edge turbines to the domain
  boundary was confirmed to remove wall effects on the edge column.
- Used for P2-3 (power, multi-row) and the Layout B extensions of P1-1 / P1-2
- Case directory: `offshore4x4/nrel5mw_4x4_9D/`
- Extract: `benchmark/data/layout_b/`

## Boundary Conditions

- Inflow: U = 10 m/s (primary), 8 and 12 m/s (sensitivity)
- Turbulence intensity: TI = 8%
- Wind shear exponent: alpha = 0.12 (power law)
- Air density: rho = 1.225 kg/m3
- Surface roughness: z0 = 0.0002 m (offshore, Charnock 1955)
- ABL profile: neutral stability

## Turbulence Model

- k-epsilon RANS
- Neutral offshore atmospheric boundary layer
- Inlet k and epsilon derived from TI and mixing length

## Numerical Parameters

These are the values reported for the benchmark configuration.

Parameter               | Layout A              | Layout B
------------------------|-----------------------|----------------------
Solver                  | `pimpleFoam`          | `pimpleFoam`
Turbulence model        | k-epsilon RANS        | k-epsilon RANS
Turbine method          | Actuator line         | Actuator line
Mesh resolution         | 150 x 40 x 40         | 130 x 70 x 30
Total cells             | 240,000               | 273,000
Domain x (downstream)   | -300 to 2200 m        | -300 to 4200 m
Domain y (lateral)      | +/-315 m              | +/-2100 m
Domain z (height)       | 0-400 m               | 0-400 m
Time step dt            | 0.5 s                 | 0.5 s
Simulation duration     | 800 s                 | 1200 s
Result averaging        | final 20 timesteps    | final 20 timesteps
Tip-speed ratio         | 7.55                  | 7.55

The benchmark scripts average over the statistically converged tail — the last
20 dumps, or the second half of the run when fewer are available — so that the
start-up transient is excluded. Layout A ships 18 dumps, so a blind last-20
window would fold the transient in.

## Running the Simulations

### Prerequisites

A pre-built Docker image with OpenFOAM v8 and turbinesFoam is available on
Docker Hub. This is the environment used for all validation simulations in
this project.

```bash
docker pull ellaj03/openfoam8-turbinesfoam
```

### Starting the container

```bash
docker run -it --user root \
  -v /path/to/cases:/home/openfoam/cases \
  ellaj03/openfoam8-turbinesfoam bash

# Inside the container — run once per session:
source /opt/openfoam8/etc/bashrc
```

Replace `/path/to/cases` with the local directory containing your OpenFOAM
case folders. Files written there will persist after the container exits.

### Layout A

```bash
cd /home/openfoam/cases/offshore3turbine/nrel5mw_3turbines/
blockMesh
decomposePar -force
mpirun -np 8 pimpleFoam -parallel 2>&1 | tee log.pimpleFoam
reconstructPar
```

### Layout B

```bash
cd /home/openfoam/cases/offshore4x4/nrel5mw_4x4_9D/
blockMesh
decomposePar -force
mpirun -np 16 pimpleFoam -parallel 2>&1 | tee log.pimpleFoam
reconstructPar
```

## Post-Processing

Wake velocity profiles were extracted using OpenFOAM `sample` utility along
centreline probes at x/D = 1, 3, 5, 7, 10, 14 downstream of each turbine.

Per-turbine power was extracted from turbinesFoam force output files
(`postProcessing/turbineOutput/`).

```bash
# Extract velocity profiles
postProcess -func sample -latestTime

# Turbine power is written during runtime to:
# postProcessing/turbineOutput/0/turbineX.dat
```

## CFD Data Availability

The extracted validation data used by all benchmark scripts is included in
`benchmark/data/`. The full CFD case directories (mesh, field files) are not
included due to file size (several GB per case).

**Not included:** the 8 m/s and 12 m/s single-turbine sensitivity cases. The
P1-3 multi-speed generalisation figure and table therefore cannot be
regenerated from this repository — see `benchmark/README.md` §6.

### benchmark/data/ structure

```
benchmark/data/
├── layout_single/             # Isolated single turbine — primary P1 reference
│   └── wakeProfiles/
│       └── <timestep>/        # e.g. 850/
│           ├── wake_1D_U.csv  # Streamwise velocity (U) at 1D downstream
│           ├── wake_1D_k.csv  # Turbulent kinetic energy (k) at 1D
│           └── ...            # 1D, 3D, 5D, 7D, 10D, 14D
├── layout_a/                  # Layout A — 3-turbine inline row, 7D
│   ├── wakeProfiles/          # Same structure
│   ├── turbines/              # Per-turbine power output
│   │   └── 0/{turbine1,turbine2,turbine3}.csv
│   ├── actuatorLines/
│   └── actuatorLineElements/
├── layout_5D/                 # 3-turbine row at 5D — independent hold-out
│   └── wakeProfiles/
└── layout_b/                  # Layout B — 4x4 grid, 7D x 9D
    ├── wakeProfiles/
    ├── turbines/              # 16 turbine output files
    ├── actuatorLines/
    └── actuatorLineElements/
```

Wake profiles are hub-height lateral lines spanning +/-2.5D, sampled at 1D, 3D,
5D, 7D, 10D and 14D downstream of T1. Columns are lateral position (m) and the
sampled quantity.

For access to the full simulation cases, open an issue on this repository.

## Reference

- Barthelmie et al. (2009) - offshore wake validation methodology
- Bastankhah & Porté-Agel (2014) - Gaussian wake model reference
- Charnock (1955) - offshore surface roughness
- ISO 2533 - standard atmosphere air density
