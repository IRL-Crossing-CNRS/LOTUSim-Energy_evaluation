# analysis/

Supplementary spatial-field and drone hazard-zone analysis built on
`BlendedWakeModel`. These scripts are **not part of the seven benchmark
protocols** — they are the downstream application that motivates the P1-3
wake-edge gradient metric, and they produce the supplementary figures in
`../figures/supplementary/`.

## What this is for, physically

An inspection UAV flying between turbines passes through the wake shear layer.
What determines whether that is safe is not the mean velocity deficit alone but
how fast the wind changes across the wake boundary — the lateral gradient
`|dU/dy|` at the rotor-edge radius `r = 0.5D`. That is exactly the quantity
protocol P1-3 scores, and it is why LOTUSim-Blended, rather than the more
accurate power model, supplies the velocity field flown by the vehicles in
LOTUSim-Energy. See [../benchmark/README.md](../benchmark/README.md) §1.

## Files

| Script | Produces (in `../figures/supplementary/`) | Content |
|--------|-------------------------------------------|---------|
| `drone_analysis.py` | `drone_velocity_field.png`, `drone_ti_field.png`, `drone_hazard_zones.png`, `drone_lateral_profiles.png` | Single-turbine wake: velocity contours, TI field, hazard zones, lateral profiles |
| `drone_farm_analysis.py` | `drone_farm_velocity.png`, `drone_farm_hazard.png`, `drone_farm_gradient_comparison.png` | Farm-scale (3-turbine) hazard field with compounding wakes |
| `floris_spatial_comparison.py` | `floris_spatial_comparison.png` | Hub-height spatial field, LOTUSim-Blended vs FLORIS Gaussian |

`drone_farm_analysis.py`'s gradient figure was previously named
`drone_gradient_comparison.png` — the same filename as the P1-3
figure produced by `benchmark/gen_all_figures.py`. It is renamed here so the
two can never overwrite each other.

## Running them

```bash
python3 analysis/drone_analysis.py
python3 analysis/drone_farm_analysis.py
python3 analysis/floris_spatial_comparison.py
```

No environment setup and no arguments. Each script resolves the repository root
from its own location, reads any CFD reference from `benchmark/data/`, and
writes its figures to `figures/supplementary/`, so the working directory does
not matter.


## Hazard-zone classification

`BlendedWakeModel.wake_hazard_zone()` and `.farm_hazard_zone()` classify a point
on **both** velocity deficit and local turbulence intensity:

| Zone | Meaning | Deficit `1 − U/U_inf` | Local TI |
|------|---------|----------------------|----------|
| 0 | Safe | ≤ 0.15 | ≤ 0.12 |
| 1 | Caution | > 0.15 | > 0.12 |
| 2 | Restricted | > 0.25 | > 0.18 |

A point is promoted to the higher class if *either* criterion trips. All four
thresholds are constructor-style keyword arguments and can be overridden per
call.

> **`farm_hazard_field()` classifies on deficit only.** The vectorised field
> method takes `deficit_caution` / `deficit_restricted` but no TI thresholds, so
> it is not identical to calling `farm_hazard_zone()` point by point. Use the
> point method where the TI criterion matters.
>
> These thresholds are an operational convention adopted for this work, not a
> validated regulatory limit.

## Usage example

```python
import numpy as np
from models.blended import BlendedWakeModel

model = BlendedWakeModel(diameter=126.0, ct=0.75, ambient_ti=0.08)
turbines = [(0, 90, 0), (0, 90, 882), (0, 90, 1764)]     # 3-turbine row, 7D
X, Y = np.meshgrid(np.linspace(0, 2200, 200), np.linspace(-300, 300, 200))

U_field = model.farm_velocity_field(10.0, X, Y, turbines)   # m/s
hazard  = model.farm_hazard_field(10.0, X, Y, turbines)     # 0 / 1 / 2

# Wake-edge gradient at the rotor radius, 7D downstream — the P1-3 metric
g = model.velocity_gradient_at_point(10.0, x=7 * 126.0, r=63.0)   # s^-1
```

A full 200×200 hub-height field over this farm takes about 20 ms
(4.5 ms at 100×100), against roughly 150 ms for FLORIS sampling the same
points — which is what makes online path planning feasible rather than
offline-only.
