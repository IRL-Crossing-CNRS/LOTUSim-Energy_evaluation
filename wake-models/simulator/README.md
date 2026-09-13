# simulator/

LOTUSim-Energy integration layer. Provides a single `WindFarmSimulator`
class that wraps both production models.
Internally:
- Power calculations use `LarsenWakeModel`
- Spatial wake field and hazard zones use `BlendedWakeModel`

## Why two models behind one class

`WindFarmSimulator` is not a wrapper over one wake model with a choice of
options — it deliberately runs two, because the benchmark behind this repository
found that no single engineering wake model is best for both jobs:

- **`LarsenWakeModel`** drives `power_for_interval` and `energy_for_timeseries`.
  Lowest per-turbine power RMSE against actuator-line CFD (0.212 MW), farm total
  within 0.3%.
- **`BlendedWakeModel`** drives `wake_field` and `hazard_field`. Most accurate
  resolved velocity field and wake-edge velocity gradient, which is what drone
  path planning and hazard-zone delineation depend on.

Power figures and spatial-field figures from this class therefore come from
different formulations by design, and should not be cross-compared as though
they were one model's output. See [../benchmark/README.md](../benchmark/README.md)
for the protocols and [../docs/INTEGRATION.md](../docs/INTEGRATION.md) for the
full API reference.

## Basic usage

```python
from simulator.wind_farm import WindFarmSimulator
import numpy as np

# Define turbine layout: (x_lateral, y_hub_height, z_downstream)
turbines = [
    (0, 90.0,    0),   # T1 upstream
    (0, 90.0,  882),   # T2 7D downstream
    (0, 90.0, 1764),   # T3 14D downstream
]

# Initialise simulator with NREL 5MW parameters
sim = WindFarmSimulator(
    turbines=turbines,
    diameter=126.0,
    ct=0.75,
    air_density=1.225,
    cp=0.498,
    ambient_ti=0.08,
)

# Per-turbine power for a single wind condition
result = sim.power_for_interval(wind_vector=[0.0, 10.0])
print(result['velocities'])   # [10.0, 7.71, 6.13] m/s
print(result['Power_w'])      # per-turbine power in W
print(result['farm_power_w']) # total farm power in W

# Farm energy yield over a time series
wind_vectors = [[0.0, 8.0], [0.0, 10.0], [0.0, 12.0]]
times_hours  = [100, 200, 100]
total_kwh, results = sim.energy_for_timeseries(wind_vectors, times_hours)
print(f"Total energy: {total_kwh:.1f} kWh")

# Spatial wake velocity field (for visualisation or hazard mapping)
X, Y = np.meshgrid(np.linspace(0, 2200, 200), np.linspace(-300, 300, 200))
U_field = sim.wake_field(U_inf=10.0, X=X, Y=Y)

# Drone hazard zone field (0=safe, 1=caution, 2=restricted)
hazard = sim.hazard_field(U_inf=10.0, X=X, Y=Y)
```

## Wind vector convention

Wind vector is `[x, z]` in the horizontal plane:
- Turbines face south (into the +z direction)
- Pure north wind (aligned): `[0.0, 10.0]`
- Angled wind: `[3.0, 10.0]` — has a lateral component

## Turbine coordinate convention

Each turbine is `(x_lateral, y_hub_height, z_downstream)`:
- `x` — lateral position (m), positive to the right facing downwind
- `y` — hub height (m), typically 90 m for NREL 5MW
- `z` — downstream position (m), increases in wind direction
