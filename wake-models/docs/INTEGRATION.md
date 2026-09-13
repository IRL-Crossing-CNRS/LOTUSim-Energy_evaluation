# LOTUSim-Energy Integration Guide

This document is the single reference for integrating the LOTUSim wake models
into LOTUSim-Energy. It covers all inputs, outputs, and access patterns for
both the high-level simulator interface and the underlying model classes.

---

## 1. Quick Start — WindFarmSimulator

`WindFarmSimulator` is the recommended entry point. It wraps both production
models and exposes a clean interface for the three main use cases: per-turbine
power, farm energy yield, and spatial wake fields.

### Installation

```bash
git clone https://github.com/IRL-Crossing-CNRS/LOTUSim-Energy_evaluation.git
cd lotusim-wake-models
pip install -r requirements.txt
```

### Initialise

```python
from simulator.wind_farm import WindFarmSimulator

sim = WindFarmSimulator(
    turbines  = [(0, 90.0, 0), (0, 90.0, 882), (0, 90.0, 1764)],
    diameter  = 126.0,   # m
    ct        = 0.75,    # thrust coefficient
    cp        = 0.498,   # power coefficient
    air_density  = 1.225,  # kg/m3
    ambient_ti   = 0.08,   # turbulence intensity (fraction)
    cut_in_speed  = 3.0,   # m/s
    cut_out_speed = 25.0,  # m/s
)
```

#### Turbine layout format

Each turbine is a tuple `(x_lateral, y_hub_height, z_downstream)` in metres:

| Component      | Description                                      |
|----------------|--------------------------------------------------|
| `x_lateral`    | Lateral position, positive to the right downwind |
| `y_hub_height` | Hub height, typically 90 m for NREL 5MW          |
| `z_downstream` | Downstream position, increases in wind direction |

#### Wind vector format

Wind vector is `[x_lateral, z_downstream]` — the horizontal wind in the
farm coordinate frame. Turbines face into the +z direction.

| Wind condition          | Vector        |
|-------------------------|---------------|
| Aligned (north wind)    | `[0.0, 10.0]` |
| Angled (lateral offset) | `[3.0, 10.0]` |

---

### 1a. Per-turbine power — `power_for_interval`

```python
result = sim.power_for_interval(wind_vector=[0.0, 10.0])
```

**Outputs** (`result` is a dict):

| Key              | Type         | Description                              |
|------------------|--------------|------------------------------------------|
| `velocities`     | list[float]  | Effective inflow speed at each turbine (m/s) |
| `Power_w`        | list[float]  | Per-turbine power output (W)             |
| `farm_power_w`   | float        | Total farm power (W)                     |

**Example:**

```python
print(result['velocities'])    # [10.0, 7.71, 6.13] m/s
print(result['Power_w'])       # [3724000, 1707000, 858000] W
print(result['farm_power_w'])  # 6289000 W
```

---

### 1b. Farm energy yield — `energy_for_timeseries`

```python
wind_vectors = [[0.0, 8.0], [0.0, 10.0], [0.0, 12.0]]
times_hours  = [100, 200, 100]

total_kwh, results = sim.energy_for_timeseries(wind_vectors, times_hours)
```

**Inputs:**

| Parameter      | Type             | Description                              |
|----------------|------------------|------------------------------------------|
| `wind_vectors` | list[list[float]]| Wind vector `[x, z]` for each interval  |
| `times_hours`  | list[float]      | Duration of each interval in hours       |

**Outputs:**

| Variable      | Type        | Description                              |
|---------------|-------------|------------------------------------------|
| `total_kwh`   | float       | Total farm energy yield (kWh)            |
| `results`     | list[dict]  | Per-interval `power_for_interval` dicts  |

---

### 1c. Spatial wake field — `wake_field`

```python
import numpy as np

X, Y = np.meshgrid(np.linspace(0, 2200, 200), np.linspace(-300, 300, 200))
U_field = sim.wake_field(U_inf=10.0, X=X, Y=Y)
```

**Inputs:**

| Parameter | Type       | Description                              |
|-----------|------------|------------------------------------------|
| `U_inf`   | float      | Freestream wind speed (m/s)              |
| `X`       | np.ndarray | 2D grid of lateral positions (m)         |
| `Y`       | np.ndarray | 2D grid of downstream positions (m)      |

**Output:** `np.ndarray` of shape matching `X`/`Y` — wind speed at each
grid point (m/s).

---

### 1d. Drone hazard zone field — `hazard_field`

```python
hazard = sim.hazard_field(U_inf=10.0, X=X, Y=Y)
```

**Output:** `np.ndarray` (int) — hazard classification at each grid point:

| Value | Zone       | Condition                                      |
|-------|------------|------------------------------------------------|
| `0`   | Safe       | Velocity deficit < 15%, TI < 12%              |
| `1`   | Caution    | Velocity deficit 15–25%, TI 12–18%            |
| `2`   | Restricted | Velocity deficit > 25%, TI > 18%              |

---

## 2. Direct Model Classes

Use these if you need more control than `WindFarmSimulator` provides.

---

### 2a. LarsenWakeModel

Use for power output, energy yield, and LCOE. Lowest RMSE across all
power benchmark protocols (P2-1 RMSE 0.212 MW vs FLORIS 0.767 MW).

```python
from models.larsen import LarsenWakeModel

model = LarsenWakeModel(
    diameter     = 126.0,
    ct           = 0.75,
    cp           = 0.498,
    air_density  = 1.225,
    ambient_ti   = 0.08,
    cut_in_speed  = 3.0,
    cut_out_speed = 25.0,
)
```

#### Constructor parameters

| Parameter       | Required | Default | Description                  |
|-----------------|----------|---------|------------------------------|
| `diameter`      | yes      | —       | Rotor diameter (m)           |
| `ct`            | yes      | —       | Thrust coefficient           |
| `cp`            | no       | 0.35    | Power coefficient            |
| `air_density`   | no       | 1.225   | Air density (kg/m³)          |
| `ambient_ti`    | no       | 0.08    | Ambient turbulence intensity |
| `cut_in_speed`  | no       | 3.0     | Cut-in wind speed (m/s)      |
| `cut_out_speed` | no       | 25.0    | Cut-out wind speed (m/s)     |

#### Methods

**`wind_speeds_full(turbines, wind_vector)`**
Returns effective inflow speed at each turbine accounting for wake deficits
and wind shear.

```python
turbines = [(0, 90.0, 0), (0, 90.0, 882), (0, 90.0, 1764)]
turbines_sorted, velocities, order = model.wind_speeds_full(turbines, [0.0, 10.0])
# velocities → [10.0, 7.71, 6.13] m/s
```

| Output           | Type        | Description                                   |
|------------------|-------------|-----------------------------------------------|
| `turbines_sorted`| list        | Turbines reordered upstream to downstream     |
| `velocities`     | list[float] | Effective inflow speed per turbine (m/s)      |
| `order`          | list[int]   | Original indices in sorted order              |

---

**`power(wind_speed)`**
Returns power output for a single turbine at a given inflow speed.

```python
p = model.power(7.71)   # → 1707000.0 W
```

| Input        | Type  | Description          |
|--------------|-------|----------------------|
| `wind_speed` | float | Inflow speed (m/s)   |

| Output | Type  | Description       |
|--------|-------|-------------------|
| `p`    | float | Power output (W)  |

---

**`multi_speed(turbines, wind_vectors, times)`**
Farm energy yield over a wind speed/direction time series.

```python
total_kwh, per_interval = model.multi_speed(turbines, [[0,10],[0,8]], [100, 200])
```

| Output         | Type        | Description                   |
|----------------|-------------|-------------------------------|
| `total_kwh`    | float       | Total farm energy yield (kWh) |
| `per_interval` | list[dict]  | Per-interval results          |

---

**`farm_velocity_at_point(U_inf, x, y, turbines)`**
Wind speed at any point in the farm (single point query).

```python
u = model.farm_velocity_at_point(10.0, x=0, y=500, turbines=turbines)
```

| Output | Type  | Description          |
|--------|-------|----------------------|
| `u`    | float | Wind speed (m/s)     |

---

**`velocity_at_point(U_inf, x, r)`**
Wake velocity at downstream distance `x` and lateral offset `r` from a
single turbine (no farm context).

```python
u = model.velocity_at_point(U_inf=10.0, x=882, r=0)
```

---

### 2b. BlendedWakeModel

Use for spatial wake fields, velocity gradients, and drone hazard mapping.
Internally wraps a `LarsenWakeModel` for near-wake and a CFD-calibrated
Gaussian for far-wake. Less than 5% gradient error at 1D–14D vs RANS CFD.

```python
from models.blended import BlendedWakeModel

model = BlendedWakeModel(
    diameter     = 126.0,
    ct           = 0.75,
    cp           = 0.498,
    air_density  = 1.225,
    ambient_ti   = 0.08,
    cut_in_speed  = 3.0,
    cut_out_speed = 25.0,
)
```

Constructor parameters are identical to `LarsenWakeModel`.

#### Methods

**`farm_velocity_field(U_inf, X, Y, turbines)`**
Vectorised 2D velocity field across the farm at hub height.

```python
X, Y = np.meshgrid(np.linspace(0, 2200, 200), np.linspace(-300, 300, 200))
U_field = model.farm_velocity_field(10.0, X, Y, turbines)
```

| Output    | Type       | Description                            |
|-----------|------------|----------------------------------------|
| `U_field` | np.ndarray | Wind speed at each grid point (m/s)    |

---

**`farm_hazard_field(U_inf, X, Y, turbines)`**
Hazard zone classification across the full farm grid.

```python
hazard = model.farm_hazard_field(10.0, X, Y, turbines)
# 0=safe, 1=caution, 2=restricted
```

| Output   | Type          | Description                           |
|----------|---------------|---------------------------------------|
| `hazard` | np.ndarray (int) | Hazard zone per grid point (0/1/2) |

---

**`velocity_at_point(U_inf, x, r)`**
Blended wake velocity at downstream distance `x`, lateral offset `r`
from a single turbine.

```python
u = model.velocity_at_point(U_inf=10.0, x=882, r=0)
```

---

**`velocity_gradient_at_point(U_inf, x, r)`**
Lateral velocity gradient dU/dr at a point — the primary metric for
drone hazard assessment.

```python
grad = model.velocity_gradient_at_point(U_inf=10.0, x=500, r=63)
# units: s⁻¹
```

---

**`wake_hazard_zone(U_inf, x, r)`**
Hazard classification at a single point.

```python
zone = model.wake_hazard_zone(U_inf=10.0, x=500, r=0)
# → 0, 1, or 2
```

---

## 3. CFD Environment

The validation CFD simulations were run in a pre-built Docker image containing
OpenFOAM v8 and the turbinesFoam actuator line solver.

```bash
docker pull ellaj03/openfoam8-turbinesfoam

docker run -it --user root \
  -v /path/to/cases:/home/openfoam/cases \
  ellaj03/openfoam8-turbinesfoam bash

# Inside the container:
source /opt/openfoam8/etc/bashrc
```

See `docs/cfd_setup.md` for full simulation setup and reproduction instructions.

---

## 4. Recommended Models Summary

| Use case                        | Model              |
|---------------------------------|--------------------|
| Per-turbine power output        | `LarsenWakeModel`  |
| Farm energy yield / AEP / LCOE  | `LarsenWakeModel`  |
| 2D velocity field visualisation | `BlendedWakeModel` |
| Drone / UAV hazard zone mapping | `BlendedWakeModel` |
| Both (via single interface)     | `WindFarmSimulator`|
