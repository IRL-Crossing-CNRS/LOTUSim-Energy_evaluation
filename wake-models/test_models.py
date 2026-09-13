"""Quick sanity check for LarsenWakeModel and BlendedWakeModel."""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.larsen import LarsenWakeModel
from models.blended import BlendedWakeModel

D, CT, CP, TI, RHO = 126.0, 0.75, 0.498, 0.08, 1.225
# Layout A: 3-turbine aligned row at 7D spacing, (x_lateral, y_hub, z_downstream).
turbines = [(0, 90.0, 0), (0, 90.0, 882), (0, 90.0, 1764)]
# Wind vector is [x, z]: pure aligned inflow at 10 m/s travelling in +z.
# (This was [-10, -10.0] -- a 14.1 m/s 45-degree inflow -- which could never
#  produce the aligned baseline powers the assertion below checks for.)
wind_vector = [0.0, 10.0]

print("=== LarsenWakeModel ===")
larsen = LarsenWakeModel(diameter=D, ct=CT, air_density=RHO, cp=CP, ambient_ti=TI)
_, velocities, _ = larsen.wind_speeds_full(turbines, wind_vector)
powers = [larsen.power(v) / 1e6 for v in velocities]
print(f"Velocities: {[round(v,3) for v in velocities]} m/s")
print(f"Powers:     {[round(p,3) for p in powers]} MW")
print("Expected:   [3.724, 1.707, 0.858] MW  (protocol P2-1 baseline)")

expected = [3.724, 1.707, 0.858]          # protocol P2-1 baseline
ok = all(abs(p - e) < 0.05 for p, e in zip(powers, expected))
print("PASS" if ok else "FAIL - powers outside expected range")

print()
print("=== BlendedWakeModel ===")
blended = BlendedWakeModel(diameter=D, ct=CT, air_density=RHO, cp=CP, ambient_ti=TI)
X, Y = np.meshgrid(np.linspace(0, 2200, 50), np.linspace(-300, 300, 50))
U_field = blended.farm_velocity_field(10.0, X, Y, turbines)
print(f"Velocity field shape: {U_field.shape}")
print(f"Min velocity: {U_field.min():.3f} m/s")
print(f"Max velocity: {U_field.max():.3f} m/s")
ok2 = U_field.max() <= 10.0 and U_field.min() > 0.0
print("PASS" if ok2 else "FAIL - velocity field out of expected range")

print()
print("All done.")
