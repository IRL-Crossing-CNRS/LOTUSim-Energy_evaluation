"""Top view of the field LOTUSim actually delivers at hub height.

Not a drawing of a wake: it replays the real pipeline. regions_for_wind()
builds the cone segments and samples the model's radial profile into
radial_speed exactly as the running scenario does, then the sampling below
reproduces WindRegionsPlugin::ResolveWind + SampleRadial -- last matching
region wins, linear interpolation on the radial fraction.
"""
import json, numpy as np
from lotusim_sdk.agents.environment.wake.larsen import LarsenWakeModel as L
from lotusim_sdk.agents.environment.wake.blended import BlendedWakeModel as B
from lotusim_sdk.agents.environment.wake.wake_regions import WakeRegionGenerator as G

CFG = 'src/simulation_run/config/wind_wake_examples/m3_factorial_D.json'
cfg = json.load(open(CFG))
w = [a for a in cfg['agents'] if a.get('id') == 'wake'][0]
D, HUB = w['diameter'], w['turbines'][0]['z']
model = B(L(diameter=D, ct=w['ct'], ambient_ti=w['ambient_ti']))
gen = G(model, turbulence=True)
WIND = np.array([0.0, 10.0])                       # from the UI panel: Y = 10 m/s
turb = [(t['name'], t['x'], t['y'], t['z']) for t in w['turbines']]
regs = gen.regions_for_wind(turb, WIND)
print(f"config={CFG.split('/')[-1]}  D={D}m  hub={HUB}m  U_inf={np.linalg.norm(WIND)}m/s")
print(f"{len(regs)} cone segments, {len(regs[0]['radial_speed'])} radial samples each")

def sample_radial(prof, frac):
    """WindRegionsPlugin::SampleRadial."""
    n = len(prof)
    if n < 2: return prof[0] if n else 0.0
    t = np.clip(frac, 0, 1) * (n - 1)
    i = int(t)
    return prof[-1] if i >= n - 1 else prof[i] + (t - i) * (prof[i + 1] - prof[i])

def resolve(x, y, z):
    """WindRegionsPlugin::ResolveWind — last matching region wins."""
    for r in reversed(regs):
        dx, dy, dz = x - r['origin_x'], y - r['origin_y'], z - r['origin_z']
        d = dx * r['axis_x'] + dy * r['axis_y']
        if d < 0 or d > r['length']: continue
        lat = dx * (-r['axis_y']) + dy * r['axis_x']
        rad = r['r_start'] + (r['r_end'] - r['r_start']) * (d / r['length'])
        rr = lat * lat + dz * dz
        if rr > rad * rad: continue
        return sample_radial(r['radial_speed'], np.sqrt(rr) / rad if rad > 0 else 0.0)
    return float(np.linalg.norm(WIND))             # ambient outside every region

# Grid in metres, turbine at the origin, wind blowing towards +y.
NX, NY = 420, 620
xs = np.linspace(-2.2 * D, 2.2 * D, NX)
ys = np.linspace(-0.6 * D, 9.4 * D, NY)
U = np.array([[resolve(x, y, HUB) for x in xs] for y in ys])
np.savez('/tmp/wake_field.npz', U=U, xs=xs, ys=ys, D=D, hub=HUB, uinf=float(np.linalg.norm(WIND)))
print(f"grid {NX}x{NY} over x=[{xs[0]:.0f},{xs[-1]:.0f}] y=[{ys[0]:.0f},{ys[-1]:.0f}] m")
print(f"U/Uinf range {U.min()/10:.3f} .. {U.max()/10:.3f}")
for yd in (1, 2, 4, 6, 8):
    row = U[np.argmin(np.abs(ys - yd * D))] / 10.0
    inside = row < 0.995
    print(f"  y={yd}D: centreline {row[NX//2]:.3f}  width {2*np.abs(xs[inside]).max()/D:.2f}D" if inside.any() else f"  y={yd}D: no wake")
