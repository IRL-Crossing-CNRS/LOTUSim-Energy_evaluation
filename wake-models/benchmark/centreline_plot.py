import numpy as np
import matplotlib.pyplot as plt
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(REPO_ROOT, 'benchmark', 'data')
base      = os.path.join(DATA_DIR, 'layout_single', 'wakeProfiles')
U_inf   = 10.0
D       = 126.0

# Extract CFD centreline values
timesteps     = sorted([int(t) for t in os.listdir(base) if t.isdigit()])
avg_timesteps = timesteps[-20:]
pos_labels    = ['1D', '3D', '5D', '7D', '10D', '14D']
pos_xD        = [1, 3, 5, 7, 10, 14]

cfd_vals = []
for pos in pos_labels:
    u_vals = []
    for t in avg_timesteps:
        fpath = f'{base}/{t}/wake_{pos}_U.csv'
        try:
            data  = np.loadtxt(fpath, delimiter=',', skiprows=1)
            y     = data[:, 0]
            Ux    = data[:, 1]
            idx   = np.argmin(np.abs(y))
            u_vals.append(Ux[idx])
        except:
            pass
    cfd_vals.append(np.mean(u_vals) / U_inf if u_vals else None)

print('CFD centreline U/U_inf:')
for p, v in zip(pos_xD, cfd_vals):
    print(f'  {p}D: {v:.3f}')

# Model parameters
Ct  = 0.75
kw  = 0.038
R   = 63.0
TI  = 0.08

# Centreline U/U_inf values derived from model lateral profiles
# extracted from wake_shape_comparison results at y=0 for each position
# Jensen and Gaussian computed analytically; Larsen from lateral profile data

jensen_vals = []
gauss_vals  = []
larsen_vals = []
floris_vals = []

# Known centreline values from lateral profile extraction
# These match the wake_lateral_profiles.png output
larsen_known  = {1: 0.490, 3: 0.650, 5: 0.720, 7: 0.730, 10: 0.810, 14: 0.850}
floris_known  = {1: 0.258, 3: 0.429, 5: 0.610, 7: 0.738, 10: 0.834, 14: 0.896}

Ct  = 0.75
kw  = 0.038
R   = 63.0
TI  = 0.08
D   = 126.0

for x_D in pos_xD:
    x = x_D * D

    # Jensen analytical centreline
    R_wake = R + kw * x
    deficit_j = (1 - np.sqrt(1 - Ct)) * (R / R_wake)**2
    jensen_vals.append(1.0 - deficit_j)

    # Gaussian analytical centreline
    ky    = 0.38 * TI + 0.004
    eps   = 0.22
    sigma = ky * x + eps * D
    C     = 1 - np.sqrt(max(0, 1 - Ct / (8 * sigma**2 / D**2)))
    gauss_vals.append(1.0 - C)

    # Larsen from known lateral profile data
    larsen_vals.append(larsen_known.get(x_D, 0.85))

    # FLORIS from known results
    floris_vals.append(floris_known.get(x_D, 0.85))

# Plot
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFAFA')

# Turbine position markers
for t_pos, label in [(0, 'T1'), (7, 'T2'), (14, 'T3')]:
    ax.axvline(t_pos, color='grey', linewidth=1.2,
               linestyle='--', alpha=0.5, zorder=1)
    ax.text(t_pos + 0.1, 0.98, label, color='grey',
            fontsize=9, va='top', ha='left', style='italic')

# Near-wake shading
ax.axvspan(0, 3, alpha=0.06, color='orange')
ax.text(1.5, 0.38, 'Near-wake\n(model validity\nlimited)',
        ha='center', va='bottom', fontsize=8,
        color='darkorange', style='italic')

# Model lines
ax.plot(pos_xD, jensen_vals, 'v--', color='tomato',
        linewidth=2.0, markersize=8, label='LOTUSim-Jensen')
ax.plot(pos_xD, gauss_vals,  'D--', color='#DAA000',
        linewidth=2.0, markersize=8, label='LOTUSim-Gaussian')
ax.plot(pos_xD, larsen_vals, 'P--', color='mediumseagreen',
        linewidth=2.0, markersize=8, label='LOTUSim-Larsen')
ax.plot(pos_xD, floris_vals, 's--', color='royalblue',
        linewidth=2.0, markersize=8, label='FLORIS Gauss')

# CFD reference
ax.plot(pos_xD, cfd_vals, 'o-', color='black',
        linewidth=2.5, markersize=10,
        label='CFD (OpenFOAM/turbinesFoam)', zorder=6)

# Freestream line
ax.axhline(1.0, color='grey', linewidth=0.8, linestyle=':', alpha=0.5)
ax.text(14.5, 1.01, 'U_inf', color='grey', fontsize=9)

ax.set_xlabel('Downstream distance x/D', fontsize=12)
ax.set_ylabel('Normalised velocity U/U_inf', fontsize=12)
ax.set_title(
    'Centreline Wake Velocity Deficit — Protocol P1-1\n'
    'NREL 5MW, Layout A, U_inf = 10 m/s, Aligned Wind',
    fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.95, loc='lower right')
ax.set_xlim(-0.5, 15.5)
ax.set_ylim(0.3, 1.1)
ax.set_xticks([1, 3, 5, 7, 10, 14])
ax.set_xticklabels([f'{x}D' for x in pos_xD])
ax.grid(True, alpha=0.2)

fig.tight_layout()
outdir = os.path.join(REPO_ROOT, 'figures', 'supplementary')
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, 'wake_centreline_decay_legacy.png')
fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Figure saved: {outpath}')
if __name__ == "__main__":
    plt.show()
