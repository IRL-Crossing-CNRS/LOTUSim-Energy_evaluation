"""
Protocol P2-4 — Wind Direction Sweep (normalised)
=================================================
Normalised downstream-turbine power P_T2 / P_T1,0deg versus wind-direction
offset (aligned -> perpendicular), 3-turbine aligned row, 7D spacing,
U_inf = 10 m/s, Layout A.

Data are the P2-4 protocol model outputs printed by models/extended_models.py
(the wind-direction sweep section) — this script visualises exactly those values
so that the figure and the results table remain consistent by construction.
Rerun models/extended_models.py to regenerate the numbers themselves.
CFD spot checks are available at 0deg and 15deg only.

Colours/markers follow the repo-wide convention (gen_all_figures.py STYLES)
so every figure shares one visual identity per model.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# P2-4 outputs — identical to the normalised P_T2/P_T1,0deg table printed by
# models/extended_models.py.
# Normalised power ratio P_T2 / P_T1 at 0deg (aligned) reference.
# ---------------------------------------------------------------------------
offsets = np.array([0, 5, 10, 15, 20, 30, 45])            # degrees

series = {
    'LOTUSim-Jensen':   np.array([0.425, 0.645, 1.000, 1.000, 1.000, 1.000, 1.000]),
    'LOTUSim-Gaussian': np.array([0.417, 0.529, 0.771, 0.941, 0.991, 1.000, 1.000]),
    'LOTUSim-Larsen':   np.array([0.458, 0.794, 1.000, 1.000, 1.000, 1.000, 1.000]),
    'LOTUSim-Blended':  np.array([0.386, 0.746, 0.992, 1.000, 1.000, 1.000, 1.000]),
    'FLORIS':           np.array([0.440, 0.663, 0.889, 0.917, 0.918, 0.918, 0.918]),
    'FLORIS-Jensen':    np.array([0.500, 0.641, 0.918, 0.918, 0.918, 0.918, 0.918]),
}

# CFD reference — spot checks only (0deg and 15deg)
cfd_offsets = np.array([0, 15])
cfd_vals    = np.array([0.500, 0.900])

# Colour / marker / linestyle, matching STYLES in gen_all_figures.py
style = {
    'LOTUSim-Jensen':   ('tomato',         'v', '--'),
    'LOTUSim-Gaussian': ('#DAA000',        'D', '--'),
    'LOTUSim-Larsen':   ('mediumseagreen', 'P', '--'),
    'LOTUSim-Blended':  ('darkviolet',     'h', '-'),
    'FLORIS':           ('royalblue',      's', '-'),
    'FLORIS-Jensen':    ('cornflowerblue', '^', '-'),
}
DISPLAY = {'FLORIS': 'FLORIS Gaussian', 'FLORIS-Jensen': 'FLORIS Jensen'}

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
plt.rcParams.update({'font.family': 'serif'})
fig, ax = plt.subplots(figsize=(8.5, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFAFA')

for name, vals in series.items():
    col, marker, ls = style[name]
    ax.plot(offsets, vals, marker=marker, color=col, linestyle=ls,
            linewidth=2.2, markersize=10, label=DISPLAY.get(name, name))

# CFD spot checks
ax.scatter(cfd_offsets, cfd_vals, color='black', marker='*', s=320,
           zorder=6, label='CFD spot check')

# Full-recovery reference line
ax.axhline(1.0, color='k', linestyle=':', linewidth=0.9, alpha=0.4)
ax.text(46, 1.007, 'free-stream', fontsize=10, color='k', alpha=0.6,
        ha='right', va='bottom')

# Aligned-inflow marker
ax.axvline(0.0, color='grey', linestyle=':', linewidth=1.0, alpha=0.5)

ax.set_xlabel('Wind-direction offset from aligned (degrees)', fontsize=14)
ax.set_ylabel(r'Normalised power $P_{T2}/P_{T1,0^\circ}$', fontsize=14)
ax.set_title('Protocol P2-4 — Wind Direction Sweep\n'
             'NREL 5MW, Layout A, 7D spacing, $U_\\infty = 10$ m/s',
             fontsize=14, fontweight='bold')
ax.set_xticks(offsets)
ax.set_xlim(-2, 47)
ax.set_ylim(0.35, 1.08)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=12, framealpha=0.95, loc='lower right')

fig.tight_layout()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_fig   = os.path.join(REPO_ROOT, 'figures', 'main',
                         'wind_direction_normalised.png')
os.makedirs(os.path.dirname(out_fig), exist_ok=True)

fig.savefig(out_fig, dpi=600, bbox_inches='tight', facecolor='white')
print('Saved:', out_fig)
