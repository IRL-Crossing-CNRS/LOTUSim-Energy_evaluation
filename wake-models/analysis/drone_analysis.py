"""
Wake Field Spatial Analysis — Drone Safety Assessment
======================================================
Compares Gaussian and Larsen spatial wake fields for UAV operations.
Generates velocity contour maps, TI fields, hazard zones, and
velocity gradient analysis for inspection drone path planning.
"""

import os
import sys

# Repo-relative paths: this script runs identically from any working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
DATA_DIR = os.path.join(REPO_ROOT, 'benchmark', 'data')
SUPP_DIR = os.path.join(REPO_ROOT, 'figures', 'supplementary')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from models.extended_models import GaussianWakeModel, LarsenWakeModel

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'figure.dpi': 150,
})

# =============================================================================
# MODEL SETUP
# =============================================================================
D   = 126.0
U   = 10.0
Ct  = 0.75
TI  = 0.08
HH  = 90.0

gauss = GaussianWakeModel(diameter=D, ct=Ct, air_density=1.225, cp=0.498,
                           cut_in=3.0, cut_out=25.0, ambient_ti=TI, eps=0.22)

larsen = LarsenWakeModel(diameter=D, ct=Ct, air_density=1.225, cp=0.498,
                          ambient_ti=TI, cut_in_speed=3.0, cut_out_speed=25.0)

# =============================================================================
# GRID SETUP
# =============================================================================
x_vals = np.linspace(0.5*D, 14*D, 200)   # downstream
r_vals = np.linspace(0, 3*D, 150)         # lateral (positive half)
X, R   = np.meshgrid(x_vals, r_vals)

# =============================================================================
# COMPUTE FIELDS
# =============================================================================
print("Computing velocity fields...")
U_gauss  = np.zeros_like(X)
U_larsen = np.zeros_like(X)
TI_gauss  = np.zeros_like(X)
TI_larsen = np.zeros_like(X)
HZ_gauss  = np.zeros_like(X)
HZ_larsen = np.zeros_like(X)
GRAD_gauss  = np.zeros_like(X)
GRAD_larsen = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = X[i, j]
        r = R[i, j]
        U_gauss[i,j]   = gauss.velocity_at_point_2d(U, x, r) / U
        U_larsen[i,j]  = larsen.velocity_at_point(U, x, r) / U
        TI_gauss[i,j]  = gauss.ti_at_point(x, r) * 100
        TI_larsen[i,j] = larsen.ti_at_point(x, r) * 100
        HZ_gauss[i,j]  = gauss.wake_hazard_zone(U, x, r)
        HZ_larsen[i,j] = larsen.wake_hazard_zone(U, x, r)
        GRAD_gauss[i,j]  = abs(gauss.velocity_gradient_at_point(U, x, r))
        GRAD_larsen[i,j] = abs(larsen.velocity_gradient_at_point(U, x, r))

print("Done. Plotting...")

# =============================================================================
# FIGURE 1 — VELOCITY CONTOUR COMPARISON
# =============================================================================
fig1, axes = plt.subplots(2, 1, figsize=(16, 10))
fig1.patch.set_facecolor('white')

for ax, U_field, title in [
    (axes[0], U_gauss,  'Gaussian Wake Model — Normalised Velocity U/U∞'),
    (axes[1], U_larsen, 'Larsen Wake Model — Normalised Velocity U/U∞'),
]:
    ax.set_facecolor('#F0F4F8')
    U_plot = np.clip(U_field, 0.50, 1.0)
    cf = ax.contourf(X/D, R/D, U_plot,
                 levels=np.linspace(0.5, 1.0, 30),
                 cmap='RdYlGn', alpha=0.95)
    ax.contour(X/D, R/D, U_field,
               levels=[0.75, 0.85, 0.95],
               colors=['red', 'orange', 'yellow'],
               linewidths=1.2, alpha=0.7)
    plt.colorbar(cf, ax=ax, label='U/U∞')

    # Turbine symbol
    ax.plot([0, 0], [-0.5, 0.5], color='black', linewidth=3, zorder=5)
    ax.scatter(0, 0, s=80, color='black', zorder=6)

    # Deficit thresholds
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.4)

    ax.set_xlabel('Downstream distance x/D', fontsize=12)
    ax.set_ylabel('Lateral distance r/D', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.grid(True, alpha=0.15)

fig1.suptitle('Wake Velocity Field Comparison — Gaussian vs Larsen\n'
              'NREL 5MW, U∞ = 10 m/s, Hub height = 90 m',
              fontsize=13, fontweight='bold')
fig1.tight_layout()
os.makedirs(SUPP_DIR, exist_ok=True)
_out = os.path.join(SUPP_DIR, 'drone_velocity_field.png')
fig1.savefig(_out, dpi=150, bbox_inches='tight',
             facecolor='white')
print("Saved: drone_velocity_field.png")

# =============================================================================
# FIGURE 2 — TI FIELD COMPARISON
# =============================================================================
fig2, axes = plt.subplots(2, 1, figsize=(16, 10))
fig2.patch.set_facecolor('white')

for ax, TI_field, title in [
    (axes[0], TI_gauss,  'Gaussian — Turbulence Intensity TI (%)'),
    (axes[1], TI_larsen, 'Larsen — Turbulence Intensity TI (%)'),
]:
    ax.set_facecolor('#F0F4F8')
    cf = ax.contourf(X/D, R/D, TI_field,
                     levels=np.linspace(8, 25, 30),
                     cmap='YlOrRd', alpha=0.95)
    ax.contour(X/D, R/D, TI_field,
               levels=[12, 18],
               colors=['orange', 'red'],
               linewidths=1.5, alpha=0.8,
               linestyles=['--', '-'])
    plt.colorbar(cf, ax=ax, label='TI (%)')

    ax.plot([0, 0], [-0.5, 0.5], color='black', linewidth=3, zorder=5)
    ax.scatter(0, 0, s=80, color='black', zorder=6)

    # Threshold labels
    ax.clabel(ax.contour(X/D, R/D, TI_field,
                         levels=[12, 18],
                         colors=['orange', 'red'],
                         linewidths=1.5, alpha=0.8,
                         linestyles=['--', '-']),
              fmt={12: 'TI=12% caution', 18: 'TI=18% restricted'},
              fontsize=8, inline=True)

    ax.set_xlabel('Downstream distance x/D', fontsize=12)
    ax.set_ylabel('Lateral distance r/D', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.grid(True, alpha=0.15)

fig2.suptitle('Turbulence Intensity Field — Gaussian vs Larsen\n'
              'Crespo-Hernandez Model, TI_amb = 8%',
              fontsize=13, fontweight='bold')
fig2.tight_layout()
os.makedirs(SUPP_DIR, exist_ok=True)
_out = os.path.join(SUPP_DIR, 'drone_ti_field.png')
fig2.savefig(_out, dpi=150, bbox_inches='tight',
             facecolor='white')
print("Saved: drone_ti_field.png")

# =============================================================================
# FIGURE 3 — HAZARD ZONE MAP
# =============================================================================
fig3, axes = plt.subplots(2, 1, figsize=(16, 10))
fig3.patch.set_facecolor('white')

cmap_hazard = mcolors.ListedColormap(['#2ECC71', '#F39C12', '#E74C3C'])
bounds = [-0.5, 0.5, 1.5, 2.5]
norm   = mcolors.BoundaryNorm(bounds, cmap_hazard.N)

for ax, HZ_field, title in [
    (axes[0], HZ_gauss,  'Gaussian — UAV Hazard Zone Classification'),
    (axes[1], HZ_larsen, 'Larsen — UAV Hazard Zone Classification'),
]:
    ax.set_facecolor('#F0F4F8')
    cf = ax.contourf(X/D, R/D, HZ_field,
                     levels=[-0.5, 0.5, 1.5, 2.5],
                     cmap=cmap_hazard, alpha=0.90)

    ax.plot([0, 0], [-0.5, 0.5], color='black', linewidth=3, zorder=5)
    ax.scatter(0, 0, s=80, color='black', zorder=6)

    ax.set_xlabel('Downstream distance x/D', fontsize=12)
    ax.set_ylabel('Lateral distance r/D', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.grid(True, alpha=0.15)

legend_elements = [
    Patch(facecolor='#2ECC71', label='Safe — deficit < 15%, TI < 12%'),
    Patch(facecolor='#F39C12', label='Caution — deficit 15-25% or TI 12-18%'),
    Patch(facecolor='#E74C3C', label='Restricted — deficit > 25% or TI > 18%'),
]
fig3.legend(handles=legend_elements, loc='lower center',
            ncol=3, fontsize=10, framealpha=0.95,
            bbox_to_anchor=(0.5, -0.02))

fig3.suptitle('UAV Wake Hazard Zone Classification — Gaussian vs Larsen\n'
              'Thresholds: deficit 15%/25%, TI 12%/18%',
              fontsize=13, fontweight='bold')
fig3.tight_layout()
os.makedirs(SUPP_DIR, exist_ok=True)
_out = os.path.join(SUPP_DIR, 'drone_hazard_zones.png')
fig3.savefig(_out, dpi=150, bbox_inches='tight',
             facecolor='white')
print("Saved: drone_hazard_zones.png")

# =============================================================================
# FIGURE 4 — LATERAL PROFILES AT KEY POSITIONS
# =============================================================================
fig4, axes = plt.subplots(1, 3, figsize=(18, 7))
fig4.patch.set_facecolor('white')

r_plot = np.linspace(0, 3*D, 300)
positions = [(1, '1D'), (7, '7D — T2 position'), (14, '14D — T3 position')]

for ax, (x_D, title) in zip(axes, positions):
    ax.set_facecolor('#FAFAFA')
    x = x_D * D

    # Velocity profiles
    u_g = [gauss.velocity_at_point_2d(U, x, r)/U for r in r_plot]
    u_l = [larsen.velocity_at_point(U, x, r)/U for r in r_plot]

    ax.plot(r_plot/D, u_g, color='#DAA000', linewidth=2.5,
            label='Gaussian', linestyle='-')
    ax.plot(r_plot/D, u_l, color='mediumseagreen', linewidth=2.5,
            label='Larsen', linestyle='--')

    # Hazard thresholds
    ax.axhline(0.85, color='orange', linewidth=1.0,
               linestyle=':', alpha=0.7, label='Caution (15% deficit)')
    ax.axhline(0.75, color='red', linewidth=1.0,
               linestyle=':', alpha=0.7, label='Restricted (25% deficit)')

    # Rotor edge
    ax.axvline(0.5, color='grey', linewidth=1.0,
               linestyle='--', alpha=0.5, label='Rotor edge (±0.5D)')

    ax.set_xlabel('Lateral distance r/D', fontsize=12)
    ax.set_ylabel('U/U∞', fontsize=12)
    ax.set_title(f'x = {title}', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 3)
    ax.set_ylim(0.4, 1.1)
    ax.legend(fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.2)

fig4.suptitle('Lateral Wake Profiles — Gaussian vs Larsen\n'
              'With UAV operational thresholds',
              fontsize=13, fontweight='bold')
fig4.tight_layout()
os.makedirs(SUPP_DIR, exist_ok=True)
_out = os.path.join(SUPP_DIR, 'drone_lateral_profiles.png')
fig4.savefig(_out, dpi=150, bbox_inches='tight',
             facecolor='white')
print("Saved: drone_lateral_profiles.png")

# =============================================================================
# PRINT SUMMARY TABLE
# =============================================================================
print("\n" + "="*70)
print("UAV WAKE HAZARD SUMMARY — Safe approach distance from centreline")
print("="*70)
print(f"{'Position':<10} {'Gaussian caution':>18} {'Gaussian safe':>15} "
      f"{'Larsen caution':>16} {'Larsen safe':>13}")
print("-"*70)

for x_D in [1, 3, 5, 7, 10, 14]:
    x = x_D * D
    # Find caution and safe boundaries for each model
    r_test = np.linspace(0, 3*D, 500)

    g_caution = g_safe = larsen_caution = larsen_safe = None
    for r in r_test:
        gz = gauss.wake_hazard_zone(U, x, r)
        lz = larsen.wake_hazard_zone(U, x, r)
        if g_caution is None and gz <= 1:
            g_caution = r
        if g_safe is None and gz == 0:
            g_safe = r
        if larsen_caution is None and lz <= 1:
            larsen_caution = r
        if larsen_safe is None and lz == 0:
            larsen_safe = r

    gc = f"{g_caution/D:.2f}D ({g_caution:.0f}m)" if g_caution else "N/A"
    gs = f"{g_safe/D:.2f}D ({g_safe:.0f}m)" if g_safe else "N/A"
    lc = f"{larsen_caution/D:.2f}D ({larsen_caution:.0f}m)" if larsen_caution else "N/A"
    ls = f"{larsen_safe/D:.2f}D ({larsen_safe:.0f}m)" if larsen_safe else "N/A"
    print(f"{x_D}D{'':<8} {gc:>18} {gs:>15} {lc:>16} {ls:>13}")

if __name__ == "__main__":
    plt.show()
