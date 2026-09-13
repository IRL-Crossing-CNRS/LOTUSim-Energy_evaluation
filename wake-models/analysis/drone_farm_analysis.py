"""
Farm-Scale Wake Field Spatial Analysis — Drone Safety
======================================================
Computes velocity field and hazard zones for full 3-turbine farm
accounting for compounding wake effects from all upstream turbines.
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
# SETUP
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

# 3-turbine farm layout (x_lat, y_hub, z_down)
turbines = [
    (0.0, HH, 0.0),
    (0.0, HH, 7*D),
    (0.0, HH, 14*D),
]

# Grid — covers full farm domain
x_vals = np.linspace(0.1*D, 16*D, 250)   # downstream
y_vals = np.linspace(-3*D, 3*D, 200)      # lateral (both sides)
X, Y   = np.meshgrid(x_vals, y_vals)

# =============================================================================
# COMPUTE FARM FIELDS
# =============================================================================
print("Computing farm velocity fields (this may take a minute)...")

U_gauss  = np.zeros_like(X)
U_larsen = np.zeros_like(X)
TI_gauss  = np.zeros_like(X)
TI_larsen = np.zeros_like(X)
HZ_gauss  = np.zeros_like(X)
HZ_larsen = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = X[i, j]
        y = Y[i, j]
        U_gauss[i,j]   = gauss.farm_velocity_at_point(U, x, y, turbines) / U
        U_larsen[i,j]  = larsen.farm_velocity_at_point(U, x, y, turbines) / U
        TI_gauss[i,j]  = gauss.farm_ti_at_point(x, y, turbines) * 100
        TI_larsen[i,j] = larsen.farm_ti_at_point(x, y, turbines) * 100
        HZ_gauss[i,j]  = gauss.farm_hazard_zone(U, x, y, turbines)
        HZ_larsen[i,j] = larsen.farm_hazard_zone(U, x, y, turbines)
    if i % 20 == 0:
        print(f"  Progress: {100*i/X.shape[0]:.0f}%")

print("Done. Plotting...")

# =============================================================================
# FIGURE 1 — FARM VELOCITY FIELD
# =============================================================================
fig1, axes = plt.subplots(2, 1, figsize=(18, 12))
fig1.patch.set_facecolor('white')

for ax, U_field, title in [
    (axes[0], U_gauss,  'Gaussian — Farm Velocity Field U/U∞'),
    (axes[1], U_larsen, 'Larsen — Farm Velocity Field U/U∞'),
]:
    ax.set_facecolor('#F0F4F8')
    cf = ax.contourf(X/D, Y/D, U_field,
                     levels=np.linspace(0.4, 1.0, 40),
                     cmap='RdYlGn', alpha=0.95)
    ax.contour(X/D, Y/D, U_field,
               levels=[0.75, 0.85, 0.95],
               colors=['red', 'orange', 'yellow'],
               linewidths=1.0, alpha=0.6)
    plt.colorbar(cf, ax=ax, label='U/U∞')

    # Turbine symbols
    for (x_t, y_t, z_t) in turbines:
        ax.plot([z_t/D, z_t/D], [-0.5, 0.5],
                color='black', linewidth=3, zorder=5)
        ax.scatter(z_t/D, 0, s=80, color='black', zorder=6)
        ax.text(z_t/D, 0.6, f'T{turbines.index((x_t,y_t,z_t))+1}',
                ha='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('Downstream distance x/D', fontsize=12)
    ax.set_ylabel('Lateral distance y/D', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0, 16)
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.15)

fig1.suptitle('Farm-Scale Wake Velocity Field — 3-Turbine Row, 7D Spacing\n'
              'NREL 5MW, U∞ = 10 m/s, Compounding Wake Effects',
              fontsize=13, fontweight='bold')
fig1.tight_layout()
os.makedirs(SUPP_DIR, exist_ok=True)
_out = os.path.join(SUPP_DIR, 'drone_farm_velocity.png')
fig1.savefig(_out, dpi=150,
             bbox_inches='tight', facecolor='white')
print("Saved: drone_farm_velocity.png")

# =============================================================================
# FIGURE 2 — FARM HAZARD ZONE MAP
# =============================================================================
fig2, axes = plt.subplots(2, 1, figsize=(18, 12))
fig2.patch.set_facecolor('white')

cmap_hazard = mcolors.ListedColormap(['#2ECC71', '#F39C12', '#E74C3C'])
bounds = [-0.5, 0.5, 1.5, 2.5]
norm   = mcolors.BoundaryNorm(bounds, cmap_hazard.N)

for ax, HZ_field, title in [
    (axes[0], HZ_gauss,  'Gaussian — Farm UAV Hazard Zone Classification'),
    (axes[1], HZ_larsen, 'Larsen — Farm UAV Hazard Zone Classification'),
]:
    ax.set_facecolor('#F0F4F8')
    ax.contourf(X/D, Y/D, HZ_field,
                levels=[-0.5, 0.5, 1.5, 2.5],
                cmap=cmap_hazard, alpha=0.90)

    # Turbine symbols
    for idx, (x_t, y_t, z_t) in enumerate(turbines):
        ax.plot([z_t/D, z_t/D], [-0.5, 0.5],
                color='black', linewidth=3, zorder=5)
        ax.scatter(z_t/D, 0, s=80, color='black', zorder=6)
        ax.text(z_t/D, 0.65, f'T{idx+1}',
                ha='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('Downstream distance x/D', fontsize=12)
    ax.set_ylabel('Lateral distance y/D', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0, 16)
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.15)

legend_elements = [
    Patch(facecolor='#2ECC71',
          label='Safe — deficit < 15%, TI < 12%'),
    Patch(facecolor='#F39C12',
          label='Caution — deficit 15-25% or TI 12-18%'),
    Patch(facecolor='#E74C3C',
          label='Restricted — deficit > 25% or TI > 18%'),
]
fig2.legend(handles=legend_elements, loc='lower center',
            ncol=3, fontsize=10, framealpha=0.95,
            bbox_to_anchor=(0.5, -0.02))

fig2.suptitle('Farm-Scale UAV Hazard Zone Classification — 3-Turbine Row\n'
              'Thresholds: velocity deficit 15%/25%, TI 12%/18%',
              fontsize=13, fontweight='bold')
fig2.tight_layout()
os.makedirs(SUPP_DIR, exist_ok=True)
_out = os.path.join(SUPP_DIR, 'drone_farm_hazard.png')
fig2.savefig(_out, dpi=150,
             bbox_inches='tight', facecolor='white')
print("Saved: drone_farm_hazard.png")

# =============================================================================
# FIGURE 3 — GRADIENT COMPARISON vs CFD
# =============================================================================
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig3.patch.set_facecolor('white')

positions = [1, 3, 5, 7, 10, 14]
r_probe   = 63.0  # 0.5D

# CFD gradient values from earlier extraction
cfd_grad = [0.00324, 0.00347, 0.00335, 0.00489, 0.00612, 0.00701]

# Single turbine gradients
g_single  = [abs(gauss.velocity_gradient_at_point(U, p*D, r_probe)) / U
             for p in positions]
l_single  = [abs(larsen.velocity_gradient_at_point(U, p*D, r_probe)) / U
             for p in positions]

# Farm gradients (compounding)
g_farm = []
l_farm = []
for p in positions:
    x = p * D
    dU_g = (gauss.farm_velocity_at_point(U, x, r_probe + 1, turbines) -
             gauss.farm_velocity_at_point(U, x, r_probe - 1, turbines)) / (2 * U)
    dU_l = (larsen.farm_velocity_at_point(U, x, r_probe + 1, turbines) -
             larsen.farm_velocity_at_point(U, x, r_probe - 1, turbines)) / (2 * U)
    g_farm.append(abs(dU_g))
    l_farm.append(abs(dU_l))

# Plot single turbine gradients
ax1.set_facecolor('#FAFAFA')
ax1.plot(positions, cfd_grad, 'ko-', linewidth=2.5,
         markersize=10, label='CFD reference', zorder=6)
ax1.plot(positions, g_single, 'D--', color='#DAA000',
         linewidth=2.0, markersize=8, label='Gaussian — single turbine')
ax1.plot(positions, l_single, 'P--', color='mediumseagreen',
         linewidth=2.0, markersize=8, label='Larsen — single turbine')
ax1.set_xlabel('Downstream distance x/D', fontsize=12)
ax1.set_ylabel('|∂U/∂y| / U∞  (s⁻¹)', fontsize=12)
ax1.set_title('Single Turbine Wake\nVelocity Gradient at r = 0.5D',
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, framealpha=0.95)
ax1.grid(True, alpha=0.2)
ax1.set_facecolor('#FAFAFA')

# Plot farm gradients
ax2.set_facecolor('#FAFAFA')
ax2.plot(positions, cfd_grad, 'ko-', linewidth=2.5,
         markersize=10, label='CFD reference', zorder=6)
ax2.plot(positions, g_farm, 'D-', color='#DAA000',
         linewidth=2.5, markersize=8, label='Gaussian — farm (compounding)')
ax2.plot(positions, l_farm, 'P-', color='mediumseagreen',
         linewidth=2.5, markersize=8, label='Larsen — farm (compounding)')
ax2.set_xlabel('Downstream distance x/D', fontsize=12)
ax2.set_ylabel('|∂U/∂y| / U∞  (s⁻¹)', fontsize=12)
ax2.set_title('Farm Compounding Wake\nVelocity Gradient at r = 0.5D',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, framealpha=0.95)
ax2.grid(True, alpha=0.2)
ax2.set_facecolor('#FAFAFA')

fig3.suptitle('Velocity Gradient Comparison — Single Turbine vs Farm Compounding\n'
              'vs CFD Reference at r = 0.5D (63 m from centreline)',
              fontsize=13, fontweight='bold')
fig3.tight_layout()
os.makedirs(SUPP_DIR, exist_ok=True)
_out = os.path.join(SUPP_DIR, 'drone_farm_gradient_comparison.png')
fig3.savefig(_out, dpi=150,
             bbox_inches='tight', facecolor='white')
print("Saved: drone_gradient_comparison.png")

# =============================================================================
# PRINT FARM GRADIENT TABLE
# =============================================================================
print("\n" + "="*70)
print("VELOCITY GRADIENT COMPARISON — Single vs Farm at r = 0.5D")
print("="*70)
print(f"{'x/D':<6} {'CFD':>10} {'G-single':>12} {'G-farm':>10} "
      f"{'L-single':>12} {'L-farm':>10}")
print("-"*62)
for i, p in enumerate(positions):
    print(f"{p:<6} {cfd_grad[i]:>10.5f} {g_single[i]:>12.5f} "
          f"{g_farm[i]:>10.5f} {l_single[i]:>12.5f} {l_farm[i]:>10.5f}")

if __name__ == "__main__":
    plt.show()
