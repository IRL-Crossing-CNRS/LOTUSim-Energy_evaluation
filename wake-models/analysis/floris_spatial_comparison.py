import os
import sys

# Repo-relative paths: this script runs identically from any working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
DATA_DIR = os.path.join(REPO_ROOT, 'benchmark', 'data')
SUPP_DIR = os.path.join(REPO_ROOT, 'figures', 'supplementary')

import numpy as np, tempfile, yaml, matplotlib.pyplot as plt
import floris
from models.extended_models import (BlendedWakeModel, LarsenWakeModel,
                              GaussianWakeModel, CalibratedGaussianWakeModel)

plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'figure.dpi': 150})

D=126.0; U_INF=10.0; TI=0.08; RHO=1.225; HH=90.0; r_probe=63.0

blend    = BlendedWakeModel(diameter=D, ct=0.75, air_density=RHO, cp=0.498,
                             ambient_ti=TI, cut_in_speed=3.0, cut_out_speed=25.0)
larsen   = LarsenWakeModel(diameter=D, ct=0.75, air_density=RHO, cp=0.498,
                            ambient_ti=TI, cut_in_speed=3.0, cut_out_speed=25.0)
gauss    = GaussianWakeModel(diameter=D, ct=0.75, air_density=RHO, cp=0.498,
                              cut_in=3.0, cut_out=25.0, ambient_ti=TI, eps=0.22)
cal_gauss = CalibratedGaussianWakeModel(diameter=D, ct=0.75, air_density=RHO,
                                         cp=0.498, cut_in=3.0, cut_out=25.0,
                                         ambient_ti=TI, eps=0.22)

# FLORIS single turbine setup
floris_path = os.path.dirname(floris.__file__)
with open(floris_path + '/default_inputs.yaml') as f:
    config = yaml.safe_load(f)
config['farm']['layout_x'] = [0.0]
config['farm']['layout_y'] = [0.0]
config['farm']['turbine_type'] = ['nrel_5MW']
config['flow_field']['wind_speeds'] = [U_INF]
config['flow_field']['wind_directions'] = [270.0]
config['flow_field']['turbulence_intensities'] = [TI]
config['flow_field']['air_density'] = RHO
config['wake']['model_strings']['velocity_model'] = 'gauss'
tmp = os.path.join(tempfile.gettempdir(), 'floris_spatial.yaml')
with open(tmp, 'w') as f:
    yaml.dump(config, f)
from floris import FlorisModel
fm = FlorisModel(tmp)
fm.run()

positions = [1, 3, 5, 7, 10, 14]
dr = 1.0

floris_grad=[]; blend_grad=[]; larsen_grad=[]
gauss_grad=[]; cal_grad=[]

print('Computing gradients...')
for x_D in positions:
    x = x_D * D
    # FLORIS gradient
    try:
        u_plus  = fm.sample_flow_at_points(
            np.array([x]), np.array([r_probe+dr]), np.array([HH]))[0]
        u_minus = fm.sample_flow_at_points(
            np.array([x]), np.array([r_probe-dr]), np.array([HH]))[0]
        floris_grad.append(abs((u_plus[0]-u_minus[0])/(2*dr)/U_INF))
    except Exception as e:
        print(f'  FLORIS {x_D}D: {e}')
        floris_grad.append(0)
    blend_grad.append(abs(blend.velocity_gradient_at_point(U_INF,x,r_probe))/U_INF)
    larsen_grad.append(abs(larsen.velocity_gradient_at_point(U_INF,x,r_probe))/U_INF)
    gauss_grad.append(abs(gauss.velocity_gradient_at_point(U_INF,x,r_probe))/U_INF)
    cal_grad.append(abs(cal_gauss.velocity_gradient_at_point(U_INF,x,r_probe))/U_INF)

# CFD reference
base = os.path.join(DATA_DIR, 'layout_single', 'wakeProfiles')
timesteps = sorted([int(t) for t in os.listdir(base) if t.isdigit()])
avg_t = timesteps[-20:]
cfd_grad=[]; cfd_ci=[]
for pos in ['1D','3D','5D','7D','10D','14D']:
    grads=[]
    for t in avg_t:
        try:
            data=np.loadtxt(f'{base}/{t}/wake_{pos}_U.csv',delimiter=',',skiprows=1)
            y=data[:,0]; Ux=data[:,1]
            idx=np.argmin(np.abs(y-r_probe))
            if 0<idx<len(y)-1:
                dy=y[idx+1]-y[idx-1]
                grads.append(abs((Ux[idx+1]-Ux[idx-1])/dy/U_INF))
        except: pass
    cfd_grad.append(np.mean(grads))
    cfd_ci.append(1.96*np.std(grads)/np.sqrt(len(grads)))

# Print table
print(f'\n{"x/D":<6} {"CFD":>10} {"FLORIS":>10} {"Blended":>10} '
      f'{"Larsen":>10} {"CalGauss":>10} {"Gauss":>10}')
print('-'*68)
for i,x_D in enumerate(positions):
    fl_err = abs(floris_grad[i]-cfd_grad[i])/cfd_grad[i]*100
    b_err  = abs(blend_grad[i]-cfd_grad[i])/cfd_grad[i]*100
    print(f'{x_D:<6} {cfd_grad[i]:>10.5f} {floris_grad[i]:>10.5f} '
          f'{blend_grad[i]:>10.5f} {larsen_grad[i]:>10.5f} '
          f'{cal_grad[i]:>10.5f} {gauss_grad[i]:>10.5f}')

# Figure
fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFAFA')

cfd_arr=np.array(cfd_grad); ci_arr=np.array(cfd_ci)
ax.fill_between(positions, cfd_arr-ci_arr, cfd_arr+ci_arr,
                alpha=0.15, color='black')
ax.errorbar(positions, cfd_grad, yerr=cfd_ci, fmt='ko-',
            linewidth=2.5, markersize=10, capsize=5, capthick=2,
            elinewidth=1.5, label='CFD (single turbine)', zorder=6)
ax.plot(positions, gauss_grad, 'D--', color='#DAA000',
        linewidth=1.5, markersize=7,
        label='LOTUSim-Gaussian (standard)', alpha=0.7)
ax.plot(positions, larsen_grad, 'P--', color='mediumseagreen',
        linewidth=1.5, markersize=7, label='LOTUSim-Larsen', alpha=0.7)
ax.plot(positions, cal_grad, 'X-', color='steelblue',
        linewidth=2.0, markersize=9, label='LOTUSim-Calibrated Gaussian')
ax.plot(positions, blend_grad, 'h-', color='darkviolet',
        linewidth=3.0, markersize=11, label='LOTUSim-Blended', zorder=5)
ax.plot(positions, floris_grad, 's--', color='royalblue',
        linewidth=2.0, markersize=9, label='FLORIS Gauss')

ax.set_xlabel('Downstream distance x/D', fontsize=12)
ax.set_ylabel('|∂U/∂y| / U\u221e  (s\u207b\u00b9)', fontsize=12)
ax.set_title('Wake Velocity Gradient at r = 0.5D\n'
             'LOTUSim Models vs FLORIS Gauss vs Single Turbine CFD',
             fontsize=13, fontweight='bold')
ax.set_xlim(0.5, 15); ax.set_ylim(0, 0.010)
ax.set_xticks(positions)
ax.legend(fontsize=9, framealpha=0.95, loc='upper right')
ax.grid(True, alpha=0.2)
fig.tight_layout()
os.makedirs(SUPP_DIR, exist_ok=True)
outpath = os.path.join(SUPP_DIR, 'floris_spatial_comparison.png')
fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved: {outpath}')
if __name__ == "__main__":
    plt.show()
