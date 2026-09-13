"""
Master figure generation script — all primary figures
"""
import numpy as np, os, sys, tempfile, yaml, matplotlib.pyplot as plt
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Extracted CFD reference data. Every path in this script is derived from
# DATA_DIR so the script runs identically on any machine and from any working
# directory -- no absolute paths to the original OpenFOAM case tree.
DATA_DIR = os.path.join(REPO_ROOT, 'benchmark', 'data')


def tmp_path(name):
    """Scratch file in the platform temp directory (not a hardcoded /tmp)."""
    return os.path.join(tempfile.gettempdir(), name)
from models.extended_models import (JensenWakeModel, GaussianWakeModel,
                              LarsenWakeModel, BlendedWakeModel)

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
DPI    = 600
FS     = 14   # base font size
FS_SM  = 12   # small
FS_LG  = 16   # large (titles)
LW     = 2.5  # line width
LW_CFD = 3.0  # CFD line width
MS     = 9    # marker size
D      = 126.0; U = 10.0; Ct = 0.75; TI = 0.08; RHO = 1.225; HUB = 90.0

plt.rcParams.update({'font.family':'serif', 'font.size':FS,
                     'axes.titlesize':FS_LG, 'axes.labelsize':FS,
                     'xtick.labelsize':FS_SM, 'ytick.labelsize':FS_SM,
                     'legend.fontsize':FS_SM})

OUTDIR = os.path.join(REPO_ROOT, 'figures', 'main')

STYLES = {
    'CFD':              dict(color='black',          lw=LW_CFD, ls='-',  marker='o', ms=MS+1),
    'LOTUSim-Jensen':   dict(color='tomato',         lw=LW,     ls='--', marker='v', ms=MS),
    'LOTUSim-Gaussian': dict(color='#DAA000',        lw=LW,     ls='--', marker='D', ms=MS),
    'LOTUSim-Larsen':   dict(color='mediumseagreen', lw=LW,     ls='--', marker='P', ms=MS),
    'LOTUSim-Blended':  dict(color='darkviolet',     lw=LW+0.5, ls='-',  marker='h', ms=MS+1),
    'FLORIS':           dict(color='royalblue',      lw=LW,     ls='--', marker='s', ms=MS),
    # ADDED FOR: "add all FLORIS models" request. Matches the
    # cornflowerblue/'^' convention already used for FLORIS Jensen
    # elsewhere in this repo (models/extended_models.py STYLES dict).
    'FLORIS-Jensen':    dict(color='cornflowerblue', lw=LW,     ls='--', marker='^', ms=MS),
}

# Alias so the consistent name 'FLORIS-Gaussian' (matching the results tables and
# 'FLORIS-Jensen') resolves to the same style as the legacy 'FLORIS' key. Both
# keys are valid, so lookups written either way work.
STYLES['FLORIS-Gaussian'] = STYLES['FLORIS']

turbines_3t = [(0.0, HUB, 0.0), (0.0, HUB, 7*D), (0.0, HUB, 14*D)]
turbines_5D = [(0.0, HUB, 0.0), (0.0, HUB, 5*D), (0.0, HUB, 10*D)]

jensen  = JensenWakeModel(diameter=D, ct=Ct, air_density=RHO, cp=0.498, cut_in=3.0, cut_out=25.0, kw=0.03)
gauss   = GaussianWakeModel(diameter=D, ct=Ct, air_density=RHO, cp=0.498, cut_in=3.0, cut_out=25.0, ambient_ti=TI, eps=0.22)
larsen  = LarsenWakeModel(diameter=D, ct=Ct, air_density=RHO, cp=0.498, ambient_ti=TI, cut_in_speed=3.0, cut_out_speed=25.0)
blend   = BlendedWakeModel(diameter=D, ct=Ct, air_density=RHO, cp=0.498, ambient_ti=TI, cut_in_speed=3.0, cut_out_speed=25.0)

def load_cfd_profiles(base, positions, n_avg=20):
    """
    Time-average wake profiles over the statistically converged tail.
    Averaging blindly over the last n_avg timesteps pulls in the startup
    transient whenever fewer than n_avg timesteps are available (e.g.
    Layout A has only 18 dumps but n_avg=20 was requested), which
    silently contaminates the "converged" CFD reference with early
    transient values. Average over whichever window is smaller: the
    last n_avg steps, or the second half of the run (standard
    transient-exclusion convention).
    """
    timesteps = sorted([int(t) for t in os.listdir(base) if t.isdigit()])
    start = max(len(timesteps) // 2, len(timesteps) - n_avg)
    avg_t = timesteps[start:]
    profiles = {}
    for pos in positions:
        u_list = []
        for t in avg_t:
            try:
                data = np.loadtxt(f'{base}/{t}/wake_{pos}_U.csv', delimiter=',', skiprows=1)
                u_list.append(data[:,1])
            except: pass
        if u_list:
            profiles[pos] = (data[:,0], np.mean(u_list, axis=0))
    return profiles

# =============================================================================
# P1-1 — CENTRELINE DECAY
# =============================================================================
print('Generating P1-1 centreline decay...')
# base_single is the genuinely-isolated single-turbine OpenFOAM case
# (one active rotor, centreline/lateral probes at 1D-14D). This is the
# CORRECT reference for a single-turbine protocol: unlike layout_a (the
# 3-turbine inline row, whose 7D/10D/14D probes coincide with turbine2/
# turbine3 and are contaminated by their induction), the single-turbine
# wake at 7D-14D shows genuine recovery (~0.71) rather than the deep
# compounded deficit seen in layout_a (~0.34-0.56). Recovered from the
# source OpenFOAM run and extracted into benchmark/data/layout_single.
base_single = os.path.join(DATA_DIR, 'layout_single', 'wakeProfiles')
base_1t = base_single
# NOTE: named p11_* rather than the shared pos_labels/pos_xD used by later
# sections (P1-2, P1-4 gradient) -- those sections still need the full
# 1D-14D position set and must not be truncated by this protocol's scope.
p11_pos_labels = ['1D', '3D', '5D', '7D', '10D', '14D']
p11_pos_xD     = [1, 3, 5, 7, 10, 14]

profs = load_cfd_profiles(base_1t, p11_pos_labels)
cfd_cl = []
for pos in p11_pos_labels:
    if pos in profs:
        y, u = profs[pos]
        cfd_cl.append(u[np.argmin(np.abs(y))] / U)
    else:
        cfd_cl.append(np.nan)

# Model centreline values
larsen_known = {1:0.490,3:0.650,5:0.720,7:0.730,10:0.810,14:0.850}
larsen_cl = [larsen_known[x] for x in p11_pos_xD]
floris_known = {1:0.258,3:0.429,5:0.610,7:0.738,10:0.834,14:0.896}
floris_cl = [floris_known[x] for x in p11_pos_xD]
# FLORIS-Jensen single-turbine centreline, computed the same way as
# floris_known above (FlorisModel, layout_x=[0.0], velocity_model='jensen',
# deflection_model='jimenez', sampled at hub height, y=0). Hardcoded here
# for the same reason floris_known/larsen_known are: avoids re-running
# FLORIS (and writing /tmp/*.yaml) every time this script is executed.
# ADDED FOR: "add all FLORIS models" request. 7D/10D/14D computed the
# same way as 1D/3D/5D (see the P1-1 table update in eval.tex).
floris_jensen_known = {1:0.558, 3:0.684, 5:0.762, 7:0.815, 10:0.866, 14:0.907}
floris_jensen_cl = [floris_jensen_known[x] for x in p11_pos_xD]

jensen_cl = []
gauss_cl  = []
for x_D in p11_pos_xD:
    x = x_D * D
    # Jensen
    R_wake = 63.0 + 0.03*x
    deficit_j = (1 - np.sqrt(1 - Ct)) * (63.0/R_wake)**2
    jensen_cl.append(1.0 - deficit_j)
    # Gaussian
    ky = 0.38*TI + 0.004
    sigma = ky*x + 0.22*D
    C = 1 - np.sqrt(max(0, 1 - Ct/(8*sigma**2/D**2)))
    gauss_cl.append(1.0 - C)

# Blended centreline: computed live from the model (not hardcoded), for
# reference only. Its Gaussian sub-model was calibrated against the same
# turbine1 wake profiles used as the CFD reference here (see
# CalibratedGaussianWakeModel), so this is an in-sample fit, not an
# independent prediction — plotted with a distinct style and excluded
# from the headline RMSE ranking.
blended_cl = [blend.velocity_at_point(U, x_D * D, 0.0) / U for x_D in p11_pos_xD]

print('P1-1 CFD centreline U/Uinf (converged tail):', [round(v, 3) if not np.isnan(v) else v for v in cfd_cl])
for name, vals in [('Jensen', jensen_cl), ('Gaussian', gauss_cl), ('Larsen', larsen_cl),
                    ('FLORIS-Gaussian', floris_cl), ('FLORIS-Jensen', floris_jensen_cl),
                    ('Blended (in-sample)', blended_cl)]:
    rmse = float(np.sqrt(np.nanmean((np.array(cfd_cl) - np.array(vals)) ** 2)))
    print(f'  {name} RMSE = {rmse:.3f}')

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('white'); ax.set_facecolor('#FAFAFA')

ax.axvspan(0, 3, alpha=0.06, color='orange')
ax.text(1.5, 0.35, 'Near-wake\n(model validity\nlimited)',
        ha='center', va='bottom', fontsize=FS_SM-1, color='darkorange', style='italic')

# Single isolated turbine at x=0 only (no downstream turbines -- this is the
# single-turbine reference case, so the old T2/T3 markers at 7D/14D that
# belonged to the 3-turbine layout are intentionally not drawn here).
ax.axvline(0, color='grey', lw=1.2, ls='--', alpha=0.5)
ax.text(0.15, 1.07, 'T1', color='grey', fontsize=FS_SM, style='italic')

ax.plot(p11_pos_xD, jensen_cl, **{**STYLES['LOTUSim-Jensen'],  'label':'LOTUSim-Jensen'})
ax.plot(p11_pos_xD, gauss_cl,  **{**STYLES['LOTUSim-Gaussian'],'label':'LOTUSim-Gaussian'})
ax.plot(p11_pos_xD, larsen_cl, **{**STYLES['LOTUSim-Larsen'],  'label':'LOTUSim-Larsen'})
ax.plot(p11_pos_xD, floris_cl, **{**STYLES['FLORIS'],          'label':'FLORIS Gaussian'})
# ADDED FOR: "add all FLORIS models" request. Remove this line (and the
# floris_jensen_known/floris_jensen_cl definitions above) if it clutters
# a 3-point plot too much -- consider the table instead in that case.
ax.plot(p11_pos_xD, floris_jensen_cl, **{**STYLES['FLORIS-Jensen'], 'label':'FLORIS Jensen'})
ax.plot(p11_pos_xD, blended_cl, **{**STYLES['LOTUSim-Blended'], 'ls':':',
        'label':'LOTUSim-Blended (in-sample fit, not independently validated here)'})
ax.plot(p11_pos_xD, cfd_cl,    **{**STYLES['CFD'],             'label':'CFD reference'})

ax.axhline(1.0, color='grey', lw=0.8, ls=':', alpha=0.5)
ax.set_xlabel('Downstream distance x/D', fontsize=FS)
ax.set_ylabel('Normalised velocity U/U\u221e', fontsize=FS)
ax.set_title('Protocol P1-1 — Wake Centreline Velocity Decay',
             fontsize=FS_LG, fontweight='bold')
ax.legend(fontsize=FS_SM, framealpha=0.95, loc='upper left')
ax.set_xticks(p11_pos_xD); ax.set_xticklabels([f'{x}D' for x in p11_pos_xD])
ax.set_xlim(-0.5, 15.5); ax.set_ylim(0.3, 1.1)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(f'{OUTDIR}/wake_centreline_decay.png', dpi=DPI, bbox_inches='tight', facecolor='white')
print('  Saved: wake_centreline_decay.png')
plt.close()

# =============================================================================
# P1-2 — LATERAL PROFILES (3-turbine)
# =============================================================================
print('Generating P1-2 lateral profiles...')
# base_3t (layout_a, 3-turbine row) is retained here because P1-3's 7D-farm
# calibration panel below genuinely needs it. P1-2 itself, however, compares
# SINGLE-TURBINE model predictions (model_lateral uses farm=[turbines_3t[0],
# probe] -- one rotor) against CFD, so its CFD reference must also be the
# single-turbine case; using layout_a here would reintroduce the exact
# turbine2/turbine3 contamination at 7D-14D that P1-1 corrects.
base_3t     = os.path.join(DATA_DIR, 'layout_a', 'wakeProfiles')
pos_labels = ['1D', '3D', '5D', '7D', '10D', '14D']
pos_xD     = [1, 3, 5, 7, 10, 14]
profs_3t = load_cfd_profiles(base_single, pos_labels)

def model_lateral(model, mtype, x, y_range=2.5, n=101):
    y_D = np.linspace(-y_range, y_range, n)
    U_norm = np.ones(n)
    for i, yd in enumerate(y_D):
        probe = (yd*D, HUB, x)
        farm  = [turbines_3t[0], probe]
        try:
            if mtype in ('jensen','gaussian'):
                _, vels, _ = model.wind_speeds_full(farm, U, [0.0, U])
            else:
                _, vels, _ = model.wind_speeds_full(farm, [0.0, U])
            U_norm[i] = vels[1] / U
        except:
            U_norm[i] = 1.0
    return y_D, U_norm

fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.patch.set_facecolor('white')
axes = axes.flatten()

# ADDED FOR: "add all FLORIS models"/"update the tables" request. RMSE
# per model, accumulated across all 6 positions by interpolating each
# model's y_D/U_norm curve onto the CFD probe's own y grid (CFD points
# are not on the same grid as the model curves). Feeds tab:p12_results.
# CFD reference is the single-turbine case (layout_single), matching the
# single-turbine model predictions computed here.
p12_sq_errors = {name: [] for name in
                  ['LOTUSim-Jensen','LOTUSim-Gaussian','LOTUSim-Larsen',
                   'LOTUSim-Blended','FLORIS-Gaussian','FLORIS-Jensen']}

for idx, (pos, x_D) in enumerate(zip(pos_labels, pos_xD)):
    ax = axes[idx]; ax.set_facecolor('#FAFAFA')
    x = x_D * D

    y_cfd_D, u_cfd_norm = None, None
    if pos in profs_3t:
        y, u = profs_3t[pos]
        ax.plot(y/D, u/U, **{**STYLES['CFD'], 'label':'CFD reference', 'marker':None})
        y_cfd_D, u_cfd_norm = y/D, u/U

    # FLORIS lateral profile
    import floris, yaml
    floris_path = os.path.dirname(floris.__file__)
    with open(floris_path+'/default_inputs.yaml') as f:
        config = yaml.safe_load(f)
    config['farm']['layout_x']=[0.0]; config['farm']['layout_y']=[0.0]
    config['farm']['turbine_type']=['nrel_5MW']
    config['flow_field']['wind_speeds']=[U]; config['flow_field']['wind_directions']=[270.0]
    config['flow_field']['turbulence_intensities']=[TI]; config['flow_field']['air_density']=RHO
    config['wake']['model_strings']['velocity_model']='gauss'
    _cfg = tmp_path('fl.yaml')
    with open(_cfg,'w') as f: yaml.dump(config,f)
    from floris import FlorisModel
    fm = FlorisModel(_cfg); fm.run()
    y_D_fl = np.linspace(-2.5, 2.5, 101)
    u_fl = np.array([fm.sample_flow_at_points(
        np.array([x]), np.array([yd*D]), np.array([HUB]))[0][0]/U
        for yd in y_D_fl])
    st = STYLES['FLORIS']
    ax.plot(y_D_fl, u_fl, color=st['color'], lw=st['lw'], ls=st['ls'], label='FLORIS Gaussian')
    if y_cfd_D is not None:
        u_fl_on_cfd = np.interp(y_cfd_D, y_D_fl, u_fl)
        p12_sq_errors['FLORIS-Gaussian'].extend((u_fl_on_cfd - u_cfd_norm) ** 2)

    # ADDED FOR: "add all FLORIS models" request. Second FLORIS run with
    # velocity_model='jensen' (+ jimenez deflection, required by FLORIS
    # for the Jensen model -- see models/extended_models.py run_floris()
    # for the same pairing). Re-uses the same single-turbine farm layout
    # as the FLORIS Gauss block above, just swapping the wake model.
    config['wake']['model_strings']['velocity_model'] = 'jensen'
    config['wake']['model_strings']['deflection_model'] = 'jimenez'
    _cfg_j = tmp_path('fl_jensen.yaml')
    with open(_cfg_j, 'w') as f: yaml.dump(config, f)
    fm_j = FlorisModel(_cfg_j); fm_j.run()
    u_fl_j = np.array([fm_j.sample_flow_at_points(
        np.array([x]), np.array([yd*D]), np.array([HUB]))[0][0]/U
        for yd in y_D_fl])
    st_j = STYLES['FLORIS-Jensen']
    ax.plot(y_D_fl, u_fl_j, color=st_j['color'], lw=st_j['lw'], ls=st_j['ls'], label='FLORIS Jensen')
    if y_cfd_D is not None:
        u_fl_j_on_cfd = np.interp(y_cfd_D, y_D_fl, u_fl_j)
        p12_sq_errors['FLORIS-Jensen'].extend((u_fl_j_on_cfd - u_cfd_norm) ** 2)

    for name, mtype, model in [
        ('LOTUSim-Jensen',   'jensen',   jensen),
        ('LOTUSim-Gaussian', 'gaussian', gauss),
        ('LOTUSim-Larsen',   'larsen',   larsen),
        ('LOTUSim-Blended',  'blended',  blend),
    ]:
        if mtype == 'blended':
            y_D = np.linspace(-2.5, 2.5, 101)
            u_n = np.array([blend.velocity_at_point(U, x, abs(yd*D))/U for yd in y_D])
            st = STYLES['LOTUSim-Blended']
            ax.plot(y_D, u_n, color=st['color'], lw=st['lw'], ls=st['ls'], label=name)
        else:
            y_D, u_n = model_lateral(model, mtype, x)
            st = STYLES[name]
            ax.plot(y_D, u_n, color=st['color'], lw=st['lw'], ls=st['ls'], label=name)
        if y_cfd_D is not None:
            u_n_on_cfd = np.interp(y_cfd_D, y_D, u_n)
            p12_sq_errors[name].extend((u_n_on_cfd - u_cfd_norm) ** 2)

    ax.axvline(-0.5, color='steelblue', lw=1.0, ls=':', alpha=0.4)
    ax.axvline( 0.5, color='steelblue', lw=1.0, ls=':', alpha=0.4)
    ax.axhline( 1.0, color='k', lw=0.7, ls=':', alpha=0.3)
    ax.set_xlabel('y/D', fontsize=FS)
    ax.set_ylabel('U/U\u221e', fontsize=FS)
    ax.set_title(f'{pos}  ({int(x_D*D)} m downstream)', fontsize=FS_LG, fontweight='bold')
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(0.2, 1.15)
    ax.grid(True, alpha=0.22)
    if idx == 0:
        ax.legend(fontsize=FS_SM, loc='lower center', ncol=2, framealpha=0.95)

fig.tight_layout()
fig.savefig(f'{OUTDIR}/wake_lateral_profiles.png', dpi=DPI, bbox_inches='tight', facecolor='white')
print('  Saved: wake_lateral_profiles.png')
plt.close()

print('P1-2 average lateral-profile RMSE vs CFD (all 6 positions, layout_single):')
for name, sq_errs in p12_sq_errors.items():
    print(f'  {name} RMSE = {np.sqrt(np.mean(sq_errs)):.4f}')

# =============================================================================
# P1-3 — WAKE-EDGE GRADIENT COMPARISON
# =============================================================================
print('Generating P1-3 gradient comparison...')

def get_cfd_grads(base, r=63.0):
    # Degrade gracefully (NaN) for CFD cases genuinely absent from the repo,
    # rather than crashing the whole script.
    if not os.path.isdir(base):
        print(f'  WARNING: CFD directory not found, skipping: {base}')
        nan_arr = np.full(len(pos_labels), np.nan)
        return nan_arr, nan_arr
    timesteps = sorted([int(t) for t in os.listdir(base) if t.isdigit()])
    # Converged tail only (second half / last 20), matching load_cfd_profiles:
    # these cases have ~18 timesteps, so a blind [-20:] would fold the startup
    # transient (t=0 uniform freestream, gradient 0) into the average and bias
    # the gradient low. Average over the statistically converged window instead.
    start = max(len(timesteps) // 2, len(timesteps) - 20)
    avg_t = timesteps[start:]
    grads=[]; cis=[]
    for pos in pos_labels:
        g=[]
        for t in avg_t:
            try:
                data=np.loadtxt(f'{base}/{t}/wake_{pos}_U.csv',delimiter=',',skiprows=1)
                y=data[:,0]; Ux=data[:,1]
                idx=np.argmin(np.abs(y-r))
                if 0<idx<len(y)-1:
                    dy=y[idx+1]-y[idx-1]
                    g.append(abs((Ux[idx+1]-Ux[idx-1])/dy/U))
            except: pass
        grads.append(np.mean(g)); cis.append(1.96*np.std(g)/np.sqrt(len(g)))
    return np.array(grads), np.array(cis)

# All three P1-3 gradient references now available in-repo:
#   single-turbine  -> layout_single
#   3-turbine 7D    -> layout_a  (base_3t)
#   3-turbine 5D    -> layout_5D (the independent hold-out case)
# The 5D case was recovered separately; its centreline signature (deep
# deficits at 5D and 10D, the T2/T3 rotor planes for 5D spacing) confirms
# it as a genuine 3-turbine 5D-spacing farm.
base_5D_path = os.path.join(DATA_DIR, 'layout_5D', 'wakeProfiles')
cfd_1t, ci_1t = get_cfd_grads(base_single)
cfd_7D, ci_7D = get_cfd_grads(base_3t)
cfd_5D, ci_5D = get_cfd_grads(base_5D_path)

r=63.0; dr=1.0
import floris, yaml
floris_path = os.path.dirname(floris.__file__)
# ADDED FOR: "add all FLORIS models" request -- velocity_model is now a
# parameter (was hardcoded to 'gauss') so this same function can produce
# both the Gauss and Jensen gradient curves below.
def floris_farm_grad(layout_x, velocity_model='gauss'):
    with open(floris_path+'/default_inputs.yaml') as f:
        config = yaml.safe_load(f)
    config['farm']['layout_x']=layout_x; config['farm']['layout_y']=[0.0]*len(layout_x)
    config['farm']['turbine_type']=['nrel_5MW']*len(layout_x)
    config['flow_field']['wind_speeds']=[U]; config['flow_field']['wind_directions']=[270.0]
    config['flow_field']['turbulence_intensities']=[TI]; config['flow_field']['air_density']=RHO
    config['wake']['model_strings']['velocity_model']=velocity_model
    if velocity_model == 'jensen':
        config['wake']['model_strings']['deflection_model'] = 'jimenez'
    _cfg_s = tmp_path('fs.yaml')
    with open(_cfg_s,'w') as f: yaml.dump(config,f)
    from floris import FlorisModel
    fm=FlorisModel(_cfg_s); fm.run()
    return [abs((fm.sample_flow_at_points(np.array([p*D]),np.array([r+dr]),np.array([HUB]))[0][0]-
                 fm.sample_flow_at_points(np.array([p*D]),np.array([r-dr]),np.array([HUB]))[0][0])/(2*dr)/U)
            for p in pos_xD]

fl_1t=floris_farm_grad([0.0])
fl_7D=floris_farm_grad([0.0,7*D,14*D])
fl_5D=floris_farm_grad([0.0,5*D,10*D])

# NOTE: FLORIS-Jensen was tried here for the "add all FLORIS models"
# request and deliberately removed again. FLORIS's Jensen implementation
# is a top-hat wake model, so |dU/dy| at a fixed radius is structurally
# ~0 almost everywhere except exactly at the wake edge -- this is the
# same reason eval.tex already excludes LOTUSim-Jensen from this protocol
# ("top-hat formulation does not define a velocity gradient at the wake
# boundary"). Adding FLORIS-Jensen here produced a degenerate near-zero
# curve, not a meaningful comparison; do not re-add without a different
# metric (e.g. wake half-width) that is actually defined for top-hat models.

g_s=[abs(gauss.velocity_gradient_at_point(U,p*D,r))/U for p in pos_xD]
l_s=[abs(larsen.velocity_gradient_at_point(U,p*D,r))/U for p in pos_xD]
b_s=[abs(blend.velocity_gradient_at_point(U,p*D,r))/U for p in pos_xD]

# Farm-scale gradients via central difference on farm_velocity_at_point.
# ADDED FOR: "add lotusim gaussian and larsen to the 7D and 5D of P1-3"
# request -- previously only Blended was shown in the farm panels.
def farm_grad(model, turbines):
    return [abs(model.farm_velocity_at_point(U,p*D,r+1,turbines)
                - model.farm_velocity_at_point(U,p*D,r-1,turbines))/(2*U)
            for p in pos_xD]
b_7D = farm_grad(blend,  turbines_3t)
b_5D = farm_grad(blend,  turbines_5D)
g_7D = farm_grad(gauss,  turbines_3t)
g_5D = farm_grad(gauss,  turbines_5D)
l_7D = farm_grad(larsen, turbines_3t)
l_5D = farm_grad(larsen, turbines_5D)

# Single-turbine gradient errors vs CFD (for P1-3 prose / tab:p14_validation)
if not np.all(np.isnan(cfd_1t)):
    print('P1-3 single-turbine |dU/dy|/Uinf, mean abs % error vs CFD (1D-14D):')
    for name, vals in [('LOTUSim-Gaussian',g_s),('LOTUSim-Larsen',l_s),
                        ('LOTUSim-Blended',b_s),('FLORIS-Gaussian',fl_1t)]:
        v = np.array(vals); mask = cfd_1t > 0
        mape = 100*np.mean(np.abs(v[mask]-cfd_1t[mask])/cfd_1t[mask])
        print(f'  {name}: {mape:.1f}%   vals={[round(x,5) for x in vals]}')
    print(f'  CFD single-turbine: {[round(x,5) for x in cfd_1t]}')

fig, axes = plt.subplots(1,3,figsize=(22,7))
fig.patch.set_facecolor('white')

panel_data = [
    (axes[0], cfd_1t, ci_1t, 'CFD reference (single turbine)', fl_1t,
     [('LOTUSim-Gaussian',g_s),('LOTUSim-Larsen',l_s),('LOTUSim-Blended',b_s)],
     'Single Turbine — Model Comparison', []),
    (axes[1], cfd_7D, ci_7D, 'CFD reference (3-turbine 7D)', fl_7D,
     [('LOTUSim-Gaussian',g_7D),('LOTUSim-Larsen',l_7D),('LOTUSim-Blended',b_7D)],
     '7D Farm — Calibration', [(7,'T2'),(14,'T3')]),
    (axes[2], cfd_5D, ci_5D, 'CFD reference (3-turbine 5D)', fl_5D,
     [('LOTUSim-Gaussian',g_5D),('LOTUSim-Larsen',l_5D),('LOTUSim-Blended',b_5D)],
     '5D Farm — Independent Validation', [(5,'T2'),(10,'T3')]),
]

for ax, cfd, ci, cfd_lbl, fl, models, title, tmarkers in panel_data:
    ax.set_facecolor('#FAFAFA')
    # ADDED FOR: "run everything" request -- skip the CFD series (and its
    # legend entry) entirely when the underlying directory was missing
    # (all-NaN), instead of drawing an invisible line with a phantom label.
    if not np.all(np.isnan(cfd)):
        ax.fill_between(pos_xD, cfd-ci, cfd+ci, alpha=0.15, color='black')
        ax.errorbar(pos_xD, cfd, yerr=ci, fmt='ko-', lw=LW_CFD, ms=MS+1,
                    capsize=5, capthick=2, elinewidth=1.5, label=cfd_lbl, zorder=6)
    for name, vals in models:
        st = STYLES[name]
        ax.plot(pos_xD, vals, color=st['color'], lw=st['lw'], ls=st['ls'],
                marker=st['marker'], ms=st['ms'], label=name)
    ax.plot(pos_xD, fl, **{**STYLES['FLORIS'], 'label':'FLORIS Gaussian'})
    for t_pos, t_lbl in tmarkers:
        ax.axvline(t_pos, color='grey', lw=1.0, ls=':', alpha=0.5)
        ax.text(t_pos+0.2, 0.0093, t_lbl, fontsize=FS_SM, color='grey', style='italic')
    ax.set_xlabel('Downstream distance x/D', fontsize=FS)
    ax.set_ylabel('|dU/dy| / U\u221e  (1/m)', fontsize=FS)
    ax.set_title(title, fontsize=FS_LG, fontweight='bold')
    ax.set_xlim(0.5,15); ax.set_ylim(0,0.010)
    ax.set_xticks(pos_xD); ax.legend(fontsize=FS_SM, framealpha=0.95)
    ax.grid(True, alpha=0.2)

fig.tight_layout()
fig.savefig(f'{OUTDIR}/drone_gradient_comparison.png', dpi=DPI, bbox_inches='tight', facecolor='white')
print('  Saved: drone_gradient_comparison.png')
plt.close()

# =============================================================================
# GRID 4x4 COMPARISON — P2-3
# =============================================================================
print('Generating P2-3 4x4 grid comparison...')
rows=[1,2,3,4]; row_labels=['R1\n(upstream)','R2\n(9D)','R3\n(18D)','R4\n(27D)']
cfd_rows=[3.964,2.046,0.619,0.190]
jensen_rows=[3.724,1.829,1.266,1.009]
gauss_rows=[3.724,2.023,1.635,1.460]
larsen_rows=[3.724,1.767,0.918,0.503]
floris_rows=[3.418,2.004,2.164,2.191]
# ADDED FOR: "add all FLORIS models" request. FLORIS-Jensen row-averaged
# power (MW), 16-turbine 4x4 grid, 7D lateral x 9D row spacing, U=10 m/s.
# Computed directly via FlorisModel (velocity_model='jensen',
# deflection_model='jimenez'), get_turbine_powers() reshaped to (row,col)
# and averaged per row -- same farm geometry as floris_rows above.
# RMSE vs CFD (per-row): 1.173 MW (cf. FLORIS-Gaussian: 1.294 MW).
floris_jensen_rows=[3.418,2.138,2.022,1.986]
# Blended: reference only — farm_velocity_at_point, no TI-rescaling/wake-meandering
blended_rows=[3.635,1.308,0.429,0.077]
cfd_ci_norm=[0.000,0.010,0.040,0.080]

def norm(vals): return [v/vals[0] for v in vals]
cfd_n=norm(cfd_rows); jensen_n=norm(jensen_rows)
gauss_n=norm(gauss_rows); larsen_n=norm(larsen_rows); floris_n=norm(floris_rows)
floris_jensen_n=norm(floris_jensen_rows)
blended_n=norm(blended_rows)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(18,7))
fig.patch.set_facecolor('white')

ax1.set_facecolor('#FAFAFA')
ax1.fill_between(rows,[c-e for c,e in zip(cfd_n,cfd_ci_norm)],
                 [c+e for c,e in zip(cfd_n,cfd_ci_norm)],alpha=0.15,color='black')
ax1.errorbar(rows,cfd_n,yerr=cfd_ci_norm,fmt='ko-',lw=LW_CFD,ms=MS+1,
             capsize=5,capthick=2,elinewidth=1.5,label='CFD reference',zorder=6)
for name,vals in [('LOTUSim-Jensen',jensen_n),('LOTUSim-Gaussian',gauss_n),
                  ('LOTUSim-Larsen',larsen_n),('FLORIS-Gaussian',floris_n),
                  ('FLORIS-Jensen',floris_jensen_n)]:
    st=STYLES[name]
    ax1.plot(rows,vals,color=st['color'],lw=st['lw'],ls=st['ls'],
             marker=st['marker'],ms=st['ms'],label='FLORIS Jensen' if name=='FLORIS-Jensen' else name)
# Blended: dashed, semi-transparent, labelled as reference only
ax1.plot(rows,blended_n,color='darkviolet',lw=1.5,ls='--',
         marker='h',ms=MS,alpha=0.55,label='LOTUSim-Blended (ref.†)')
for r,v in zip(rows,cfd_n):
    ax1.annotate(f'{v:.3f}',xy=(r,v),xytext=(r+0.06,v+0.03),fontsize=FS_SM,fontweight='bold')
ax1.set_xticks(rows); ax1.set_xticklabels(row_labels,fontsize=FS)
ax1.set_ylabel('Normalised row power P/P$_{R1}$',fontsize=FS)
ax1.set_title('Normalised Power vs Row',fontsize=FS_LG,fontweight='bold')
ax1.legend(fontsize=FS_SM,framealpha=0.95); ax1.grid(True,alpha=0.25)
ax1.set_ylim(-0.05,1.15); ax1.axhline(1.0,color='k',ls=':',lw=0.8,alpha=0.3)

ax2.set_facecolor('#FAFAFA')
# ADDED FOR: "add all FLORIS models" request -- widened from 4 to 5 bar
# groups (bar width w narrowed slightly, offsets shifted) to fit the new
# FLORIS-Jensen series without overlapping the existing bars.
x=np.arange(4); w=0.13
err_data=[
    ('LOTUSim-Jensen',  [j-c for j,c in zip(jensen_rows,cfd_rows)], 'tomato',         -2*w),
    ('LOTUSim-Gaussian',[g-c for g,c in zip(gauss_rows,cfd_rows)],  '#DAA000',        -1*w),
    ('LOTUSim-Larsen',  [l-c for l,c in zip(larsen_rows,cfd_rows)], 'mediumseagreen',  0*w),
    ('FLORIS-Gaussian',          [f-c for f,c in zip(floris_rows,cfd_rows)],  'royalblue',      +1*w),
    ('FLORIS-Jensen',   [f-c for f,c in zip(floris_jensen_rows,cfd_rows)], 'cornflowerblue', +2*w),
]
for name,errs,col,offset in err_data:
    bars=ax2.bar(x+offset,errs,w,label=name,color=col,alpha=0.88,edgecolor='k',lw=0.4)
    for bar,val in zip(bars,errs):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 val+0.03 if val>=0 else val-0.18,
                 f'{val:+.2f}',ha='center',va='bottom',fontsize=FS_SM-1)
# Blended: hatched bars, shifted to +3w now that FLORIS-Jensen occupies +2w
blended_errs=[b-c for b,c in zip(blended_rows,cfd_rows)]
bars=ax2.bar(x+3*w,blended_errs,w,label='LOTUSim-Blended (ref.†)',
             color='darkviolet',alpha=0.35,edgecolor='darkviolet',lw=0.8,hatch='//')
for bar,val in zip(bars,blended_errs):
    ax2.text(bar.get_x()+bar.get_width()/2,
             val+0.03 if val>=0 else val-0.18,
             f'{val:+.2f}',ha='center',va='bottom',fontsize=FS_SM-2,color='darkviolet')
ax2.axhline(0,color='k',lw=1.5)
ax2.set_xticks(x); ax2.set_xticklabels(row_labels,fontsize=FS)
ax2.set_ylabel('Power error vs CFD (MW)\n(positive = overestimate)',fontsize=FS)
ax2.set_title('Model Error vs CFD per Row',fontsize=FS_LG,fontweight='bold')
ax2.legend(fontsize=FS_SM,framealpha=0.95); ax2.grid(True,alpha=0.25,axis='y')
ax2.set_facecolor('#FAFAFA')
ax2.text(0.01,0.01,'†Blended: simplified pipeline (no TI-rescaling/wake-meandering); reference only, not a power candidate.',
         transform=ax2.transAxes,fontsize=FS_SM-2,color='darkviolet',va='bottom')

fig.tight_layout()
fig.savefig(f'{OUTDIR}/grid_4x4_comparison.png', dpi=DPI, bbox_inches='tight', facecolor='white')
print('  Saved: grid_4x4_comparison.png')
plt.close()

print('\nAll figures generated successfully.')

# =============================================================================
# WAKE CENTRELINE 4x4 — P1-1 (Layout B extension)
# =============================================================================
print('Generating 4x4 centreline decay...')
base_4x4 = os.path.join(DATA_DIR, 'layout_b', 'wakeProfiles')
turbines_4x4 = [((-1323+col*882), 90.0, row*1134) for row in range(4) for col in range(4)]

pos_4x4 = ['1D','3D','5D','7D','9D','11D','13D','18D','27D']
pos_4x4_xD = [1,3,5,7,9,11,13,18,27]

profs_4x4 = load_cfd_profiles(base_4x4, pos_4x4, n_avg=3)
cfd_4x4_cl = []
for pos in pos_4x4:
    if pos in profs_4x4:
        y, u = profs_4x4[pos]
        idx_cl = np.argmin(np.abs(y-(-441)))
        cfd_4x4_cl.append(u[idx_cl]/U)
    else:
        cfd_4x4_cl.append(np.nan)

gauss_4x4  = [gauss.farm_velocity_at_point(U,x*D,-441,turbines_4x4)/U for x in pos_4x4_xD]
larsen_4x4 = [larsen.farm_velocity_at_point(U,x*D,-441,turbines_4x4)/U for x in pos_4x4_xD]
blend_4x4  = [blend.farm_velocity_at_point(U,x*D,-441,turbines_4x4)/U for x in pos_4x4_xD]

# ADDED FOR: "add the lotusim jensen too" request. JensenWakeModel has no
# farm_velocity_at_point (unlike Gaussian/Larsen/Blended), only
# wind_speeds_full(turbines, ogWind, wind_vector) -- the same multi-turbine
# sequential-superposition method already used for Jensen in P1-2's
# model_lateral(). This is a real, valid farm-scale calculation (top-hat
# wake superposition is standard, not a structural exclusion like P1-3's
# gradient metric), so LOTUSim-Jensen is legitimately addable here.
# A probe is appended to the 16 real turbines; at R2/R3/R4 the probe's
# (y=-441, x=row*1134) coordinate exactly coincides with a real column-2
# turbine, so a tiny downstream epsilon disambiguates the probe entry in
# the sorted output (harmless: x_dist>1e-9 already excludes zero-distance
# self-interaction, so this only affects which duplicate we read back).
def jensen_farm_centreline(x_D):
    probe = (-441.0, HUB, x_D*D + 1e-3)
    farm = turbines_4x4 + [probe]
    turbines_sorted, velocities, _ = jensen.wind_speeds_full(farm, U, [0.0, U])
    dists = [abs(t[0]-probe[0]) + abs(t[2]-probe[2]) for t in turbines_sorted]
    return velocities[int(np.argmin(dists))]
jensen_4x4 = [jensen_farm_centreline(x)/U for x in pos_4x4_xD]

# ADDED FOR: "regenerate P1-1 fig for layout B with the two FLORIS models"
# request. Full 16-turbine FLORIS run (both velocity models), sampled at
# the same column-2 centreline (y=-441) used for the CFD/LOTUSim curves
# above. Previously this figure/protocol explicitly excluded FLORIS
# ("characterised through the power metric in P2-3 instead" -- see
# eval.tex); this adds the actual spatial-field comparison on request.
import floris, yaml
floris_path = os.path.dirname(floris.__file__)
layout_y_4x4 = [t[0] for t in turbines_4x4]   # lateral position
layout_x_4x4 = [t[2] for t in turbines_4x4]   # downstream position

def floris_4x4_centreline(velocity_model):
    with open(floris_path+'/default_inputs.yaml') as f:
        config = yaml.safe_load(f)
    config['farm']['layout_x']=layout_x_4x4; config['farm']['layout_y']=layout_y_4x4
    config['farm']['turbine_type']=['nrel_5MW']*16
    config['flow_field']['wind_speeds']=[U]; config['flow_field']['wind_directions']=[270.0]
    config['flow_field']['turbulence_intensities']=[TI]; config['flow_field']['air_density']=RHO
    config['wake']['model_strings']['velocity_model']=velocity_model
    if velocity_model == 'jensen':
        config['wake']['model_strings']['deflection_model'] = 'jimenez'
    _cfg_4 = tmp_path(f'fl_4x4_{velocity_model}.yaml')
    with open(_cfg_4,'w') as f: yaml.dump(config,f)
    from floris import FlorisModel
    fm = FlorisModel(_cfg_4); fm.run()
    return [fm.sample_flow_at_points(np.array([x*D]), np.array([-441.0]), np.array([HUB]))[0][0]/U
            for x in pos_4x4_xD]

floris_gauss_4x4  = floris_4x4_centreline('gauss')
floris_jensen_4x4 = floris_4x4_centreline('jensen')

print('4x4 centreline at R2(9D)/R3(18D)/R4(27D), for tab:4x4_centreline:')
row_labels_4x4 = {9: 'R2', 18: 'R3', 27: 'R4'}
r_idx = {r: pos_4x4_xD.index(r) for r in row_labels_4x4}
cfd_r = {r: cfd_4x4_cl[i] for r, i in r_idx.items()}
for name, vals in [('LOTUSim-Jensen', jensen_4x4),
                    ('FLORIS-Gaussian', floris_gauss_4x4), ('FLORIS-Jensen', floris_jensen_4x4)]:
    for r, i in r_idx.items():
        pct = 100 * (vals[i] - cfd_r[r]) / cfd_r[r]
        print(f'  {name} {row_labels_4x4[r]} ({r}D): {vals[i]:.2f} ({pct:+.0f}%)')

fig, ax = plt.subplots(figsize=(13,7))
fig.patch.set_facecolor('white'); ax.set_facecolor('#FAFAFA')

ax.plot(pos_4x4_xD, cfd_4x4_cl,  **{**STYLES['CFD'],             'label':'CFD reference'})
ax.plot(pos_4x4_xD, jensen_4x4,  **{**STYLES['LOTUSim-Jensen'],  'label':'LOTUSim-Jensen'})
ax.plot(pos_4x4_xD, gauss_4x4,   **{**STYLES['LOTUSim-Gaussian'],'label':'LOTUSim-Gaussian'})
ax.plot(pos_4x4_xD, larsen_4x4,  **{**STYLES['LOTUSim-Larsen'],  'label':'LOTUSim-Larsen'})
ax.plot(pos_4x4_xD, blend_4x4,   **{**STYLES['LOTUSim-Blended'], 'label':'LOTUSim-Blended'})
ax.plot(pos_4x4_xD, floris_gauss_4x4,  **{**STYLES['FLORIS'],        'label':'FLORIS Gaussian'})
ax.plot(pos_4x4_xD, floris_jensen_4x4, **{**STYLES['FLORIS-Jensen'], 'label':'FLORIS Jensen'})

for x_D, lbl in [(9,'R2'),(18,'R3'),(27,'R4')]:
    ax.axvline(x_D, color='red', lw=1.2, ls='--', alpha=0.5)
    ax.text(x_D+0.3, 0.97, lbl, color='darkred', fontsize=FS, fontweight='bold')

ax.set_xlabel('Downstream distance x/D', fontsize=FS)
ax.set_ylabel('U/U\u221e at turbine centreline', fontsize=FS)
ax.set_title('4\u00d74 Farm Wake Centreline Decay', fontsize=FS_LG, fontweight='bold')
ax.set_xlim(0,29); ax.set_ylim(0.2,1.05)
ax.set_xticks(pos_4x4_xD)
ax.legend(fontsize=FS_SM, framealpha=0.95)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(f'{OUTDIR}/wake_centreline_4x4.png', dpi=DPI, bbox_inches='tight', facecolor='white')
print('  Saved: wake_centreline_4x4.png')
plt.close()

# =============================================================================
# P2-1 — BASELINE POWER COMPARISON
# =============================================================================
print('Generating P2-1 baseline power...')
cfd_powers   = [3.806, 1.904, 0.560]
floris_g     = [3.418, 1.638, 1.803]
floris_j     = [3.418, 1.863, 1.702]
lotusim_j    = [3.724, 1.583, 0.981]
lotusim_g    = [3.724, 1.552, 1.130]
lotusim_l    = [3.724, 1.707, 0.858]
lotusim_b    = [3.803, 1.470, 0.595]

turb_labels = ['T1\n(upstream)', 'T2\n(7D)', 'T3\n(14D)']
x = np.arange(3); w = 0.11
datasets = [
    ('CFD reference',   cfd_powers, 'black'),
    ('FLORIS Gaussian',    floris_g,   'royalblue'),
    ('FLORIS Jensen',   floris_j,   'cornflowerblue'),
    ('LOTUSim-Jensen',  lotusim_j,  'tomato'),
    ('LOTUSim-Gaussian',lotusim_g,  '#DAA000'),
    ('LOTUSim-Larsen',  lotusim_l,  'mediumseagreen'),
    ('LOTUSim-Blended', lotusim_b,  'darkviolet'),
]
offsets = np.linspace(-3*w, 3*w, 7)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,7))
fig.patch.set_facecolor('white')

ax1.set_facecolor('#FAFAFA')
for (name, vals, col), offset in zip(datasets, offsets):
    ax1.bar(x+offset, vals, w, label=name, color=col, alpha=0.88, edgecolor='k', lw=0.4)
ax1.set_xticks(x); ax1.set_xticklabels(turb_labels, fontsize=FS)
ax1.set_ylabel('Power (MW)', fontsize=FS)
ax1.set_title('Per-Turbine Absolute Power', fontsize=FS_LG, fontweight='bold')
ax1.legend(fontsize=FS_SM, framealpha=0.95, loc='upper right')
ax1.grid(True, axis='y', alpha=0.25)

positions_D = [0, 7, 14]
ax2.set_facecolor('#FAFAFA')
cfd_norm = [p/cfd_powers[0] for p in cfd_powers]
ax2.plot(positions_D, cfd_norm, **{**STYLES['CFD'], 'label':'CFD reference'})
for name, vals, col in datasets[1:]:
    norm = [p/vals[0] for p in vals]
    st = STYLES.get(name, dict(color=col, lw=LW, ls='--', marker='o', ms=MS))
    ax2.plot(positions_D, norm, color=col, lw=st['lw'], ls=st['ls'],
             marker=st['marker'], ms=st['ms'], label=name)

for i, (v, lbl) in enumerate(zip(cfd_norm, ['CFD: 1.00','CFD: 0.50','CFD: 0.15'])):
    ax2.annotate(lbl, xy=(positions_D[i], v),
                 xytext=(positions_D[i]+0.3, v+0.03),
                 fontsize=FS_SM-1, fontweight='bold')

ax2.set_xlabel('Downstream distance (D)', fontsize=FS)
ax2.set_ylabel('Normalised power P/P_T1', fontsize=FS)
ax2.set_title('Wake Loss Profile', fontsize=FS_LG, fontweight='bold')
ax2.set_xticks([0,7,14]); ax2.set_xticklabels(['T1 (0D)','T2 (7D)','T3 (14D)'], fontsize=FS)
ax2.legend(fontsize=FS_SM, framealpha=0.95)
ax2.grid(True, alpha=0.25); ax2.set_ylim(0, 1.1)

fig.tight_layout()
fig.savefig(f'{OUTDIR}/wake_comparison_power.png', dpi=DPI, bbox_inches='tight', facecolor='white')
print('  Saved: wake_comparison_power.png')
plt.close()

# =============================================================================
# P1-3 MULTI-SPEED — BLENDED VALIDATION
# =============================================================================
# Single-turbine CFD at three inflow speeds, each with its own Ct. Only the
# 10 m/s case (layout_single) ships with this repository; the 8 and 12 m/s
# sensitivity runs are available on request. Drop their extracted
# wakeProfiles trees at the paths below and this figure regenerates in full.
print('Generating P1-3 multi-speed validation...')
cases = {
    8:  {'path': os.path.join(DATA_DIR, 'layout_single_8ms',  'wakeProfiles'), 'ct': 0.80},
    10: {'path': os.path.join(DATA_DIR, 'layout_single',      'wakeProfiles'), 'ct': 0.75},
    12: {'path': os.path.join(DATA_DIR, 'layout_single_12ms', 'wakeProfiles'), 'ct': 0.53},
}
speed_colors = {8:'tomato', 10:'darkviolet', 12:'royalblue'}
r=63.0
missing_speeds = []

fig, ax = plt.subplots(figsize=(13,7))
fig.patch.set_facecolor('white'); ax.set_facecolor('#FAFAFA')

for U_inf, case in cases.items():
    b = BlendedWakeModel(diameter=D, ct=case['ct'], air_density=RHO, cp=0.498,
                          ambient_ti=TI, cut_in_speed=3.0, cut_out_speed=25.0)
    base = case['path']
    if not os.path.isdir(base):
        missing_speeds.append(U_inf)
        print(f'  NOTE: no CFD data for U={U_inf} m/s at {base} -- series skipped')
        continue
    timesteps = sorted([int(t) for t in os.listdir(base) if t.isdigit()])
    avg_t = timesteps[-20:]
    cfd_g=[]; cfd_ci=[]; blend_g=[]
    for pos in pos_labels:
        grads=[]
        for t in avg_t:
            try:
                data=np.loadtxt(f'{base}/{t}/wake_{pos}_U.csv',delimiter=',',skiprows=1)
                y=data[:,0]; Ux=data[:,1]
                idx=np.argmin(np.abs(y-r))
                if 0<idx<len(y)-1:
                    dy=y[idx+1]-y[idx-1]
                    grads.append(abs((Ux[idx+1]-Ux[idx-1])/dy/U_inf))
            except: pass
        cfd_g.append(np.mean(grads))
        cfd_ci.append(1.96*np.std(grads)/np.sqrt(len(grads)))
    for x_D in pos_xD:
        blend_g.append(abs(b.velocity_gradient_at_point(U_inf,x_D*D,r))/U_inf)
    col = speed_colors[U_inf]
    ax.errorbar(pos_xD, cfd_g, yerr=cfd_ci, fmt='o-', color=col,
                lw=LW_CFD, ms=MS, capsize=5, capthick=2, elinewidth=1.5,
                label=f'CFD reference {U_inf} m/s', zorder=6)
    ax.plot(pos_xD, blend_g, 'h--', color=col, lw=LW, ms=MS-1, alpha=0.75,
            label=f'LOTUSim-Blended {U_inf} m/s')

ax.set_xlabel('Downstream distance x/D', fontsize=FS)
ax.set_ylabel('|dU/dy| / U\u221e  (1/m)', fontsize=FS)
ax.set_title('LOTUSim-Blended Multi-Speed Validation', fontsize=FS_LG, fontweight='bold')
ax.set_xlim(0.5,15); ax.set_ylim(0.0015,0.0055)
ax.set_xticks(pos_xD)
ax.legend(fontsize=FS_SM, framealpha=0.95, ncol=2)
ax.grid(True, alpha=0.2)
fig.tight_layout()
if missing_speeds:
    # Writing a partial figure here would silently replace the committed
    # three-speed version with an incomplete one, so leave it untouched.
    print(f'  SKIPPED writing blended_multispeed_validation.png: CFD data missing '
          f'for U = {missing_speeds} m/s. The committed figure was produced with '
          f'all three speeds and is left in place. Add the missing cases under '
          f'{DATA_DIR} (layout_single_8ms / layout_single_12ms) to regenerate it.')
else:
    fig.savefig(f'{OUTDIR}/blended_multispeed_validation.png', dpi=DPI,
                bbox_inches='tight', facecolor='white')
    print('  Saved: blended_multispeed_validation.png')
plt.close()

print('\nAll figures complete.')

# =============================================================================
# P2-2 — WIND SPEED SWEEP (normalised)
# =============================================================================
print('Generating P2-2 wind speed sweep...')

# Real Cp/Ct curves from FLORIS
import floris, yaml
floris_path = os.path.dirname(floris.__file__)
with open(floris_path+'/default_inputs.yaml') as f:
    config_base = yaml.safe_load(f)

from floris.turbine_library import TurbineInterface
nrel = config_base['farm']['turbine_type']

def get_cp(u):
    speeds = [3,4,5,6,7,8,9,10,11,11.4,12,13,14,15,25]
    cps    = [0.0,0.178,0.330,0.450,0.482,0.498,0.498,0.498,0.498,0.498,0.390,0.280,0.210,0.150,0.0]
    return float(np.interp(u, speeds, cps))

def get_ct(u):
    speeds = [3,4,5,6,7,8,9,10,11,11.4,12,13,14,15,25]
    cts    = [0.0,0.820,0.810,0.800,0.790,0.780,0.770,0.750,0.700,0.640,0.530,0.410,0.330,0.260,0.0]
    return float(np.interp(u, speeds, cts))

def power_mw(u):
    A = np.pi*(D/2)**2
    return max(0.0, 0.5*RHO*A*get_cp(u)*u**3/1e6)

wind_speeds = [5,6,7,8,9,10,11,12,13,14,15]
wv = [0.0, 1.0]

results = {n: {'T1':[],'T2':[],'T3':[]} for n in
           ['LOTUSim-Jensen','LOTUSim-Gaussian','LOTUSim-Larsen','FLORIS']}

for U_s in wind_speeds:
    ct_s = get_ct(U_s); cp_s = get_cp(U_s)
    wv_s = [0.0, U_s]
    j = JensenWakeModel(diameter=D,ct=ct_s,air_density=RHO,cp=cp_s,cut_in=3.0,cut_out=25.0,kw=0.03)
    g = GaussianWakeModel(diameter=D,ct=ct_s,air_density=RHO,cp=cp_s,cut_in=3.0,cut_out=25.0,ambient_ti=TI,eps=0.22)
    l = LarsenWakeModel(diameter=D,ct=ct_s,air_density=RHO,cp=cp_s,ambient_ti=TI,cut_in_speed=3.0,cut_out_speed=25.0)

    _,vj,_ = j.wind_speeds_full(turbines_3t,U_s,wv_s)
    _,vg,_ = g.wind_speeds_full(turbines_3t,U_s,wv_s)
    _,vl,_ = l.wind_speeds_full(turbines_3t,wv_s)

    for name, vels in [('LOTUSim-Jensen',vj),('LOTUSim-Gaussian',vg),('LOTUSim-Larsen',vl)]:
        for ti, tk in enumerate(['T1','T2','T3']):
            results[name][tk].append(power_mw(vels[ti]))

    # FLORIS
    try:
        cfg = yaml.safe_load(open(floris_path+'/default_inputs.yaml'))
        cfg['farm']['layout_x']=[t[2] for t in turbines_3t]
        cfg['farm']['layout_y']=[t[0] for t in turbines_3t]
        cfg['farm']['turbine_type']=['nrel_5MW']
        cfg['flow_field']['wind_speeds']=[U_s]
        cfg['flow_field']['wind_directions']=[270.0]
        cfg['flow_field']['turbulence_intensities']=[TI]
        cfg['wake']['model_strings']['velocity_model']='gauss'
        _cfg_w = tmp_path('fws.yaml')
        with open(_cfg_w,'w') as f: yaml.dump(cfg,f)
        from floris import FlorisModel
        fm=FlorisModel(_cfg_w); fm.run()
        fp=[p/1e6 for p in fm.get_turbine_powers().flatten()]
        for ti,tk in enumerate(['T1','T2','T3']):
            results['FLORIS'][tk].append(fp[ti])
    except:
        for tk in ['T1','T2','T3']:
            results['FLORIS'][tk].append(0)

# CFD reference
cfd_data = {
    6:  {'T1':0.943,'T2':0.676,'T3':0.563},
    8:  {'T1':2.215,'T2':1.590,'T3':1.314},
    10: {'T1':3.806,'T2':1.904,'T3':None},
    12: {'T1':7.243,'T2':5.231,'T3':4.325},
    14: {'T1':11.216,'T2':8.145,'T3':6.742},
}
cfd_speeds = sorted(cfd_data.keys())

colors2 = {
    'LOTUSim-Jensen':   ('tomato',         'v', '--'),
    'LOTUSim-Gaussian': ('#DAA000',        'D', '--'),
    'LOTUSim-Larsen':   ('mediumseagreen', 'P', '--'),
    'FLORIS':           ('royalblue',      's', '-'),
}

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig2.patch.set_facecolor('white')

for ax, turb, title in [(ax1,'T2','T2/T1 — 7D downstream'),(ax2,'T3','T3/T1 — 14D downstream')]:
    ax.set_facecolor('#FAFAFA')
    for name, (col, marker, ls) in colors2.items():
        t1v = results[name]['T1']
        norm = [results[name][turb][i]/t1v[i] if t1v[i]>0 else 0 for i in range(len(wind_speeds))]
        ax.plot(wind_speeds, norm, marker=marker, color=col, lw=LW, ls=ls, ms=MS, label=name)
    cfd_ns = [U_s for U_s in cfd_speeds if cfd_data[U_s][turb] is not None]
    cfd_nv = [cfd_data[U_s][turb]/cfd_data[U_s]['T1'] for U_s in cfd_ns]
    ax.scatter(cfd_ns, cfd_nv, color='black', marker='*', s=250, zorder=6, label='CFD reference')
    ax.axvline(11.4, color='grey', ls=':', lw=1.0, alpha=0.6, label='Rated (11.4 m/s)')
    ax.set_xlabel('Wind speed (m/s)', fontsize=FS)
    ax.set_ylabel('P / P_T1', fontsize=FS)
    ax.set_title(title, fontsize=FS_LG, fontweight='bold')
    ax.legend(fontsize=FS_SM, framealpha=0.95)
    ax.set_xticks(wind_speeds); ax.set_ylim(0,1.1)
    ax.axhline(1.0, color='k', ls=':', lw=0.8, alpha=0.4)
    ax.grid(True, alpha=0.25)

fig2.suptitle('Protocol P2-2 — Wind Speed Sweep', fontsize=FS_LG, fontweight='bold')
fig2.tight_layout()
fig2.savefig(f'{OUTDIR}/wind_speed_sweep_normalised.png', dpi=DPI, bbox_inches='tight', facecolor='white')
print('  Saved: wind_speed_sweep_normalised.png')
plt.close()

print('\nAll figures complete.')
