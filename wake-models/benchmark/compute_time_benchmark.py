#!/usr/bin/env python3
"""
Computational-cost benchmark: per-evaluation wall-clock time to predict the
per-turbine power of the 3-turbine Layout A row (7D spacing, NREL 5MW,
U_inf = 10 m/s), for the four LOTUSim engineering models and for FLORIS v4 in
BOTH its Gaussian and Jensen velocity-model configurations.

Methodology
-----------
- The "task" is one full power prediction of the 3-turbine farm at the baseline
  condition (the same task as protocol P2-1 / Table tab:compute_performance).
- Model construction / FLORIS FlorisModel construction is one-time SETUP and is
  excluded from the timing; we time only the repeated evaluation, which is the
  cost paid every step inside the LOTUSim-Energy loop.
- Each model is warmed up once, then timed over N repetitions; we report the
  mean and the min (min is the cleanest estimate of pure compute cost, least
  perturbed by OS scheduling).

The four LOTUSim model classes and run_floris() are loaded from
models/extended_models.py WITHOUT executing that file's plotting/script body
(we exec the source up to the first module-level side effect).
"""
import matplotlib
matplotlib.use("Agg")          # never pop up / block on plt.show()
import os, sys, tempfile, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Load the production model classes + run_floris() from extended_models.py,
# stopping before its module-level script body (FLORIS runs + figure code).
# ---------------------------------------------------------------------------
src_path = os.path.join(ROOT, "models", "extended_models.py")
src = open(src_path, encoding="utf-8").read()
MARKER = '\nprint("Running FLORIS Gaussian...")'
assert MARKER in src, "extended_models.py layout changed; update the marker."
src_defs = src[: src.index(MARKER)]

ns = {"__name__": "extended_models_defs", "__file__": src_path}
exec(compile(src_defs, src_path, "exec"), ns)

JensenWakeModel   = ns["JensenWakeModel"]
GaussianWakeModel = ns["GaussianWakeModel"]
LarsenWakeModel   = ns["LarsenWakeModel"]
BlendedWakeModel  = ns["BlendedWakeModel"]
run_floris        = ns["run_floris"]          # builds+runs a FlorisModel each call

# Baseline constants (as defined in extended_models.py)
D, HUB, U_INF, TI, RHO, CT, CP = (ns["D"], ns["HUB_HEIGHT"], ns["U_INF"],
                                  ns["TI"], ns["RHO"], ns["CT"], ns["CP_ENG"])
turbines    = ns["turbines"]        # 3-turbine row, 7D
wind_vector = ns["wind_vector"]     # [0.0, U_INF]

print(f"Baseline: D={D} m, U_inf={U_INF} m/s, TI={TI}, Ct={CT}, Cp={CP}, "
      f"{len(turbines)}-turbine row")

# ---------------------------------------------------------------------------
# Build the evaluation closures. Construction = one-time setup (excluded).
# Each closure performs one full 3-turbine power prediction.
# ---------------------------------------------------------------------------
def P(u):                                   # power (MW) from effective inflow
    return 0.5 * RHO * np.pi * (D / 2) ** 2 * CP * u ** 3 / 1e6

# LOTUSim-Jensen
mj = JensenWakeModel(diameter=D, ct=CT, air_density=RHO, cp=CP,
                     cut_in=3.0, cut_out=25.0, kw=0.03)
def eval_jensen():
    _, v, _ = mj.wind_speeds_full(turbines, U_INF, wind_vector)
    return [P(x) for x in v]

# LOTUSim-Gaussian
mg = GaussianWakeModel(diameter=D, ct=CT, air_density=RHO, cp=CP,
                       cut_in=3.0, cut_out=25.0, ambient_ti=TI, eps=0.22)
def eval_gaussian():
    _, v, _ = mg.wind_speeds_full(turbines, U_INF, wind_vector)
    return [P(x) for x in v]

# LOTUSim-Larsen
ml = LarsenWakeModel(diameter=D, ct=CT, air_density=RHO, cp=CP,
                     ambient_ti=TI, cut_in_speed=3.0, cut_out_speed=25.0)
def eval_larsen():
    _, v, _ = ml.wind_speeds_full(turbines, wind_vector)
    return [P(x) for x in v]

# LOTUSim-Blended
mb = BlendedWakeModel(diameter=D, ct=CT, air_density=RHO, cp=CP,
                      ambient_ti=TI, cut_in_speed=3.0, cut_out_speed=25.0)
def eval_blended():
    ups, out = [], []
    for i, (x_t, y_t, z_t) in enumerate(turbines):
        u = U_INF if i == 0 else mb.farm_velocity_at_point(U_INF, z_t, x_t, ups)
        out.append(P(u))
        ups.append((x_t, y_t, z_t))
    return out

# ---------------------------------------------------------------------------
# FLORIS: build the FlorisModel ONCE per velocity model (setup, excluded),
# then time only fm.run() + get_turbine_powers().
# ---------------------------------------------------------------------------
def build_floris(velocity_model):
    import floris, yaml
    fp = os.path.dirname(floris.__file__)
    with open(fp + "/default_inputs.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["farm"]["layout_x"] = [t[2] for t in turbines]
    cfg["farm"]["layout_y"] = [t[0] for t in turbines]
    cfg["farm"]["turbine_type"] = ["nrel_5MW"]
    cfg["flow_field"]["wind_speeds"] = [U_INF]
    cfg["flow_field"]["wind_directions"] = [270.0]
    cfg["flow_field"]["turbulence_intensities"] = [TI]
    cfg["flow_field"]["air_density"] = RHO
    cfg["wake"]["model_strings"]["velocity_model"] = velocity_model
    if velocity_model == "jensen":
        cfg["wake"]["model_strings"]["deflection_model"] = "jimenez"
        cfg["wake"]["enable_secondary_steering"] = False
        cfg["wake"]["enable_yaw_added_recovery"] = False
        cfg["wake"]["enable_transverse_velocities"] = False
    tmp = os.path.join(tempfile.gettempdir(), f"floris_time_{velocity_model}.yaml")
    with open(tmp, "w") as f:
        yaml.dump(cfg, f)
    from floris import FlorisModel
    return FlorisModel(tmp)

fm_g = build_floris("gauss")
fm_j = build_floris("jensen")
def eval_floris_gauss():
    fm_g.run(); return [p / 1e6 for p in fm_g.get_turbine_powers().flatten()]
def eval_floris_jensen():
    fm_j.run(); return [p / 1e6 for p in fm_j.get_turbine_powers().flatten()]

# ---------------------------------------------------------------------------
# Timing driver
# ---------------------------------------------------------------------------
def timeit(fn, n):
    fn()                                    # warm-up (excluded)
    ts = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        fn()
        ts[i] = time.perf_counter() - t0
    return ts.mean() * 1e3, ts.min() * 1e3   # ms

N_LOT = 2000     # LOTUSim models are cheap -> many reps
N_FLO = 300      # FLORIS ~ms/run -> fewer reps, still robust

cases = [
    ("LOTUSim-Jensen",   eval_jensen,        N_LOT),
    ("LOTUSim-Gaussian", eval_gaussian,      N_LOT),
    ("LOTUSim-Larsen",   eval_larsen,        N_LOT),
    ("LOTUSim-Blended",  eval_blended,       N_LOT),
    ("FLORIS-Gaussian",  eval_floris_gauss,  N_FLO),
    ("FLORIS-Jensen",    eval_floris_jensen, N_FLO),
]

# sanity: print the powers each method predicts (should match the reported tables)
print("\nPredicted per-turbine power (MW) [T1, T2, T3]:")
for name, fn, _ in cases:
    print(f"  {name:<17}: {[round(x,3) for x in fn()]}")

print("\nComputational cost (3-turbine power prediction, model reused):")
print(f"  {'Model':<17} {'mean (ms)':>11} {'min (ms)':>10} {'reps':>7}")
res = {}
for name, fn, n in cases:
    mean_ms, min_ms = timeit(fn, n)
    res[name] = (mean_ms, min_ms)
    print(f"  {name:<17} {mean_ms:>11.4f} {min_ms:>10.4f} {n:>7}")

# Speedups vs each FLORIS configuration (mean)
fg = res["FLORIS-Gaussian"][0]
fj = res["FLORIS-Jensen"][0]
print("\nSpeedup vs FLORIS (mean):")
print(f"  {'Model':<17} {'x vs FLORIS-Gauss':>18} {'x vs FLORIS-Jensen':>19}")
for name in ["LOTUSim-Jensen", "LOTUSim-Gaussian", "LOTUSim-Larsen", "LOTUSim-Blended"]:
    m = res[name][0]
    print(f"  {name:<17} {fg/m:>17.1f}x {fj/m:>18.1f}x")
print(f"  {'FLORIS-Jensen':<17} {fg/fj:>17.2f}x {'1.00x':>19}")

# ---------------------------------------------------------------------------
# SPATIAL-FIELD GENERATION: LOTUSim-Blended vs FLORIS-Gaussian.
# "NxN" is the number of hub-height query points (N downstream x N lateral),
# NOT a physical size. We sample a fixed hub-height domain covering the
# 3-turbine row and its wake: downstream x in [-100, 2400] m, lateral
# y in [-400, 400] m, at z = hub height. FLORIS-Jensen is excluded: its
# top-hat profile has no defined lateral gradient, so it is not a comparator
# for the spatial-field / gradient use case.
# ---------------------------------------------------------------------------
DOWNSTREAM = (-100.0, 2400.0)   # metres (streamwise), covers T1..T3 + wake
LATERAL    = (-400.0, 400.0)    # metres (cross-stream), ~ +/- 3.2 D

def grid(n):
    X, Y = np.meshgrid(np.linspace(*DOWNSTREAM, n), np.linspace(*LATERAL, n))
    return X, Y

def blended_field(n):
    X, Y = grid(n)                                   # grid build = setup (excluded)
    return lambda: mb.farm_velocity_field(U_INF, X, Y, turbines)

def floris_field(n):
    X, Y = grid(n)
    xf, yf = X.ravel(), Y.ravel()
    zf = np.full(xf.shape, HUB)
    fm_g.run()                                       # solve once (setup, excluded)
    return lambda: fm_g.sample_flow_at_points(xf, yf, zf)

print("\nSpatial-field generation (hub-height velocity field; Jensen excluded):")
print(f"  domain: downstream {DOWNSTREAM} m x lateral {LATERAL} m, z=hub")
print(f"  {'Grid':<10} {'points':>8} {'Blended min (ms)':>18} "
      f"{'FLORIS-G min (ms)':>18} {'speedup':>9}")
for n in (100, 200):
    tb = timeit(blended_field(n), 200)[1]            # min (ms)
    tf = timeit(floris_field(n), 40)[1]              # min (ms)
    res_m = f"{n}m" if False else f"{DOWNSTREAM[1]-DOWNSTREAM[0]:.0f}x{LATERAL[1]-LATERAL[0]:.0f}m"
    print(f"  {n}x{n:<6} {n*n:>8} {tb:>18.3f} {tf:>18.3f} {tf/tb:>8.1f}x")
print(f"  (each grid spans the same {DOWNSTREAM[1]-DOWNSTREAM[0]:.0f} m x "
      f"{LATERAL[1]-LATERAL[0]:.0f} m physical domain; NxN is the point count / "
      f"resolution)")
