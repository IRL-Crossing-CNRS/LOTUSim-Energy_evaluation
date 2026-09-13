#!/usr/bin/env python3
"""2x2 factorial analysis of the M1 transect: mean field x turbulence.

    python3 factorial_analysis.py [--manifest factorial_manifest.txt]

Cells (see factorial.sh):
    A uniform mean, ambient TI      B uniform mean, wake TI
    C wake mean,    ambient TI      D wake mean,    wake TI

Contrasts, all paired within seed (every seed contributes all four cells, so
each contrast is a within-seed difference and the gust realisation cancels):

    B - A          effect of turbulence, with no deficit
    C - A          effect of the deficit, with no added turbulence
    D - A          the real wake, both factors together
    D - B - C + A  interaction: is a gust worse inside a deficit than the same
                   gust in clean air, beyond what the larger sigma explains?

LEG SEGMENTATION.  The mission is an out-and-back along the wind, so its two
legs sit at very different tilts (~19 deg downwind, ~35-41 deg upwind) and must
never be pooled — how much of each a run captures depends on PX4 startup
jitter, which on the earlier campaign flipped the answer outright (p=0.58
pooled vs p=0.0156 per leg on identical data).

The run is also longer than one out-and-back, so the log holds a partial second
lap. Taking every northbound sample would pool complete legs with a truncated
one and reintroduce exactly that bias, so this uses every COMPLETE block in
each direction and discards any block cut off by the end of the log. Complete
blocks are interchangeable — same path, same field — so pooling them is free
statistics; it is only the truncated one that biases.

The same split serves all three missions, but not on the same axis. M1 transits
along the wind, so its two directions are downwind and upwind legs on the
north axis. M3 traverses across the wind: its north velocity never leaves
+/-3 m/s while its east velocity spans +/-12, so splitting on north yields no
legs at all. The axis is therefore chosen per run as the one the vehicle
actually travels along -- the larger positional extent -- with a near-tie
falling back to north, which is the case for M2, whose orbit is equally wide
in both and whose meaningful halves are the upwind and downwind ones.
"""
from __future__ import annotations
import argparse, os, sys
from collections import defaultdict
import numpy as np
from pyulog import ULog

def run_dir(path: str) -> str:
    """Expand a manifest's run directory.

    The committed manifests record where each run's logs sat on the machine
    that flew it, with that machine's workspace written as the literal
    ``<scenario_ws>``. Substituting $LOTUSIM_SCENARIO_WS makes them usable on a
    machine that holds the same logs under a different root, and leaves an
    already-absolute path alone.
    """
    path = path.strip().rstrip("/")
    if "<scenario_ws>" in path:
        ws = os.environ.get("LOTUSIM_SCENARIO_WS", "")
        path = path.replace("<scenario_ws>", ws)
    return path

CELLS = ("A", "B", "C", "D")
CELL_DESC = {"A": "uniform mean, ambient TI", "B": "uniform mean, wake TI",
             "C": "wake mean, ambient TI", "D": "wake mean, wake TI"}
# Labels for M1's two vehicles; other missions have more (M3 flies four), so
# anything not listed is reported by its instance number rather than mislabelled.
INSTANCES = {0: "IN-WAKE", 1: "CLEAR-AIR"}
LEG_NAMES = {"north": ("NORTHBOUND", "SOUTHBOUND"), "east": ("EASTBOUND", "WESTBOUND")}
from ulog_extract import KT, KQ, W_MAX, ETA  # x500_px4/model.sdf


def _orbit_laps(e, n, t):
    """Complete laps of a closed orbit, as a list of index arrays, or None if
    the path is not an orbit.

    A closed orbit has no legs: splitting it on velocity sign the way a transit
    is split chops each half into fragments every time the vehicle crosses the
    threshold, which on an 8 m/s orbit happens constantly (observed: 15-33
    fragments per flight, most under 2 s). The natural unit is instead one
    whole lap, which averages over every heading by construction -- and is also
    the operationally meaningful quantity, since an inspection orbit is flown
    as laps.

    Laps are counted on the unwrapped azimuth about the path centroid, so the
    cut is geometric and immune to velocity noise near a threshold.
    """
    cx, cy = e.mean(), n.mean()
    # Accumulated azimuth alone is not enough. A there-and-back transit runs
    # out and back along one line, so its centroid sits ON the path: every
    # pass through the middle swings the azimuth by pi, and two passes look
    # like a full lap. Four of M3's 32 runs were classified as orbits that
    # way, which drops those seeds from the paired analysis entirely.
    #
    # A real orbit never approaches its own centre, and the separation is not
    # marginal: r_min/r_max is 0.88 on an M2 orbit and 0.001 on an M3 transit.
    r = np.hypot(e - cx, n - cy)
    if r.max() <= 0.0 or r.min() / r.max() < 0.3:
        return None
    phi = np.unwrap(np.arctan2(n - cy, e - cx))
    span = phi[-1] - phi[0]
    if abs(span) < 2 * np.pi:
        return None
    direction = 1.0 if span > 0 else -1.0
    phi_d = direction * phi                       # monotonically increasing
    start = phi_d[0]
    edges = []
    k = 1
    while True:
        idx = np.searchsorted(phi_d, start + k * 2 * np.pi)
        if idx >= len(phi_d):
            break
        edges.append(idx); k += 1
    if len(edges) < 2:
        return None
    return [np.arange(a, b) for a, b in zip(edges[:-1], edges[1:])]


def _blocks(mask: np.ndarray):
    """Contiguous True runs as (start, stop) index pairs."""
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def legs(path: str, alt_min=45.0, v_min=5.0, full_frac=0.8, min_secs=10.0):
    u = ULog(path, ["vehicle_attitude_groundtruth", "vehicle_local_position_groundtruth",
                    "actuator_motors"])
    have = {d.name: d.data for d in u.data_list}
    a = have["vehicle_attitude_groundtruth"]
    t = np.array(a["timestamp"], dtype=float) * 1e-6
    w, x, y, z = a["q[0]"], a["q[1]"], a["q[2]"], a["q[3]"]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    tilt = np.degrees(np.arccos(np.clip(np.cos(roll) * np.cos(pitch), -1, 1)))

    p = have["vehicle_local_position_groundtruth"]
    tp = np.array(p["timestamp"], dtype=float) * 1e-6
    # Interpolated, never index-paired: attitude logs ~57 Hz, position 50 Hz.
    alt = np.interp(t, tp, -np.array(p["z"]))
    # world North = px4 x, world East = px4 y.
    pos_n, pos_e = np.array(p["x"]), np.array(p["y"])
    v_north = np.interp(t, tp, np.array(p["vx"]))
    v_east = np.interp(t, tp, np.array(p["vy"]))
    # Split on the axis the vehicle actually travels along. A near-tie (an
    # orbit is equally wide both ways) falls back to north, so M2's halves stay
    # the upwind and downwind ones rather than an arbitrary choice.
    ext_n, ext_e = np.ptp(pos_n), np.ptp(pos_e)   # ndarray.ptp() removed in NumPy 2
    axis = "east" if ext_e > 1.2 * ext_n else "north"
    v_along = v_east if axis == "east" else v_north
    fwd, back = LEG_NAMES[axis]

    power = None
    rotor = None
    if "actuator_motors" in have:
        m = have["actuator_motors"]
        tm = np.array(m["timestamp"], dtype=float) * 1e-6
        ctrl = np.stack([np.clip(np.array(m[f"control[{i}]"], dtype=float), 0, 1) for i in range(4)])
        # P_shaft = kQ*kT*w^3 per rotor, w = control * maxRotVelocity; /eta electrical.
        p_el = (KQ * KT * (ctrl * W_MAX) ** 3).sum(axis=0) / ETA
        power = np.interp(t, tm, p_el)
        rotor = np.interp(t, tm, ctrl.max(axis=0))

    cruise = alt > alt_min
    dt = np.gradient(t)
    out = {}

    # An orbit is analysed as whole laps; a transit as complete legs.
    laps = _orbit_laps(pos_e[np.searchsorted(tp, t[cruise])[:cruise.sum()] - 1],
                       pos_n[np.searchsorted(tp, t[cruise])[:cruise.sum()] - 1],
                       t[cruise]) if cruise.sum() > 100 else None
    if laps is not None:
        cidx = np.flatnonzero(cruise)
        sel = np.concatenate([cidx[l] for l in laps])
        srs = tilt[sel]
        rec = dict(n=int(srs.size), blocks=len(laps), secs=float(dt[sel].sum()),
                   mean=float(srs.mean()), sd=float(srs.std(ddof=1)),
                   p95=float(np.percentile(srs, 95)),
                   sat=float(100.0 * (srs > 44.0).mean()), var=float(srs.var(ddof=1)))
        rec["power"] = float(np.mean(power[sel])) if power is not None else float("nan")
        # Energy actually drawn over the pattern, in Wh: the number that sets
        # endurance. On an orbit this is the metric that survives when the mean
        # does not -- the vehicle flies every heading, so the mean-field effect
        # on attitude cancels across the lap, but power goes as the cube of
        # rotor speed and a cubic does not average out. M2's deficit moves mean
        # attitude by -2.4 deg (easy to dismiss) and mean power by -10.8 W in
        # 8 seeds out of 8 (not).
        rec["energy"] = (float(np.trapezoid(power[sel], t[sel])) / 3600.0
                         if power is not None else float("nan"))
        rec["rotor"] = (float(np.percentile(rotor[sel], 99))
                        if rotor is not None else float("nan"))
        # Along-pattern coordinate for the profile subtraction. On an orbit that
        # is the azimuth: "where around the lap" is what position means there.
        j = np.clip(np.searchsorted(tp, t[sel]) - 1, 0, len(tp) - 1)
        # Centre on the ORBIT, not on the whole flight: pos_*.mean() over every
        # sample includes climb-out and the transit to the pattern, which pulls
        # the centre off the circle and makes the azimuth, and therefore the
        # profile subtraction, meaningless.
        cx, cy = pos_e[j].mean(), pos_n[j].mean()
        rec["_pos"] = np.mod(np.arctan2(pos_n[j] - cy, pos_e[j] - cx), 2 * np.pi)
        rec["_tilt"] = srs
        return {"FULL-LAP": rec}
    for name, m in ((fwd, cruise & (v_along > v_min)),
                    (back, cruise & (v_along < -v_min))):
        # A complete pass takes essentially the same time every time, so
        # "complete" is defined relative to the longest untruncated pass in
        # this direction rather than by an absolute duration: M1's legs run
        # 144 s and M3's 44 s, and a fixed floor tuned to one silently discards
        # every leg of the other. This also drops the short first pass that the
        # climb-out truncates, which would otherwise be pooled with whole ones.
        cand = [(i0, i1, dt[i0:i1].sum()) for i0, i1 in _blocks(m) if i1 < len(m)]
        cand = [c for c in cand if c[2] >= min_secs]
        if not cand:
            return None
        longest = max(c[2] for c in cand)
        keep = [np.arange(i0, i1) for i0, i1, d in cand if d >= full_frac * longest]
        nblocks = len(keep)
        if not keep:
            return None
        chosen = np.concatenate(keep)
        s = tilt[chosen]
        # `sat` is a censoring check as much as a metric: the operating point
        # was chosen for 0% saturation in a deterministic field, and a gust can
        # legitimately push a quad to its limit. If it climbs, tilt is being
        # clipped at the top and the sd contrast understates the effect.
        rec = dict(n=int(s.size), blocks=nblocks, secs=float(dt[chosen].sum()), mean=float(s.mean()),
                   sd=float(s.std(ddof=1)), p95=float(np.percentile(s, 95)),
                   sat=float(100.0 * (s > 44.0).mean()),
                   # Variance, not sd, is the additive scale for independent
                   # disturbances: two uncorrelated sources give
                   # sd_total = sqrt(sd_1^2 + sd_2^2), so a factorial run on sd
                   # shows a spurious negative interaction even when the two
                   # factors combine perfectly independently. Test the
                   # interaction here.
                   var=float(s.var(ddof=1)))
        rec["power"] = float(np.mean(power[chosen])) if power is not None else float("nan")
        rec["energy"] = (float(np.trapezoid(power[chosen], t[chosen])) / 3600.0
                         if power is not None else float("nan"))
        rec["rotor"] = (float(np.percentile(rotor[chosen], 99))
                        if rotor is not None else float("nan"))
        along = pos_e if axis == "east" else pos_n
        j = np.clip(np.searchsorted(tp, t[chosen]) - 1, 0, len(tp) - 1)
        rec["_pos"] = along[j]
        rec["_tilt"] = s
        out[name] = rec
    return out


def bootstrap_ci(x, n=20000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    d = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(d, 100 * alpha / 2)), float(np.percentile(d, 100 * (1 - alpha / 2)))


def contrast(name, vals, expl):
    v = np.array(vals)
    lo, hi = bootstrap_ci(v)
    line = (f"  {name:<16} {v.mean():+8.3f}  [{lo:+.3f}, {hi:+.3f}]  "
            f"{int((v > 0).sum())}/{len(v)}>0")
    try:
        from scipy.stats import wilcoxon
        if len(v) >= 6 and np.any(v != 0):
            line += f"  p={wilcoxon(v).pvalue:.4f}"
    except ImportError:
        pass
    print(line + f"   {expl}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "manifests", "factorial_manifest_m1_factorial.txt"))
    ap.add_argument("--instance", type=int, default=0, help="0 = in-wake path, 1 = clear-air reference")
    # resvar is `var` with the deterministic along-track profile removed first.
    # Raw variance counts two different things at once: how much the vehicle is
    # disturbed, and how much its steady operating point varies with position.
    # A mean deficit changes the second without disturbing anything, so on raw
    # variance it looks like a disturbance the size of turbulence. It is not:
    # on M1's downwind leg the deficit contributes +12.9 deg^2 raw but -0.95
    # residual -- slower air is calmer air -- against turbulence's +7.83
    # residual in 8 seeds out of 8.
    ap.add_argument("--metric", default="sd",
                    choices=("sd", "mean", "p95", "power", "sat", "var", "resvar",
                             "energy", "rotor"))
    # Sensitivity sweeps run cells A and D only: enough to show whether the
    # ordering survives a parameter change, without repeating the B/C
    # decomposition at every value. With two cells only D-A is defined.
    ap.add_argument("--cells", default="ABCD")
    a = ap.parse_args()

    cells = tuple(a.cells)
    runs = defaultdict(dict)   # seed -> cell -> legs
    for line in open(a.manifest):
        line = line.strip()
        if not line:
            continue
        cell, seed, log_dir = line.split(None, 2)
        p = os.path.join(run_dir(log_dir), "ulogs", f"instance_{a.instance}.ulg")
        if not os.path.exists(p):
            print(f"  cell {cell} seed {seed}: no ULog — excluded"); continue
        L = legs(p)
        if L is None:
            print(f"  cell {cell} seed {seed}: no complete leg in one direction — excluded"); continue
        runs[int(seed)][cell] = L

    complete = sorted(s for s, c in runs.items() if all(k in c for k in cells))
    dropped = sorted(set(runs) - set(complete))
    print(f"instance {a.instance} ({INSTANCES.get(a.instance, f'inst{a.instance}')}), metric = tilt {a.metric}")
    print(f"complete 2x2 blocks: {len(complete)} seeds {complete}"
          + (f"   incomplete, dropped: {dropped}" if dropped else ""))
    if not complete:
        return 1

    # resvar needs the profile averaged ACROSS seeds, so it is computed here
    # rather than inside legs(), which only ever sees one flight.
    if a.metric == "resvar":
        for leg in runs[complete[0]][cells[0]].keys():
            for c in cells:
                pos = np.concatenate([runs[s][c][leg]["_pos"] for s in complete])
                edges = np.linspace(pos.min(), pos.max(), 96)
                prof, _ = np.histogram(pos, edges, weights=np.concatenate(
                    [runs[s][c][leg]["_tilt"] for s in complete]))
                cnt, _ = np.histogram(pos, edges)
                prof = np.divide(prof, cnt, out=np.zeros_like(prof), where=cnt > 0)
                for s in complete:
                    r = runs[s][c][leg]
                    k = np.clip(np.digitize(r["_pos"], edges) - 1, 0, len(prof) - 1)
                    r["resvar"] = float(np.var(r["_tilt"] - prof[k], ddof=1))

    legnames = list(runs[complete[0]][cells[0]].keys())
    for leg in legnames:
        secs = [runs[s][c][leg]["secs"] for s in complete for c in cells]
        print(f"\n{'=' * 78}\n{leg}   leg duration {min(secs):.0f}-{max(secs):.0f} s "
              f"(spread {max(secs) - min(secs):.0f} s across all cells and seeds)")
        print(f"  {'seed':>4}" + "".join(f"{c:>9}" for c in cells))
        for s in complete:
            print(f"  {s:>4}" + "".join(f"{runs[s][c][leg][a.metric]:>9.3f}" for c in cells))
        cellmean = {c: np.mean([runs[s][c][leg][a.metric] for s in complete]) for c in cells}
        print(f"  {'mean':>4}" + "".join(f"{cellmean[c]:>9.3f}" for c in cells))
        for c in cells:
            print(f"       {c} = {CELL_DESC[c]}")

        # Censoring guard. A cell spending real time against the vehicle's
        # attitude limit has its upper tail clipped, so its variance is an
        # underestimate — and because the interaction is a difference of
        # differences, one clipped cell inflates it. Report it rather than let
        # a censored cell masquerade as an interaction effect.
        clipped = [c for c in cells
                   if np.mean([runs[s][c][leg]["sat"] for s in complete]) > 5.0]
        if clipped:
            pct = {c: np.mean([runs[s][c][leg]["sat"] for s in complete]) for c in clipped}
            print("\n  !! CENSORED: " + ", ".join(f"cell {c} spends {pct[c]:.1f}% above 44 deg"
                                                  for c in clipped))
            print("     Variance in those cells is clipped and understated; every contrast")
            print("     involving them, the interaction most of all, is not interpretable.")

        g = lambda s, c: runs[s][c][leg][a.metric]
        print(f"\n  contrast          effect      95% CI            sign")
        if "B" in cells:
            contrast("B - A", [g(s, "B") - g(s, "A") for s in complete], "turbulence alone")
        if "C" in cells:
            contrast("C - A", [g(s, "C") - g(s, "A") for s in complete], "deficit alone")
        contrast("D - A", [g(s, "D") - g(s, "A") for s in complete], "the real wake")
        if "B" in cells and "C" in cells:
            contrast("D-B-C+A", [g(s, "D") - g(s, "B") - g(s, "C") + g(s, "A") for s in complete],
                     "interaction")
    return 0


if __name__ == "__main__":
    sys.exit(main())
