#!/usr/bin/env python3
"""Does the in-wake attitude-variance excess survive repetition?

Reads the turbulence-ON runs listed in rep_manifest.txt (one gust seed each)
against a single deterministic OFF baseline, and tests the quantity the smoke
run suggested might discriminate a wake where the mean-field metrics could not:
how much more attitude variance the in-wake drone gains from turbulence than
the clear-air drone does.

    python3 rep_analysis.py --off <off_log_dir> [--manifest rep_manifest.txt]

WHY THE STATISTIC IS PER LEG.  The mission is an out-and-back along the wind,
so its two legs are flown downwind and upwind and sit at completely different
tilts: about 19 deg northbound against 35-41 deg southbound. The scenario is
cut off by a wall-clock timer, so how much of the upwind leg a run captures
depends on PX4 startup jitter — in this campaign, 23-28 s for four seeds and
43-44 s for two. Pooling the legs therefore measures how much upwind flying
happened to fit inside the window, which swamps the gust entirely: pooled, this
same data gives p=0.58 and two of six seeds with the sign flipped, purely
because those two caught 60% more of the high-tilt leg.

The northbound leg runs a near-identical 144-145 s in every run including the
baseline, so it is the primary comparison. The southbound leg is reported too,
flagged, because its duration is exactly what varies.

WHY DIFFERENCE-IN-DIFFERENCES.  The two drones fly different air by design, so
their sds are not interchangeable. What is comparable is how far each moves
from its OWN turbulence-off value:

    per seed:  d_wake  = sd_on(in-wake)   - sd_off(in-wake)
               d_clear = sd_on(clear-air) - sd_off(clear-air)
               D       = d_wake - d_clear      <- the statistic

Paired Wilcoxon signed-rank over seeds, plus a bootstrap CI on mean(D). Six
seeds is the floor at which a one-sided test can reach p<0.05 at all (min
p=0.0156); a null at n=6 means "not shown", not "no effect".

Mean tilt is reported as a CONTROL, not a result: the gust is zero-mean by
construction, so a systematic shift in mean tilt would mean the implementation
is wrong and the variance result should not be believed.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
from pyulog import ULog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factorial_analysis import run_dir

INSTANCES = {0: "IN-WAKE", 1: "CLEAR-AIR"}
# Northbound is downwind for the wind these scenarios set ([0, +10] ENU, i.e.
# blowing toward +y/North). Named by heading, not by upwind/downwind, so the
# labels stay true if the wind is ever reversed.
LEGS = ("NORTHBOUND", "SOUTHBOUND")


def leg_stats(path: str, alt_min: float = 45.0, v_min: float = 5.0):
    """Per-leg cruise tilt statistics, split on along-track velocity sign.

    Altitude and velocity are interpolated onto the attitude timestamps rather
    than index-paired: attitude logs at ~57 Hz and local position at 50 Hz, so
    pairing sample i with sample i drifts about 25 s over a 190 s flight and
    truncating to the shorter array drops the tail — which for this mission is
    the high-tilt upwind leg.
    """
    u = ULog(path, ["vehicle_attitude_groundtruth", "vehicle_local_position_groundtruth"])
    a = u.get_dataset("vehicle_attitude_groundtruth").data
    t = np.array(a["timestamp"], dtype=float) * 1e-6
    w, x, y, z = a["q[0]"], a["q[1]"], a["q[2]"], a["q[3]"]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    tilt = np.degrees(np.arccos(np.clip(np.cos(roll) * np.cos(pitch), -1, 1)))

    p = u.get_dataset("vehicle_local_position_groundtruth").data
    tp = np.array(p["timestamp"], dtype=float) * 1e-6
    alt = np.interp(t, tp, -np.array(p["z"]))
    v_north = np.interp(t, tp, np.array(p["vx"]))  # world North = px4 x

    cruise = alt > alt_min
    dt = np.gradient(t)
    out = {}
    for leg, mask in zip(LEGS, (cruise & (v_north > v_min), cruise & (v_north < -v_min))):
        if mask.sum() < 50:
            return None
        s = tilt[mask]
        out[leg] = dict(n=int(mask.sum()), secs=float(dt[mask].sum()),
                        mean=float(s.mean()), sd=float(s.std(ddof=1)),
                        p95=float(np.percentile(s, 95)))
    return out


def load(log_dir: str):
    out = {}
    for inst in INSTANCES:
        p = os.path.join(log_dir.rstrip("/"), "ulogs", f"instance_{inst}.ulg")
        if not os.path.exists(p):
            return None
        s = leg_stats(p)
        if s is None:
            return None
        out[inst] = s
    return out


def bootstrap_ci(x, n=20000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    draws = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


def report_leg(leg: str, off, rows, primary: bool):
    print(f"\n{'=' * 78}\n{leg}   ({'primary' if primary else 'secondary — see duration column'})")
    print(f"  baseline (turbulence off)   sd wake {off[0][leg]['sd']:.2f}   "
          f"sd clear {off[1][leg]['sd']:.2f}   ({off[0][leg]['secs']:.0f} s)")
    print(f"  {'seed':>4} {'secs':>5} {'sd wake':>8} {'sd clear':>9} {'d_wake':>8} "
          f"{'d_clear':>8} {'D':>7}   {'mean wake':>9} {'mean clear':>10}")
    D, mw, mc = [], [], []
    for seed, on in sorted(rows):
        dw = on[0][leg]["sd"] - off[0][leg]["sd"]
        dc = on[1][leg]["sd"] - off[1][leg]["sd"]
        D.append(dw - dc); mw.append(on[0][leg]["mean"]); mc.append(on[1][leg]["mean"])
        print(f"  {seed:>4} {on[0][leg]['secs']:>5.0f} {on[0][leg]['sd']:>8.2f} "
              f"{on[1][leg]['sd']:>9.2f} {dw:>8.2f} {dc:>8.2f} {dw - dc:>7.2f}   "
              f"{on[0][leg]['mean']:>9.2f} {on[1][leg]['mean']:>10.2f}")

    D = np.array(D)
    lo, hi = bootstrap_ci(D)
    print(f"\n  D = (in-wake sd gain) - (clear-air sd gain), degrees")
    print(f"  mean {D.mean():+.3f}   median {np.median(D):+.3f}   "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]   {int((D > 0).sum())}/{len(D)} positive")
    try:
        from scipy.stats import wilcoxon
        if len(D) >= 6:
            r = wilcoxon(D, alternative="greater")
            print(f"  paired Wilcoxon signed-rank (one-sided, D>0): W={r.statistic:.1f}  p={r.pvalue:.4f}")
        else:
            print(f"  n={len(D)}: below the 6 needed for p<0.05 to be reachable — CI only")
    except ImportError:
        print("  scipy unavailable — bootstrap CI only")

    print(f"\n  control: mean tilt (a zero-mean gust must not move it)")
    for lab, m, o in (("IN-WAKE", mw, off[0][leg]["mean"]), ("CLEAR-AIR", mc, off[1][leg]["mean"])):
        m = np.array(m)
        sd = f"{m.std(ddof=1):.2f}" if len(m) > 1 else "n/a"
        print(f"    {lab:<11} OFF {o:6.2f}   ON {m.mean():6.2f} +/- {sd}   "
              f"shift {100 * (m.mean() - o) / o:+.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True, help="log dir of the deterministic (turbulence OFF) run")
    ap.add_argument("--manifest", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "manifests", "rep_manifest.txt"))
    a = ap.parse_args()

    off = load(a.off)
    if off is None:
        print(f"OFF baseline unusable: {a.off}", file=sys.stderr)
        return 1

    rows = []
    for line in open(a.manifest):
        line = line.strip()
        if not line:
            continue
        seed, log_dir = line.split(None, 1)
        on = load(run_dir(log_dir))
        if on is None:
            print(f"  seed {seed}: ULogs missing or a leg too short — excluded")
            continue
        rows.append((int(seed), on))
    if not rows:
        print("no usable runs in the manifest", file=sys.stderr)
        return 1

    print(f"OFF baseline: {a.off}\nturbulence ON: {len(rows)} seeds")
    report_leg("NORTHBOUND", off, rows, primary=True)
    report_leg("SOUTHBOUND", off, rows, primary=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
