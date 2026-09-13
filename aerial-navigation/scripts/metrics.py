#!/usr/bin/env python3
"""Per-run navigation metrics -> the rows of the results table.

Takes a replayed track (ulog_extract -> wake_replay) plus the scenario, and
produces one summary dict per flight.

Energy. NOT LOTUSim's marine energy plugin: that model reads an RPM out of
`vessel_cmd_array` and its polynomial is fitted to a marine
thruster, an actuator a PX4 quadrotor neither uses nor writes to. Aerial shaft
power is computed instead from the simulator's own rotor constants, so the
energy figure is internally consistent with the physics that actually flew:

    omega   = control * maxRotVelocity          (gz MulticopterMotorModel)
    thrust  = kT * omega^2
    torque  = kQ * thrust
    P_shaft = torque * omega = kQ * kT * omega^3

with the same electromechanical efficiency eta used elsewhere.
Sanity check: this predicts a hover command of 0.770 for the X500's mass,
against 0.742 measured in level flight.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

# from assets/models/x500_px4/model.sdf
KT, KQ, W_MAX, ETA = 8.54858e-06, 0.016, 1000.0, 0.9
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def planned_polyline(scenario: dict, agent_id: str):
    a = next(x for x in scenario["agents"] if x.get("id") == agent_id)
    m = a["missions"][0]["params"]
    pts = [(m["spawn"]["x"], m["spawn"]["y"])] + [(w["x"], w["y"]) for w in m["waypoints"]]
    return np.array(pts, float)


def cross_track(df: pd.DataFrame, poly: np.ndarray) -> np.ndarray:
    """Distance to the nearest segment of the planned path. Uses nearest-segment
    rather than current-leg because a looping patrol retraces the same legs."""
    P = np.stack([df["x"].to_numpy(), df["y"].to_numpy()], axis=1)
    best = np.full(len(P), np.inf)
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a
        L2 = ab @ ab
        if L2 < 1e-9:
            continue
        t = np.clip(((P - a) @ ab) / L2, 0, 1)
        d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        best = np.minimum(best, d)
    return best


def edge_events(df: pd.DataFrame, xtrack: np.ndarray, pre=(4.0, 1.0), post=3.0):
    """Peak *excess* cross-track around each wake-boundary crossing, over a
    baseline taken just before that crossing.

    The obvious version -- peak absolute cross-track within a window, and
    "recovered" when it falls back under the whole-flight RMS -- does not
    measure a transient. On a leg with a steady crosswind the signal already
    oscillates by several metres, so the window peak is that oscillation and
    the recovery test is satisfied by the next zero crossing (measured: a
    single 0.05 s sample). Both numbers looked like results and were not.

    Baseline is the median over [t-pre[0], t-pre[1]] before the crossing, so
    each event is judged against the vehicle's own immediately preceding
    tracking, not against the flight as a whole.
    """
    if "in_wake" not in df:
        return []
    w = df["in_wake"].to_numpy().astype(bool)
    t = df["t"].to_numpy()
    out = []
    for i in np.flatnonzero(w[1:] != w[:-1]):
        t0 = t[i]
        pre_sel = (t >= t0 - pre[0]) & (t <= t0 - pre[1])
        post_sel = (t >= t0) & (t <= t0 + post)
        if pre_sel.sum() < 5 or post_sel.sum() < 5:
            continue
        base = float(np.median(xtrack[pre_sel]))
        seg = xtrack[post_sel]
        peak = float(seg.max())
        excess = peak - base
        # recovery: first return to within 20% of the excess above baseline
        thr = base + 0.2 * max(excess, 1e-9)
        after = np.flatnonzero((t > t0 + float(t[post_sel][seg.argmax()] - t0)) & (xtrack <= thr))
        rec = float(t[after[0]] - t0) if len(after) else float("nan")
        out.append(dict(t=float(t0), entering=bool(w[i + 1]),
                        baseline=base, peak=peak, excess=excess, recovery_s=rec))
    return out


def orbit_metrics(df: pd.DataFrame, poly: np.ndarray, tol: float):
    """Radial error about the orbit centre, and standoff-corridor violations.

    Inferred from the planned path rather than configured: if the waypoints lie
    on a circle to within 5% they are treated as an orbit, and the centre and
    radius come from them. Radial error is the operationally meaningful
    quantity for a close inspection -- distance to the structure -- whereas
    cross-track to the sampled polygon carries the discretisation of that
    polygon as well.
    """
    pts = np.unique(poly, axis=0)
    c = pts.mean(axis=0)
    rr = np.linalg.norm(pts - c, axis=1)
    R = float(rr.mean())
    if R < 1e-6 or float(rr.std()) / R > 0.05:
        return {}
    r = np.linalg.norm(np.stack([df["x"], df["y"]], axis=1) - c, axis=1)
    e = r - R
    return {
        "orbit_R_m": round(R, 1),
        "radial_rms_m": round(float(np.sqrt(np.mean(e ** 2))), 3),
        "radial_bias_m": round(float(e.mean()), 3),
        "radial_max_out_m": round(float(e.max()), 3),
        "radial_max_in_m": round(float(e.min()), 3),
        "standoff_violation_pct": round(float(100 * (np.abs(e) > tol).mean()), 2),
        "standoff_tol_m": tol,
    }


def summarise(df: pd.DataFrame, scenario: dict, agent_id: str, sat=0.95,
              cruise_alt: float | None = None, standoff_tol: float = 2.0) -> dict:
    """cruise_alt gates every statistic to level flight, excluding the climb and
    the turnaround transients. Energy stays over the whole mission -- it is a
    total, not a steady-state statistic. Keep this consistent with
    tilt_stats.py's gate or the two tools will disagree on the same flight."""
    full = df
    if cruise_alt is not None:
        df = df[df["z"] > cruise_alt].reset_index(drop=True)
    poly = planned_polyline(scenario, agent_id)
    xt = cross_track(df, poly)
    t = df["t"].to_numpy()

    mcols = [c for c in ("m0", "m1", "m2", "m3") if c in df]
    P = sum(KQ * KT * (df[c].to_numpy() * W_MAX) ** 3 for c in mcols) / ETA
    Pf = sum(KQ * KT * (full[c].to_numpy() * W_MAX) ** 3 for c in mcols) / ETA
    energy_wh = float(TRAPZ(Pf, full["t"].to_numpy()) / 3600.0)

    ev = edge_events(df, xt)
    peaks = [e["excess"] for e in ev]
    recs = [e["recovery_s"] for e in ev if np.isfinite(e["recovery_s"])]

    s = {
        "agent": agent_id,
        "duration_s": round(float(t[-1]), 1),
        "xtrack_rms_m": round(float(np.sqrt(np.mean(xt ** 2))), 3),
        "xtrack_mean_m": round(float(xt.mean()), 3),
        "xtrack_max_m": round(float(xt.max()), 3),
        "tilt_rms_deg": round(float(np.sqrt(np.mean(df["tilt_deg"] ** 2))), 2) if "tilt_deg" in df else None,
        "tilt_max_deg": round(float(df["tilt_deg"].max()), 2) if "tilt_deg" in df else None,
        "rotor_sat_pct": round(float(100 * (df["m_max"] > sat).mean()), 2) if "m_max" in df else None,
        "mean_power_w": round(float(P.mean()), 1),
        "energy_wh": round(energy_wh, 3),
        "mean_speed_ms": round(float(np.hypot(df["vx"], df["vy"]).mean()), 2),
        "edge_crossings": len(ev),
        "edge_excess_max_m": round(float(np.max(peaks)), 3) if peaks else None,
        "edge_excess_median_m": round(float(np.median(peaks)), 3) if peaks else None,
        "edge_recovery_s_median": round(float(np.median(recs)), 2) if recs else None,
    }
    s.update(orbit_metrics(df, poly, standoff_tol))
    if "deficit" in df:
        s["in_wake_pct"] = round(float(100 * df["in_wake"].mean()), 1)
        s["deficit_median_pct"] = round(float(100 * df["deficit"].median()), 1)
        s["dUdl_p95"] = round(float(np.percentile(df["dUdl"], 95)), 5)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("pairs", nargs="+", help="agent_id=replayed.csv")
    ap.add_argument("--out")
    ap.add_argument("--cruise-alt", type=float, default=None)
    ap.add_argument("--standoff-tol", type=float, default=2.0)
    a = ap.parse_args()
    scenario = json.load(open(a.scenario))
    rows = []
    for p in a.pairs:
        agent, path = p.split("=", 1)
        rows.append(summarise(pd.read_csv(path), scenario, agent, cruise_alt=a.cruise_alt, standoff_tol=a.standoff_tol))
    df = pd.DataFrame(rows).set_index("agent").T
    print(df.to_string())
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=2)
        print(f"\n-> {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
