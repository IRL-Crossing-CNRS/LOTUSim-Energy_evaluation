#!/usr/bin/env python3
"""PX4 ULog -> one tidy DataFrame per flight, in LOTUSim world ENU.

Frame. PX4 logs a local NED frame whose origin is the *world* origin, not the
vehicle spawn (verified against scenario spawns: an agent spawned at world
y=-1000 logs north=-1000). So:

    world_x (East)  = px4 y
    world_y (North) = px4 x
    world_z (Up)    = -px4 z

Ground truth. Position and attitude are taken from the *_groundtruth topics,
so tracking error is not contaminated by estimator error; the estimated
topics are kept alongside where present, which is what lets the two be
separated later.

Topics are logged at different rates, so everything is interpolated onto one
uniform grid (--rate, default 20 Hz) spanning the overlap.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

# Rotor constants from x500_px4/model.sdf. Defined here, and imported by
# factorial_analysis, so the per-rotor and total power can never drift.
KT, KQ, W_MAX, ETA = 8.54858e-06, 0.016, 1000.0, 0.9
import pandas as pd
from pyulog import ULog

WANT = ["vehicle_local_position_groundtruth", "vehicle_attitude_groundtruth",
        "vehicle_local_position_setpoint", "actuator_motors", "battery_status"]


def _q_to_euler(w, x, y, z):
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def extract(path: str, rate: float = 20.0) -> pd.DataFrame:
    u = ULog(path, WANT)
    have = {d.name: d.data for d in u.data_list}
    if "vehicle_local_position_groundtruth" not in have:
        raise SystemExit(f"{path}: no vehicle_local_position_groundtruth")

    p = have["vehicle_local_position_groundtruth"]
    t0, t1 = p["timestamp"][0], p["timestamp"][-1]
    grid = np.arange(t0, t1, 1e6 / rate)
    out = {"t": (grid - t0) / 1e6}

    def interp(src, key, dst):
        if src is None or key not in src:
            return
        out[dst] = np.interp(grid, src["timestamp"], src[key])

    # position / velocity, mapped into world ENU
    interp(p, "y", "x");  interp(p, "x", "y")
    out["z"] = -np.interp(grid, p["timestamp"], p["z"])
    interp(p, "vy", "vx"); interp(p, "vx", "vy")
    out["vz"] = -np.interp(grid, p["timestamp"], p["vz"])

    # attitude
    a = have.get("vehicle_attitude_groundtruth")
    if a is not None:
        q = [np.interp(grid, a["timestamp"], a[f"q[{i}]"]) for i in range(4)]
        roll, pitch, yaw = _q_to_euler(*q)
        out["roll"], out["pitch"], out["yaw"] = roll, pitch, yaw
        out["tilt_deg"] = np.degrees(np.arccos(np.clip(np.cos(roll) * np.cos(pitch), -1, 1)))

    # commanded setpoint, same frame mapping
    sp = have.get("vehicle_local_position_setpoint")
    if sp is not None:
        interp(sp, "y", "sp_x"); interp(sp, "x", "sp_y")
        out["sp_z"] = -np.interp(grid, sp["timestamp"], sp["z"])

    # rotor commands -> effort and saturation
    m = have.get("actuator_motors")
    if m is not None:
        cols = [c for c in (f"control[{i}]" for i in range(4)) if c in m]
        for i, c in enumerate(cols):
            out[f"m{i}"] = np.interp(grid, m["timestamp"], m[c])
        if cols:
            M = np.vstack([out[f"m{i}"] for i in range(len(cols))])
            out["m_mean"] = M.mean(axis=0)
            out["m_max"] = M.max(axis=0)
            # Per-rotor electrical power, P = kQ*kT*w^3 / eta with
            # w = control * maxRotVelocity -- the same model factorial_analysis
            # sums for the total, exposed per rotor so a load imbalance (one
            # rotor working the wake edge harder than its opposite) is visible
            # rather than averaged away. battery_status cannot answer this:
            # PX4's SITL model reports current_a as a constant -1.
            for i in range(len(cols)):
                out[f"p{i}"] = KQ * KT * (np.clip(out[f"m{i}"], 0, 1) * W_MAX) ** 3 / ETA
            out["p_total"] = sum(out[f"p{i}"] for i in range(len(cols)))

    # battery (PX4's own SITL model; see README on why this is not our model)
    b = have.get("battery_status")
    if b is not None:
        for key, dst in (("voltage_v", "batt_v"), ("current_a", "batt_a"),
                         ("discharged_mah", "batt_mah"), ("remaining", "batt_frac")):
            interp(b, key, dst)

    df = pd.DataFrame(out)
    df.attrs["source"] = os.path.abspath(path)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ulg", nargs="+")
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--out", help="write CSV next to each ULog (or to this dir)")
    a = ap.parse_args()
    for f in a.ulg:
        df = extract(f, a.rate)
        print(f"{os.path.basename(f)}: {len(df)} rows @ {a.rate}Hz, "
              f"{df['t'].iloc[-1]:.1f}s, cols={len(df.columns)}")
        print("  " + ", ".join(df.columns))
        if a.out:
            d = a.out if os.path.isdir(a.out) else os.path.dirname(f)
            dst = os.path.join(d, os.path.basename(f).replace(".ulg", ".csv"))
            df.to_csv(dst, index=False)
            print(f"  -> {dst}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
