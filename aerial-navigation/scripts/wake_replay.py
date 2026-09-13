#!/usr/bin/env python3
"""Recover the disturbance a logged flight actually met, offline.

The simulator never records the wind at the vehicle. It does not need to:
the wake field is a deterministic function of position, so replaying a logged
trajectory through the *same* model the run used recovers both the local wind
speed and its lateral gradient exactly, with nothing added to the control loop.

Adds per-sample:
    U_local   local wind speed (m/s), farm-compounded
    deficit   (U_inf - U_local) / U_inf
    dUdl      lateral gradient |dU/dl| across the wake (m/s per m), central
              difference perpendicular to the wind
    in_wake   deficit above --threshold

`dUdl` is the regressor for the headline figure: tracking error against the
gradient the vehicle was crossing at that instant.

Caveat worth carrying into any reported figure: this reconstructs the *model's*
continuous field. What the vehicle actually felt is that field discretised
into cone segments, uniform within each, so the delivered lateral edge is a
step rather than this gradient. The discretisation sweep quantifies the gap.
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
import pandas as pd


def build_model(scenario: dict):
    from lotusim_sdk.agents.environment.wake.larsen import LarsenWakeModel
    from lotusim_sdk.agents.environment.wake.blended import BlendedWakeModel
    w = next(a for a in scenario["agents"] if a.get("class") == "Wake")
    m = LarsenWakeModel(
        diameter=w["diameter"], ct=w["ct"], cp=w["cp"], air_density=w["air_density"],
        cut_in=w["cut_in"], cut_out=w["cut_out"], ambient_ti=w["ambient_ti"],
        shear_exponent=w["shear_exponent"])
    turbines = [(t["x"], t["y"], t["z"]) for t in w["turbines"]]
    return BlendedWakeModel(m), turbines


def find_wind(scenario: dict):
    """The ambient vector the run was flown at, from the Wind agent's set_wind."""
    def walk(nodes):
        for n in nodes or []:
            if n.get("task") == "set_wind":
                p = n.get("params", {})
                return [float(p.get("x", 0.0)), float(p.get("y", 0.0))]
            got = walk(n.get("children"))
            if got:
                return got
        return None
    for a in scenario["agents"]:
        if a.get("class") == "Wind":
            got = walk(a.get("missions"))
            if got:
                return got
    raise SystemExit("no set_wind found in scenario; pass --wind X Y")


def replay(df: pd.DataFrame, blended, turbines, wind, h=2.0, threshold=0.05) -> pd.DataFrame:
    wind = np.asarray(wind, float)
    U_inf = float(np.hypot(*wind))
    if U_inf < 1e-6:
        raise SystemExit("ambient wind is zero")
    unit = wind / U_inf
    perp = np.array([-unit[1], unit[0]])          # lateral, across the wake

    xs, ys = df["x"].to_numpy(), df["y"].to_numpy()
    f = lambda X, Y: np.array([
        blended.farm_velocity_at_point(U_inf, wind, x, y, turbines) for x, y in zip(X, Y)])

    U = f(xs, ys)
    Up = f(xs + perp[0] * h, ys + perp[1] * h)
    Um = f(xs - perp[0] * h, ys - perp[1] * h)

    out = df.copy()
    out["U_local"] = U
    out["deficit"] = (U_inf - U) / U_inf
    out["dUdl"] = np.abs(Up - Um) / (2 * h)
    out["in_wake"] = out["deficit"] > threshold
    out.attrs["U_inf"] = U_inf
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("track", help="CSV from ulog_extract.py, or a .ulg")
    ap.add_argument("--wind", type=float, nargs=2)
    ap.add_argument("--h", type=float, default=2.0)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--out")
    a = ap.parse_args()

    scenario = json.load(open(a.scenario))
    blended, turbines = build_model(scenario)
    wind = a.wind or find_wind(scenario)

    if a.track.endswith(".ulg"):
        from ulog_extract import extract
        df = extract(a.track)
    else:
        df = pd.read_csv(a.track)

    out = replay(df, blended, turbines, wind, a.h, a.threshold)
    U_inf = out.attrs["U_inf"]
    inw = out["in_wake"]
    print(f"ambient {U_inf:.2f} m/s, {len(out)} samples, {len(turbines)} turbines")
    print(f"  in wake      : {100*inw.mean():5.1f}% of samples")
    print(f"  U_local      : {out['U_local'].min():5.2f} .. {out['U_local'].max():5.2f} m/s")
    print(f"  deficit      : max {100*out['deficit'].max():5.1f}%")
    print(f"  |dU/dl|      : max {out['dUdl'].max():.4f} m/s/m, "
          f"p95 {np.percentile(out['dUdl'],95):.4f}")
    if a.out:
        out.to_csv(a.out, index=False)
        print(f"  -> {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
