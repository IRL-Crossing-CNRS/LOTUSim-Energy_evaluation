#!/usr/bin/env python3
"""Decide whether a run is analysable. Exit 0 = PASS, 1 = FAIL.

Three assertions, all required. The first two passed on a run where neither
drone ever left the ground, which is why the third exists:

  A1  the wake plugin loaded          (gz_aerialWorld.log)
  A2  the wake field was published    (run_assertions.json, from probe_run.py)
  A3  every vehicle actually flew     (PX4 ULog: altitude, cruise samples, track)

With --collect the ULogs are copied into <log_dir>/ulogs/ before checking, so
the run -> log mapping survives a batch that would otherwise leave them in
PX4's own rootfs tree with only a timestamp to match on.
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, glob
import numpy as np

PX4_ROOT = os.path.expanduser("~/PX4-Autopilot/build/px4_sitl_default/rootfs")


def newest_ulogs(instances=(0, 1)):
    found = {}
    for i in instances:
        c = glob.glob(f"{PX4_ROOT}/{i}/log/*/*.ulg")
        if c:
            found[i] = max(c, key=os.path.getmtime)
    return found


def flight_stats(path, min_alt):
    from pyulog import ULog
    u = ULog(path, ["vehicle_local_position_groundtruth"])
    names = {d.name for d in u.data_list}
    if "vehicle_local_position_groundtruth" not in names:
        return None
    p = u.get_dataset("vehicle_local_position_groundtruth").data
    alt = -p["z"]
    t = np.asarray(p["timestamp"], dtype=float) * 1e-6
    track = float(np.hypot(p["x"].max() - p["x"].min(), p["y"].max() - p["y"].min()))
    return dict(max_alt=float(alt.max()), cruise=int((alt > min_alt).sum()), track_m=track,
                flight_s=float(t[-1] - t[0]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--min-alt", type=float, default=45.0)
    ap.add_argument("--min-cruise", type=int, default=500)
    ap.add_argument("--min-track", type=float, default=100.0)
    # A run whose ULog stops early looks healthy on the other three tests: it
    # reached altitude, logged plenty of cruise samples and covered far more
    # than 100 m. Forty-plus runs passed A3 having flown a fifth of their
    # mission, because a full disk truncates the log without failing anything.
    # Expressed as a fraction of the run window so it follows RUN_S.
    ap.add_argument("--run-seconds", type=float, default=None,
                    help="the scenario's wall-clock window; A3 then also requires "
                         "a flight of at least --min-flight-frac of it")
    ap.add_argument("--min-flight-frac", type=float, default=0.6)
    ap.add_argument("--instances", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--no-wake", action="store_true",
                    help="force: treat as a scenario with no wind regions (normally inferred)")
    a = ap.parse_args()
    d = a.log_dir.rstrip("/")
    fails = []

    # A1
    gz = os.path.join(d, "gz_aerialWorld.log")
    a1 = os.path.isfile(gz) and "WindRegionsPlugin ready" in open(gz, errors="replace").read()
    if not a1:
        fails.append("A1 wind_regions plugin did not load")

    # A2. What counts as a pass depends on what the scenario ASKED for, read
    # from the config copy the run left behind. A uniform-field cell that
    # deliberately declares no wind_regions must assert the opposite — zero
    # regions — or a stray latched publication from a previous run would go
    # unnoticed and the "uniform" baseline would quietly contain a wake.
    expect_regions = not a.no_wake
    if not a.no_wake:
        for cfg_path in sorted(glob.glob(os.path.join(d, "config", "*.json"))):
            try:
                cfg = json.load(open(cfg_path))
            except (ValueError, OSError):
                continue
            wake = [ag for ag in cfg.get("agents", []) if ag.get("class") == "Wake"]
            if wake:
                expect_regions = any("wind_regions" in ag for ag in wake)
                break

    ap_path = os.path.join(d, "run_assertions.json")
    probe = json.load(open(ap_path)) if os.path.isfile(ap_path) else None
    if probe is None:
        a2, a2txt = False, "no run_assertions.json — probe_run.py was not run during the run"
        fails.append("A2 " + a2txt)
    else:
        n_reg = probe.get("wake_regions", 0)
        a2txt = f"{n_reg} regions, {probe.get('turbines', 0)} turbines"
        if expect_regions:
            a2 = n_reg > 0
            if not a2:
                fails.append("A2 wake field empty — " + a2txt)
        else:
            a2 = n_reg == 0
            a2txt += " (uniform-field cell: expected 0)"
            if not a2:
                fails.append("A2 uniform-field cell was not uniform — " + a2txt)

    # A3. Prefer ULogs already collected into this run's directory: reading
    # PX4's rootfs instead would validate whichever run finished most recently,
    # silently passing an old run against fresh logs.
    collected = {int(os.path.basename(f).split("_")[1].split(".")[0]): f
                 for f in sorted(glob.glob(os.path.join(d, "ulogs", "instance_*.ulg")))}
    if collected:
        ulogs, a.collect = collected, False
    else:
        ulogs = newest_ulogs(a.instances)
        if not a.collect:
            print("  note: using ULogs from PX4 rootfs (not collected into this run dir);"
                  " correct only for the run that just finished")
    if a.collect and ulogs:
        os.makedirs(os.path.join(d, "ulogs"), exist_ok=True)
        for i, p in list(ulogs.items()):
            dst = os.path.join(d, "ulogs", f"instance_{i}.ulg")
            shutil.copy2(p, dst)
            ulogs[i] = dst
    rows = []
    if not ulogs:
        fails.append("A3 no ULog found")
    for i, p in sorted(ulogs.items()):
        s = flight_stats(p, a.min_alt)
        if s is None:
            fails.append(f"A3 instance {i}: no ground-truth position in ULog")
            continue
        ok = s["max_alt"] >= a.min_alt and s["cruise"] >= a.min_cruise and s["track_m"] >= a.min_track
        need = a.min_flight_frac * a.run_seconds if a.run_seconds else 0.0
        truncated = bool(need) and s["flight_s"] < need
        ok = ok and not truncated
        rows.append((i, s, ok))
        if not ok:
            why = (f"log ends after {s['flight_s']:.0f}s of a {a.run_seconds:.0f}s window, "
                   f"needs {need:.0f}s; a truncated ULog usually means the filesystem PX4 "
                   f"logs to ran out of space") if truncated else (
                   f"max_alt {s['max_alt']:.1f}m, cruise {s['cruise']}, track {s['track_m']:.0f}m")
            fails.append(f"A3 instance {i} never flew the mission ({why})")

    print(f"run: {d}")
    print(f"  A1 plugin loaded      : {'PASS' if a1 else 'FAIL'}")
    print(f"  A2 wake published     : {'PASS' if a2 else 'FAIL'}  {a2txt}")
    for i, s, ok in rows:
        print(f"  A3 instance {i} flew    : {'PASS' if ok else 'FAIL'}  "
              f"max_alt {s['max_alt']:6.1f}m  cruise {s['cruise']:6d}  track {s['track_m']:7.0f}m"
              f"  flight {s['flight_s']:5.0f}s")
    verdict = "PASS" if not fails else "FAIL"
    print(f"  VERDICT: {verdict}")
    for f in fails:
        print(f"    - {f}")
    return 0 if not fails else 1

if __name__ == "__main__":
    sys.exit(main())
