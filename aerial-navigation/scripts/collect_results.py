#!/usr/bin/env python3
"""Compute every reported contrast, once, into one JSON file.

Each reported number is otherwise recomputed from the ULogs on demand, which
means it depends on whatever the analysis code says today. This writes them
down: metric, contrast, CI, seed count, and the manifest and commit they came
from, so a number can be traced to the runs that produced it without re-running
anything.
"""
from __future__ import annotations
import json, subprocess, sys, os
from datetime import date
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from factorial_analysis import legs, bootstrap_ci, run_dir

METRICS = ("mean", "sd", "p95", "var", "resvar", "sat", "power", "energy", "rotor")
JOBS = [
    ("M1", "factorial_manifest_m1_factorial.txt", [0]),
    ("M3", "factorial_manifest_m3_factorial.txt", [0, 1, 2, 3]),
    ("M2", "factorial_manifest_m2_factorial.txt", [0, 1]),
    ("kappa0.25", "factorial_manifest_m1_factorial_kappa0.25.txt", [0]),
    ("kappa0.50", "factorial_manifest_m1_factorial_kappa0.50.txt", [0]),
    ("tau0.5",    "factorial_manifest_m1_factorial_tau0.5.txt",    [0]),
    ("tau3.0",    "factorial_manifest_m1_factorial_tau3.0.txt",    [0]),
]
here = os.path.dirname(os.path.abspath(__file__))
study = os.path.dirname(here)                 # aerial-navigation/
repo = os.path.dirname(study)                 # repository root
manifests = os.path.join(study, "results", "manifests")

def commit():
    try:
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"

out = {"generated": str(date.today()), "commit": commit(), "missions": {}}
for name, man, instances in JOBS:
    path = os.path.join(manifests, man)
    if not os.path.exists(path):
        print(f"  {name}: manifest missing, skipped"); continue
    entries = [l.split(None, 2) for l in open(path) if l.strip()]
    for inst in instances:
        runs = {}
        for cell, seed, d in entries:
            ulg = f"{run_dir(d)}/ulogs/instance_{inst}.ulg"
            if not os.path.exists(ulg): continue
            L = legs(ulg, 45.0)
            if L: runs.setdefault(int(seed), {})[cell] = L
        cells = sorted({c for v in runs.values() for c in v})
        complete = sorted(s for s, v in runs.items() if len(v) == len(cells))
        if not complete: continue
        legnames = list(runs[complete[0]][cells[0]].keys())
        key = f"{name}_inst{inst}"
        out["missions"][key] = {"manifest": man, "instance": inst,
                                "seeds": complete, "cells": cells, "legs": {}}
        for leg in legnames:
            rec = {}
            for metric in METRICS:
                if metric == "resvar":      # needs the cross-seed profile; skip here
                    continue
                vals = {c: [runs[s][c][leg][metric] for s in complete] for c in cells}
                rec[metric] = {"cell_means": {c: float(np.mean(vals[c])) for c in cells}}
                for lbl, x, y in (("turbulence", "B", "A"), ("deficit", "C", "A"), ("wake", "D", "A")):
                    if x not in cells or y not in cells: continue
                    d_ = np.array(vals[x]) - np.array(vals[y])
                    lo, hi = bootstrap_ci(d_)
                    rec[metric][lbl] = {"delta": float(d_.mean()),
                                        "ci95": [float(lo), float(hi)],
                                        "n_positive": int((d_ > 0).sum()), "n": len(d_)}
            out["missions"][key]["legs"][leg] = rec
        print(f"  {key}: {len(complete)} seeds, cells {''.join(cells)}, legs {legnames}")

dst = os.path.join(study, "results", "factorial_results.json")

# Refuse to write an empty file over a stored one. Without the ULogs the loop
# above collects nothing, and overwriting on that path would destroy the record
# this file exists to keep -- silently, and exactly when the logs are missing.
if not out["missions"]:
    print("\nno mission produced results: manifests or ULogs missing.")
    print(f"leaving {dst} as it is.")
    sys.exit(1)

os.makedirs(os.path.dirname(dst), exist_ok=True)
json.dump(out, open(dst, "w"), indent=1)
print(f"\nwritten: {dst}")
