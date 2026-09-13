#!/usr/bin/env python3
"""Derive the four factorial cells from a base scenario.

    python3 make_factorial_configs.py <base.json> <prefix>

writes <prefix>_A.json .. <prefix>_D.json beside the base config:

           | ambient TI              | wake TI
    -------+-------------------------+--------------------------
    uniform| A  no wind_regions      | B  turbulence_only
    wake   | C  turbulence_sigma=0.8 | D  wake as published

Only the Wake agent's `wind_regions` block differs; the turbines, the wind, the
vehicles and their missions are copied through untouched, so the four cells put
the same vehicle on the same path and change nothing but the field.

Cell C sets sigma explicitly to the ambient value rather than disabling
turbulence: with turbulence off, regions would carry sigma 0 while the air
around them still had the world's ambient sigma, making C "the deficit, in
unnaturally still air" rather than "the deficit alone".

Cell B is not a physical field. Nowhere has freestream speed and wake-level
turbulence at once; it is a manipulation that breaks the confound a real wake
always imposes, and it must be described as a synthetic control rather than
as a scenario. Its region geometry is identical to C and D — the
segment set is chosen on the true deficit before the published velocity is
decided — so no pair of cells differs in more than one factor.
"""
import copy, json, pathlib, sys

CELLS = {
    "A": ("uniform mean, ambient TI", None),
    "B": ("uniform mean, wake TI", {"turbulence": True, "turbulence_only": True}),
    "C": ("wake mean, ambient TI", {"turbulence": True, "turbulence_sigma": 0.8}),
    "D": ("wake mean, wake TI", {"turbulence": True}),
}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    base_path = pathlib.Path(sys.argv[1])
    prefix = sys.argv[2]
    base = json.loads(base_path.read_text())

    wakes = [a for a in base.get("agents", []) if a.get("class") == "Wake"]
    if not wakes:
        print(f"{base_path}: no Wake agent — nothing to vary", file=sys.stderr)
        return 1
    # Keep whatever the base already tuned (segment size, wake length, hysteresis)
    # so the cells differ from the base only in the two factors under study.
    base_wr = {k: v for k, v in (wakes[0].get("wind_regions") or {}).items()
               if k not in ("turbulence", "turbulence_only", "turbulence_sigma")}

    for cell, (desc, extra) in CELLS.items():
        cfg = copy.deepcopy(base)
        for agent in (a for a in cfg["agents"] if a.get("class") == "Wake"):
            if extra is None:
                agent.pop("wind_regions", None)
            else:
                agent["wind_regions"] = {**base_wr, **extra}
        out = base_path.with_name(f"{prefix}_{cell}.json")
        out.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"{out.name:<28} {desc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
