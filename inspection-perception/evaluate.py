#!/usr/bin/env python3
"""Score the model on one evaluation domain.

    python3 evaluate.py underwater_clahe

Writes an Ultralytics validation run to results/runs/<domain>/: the PR, P, R and
F1 curves for boxes and masks, the confusion matrices, a `predictions.json` that
`precision_recall.py` reads back, and side-by-side label/prediction previews.

The committed runs were produced exactly this way. Re-running overwrites them,
so pass --into to write somewhere else if you want to keep both.
"""
from __future__ import annotations
import argparse
from ultralytics import YOLO

import domains


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    domains.add_domain_argument(ap)
    ap.add_argument("--into", default=str(domains.RUNS),
                    help="directory to write the run under (default: results/runs)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=0, help="0 for the first GPU, or 'cpu'")
    a = ap.parse_args()

    data = domains.DATASETS / a.domain / "dataset.yaml"
    if not data.exists():
        print(f"no dataset.yaml at {data}")
        return 1

    results = YOLO(str(domains.MODEL)).val(
        data=str(data),
        # Every domain's yaml points its `val` split at the full image set: these
        # are evaluation sets, not training splits, so there is nothing held out
        # within them. 'train' is the split name the yamls use for that set.
        split="train",
        imgsz=a.imgsz, batch=a.batch, device=a.device,
        save_json=True, save_txt=True, save_conf=True,
        project=a.into, name=a.domain, exist_ok=True,
    )

    print(f"\n{a.domain}")
    for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        print(f"  {key}: {results.results_dict[key]:.4f}")
    print(f"\nBootstrap CI:      python3 bootstrap_ap50.py {a.domain}")
    print(f"Fixed-threshold P/R: python3 precision_recall.py {a.domain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
