#!/usr/bin/env python3
"""Bootstrap a 95% confidence interval on AP50 for one evaluation domain.

    python3 bootstrap_ap50.py aerial

Resamples whole images with replacement, 1000 times, and recomputes AP50 from
the per-image detection records each time. Whole images rather than detections,
because detections within an image are not independent. AP is computed over the
full score range, NOT at the fixed confidence threshold precision_recall.py
uses -- the two answer different questions.

The interval assumes images are independent of each other.
"""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils.metrics import compute_ap

import argparse
import domains

ap = argparse.ArgumentParser(description=__doc__)
domains.add_domain_argument(ap)
ap.add_argument("--bootstraps", type=int, default=1000)
ap.add_argument("--seed", type=int, default=42)
ARGS = ap.parse_args()

DOMAIN = ARGS.domain
IMAGE_DIR = domains.images(DOMAIN)
BASE_DIR = domains.DATASETS
EXPECTED_IMAGES = domains.expected_images(DOMAIN)
N_BOOTSTRAPS = ARGS.bootstraps
SEED = ARGS.seed


class BootstrapValidator(SegmentationValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        self.records = []
        BootstrapValidator.latest = self

    def _process_batch(self, preds, batch):
        result = super()._process_batch(preds, batch)
        # Keep every image, including those with no predictions or labels.
        self.records.append((
            Path(batch["im_file"]).name,
            preds["conf"].cpu().numpy(),
            result["tp"][:, 0],  # Bounding boxes at IoU 0.5, not masks.
            len(batch["cls"]),
        ))
        return result


def ap50(records, indices):
    sample = [records[i] for i in indices]
    confidence = np.concatenate([row[1] for row in sample])
    correct = np.concatenate([row[2] for row in sample])
    targets = sum(row[3] for row in sample)
    if targets == 0 or confidence.size == 0:
        return 0.0
    tp = np.cumsum(correct[np.argsort(-confidence, kind="stable")])
    recall = tp / targets
    precision = tp / np.arange(1, len(tp) + 1)
    return float(compute_ap(recall, precision)[0])


def main():
    # Temporary config/output: existing evaluation results stay untouched.
    with TemporaryDirectory(prefix="ap50-bootstrap-") as temporary:
        data = Path(temporary) / "dataset.yaml"
        data.write_text(yaml.safe_dump({
            "path": str(BASE_DIR),
            "train": str(IMAGE_DIR),
            "val": str(IMAGE_DIR),
            "names": {0: "crack"},
        }))
        YOLO(str(domains.MODEL)).val(
            validator=BootstrapValidator, data=str(data), split="train",
            imgsz=640, batch=16, device=0, workers=2,
            conf=0.001, iou=0.7, max_det=300,
            save_json=False, save_txt=False, plots=False,
            project=temporary, name="validation", verbose=False,
        )
        records = sorted(BootstrapValidator.latest.records, key=lambda row: row[0])

    n = len(records)
    if n != EXPECTED_IMAGES:
        raise ValueError(f"Expected {EXPECTED_IMAGES} images, found {n}. Check IMAGE_DIR.")
    # AP uses the full score range, NOT the fixed P/R confidence threshold.
    rng = np.random.default_rng(SEED)
    scores = np.array([
        ap50(records, rng.choice(n, size=n, replace=True))
        for _ in range(N_BOOTSTRAPS)
    ])
    lower, upper = np.percentile(scores, [2.5, 97.5])
    print(f"\n{DOMAIN} ({n} images)")
    print(f"AP50(B): {ap50(records, np.arange(n)):.4f}")
    print(f"95% CI: [{lower:.4f}, {upper:.4f}]")
    print(f"{N_BOOTSTRAPS} whole-image resamples; seed {SEED}.")
    print("Interval assumes independent images.")


if __name__ == "__main__":
    main()
