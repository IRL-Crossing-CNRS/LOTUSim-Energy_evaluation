#!/usr/bin/env python3
"""Precision and recall at the domain's fixed confidence threshold.

    python3 precision_recall.py aerial

Reads back the predictions.json that evaluate.py wrote, keeps every detection
above the threshold, and matches them one-to-one against the labels at
IoU >= 0.5. Segmentation polygons are reduced to their enclosing boxes, so this
is a box-level score, the same one the results table reports.

The threshold is the modality's, not the domain's -- see domains.py for why.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import domains

_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
domains.add_domain_argument(_ap)
_ap.add_argument("--run", default=None,
                 help="run directory holding predictions.json (default: results/runs/<domain>)")
ARGS = _ap.parse_args()

DOMAIN = ARGS.domain
IMAGE_DIR = domains.images(DOMAIN)
LABEL_DIR = domains.labels(DOMAIN)
RUN_DIR = Path(ARGS.run) if ARGS.run else domains.RUNS / DOMAIN
PREDICTION_PATH = RUN_DIR / "predictions.json"

CONFIDENCE_THRESHOLD = domains.confidence(DOMAIN)
IOU_THRESHOLD = domains.IOU_THRESHOLD

if not PREDICTION_PATH.exists():
    raise SystemExit(f"no predictions.json at {PREDICTION_PATH}\n"
                     f"run:  python3 evaluate.py {DOMAIN}")


def ground_truth_boxes(image_path):
    """Convert YOLO segmentation polygons into enclosing bounding boxes."""
    width, height = Image.open(image_path).size
    label_path = LABEL_DIR / f"{image_path.stem}.txt"
    boxes = []

    if not label_path.exists():
        return np.empty((0, 4), dtype=float)

    for line in label_path.read_text().splitlines():
        values = [float(value) for value in line.split()]
        points = np.asarray(values[1:], dtype=float).reshape(-1, 2)
        boxes.append(
            [
                points[:, 0].min() * width,
                points[:, 1].min() * height,
                points[:, 0].max() * width,
                points[:, 1].max() * height,
            ]
        )

    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def box_iou(targets, predictions):
    if len(targets) == 0 or len(predictions) == 0:
        return np.zeros((len(targets), len(predictions)))

    top_left = np.maximum(targets[:, None, :2], predictions[None, :, :2])
    bottom_right = np.minimum(targets[:, None, 2:], predictions[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    target_area = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
    prediction_area = (predictions[:, 2] - predictions[:, 0]) * (
        predictions[:, 3] - predictions[:, 1]
    )
    union = target_area[:, None] + prediction_area[None, :] - intersection
    return intersection / (union + 1e-7)


def count_matches(targets, predictions):
    """Match predictions and labels one-to-one at IoU >= 0.5."""
    ious = box_iou(targets, predictions)
    possible_matches = np.argwhere(ious >= IOU_THRESHOLD)

    if len(possible_matches) == 0:
        return 0

    possible_matches = possible_matches[
        np.argsort(ious[possible_matches[:, 0], possible_matches[:, 1]])[::-1]
    ]
    possible_matches = possible_matches[
        np.unique(possible_matches[:, 1], return_index=True)[1]
    ]
    possible_matches = possible_matches[
        np.unique(possible_matches[:, 0], return_index=True)[1]
    ]
    return len(possible_matches)


predictions_by_image = defaultdict(list)
for prediction in json.loads(PREDICTION_PATH.read_text()):
    if prediction["score"] < CONFIDENCE_THRESHOLD:
        continue

    x, y, width, height = prediction["bbox"]
    predictions_by_image[Path(prediction["file_name"]).stem].append(
        [x, y, x + width, y + height]
    )

image_paths = sorted(
    path
    for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
)

true_positives = 0
total_predictions = 0
total_labels = 0

for image_path in image_paths:
    targets = ground_truth_boxes(image_path)
    predictions = np.asarray(
        predictions_by_image[image_path.stem], dtype=float
    ).reshape(-1, 4)
    true_positives += count_matches(targets, predictions)
    total_predictions += len(predictions)
    total_labels += len(targets)

false_positives = total_predictions - true_positives
false_negatives = total_labels - true_positives
precision = true_positives / total_predictions if total_predictions else 0.0
recall = true_positives / total_labels if total_labels else 0.0

print(f"{DOMAIN} ({len(image_paths)} images)")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.3f}")
print(f"IoU threshold: {IOU_THRESHOLD:.1f}")
print(f"TP={true_positives}, FP={false_positives}, FN={false_negatives}")
print(f"Precision: {precision:.6f}")
print(f"Recall: {recall:.6f}")
