#!/usr/bin/env python3
"""The six evaluation domains, in one place.

One model is scored on six image sets. They differ only in where the images
live, how many there are, and the confidence threshold the fixed-threshold
precision/recall is read at -- so every script below takes a domain name and
looks the rest up here rather than carrying its own copy.

The threshold is per modality, not per domain: it is the value that maximised
F1 on the REAL set of that modality (0.485 in air, 0.056 underwater), and the
synthetic counterpart is then read at the same value. Re-tuning it per domain
would compare two different operating points and hide the transfer gap that is
being measured.
"""
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATASETS = BASE / "datasets"
MODEL = BASE / "model" / "best.pt"
RUNS = BASE / "results" / "runs"

# name -> (images, labels, image count, confidence threshold)
DOMAINS = {
    "aerial":                     ("aerial/images",                    "aerial/labels",                    312, 0.485),
    "synthetic_aerial":           ("synthetic_aerial/images/train",    "synthetic_aerial/labels/train",    137, 0.485),
    "underwater":                 ("underwater/valid/images",          "underwater/valid/labels",          175, 0.056),
    "underwater_clahe":           ("underwater_clahe/valid/images",    "underwater_clahe/valid/labels",    175, 0.056),
    "synthetic_underwater":       ("synthetic_underwater/images/train", "synthetic_underwater/labels/train", 148, 0.056),
    "synthetic_underwater_clahe": ("synthetic_underwater_clahe/train/images", "synthetic_underwater_clahe/train/labels", 148, 0.056),
}

IOU_THRESHOLD = 0.5


def images(name: str) -> Path:
    return DATASETS / DOMAINS[name][0]


def labels(name: str) -> Path:
    return DATASETS / DOMAINS[name][1]


def expected_images(name: str) -> int:
    return DOMAINS[name][2]


def confidence(name: str) -> float:
    return DOMAINS[name][3]


def add_domain_argument(parser):
    parser.add_argument("domain", choices=sorted(DOMAINS),
                        help="which evaluation set to score")
    return parser
