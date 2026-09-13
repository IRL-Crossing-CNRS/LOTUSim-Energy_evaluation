# Corrosion and crack detection: does it work on rendered imagery?

An offshore turbine is inspected for two things: corroded steel and cracked
concrete. The pipeline looks for both. As the vehicle flies a tower, its camera
stream feeds two detectors over ROS 2 — **corrosion**, by colour thresholding in
HSV and CIELab, and **cracks**, by a YOLO segmentation model — and their
labelled output is what the mission acts on. That loop is only worth building if
a detector behaves on rendered imagery as it would on the real thing; otherwise
everything tuned in simulation has to be re-tuned in the field.

This directory measures that, for the crack detector. It is used here as a
**fixed instrument for measuring the imagery**, not as a capability being
claimed: a public YOLO11n-seg checkpoint fine-tuned on a public dataset, a
reproducible stand-in for whatever detector a user would bring. The corrosion
threshold is implemented in the pipeline and is not characterised here.

The model, trained only on real photographs, is scored on six image sets — real
and rendered, in air and underwater, with and without contrast preprocessing —
and the gap between each real set and its rendered counterpart is the result.

Everything here is offline — the model, the six image sets and the six
evaluation runs are all committed — so it needs no running simulator. Feeding a
live scenario is `publish_compressed_image.py`, at the bottom of this page.

**Results and what they mean: [`results/summary.md`](results/summary.md).**

## Quick start

Everything runs from this directory.

```bash
cd inspection-perception
pip install ultralytics opencv-python numpy pillow

python3 precision_recall.py aerial       # no GPU, no model: reads the committed run
python3 bootstrap_ap50.py  aerial        # needs a GPU; re-runs the model
python3 evaluate.py        aerial        # needs a GPU; regenerates the whole run
```

Domain names are the six directories under `datasets/`: `aerial`,
`synthetic_aerial`, `underwater`, `underwater_clahe`, `synthetic_underwater`,
`synthetic_underwater_clahe`.

`precision_recall.py` is the cheapest check that this repository is intact: it
reads `results/runs/<domain>/predictions.json` and recomputes the Precision and
Recall columns of the results table in seconds, on any machine.

## What is here

| Path | |
|---|---|
| `domains.py` | the six domains in one table: images, labels, count, confidence threshold. Every script looks them up here |
| `model/best.pt` | the evaluated checkpoint — YOLO11n-seg, the only model in the study |
| `datasets/<domain>/` | images, labels and a `dataset.yaml` per domain |
| `evaluate.py` | score the model on one domain → `results/runs/<domain>/` |
| `precision_recall.py` | precision and recall at the fixed threshold, from a run's `predictions.json` |
| `bootstrap_ap50.py` | AP50 with a bootstrap 95 % confidence interval |
| `results/summary.md` | the results table, the transfer gaps, and what they mean |
| `results/runs/<domain>/` | the committed evaluation runs: curves, confusion matrices, predictions, previews |
| `preprocessing/` | how the CLAHE and domain-adaptation variants were produced, and how the model was trained |
| `publish_compressed_image.py` | put a still on a scenario's inspection camera topic |

## The model

YOLO11n-seg (Ultralytics), one class, `crack`, trained on the public
`crack-bphdr` dataset — Roboflow Universe, Public Domain,
<https://universe.roboflow.com/university-bswxt/crack-bphdr>. Release 1's
1239 training images; its 312-image test split is the `aerial` domain here.

`preprocessing/train_model.py` carries the exact hyperparameters: 100 epochs at
832 px, SGD, `lr0=0.00777`, `mask_ratio=4`, `seed=0`, `deterministic=True`.
Retraining needs the training split, which is **not** committed here — this
repository ships the trained checkpoint instead, which is all the reported
numbers depend on. Download the training split from the Roboflow link above if
you want to retrain.

```bash
python3 preprocessing/checkpoint_settings.py   # print a checkpoint's epoch and training args
```

## The six domains

| Domain | Images | What it is |
|---|---:|---|
| `aerial` | 312 | real photographs; release 1's held-out test split |
| `synthetic_aerial` | 137 | rendered views of cracked turbine towers |
| `underwater` | 175 | real underwater photographs of cracked concrete |
| `underwater_clahe` | 175 | the same 175, after CLAHE |
| `synthetic_underwater` | 148 | rendered underwater views |
| `synthetic_underwater_clahe` | 148 | the same 148, after CLAHE |

The rendered sets come from LOTUSim-Energy's Unity project. For the aerial set,
`datasets/synthetic_aerial/capture_log.csv` records one row per image: the scene
(`Assets/Scenes/Energy/demo_facet.unity`), the crack decal object on the turbine,
the material and the base-colour texture it used — four crack textures across
the set. `previews/` holds the same frames with the label drawn on, for
eyeballing the annotation.

Every `dataset.yaml` points its `val` split at the full image set. These are
evaluation sets, not training splits: nothing is held out inside them.

## Preprocessing

CLAHE is the one preprocessing step in the reported results. The committed
setting is clip limit 2.0, 8×8 tiles, a 7×7 Gaussian blur and an unsharp mask at
weights 1.2 / −0.2:

```bash
python3 preprocessing/apply_clahe.py              # writes a timestamped copy beside the input
python3 preprocessing/apply_clahe_and_validate.py # the same, then scores the result in one step
python3 preprocessing/grid_search_clahe.py        # sweep clip limit, tile grid, blur, sharpen; ranks on mAP50
```

Both scripts resolve their input relative to themselves and expect the dataset
layout they were written against — read the paths at the top of each before
running, and point them at the domain you mean.

Two further scripts move a rendered image towards the appearance of a real one
instead of correcting it towards the training domain. They are exploratory: no
reported number depends on them.

```bash
python3 preprocessing/fda.py                 # Fourier domain adaptation: swap low-frequency amplitude
python3 preprocessing/diffusion_texture.py   # low-strength img2img; downloads a ~5 GB model on first run
```

## Putting the model in the loop

```bash
source /opt/ros/jazzy/setup.bash
python3 publish_compressed_image.py test_image_2.png --topic /<scenario>/inspection/image --once
```

Publishes a still as `sensor_msgs/CompressedImage` on an inspection camera
topic, which is how a segmentation model is fed from a running scenario without
a camera. The default topic is the inspection feed of a `patrol_then_inspect`
scenario; pass `--topic` for any other.

## Not committed

Training splits (see the Roboflow link above), Ultralytics `runs/` directories
from retraining, grid-search working copies, and `*.pt` files other than
`model/best.pt`.
