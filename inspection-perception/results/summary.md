# Results

One model, `model/best.pt`, scored on six image sets. Every figure below is
reproduced by the scripts in this directory's parent — see the commands under
each table.

Precision and recall are box-level, at a fixed confidence threshold, matching
predictions to labels one-to-one at IoU ≥ 0.5. AP50 uses the full score range
instead, with a 95 % confidence interval from 1000 whole-image bootstrap
resamples, seed 42.

The threshold is per modality, not per domain: it is the value that maximised F1
on the **real** set of that modality, and the synthetic counterpart is read at
the same value. Re-tuning per domain would compare two different operating
points and hide the transfer gap being measured.

| Domain | Images | Threshold | Precision | Recall | AP50 | 95 % CI |
|---|---:|---:|---:|---:|---:|---|
| `aerial` (real) | 312 | 0.485 | 0.858 | 0.714 | 0.771 | [0.715, 0.828] |
| `synthetic_aerial` | 137 | 0.485 | 0.950 | 0.542 | 0.688 | [0.624, 0.748] |
| `underwater` (real) | 175 | 0.056 | 0.800 | 0.222 | 0.272 | [0.203, 0.342] |
| `underwater_clahe` (real) | 175 | 0.056 | 0.824 | 0.519 | 0.619 | [0.551, 0.691] |
| `synthetic_underwater` | 148 | 0.056 | 0.796 | 0.206 | 0.233 | [0.204, 0.266] |
| `synthetic_underwater_clahe` | 148 | 0.056 | 0.552 | 0.264 | 0.264 | [0.232, 0.299] |

```bash
python3 precision_recall.py <domain>    # the Precision and Recall columns
python3 bootstrap_ap50.py  <domain>     # the AP50 column and its interval
```

`precision_recall.py` reads the committed `runs/<domain>/predictions.json`, so
it needs neither a GPU nor the model — it reproduces its two columns in seconds
on any machine. `bootstrap_ap50.py` re-runs the model and does need one.

## The transfer gap

The quantity of interest is the drop from a real set to its synthetic
counterpart, read at the same threshold:

| Comparison | Real | Synthetic | Δ AP50 |
|---|---:|---:|---:|
| Aerial | 0.771 | 0.688 | **0.083** |
| Underwater, no preprocessing | 0.272 | 0.233 | **0.039** |
| Underwater, CLAHE | 0.619 | 0.264 | **0.355** |

Read together these say something the individual rows do not.

**The rendered aerial imagery transfers well.** A model that never saw a
rendered image loses 0.083 AP50 on them — the smallest gap of the three, and
the two intervals overlap.

**Raw underwater transfers well too, but at a level where it hardly matters.**
Δ is 0.039, the smallest of all, and both sides sit near 0.25: the model barely
works underwater in either domain, so agreeing is cheap.

**CLAHE is where the simulator and reality part company.** On real underwater
imagery it more than doubles AP50, 0.272 → 0.619, and lifts recall from 0.222
to 0.519. On rendered underwater imagery it moves almost nothing, 0.233 →
0.264, and costs precision (0.796 → 0.552) by promoting false positives —
77 against 19. The gap therefore widens from 0.039 to 0.355, by far the largest
here.

The reading: CLAHE recovers contrast that real water removed, and the renderer
is not removing contrast the same way. Its underwater views are degraded, but
not degraded by the physical mechanism a histogram equaliser inverts. Anything
tuned on rendered underwater imagery — a preprocessing chain, a confidence
threshold — should therefore not be expected to carry to the real thing without
being re-checked there; the aerial case carries a good deal better.

## Files

| | |
|---|---|
| `runs/<domain>/predictions.json` | every detection the model made, with scores; what `precision_recall.py` reads |
| `runs/<domain>/Box*_curve.png`, `Mask*_curve.png` | precision, recall, F1 and PR curves for boxes and masks |
| `runs/<domain>/confusion_matrix*.png` | counts and row-normalised |
| `runs/<domain>/val_batch*_labels.jpg`, `_pred.jpg` | labels beside predictions, for eyeballing |
| `runs/<domain>/labels/` | the per-image detections as text |
