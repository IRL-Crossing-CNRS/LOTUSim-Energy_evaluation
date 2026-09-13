import csv
from itertools import product
from pathlib import Path
from shutil import copy2

import cv2
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
INPUT_IMAGES = BASE_DIR / "underwater" / "valid" / "images"
INPUT_LABELS = BASE_DIR / "underwater" / "valid" / "labels"
CHECKPOINT = BASE_DIR / "runs/segment/v1_img832/weights/best.pt"

# Inclusive integer search bounds. The script tests every combination within
# these bounds; Gaussian kernel sizes are restricted to valid odd integers.
CLAHE_CLIP_MIN = 1
CLAHE_CLIP_MAX = 10
CLAHE_GRID_MIN = 1
CLAHE_GRID_MAX = 16
GAUSSIAN_KERNEL_MIN = 1
GAUSSIAN_KERNEL_MAX = 15
SHARPEN_AMOUNT_MIN = 1
SHARPEN_AMOUNT_MAX = 2

IMGSZ = 640
BATCH = 16
DEVICE = 0
SCORE_KEY = "metrics/mAP50(B)"

SEARCH_ROOT = BASE_DIR / "underwater_grid_search"
WORK_ROOT = SEARCH_ROOT / "working_dataset"
WORK_IMAGES = WORK_ROOT / "valid" / "images"
WORK_LABELS = WORK_ROOT / "valid" / "labels"
RESULTS_CSV = SEARCH_ROOT / "results.csv"
DATA_YAML = WORK_ROOT / "data.yaml"

METRIC_KEYS = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP50(M)",
    "metrics/mAP50-95(M)",
    "fitness",
]
CSV_FIELDS = [
    "trial_id",
    "clahe_clip_limit",
    "grid_width",
    "grid_height",
    "gaussian_width",
    "gaussian_height",
    "sharpen_original_weight",
    "sharpen_blurred_weight",
    *METRIC_KEYS,
]


def number_for_id(value):
    return str(value).replace(".", "p").replace("-", "m")


def make_trial_id(clip_limit, grid_size, kernel, sharpen_amount):
    return (
        f"clip_{number_for_id(clip_limit)}_"
        f"grid_{grid_size[0]}x{grid_size[1]}_"
        f"gaussian_{kernel[0]}x{kernel[1]}_"
        f"sharpen_{number_for_id(sharpen_amount)}"
    )


def load_existing_results():
    if not RESULTS_CSV.exists():
        return []
    with RESULTS_CSV.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def prepare_working_dataset(image_paths):
    WORK_IMAGES.mkdir(parents=True, exist_ok=True)
    WORK_LABELS.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        label_path = INPUT_LABELS / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label for {image_path.name}: {label_path}"
            )
        copy2(label_path, WORK_LABELS / label_path.name)

    DATA_YAML.write_text(
        f"path: {WORK_ROOT}\n"
        "train: valid/images\n"
        "val: valid/images\n"
        "names:\n"
        "  0: crack\n"
    )


def enhance_images(image_paths, clip_limit, grid_size, kernel, sharpen_amount):
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=grid_size,
    )
    blurred_weight = 1 - sharpen_amount

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read {image_path}")

        clahe_image = clahe.apply(image)
        blurred_image = cv2.GaussianBlur(clahe_image, kernel, 0)
        final_image = cv2.addWeighted(
            clahe_image,
            sharpen_amount,
            blurred_image,
            blurred_weight,
            0,
        )

        output_path = WORK_IMAGES / f"{image_path.stem}.jpg"
        if not cv2.imwrite(str(output_path), final_image):
            raise RuntimeError(f"Failed to save {output_path}")


def append_result(row):
    write_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def best_result(rows):
    scored_rows = [row for row in rows if row.get(SCORE_KEY) not in (None, "")]
    if not scored_rows:
        return None
    return max(scored_rows, key=lambda row: float(row[SCORE_KEY]))


def main():
    if not INPUT_IMAGES.is_dir():
        raise FileNotFoundError(f"Input image directory not found: {INPUT_IMAGES}")
    if not INPUT_LABELS.is_dir():
        raise FileNotFoundError(f"Input label directory not found: {INPUT_LABELS}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT}")

    image_paths = sorted(
        path
        for path in INPUT_IMAGES.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {INPUT_IMAGES}")

    SEARCH_ROOT.mkdir(parents=True, exist_ok=True)
    prepare_working_dataset(image_paths)

    clahe_clip_limits = range(CLAHE_CLIP_MIN, CLAHE_CLIP_MAX + 1)
    clahe_grid_sizes = (
        (size, size) for size in range(CLAHE_GRID_MIN, CLAHE_GRID_MAX + 1)
    )
    gaussian_kernels = (
        (size, size)
        for size in range(GAUSSIAN_KERNEL_MIN, GAUSSIAN_KERNEL_MAX + 1)
        if size % 2 == 1
    )
    sharpen_amounts = range(SHARPEN_AMOUNT_MIN, SHARPEN_AMOUNT_MAX + 1)
    combinations = list(
        product(
            clahe_clip_limits,
            clahe_grid_sizes,
            gaussian_kernels,
            sharpen_amounts,
        )
    )
    existing_rows = load_existing_results()
    completed_ids = {row["trial_id"] for row in existing_rows}
    remaining = sum(
        make_trial_id(*parameters) not in completed_ids
        for parameters in combinations
    )

    print(f"Images: {len(image_paths)}")
    print(f"Grid combinations: {len(combinations)}")
    print(f"Already completed: {len(combinations) - remaining}")
    print(f"Remaining: {remaining}")
    print(f"Results file: {RESULTS_CSV}")

    model = YOLO(CHECKPOINT)
    completed_this_run = 0

    for clip_limit, grid_size, kernel, sharpen_amount in combinations:
        trial_id = make_trial_id(
            clip_limit,
            grid_size,
            kernel,
            sharpen_amount,
        )
        if trial_id in completed_ids:
            continue

        completed_this_run += 1
        print(
            f"\n[{completed_this_run}/{remaining}] {trial_id} "
            f"(blurred weight={1 - sharpen_amount})"
        )
        enhance_images(
            image_paths,
            clip_limit,
            grid_size,
            kernel,
            sharpen_amount,
        )

        results = model.val(
            data=str(DATA_YAML),
            split="val",
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            plots=False,
            verbose=False,
            project=str(SEARCH_ROOT / "validation_runs"),
            name="current",
            exist_ok=True,
        )
        metrics = results.results_dict
        row = {
            "trial_id": trial_id,
            "clahe_clip_limit": clip_limit,
            "grid_width": grid_size[0],
            "grid_height": grid_size[1],
            "gaussian_width": kernel[0],
            "gaussian_height": kernel[1],
            "sharpen_original_weight": sharpen_amount,
            "sharpen_blurred_weight": 1 - sharpen_amount,
        }
        for key in METRIC_KEYS:
            row[key] = float(metrics[key]) if key in metrics else ""

        append_result(row)
        existing_rows.append(row)
        completed_ids.add(trial_id)
        print(f"{SCORE_KEY}: {float(row[SCORE_KEY]):.6f}")

        current_best = best_result(existing_rows)
        print(
            f"Best so far: {current_best['trial_id']} "
            f"({SCORE_KEY}={float(current_best[SCORE_KEY]):.6f})"
        )

    winner = best_result(existing_rows)
    if winner is None:
        print("No completed results were found.")
        return

    print("\nGrid search complete.")
    print(f"Best trial: {winner['trial_id']}")
    for key in CSV_FIELDS[1:]:
        print(f"{key}: {winner[key]}")


if __name__ == "__main__":
    main()
