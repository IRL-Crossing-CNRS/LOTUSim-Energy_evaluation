import cv2
from pathlib import Path
from datetime import datetime
from shutil import copy2


BASE_DIR = Path(__file__).resolve().parent
INPUT_IMAGES = BASE_DIR / "underwater" / "valid" / "images"
INPUT_LABELS = BASE_DIR / "underwater" / "valid" / "labels"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = BASE_DIR / f"underwater_clahe_{timestamp}" / "valid"
OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_LABELS = OUTPUT_ROOT / "labels"

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

print(f"Saving processed dataset to: {OUTPUT_ROOT}")

for entry in sorted(INPUT_IMAGES.iterdir()):
    if entry.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        continue

    image = cv2.imread(str(entry), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read {entry}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(image)
    blurred_image = cv2.GaussianBlur(clahe_image, (7, 7), 0)
    final_image = cv2.addWeighted(clahe_image, 1.2, blurred_image, -0.2, 0)

    output_image = OUTPUT_IMAGES / f"{entry.stem}.jpg"
    if not cv2.imwrite(str(output_image), final_image):
        raise RuntimeError(f"Failed to save {output_image}")

    input_label = INPUT_LABELS / f"{entry.stem}.txt"
    if input_label.exists():
        copy2(input_label, OUTPUT_LABELS / input_label.name)

    print(f"Processed and saved: {output_image.name}")

print(f"Finished. Output dataset: {OUTPUT_ROOT}")
