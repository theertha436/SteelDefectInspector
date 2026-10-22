
from pathlib import Path
from collections import Counter
import csv
import json
import random
import shutil

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
ROOT = Path("data/raw")
OUT = Path("data/prepared")
SEED = 42

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]

# Hold out scratches as the unknown defect class.
UNKNOWN_CLASS = "scratches"

random.seed(SEED)


# ------------------------------------------------------------
# Validate and discover dataset images
# ------------------------------------------------------------
if not ROOT.exists():
    raise SystemExit(
        f"Dataset folder not found: {ROOT.resolve()}"
    )

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

images = [
    path
    for path in ROOT.rglob("*")
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
]


# ------------------------------------------------------------
# Map filenames to defect classes
# ------------------------------------------------------------
def get_class(path: Path):
    """Return the class label inferred from an image filename."""
    name = path.stem.lower().replace("-", "_")

    for label in CLASSES:
        if name.startswith(label):
            return label

    return None


by_class = {label: [] for label in CLASSES}
unmatched = []

for image_path in images:
    label = get_class(image_path)

    if label is None:
        unmatched.append(image_path.name)
    else:
        by_class[label].append(image_path)


# ------------------------------------------------------------
# Validate dataset counts before modifying output
# ------------------------------------------------------------
print("Images found:", len(images))
print("Class counts:")

for label, paths in by_class.items():
    print(f"  {label}: {len(paths)}")

if unmatched:
    print("\nUnrecognized filenames (first 10):")
    print(*unmatched[:10], sep="\n")
    raise SystemExit(
        "Fix filename mapping before preparing data."
    )

if len(images) != 1800 or any(
    len(paths) != 300 for paths in by_class.values()
):
    raise SystemExit(
        "Expected 1,800 images and 300 per class. "
        "Review the counts above before continuing."
    )


# ------------------------------------------------------------
# Prepare output directory
# ------------------------------------------------------------
# Only remove the prepared output after validating the source data.
if OUT.exists():
    shutil.rmtree(OUT)


def copy_split(split: str, label: str, paths: list):
    """Copy images into data/prepared/<split>/<class>/."""
    destination = OUT / split / label
    destination.mkdir(parents=True, exist_ok=True)

    for path in paths:
        shutil.copy2(path, destination / path.name)


# ------------------------------------------------------------
# Split known classes and the unknown class
# ------------------------------------------------------------
for label, class_paths in by_class.items():
    paths = class_paths.copy()
    random.shuffle(paths)

    if label == UNKNOWN_CLASS:
        # Unknown defect: 45 validation, 5 few-shot, 250 test.
        copy_split("unknown_validation", label, paths[:45])
        copy_split("few_shot_pool", label, paths[45:50])
        copy_split("unknown_test", label, paths[50:])
    else:
        # Known classes: 210 train, 45 validation, 45 test.
        copy_split("train", label, paths[:210])
        copy_split("validation", label, paths[210:255])
        copy_split("test", label, paths[255:])


# ------------------------------------------------------------
# Report split counts
# ------------------------------------------------------------
print("\nPrepared dataset at:", OUT.resolve())
print("Known training images: 1,050")
print("Known validation images: 225")
print("Known test images: 225")
print("Unknown validation images: 45")
print("Five-shot pool images: 5")
print("Unknown test images: 250")


# ------------------------------------------------------------
# Generate dataset manifest
# ------------------------------------------------------------
manifest_path = OUT / "manifest.csv"
boxes_path = OUT / "boxes.json"

split_names = [
    "train",
    "validation",
    "test",
    "unknown_validation",
    "few_shot_pool",
    "unknown_test",
]

with manifest_path.open(
    "w", newline="", encoding="utf-8"
) as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["relative_path", "class", "split"])

    for split_name in split_names:
        split_root = OUT / split_name

        if not split_root.exists():
            continue

        for class_dir in sorted(split_root.iterdir()):
            if not class_dir.is_dir():
                continue

            for image_path in sorted(class_dir.iterdir()):
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    relative_path = image_path.relative_to(
                        OUT
                    ).as_posix()

                    writer.writerow(
                        [relative_path, class_dir.name, split_name]
                    )


# ------------------------------------------------------------
# Generate boxes placeholder
# ------------------------------------------------------------
# This is an empty placeholder, not actual bounding-box annotations.
with boxes_path.open("w", encoding="utf-8") as jsonfile:
    json.dump({}, jsonfile, indent=2)


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------
print("\nDataset preparation completed.")
print("Manifest:", manifest_path.resolve())
print("Boxes placeholder:", boxes_path.resolve())
