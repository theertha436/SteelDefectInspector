from pathlib import Path
from collections import Counter
import random
import shutil

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

if not ROOT.exists():
    raise SystemExit(f"Dataset folder not found: {ROOT.resolve()}")

# Identify images by filename, not by parent folder.
images = [
    p for p in ROOT.rglob("*")
    if p.is_file()
    and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
]

def get_class(path):
    name = path.stem.lower()
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

print("Images found:", len(images))
print("Class counts:")
for label, paths in by_class.items():
    print(f"  {label}: {len(paths)}")

if unmatched:
    print("\nUnrecognized filenames (first 10):")
    print(*unmatched[:10], sep="\n")
    raise SystemExit("Fix filename mapping before preparing data.")

if len(images) != 1800 or any(len(v) != 300 for v in by_class.values()):
    raise SystemExit(
        "Expected 1,800 images and 300 per class. "
        "Review the counts above before continuing."
    )

if OUT.exists():
    shutil.rmtree(OUT)

def copy_split(split, label, paths):
    destination = OUT / split / label
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, destination / path.name)

for label, paths in by_class.items():
    paths = paths.copy()
    random.shuffle(paths)

    if label == UNKNOWN_CLASS:
        # Separate unknown validation, operator examples, and test images.
        copy_split("unknown_validation", label, paths[:45])
        copy_split("few_shot_pool", label, paths[45:50])
        copy_split("unknown_test", label, paths[50:])
    else:
        # 210 train, 45 validation, 45 test per known class.
        copy_split("train", label, paths[:210])
        copy_split("validation", label, paths[210:255])
        copy_split("test", label, paths[255:])

print("\nPrepared dataset at:", OUT.resolve())
print("Known training images: 1,050")
print("Known validation images: 225")
print("Known test images: 225")
print("Unknown validation images: 45")
print("Five-shot pool images: 5")
print("Unknown test images: 250")
print("\nDataset preparation completed.")