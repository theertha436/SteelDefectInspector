from pathlib import Path
from collections import Counter

DATA_DIR = Path("data/raw")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

if not DATA_DIR.exists():
    print("ERROR: data/raw folder not found.")
    raise SystemExit(1)

images = [
    path for path in DATA_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in EXTENSIONS
]

print("Total images found:", len(images))
print("\nImages by parent folder:")

counts = Counter(path.parent.name for path in images)
for folder, count in sorted(counts.items()):
    print(f"{folder}: {count}")

print("\nFirst 10 image paths:")
for path in images[:10]:
    print(path)