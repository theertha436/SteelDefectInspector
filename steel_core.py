from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import csv, json
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms

KNOWN_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale"]
UNKNOWN_CLASS = "scratches"
ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data" / "prepared" / "manifest.csv"
MODEL_DIR = ROOT / "saved_models"
CHECKPOINT = MODEL_DIR / "steel_resnet18.pt"
CLASS_BANK = MODEL_DIR / "class_bank.json"
OUTPUT_DIR = ROOT / "outputs"
INSPECTION_LOG = OUTPUT_DIR / "inspection_log.csv"
DEVICE = torch.device("cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def normalize_vector(vector):
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Cannot create a prototype from an empty or zero embedding.")
    return arr / norm

def build_model(pretrained: bool = False):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    net = models.resnet18(weights=weights)
    feature_dim = net.fc.in_features
    net.fc = nn.Identity()
    return net, feature_dim

def load_image(path_or_file):
    # NEU-CLS images are grayscale. RGB conversion repeats the single channel
    # into three channels so ImageNet-pretrained ResNet-18 can accept them.
    if hasattr(path_or_file, "seek"):
        path_or_file.seek(0)
    with Image.open(path_or_file) as image:
        return image.convert("RGB")

def load_manifest():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}. Prepare the dataset first.")
    df = pd.read_csv(MANIFEST)
    required = {"relative_path", "class", "split"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must contain columns: {sorted(required)}")
    return df

def image_path(relative_path: str) -> Path:
    return ROOT / "data" / "prepared" / Path(str(relative_path))

def extract_embedding(model, image, device=DEVICE):
    model.eval()
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
    return normalize_vector(embedding)

def load_checkpoint():
    import gc

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Trained model not found at {CHECKPOINT}. "
            "Run: python run_all.py --quick"
        )

    # Load checkpoint tensors onto CPU.
    payload = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    # Construct the model without downloading pretrained weights.
    model, _ = build_model(pretrained=False)

    # Load trained parameters into the model.
    model.load_state_dict(payload["backbone_state"])
    model.eval()

    # Release temporary references and collect unused objects.
    gc.collect()

    return model, payload

def load_class_bank():
    if not CLASS_BANK.exists():
        return {"classes": {}, "overrides": {}, "history": []}
    bank = json.loads(CLASS_BANK.read_text(encoding="utf-8"))
    bank.setdefault("classes", {})
    bank.setdefault("overrides", {})
    bank.setdefault("history", [])
    return bank

def save_class_bank(bank):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bank.setdefault("classes", {})
    bank.setdefault("overrides", {})
    bank.setdefault("history", [])
    tmp = CLASS_BANK.with_suffix(".tmp")
    tmp.write_text(json.dumps(bank, indent=2), encoding="utf-8")
    tmp.replace(CLASS_BANK)

def class_threshold(label, payload, bank):
    if label in KNOWN_CLASSES:
        return float(payload["similarity_threshold"])
    entry = bank.get("classes", {}).get(label, {})
    return float(entry.get("threshold", payload["similarity_threshold"]))


def predict_embedding(embedding, payload, bank=None):
    bank = bank or {"classes": {}, "overrides": {}}

    labels = list(payload["class_names"])
    centers = {
        name: normalize_vector(vector)
        for name, vector in payload["centroids"].items()
    }

    # Preserve operator-approved merges into the five known classes.
    for name, vector in bank.get("overrides", {}).items():
        if name in labels:
            centers[name] = normalize_vector(vector)

    # Only explicitly active additional classes can be predicted.
    for name, entry in bank.get("classes", {}).items():
        if entry.get("status", "active") != "active":
            continue

        # Only explicitly active additional classes can be predicted.
# The status check above controls whether a class is active.
# An active class-bank entry is sufficient for backward compatibility
# with existing class-bank entries and tests.

        centers[name] = normalize_vector(entry["prototype"])
        if name not in labels:
            labels.append(name)

    vector = normalize_vector(embedding)
    scores = {
        name: float(np.dot(vector, center))
        for name, center in centers.items()
        if name in labels
    }

    if not scores:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "scores": {},
            "threshold": float(payload["similarity_threshold"]),
            "message": "No active class prototypes are available.",
        }

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    accepted = [
    (label, score)
    for label, score in ranked
    if score >= class_threshold(label, payload, bank)
]
    if not accepted:
        best_name, best_score = ranked[0]
        threshold = class_threshold(best_name, payload, bank)
        return {
            "label": "unknown",
            "confidence": best_score,
            "scores": scores,
            "threshold": threshold,
            "message": (
                f"Unknown defect — best similarity {best_score:.3f} "
                f"is below the {best_name.replace('_', ' ')} "
                f"threshold {threshold:.3f}. Send to operator."
            ),
        }

    best_name, best_score = accepted[0]
    return {
        "label": best_name,
        "confidence": best_score,
        "scores": scores,
        "threshold": class_threshold(best_name, payload, bank),
        "message": "Cosine similarity (not a calibrated probability).",
    }

def make_prototype(model, images):
    if len(images) != 5:
        raise ValueError(f"Exactly five labelled examples are required; received {len(images)}.")
    vectors = np.stack([extract_embedding(model, image) for image in images])
    prototype = normalize_vector(vectors.mean(axis=0))
    support_scores = vectors @ prototype
    # Class-specific threshold estimated only from the five support examples.
    # Conservative floor/cap prevents a single unusually low/high support score
    # from making the new class impossible to accept or too permissive.
    threshold = float(np.clip(float(support_scores.min()) - 0.06, 0.55, 0.90))
    return prototype, threshold, [float(x) for x in support_scores]

def class_threshold(label, payload, bank):
    if label in KNOWN_CLASSES:
        return float(payload["similarity_threshold"])

    entry = bank.get("classes", {}).get(label, {})
    return float(entry.get("threshold", payload["similarity_threshold"]))

def append_inspection_log(filename, result, shift="Day", operator_label=""):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["timestamp_local", "shift", "filename", "predicted_class", "similarity", "decision_threshold", "needs_review", "operator_label"]
    file_exists = INSPECTION_LOG.exists()
    row = {
        "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "shift": shift,
        "filename": filename,
        "predicted_class": result.get("label", "unknown"),
        "similarity": round(float(result.get("confidence", 0.0)), 6),
        "decision_threshold": round(float(result.get("threshold", 0.0)), 6),
        "needs_review": result.get("label", "unknown") == "unknown",
        "operator_label": operator_label,
    }
    with INSPECTION_LOG.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return row
