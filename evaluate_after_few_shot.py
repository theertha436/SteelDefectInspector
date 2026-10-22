"""Evaluate the newly added class using test images excluded from the 5-shot support set."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from steel_core import (ROOT, OUTPUT_DIR, KNOWN_CLASSES, UNKNOWN_CLASS, load_manifest,
                        load_checkpoint, load_class_bank, image_path, load_image,
                        extract_embedding, predict_embedding)

def main():
    model, payload = load_checkpoint()
    bank = load_class_bank()
    df = load_manifest()
    known_test = df[(df["split"] == "test") & df["class"].isin(KNOWN_CLASSES)].copy()
    new_test = df[(df["split"] == "unknown_test") & (df["class"] == UNKNOWN_CLASS)].copy()
    if len(known_test) == 0 or len(new_test) == 0:
        raise SystemExit("Expected known test images and held-out scratches in unknown_test.")
    labels = KNOWN_CLASSES + [UNKNOWN_CLASS]
    y_true, y_pred = [], []
    for _, row in known_test.iterrows():
        image = load_image(image_path(row["relative_path"]))
        result = predict_embedding(extract_embedding(model, image), payload, bank)
        y_true.append(row["class"])
        y_pred.append(result["label"])
    for _, row in new_test.iterrows():
        image = load_image(image_path(row["relative_path"]))
        result = predict_embedding(extract_embedding(model, image), payload, bank)
        y_true.append(UNKNOWN_CLASS)
        y_pred.append(result["label"])
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    result = {
        "evaluation_protocol": "after_few_shot_update",
        "new_class": UNKNOWN_CLASS,
        "few_shot_support_count": int(bank.get("classes", {}).get(UNKNOWN_CLASS, {}).get("num_examples", 0)),
        "known_test_count": int(len(known_test)),
        "new_class_test_count": int(len(new_test)),
        "combined_accuracy": float(accuracy_score(y_true, y_pred)),
        "known_class_accuracy": float(np.mean([a == b for a, b in zip(y_true[:len(known_test)], y_pred[:len(known_test)])])),
        "new_class_recall": float(report.get(UNKNOWN_CLASS, {}).get("recall", 0.0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
        "note": "The five few-shot support examples are excluded; unknown_test images are used for post-update evaluation. After learning scratches, this test measures scratches recognition, not rejection of a different unseen class."
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "few_shot_evaluation.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["new_class", "few_shot_support_count", "known_test_count", "new_class_test_count", "combined_accuracy", "known_class_accuracy", "new_class_recall"]}, indent=2))
    print(f"Full report saved to: {out}")

if __name__ == "__main__":
    main()
