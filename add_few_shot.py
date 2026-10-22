from __future__ import annotations
import argparse
from steel_core import (MODEL_DIR, load_checkpoint, load_class_bank, save_class_bank,
                        load_manifest, image_path, load_image, make_prototype, KNOWN_CLASSES)

def add_class(class_name: str, commit: bool = False, replace: bool = False):
    name = class_name.strip().lower().replace(" ", "_").replace("-", "_")
    if not name:
        raise ValueError("Class name cannot be empty.")
    if name in KNOWN_CLASSES:
        raise ValueError("The five original classes are already represented by trained centroids.")
    df = load_manifest()
    pool = df[df.split == "few_shot_pool"]
    if len(pool) != 5:
        raise RuntimeError(f"Expected exactly five reserved examples, found {len(pool)}.")
    model, _ = load_checkpoint()
    images = [load_image(image_path(row.relative_path)) for _, row in pool.iterrows()]
    prototype, threshold, support_scores = make_prototype(model, images)
    bank = load_class_bank()
    bank.setdefault("classes", {})
    bank.setdefault("history", [])
    if name in bank["classes"] and not replace:
        raise ValueError(f"Class '{name}' already exists. Use --replace to update its prototype.")
    proposal = {
    "prototype": prototype.tolist(),
    "examples": pool.relative_path.tolist(),
    "num_examples": 5,
    "status": "active",
    "operator_approved": True,
    "threshold": threshold,
    "support_similarity_min": min(support_scores),
    "support_similarity_mean": sum(support_scores) / len(support_scores),
    "source": "reserved_pool",
}
    if not commit:
        print({"class_name": name, **proposal})
        print("Preview only. Re-run with --commit to save the prototype.")
        return
    bank["classes"][name] = proposal
    bank["history"].append({"action": "add_or_update", "class_name": name, "num_examples": 5,
                            "threshold": threshold, "source": "reserved_pool"})
    save_class_bank(bank)
    print(f"Added class '{name}' from five reserved examples; threshold={threshold:.4f}.")
    print("Neural network was not retrained.")
    print(f"Class bank: {MODEL_DIR / 'class_bank.json'}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Add/update a class using five reserved few-shot examples")
    p.add_argument("--class-name", required=True, help="Operator-approved label for the new class")
    p.add_argument("--commit", action="store_true", help="Persist the prototype (default is preview)")
    p.add_argument("--replace", action="store_true", help="Replace an existing prototype with the same name")
    a = p.parse_args()
    add_class(a.class_name, a.commit, a.replace)
