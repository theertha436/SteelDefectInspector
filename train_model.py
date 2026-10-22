from __future__ import annotations
import argparse, json, random, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from steel_core import (ROOT, MANIFEST, MODEL_DIR, CHECKPOINT, CLASS_BANK, KNOWN_CLASSES,
                        UNKNOWN_CLASS, TRANSFORM, build_model, image_path,
                        load_manifest, extract_embedding, load_image)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

class SteelDataset(Dataset):
    def __init__(self, frame, class_to_idx, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.augment = augment
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        img = load_image(image_path(row.relative_path))
        x = TRANSFORM(img)
        if self.augment and random.random() < 0.5:
            x = torch.flip(x, dims=[2])
        if self.augment and random.random() < 0.25:
            x = torch.flip(x, dims=[1])
        return x, self.class_to_idx[row["class"]], row.relative_path

def get_backbone(pretrained):
    if pretrained:
        try:
            net, dim = build_model(pretrained=True)
            print("Using torchvision ResNet-18 ImageNet weights.")
            return net, dim, "imagenet"
        except Exception as exc:
            warnings.warn(f"Could not load ImageNet weights ({exc}); using random initialization.")
    net, dim = build_model(pretrained=False)
    print("WARNING: training with random initialization.")
    return net, dim, "random"

def train(quick=False):
    started = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)
    df = load_manifest()
    train_df = df[(df.split == "train") & df["class"].isin(KNOWN_CLASSES)].copy()
    val_df = df[df.split == "validation"].copy()
    test_df = df[df.split == "test"].copy()
    unk_val = df[df.split == "unknown_validation"].copy()
    unk_test = df[df.split == "unknown_test"].copy()
    fewshot = df[df.split == "few_shot_pool"].copy()
    if set(train_df["class"].unique()) != set(KNOWN_CLASSES):
        raise RuntimeError("Training manifest must contain exactly the five known classes.")
    if (df[(df.split == "train")]["class"] == UNKNOWN_CLASS).any():
        raise RuntimeError("SCRATCHES must never appear in initial training.")
    class_to_idx = {name: i for i, name in enumerate(KNOWN_CLASSES)}
    model, feature_dim, init_method = get_backbone(pretrained=True)
    # Freeze most of the network for CPU efficiency; train the last residual block and head.
    for p in model.parameters(): p.requires_grad = False
    # model's fc is Identity; add a small classification head on top of pooled features.
    model.fc = nn.Linear(feature_dim, len(KNOWN_CLASSES))
    # train layer4 and head when available
    for p in model.layer4.parameters(): p.requires_grad = True
    for p in model.fc.parameters(): p.requires_grad = True
    model.to("cpu")
    train_loader = DataLoader(SteelDataset(train_df, class_to_idx, True),
                              batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(SteelDataset(val_df, class_to_idx, False),
                            batch_size=64, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
    criterion = nn.CrossEntropyLoss()
    epochs = 2 if quick else 8
    history = []
    t_train = time.time()
    for epoch in range(epochs):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for x, y, _ in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(y)
            correct += int((logits.argmax(1) == y).sum())
            total += len(y)
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                pred = model(x).argmax(1)
                val_correct += int((pred == y).sum())
                val_total += len(y)
        item = {"epoch": epoch + 1, "train_loss": loss_sum / max(total,1),
                "train_accuracy": correct / max(total,1),
                "validation_accuracy": val_correct / max(val_total,1)}
        history.append(item)
        print(f"Epoch {epoch+1}/{epochs}: loss={item['train_loss']:.4f}, "
              f"train_acc={item['train_accuracy']:.3f}, val_acc={item['validation_accuracy']:.3f}")
    train_seconds = time.time() - t_train

    # Save only the backbone and separately trainable classification head for reproducibility.
    model.eval()
    # Turn the classifier into an embedding extractor (ResNet's pooled features).
    embedder = models.resnet18(weights=None)
    embedder.load_state_dict({k:v for k,v in model.state_dict().items()
                              if k in embedder.state_dict() and v.shape == embedder.state_dict()[k].shape},
                             strict=False)
    embedder.fc = nn.Identity()
    embedder.eval()

    def embeddings(frame):
        out_x, out_y = [], []
        for _, row in frame.iterrows():
            image = load_image(image_path(row.relative_path))
            e = extract_embedding(embedder, image)
            out_x.append(e)
            out_y.append(row["class"])
        return np.stack(out_x), np.asarray(out_y)

    # The centroid bank uses known training examples only.
    x_train, y_train = embeddings(train_df)
    centroids = {}
    for cls in KNOWN_CLASSES:
        c = x_train[y_train == cls].mean(axis=0)
        c = c / max(float(np.linalg.norm(c)), 1e-12)
        centroids[cls] = c.tolist()

    # Calibrate a cosine-similarity threshold using held-out known validation and
    # unknown_validation scratches, never using test or few-shot images.
    x_val, y_val = embeddings(val_df)
    x_unk_val, _ = embeddings(unk_val)
    def max_similarity(x):
        centers = np.asarray([centroids[c] for c in KNOWN_CLASSES], dtype=np.float32)
        return (x @ centers.T).max(axis=1)
    known_scores = max_similarity(x_val)
    unknown_scores = max_similarity(x_unk_val)
    # Threshold balances known acceptance and unknown rejection using validation only.
    candidates = np.linspace(min(known_scores.min(), unknown_scores.min()),
                             max(known_scores.max(), unknown_scores.max()), 201)
    best = None
    for threshold in candidates:
        known_accept = (known_scores >= threshold).mean()
        unknown_reject = (unknown_scores < threshold).mean()
        balanced = (known_accept + unknown_reject) / 2
        if best is None or balanced > best[0]:
            best = (balanced, float(threshold), float(known_accept), float(unknown_reject))
    threshold = best[1]

    # Evaluation on known test + unknown test; held-out class is mapped to unknown.
    x_test, y_test = embeddings(test_df)
    x_unk_test, _ = embeddings(unk_test)
    centers = np.asarray([centroids[c] for c in KNOWN_CLASSES], dtype=np.float32)
    test_sim = x_test @ centers.T
    unk_sim = x_unk_test @ centers.T
    test_pred_idx = test_sim.argmax(axis=1)
    unk_best_sim = unk_sim.max(axis=1)
    unk_pred = np.where(unk_best_sim >= threshold,
                        np.asarray(KNOWN_CLASSES)[unk_sim.argmax(axis=1)], "unknown")
    known_best_sim = test_sim.max(axis=1)
    known_pred = np.where(known_best_sim >= threshold,
                          np.asarray(KNOWN_CLASSES)[test_pred_idx], "unknown")
    y_true = np.concatenate([
    y_test,
    np.full(len(x_unk_test), "unknown", dtype=object)
])
    y_pred = np.concatenate([known_pred, unk_pred])
    labels = KNOWN_CLASSES + ["unknown"]
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    # AUROC based on maximum known-class cosine similarity; higher means known.
    auroc = float(roc_auc_score(np.concatenate([np.ones(len(known_best_sim)), np.zeros(len(unk_best_sim))]),
                                np.concatenate([known_best_sim, unk_best_sim])))
    plt.figure(figsize=(9,7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("SteelDefectInspector — known vs unknown")
    plt.tight_layout()
    plt.savefig(ROOT / "outputs" / "confusion_matrix.png", dpi=160)
    plt.close()

    payload = {
        "class_names": KNOWN_CLASSES,
        "centroids": centroids,
        "similarity_threshold": threshold,
        "initialization": init_method,
        "epochs": epochs,
        "seed": SEED,
        "feature_dim": int(x_train.shape[1]),
        "train_seconds": train_seconds,
        "threshold_validation_balanced_accuracy": best[0],
        "known_validation_acceptance": best[2],
        "unknown_validation_rejection": best[3],
        "history": history,
        "evaluation": {"combined_accuracy": acc, "known_vs_unknown_auroc": auroc,
                       "classification_report": report, "confusion_matrix": cm.tolist(),
                       "labels": labels},
    }
    # Store the feature extractor weights (ResNet18 without FC).
    payload["backbone_state"] = {k:v.cpu() for k,v in embedder.state_dict().items()}
    torch.save(payload, CHECKPOINT)
    # Prototypes depend on this exact embedding backbone. Clear stale prototypes
    # whenever the backbone is retrained so old vectors cannot silently be reused.
    CLASS_BANK.write_text(json.dumps({
        "classes": {}, "overrides": {},
        "history": [{"action": "reset_after_model_training", "reason": "Embedding model weights changed"}]
    }, indent=2), encoding="utf-8")
    (ROOT / "outputs" / "metrics.json").write_text(json.dumps({
        k:v for k,v in payload.items() if k != "backbone_state"
    }, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(ROOT / "outputs" / "training_history.csv", index=False)
    print("\nEvaluation results (actual run):")
    print("Combined accuracy:", round(acc, 4))
    print("Known-vs-unknown AUROC:", round(auroc, 4))
    print("Calibrated similarity threshold:", round(threshold, 4))
    print("Confusion matrix:", ROOT / "outputs" / "confusion_matrix.png")
    print("Checkpoint:", CHECKPOINT)
    print("Total elapsed seconds:", round(time.time()-started, 2))
    print("Reserved five-shot examples:", len(fewshot))
    print("NOTE: Five-shot examples were not used for training or evaluation.")
    return payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run two epochs for smoke testing.")
    args = parser.parse_args()
    train(quick=args.quick)
