"""
Evaluate a trained checkpoint on its held-out test split.

Usage:
    python -m src.evaluate --task style --checkpoint checkpoints/style_best.pt
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.dataset import ArtworkDataset
from src.model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["style", "artist", "era", "genre"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-split", default=None,
                         help="Defaults to checkpoints/<task>_test_split.csv as saved by train.py")
    args = parser.parse_args()

    test_split_path = args.test_split or f"{config.CHECKPOINT_DIR}/{args.task}_test_split.csv"
    label_col = {"style": "style", "artist": "artist", "era": "era_label", "genre": "genre"}[args.task]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    test_df = pd.read_csv(test_split_path)
    test_ds = ArtworkDataset(test_df, label_col, label_to_idx, train=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                              num_workers=config.NUM_WORKERS)

    model = build_model(ckpt["backbone"], num_classes=len(label_to_idx)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_labels, all_confidences = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = probs.max(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(confidences.cpu().numpy())

    label_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    # Per-image results CSV — the main file for Tableau. One row per test
    # image with true label, predicted label, confidence, and a correct
    # flag, so you can build confusion-matrix heatmaps, per-class accuracy
    # bars, confidence distributions, and drill into misclassified examples.
    results_df = test_df.reset_index(drop=True).copy()
    results_df["true_label"] = [idx_to_label[i] for i in all_labels]
    results_df["predicted_label"] = [idx_to_label[i] for i in all_preds]
    results_df["confidence"] = all_confidences
    results_df["correct"] = results_df["true_label"] == results_df["predicted_label"]

    Path(config.LOG_DIR).mkdir(exist_ok=True)
    predictions_path = f"{config.LOG_DIR}/{args.task}_predictions.csv"
    results_df.to_csv(predictions_path, index=False)
    print(f"Per-image predictions (Tableau-ready) saved to: {predictions_path}")

    print(f"\n=== Classification report: {args.task} ===")
    print(classification_report(all_labels, all_preds, target_names=label_names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)

    # Long-format confusion matrix CSV for Tableau (Tableau heatmaps expect
    # one row per cell: true_label, predicted_label, count — not a wide
    # matrix). The PNG below is a quick local look; use this CSV in Tableau.
    cm_rows = [
        {"true_label": label_names[i], "predicted_label": label_names[j], "count": int(cm[i, j])}
        for i in range(len(label_names)) for j in range(len(label_names))
    ]
    cm_csv_path = f"{config.LOG_DIR}/{args.task}_confusion_matrix.csv"
    pd.DataFrame(cm_rows).to_csv(cm_csv_path, index=False)
    print(f"Confusion matrix (long format, Tableau-ready) saved to: {cm_csv_path}")

    fig_size = max(6, len(label_names) * 0.4)
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix — {args.task}")
    plt.colorbar()
    tick_marks = range(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=90, fontsize=6)
    plt.yticks(tick_marks, label_names, fontsize=6)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    out_path = f"{config.LOG_DIR}/{args.task}_confusion_matrix.png"
    Path(config.LOG_DIR).mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Confusion matrix saved to {out_path}")


if __name__ == "__main__":
    main()