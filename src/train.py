"""
Train one of the three tasks: style, artist, or era.

Usage:
    python -m src.train --task style --metadata metadata.csv --epochs 10
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.dataset import ArtworkDataset, load_splits
from src.model import build_model


def compute_class_weights(train_df, label_col, label_to_idx, device):
    counts = train_df[label_col].value_counts()
    weights = torch.zeros(len(label_to_idx))
    for label, idx in label_to_idx.items():
        weights[idx] = 1.0 / max(counts.get(label, 1), 1)
    weights = weights / weights.sum() * len(label_to_idx)  # normalize
    return weights.to(device)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_n += images.size(0)

    return total_loss / total_n, total_correct / total_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["style", "artist", "era", "genre"])
    parser.add_argument("--metadata", default="metadata.csv")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--backbone", default=config.BACKBONE)
    parser.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE,
                         help="Stop if val_acc hasn't improved for this many epochs "
                              "(0 disables early stopping).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("  WARNING: training on CPU will be very slow for a real WikiArt-sized dataset.")

    label_col = {"style": "style", "artist": "artist", "era": "era_label", "genre": "genre"}[args.task]
    train_df, val_df, test_df, label_to_idx = load_splits(args.metadata, args.task)

    train_ds = ArtworkDataset(train_df, label_col, label_to_idx, train=True)
    val_ds = ArtworkDataset(val_df, label_col, label_to_idx, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=config.NUM_WORKERS, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=(device.type == "cuda"))

    model = build_model(args.backbone, num_classes=len(label_to_idx)).to(device)

    class_weights = compute_class_weights(train_df, label_col, label_to_idx, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    # Cosine schedule: LR eases down to ~0 by the final epoch instead of
    # staying constant. This is usually why the last couple epochs of a
    # fixed-LR run make val_acc *worse* (as in the runs so far) — a high LR
    # late in training keeps knocking the model out of a good minimum.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    Path(config.CHECKPOINT_DIR).mkdir(exist_ok=True)
    Path(config.LOG_DIR).mkdir(exist_ok=True)
    best_val_acc = 0.0
    epochs_since_improvement = 0
    history = []  # one row per epoch, written to CSV for Tableau/plotting

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        history.append({
            "task": args.task, "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_since_improvement = 0
            ckpt_path = Path(config.CHECKPOINT_DIR) / f"{args.task}_best.pt"
            torch.save({
                "model_state": model.state_dict(),
                "label_to_idx": label_to_idx,
                "backbone": args.backbone,
                "task": args.task,
            }, ckpt_path)
            print(f"  Saved new best checkpoint to {ckpt_path} (val_acc={val_acc:.4f})")
        else:
            epochs_since_improvement += 1
            if args.patience and epochs_since_improvement >= args.patience:
                print(f"  No val_acc improvement for {epochs_since_improvement} epochs "
                      f"(patience={args.patience}); stopping early.")
                break

    # Save test split for evaluate.py to use later, so eval matches this exact split.
    test_df.to_csv(f"{config.CHECKPOINT_DIR}/{args.task}_test_split.csv", index=False)

    # Save per-epoch metrics — this is the file to plug into Tableau for
    # train/val loss and accuracy curves (across one or all three tasks,
    # since each row is tagged with 'task').
    history_path = Path(config.LOG_DIR) / f"{args.task}_training_history.csv"
    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print(f"\nBest val_acc for task '{args.task}': {best_val_acc:.4f}")
    print(f"Test split saved for evaluate.py: {config.CHECKPOINT_DIR}/{args.task}_test_split.csv")
    print(f"Training history (Tableau-ready) saved to: {history_path}")


if __name__ == "__main__":
    main()