"""
Classifier training with proper validation, early stopping,
per-class clinical metrics, and saved learning curves.

Key improvements over the original medical.py:
  1. Uses val/ split for model selection (no test-set leakage)
  2. Early stopping on validation loss
  3. Full per-class metrics: precision, recall, F1, specificity, sensitivity
  4. Confusion matrix saved as a plot
  5. AUC-ROC score
  6. Learning curves (train loss, val acc) saved as plots
  7. Results logged to JSON for reproducibility
  8. Model checkpointing (saves best model weights)
"""

import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from data import ImagePathDataset, get_train_transforms, get_eval_transforms


# ─── Confusion Matrix Plot ────────────────────────────────────────────


def _save_confusion_matrix(cm, class_names, save_path):
    """Save a nicely formatted confusion matrix as an image."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ─── Learning Curves Plot ────────────────────────────────────────────


def _save_learning_curves(history, save_path):
    """Save train loss and validation accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Train loss
    ax1.plot(epochs, history["train_loss"], "b-", linewidth=2)
    ax1.set_title("Training Loss", fontsize=13)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    # Val & Test accuracy
    ax2.plot(epochs, history["val_acc"], "g-", linewidth=2, label="Validation")
    ax2.plot(epochs, history["test_acc"], "r--", linewidth=2, label="Test")
    ax2.set_title("Accuracy", fontsize=13)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ─── Core Training Function ──────────────────────────────────────────


def train_and_evaluate(
    experiment_name: str,
    train_files: list[tuple[str, int]],
    val_files: list[tuple[str, int]],
    test_files: list[tuple[str, int]],
    device: torch.device,
) -> dict:
    """
    Train a ResNet18 classifier and evaluate with full clinical metrics.

    Uses validation set for model selection (early stopping) and
    reports final metrics on the held-out test set.

    Args:
        experiment_name: Human-readable experiment label
        train_files:     (path, label) pairs for training
        val_files:       (path, label) pairs for validation
        test_files:      (path, label) pairs for final test
        device:          Compute device

    Returns:
        dict with: accuracy, precision, recall, f1, sensitivity,
                   specificity, auc_roc, confusion_matrix, history
    """
    safe_name = experiment_name.lower().replace(" ", "_").replace("+", "")

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {experiment_name}")
    print(f"{'='*60}")

    # ── Dataset statistics ────────────────────────────────────────
    normal_count = sum(1 for _, l in train_files if l == 0)
    pneumonia_count = sum(1 for _, l in train_files if l == 1)
    print(f"  Train: {len(train_files)} (Normal: {normal_count}, Pneumonia: {pneumonia_count})")
    print(f"  Val:   {len(val_files)}")
    print(f"  Test:  {len(test_files)}")

    # ── Data loaders ──────────────────────────────────────────────
    train_dataset = ImagePathDataset(train_files, transform=get_train_transforms())
    val_dataset = ImagePathDataset(val_files, transform=get_eval_transforms())
    test_dataset = ImagePathDataset(test_files, transform=get_eval_transforms())

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.CLASSIFIER_BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.CLASSIFIER_BATCH_SIZE,
        shuffle=False, num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.CLASSIFIER_BATCH_SIZE,
        shuffle=False, num_workers=2,
    )

    # ── Class weights for imbalanced data ─────────────────────────
    total = len(train_files)
    if normal_count > 0 and pneumonia_count > 0:
        w_normal = total / (2 * normal_count)
        w_pneumonia = total / (2 * pneumonia_count)
        weights = torch.tensor([w_normal, w_pneumonia]).to(device)
        print(f"  Class weights → Normal: {w_normal:.2f}, Pneumonia: {w_pneumonia:.2f}")
    else:
        weights = None
        print("  ⚠️ One class has 0 samples; no class weighting.")

    # ── Model ─────────────────────────────────────────────────────
    print("  Loading ResNet18 (pretrained) ...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 2))
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.CLASSIFIER_LR,
        weight_decay=cfg.CLASSIFIER_WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # ── Training with early stopping ──────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    history = {
        "train_loss": [],
        "val_acc": [],
        "test_acc": [],
    }

    for epoch in range(cfg.CLASSIFIER_EPOCHS):
        # — Train —
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # — Validate —
        val_acc, val_loss = _evaluate_loader(model, val_loader, loss_fn, device)

        # — Test (tracked for plotting, NOT used for model selection) —
        test_acc, _ = _evaluate_loader(model, test_loader, loss_fn, device)

        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

        print(
            f"  Epoch {epoch+1:>2}/{cfg.CLASSIFIER_EPOCHS} | "
            f"Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}%"
        )

        # Early stopping on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
                print(f"  [Early stopping] No improvement for {cfg.EARLY_STOPPING_PATIENCE} epochs.")
                break

    # ── Restore best model ────────────────────────────────────────
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("  Restored best model (by validation loss).")

    # ── Final test evaluation with full metrics ───────────────────
    results = _compute_full_metrics(model, test_loader, device, experiment_name)
    results["history"] = history

    # ── Save artifacts ────────────────────────────────────────────
    # Learning curves
    curves_path = os.path.join(cfg.RESULTS_DIR, f"curves_{safe_name}.png")
    _save_learning_curves(history, curves_path)
    print(f"  Learning curves → {curves_path}")

    # Confusion matrix
    cm_path = os.path.join(cfg.RESULTS_DIR, f"cm_{safe_name}.png")
    _save_confusion_matrix(
        results["confusion_matrix"],
        ["Normal", "Pneumonia"],
        cm_path,
    )
    print(f"  Confusion matrix → {cm_path}")

    # Save best model weights
    model_path = os.path.join(cfg.CHECKPOINTS_DIR, f"classifier_{safe_name}.pt")
    torch.save(best_model_state or model.state_dict(), model_path)
    print(f"  Model weights → {model_path}")

    return results


# ─── Evaluation Helpers ───────────────────────────────────────────────


def _evaluate_loader(model, loader, loss_fn, device):
    """Quick accuracy + loss evaluation on a DataLoader."""
    model.eval()
    correct = total = 0
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += loss_fn(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    acc = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    return acc, avg_loss


@torch.no_grad()
def _compute_full_metrics(model, test_loader, device, experiment_name):
    """
    Compute detailed clinical metrics on the test set.

    Returns dict with:
        accuracy, precision, recall, f1, sensitivity, specificity,
        auc_roc, confusion_matrix, classification_report
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(y.tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())  # P(PNEUMONIA)

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    probs = np.array(all_probs)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    accuracy = 100.0 * (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall for PNEUMONIA
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Recall for NORMAL

    # Per-class precision, recall, F1
    prec, rec, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1], zero_division=0
    )

    # Macro-averaged
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    # AUC-ROC
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0

    # Classification report string
    report = classification_report(
        labels, preds, target_names=["Normal", "Pneumonia"], zero_division=0
    )

    # Print summary
    print(f"\n  ── {experiment_name}: Test Set Results ──")
    print(f"  Accuracy    : {accuracy:.2f}%")
    print(f"  Sensitivity : {sensitivity:.4f} (PNEUMONIA recall — higher = fewer missed cases)")
    print(f"  Specificity : {specificity:.4f} (NORMAL recall — higher = fewer false alarms)")
    print(f"  AUC-ROC     : {auc:.4f}")
    print(f"\n  Per-class metrics:")
    print(f"  {'':>12} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'Normal':>12} {prec[0]:>10.4f} {rec[0]:>8.4f} {f1[0]:>8.4f}")
    print(f"  {'Pneumonia':>12} {prec[1]:>10.4f} {rec[1]:>8.4f} {f1[1]:>8.4f}")
    print(f"  {'Macro avg':>12} {macro_prec:>10.4f} {macro_rec:>8.4f} {macro_f1:>8.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"              Pred Normal  Pred Pneumonia")
    print(f"  True Normal     {tn:>5}         {fp:>5}")
    print(f"  True Pneum.     {fn:>5}         {tp:>5}")

    return {
        "experiment": experiment_name,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc_roc": auc,
        "precision_normal": float(prec[0]),
        "recall_normal": float(rec[0]),
        "f1_normal": float(f1[0]),
        "precision_pneumonia": float(prec[1]),
        "recall_pneumonia": float(rec[1]),
        "f1_pneumonia": float(f1[1]),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
    }
