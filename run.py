"""
Main pipeline: ties together diffusion training, evaluation, and classifier experiments.

Usage:
    uv run python run.py

Output structure:
    results/
    ├── generation_metrics.json        # FID, Precision, Recall, Diversity
    ├── classifier_results.json        # All experiment metrics
    ├── tsne_real_vs_synthetic.png      # Feature space visualisation
    ├── curves_*.png                    # Learning curves per experiment
    ├── cm_*.png                        # Confusion matrices per experiment
    └── diffusion_loss.json            # Diffusion training loss curve

    checkpoints/
    ├── diffusion_latest.pt            # Diffusion model (resumable)
    └── classifier_*.pt               # Best classifier per experiment

    samples/
    └── samples_epoch_*.png            # Sample grids during training

    synthetic_v2/PNEUMONIA/
    └── syn_*.png                      # Generated images
"""

import os
import glob
import json
from datetime import datetime

import torch

import config as cfg
from data import download_dataset, get_data_splits, build_file_list
from train_diffusion import train_diffusion
from train_classifier import train_and_evaluate
from evaluate import evaluate_generated_images


def main():
    # ── 0. Setup ──────────────────────────────────────────────────
    cfg.set_seed()
    device = cfg.get_device()
    cfg.ensure_dirs()

    print(f"\n{'='*60}")
    print("  SYNTHETIC MEDICAL IMAGE GENERATION PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # ── 1. Dataset ────────────────────────────────────────────────
    dataset_root = download_dataset()
    splits = get_data_splits(dataset_root)

    # ── 2. Diffusion Training or Load Cached ──────────────────────
    existing_syn = _list_synthetic_images()

    if len(existing_syn) >= cfg.NUM_SYNTHETIC_SAMPLES:
        print(
            f"\n[Pipeline] Found {len(existing_syn)} existing synthetic images. "
            f"Skipping diffusion training."
        )
        synthetic_paths = existing_syn
        real_pneumonia_paths = splits["train"]["PNEUMONIA"]
    else:
        real_pneumonia_paths, synthetic_paths = train_diffusion(dataset_root, device)

    # ── 3. Evaluate Generated Images ──────────────────────────────
    gen_metrics = evaluate_generated_images(
        real_pneumonia_paths, synthetic_paths, device
    )

    # ── 4. Classifier Experiments ─────────────────────────────────
    # Shared val and test sets (never change across experiments)
    val_files = build_file_list(splits, "val")
    test_files = build_file_list(splits, "test")

    all_results = {}

    # — Experiment 1: Real Data Only —
    train_real = build_file_list(splits, "train", mode="real_only")
    results_real = train_and_evaluate(
        "Real Data Only", train_real, val_files, test_files, device
    )
    all_results["real_only"] = results_real

    # — Experiment 2: Synthetic Pneumonia + Real Normal —
    train_syn = build_file_list(
        splits, "train", synthetic_paths=synthetic_paths, mode="synthetic_pneumonia"
    )
    results_syn = train_and_evaluate(
        "Synthetic Pneumonia + Real Normal",
        train_syn, val_files, test_files, device,
    )
    all_results["synthetic_pneumonia"] = results_syn

    # — Experiment 3: Mixed (Real + Synthetic Augmentation) —
    train_mix = build_file_list(
        splits, "train", synthetic_paths=synthetic_paths, mode="mixed"
    )
    results_mix = train_and_evaluate(
        "Mixed (Real + Synthetic)", train_mix, val_files, test_files, device
    )
    all_results["mixed"] = results_mix

    # ── 5. Final Summary ──────────────────────────────────────────
    _print_final_summary(all_results, gen_metrics)

    # ── 6. Save Everything ────────────────────────────────────────
    # Remove non-serialisable items (history tensors are already lists)
    save_results = {}
    for key, val in all_results.items():
        save_results[key] = {
            k: v for k, v in val.items() if k != "confusion_matrix_obj"
        }
    save_results["generation_metrics"] = gen_metrics

    results_path = os.path.join(cfg.RESULTS_DIR, "classifier_results.json")
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n[Pipeline] All results saved → {results_path}")
    print(f"[Pipeline] Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ─── Helpers ──────────────────────────────────────────────────────────


def _list_synthetic_images() -> list[str]:
    """List existing synthetic images in SYNTHETIC_DIR."""
    if not os.path.isdir(cfg.SYNTHETIC_DIR):
        return []
    files = glob.glob(os.path.join(cfg.SYNTHETIC_DIR, "*"))
    files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    return sorted(files)


def _print_final_summary(all_results: dict, gen_metrics: dict):
    """Print a comparison table of all experiments."""
    print("\n" + "=" * 70)
    print("  FINAL RESULTS COMPARISON")
    print("=" * 70)

    # Generation metrics
    print("\n  ── Generation Quality ──")
    print(f"  FID              : {gen_metrics.get('fid', 'N/A'):.2f}")
    print(f"  Precision        : {gen_metrics.get('precision', 'N/A'):.4f}")
    print(f"  Recall           : {gen_metrics.get('recall', 'N/A'):.4f}")
    print(f"  Diversity (SSIM) : {gen_metrics.get('diversity_ssim', 'N/A'):.4f}")

    # Classifier comparison table
    print("\n  ── Classifier Performance ──")
    header = f"  {'Experiment':<35} {'Acc%':>6} {'Sens':>6} {'Spec':>6} {'AUC':>6} {'F1':>6}"
    print(header)
    print(f"  {'─'*35} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")

    for key, r in all_results.items():
        name = r.get("experiment", key)[:35]
        print(
            f"  {name:<35} "
            f"{r['accuracy']:>5.1f}% "
            f"{r['sensitivity']:>6.4f} "
            f"{r['specificity']:>6.4f} "
            f"{r['auc_roc']:>6.4f} "
            f"{r['macro_f1']:>6.4f}"
        )

    print("=" * 70)

    # Key insight
    accs = {k: v["accuracy"] for k, v in all_results.items()}
    best = max(accs, key=accs.get)
    print(f"\n  🏆 Best accuracy: {all_results[best]['experiment']} ({accs[best]:.2f}%)")


if __name__ == "__main__":
    main()
