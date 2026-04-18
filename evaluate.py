"""
Evaluation metrics for synthetic medical image quality.

Implements three research-standard metrics:
  1. FID  (Fréchet Inception Distance)  — distribution-level fidelity
  2. Manifold Precision & Recall        — realism vs. coverage (Kynkäänniemi et al., 2019)
  3. Intra-set SSIM Diversity           — mode collapse detection

Also provides a t-SNE visualisation of real vs. synthetic feature spaces.
"""

import os
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from scipy import linalg
from skimage.metrics import structural_similarity
from sklearn.manifold import TSNE
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

import config as cfg


# ─── Feature Extraction (shared by FID and P&R) ──────────────────────


class _PathDataset(Dataset):
    """Minimal dataset that loads images from file paths."""

    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            return self.transform(img)
        except Exception:
            return torch.zeros(3, 299, 299)


def _get_inception_model(device: torch.device) -> nn.Module:
    """Load InceptionV3 as a frozen 2048-d feature extractor."""
    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.DEFAULT,
        transform_input=False,
    )
    inception.fc = nn.Identity()
    inception.aux_logits = False
    inception = inception.to(device)
    inception.eval()
    for p in inception.parameters():
        p.requires_grad = False
    return inception


def extract_inception_features(
    file_paths: list[str],
    model: nn.Module,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract InceptionV3 pool-3 features (2048-dim) from images.

    Args:
        file_paths: Image paths
        model:      InceptionV3 with Identity() fc
        device:     Compute device
        batch_size: Loader batch size

    Returns:
        np.ndarray of shape (N, 2048)
    """
    tfm = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    loader = DataLoader(
        _PathDataset(file_paths, tfm),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    features = []
    with torch.no_grad():
        for batch in loader:
            feat = model(batch.to(device))
            features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


# ─── Metric 1: FID ───────────────────────────────────────────────────


def compute_fid(
    real_paths: list[str],
    synthetic_paths: list[str],
    device: torch.device,
) -> float:
    """
    Fréchet Inception Distance between real and synthetic sets.

    FID = ||μ_r − μ_g||² + Tr(Σ_r + Σ_g − 2·(Σ_r·Σ_g)^½)

    Lower → better.  Typical medical imaging references:
        <50  : good     50-100 : moderate     >100 : poor
    """
    print("\n  [FID] Loading InceptionV3 ...")
    inception = _get_inception_model(device)

    # Cap images for efficiency
    n_real = min(len(real_paths), cfg.MAX_EVAL_IMAGES)
    n_syn = min(len(synthetic_paths), cfg.MAX_EVAL_IMAGES)

    print(f"  [FID] Extracting features: {n_real} real, {n_syn} synthetic ...")
    real_feats = extract_inception_features(
        real_paths[:n_real], inception, device, cfg.FID_BATCH_SIZE
    )
    syn_feats = extract_inception_features(
        synthetic_paths[:n_syn], inception, device, cfg.FID_BATCH_SIZE
    )

    # Gaussian statistics
    mu_r, sigma_r = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu_g, sigma_g = syn_feats.mean(axis=0), np.cov(syn_feats, rowvar=False)

    # FID formula
    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))
    print(f"  [FID] Score: {fid:.2f}")
    return fid


# ─── Metric 2: Manifold Precision & Recall ────────────────────────────


def _pairwise_l2(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise L2 distances.  (N, D), (M, D) → (N, M)"""
    XX = (X * X).sum(dim=1, keepdim=True)  # (N, 1)
    YY = (Y * Y).sum(dim=1, keepdim=True)  # (M, 1)
    XY = X @ Y.T  # (N, M)
    dists = XX + YY.T - 2 * XY
    return dists.clamp(min=0).sqrt()


def compute_manifold_precision_recall(
    real_paths: list[str],
    synthetic_paths: list[str],
    device: torch.device,
    k: int = cfg.PR_K_NEAREST,
) -> dict:
    """
    Manifold-based Precision & Recall (Kynkäänniemi et al., NeurIPS 2019).

    Uses InceptionV3 features and k-NN to estimate data manifolds:
      • Precision: fraction of synthetic samples that fall within the
        support of the real data manifold → measures realism.
      • Recall: fraction of real samples that fall within the support
        of the synthetic manifold → measures coverage / diversity.

    This is the community-standard metric, NOT the classifier-based
    version that was in the original code.

    Args:
        real_paths:      Real PNEUMONIA image paths
        synthetic_paths: Generated image paths
        device:          torch.device
        k:               Number of nearest neighbours for manifold estimation

    Returns:
        {"precision": float, "recall": float}
    """
    print(f"\n  [P&R] Computing manifold Precision & Recall (k={k}) ...")

    inception = _get_inception_model(device)

    n_real = min(len(real_paths), cfg.MAX_EVAL_IMAGES)
    n_syn = min(len(synthetic_paths), cfg.MAX_EVAL_IMAGES)

    real_feats = extract_inception_features(
        real_paths[:n_real], inception, device, cfg.FID_BATCH_SIZE
    )
    syn_feats = extract_inception_features(
        synthetic_paths[:n_syn], inception, device, cfg.FID_BATCH_SIZE
    )

    real_t = torch.from_numpy(real_feats).float()
    syn_t = torch.from_numpy(syn_feats).float()

    # ── k-NN radii within each set ────────────────────────────────
    # Real-to-real distances → defines real manifold ball radii
    rr_dists = _pairwise_l2(real_t, real_t)
    rr_dists.fill_diagonal_(float("inf"))
    real_radii = rr_dists.kthvalue(k, dim=1).values  # (N_real,)

    # Synthetic-to-synthetic → defines synthetic manifold ball radii
    ss_dists = _pairwise_l2(syn_t, syn_t)
    ss_dists.fill_diagonal_(float("inf"))
    syn_radii = ss_dists.kthvalue(k, dim=1).values  # (N_syn,)

    # ── Cross-distances ───────────────────────────────────────────
    # syn-to-real: (N_syn, N_real)
    sr_dists = _pairwise_l2(syn_t, real_t)

    # Precision: for each synthetic sample, is it inside any real ball?
    # sr_dists[i, j] < real_radii[j] → syn i is in real j's ball
    in_real = (sr_dists < real_radii.unsqueeze(0)).any(dim=1)
    precision = in_real.float().mean().item()

    # Recall: for each real sample, is it inside any synthetic ball?
    # sr_dists.T is (N_real, N_syn);  sr_dists.T[i, j] < syn_radii[j]
    in_syn = (sr_dists.T < syn_radii.unsqueeze(0)).any(dim=1)
    recall = in_syn.float().mean().item()

    print(f"  [P&R] Precision: {precision:.4f} | Recall: {recall:.4f}")
    return {"precision": precision, "recall": recall}


# ─── Metric 3: Intra-set SSIM Diversity ──────────────────────────────


def compute_diversity_ssim(
    synthetic_paths: list[str],
    num_pairs: int = 500,
) -> float:
    """
    Compute mean SSIM between random PAIRS of synthetic images.

    This measures diversity / mode-collapse detection:
      • High SSIM (~0.9+) between synthetics → mode collapse (bad)
      • Moderate SSIM (~0.3–0.6)             → healthy diversity (good)

    This is more meaningful than the original code's random real-vs-synthetic
    SSIM, which conflates structural differences between patients with
    generation quality.

    Args:
        synthetic_paths: Generated image paths
        num_pairs:       Number of random pairs to evaluate

    Returns:
        Mean SSIM (closer to 0 = more diverse)
    """
    print(f"\n  [Diversity] Computing intra-synthetic SSIM ({num_pairs} pairs) ...")

    if len(synthetic_paths) < 2:
        print("  [Diversity] Not enough images for comparison.")
        return 0.0

    tfm = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE_DIFFUSION, cfg.IMG_SIZE_DIFFUSION)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    n = min(num_pairs, len(synthetic_paths) * (len(synthetic_paths) - 1) // 2)
    idx_a = np.random.randint(0, len(synthetic_paths), n)
    idx_b = np.random.randint(0, len(synthetic_paths), n)
    # Ensure different images
    mask = idx_a == idx_b
    idx_b[mask] = (idx_b[mask] + 1) % len(synthetic_paths)

    scores = []
    for a, b in zip(idx_a, idx_b):
        try:
            img_a = tfm(Image.open(synthetic_paths[a]).convert("RGB")).squeeze(0).numpy()
            img_b = tfm(Image.open(synthetic_paths[b]).convert("RGB")).squeeze(0).numpy()
            score = structural_similarity(img_a, img_b, data_range=1.0)
            scores.append(score)
        except Exception as e:
            print(f"  [Diversity] Skipping pair ({a}, {b}): {e}")

    mean_ssim = float(np.mean(scores)) if scores else 0.0
    print(f"  [Diversity] Mean intra-synthetic SSIM: {mean_ssim:.4f}")
    print(
        f"  [Diversity] Interpretation: "
        f"{'⚠️ Possible mode collapse' if mean_ssim > 0.8 else '✓ Healthy diversity'}"
    )
    return mean_ssim


# ─── Visualisation: t-SNE ─────────────────────────────────────────────


def plot_tsne(
    real_paths: list[str],
    synthetic_paths: list[str],
    device: torch.device,
    save_path: str | None = None,
):
    """
    t-SNE scatter plot of InceptionV3 features: real vs. synthetic.

    Visually shows how well synthetic samples overlap with the real
    data distribution in feature space.
    """
    print("\n  [t-SNE] Computing feature embeddings ...")

    inception = _get_inception_model(device)
    n = min(500, len(real_paths), len(synthetic_paths))

    real_feats = extract_inception_features(
        real_paths[:n], inception, device, cfg.FID_BATCH_SIZE
    )
    syn_feats = extract_inception_features(
        synthetic_paths[:n], inception, device, cfg.FID_BATCH_SIZE
    )

    combined = np.concatenate([real_feats, syn_feats], axis=0)
    labels = np.array([0] * len(real_feats) + [1] * len(syn_feats))

    print("  [t-SNE] Running t-SNE (this may take a minute) ...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=cfg.SEED, n_iter=1000)
    coords = tsne.fit_transform(combined)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(
        coords[labels == 0, 0],
        coords[labels == 0, 1],
        c="#2196F3",
        alpha=0.5,
        s=15,
        label=f"Real (n={len(real_feats)})",
    )
    ax.scatter(
        coords[labels == 1, 0],
        coords[labels == 1, 1],
        c="#FF5722",
        alpha=0.5,
        s=15,
        label=f"Synthetic (n={len(syn_feats)})",
    )
    ax.set_title("t-SNE: Real vs Synthetic Feature Space", fontsize=14)
    ax.legend(fontsize=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()

    save_path = save_path or os.path.join(cfg.RESULTS_DIR, "tsne_real_vs_synthetic.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [t-SNE] Saved → {save_path}")


# ─── Unified Evaluation Runner ────────────────────────────────────────


def evaluate_generated_images(
    real_paths: list[str],
    synthetic_paths: list[str],
    device: torch.device,
) -> dict:
    """
    Run all evaluation metrics and produce a summary report.

    Metrics computed:
      1. FID (lower = better)
      2. Manifold Precision (higher = more realistic)
      3. Manifold Recall (higher = more diverse coverage)
      4. Intra-SSIM Diversity (lower = more diverse, higher = mode collapse)

    Also generates a t-SNE plot.

    Returns:
        dict with all metric values
    """
    print("\n" + "=" * 60)
    print("  GENERATIVE MODEL EVALUATION")
    print("=" * 60)

    results = {}

    # 1. FID
    results["fid"] = compute_fid(real_paths, synthetic_paths, device)

    # 2. Manifold Precision & Recall
    pr = compute_manifold_precision_recall(real_paths, synthetic_paths, device)
    results["precision"] = pr["precision"]
    results["recall"] = pr["recall"]

    # 3. Diversity (Intra-synthetic SSIM)
    results["diversity_ssim"] = compute_diversity_ssim(synthetic_paths)

    # 4. t-SNE Visualisation
    plot_tsne(real_paths, synthetic_paths, device)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  FID Score        : {results['fid']:.2f}  (lower = better)")
    print(f"  Precision        : {results['precision']:.4f}  (realism)")
    print(f"  Recall           : {results['recall']:.4f}  (coverage)")
    print(f"  Diversity (SSIM) : {results['diversity_ssim']:.4f}  (lower = more diverse)")
    print("=" * 60)

    # Save to JSON
    save_path = os.path.join(cfg.RESULTS_DIR, "generation_metrics.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {save_path}\n")

    return results
