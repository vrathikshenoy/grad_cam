"""
Centralized configuration, reproducibility, and device setup.
All hyperparameters in one place for easy experimentation.
"""

import os
import random
import torch
import numpy as np

# ─── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, "synthetic_v2", "PNEUMONIA")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "samples")

# ─── Reproducibility ────────────────────────────────────────────────
SEED = 42

# ─── Diffusion Model ────────────────────────────────────────────────
IMG_SIZE_DIFFUSION = 128  # ↑ from 64 – major quality improvement
DIFFUSION_EPOCHS = 200  # ↑ from 100 – allows convergence
DIFFUSION_LR = 1e-4
DIFFUSION_BATCH_SIZE = 16  # Smaller for 128×128 to fit in VRAM
EMA_DECAY = 0.9999  # Exponential moving average for stable generation
GRADIENT_CLIP_NORM = 1.0  # Prevent training instability
NUM_TRAIN_TIMESTEPS = 1000
NUM_INFERENCE_STEPS = 50  # DDIM steps (↓ from 1000 DDPM – 20× faster)
NUM_SYNTHETIC_SAMPLES = 1000  # ↑ from 200 – meaningful augmentation
SAVE_SAMPLES_EVERY = 50  # Save sample grid every N epochs
LR_WARMUP_STEPS = 500  # Linear warmup before cosine decay

# ─── Classifier ──────────────────────────────────────────────────────
IMG_SIZE_CLASSIFIER = 224
CLASSIFIER_EPOCHS = 20  # ↑ from 10
CLASSIFIER_LR = 1e-4
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5  # Stop if val loss doesn't improve

# ─── Evaluation ──────────────────────────────────────────────────────
FID_BATCH_SIZE = 32
MAX_EVAL_IMAGES = 1000
PR_K_NEAREST = 5  # k for manifold-based Precision & Recall


def set_seed(seed: int = SEED):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Config] Random seed set to {seed}")


def get_device() -> torch.device:
    """Detect and return the best available compute device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Config] Device: {device}")
    if device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  VRAM: {mem:.1f} GB")
    return device


def ensure_dirs():
    """Create all required output directories."""
    for d in [SYNTHETIC_DIR, RESULTS_DIR, CHECKPOINTS_DIR, SAMPLE_DIR]:
        os.makedirs(d, exist_ok=True)
    print("[Config] Output directories ready.")
