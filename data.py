"""
Data loading, dataset download, and path management.

Handles the Kaggle Chest X-ray Pneumonia dataset, which has the structure:
    chest_xray/
    ├── train/   {NORMAL, PNEUMONIA}
    ├── val/     {NORMAL, PNEUMONIA}   ← used for hyperparameter tuning
    └── test/    {NORMAL, PNEUMONIA}   ← held-out final evaluation
"""

import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import kagglehub

import config as cfg


# ─── Dataset Classes ─────────────────────────────────────────────────


class ImagePathDataset(Dataset):
    """Load images from (path, label) pairs with a given transform."""

    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"[Data] Error loading {path}: {e}")
            dummy = torch.zeros(3, cfg.IMG_SIZE_CLASSIFIER, cfg.IMG_SIZE_CLASSIFIER)
            return dummy, label


# ─── Dataset Download & Path Discovery ───────────────────────────────


def download_dataset() -> str:
    """Download the Chest X-ray Pneumonia dataset and return its root path."""
    print("[Data] Downloading dataset via kagglehub ...")
    handle = "paultimothymooney/chest-xray-pneumonia"
    download_path = kagglehub.dataset_download(handle)

    # The dataset sometimes nests an extra level
    root = os.path.join(download_path, "chest_xray", "chest_xray")
    if not os.path.isdir(root):
        root = os.path.join(download_path, "chest_xray")

    print(f"[Data] Dataset root: {root}")
    return root


def _list_images(folder: str) -> list[str]:
    """Return sorted list of image paths in a folder."""
    if not os.path.isdir(folder):
        return []
    files = glob.glob(os.path.join(folder, "*"))
    files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    return sorted(files)


def get_data_splits(dataset_root: str) -> dict:
    """
    Return all image paths organised by split and class.

    Returns:
        {
            "train": {"NORMAL": [...], "PNEUMONIA": [...]},
            "val":   {"NORMAL": [...], "PNEUMONIA": [...]},
            "test":  {"NORMAL": [...], "PNEUMONIA": [...]},
        }
    """
    splits = {}
    for split in ["train", "val", "test"]:
        splits[split] = {}
        for label in ["NORMAL", "PNEUMONIA"]:
            folder = os.path.join(dataset_root, split, label)
            paths = _list_images(folder)
            splits[split][label] = paths

    # Print dataset statistics
    print("\n[Data] Dataset statistics:")
    print(f"  {'Split':<8} {'NORMAL':>8} {'PNEUMONIA':>10} {'Total':>8}")
    print(f"  {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
    for split in ["train", "val", "test"]:
        n = len(splits[split]["NORMAL"])
        p = len(splits[split]["PNEUMONIA"])
        print(f"  {split:<8} {n:>8} {p:>10} {n+p:>8}")
    print()

    return splits


# ─── Transform Factories ─────────────────────────────────────────────


def get_train_transforms() -> transforms.Compose:
    """Augmented transforms for classifier training."""
    return transforms.Compose([
        transforms.RandomResizedCrop(cfg.IMG_SIZE_CLASSIFIER, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Deterministic transforms for validation / test."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(cfg.IMG_SIZE_CLASSIFIER),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_diffusion_transforms() -> transforms.Compose:
    """Transforms for diffusion model training (→ [-1, 1])."""
    return transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE_DIFFUSION, cfg.IMG_SIZE_DIFFUSION)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


# ─── Helper: Build (path, label) lists ───────────────────────────────


def build_file_list(
    splits: dict,
    split_name: str,
    synthetic_paths: list[str] | None = None,
    mode: str = "real_only",
) -> list[tuple[str, int]]:
    """
    Build a (path, label) list for a given experiment mode.

    Args:
        splits:          Output of get_data_splits()
        split_name:      "train", "val", or "test"
        synthetic_paths: List of synthetic PNEUMONIA image paths
        mode:            "real_only" | "synthetic_pneumonia" | "mixed"

    Returns:
        List of (path, label) tuples.  label: 0=NORMAL, 1=PNEUMONIA
    """
    normal = [(p, 0) for p in splits[split_name]["NORMAL"]]
    real_pneumonia = [(p, 1) for p in splits[split_name]["PNEUMONIA"]]
    syn_pneumonia = [(p, 1) for p in (synthetic_paths or [])]

    if mode == "real_only":
        return normal + real_pneumonia
    elif mode == "synthetic_pneumonia":
        # Real NORMAL + Synthetic PNEUMONIA (replaces real pneumonia)
        return normal + syn_pneumonia
    elif mode == "mixed":
        # All real data + synthetic pneumonia as augmentation
        return normal + real_pneumonia + syn_pneumonia
    else:
        raise ValueError(f"Unknown mode: {mode}")
