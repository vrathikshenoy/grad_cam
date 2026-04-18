import os
import glob
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models, utils as vutils
from PIL import Image
import kagglehub
import numpy as np
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline

# ============================================================
# NEW CODE: Additional imports for evaluation metrics
# ============================================================
from scipy import linalg  # For FID matrix sqrt
from skimage.metrics import structural_similarity  # For SSIM
from sklearn.metrics import (  # For Precision / Recall
    precision_score,
    recall_score,
    confusion_matrix,
)
# ============================================================

# Configuration
BATCH_SIZE = 32
CLASSIFIER_EPOCHS = 10
DIFFUSION_EPOCHS = 100
IMG_SIZE_DIFFUSION = 64
IMG_SIZE_CLASSIFIER = 224
NUM_SYNTHETIC_SAMPLES = 200
SYNTHETIC_DIR = "synthetic/PNEUMONIA"


# ============================================================
# NEW CODE: Evaluation helper – load images as tensors
# ============================================================
def load_images_as_tensors(file_paths, img_size, max_images=None, device="cpu"):
    """
    Load a list of image file paths into a single float32 tensor.

    Args:
        file_paths : list of str – paths to .png / .jpg images
        img_size   : int         – resize both dims to this value
        max_images : int | None  – cap number of loaded images
        device     : str         – 'cpu' or 'cuda'

    Returns:
        Tensor of shape (N, 3, img_size, img_size), values in [0, 1]
    """
    tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # → [0, 1]
        ]
    )
    tensors = []
    paths = file_paths[:max_images] if max_images else file_paths
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            tensors.append(tfm(img))
        except Exception as e:
            print(f"  [load_images] skipping {p}: {e}")
    if not tensors:
        return torch.zeros((0, 3, img_size, img_size), device=device)
    return torch.stack(tensors).to(device)


# ============================================================


# ============================================================
# NEW CODE: Metric 1 – Precision & Recall via pretrained CNN
# ============================================================
def compute_precision_recall(real_paths, synthetic_paths, device, batch_size=32):
    """
    Classify real (label=0) vs synthetic (label=1) images with a frozen
    ResNet18 feature extractor + small linear head, then compute precision
    and recall from the confusion matrix.

    This is the "classifier precision / recall" formulation used to measure
    whether the generator is producing realistic samples:
      • High precision  → most synthetic images look real
      • High recall     → the generator covers the real distribution

    Args:
        real_paths      : list of str – paths to real PNEUMONIA images
        synthetic_paths : list of str – paths to generated images
        device          : torch.device

    Returns:
        dict with keys: precision, recall, confusion_matrix
    """
    print("\n  [Precision/Recall] Building dataset …")

    # --- 1. Transforms (ImageNet stats – matches the ResNet backbone) ---
    tfm = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE_CLASSIFIER, IMG_SIZE_CLASSIFIER)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # --- 2. Balanced sampling (cap at min class size) ---
    n = min(len(real_paths), len(synthetic_paths), 500)  # cap for speed
    real_sample = real_paths[:n]
    syn_sample = synthetic_paths[:n]

    all_paths = real_sample + syn_sample
    all_labels = [0] * n + [1] * n  # 0=real, 1=synthetic

    # --- 3. Dataset ---
    class _PairDataset(Dataset):
        def __init__(self, paths, labels, transform):
            self.paths = paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            try:
                img = Image.open(self.paths[i]).convert("RGB")
                return self.transform(img), self.labels[i]
            except Exception:
                return torch.zeros(
                    3, IMG_SIZE_CLASSIFIER, IMG_SIZE_CLASSIFIER
                ), self.labels[i]

    dataset = _PairDataset(all_paths, all_labels, tfm)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # --- 4. Lightweight linear probe on top of frozen ResNet18 ---
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Freeze all layers except the final classifier
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.fc = nn.Linear(backbone.fc.in_features, 2)  # 2 classes
    backbone = backbone.to(device)

    opt = optim.Adam(backbone.fc.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    # Quick 3-epoch fine-tune
    backbone.train()
    for ep in range(3):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            ce(backbone(x), y).backward()
            opt.step()

    # --- 5. Collect predictions ---
    backbone.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            preds = backbone(x.to(device)).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_true.extend(y.tolist())

    # --- 6. Compute metrics ---
    prec = precision_score(all_true, all_preds, pos_label=1, zero_division=0)
    rec = recall_score(all_true, all_preds, pos_label=1, zero_division=0)
    cm = confusion_matrix(all_true, all_preds)

    print(f"  [Precision/Recall] Precision: {prec:.4f} | Recall: {rec:.4f}")
    print(f"  [Precision/Recall] Confusion Matrix:\n{cm}")

    return {"precision": prec, "recall": rec, "confusion_matrix": cm}


# ============================================================


# ============================================================
# NEW CODE: Metric 2 – SSIM (Structural Similarity Index)
# ============================================================
def compute_ssim(
    real_paths, synthetic_paths, img_size=IMG_SIZE_DIFFUSION, num_pairs=200
):
    """
    Compute mean SSIM between randomly-paired real and synthetic images.

    Images are converted to grayscale before comparison (appropriate for
    chest X-rays). Both images are resized to `img_size` × `img_size`.

    Args:
        real_paths      : list of str
        synthetic_paths : list of str
        img_size        : int – resize target (default: 64, matching diffusion output)
        num_pairs       : int – number of random pairs to evaluate

    Returns:
        float – mean SSIM score in [−1, 1] (1 = identical)
    """
    print("\n  [SSIM] Computing structural similarity …")

    tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(),  # single-channel for medical images
            transforms.ToTensor(),  # → [0, 1]
        ]
    )

    # Cap to available images
    n = min(len(real_paths), len(synthetic_paths), num_pairs)
    real_idx = np.random.choice(len(real_paths), n, replace=False)
    syn_idx = np.random.choice(len(synthetic_paths), n, replace=False)

    ssim_scores = []
    for ri, si in zip(real_idx, syn_idx):
        try:
            real_img = tfm(Image.open(real_paths[ri]).convert("RGB")).squeeze(0).numpy()
            syn_img = (
                tfm(Image.open(synthetic_paths[si]).convert("RGB")).squeeze(0).numpy()
            )

            # skimage expects (H, W), values in [0, 1], data_range=1.0
            score = structural_similarity(real_img, syn_img, data_range=1.0)
            ssim_scores.append(score)
        except Exception as e:
            print(f"  [SSIM] skipping pair ({ri}, {si}): {e}")

    mean_ssim = float(np.mean(ssim_scores)) if ssim_scores else 0.0
    print(f"  [SSIM] Mean SSIM over {len(ssim_scores)} pairs: {mean_ssim:.4f}")
    return mean_ssim


# ============================================================


# ============================================================
# NEW CODE: Metric 3 – FID (Fréchet Inception Distance)
# ============================================================
def _extract_inception_features(file_paths, inception_model, device, batch_size=32):
    """
    Helper: run images through InceptionV3 (pool3 features, 2048-dim).

    Images are resized to 299×299 as required by InceptionV3.
    """
    tfm = transforms.Compose(
        [
            transforms.Resize((299, 299)),  # InceptionV3 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    class _PathDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            try:
                return self.transform(Image.open(self.paths[i]).convert("RGB"))
            except Exception:
                return torch.zeros(3, 299, 299)

    loader = DataLoader(
        _PathDataset(file_paths, tfm),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    features = []
    inception_model.eval()
    with torch.no_grad():
        for batch in loader:
            feat = inception_model(batch.to(device))  # (N, 2048)
            features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)  # (N, 2048)


def compute_fid(real_paths, synthetic_paths, device, batch_size=32):
    """
    Compute the Fréchet Inception Distance between real and synthetic images.

    FID = ||μ_r − μ_g||² + Tr(Σ_r + Σ_g − 2·(Σ_r·Σ_g)^½)

    Lower FID → generated images are closer to the real distribution.
    Typical reference values for medical imaging:
        < 50  : good quality
        50-100: moderate
        > 100 : poor quality

    Args:
        real_paths      : list of str – real PNEUMONIA image paths
        synthetic_paths : list of str – generated image paths
        device          : torch.device
        batch_size      : int

    Returns:
        float – FID score
    """
    print("\n  [FID] Loading InceptionV3 …")

    # Build InceptionV3 with the final avg-pool layer as output (2048 features)
    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.DEFAULT, transform_input=False
    )
    # Remove the classification head; expose pool3 (2048-d) features
    inception.fc = nn.Identity()
    inception.aux_logits = False
    inception = inception.to(device)
    for p in inception.parameters():
        p.requires_grad = False

    print("  [FID] Extracting real features …")
    real_feats = _extract_inception_features(real_paths, inception, device, batch_size)

    print("  [FID] Extracting synthetic features …")
    syn_feats = _extract_inception_features(
        synthetic_paths, inception, device, batch_size
    )

    # --- Gaussian statistics ---
    mu_r, sigma_r = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu_g, sigma_g = syn_feats.mean(axis=0), np.cov(syn_feats, rowvar=False)

    # --- FID formula ---
    diff = mu_r - mu_g
    # Matrix square-root via scipy (numerically stable)
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real  # discard negligible imaginary part

    fid = float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))
    print(f"  [FID] Score: {fid:.4f}")
    return fid


# ============================================================


# ============================================================
# NEW CODE: Unified evaluation runner
# ============================================================
def evaluate_generated_images(real_paths, synthetic_paths, device):
    """
    Run all three evaluation metrics and print a summary report.

    Call this after image generation is complete (end of train_diffusion).

    Args:
        real_paths      : list of str – real PNEUMONIA training images
        synthetic_paths : list of str – generated image paths
        device          : torch.device

    Returns:
        dict with keys: fid, ssim, precision, recall
    """
    print("\n" + "=" * 55)
    print("  GENERATIVE MODEL EVALUATION METRICS")
    print("=" * 55)

    results = {}

    # 1. FID
    results["fid"] = compute_fid(real_paths, synthetic_paths, device)

    # 2. SSIM
    results["ssim"] = compute_ssim(real_paths, synthetic_paths)

    # 3. Precision & Recall
    pr = compute_precision_recall(real_paths, synthetic_paths, device)
    results["precision"] = pr["precision"]
    results["recall"] = pr["recall"]

    print("\n" + "=" * 55)
    print("  EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  FID Score  : {results['fid']:.4f}  (lower is better)")
    print(f"  Mean SSIM  : {results['ssim']:.4f}  (higher is better, max=1)")
    print(f"  Precision  : {results['precision']:.4f}  (how realistic the samples are)")
    print(f"  Recall     : {results['recall']:.4f}  (how well distribution is covered)")
    print("=" * 55 + "\n")

    return results


# ============================================================


# --- Diffusion Logic (using diffusers) ---


def train_diffusion(dataset_root, output_path, device):
    print("\n=== Initializing Diffusion Model Training (diffusers) ===")
    print("Preparing training data for Diffusion (Resizing to 64x64)...")

    # 1. Dataset & DataLoader
    dataset = datasets.ImageFolder(
        root=os.path.join(dataset_root, "train"),
        transform=transforms.Compose(
            [
                transforms.Resize((IMG_SIZE_DIFFUSION, IMG_SIZE_DIFFUSION)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    # Filter PNEUMONIA class
    pneumonia_idx = dataset.class_to_idx.get("PNEUMONIA")
    indices = [
        i for i, (_, label) in enumerate(dataset.samples) if label == pneumonia_idx
    ]
    # ============================================================
    # NEW CODE: Keep real PNEUMONIA paths for metric computation
    # ============================================================
    real_pneumonia_paths = [dataset.samples[i][0] for i in indices]
    # ============================================================
    print(f"Found {len(indices)} PNEUMONIA images for Diffusion training.")

    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 2. Model & Scheduler
    model = UNet2DModel(
        sample_size=IMG_SIZE_DIFFUSION,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 128, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    mse = nn.MSELoss()

    print(f"Starting Diffusion training for {DIFFUSION_EPOCHS} epochs...")

    for epoch in range(DIFFUSION_EPOCHS):
        model.train()
        epoch_loss = 0
        for step, (batch, _) in enumerate(dataloader):
            clean_images = batch.to(device)
            bs = clean_images.shape[0]

            noise = torch.randn(clean_images.shape).to(clean_images.device)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            loss = mse(noise_pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{DIFFUSION_EPOCHS} | Loss: {epoch_loss / len(dataloader):.4f}"
        )

    print("Diffusion Training Finished.")

    # --- Sampling / Generation ---
    print(f"Generating {NUM_SYNTHETIC_SAMPLES} synthetic images using DDPMPipeline...")
    os.makedirs(output_path, exist_ok=True)

    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    pipeline.to(device)

    batch_size_gen = 16
    num_batches = math.ceil(NUM_SYNTHETIC_SAMPLES / batch_size_gen)

    generated_count = 0
    # ============================================================
    # NEW CODE: Track generated paths for metric computation
    # ============================================================
    generated_paths = []
    # ============================================================
    for i in range(num_batches):
        current_batch_size = min(
            batch_size_gen, NUM_SYNTHETIC_SAMPLES - generated_count
        )
        images = pipeline(
            batch_size=current_batch_size, num_inference_steps=1000
        ).images

        for idx, image in enumerate(images):
            save_path = f"{output_path}/diff_syn_{generated_count + idx:04d}.png"
            image.save(save_path)
            # ====================================================
            # NEW CODE: Append saved path for evaluation
            # ====================================================
            generated_paths.append(save_path)
            # ====================================================
        generated_count += current_batch_size

    print(f"Saved synthetic images to {output_path}")

    # ============================================================
    # NEW CODE: Run evaluation after generation is complete
    # ============================================================
    evaluate_generated_images(real_pneumonia_paths, generated_paths, device)
    # ============================================================


# --- Standard Helper Functions ---


class SimpleDataset(Dataset):
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
            print(f"Error loading {path}: {e}")
            return torch.zeros((3, IMG_SIZE_CLASSIFIER, IMG_SIZE_CLASSIFIER)), label


def get_data_paths(dataset_root):
    data = {
        "train": {"NORMAL": [], "PNEUMONIA": []},
        "test": {"NORMAL": [], "PNEUMONIA": []},
    }
    for split in ["train", "test"]:
        for label_name in ["NORMAL", "PNEUMONIA"]:
            folder = os.path.join(dataset_root, split, label_name)
            if not os.path.exists(folder):
                continue
            files = glob.glob(os.path.join(folder, "*"))
            files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            data[split][label_name] = files
    return data


def setup_environment(device):
    print("1. Setting up Dataset...")
    dataset_handle = "paultimothymooney/chest-xray-pneumonia"
    download_path = kagglehub.dataset_download(dataset_handle)

    dataset_root = os.path.join(download_path, "chest_xray", "chest_xray")
    if not os.path.exists(dataset_root):
        dataset_root = os.path.join(download_path, "chest_xray")

    print("2. Checking/Generating Synthetic Data...")
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    existing_syn = glob.glob(os.path.join(SYNTHETIC_DIR, "*"))
    existing_syn = [
        f for f in existing_syn if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if len(existing_syn) < NUM_SYNTHETIC_SAMPLES:
        train_diffusion(dataset_root, SYNTHETIC_DIR, device)
        existing_syn = glob.glob(os.path.join(SYNTHETIC_DIR, "*"))
        existing_syn = [
            f for f in existing_syn if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    else:
        print(f"Found {len(existing_syn)} synthetic images. Skipping generation.")

    return dataset_root, existing_syn


def train_and_evaluate(experiment_name, train_files, test_files, device):
    print(f"\n=== Running Experiment: {experiment_name} ===")
    print(f"Training set size: {len(train_files)}")

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE_CLASSIFIER, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE_CLASSIFIER),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = SimpleDataset(train_files, transform=train_transform)
    test_dataset = SimpleDataset(test_files, transform=test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    normal_count = sum(1 for _, label in train_files if label == 0)
    pneumonia_count = sum(1 for _, label in train_files if label == 1)
    total_count = len(train_files)

    if normal_count == 0 or pneumonia_count == 0:
        weights = None
        print("Warning: One class has 0 samples. weights=None")
    else:
        weight_normal = total_count / (2 * normal_count)
        weight_pneumonia = total_count / (2 * pneumonia_count)
        weights = torch.tensor([weight_normal, weight_pneumonia]).to(device)
        print(
            f"Class Weights -> Normal: {weight_normal:.2f}, Pneumonia: {weight_pneumonia:.2f}"
        )

    print("Loading ResNet18 (Pretrained)...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 2))
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    best_acc = 0.0

    for epoch in range(CLASSIFIER_EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = 100 * correct / total
        print(
            f"Epoch {epoch + 1}/{CLASSIFIER_EPOCHS} | Loss: {total_loss / len(train_loader):.4f} | Test Acc: {acc:.2f}%"
        )

        if acc > best_acc:
            best_acc = acc

    print(f"Experiment: {experiment_name} | Best Test Accuracy: {best_acc:.2f}%")
    return best_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_root, synthetic_files = setup_environment(device)
    data_paths = get_data_paths(dataset_root)

    # Prepare Test Set
    test_set = []
    test_set.extend([(p, 0) for p in data_paths["test"]["NORMAL"]])
    test_set.extend([(p, 1) for p in data_paths["test"]["PNEUMONIA"]])

    if not test_set:
        print("Error: Test set is empty.")
        return

    # --- Experiment 1: Real Data Only ---
    train_real = []
    train_real.extend([(p, 0) for p in data_paths["train"]["NORMAL"]])
    train_real.extend([(p, 1) for p in data_paths["train"]["PNEUMONIA"]])

    acc_real = train_and_evaluate("Real Data Only", train_real, test_set, device)

    # --- Experiment 2: Synthetic Data Only ---
    train_syn = []
    train_syn.extend([(p, 0) for p in data_paths["train"]["NORMAL"]])
    train_syn.extend([(p, 1) for p in synthetic_files])

    acc_syn = train_and_evaluate(
        "Synthetic Data (Real Normal + Syn Pneumonia)", train_syn, test_set, device
    )

    # --- Experiment 3: Mixture ---
    train_mix = train_real + [(p, 1) for p in synthetic_files]

    acc_mix = train_and_evaluate("Mixture (Real + Syn)", train_mix, test_set, device)

    print("\n=== Final Results Summary ===")
    print(f"Real Data Only:       {acc_real:.2f}%")
    print(f"Synthetic Data Only:  {acc_syn:.2f}%")
    print(f"Mixture Data:         {acc_mix:.2f}%")

    # ============================================================
    # NEW CODE: Re-run evaluation if synthetic images were
    #           already cached (generation skipped in setup_environment)
    # ============================================================
    if synthetic_files:
        real_pneumonia_paths = data_paths["train"]["PNEUMONIA"]
        print(
            "\n[Post-experiment] Running generative evaluation on cached synthetic images …"
        )
        evaluate_generated_images(real_pneumonia_paths, synthetic_files, device)
    # ============================================================


if __name__ == "__main__":
    main()
