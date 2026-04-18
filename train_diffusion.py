"""
Improved DDPM diffusion model training for synthetic medical image generation.

Key improvements over the original medical.py:
  1. Resolution:       128×128 (was 64×64)
  2. Architecture:     Larger UNet with more attention at lower resolutions
  3. Noise schedule:   Cosine (squaredcos_cap_v2) instead of linear
  4. EMA:              Exponential moving average for stable generation
  5. Mixed precision:  AMP for faster training + lower VRAM usage
  6. Gradient clipping: Prevents training instability
  7. LR scheduling:    Linear warmup + cosine annealing
  8. Checkpointing:    Resume training from saved states
  9. DDIM sampling:    50 steps instead of 1000 (20× faster, same quality)
  10. Sample grids:    Visual progress monitoring every N epochs
"""

import os
import math
import json
import copy
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, utils as vutils
from tqdm import tqdm

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, DDPMPipeline

import config as cfg


# ─── EMA (Exponential Moving Average) ────────────────────────────────


class EMAModel:
    """
    Maintain an exponential moving average of model parameters.

    EMA produces much smoother and higher-quality samples than
    using the raw training weights directly. Standard practice for
    diffusion models (Dhariwal & Nichol, 2021).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update shadow weights with current model weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module):
        """Replace model weights with EMA weights (for inference)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original training weights after inference."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ─── Learning Rate Scheduler with Warmup ──────────────────────────────


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup followed by cosine decay to 0."""

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Sample Grid Saving ──────────────────────────────────────────────


@torch.no_grad()
def save_sample_grid(model, scheduler, device, epoch, save_dir):
    """Generate and save a 4×4 grid of samples for visual monitoring."""
    model.eval()

    # Use DDIM for fast preview
    ddim = DDIMScheduler.from_config(scheduler.config)
    pipeline = DDPMPipeline(unet=model, scheduler=ddim)
    pipeline.to(device)

    images = pipeline(
        batch_size=16,
        num_inference_steps=cfg.NUM_INFERENCE_STEPS,
        output_type="np",
    ).images  # (16, H, W, 3) numpy in [0, 1]

    # Convert to tensor for grid saving (C, H, W)
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
    grid = vutils.make_grid(tensor, nrow=4, padding=2, normalize=False)

    save_path = os.path.join(save_dir, f"samples_epoch_{epoch:04d}.png")
    vutils.save_image(grid, save_path)
    print(f"  [Samples] Grid saved → {save_path}")


# ─── Main Training Function ──────────────────────────────────────────


def train_diffusion(dataset_root: str, device: torch.device) -> list[str]:
    """
    Train a DDPM on PNEUMONIA chest X-rays and generate synthetic samples.

    Returns:
        list of paths to generated synthetic images
    """
    print("\n" + "=" * 60)
    print("  DIFFUSION MODEL TRAINING")
    print("=" * 60)

    # ── 1. Dataset (filter PNEUMONIA class only) ──────────────────
    from data import get_diffusion_transforms

    full_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_root, "train"),
        transform=get_diffusion_transforms(),
    )

    pneumonia_idx = full_dataset.class_to_idx.get("PNEUMONIA")
    if pneumonia_idx is None:
        raise ValueError("PNEUMONIA class not found in dataset!")

    indices = [
        i
        for i, (_, label) in enumerate(full_dataset.samples)
        if label == pneumonia_idx
    ]
    real_pneumonia_paths = [full_dataset.samples[i][0] for i in indices]
    print(f"[Diffusion] PNEUMONIA images: {len(indices)}")

    subset = Subset(full_dataset, indices)
    dataloader = DataLoader(
        subset,
        batch_size=cfg.DIFFUSION_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # ── 2. Model: Larger UNet for 128×128 ─────────────────────────
    model = UNet2DModel(
        sample_size=cfg.IMG_SIZE_DIFFUSION,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
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
        norm_num_groups=32,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Diffusion] UNet parameters: {param_count:.1f}M")

    # ── 3. Cosine noise schedule (better than linear) ─────────────
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.NUM_TRAIN_TIMESTEPS,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )

    # ── 4. Optimizer + LR schedule ────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=cfg.DIFFUSION_LR)
    total_steps = cfg.DIFFUSION_EPOCHS * len(dataloader)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, cfg.LR_WARMUP_STEPS, total_steps
    )

    # ── 5. EMA ────────────────────────────────────────────────────
    ema = EMAModel(model, decay=cfg.EMA_DECAY)

    # ── 6. Mixed precision ────────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── 7. Resume from checkpoint if available ────────────────────
    ckpt_path = os.path.join(cfg.CHECKPOINTS_DIR, "diffusion_latest.pt")
    start_epoch = 0
    loss_history = []

    if os.path.isfile(ckpt_path):
        print(f"[Diffusion] Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = ckpt["epoch"] + 1
        loss_history = ckpt.get("loss_history", [])
        # Restore EMA
        if "ema" in ckpt:
            ema.shadow = ckpt["ema"]
        print(f"  Resuming from epoch {start_epoch}")

    # ── 8. Training loop ──────────────────────────────────────────
    print(
        f"\n[Diffusion] Training for {cfg.DIFFUSION_EPOCHS} epochs "
        f"({cfg.IMG_SIZE_DIFFUSION}×{cfg.IMG_SIZE_DIFFUSION}) ...\n"
    )

    mse = nn.MSELoss()

    for epoch in range(start_epoch, cfg.DIFFUSION_EPOCHS):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.DIFFUSION_EPOCHS}")
        for batch, _ in pbar:
            clean_images = batch.to(device)
            bs = clean_images.shape[0]

            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=device,
            ).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Forward pass with mixed precision
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = mse(noise_pred, noise)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.GRADIENT_CLIP_NORM
            )
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            # Update EMA
            ema.update(model)

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        current_lr = lr_scheduler.get_last_lr()[0]
        print(
            f"  Epoch {epoch+1} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}"
        )

        # Save sample grid periodically
        if (epoch + 1) % cfg.SAVE_SAMPLES_EVERY == 0 or epoch == 0:
            ema.apply_shadow(model)
            save_sample_grid(model, noise_scheduler, device, epoch + 1, cfg.SAMPLE_DIR)
            ema.restore(model)

        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0 or (epoch + 1) == cfg.DIFFUSION_EPOCHS:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "ema": ema.shadow,
                    "loss_history": loss_history,
                },
                ckpt_path,
            )
            print(f"  [Checkpoint] Saved → {ckpt_path}")

    # Save loss curve data
    with open(os.path.join(cfg.RESULTS_DIR, "diffusion_loss.json"), "w") as f:
        json.dump({"epochs": list(range(1, len(loss_history) + 1)), "loss": loss_history}, f)

    print("\n[Diffusion] Training complete.")

    # ── 9. Generate synthetic images with EMA + DDIM ──────────────
    generated_paths = generate_samples(model, ema, noise_scheduler, device)

    return real_pneumonia_paths, generated_paths


# ─── Sample Generation ────────────────────────────────────────────────


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    ema: EMAModel,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
) -> list[str]:
    """
    Generate synthetic images using EMA weights and DDIM sampling.

    Returns:
        List of file paths to the generated images.
    """
    print(
        f"\n[Generation] Producing {cfg.NUM_SYNTHETIC_SAMPLES} synthetic images "
        f"via DDIM ({cfg.NUM_INFERENCE_STEPS} steps) ..."
    )

    os.makedirs(cfg.SYNTHETIC_DIR, exist_ok=True)

    # Apply EMA weights for better sample quality
    ema.apply_shadow(model)
    model.eval()

    # DDIM scheduler for fast, high-quality sampling
    ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    pipeline = DDPMPipeline(unet=model, scheduler=ddim_scheduler)
    pipeline.to(device)

    batch_size_gen = 16
    num_batches = math.ceil(cfg.NUM_SYNTHETIC_SAMPLES / batch_size_gen)
    generated_paths = []
    generated_count = 0

    for i in tqdm(range(num_batches), desc="Generating"):
        current_bs = min(batch_size_gen, cfg.NUM_SYNTHETIC_SAMPLES - generated_count)
        images = pipeline(
            batch_size=current_bs,
            num_inference_steps=cfg.NUM_INFERENCE_STEPS,
        ).images

        for idx, image in enumerate(images):
            path = os.path.join(
                cfg.SYNTHETIC_DIR, f"syn_{generated_count + idx:04d}.png"
            )
            image.save(path)
            generated_paths.append(path)

        generated_count += current_bs

    # Restore training weights
    ema.restore(model)

    print(f"[Generation] Saved {len(generated_paths)} images → {cfg.SYNTHETIC_DIR}")
    return generated_paths
