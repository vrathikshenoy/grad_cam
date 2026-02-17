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

# Configuration
BATCH_SIZE = 32
CLASSIFIER_EPOCHS = 10
DIFFUSION_EPOCHS = 100 
IMG_SIZE_DIFFUSION = 64  
IMG_SIZE_CLASSIFIER = 224
NUM_SYNTHETIC_SAMPLES = 200
SYNTHETIC_DIR = "synthetic/PNEUMONIA"

# --- Diffusion Logic (using diffusers) ---

def train_diffusion(dataset_root, output_path, device):
    print("\n=== Initializing Diffusion Model Training (diffusers) ===")
    print("Preparing training data for Diffusion (Resizing to 64x64)...")
    
    # 1. Dataset & DataLoader
    dataset = datasets.ImageFolder(
        root=os.path.join(dataset_root, "train"),
        transform=transforms.Compose([
            transforms.Resize((IMG_SIZE_DIFFUSION, IMG_SIZE_DIFFUSION)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # [-1, 1] range for DDPM
        ])
    )
    
    # Filter PNEUMONIA class
    pneumonia_idx = dataset.class_to_idx.get('PNEUMONIA')
    indices = [i for i, (_, label) in enumerate(dataset.samples) if label == pneumonia_idx]
    print(f"Found {len(indices)} PNEUMONIA images for Diffusion training.")
    
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # 2. Model & Scheduler
    # A standard U-Net configuration for 64x64 generation
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

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            # Predict the noise residual
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            
            loss = mse(noise_pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{DIFFUSION_EPOCHS} | Loss: {epoch_loss/len(dataloader):.4f}")
        
    print("Diffusion Training Finished.")
    
    # --- Sampling / Generation ---
    print(f"Generating {NUM_SYNTHETIC_SAMPLES} synthetic images using DDPMPipeline...")
    os.makedirs(output_path, exist_ok=True)
    
    # Create pipeline for easy sampling
    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    pipeline.to(device)
    
    # Generate in batches to save memory
    batch_size_gen = 16
    num_batches = math.ceil(NUM_SYNTHETIC_SAMPLES / batch_size_gen)
    
    generated_count = 0
    for i in range(num_batches):
        current_batch_size = min(batch_size_gen, NUM_SYNTHETIC_SAMPLES - generated_count)
        
        # Pipeline output is a dictionary, "images" key contains PIL images by default
        # But we can ask for numpy or tensor if needed. Default is PIL.
        images = pipeline(batch_size=current_batch_size, num_inference_steps=1000).images
        
        for idx, image in enumerate(images):
            save_path = f"{output_path}/diff_syn_{generated_count + idx:04d}.png"
            image.save(save_path)
            
        generated_count += current_batch_size
            
    print(f"Saved synthetic images to {output_path}")

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
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # Fallback for corrupted images
            print(f"Error loading {path}: {e}")
            return torch.zeros((3, IMG_SIZE_CLASSIFIER, IMG_SIZE_CLASSIFIER)), label

def get_data_paths(dataset_root):
    data = {'train': {'NORMAL': [], 'PNEUMONIA': []}, 'test': {'NORMAL': [], 'PNEUMONIA': []}}
    for split in ['train', 'test']:
        for label_name in ['NORMAL', 'PNEUMONIA']:
            folder = os.path.join(dataset_root, split, label_name)
            if not os.path.exists(folder): continue
            files = glob.glob(os.path.join(folder, "*"))
            files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
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
    existing_syn = [f for f in existing_syn if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(existing_syn) < NUM_SYNTHETIC_SAMPLES:
        train_diffusion(dataset_root, SYNTHETIC_DIR, device)
        existing_syn = glob.glob(os.path.join(SYNTHETIC_DIR, "*"))
        existing_syn = [f for f in existing_syn if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        print(f"Found {len(existing_syn)} synthetic images. Skipping generation.")
        
    return dataset_root, existing_syn

def train_and_evaluate(experiment_name, train_files, test_files, device):
    print(f"\n=== Running Experiment: {experiment_name} ===")
    print(f"Training set size: {len(train_files)}")
    
    # 1. Advanced Data Augmentation for Training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE_CLASSIFIER, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Standard Transforms for Testing
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE_CLASSIFIER),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = SimpleDataset(train_files, transform=train_transform)
    test_dataset = SimpleDataset(test_files, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 2. Calculate Class Weights for Imbalance
    # Labels: 0 = NORMAL, 1 = PNEUMONIA
    normal_count = sum(1 for _, label in train_files if label == 0)
    pneumonia_count = sum(1 for _, label in train_files if label == 1)
    total_count = len(train_files)
    
    if normal_count == 0 or pneumonia_count == 0:
        weights = None
        print("Warning: One class has 0 samples. weights=None")
    else:
        # Standard weighting: total / (num_classes * class_count)
        weight_normal = total_count / (2 * normal_count)
        weight_pneumonia = total_count / (2 * pneumonia_count)
        weights = torch.tensor([weight_normal, weight_pneumonia]).to(device)
        print(f"Class Weights -> Normal: {weight_normal:.2f}, Pneumonia: {weight_pneumonia:.2f}")

    # 3. Model Setup (ResNet18)
    print("Loading ResNet18 (Pretrained)...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Replace head with Dropout + Linear
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 2)
    )
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
        
        # Validation on Test Set (to track progress)
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
        print(f"Epoch {epoch+1}/{CLASSIFIER_EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}%")
        
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
    test_set.extend([(p, 0) for p in data_paths['test']['NORMAL']])
    test_set.extend([(p, 1) for p in data_paths['test']['PNEUMONIA']])
    
    if not test_set:
        print("Error: Test set is empty.")
        return

    # --- Experiment 1: Real Data Only ---
    train_real = []
    train_real.extend([(p, 0) for p in data_paths['train']['NORMAL']])
    train_real.extend([(p, 1) for p in data_paths['train']['PNEUMONIA']])
    
    acc_real = train_and_evaluate("Real Data Only", train_real, test_set, device)
    
    # --- Experiment 2: Synthetic Data Only ---
    train_syn = []
    train_syn.extend([(p, 0) for p in data_paths['train']['NORMAL']]) 
    train_syn.extend([(p, 1) for p in synthetic_files])              
    
    acc_syn = train_and_evaluate("Synthetic Data (Real Normal + Syn Pneumonia)", train_syn, test_set, device)
    
    # --- Experiment 3: Mixture ---
    train_mix = train_real + [(p, 1) for p in synthetic_files]
    
    acc_mix = train_and_evaluate("Mixture (Real + Syn)", train_mix, test_set, device)
    
    print("\n=== Final Results Summary ===")
    print(f"Real Data Only:       {acc_real:.2f}%")
    print(f"Synthetic Data Only:  {acc_syn:.2f}%")
    print(f"Mixture Data:         {acc_mix:.2f}%")

if __name__ == "__main__":
    main()
