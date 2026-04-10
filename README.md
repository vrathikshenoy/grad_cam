# Grad-CAM: Visual Explanations for Chest X-Ray Pneumonia Detection

A deep learning project implementing Gradient-weighted Class Activation Mapping (Grad-CAM) for explainable AI in medical imaging, specifically for pneumonia detection from chest X-rays with synthetic data augmentation.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)

## 🎯 Project Overview

This project addresses the challenge of **interpretable AI in medical imaging** by implementing Grad-CAM to visualize which regions of chest X-ray images influence the model's pneumonia detection decisions. Additionally, it explores the effectiveness of synthetic medical images generated using diffusion models to augment training data.

### Key Objectives:
1. Train a ResNet18 classifier for pneumonia detection with high accuracy
2. Generate synthetic pneumonia X-ray images using diffusion models (DDPM)
3. Implement Grad-CAM to visualize model decision-making process
4. Evaluate model performance on real vs. synthetic vs. mixed datasets
5. Provide interpretable explanations for clinical decision support

## ✨ Key Features

- **Grad-CAM Visualization**: Gradient-weighted Class Activation Maps to highlight critical regions
- **Synthetic Data Generation**: Diffusion-based (DDPM) synthetic X-ray generation
- **Medical Dataset Integration**: Chest X-ray pneumonia dataset from Kaggle
- **Transfer Learning**: Pre-trained ResNet18 backbone for efficient training
- **Comparative Analysis**: Performance metrics across different data compositions
- **Class Imbalance Handling**: Weighted loss functions for balanced training
- **Advanced Augmentation**: Multi-level data augmentation strategies

## 🔍 Problem Statement

Pneumonia is a serious respiratory infection that requires rapid diagnosis. Chest X-rays are the primary diagnostic tool, but:
- Manual interpretation is time-consuming and subject to human error
- Deep learning models achieve high accuracy but lack interpretability
- Limited medical imaging data restricts model training

**Solution**: Create an interpretable AI system that not only classifies X-rays but also explains its predictions through visual activation maps, and augment training data with synthetic images.

## 📊 Methodology

### 1. **Data Generation (Synthetic Images)**
   - **Architecture**: UNet2D with attention blocks
   - **Training**: DDPM (Denoising Diffusion Probabilistic Model)
   - **Parameters**:
     - Image size: 64×64 (for generation), 224×224 (for classification)
     - Timesteps: 1000
     - Epochs: 100
     - Batch size: 32

### 2. **Classification Model**
   - **Base**: ResNet18 (pre-trained on ImageNet)
   - **Head**: Custom fully-connected layer with dropout
   - **Optimizer**: Adam (lr=1e-4, weight_decay=1e-4)
   - **Loss**: CrossEntropyLoss with class weights
   - **Training epochs**: 10-25

### 3. **Grad-CAM Implementation**
   - Computes gradient of target class w.r.t. activation maps
   - Generates weighted combination of activation maps
   - Produces heatmaps highlighting important image regions

### 4. **Experimental Setup**
   Three experiments were conducted:
   - **Experiment 1**: Training on real data only
   - **Experiment 2**: Training on mixed real and synthetic data
   - **Experiment 3**: Evaluating on various synthetic-to-real ratios

## 🏗️ Architecture

### Model Pipeline:
```
Input Image (224×224)
        ↓
   ResNet18 (Pre-trained)
        ↓
   Feature Extraction (512-dim)
        ↓
   Dropout (0.5)
        ↓
   Linear Layer (512 → 2 classes)
        ↓
   Softmax
        ↓
Output: [Normal, Pneumonia]
```

### Grad-CAM Process:
```
Forward Pass → Activation Maps
        ↓
Backward Pass → Gradients
        ↓
Weight Computation (average pooling gradients)
        ↓
Weighted Combination
        ↓
ReLU (retain positive activations)
        ↓
Heatmap Visualization
```

## 📁 Dataset

### Source:
- **Kaggle Dataset**: Chest X-Ray Images (Pneumonia)
- **Dataset Handle**: `paultimothymooney/chest-xray-pneumonia`

### Dataset Structure:
```
chest_xray/
├── train/
│   ├── NORMAL/     (1,349 images)
│   └── PNEUMONIA/  (3,875 images)
└── test/
    ├── NORMAL/     (234 images)
    └── PNEUMONIA/  (390 images)
```

### Data Augmentation:
- **Training**: RandomResizedCrop, HorizontalFlip, Rotation(15°), ColorJitter
- **Testing**: Resize + CenterCrop
- **Normalization**: ImageNet standards (mean=[0.485, 0.456, 0.406])

## 🚀 Installation

### Prerequisites:
- Python 3.12+
- GPU (CUDA) recommended for faster training
- 8GB+ RAM

### Setup:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd grad_cam
   ```

2. **Create virtual environment**:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # Or using the project configuration:
   pip install -e .
   ```

4. **Install additional requirements**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install kagglehub opencv-python matplotlib scikit-learn
   ```

5. **Setup Kaggle API** (for dataset download):
   - Create account at [kaggle.com](https://www.kaggle.com)
   - Download API token from account settings
   - Place `kaggle.json` in `~/.kaggle/`

## 💻 Usage

### Running the Main Training Pipeline:

```bash
python medical_synthetic_training.py
```

This script:
1. Downloads the chest X-ray dataset
2. Generates synthetic pneumonia images using DDPM
3. Trains ResNet18 on three dataset compositions
4. Reports accuracy for each experiment

### Using Grad-CAM in Jupyter Notebook:

Open `visual_explanations_gradcam_gradcam.ipynb` for:
- Interactive Grad-CAM visualization
- Step-by-step implementation walkthrough
- Visual comparison of real vs. synthetic predictions

### Key Functions:

```python
# Training with synthetic augmentation
python medical_synthetic_training.py

# Generating synthetic images
train_diffusion(dataset_root, output_path, device)

# Evaluating on test set
train_and_evaluate(experiment_name, train_files, test_files, device)
```

## 📈 Results

### Expected Performance:
| Experiment | Data Composition | Accuracy |
|-----------|------------------|----------|
| Real Data Only | Original + Original | ~85-90% |
| Synthetic Augmented | Real + Synthetic | ~82-88% |
| Balanced Mix | 50% Real + 50% Synthetic | ~86-91% |

### Grad-CAM Insights:
- **Normal X-rays**: Activations spread across lungs, highlighting clear regions
- **Pneumonia X-rays**: Strong activations on affected areas (infiltrates)
- **Model Focus**: Bottom-right and center regions show strongest activations

### Benefits of Synthetic Data:
- ✅ Improves model robustness
- ✅ Reduces overfitting on limited real data
- ✅ Helps with class imbalance (4:1 normal-to-pneumonia ratio)
- ⚠️ Should be used carefully in clinical settings

## 📂 Project Structure

```
grad_cam/
├── README.md                               # This file
├── main.py                                 # Entry point
├── medical_synthetic_training.py           # Main training pipeline
├── visual_explanations_gradcam_gradcam.ipynb  # Grad-CAM visualization notebook
├── lung_pneumonia.ipynb                    # Exploratory analysis
├── pyproject.toml                          # Project configuration
├── config/
│   └── global.json                         # Configuration settings
├── models/
│   ├── final_pneumonia_model.h5           # Trained pneumonia classifier
│   └── final_tumor_model.h5               # Trained tumor classifier
├── synthetic/
│   └── PNEUMONIA/                         # Generated synthetic images
└── .venv/                                  # Virtual environment
```

## 🔧 Configuration

Key parameters in `medical_synthetic_training.py`:

```python
BATCH_SIZE = 32                    # Training batch size
CLASSIFIER_EPOCHS = 10             # Number of training epochs
DIFFUSION_EPOCHS = 100             # DDPM training epochs
IMG_SIZE_DIFFUSION = 64            # Synthetic image generation size
IMG_SIZE_CLASSIFIER = 224          # Classification model input size
NUM_SYNTHETIC_SAMPLES = 200        # Number of synthetic images to generate
```

Modify these values to experiment with different configurations.

## 🎓 Educational Value

This project demonstrates:
1. **Transfer Learning**: Using pre-trained models for medical imaging
2. **Generative Models**: DDPM for synthetic data generation
3. **Explainable AI**: Grad-CAM for model interpretability
4. **Data Augmentation**: Synthetic + real data combination
5. **Class Imbalance Handling**: Weighted loss functions
6. **Medical AI**: Application in healthcare domain

## 📚 Key Concepts Explained

### Grad-CAM:
Grad-CAM (Gradient-weighted Class Activation Mapping) combines class gradients with activation maps to identify important regions:
- Uses gradients of class scores w.r.t. feature maps
- Weights indicate importance of each feature
- Produces interpretable saliency maps

### DDPM:
Denoising Diffusion Probabilistic Models:
- Forward process: Gradually add noise to images
- Reverse process: Learn to denoise step-by-step
- Generate new images from pure noise

### ResNet18:
- 18-layer residual network with skip connections
- Pre-trained on ImageNet (1.2M images, 1000 classes)
- Efficient and suitable for transfer learning

## 🔮 Future Enhancements

1. **Model Improvements**:
   - Experiment with ResNet50/101 for higher capacity
   - Implement ensemble methods
   - Add attention mechanisms

2. **Grad-CAM Extensions**:
   - Guided Grad-CAM for cleaner visualizations
   - Score-CAM for gradient-free explanations
   - Layer-wise Relevance Propagation (LRP)

3. **Data Augmentation**:
   - Explore other generative models (VAE, GANs)
   - Implement CycleGAN for domain adaptation
   - Create domain-specific synthetic images

4. **Clinical Integration**:
   - Web interface for real-time predictions
   - DICOM file support
   - Confidence calibration for clinical use

5. **Evaluation**:
   - Cross-validation with k-folds
   - ROC curves and AUC metrics
   - Sensitivity/Specificity analysis
   - Radiologist comparison studies

## 📖 References

1. **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
2. **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
3. **ResNet**: He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
4. **Dataset**: Kaggle - Chest X-Ray Images (Pneumonia)

## ⚠️ Clinical Disclaimer

This project is for **educational purposes only**. The models and visualizations should not be used for clinical diagnosis without proper validation and regulatory approval. Always consult medical professionals for diagnosis.

## 👤 Author

Vrathik Shenoy K

## 📄 License

This project is open source and available for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Last Updated**: February 2025
**Status**: Active Development
