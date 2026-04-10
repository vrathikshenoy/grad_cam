# Brain Tumor Detection with Grad-CAM & Grad-CAM++ Visualization

A deep learning project implementing Grad-CAM and Grad-CAM++ for explainable AI in medical imaging, specifically for detecting brain tumors and visualizing which regions of medical images the CNN focuses on when making predictions.

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

## 🎯 Project Overview

This project addresses the challenge of **interpretable AI in medical imaging** by implementing Grad-CAM and Grad-CAM++ to visualize which regions of brain tumor medical images influence the model's classification decisions.

### Key Objectives:
1. Train a DenseNet121 classifier for brain tumor detection with high accuracy
2. Implement Grad-CAM to visualize which image regions the CNN uses for predictions
3. Implement Grad-CAM++ for improved spatial localization of tumor areas
4. Generate interpretable heatmaps that highlight regions of interest
5. Provide visual explanations for clinical decision support

## ✨ Key Features

- **Dual Visualization Methods**: Grad-CAM and Grad-CAM++ for different visualization styles
- **Pre-trained DenseNet121**: Efficient feature extraction with ImageNet weights
- **Brain Tumor Dataset**: Real medical imaging data from Kaggle
- **Transfer Learning**: Leverages pre-trained weights for better convergence
- **Data Augmentation**: Rotation, shifts, flips, and zoom for robust training
- **Comprehensive Metrics**: Tracks accuracy, precision, recall alongside training loss
- **Interactive Visualizations**: Heatmap overlays on original medical images

## 🔍 Problem Statement

Brain tumors require accurate and rapid diagnosis. However:
- Manual interpretation by radiologists is time-consuming and subject to inter-observer variability
- Deep learning models achieve high accuracy but lack interpretability (black-box problem)
- Clinicians need to understand which regions the model focuses on to trust AI predictions

**Solution**: Create an interpretable AI system that classifies brain tumors while visualizing which image regions drive the model's decisions through gradient-based activation mapping.

## 📊 Methodology

### 1. **Classification Model**
   - **Base**: DenseNet121 (pre-trained on ImageNet)
   - **Custom Head**: 
     - Flatten layer
     - Dropout (0.7)
     - BatchNormalization
     - Dense layer (16 units, ReLU)
     - Dropout (0.5)
     - BatchNormalization
     - Dense layer (2 units, Softmax) → Binary classification output
   - **Optimizer**: Adam (lr=0.0001)
   - **Loss**: Binary Crossentropy
   - **Training epochs**: 25

### 2. **Grad-CAM Implementation**
   - Computes gradients of class score w.r.t. feature maps
   - Weights feature maps by importance (mean gradient magnitude)
   - Generates heatmap highlighting regions influencing prediction
   - Resizes heatmap to match input image dimensions (224×224)

### 3. **Grad-CAM++ Implementation**
   - Computes higher-order gradients (first, second, third derivatives)
   - Applies spatial consistency weighting
   - Provides better localization for multiple activation regions
   - More robust to multiple objects/lesions in images

## 🏗️ Architecture

### Model Pipeline:
```
Input Image (224×224×3)
        ↓
   DenseNet121 (Pre-trained on ImageNet)
        ↓
   Flatten Layer
        ↓
   Dropout (0.7)
        ↓
   BatchNormalization
        ↓
   Dense (16 units, ReLU)
        ↓
   Dropout (0.5)
        ↓
   BatchNormalization
        ↓
   Dense (2 units, Softmax)
        ↓
Output: [No Tumor, Tumor]
```

### Grad-CAM Process:
```
Forward Pass → Convolutional Feature Maps
        ↓
Backward Pass → Compute Gradients
        ↓
Weight Computation (mean absolute gradient)
        ↓
Weighted Feature Combination
        ↓
ReLU (retain positive activations only)
        ↓
Normalization & Upsampling to (224×224)
        ↓
Heatmap Visualization
```

### Grad-CAM++ Process:
```
Forward Pass → Convolutional Feature Maps
        ↓
Higher-Order Gradients (1st, 2nd, 3rd order)
        ↓
Spatial Consistency Weighting
        ↓
Normalized Weighted Feature Combination
        ↓
Improved Spatial Localization Heatmap
```

## 📁 Dataset

### Source:
- **Kaggle Dataset**: Brain Tumor Dataset
- **Dataset Handle**: `preetviradiya/brian-tumor-dataset`
- **Downloaded via**: Kaggle API (`kagglehub`)

### Dataset Structure:
```
Brain Tumor Data Set/
├── Brain Tumor/     (Medical images with brain tumors)
└── No Brain Tumor/  (Control medical images without tumors)
```

### Data Properties:
- **Input Size**: 224×224×3 (RGB images)
- **Classes**: Binary classification (Tumor vs. No Tumor)
- **Preprocessing**: Images normalized to [0, 1] range
- **Resizing**: All images resized to 224×224

### Data Augmentation (Training):
- **Rescaling**: Normalize pixel values to [0, 1]
- **Rotation**: Random rotations up to 0.2 radians
- **Width/Height Shifts**: ±5% random translation
- **Horizontal & Vertical Flips**: Random mirroring
- **Zoom**: Random zoom up to 0.2
- **Validation Split**: 80% train / 20% validation
- **Batch Size**: 32

## 🚀 Installation

### Prerequisites:
- Python 3.8+
- GPU (CUDA) recommended for faster training (optional)
- 6GB+ RAM

### Setup:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd grad_cam
   ```

2. **Create virtual environment** (optional):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install tensorflow keras numpy pandas opencv-python matplotlib pillow kagglehub scikit-learn
   ```

4. **Setup Kaggle API** (for automatic dataset download):
   - kagglehub

## 💻 Usage

### Running the Main Notebook:

Open the Jupyter notebook with the dataset and training code:

```bash
jupyter notebook visual_explanations_gradcam_gradcam.ipynb
```

The notebook will automatically:
1. Download the Brain Tumor Dataset via Kaggle API
2. Preprocess and augment the medical images
3. Train a DenseNet121 model for brain tumor classification
4. Generate Grad-CAM heatmaps for predictions
5. Generate Grad-CAM++ heatmaps for improved visualizations
6. Display side-by-side comparisons of original images with heatmaps

### Step-by-Step Workflow:

1. **Data Preparation**: Download and augment brain tumor images
2. **Model Training**: Train DenseNet121 on augmented dataset for 25 epochs
3. **Model Evaluation**: Test on validation set, display metrics
4. **Grad-CAM Visualization**: Generate heatmaps showing regions of interest
5. **Grad-CAM++ Visualization**: Generate improved heatmaps with better localization
6. **Comparison Plots**: Display 40 sample images with both visualization methods

### Key Functions Used:

```python
# Load and preprocess images
readtumorImages(image_paths)  # Reads, normalizes, and resizes images

# Generate Grad-CAM heatmap
gradCam(image, true_label, layer_conv_name)  # Returns heatmap and image

# Generate Grad-CAM++ heatmap  
grad_cam_plus_plus(image, true_label, conv_layer)  # Returns improved heatmap

# Visualize results
draw_compare(images, gradcam_maps, gradcam_plus_maps, labels)  # Side-by-side display
```

## 📈 Results

### Expected Performance:
- **Accuracy**: ~85-95% binary classification accuracy
- **Precision**: High precision for tumor detection (minimizes false positives)
- **Recall**: Good recall for sensitivity (minimizes false negatives)
- **Training Time**: ~25 epochs, typically 10-30 minutes (GPU) or 1-2 hours (CPU)

### Model Weights Initialization:
The model uses pre-trained ImageNet weights for all DenseNet121 layers (not frozen), providing:
- ✅ Good weight initialization (better than random initialization)
- ✅ Faster convergence during training
- ✅ Better feature extraction for medical images
- ✅ Improved generalization with limited data

### Grad-CAM Insights:
- **Tumor Images**: Strong heat map activations concentrated on tumor regions
- **No Tumor Images**: Activations distributed across normal brain tissue
- **Localization**: Heatmaps identify specific areas the CNN uses for decisions
- **Clinical Relevance**: Visualizations help radiologists understand model predictions

### Grad-CAM vs Grad-CAM++:
- **Grad-CAM**: Faster, general-purpose visualization, works well for single objects
- **Grad-CAM++**: Better spatial consistency, improved for multiple activation regions, superior localization
- **Usage**: Both methods complement each other for thorough interpretation

## 📂 Project Structure

```
grad_cam/
├── README.md                               # This file
├── visual_explanations_gradcam_gradcam.ipynb  # Main notebook with complete pipeline
├── models/
│   ├── final_tumor_model.h5               # Trained DenseNet121 brain tumor model
│   └── final_pneumonia_model.h5           # Pneumonia model (separate project)
└── .venv/                                  # Virtual environment (optional)
```

## 🔧 Configuration

Key parameters in the notebook:

```python
INPUT_SHAPE = (224, 224, 3)        # Model input dimensions
BATCH_SIZE = 32                    # Training batch size
TRAINING_EPOCHS = 25               # Number of training epochs
DROPOUT_RATE_1 = 0.7              # First dropout layer
DROPOUT_RATE_2 = 0.5              # Second dropout layer
LEARNING_RATE = 0.0001            # Adam optimizer learning rate
DENSE_UNITS = 16                  # Hidden dense layer units
```

Modify these values in the notebook to experiment with different configurations.

## 🎓 Educational Value

This project demonstrates:
1. **Transfer Learning**: Using pre-trained DenseNet121 for medical imaging
2. **Explainable AI**: Grad-CAM and Grad-CAM++ for model interpretability
3. **Gradient-based Methods**: Computing gradients w.r.t. activation maps
4. **Data Augmentation**: Comprehensive augmentation strategies for robust training
5. **Medical AI**: Application in healthcare domain for brain tumor detection
6. **Visualization Techniques**: Converting neural network activations into human-interpretable heatmaps

## 📚 Key Concepts Explained

### Grad-CAM:
Gradient-weighted Class Activation Mapping combines class gradients with activation maps:
- Computes gradients of class score w.r.t. convolutional feature maps
- Weights feature maps by mean absolute gradient (importance)
- Uses positive activations (ReLU) to highlight regions
- Produces saliency heatmaps highlighting important regions

### Grad-CAM++:
An improved version of Grad-CAM with better spatial localization:
- Computes higher-order gradients (1st, 2nd, 3rd derivatives)
- Applies spatial consistency weighting
- Better handles multiple activation regions
- More robust for complex medical images with multiple lesions

### DenseNet121:
- Dense Connections: Each layer connects to all previous layers
- 121 layers total with efficient gradient flow
- Pre-trained on ImageNet (1.2M images, 1000 classes)
- Excellent for transfer learning with limited medical data

## 🔮 Future Enhancements

1. **Model Architecture**:
   - Experiment with ResNet50/101 alternatives
   - Implement ensemble methods combining multiple models
   - Add attention mechanisms for attention-based explanations

2. **Visualization Extensions**:
   - Guided Grad-CAM for cleaner visualizations
   - Score-CAM for gradient-free explanations
   - Layer-wise Relevance Propagation (LRP)
   - Integrated Gradients for attribution

3. **Clinical Integration**:
   - Web interface for real-time predictions
   - DICOM file support for medical imaging format
   - Confidence calibration for clinical decision support
   - Multi-class tumor type classification

4. **Evaluation Metrics**:
   - 5-fold cross-validation for robust metrics
   - ROC curves and AUC analysis
   - Sensitivity/Specificity optimization
   - Comparison with radiologist interpretations

## 📖 References

1. **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
2. **Grad-CAM++**: Chattopadhyay et al., "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks" (WACV 2018)
3. **DenseNet**: Huang et al., "Densely Connected Convolutional Networks" (CVPR 2017)
4. **Dataset**: Kaggle - Brain Tumor Dataset by Preet Viradiya
5. **TensorFlow/Keras Documentation**: https://www.tensorflow.org/

## ⚠️ Clinical Disclaimer

This project is for **educational and research purposes only**. The models and visualizations should not be used for clinical diagnosis without proper validation, regulatory approval (FDA, CE marking, etc.), and clinical trials. Always consult qualified medical professionals and radiologists for diagnosis and treatment decisions.

## 👤 Author

Vrathik Shenoy K

## 📄 License

This project is open source and available for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Project Focus**: Brain Tumor Detection with Visual Explanations  
**Last Updated**: April 2026  
**Status**: Complete Implementation
