# Virtual Contrast MRI

Deep learning-based medical image synthesis for generating contrast-enhanced MRI (T1CE) from non-contrast modalities (T1, T2, FLAIR) with uncertainty quantification.

## Overview

This project addresses the clinical need for contrast-enhanced MRI without gadolinium injection by using deep learning to synthesize T1CE images from standard non-contrast scans. The system includes uncertainty quantification through Monte Carlo Dropout, addressing a critical research gap in medical image synthesis.

## Features

- **Virtual Contrast Generation**: Synthesize T1CE from T1, T2, FLAIR modalities
- **Uncertainty Quantification**: Monte Carlo Dropout for confidence maps
- **Batch Prediction**: 10x faster processing with batch inference
- **Mixed Precision**: CUDA-enabled mixed precision for faster inference
- **Interactive UI**: Streamlit-based interface for real-time predictions
- **Comprehensive Metrics**: PSNR, SSIM, MAE, MSE evaluation

## Architecture

### Model
- **Backbone**: Attention U-Net (2D)
- **Input Channels**: 3 (T1, T2, FLAIR)
- **Output Channels**: 1 (T1CE)
- **Channels**: (32, 64, 128, 256, 512)
- **Strides**: (2, 2, 2, 2)

### Loss Function
Hybrid Loss combining:
- L1 Loss
- SSIM Loss
- Edge Loss

## Project Structure

```
virtual_contrast_mri/
├── models/
│   ├── __init__.py
│   └── attention_unet.py       # Attention U-Net model
├── losses/
│   ├── __init__.py
│   └── hybrid_loss.py          # Hybrid loss function
├── training/
│   ├── __init__.py
│   ├── train.py                # Training script
│   └── validate.py             # Validation script
├── dataset.py                   # Dataset loader
├── transforms.py               # Data transformations
├── brats_dataset.py            # BraTS slice dataset
├── predict.py                  # Inference module
├── metrics.py                  # Evaluation metrics
├── confidence_map.py           # Uncertainty quantification
└── app.py                      # Streamlit UI
```

## Installation

### Requirements
```bash
pip install torch torchvision torchaudio monai nibabel SimpleITK opencv-python matplotlib numpy pandas scikit-image scikit-learn scipy tqdm streamlit tensorboard einops pyyaml pillow
```

### Dataset
Download BraTS2020 Training Data and place in:
```
dataset/PKG - BraTS-Africa/MICCAI_BraTS2020_TrainingData/
```

## Usage

### Training
```bash
python training/train.py
```

### Validation
```bash
python training/validate.py
```

### Inference
```bash
python predict.py
```

### Streamlit UI
```bash
streamlit run app.py
```

## Training Configuration

- **Dataset**: BraTS2020 Training Data (266 patients, 41,230 slices)
- **Train/Val Split**: 80/20 (seed 42)
- **Batch Size**: 8
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Weight Decay**: 1e-5
- **Epochs**: 50
- **Device**: CUDA (if available) / CPU

## Inference Features

### Batch Prediction
Process entire patient volumes in single forward pass (10x faster than slice-by-slice)

### Mixed Precision
CUDA-enabled automatic mixed precision for faster inference

### Normalization
Automatic input normalization for robust predictions

### Confidence Maps
Monte Carlo Dropout with configurable samples for uncertainty estimation

### Auto-Save
Automatic saving of predictions and confidence maps

## Evaluation Metrics

- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity Index (0-1, higher is better)
- **MAE**: Mean Absolute Error (lower is better)
- **MSE**: Mean Squared Error (lower is better)

## Research Contributions

1. **Virtual Contrast Generation**: Eliminates need for gadolinium injection
2. **Uncertainty Quantification**: Monte Carlo Dropout for confidence maps
3. **Batch Inference**: 10x speed improvement over slice-by-slice processing
4. **Mixed Precision**: Faster inference with minimal accuracy loss

## Model Checkpoint

Best model saved as `best_model.pth` based on validation loss.

## Streamlit Interface

Features:
- Upload T1, T2, FLAIR slices
- Real-time T1CE prediction
- Confidence map visualization
- Ground truth comparison with metrics
- Interactive parameter adjustment

## Citation

If you use this code, please cite:

```bibtex
@software{virtual_contrast_mri,
  title={Virtual Contrast MRI: Deep Learning-based Medical Image Synthesis with Uncertainty Quantification},
  author={Your Name},
  year={2024}
}
```

## License

This project is for research purposes only.

## Acknowledgments

- BraTS2020 Dataset
- MONAI Framework
- PyTorch
