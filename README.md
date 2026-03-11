# Lung X-Ray Disease Detection with ConvNeXt

AI-powered multi-label classification and bounding box detection for 14 lung diseases using NIH ChestX-ray14 dataset.

**Best Result: Mean AUC-ROC = 0.8469** (competitive with CheXNet)

## Overview

This project trains a deep learning model to detect 14 thoracic diseases from chest X-ray images. It uses a ConvNeXt-Base backbone pretrained on ImageNet-22k, enhanced with patient metadata fusion (age, gender, view position, follow-up number) and bounding box detection for disease localization.

### Diseases Detected

| Disease | AUC | Disease | AUC |
|---------|------|---------|------|
| Emphysema | 0.939 | Consolidation | 0.825 |
| Hernia | 0.933 | Pleural Thickening | 0.817 |
| Cardiomegaly | 0.919 | Atelectasis | 0.814 |
| Edema | 0.895 | Nodule | 0.806 |
| Pneumothorax | 0.889 | Pneumonia | 0.759 |
| Effusion | 0.879 | Infiltration | 0.701 |
| Mass | 0.853 | | |
| Fibrosis | 0.828 | **Mean AUC** | **0.847** |

## Architecture

- **Backbone**: ConvNeXt-Base (`convnext_base.fb_in22k_ft_in1k_384`) via timm
- **Classification Head**: 14-class multi-label sigmoid output
- **Detection Head**: Per-class bounding box regression with spatial attention
- **Metadata Fusion**: MLP encoder for patient age, gender, view position, follow-up
- **Parameters**: 89.1M total

## Training Strategy

### Two-Phase Training
1. **Phase 1 (3 epochs)**: Backbone frozen, train heads only (1.5M params) with higher LR
2. **Phase 2 (remaining epochs)**: Backbone unfrozen with differential LR (backbone 10x lower than heads) and linear warmup

### Key Techniques
- **EMA (Exponential Moving Average)**: Smoothed model weights for stable validation
- **Focal Loss**: Handles severe class imbalance (e.g., Hernia: 0.2% prevalence)
- **Mixed Precision (AMP)**: FP16 training with gradient scaling for RTX 3090
- **Gradient Accumulation**: Effective batch size 32 (16 x 2 accumulation steps)
- **ReduceLROnPlateau**: Adapts learning rate when improvement stalls
- **Early Stopping**: Patience of 15 epochs on validation AUC
- **Patient-Level Split**: No data leakage between train/val sets

### Grayscale Handling
- Load images in grayscale mode ("L")
- Histogram equalization for scanner standardization
- Replicate to 3 channels for pretrained backbone compatibility
- Grayscale-specific normalization (mean=0.5024, std=0.2898)

## Dataset

- **NIH ChestX-ray14**: 112,120 frontal chest X-rays from 30,805 patients
- **Labels**: NLP-extracted from radiology reports (14 diseases + "No Finding")
- **Bounding Boxes**: 984 annotations across 8 disease types
- **Train/Val Split**: 89,826 / 22,294 images (patient-level, no leakage)

### Required Data Files
Place these in the `data/` directory:
- `Data_Entry_2017_v2020.csv` - Image metadata and disease labels
- `BBox_List_2017.csv` - Bounding box annotations
- `images/` - X-ray image files (PNG format)

Data available from [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC).

## Setup

### Prerequisites
- Python 3.11+
- NVIDIA GPU with 24GB+ VRAM (tested on RTX 3090)
- CUDA 12.x

### Installation

```bash
# Create conda environment
conda create -n cuda12-py311 python=3.11
conda activate cuda12-py311

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Default training (auto-resumes from latest checkpoint)
python main.py

# Custom parameters
python main.py --batch_size 16 --lr 1e-4 --epochs 50 --image_size 512

# Resume from specific checkpoint
python main.py --resume outputs/checkpoints/epoch_010.pth
```

### Configuration

Key parameters in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_size` | 512 | Input image resolution |
| `batch_size` | 16 | Per-GPU batch size |
| `grad_accum_steps` | 2 | Gradient accumulation (effective batch = 32) |
| `lr` | 1e-4 | Base learning rate (backbone) |
| `head_lr_factor` | 10.0 | Head LR multiplier vs backbone |
| `epochs` | 50 | Maximum training epochs |
| `freeze_backbone_epochs` | 3 | Phase 1 duration |
| `patience` | 15 | Early stopping patience |
| `ema_decay` | 0.999 | EMA decay rate |

## Project Structure

```
aimedimages/
├── main.py                  # Entry point with CLI arguments
├── requirements.txt         # Python dependencies
├── src/
│   ├── config.py            # Configuration dataclass
│   ├── dataset.py           # Data loading, transforms, DataLoader
│   ├── model.py             # LungDiseaseNet (ConvNeXt + metadata + bbox)
│   ├── trainer.py           # Training loop, checkpointing, EMA
│   └── utils.py             # Losses, metrics, logging, seeding
├── data/                    # CSV files and images (not in repo)
└── outputs/                 # Checkpoints, logs, TensorBoard (not in repo)
    ├── checkpoints/
    ├── logs/
    └── tensorboard/
```

## Training Results

Training converged at epoch 13 with mean AUC = 0.8469:

```
Epoch  Phase     Val Loss  Mean AUC  Status
-----  --------  --------  --------  ------
  0    Phase 1    0.3876    0.7110   New best
  1    Phase 1    0.3772    0.7137   New best
  2    Phase 1    0.3721    0.7193   New best
  3    Phase 2    0.3451    0.7701   New best (+0.051)
  4    Phase 2    0.3267    0.7998   New best
  5    Phase 2    0.3101    0.8185   New best
  6    Phase 2    0.2936    0.8311   New best
  7    Phase 2    0.2881    0.8366   New best
  8    Phase 2    0.2912    0.8405   New best
  9    Phase 2    0.2896    0.8435   New best
 10    Phase 2    0.2909    0.8451   New best
 11    Phase 2    0.2936    0.8456   New best
 12    Phase 2    0.2961    0.8464   New best
 13    Phase 2    0.3007    0.8469   Best model saved
 14+   Phase 2    rising    declining Early stopping at epoch 28
```

## Hardware

Developed and tested on:
- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **CPU**: Intel Xeon E5-2690 v4 (HP Z840)
- **Training time**: ~15 hours (29 epochs x ~31 min/epoch)

## License

This project is for research and educational purposes.
