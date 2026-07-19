# CIFAR-10 Image Classifier

## PyTorch Implementation | Tech Prime Internship

## Overview

Convolutional Neural Network for CIFAR-10 classification using PyTorch. Achieved **76.58% validation accuracy** through 3-layer CNN architecture with batch normalization and dropout.

**Kaggle Notebook:** [Link](https://www.kaggle.com/code/sanaullah03041417973/image-classifier-cifar-10)


### Week 1 Presentation Preview

<p align="center">
  <a href="https://htmlpreview.github.io/?https://github.com/SANAULLAH-AI/Tech-Prime-Intern/blob/main/Week1/showcase.html">
    <img src="https://img.shields.io/badge/Click_Here-Preview_Presentation-0a0a1a?style=for-the-badge&logo=github" alt="Preview Presentation">
  </a>
</p>


---

## Model Architecture

```
Input (3×32×32)
    ↓
Conv2d(3→32) → BN → ReLU → MaxPool(2×2)
    ↓
Conv2d(32→64) → BN → ReLU → MaxPool(2×2)
    ↓
Conv2d(64→128) → BN → ReLU → MaxPool(2×2)
    ↓
Flatten → Linear(2048→256) → ReLU → Dropout(0.5)
    ↓
Linear(256→10)
```

| Parameter | Value |
|-----------|-------|
| Total Parameters | 620,810 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Epochs | 10 |
| Device | CUDA GPU |

---

## Results

### Training Progress

| Epoch | Train Acc | Val Acc | Best |
|-------|-----------|---------|------|
| 1 | 44.48% | 57.38% | ✓ |
| 2 | 57.47% | 67.16% | ✓ |
| 3 | 62.35% | 69.77% | ✓ |
| 4 | 64.94% | 71.75% | ✓ |
| 5 | 67.61% | 74.40% | ✓ |
| 6 | 69.07% | 75.29% | ✓ |
| 7 | 70.38% | 75.92% | ✓ |
| 8 | 71.67% | 75.15% | |
| 9 | 72.58% | 76.58% | ✓ |
| 10 | 73.49% | 76.51% | |

**Best Validation Accuracy:** 76.58%

---

### Per-Class Performance

| Class | Accuracy |
|-------|----------|
| Ship | 91.10% |
| Truck | 90.80% |
| Frog | 90.10% |
| Automobile | 84.20% |
| Horse | 78.40% |
| Airplane | 73.50% |
| Deer | 68.50% |
| Cat | 67.70% |
| Bird | 66.50% |
| Dog | 55.00% |

---

### Observations

- **Strong Performance:** Ship (91.1%), Truck (90.8%), Frog (90.1%)
- **Weak Performance:** Dog (55.0%), Bird (66.5%), Cat (67.7%)
- **Training Time:** 2.23 minutes on GPU
- **Gap:** Validation accuracy ~3% higher than training, indicating good generalization

---

## Installation

```bash
pip install torch torchvision matplotlib numpy
```

---

## Output Files & Visualizations

| File | Preview / Link |
|------|----------------|
| `cifar10_best_model.pth` | [⬇ Download model](https://github.com/SANAULLAH-AI/Tech-Prime-Intern/blob/main/Week1/Project/image-classifier-cifar-10/cifar10_best_model.pth) (620 KB) |
| `training_results.png` | ![Training Curves](https://raw.githubusercontent.com/SANAULLAH-AI/Tech-Prime-Intern/main/Week1/Project/image-classifier-cifar-10/training_results.png) |
| `predictions.png` | ![Sample Predictions](https://raw.githubusercontent.com/SANAULLAH-AI/Tech-Prime-Intern/main/Week1/Project/image-classifier-cifar-10/predictions.png) |

> *The model file can be downloaded and loaded with `torch.load('cifar10_best_model.pth')`.*

---

### Output Files

| File | Description |
|------|-------------|
| `cifar10_best_model.pth` | Trained model weights |
| `training_results.png` | Training/validation curves |
| `predictions.png` | Sample predictions |


---

## Code Structure

| Function | Purpose |
|----------|---------|
| `Config` | Hyperparameters |
| `get_data()` | CIFAR-10 loading with transforms |
| `CIFAR10CNN` | Model architecture |
| `train_epoch()` | Single epoch training |
| `validate()` | Model evaluation |
| `train_full()` | Complete training pipeline |
| `evaluate_model()` | Per-class accuracy |
| `plot_results()` | Training curves |
| `show_predictions()` | Sample predictions |

---

## Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
numpy>=1.24.0
```

---


**Sanaullah**  
[GitHub](https://github.com/sanaullah-ai) | [HuggingFace](https://huggingface.co/sanaullah7964)

---
