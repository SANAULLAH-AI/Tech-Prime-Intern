
# Medical Image Segmentation System

## Tech Prime Pvt Limited - Advanced AI/ML Internship Program

---

## Overview

A deep learning system for medical image segmentation using U-Net architecture with ResNet34 encoder. The system processes CT scan images to detect and segment regions of interest, achieving a mean Dice score of 0.8373 and IoU of 0.7297.

---

## Dataset

**Source:** COVID-19 CT Scan Dataset (Kaggle)

**Statistics:**
- Total Images: 200
- Image Size: 256x256 (Grayscale)
- Foreground Ratio: 17.39%
- Class Imbalance Ratio: 4.75 (Moderate)

**Sample Visualization:**

![Dataset Samples](https://github.com/SANAULLAH-AI/Tech-Prime-Intern/blob/main/Week3/Project/dataset_samples.png)

---

## Mask Generation

Intensity-based masks were created from raw CT images using Otsu thresholding to identify regions of interest (ground glass opacities, consolidations).

![Proper Masks](https://github.com/SANAULLAH-AI/Tech-Prime-Intern/blob/main/Week3/Project/proper_masks.png)

**Method:**
1. Gaussian blur to reduce noise
2. Otsu thresholding for binary segmentation
3. Morphological operations (close + open)
4. Small region removal (area < 100 pixels)

---

## Model Architecture

**Architecture:** U-Net with ResNet34 Encoder

- Encoder: ResNet34 (ImageNet pretrained)
- Input Channels: 1 (Grayscale)
- Output Channels: 1 (Binary Mask)
- Total Parameters: 24,430,097
- Activation: Sigmoid

**Libraries Used:**
- PyTorch (Model Implementation)
- segmentation-models-pytorch (U-Net Architecture)
- albumentations (Data Augmentation)
- OpenCV (Image Processing)

---

## Training

**Configuration:**
- Batch Size: 8
- Epochs: 15
- Optimizer: Adam (lr=1e-4)
- Loss Function: BCE + Dice Loss
- Scheduler: ReduceLROnPlateau

**Data Augmentation:**
- Horizontal Flip (p=0.5)
- Vertical Flip (p=0.3)
- Rotation (15 degrees)
- Gaussian Noise

**Training History:**

![Training History](https://github.com/SANAULLAH-AI/Tech-Prime-Intern/blob/main/Week3/Project/training_history.png)

---

## Results

**Validation Set Performance:**

| Metric | Value |
|--------|-------|
| Best Validation Dice | 0.6909 |
| Mean Dice | 0.8373 |
| Std Dice | 0.0926 |
| Mean IoU | 0.7297 |
| Std IoU | 0.1211 |
| Min Dice | 0.4954 |
| Max Dice | 0.9544 |

**Segmentation Results:**

![Segmentation Results](https://github.com/SANAULLAH-AI/Tech-Prime-Intern/blob/main/Week3/Project/segmentation_results.png)

---

## Key Technical Solutions

1. **Data Quality:** Resized all images to 256x256, normalized intensities
2. **Mask Creation:** Generated masks using Otsu thresholding from raw CT intensities
3. **Class Imbalance:** Used Combined Loss (BCE + Dice) for foreground-background imbalance
4. **Overfitting Prevention:** Applied data augmentation and weight regularization

---

## Project Structure

```
Week3/Project/
├── dataset_audit_report.txt # Dataset analysis
├── dataset_samples.png      # Sample images visualization
├── proper_masks.png         # Mask generation results
├── segmentation_results.png # Final segmentation outputs
├── training_history.png     # Loss and Dice curves
└── README.md                # Project documentation
```

---

## Requirements

```
torch >= 2.0
segmentation-models-pytorch
albumentations
opencv-python
numpy
matplotlib
tqdm
scikit-learn
```
