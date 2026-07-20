# COMPLETE WEEK 2 NOTES - Computer Vision with CNNs (ULTIMATE NOOB-FRIENDLY GUIDE)

## Professional Training Documentation
**Tech Prime Pvt Limited - Advanced AI/ML Internship Program**

---

# 📑 TABLE OF CONTENTS

1. [Computer Vision Basics for Absolute Beginners](#1-computer-vision-basics-for-absolute-beginners)
2. [CNN Architecture - Every Layer Explained](#2-cnn-architecture-every-layer-explained)
3. [ResNet - Skip Connections Made Simple](#3-resnet-skip-connections-made-simple)
4. [Transfer Learning - Why and How](#4-transfer-learning-why-and-how)
5. [Data Augmentation - Creating More Data](#5-data-augmentation-creating-more-data)
6. [EfficientNet - Smart Scaling](#6-efficientnet-smart-scaling)
7. [Object Detection - Finding Objects in Images](#7-object-detection-finding-objects-in-images)
8. [Complete Working Code with Line-by-Line Explanation](#8-complete-working-code-with-line-by-line-explanation)
9. [Common Issues and Solutions](#9-common-issues-and-solutions)
10. [Quick Reference - All Code Patterns](#10-quick-reference-all-code-patterns)

---

# 1. COMPUTER VISION BASICS FOR ABSOLUTE BEGINNERS

## 1.1 What is Computer Vision?

**Imagine:** You're looking at a photo of a cat. Your brain:
1. Sees pixels (tiny colored dots)
2. Recognizes patterns (eyes, ears, fur)
3. Says "That's a cat!"

**Computer Vision** does the same thing but with math!

### How Computers "See" Images

```
Image as Computer Sees It:
┌─────────────────────┐
│ [255, 0, 0] [255, 0, 0] [255, 0, 0] ... │
│ [0, 255, 0] [0, 255, 0] [0, 255, 0] ... │  ← Each pixel is 3 numbers (RGB)
│ [0, 0, 255] [0, 0, 255] [0, 0, 255] ... │
└─────────────────────┘
```

**Key Concept:** Every image is just a grid of numbers!

### Image Representation

```python
import torch
import numpy as np
from PIL import Image

# ============ UNDERSTANDING IMAGE DATA ============

# A single pixel is 3 numbers: [R, G, B]
pixel = [255, 0, 0]  # RED pixel
print(f"RED pixel: {pixel}")

# A 2x2 image (4 pixels)
tiny_image = torch.tensor([
    [[255, 0, 0], [0, 255, 0]],  # Row 1: Red, Green
    [[0, 0, 255], [255, 255, 255]]  # Row 2: Blue, White
])
print(f"Image shape: {tiny_image.shape}")  # Shape: (height, width, channels)
print(f"Image shape: {tiny_image.permute(2, 0, 1).shape}")  # (channels, height, width)

# In PyTorch, images are stored as (C, H, W)
# C = Channels (3 for RGB)
# H = Height (number of rows)
# W = Width (number of columns)

# A real image:
# shape = (3, 224, 224) means:
# - 3 color channels (RGB)
# - 224 pixels tall
# - 224 pixels wide
# - Total numbers: 3 × 224 × 224 = 150,528 numbers!
```

## 1.2 What is a CNN (Convolutional Neural Network)?

**Analogy:** Think of CNN like detectives with magnifying glasses.

1. **Convolution** = Detective scanning with magnifying glass
2. **Pooling** = Detective stepping back to see bigger picture
3. **Fully Connected** = Detective making final decision

### The CNN Pipeline

```python
# Simplified CNN Structure
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # STEP 1: Feature Extraction (What do we see?)
        self.conv1 = nn.Conv2d(3, 32, 3)  # 3 → 32 channels
        self.conv2 = nn.Conv2d(32, 64, 3)  # 32 → 64 channels
        self.pool = nn.MaxPool2d(2, 2)     # Reduce size by half
        
        # STEP 2: Decision Making (What is it?)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Features → 128 neurons
        self.fc2 = nn.Linear(128, 10)          # 128 → 10 classes
    
    def forward(self, x):
        # Input: (batch, 3, 32, 32)
        x = self.conv1(x)  # (batch, 32, 30, 30)
        x = self.pool(x)   # (batch, 32, 15, 15)
        x = self.conv2(x)  # (batch, 64, 13, 13)
        x = self.pool(x)   # (batch, 64, 6, 6)
        
        # Flatten: (batch, 64*6*6) → (batch, 2304)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)  # (batch, 128)
        x = self.fc2(x)  # (batch, 10)
        return x
```

---

# 2. CNN ARCHITECTURE - EVERY LAYER EXPLAINED

## 2.1 Convolution Layer (nn.Conv2d) - THE MOST IMPORTANT LAYER!

### What Does Convolution Do?

**Analogy:** Imagine you have a small window (filter) that slides over your image, looking for specific patterns (edges, curves, textures).

```python
# ============ CONV2D EXPLAINED ============

conv = nn.Conv2d(
    in_channels=3,      # INPUT: How many colors? (RGB = 3)
    out_channels=32,    # OUTPUT: How many filters/patterns to find?
    kernel_size=3,      # WINDOW: Size of detector (3×3 pixels)
    stride=1,           # STEP: How many pixels to move each time
    padding=1           # BORDER: Add zeros around image
)

# Example with real numbers:
input_image = torch.randn(1, 3, 32, 32)  # 1 image, 3 channels, 32×32
output = conv(input_image)
print(f"Input shape: {input_image.shape}")   # (1, 3, 32, 32)
print(f"Output shape: {output.shape}")       # (1, 32, 32, 32)

# Explanation:
# 1 image → Still 1 image
# 3 channels (RGB) → 32 channels (features found)
# 32×32 image → Still 32×32 (because padding=1 keeps size same)
# If padding=0, shape would be 30×30
```

### Visual Understanding of Convolution

```
Original Image (5×5):        Filter (3×3):        Output (3×3):
┌─────────────┐              ┌─────┐              ┌─────────┐
│ 1 2 3 4 5   │              │ 1 0 1│              │ 1+0+3+... │
│ 6 7 8 9 10  │              │ 0 1 0│              │          │
│11 12 13 14 15│              │ 1 0 1│              │          │
│16 17 18 19 20│              └─────┘              └─────────┘
│21 22 23 24 25│
└─────────────┘

Step 1: Filter at top-left (1,2,3,6,7,8,11,12,13)
Step 2: Multiply each number with filter values
Step 3: Sum everything = Output pixel
Step 4: Slide filter to next position (stride=1)
Step 5: Repeat until all positions covered
```

### Important Parameters Explained

```python
# ============ KERNEL_SIZE (Filter Size) ============
conv_3x3 = nn.Conv2d(3, 32, kernel_size=3)  # 3×3 filter (most common)
conv_5x5 = nn.Conv2d(3, 32, kernel_size=5)  # 5×5 filter (bigger, more parameters)
conv_1x1 = nn.Conv2d(3, 32, kernel_size=1)  # 1×1 filter (changes channels only)

# ============ STRIDE (Step Size) ============
conv_stride_1 = nn.Conv2d(3, 32, 3, stride=1)  # Moves 1 pixel each time
conv_stride_2 = nn.Conv2d(3, 32, 3, stride=2)  # Moves 2 pixels each time (reduces size)

# ============ PADDING (Border Handling) ============
conv_pad_0 = nn.Conv2d(3, 32, 3, padding=0)   # No padding → Image shrinks
conv_pad_1 = nn.Conv2d(3, 32, 3, padding=1)   # Add 1 pixel border → Same size
conv_pad_2 = nn.Conv2d(3, 32, 3, padding=2)   # Add 2 pixel border → Image grows

# ============ DILATION (Spacing between filter values) ============
conv_normal = nn.Conv2d(3, 32, 3, dilation=1)    # Normal (no gaps)
conv_dilated = nn.Conv2d(3, 32, 3, dilation=2)   # Bigger receptive field

# ============ GROUPS (Depthwise Convolution) ============
conv_groups_1 = nn.Conv2d(3, 32, 3, groups=1)    # Normal
conv_groups_3 = nn.Conv2d(3, 32, 3, groups=3)    # Depthwise (each channel processed separately)
```

### Input and Output Size Calculation

**Formula:** 
```
Output_Size = (Input_Size - Kernel_Size + 2×Padding) / Stride + 1
```

```python
# ============ CALCULATING SHAPES ============

def calculate_conv_output(input_size, kernel_size, padding, stride):
    """Calculate output size after convolution"""
    output_size = (input_size - kernel_size + 2 * padding) // stride + 1
    return output_size

# Examples:
print(f"5×5 with 3×3 kernel, pad=0, stride=1: {calculate_conv_output(5, 3, 0, 1)}")  # 3
print(f"5×5 with 3×3 kernel, pad=1, stride=1: {calculate_conv_output(5, 3, 1, 1)}")  # 5
print(f"5×5 with 3×3 kernel, pad=0, stride=2: {calculate_conv_output(5, 3, 0, 2)}")  # 2
print(f"32×32 with 3×3 kernel, pad=1, stride=1: {calculate_conv_output(32, 3, 1, 1)}")  # 32

# ============ NUMBER OF PARAMETERS ============
def count_conv_params(in_channels, out_channels, kernel_size):
    """Count parameters in a conv layer"""
    # Each filter has: in_channels × kernel_size × kernel_size weights + 1 bias
    params = (in_channels * kernel_size * kernel_size * out_channels) + out_channels
    return params

print(f"Conv2d(3, 32, 3) parameters: {count_conv_params(3, 32, 3):,}")  # 896
print(f"Conv2d(32, 64, 3) parameters: {count_conv_params(32, 64, 3):,}")  # 18,496
print(f"Conv2d(64, 128, 3) parameters: {count_conv_params(64, 128, 3):,}")  # 73,856
```

## 2.2 Pooling Layer (MaxPool2d) - Downsampling

### What Does Pooling Do?

**Analogy:** You're looking at a 32×32 image, but you step back 2 steps. Now you see a 16×16 version. It preserves the important information but reduces size.

```python
# ============ MAXPOOL2D EXPLAINED ============

pool = nn.MaxPool2d(
    kernel_size=2,  # Window: 2×2
    stride=2       # Step: 2 pixels
)

# Example:
input_tensor = torch.randn(1, 32, 32, 32)  # 1 image, 32 channels, 32×32
output = pool(input_tensor)
print(f"Input: {input_tensor.shape}")   # (1, 32, 32, 32)
print(f"Output: {output.shape}")        # (1, 32, 16, 16) - Half the size!

# Different Pooling Types:
max_pool = nn.MaxPool2d(2, 2)    # Takes maximum value in window
avg_pool = nn.AvgPool2d(2, 2)    # Takes average of values in window
adaptive_pool = nn.AdaptiveAvgPool2d(1)  # Reduces to 1×1 regardless of input

# Visual Understanding:
"""
Input (4×4):                    Max Pool (2×2):
┌─────────────┐                 ┌─────────┐
│ 1 2 3 4     │                 │ max(1,2, │
│ 5 6 7 8     │                 │  5,6)=6  │ ...
│ 9 10 11 12  │                 │         │
│13 14 15 16  │                 └─────────┘
└─────────────┘
"""
```

### Why Use Pooling?

1. **Reduces Computation**: Less data to process
2. **Prevents Overfitting**: Reduces number of parameters
3. **Makes Features More Robust**: Small shifts don't matter
4. **Increase Receptive Field**: Each neuron sees more of the original image

## 2.3 Batch Normalization (BatchNorm2d) - Training Stabilizer

### What Does Batch Normalization Do?

**Analogy:** Like adjusting the volume of each channel so it's not too loud or too quiet. It ensures each layer gets inputs with consistent mean and variance.

```python
# ============ BATCH NORM EXPLAINED ============

bn = nn.BatchNorm2d(
    num_features=32,  # Number of channels from previous layer
    eps=1e-5,        # Small number to avoid division by zero
    momentum=0.1,    # For running mean/variance
    affine=True      # Learnable scale and shift
)

# How it works:
"""
For each channel independently:
1. Calculate mean of the batch
2. Calculate variance of the batch
3. Normalize: (x - mean) / sqrt(variance + eps)
4. Scale and shift: gamma * normalized + beta
"""

# Example:
x = torch.randn(4, 32, 16, 16)  # Batch of 4, 32 channels, 16×16
y = bn(x)
print(f"Mean: {y.mean():.3f}, Std: {y.std():.3f}")  # Mean ≈ 0, Std ≈ 1
```

### Why Use Batch Normalization?

1. **Faster Training**: Allow higher learning rates
2. **Less Sensitive to Initialization**: Not as picky about weight initialization
3. **Acts as Regularizer**: Slight regularization effect
4. **Enables Deeper Networks**: Helps with gradient flow

## 2.4 Activation Functions - Adding Nonlinearity

### Why Do We Need Activation Functions?

**Without activation functions**, neural networks are just linear combinations (like y = ax + b). **With activation functions**, they can learn complex patterns.

```python
# ============ RELU (Most Common) ============
# Formula: f(x) = max(0, x)
relu = nn.ReLU()
x = torch.tensor([-2, -1, 0, 1, 2])
print(f"Input: {x}")
print(f"ReLU: {relu(x)}")  # tensor([0, 0, 0, 1, 2])

# Why ReLU is popular:
# 1. Very fast to compute
# 2. Helps with vanishing gradient problem
# 3. Sparse activation (many zeros)

# ============ LEAKY RELU ============
# Formula: f(x) = max(0.01x, x)
leaky_relu = nn.LeakyReLU(0.01)
x = torch.tensor([-2, -1, 0, 1, 2])
print(f"Leaky ReLU: {leaky_relu(x)}")  # tensor([-0.02, -0.01, 0, 1, 2])

# ============ SOFTMAX (For Classification) ============
# Converts logits to probabilities (sums to 1)
softmax = nn.Softmax(dim=1)  # dim=1 means across classes
logits = torch.tensor([[1.0, 2.0, 3.0]])
probabilities = softmax(logits)
print(f"Logits: {logits}")
print(f"Probabilities: {probabilities}")  # tensor([[0.0900, 0.2447, 0.6652]])
print(f"Sum: {probabilities.sum()}")  # tensor(1.0)

# ============ SIGMOID ============
# Squeezes values between 0 and 1
sigmoid = nn.Sigmoid()
x = torch.tensor([-2, -1, 0, 1, 2])
print(f"Sigmoid: {sigmoid(x)}")  # tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])
```

## 2.5 Dropout - Preventing Overfitting

### What Does Dropout Do?

**Analogy:** During training, you randomly "turn off" some neurons so the network can't rely on any single one. This forces it to learn more robust features.

```python
# ============ DROPOUT EXPLAINED ============

dropout = nn.Dropout(
    p=0.5,          # Probability of turning off a neuron
    inplace=False   # Whether to modify in-place
)

# During training:
x = torch.randn(1, 10)
dropout.train()  # Set to training mode
y = dropout(x)
print(f"Training - {int((y == 0).sum())} neurons are 0")

# During testing:
dropout.eval()  # Set to evaluation mode
y = dropout(x)
print(f"Testing - {int((y == 0).sum())} neurons are 0")  # All neurons active!

# Important: During testing, outputs are scaled by (1 - p)
# So you don't need to do anything special
```

---

# 3. RESNET - SKIP CONNECTIONS MADE SIMPLE

## 3.1 The Problem: Vanishing Gradient

**Problem:** In very deep networks, gradients get smaller and smaller as they travel backwards through layers.

**Analogy:** Like a game of telephone - the message gets distorted as it passes through many people.

```
Layer 1 → Layer 2 → Layer 3 → ... → Layer 100
   Gradient gets smaller and smaller...
   By Layer 100, it's almost zero!
   → Network stops learning
```

## 3.2 The Solution: Skip Connections (Residual Connections)

**What is a Skip Connection?**

Instead of just passing data through layers, also add a shortcut that skips over some layers.

```python
# ============ RESNET BLOCK EXPLAINED ============

class BasicBlock(nn.Module):
    """Basic ResNet building block - LIKE A SHORTCUT"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # ============ MAIN PATH ============
        # First conv: Either 3×3 or 1×1 (if stride > 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second conv: Always 3×3
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # ============ SHORTCUT PATH (SKIP CONNECTION) ============
        # If dimensions don't match, use 1×1 conv to match
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # ============ MAIN PATH ============
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # ============ ADD SKIP CONNECTION ============
        # This is the magic! Adding input to output
        out = out + self.shortcut(x)  # ← THE SKIP CONNECTION
        out = self.relu(out)
        
        return out

# Visual Understanding:
"""
Input (x)
    │
    ├──────────────────────────────┐
    │                              │
    ↓                              │
  Conv1(3×3)                      │
    ↓                              │
  BN1                             │
    ↓                              │
  ReLU                            │
    ↓                              │
  Conv2(3×3)                      │
    ↓                              │
  BN2                             │
    │                              │
    └─────── ADD (+) ←────────────┘  ← SKIP CONNECTION
            ↓
          ReLU
            ↓
         Output
"""
```

## 3.3 Different ResNet Architectures

```python
# ============ RESNET18 ============
# 18 layers: Conv1 + 8 blocks × 2 convs each + FC layer

# ============ RESNET50 ============
# 50 layers: Conv1 + 16 blocks × 3 convs each + FC layer

# ============ RESNET152 ============
# 152 layers: Deepest standard ResNet

# ============ BOTTLENECK BLOCK (Used in ResNet50+) ============
class BottleneckBlock(nn.Module):
    """More efficient block - uses 1×1 convs to reduce computation"""
    expansion = 4  # Output channels = 4 × planes
    
    def __init__(self, in_channels, planes, stride=1):
        super().__init__()
        # 1×1 conv: Reduce channels
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 3×3 conv: Learn features
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, 
                              stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # 1×1 conv: Increase channels
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, planes * self.expansion, 
                         kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * self.expansion)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out = out + self.shortcut(x)  # Skip connection
        out = self.relu(out)
        
        return out
```

## 3.4 Building Full ResNet

```python
class ResNet(nn.Module):
    """Complete ResNet implementation"""
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        
        # ============ INITIAL LAYERS ============
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ============ RESIDUAL BLOCKS ============
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # ============ CLASSIFIER ============
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride=1):
        """Create a layer with multiple blocks"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, planes, stride))
            self.in_channels = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# ============ CREATE DIFFERENT RESNET VARIANTS ============
def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])

def resnet101():
    return ResNet(BottleneckBlock, [3, 4, 23, 3])

def resnet152():
    return ResNet(BottleneckBlock, [3, 8, 36, 3])
```

---

# 4. TRANSFER LEARNING - WHY AND HOW

## 4.1 What is Transfer Learning?

**Definition:** Using a model trained on one task as the starting point for a different task.

**Analogy:** 
- Learning to drive a car (100 hours) 
- Then learning to drive a truck (only 10 extra hours)
- You don't start from zero!

```python
# ============ TRANSFER LEARNING TYPES ============

# TYPE 1: FEATURE EXTRACTION
# Use pre-trained model as fixed feature extractor
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # ← Freeze all layers
# Only train new final layer
model.fc = nn.Linear(2048, num_classes)

# TYPE 2: FINE-TUNING
# Use pre-trained model and update some/all layers
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, num_classes)
# All layers are trainable (with lower learning rate)

# TYPE 3: PROGRESSIVE UNFREEZING
# Start with feature extraction, gradually unfreeze layers
def progressive_unfreeze(model, epoch):
    if epoch == 0:
        # Only train final layer
        for param in model.parameters():
            param.requires_grad = False
        model.fc.requires_grad = True
    elif epoch < 10:
        # Train last 2 layers
        for name, param in model.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True
    else:
        # Train everything
        for param in model.parameters():
            param.requires_grad = True
```

## 4.2 Why Use Transfer Learning?

```python
# ============ BENEFITS OF TRANSFER LEARNING ============

# 1. FASTER TRAINING
# Pre-trained: 10 epochs for good results
# From scratch: 100+ epochs

# 2. LESS DATA NEEDED
# Pre-trained: 1,000 images works
# From scratch: 100,000+ images needed

# 3. BETTER PERFORMANCE
# Pre-trained: 90%+ accuracy
# From scratch: 70% accuracy

# 4. LESS COMPUTATION
# Pre-trained: Train on CPU
# From scratch: Need multiple GPUs
```

## 4.3 When to Use Which Approach

```python
# ============ DECISION TREE ============

def choose_transfer_learning_approach(dataset_size, similarity_to_imagenet):
    """
    dataset_size: small (<1000), medium (1000-10000), large (>10000)
    similarity_to_imagenet: low (very different), medium, high (similar)
    """
    if dataset_size == 'small':
        if similarity_to_imagenet == 'high':
            return "Feature Extraction - Freeze all layers"
        else:
            return "Feature Extraction with aggressive augmentation"
    
    elif dataset_size == 'medium':
        if similarity_to_imagenet == 'high':
            return "Fine-tune last few layers (light)"
        else:
            return "Fine-tune more layers (medium)"
    
    else:  # large
        if similarity_to_imagenet == 'high':
            return "Fine-tune all layers with lower learning rate"
        else:
            return "Train from scratch or fine-tune all layers"
```

## 4.4 Implementing Transfer Learning Step by Step

```python
# ============ COMPLETE TRANSFER LEARNING PIPELINE ============

import torchvision.models as models

def setup_transfer_learning(num_classes, approach='finetune'):
    """Set up model for transfer learning"""
    
    # ============ STEP 1: LOAD PRE-TRAINED MODEL ============
    model = models.resnet50(weights='IMAGENET1K_V1')
    print("Loaded pre-trained ResNet50")
    
    # ============ STEP 2: REPLACE FINAL LAYER ============
    num_features = model.fc.in_features
    print(f"Original FC: {num_features} → 1000 classes")
    
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    print(f"New FC: {num_features} → 256 → {num_classes}")
    
    # ============ STEP 3: FREEZE LAYERS ============
    if approach == 'feature_extraction':
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        # Only train new layers
        for param in model.fc.parameters():
            param.requires_grad = True
        print("Feature extraction: Only training final layers")
    
    elif approach == 'finetune':
        # All layers are trainable
        print("Fine-tuning: Training all layers")
    
    elif approach == 'partial':
        # Freeze early layers, train later layers
        for name, param in model.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("Partial fine-tuning: Training layer4 and FC")
    
    return model

# ============ SETUP OPTIMIZER WITH DIFFERENT LRs ============
def setup_optimizer_with_diff_lr(model, base_lr=0.001):
    """Different learning rates for different layers"""
    
    # Separate parameters into groups
    fc_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'fc' in name:
                fc_params.append(param)
            else:
                other_params.append(param)
    
    optimizer = optim.SGD([
        {'params': other_params, 'lr': base_lr},           # Lower LR
        {'params': fc_params, 'lr': base_lr * 10}          # Higher LR for new layers
    ], momentum=0.9, weight_decay=1e-4)
    
    return optimizer
```

---

# 5. DATA AUGMENTATION - CREATING MORE DATA

## 5.1 What is Data Augmentation and Why Use It?

**Definition:** Creating new training samples by applying transformations to existing images.

**Why It's Important:**
- Prevents overfitting
- Increases effective dataset size
- Makes model more robust
- Helps with limited data

```python
# ============ VISUALIZING AUGMENTATION ============

def visualize_augmentations(image, transform, num_samples=8):
    """Show what different augmentations do"""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Augmented images
    for i in range(7):
        augmented = transform(Image.fromarray(image))
        axes[i+1].imshow(augmented)
        axes[i+1].set_title(f"Aug #{i+1}")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()
```

## 5.2 All Augmentation Techniques Explained

### Geometric Transformations

```python
# ============ 1. FLIPPING ============
# Horizontal flip: Mirror image left-right
transforms.RandomHorizontalFlip(p=0.5)  # 50% chance of flipping

# Vertical flip: Mirror image top-bottom
transforms.RandomVerticalFlip(p=0.1)    # 10% chance

# Why use flipping?
# - Natural: Objects can be seen from either side
# - Cheap: No computation cost
# - Effective: Helps with orientation invariance

# ============ 2. ROTATION ============
# Random rotation (in degrees)
transforms.RandomRotation(degrees=30)  # Rotate by ±30°

# Why use rotation?
# - Natural: Objects can be tilted
# - Helps with: Camera angle variation
# - Limits: Don't rotate too much for text

# ============ 3. RESIZING AND CROPPING ============
# Random crop to desired size
transforms.RandomResizedCrop(
    size=224,      # Output size
    scale=(0.8, 1.0)  # How much to zoom in/out
)

# Center crop (for validation)
transforms.CenterCrop(size=224)

# Why crop?
# - Focus on different parts of image
# - Simulates different camera distances
# - Removes unnecessary background

# ============ 4. TRANSLATION (SHIFTING) ============
# Shift image horizontally and vertically
transforms.RandomAffine(
    degrees=0,                    # No rotation
    translate=(0.1, 0.1)          # Shift by 10% of image
)

# Why translate?
# - Object not centered
# - Different framing
```

### Color Transformations

```python
# ============ 5. COLOR JITTER ============
transforms.ColorJitter(
    brightness=0.2,    # How much brightness varies
    contrast=0.2,      # How much contrast varies
    saturation=0.2,    # How much color intensity varies
    hue=0.1           # How much color hue varies
)

# Why use color jitter?
# - Lighting changes
# - Camera color variations
# - More robust to color differences

# ============ 6. GRAYSCALE ============
transforms.Grayscale(num_output_channels=3)

# Why grayscale?
# - Color might not be important
# - Reduces overfitting to colors

# ============ 7. GAUSSIAN BLUR ============
transforms.GaussianBlur(
    kernel_size=(3, 3),
    sigma=(0.1, 2.0)
)

# Why blur?
# - Simulates out-of-focus images
# - Reduces overfitting to sharp features
```

### Advanced Augmentations

```python
# ============ 8. CUTOUT ============
class Cutout:
    """Randomly mask out square regions"""
    def __init__(self, size=16):
        self.size = size
    
    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        x = np.random.randint(0, w - self.size)
        y = np.random.randint(0, h - self.size)
        img[:, y:y+self.size, x:x+self.size] = 0
        return img

# Why cutout?
# - Forces model to use context
# - Prevents overfitting to specific features
# - Like occlusion training

# ============ 9. MIXUP ============
def mixup_data(x, y, alpha=1.0):
    """Mix two images together"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# Why mixup?
# - Creates smooth transitions between classes
# - Improves generalization
# - Reduces overfitting

# ============ 10. RANDAUGMENT ============
transforms.RandAugment(
    num_ops=2,      # Number of augmentations to apply
    magnitude=9,    # How strong each augmentation is
)

# Why RandAugment?
# - Automatically chooses augmentations
# - No need to manually design pipeline
# - State-of-the-art performance
```

## 5.3 Complete Augmentation Pipeline

```python
# ============ COMPLETE AUGMENTATION PIPELINE ============

def get_augmentation_pipeline(aug_level='medium', input_size=224):
    """Create augmentation pipeline based on level"""
    
    # Normalization stats (ImageNet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if aug_level == 'none':
        # Minimal: Just resize and normalize
        return transforms.Compose([
            transforms.Resize(input_size + 32),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    elif aug_level == 'light':
        # Light augmentation
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    elif aug_level == 'medium':
        # Medium augmentation (good starting point)
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    elif aug_level == 'heavy':
        # Heavy augmentation (for small datasets)
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                                 saturation=0.3, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    else:
        raise ValueError(f"Unknown augmentation level: {aug_level}")
```

---

# 6. EFFICIENTNET - SMART SCALING

## 6.1 The Problem: How to Scale Networks

**Traditional Approach:** Just make networks deeper and wider.

**Problem:** It's like building a taller and wider building without strengthening the foundation - eventually it's not efficient.

## 6.2 What is Compound Scaling?

**Definition:** Simultaneously scaling depth, width, and resolution using a compound coefficient.

**Analogy:** Building a house:
- Depth = More floors (layers)
- Width = More rooms (channels)
- Resolution = Bigger windows (input size)

### What Each Scaling Does

```python
# ============ SCALING DIMENSIONS ============

# 1. DEPTH (Number of layers)
# More layers → More complex features
# Adds computational cost (slower)
# Example: ResNet18 → ResNet50 (3× more layers)

# 2. WIDTH (Number of channels)
# More channels → More features per layer
# Increases memory usage
# Example: 64 channels → 128 channels

# 3. RESOLUTION (Input image size)
# Higher resolution → More detail
# Increases computation quadratically
# Example: 224×224 → 448×448

# ============ COMPOUND SCALING ============
# EfficientNet scales all three with specific ratios:
# depth = α^φ
# width = β^φ  
# resolution = γ^φ
# where α·β²·γ² ≈ 2

# This means: If you double computation (φ=1), you scale:
# depth by α, width by β, resolution by γ
```

## 6.3 EfficientNet Variants

```python
# ============ EFFICIENTNET FAMILY ============

variant_info = {
    'B0': {'input': 224, 'params': 5.3, 'acc': 77.1},
    'B1': {'input': 240, 'params': 7.8, 'acc': 79.1},
    'B2': {'input': 260, 'params': 9.2, 'acc': 80.1},
    'B3': {'input': 300, 'params': 12.2, 'acc': 81.6},
    'B4': {'input': 380, 'params': 19.3, 'acc': 82.9},
    'B5': {'input': 456, 'params': 30.4, 'acc': 83.6},
    'B6': {'input': 528, 'params': 43.0, 'acc': 84.0},
    'B7': {'input': 600, 'params': 66.3, 'acc': 84.3},
}

def load_efficientnet(variant='b0', num_classes=1000):
    """Load EfficientNet with proper settings"""
    import torchvision.models as models
    
    # Map variant to function
    variants = {
        'b0': models.efficientnet_b0,
        'b1': models.efficientnet_b1,
        'b2': models.efficientnet_b2,
        'b3': models.efficientnet_b3,
        'b4': models.efficientnet_b4,
        'b5': models.efficientnet_b5,
        'b6': models.efficientnet_b6,
        'b7': models.efficientnet_b7,
    }
    
    model = variants[variant](weights='IMAGENET1K_V1')
    
    # Replace classifier
    if num_classes != 1000:
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    return model

# ============ EXAMPLE: USING EFFICIENTNET ============
model = load_efficientnet('b3', num_classes=10)
print(f"Model: EfficientNet-B3")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Input size: 300×300 (for B3)")
```

## 6.4 Training EfficientNet

```python
# ============ EFFICIENTNET TRAINING SETTINGS ============

def train_efficientnet_efficiently(model, train_loader, val_loader):
    """Train EfficientNet with recommended settings"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # ============ OPTIMIZER (EfficientNet specific) ============
    # RMSprop with specific parameters
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=0.001,
        alpha=0.9,          # RMSprop smoothing constant
        eps=1e-3,          # Stability
        momentum=0.9,
        weight_decay=1e-5
    )
    
    # ============ SCHEDULER ============
    # Exponential decay
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.97 ** (1 / 2.4)  # ~0.9874
    )
    
    # ============ LOSS ============
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # ============ TRAINING ============
    best_acc = 0.0
    
    for epoch in range(50):
        # Training
        model.train()
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        scheduler.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'efficientnet_best.pth')
    
    return model
```

---

# 7. OBJECT DETECTION - FINDING OBJECTS IN IMAGES

## 7.1 Object Detection vs Classification

```python
# ============ DIFFERENCE BETWEEN TASKS ============

# 1. CLASSIFICATION
# Question: "What is in this image?"
# Output: Single label
image_classifier = models.resnet50(pretrained=True)
output = image_classifier(image)  # tensor([0.2, 0.8, 0.0, ...])
# "This is a dog (80% confidence)"

# 2. OBJECT DETECTION
# Question: "What is where in this image?"
# Output: Multiple bounding boxes with labels
# "Dog at (100, 100, 200, 200), Cat at (300, 300, 150, 150)"
```

## 7.2 YOLO (You Only Look Once) - The Magic

**Why "You Only Look Once"?**
- Traditional methods: Look at image many times
- YOLO: Look at image just once

**Analogy:** 
- Traditional: Reading a book by scanning each page many times
- YOLO: Reading the entire book in one glance

### How YOLO Works

```
Step 1: Divide Image into Grid
┌────┬────┬────┬────┬────┬────┬────┐
│    │    │    │    │    │    │    │
├────┼────┼────┼────┼────┼────┼────┤
│    │    │    │    │    │    │    │
├────┼────┼────┼────┼────┼────┼────┤
│    │    │    │    │    │    │    │
├────┼────┼────┼────┼────┼────┼────┤
│    │    │    │    │    │    │    │
├────┼────┼────┼────┼────┼────┼────┤
│    │    │    │    │    │    │    │
├────┼────┼────┼────┼────┼────┼────┤
│    │    │    │    │    │    │    │
├────┼────┼────┼────┼────┼────┼────┤
│    │    │    │    │    │    │    │
└────┴────┴────┴────┴────┴────┴────┘

Step 2: Each Cell Predicts:
- Bounding boxes (coordinates)
- Confidence scores
- Class probabilities

Step 3: Apply Non-Maximum Suppression
- Remove duplicate detections
```

## 7.3 YOLO Architecture

```python
# ============ SIMPLIFIED YOLO ARCHITECTURE ============

class YOLOv5Backbone(nn.Module):
    """Feature extractor"""
    def __init__(self):
        super().__init__()
        # Focus layer: Reduces size, increases channels
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, 2)
        # ... more layers
    
    def forward(self, x):
        # Extract features at multiple scales
        return features

class YOLOv5Neck(nn.Module):
    """Feature Pyramid Network"""
    def __init__(self):
        super().__init__()
        # Combine features from different scales
    
    def forward(self, features):
        # Multi-scale feature fusion
        return fused_features

class YOLOv5Head(nn.Module):
    """Detection head"""
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        # Predict: boxes, confidence, classes
    
    def forward(self, features):
        # For each scale, predict:
        # (batch, num_anchors * (5 + num_classes), H, W)
        return predictions

# ============ COMPLETE YOLO ============
class YOLOv5(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = YOLOv5Backbone()
        self.neck = YOLOv5Neck()
        self.head = YOLOv5Head(num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.neck(features)
        predictions = self.head(features)
        return predictions
```

## 7.4 Understanding Bounding Boxes

```python
# ============ BOUNDING BOX FORMATS ============

# 1. XYXY Format: [x1, y1, x2, y2]
# x1, y1 = top-left corner
# x2, y2 = bottom-right corner
box_xyxy = [100, 50, 200, 150]
# Width = 200 - 100 = 100
# Height = 150 - 50 = 100

# 2. XYWH Format: [x, y, w, h]
# x, y = center of box
# w, h = width, height
box_xywh = [150, 100, 100, 100]
# x1 = 150 - 100/2 = 100
# y1 = 100 - 100/2 = 50
# x2 = 150 + 100/2 = 200
# y2 = 100 + 100/2 = 150

# 3. YOLO Format: [class, x_center, y_center, width, height]
# All values normalized to [0, 1]
box_yolo = [0, 0.5, 0.5, 0.3, 0.3]
# class_id: 0
# x_center: 50% of image width
# y_center: 50% of image height
# width: 30% of image width
# height: 30% of image height

def convert_bbox_xyxy_to_yolo(box_xyxy, img_width, img_height):
    """Convert XYXY to YOLO format"""
    x1, y1, x2, y2 = box_xyxy
    
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return [x_center, y_center, width, height]

def convert_bbox_yolo_to_xyxy(bbox_yolo, img_width, img_height):
    """Convert YOLO to XYXY format"""
    x_center, y_center, width, height = bbox_yolo
    
    x1 = (x_center - width/2) * img_width
    y1 = (y_center - height/2) * img_height
    x2 = (x_center + width/2) * img_width
    y2 = (y_center + height/2) * img_height
    
    return [int(x1), int(y1), int(x2), int(y2)]
```

## 7.5 Non-Maximum Suppression (NMS)

**Why NMS?** YOLO produces many overlapping bounding boxes. NMS removes duplicates.

```python
def non_max_suppression(predictions, conf_threshold=0.5, nms_threshold=0.4):
    """
    Remove duplicate detections
    
    Args:
        predictions: list of detections [x1, y1, x2, y2, conf, class]
        conf_threshold: Minimum confidence to keep
        nms_threshold: Overlap threshold to remove
    """
    
    # ============ STEP 1: Filter by confidence ============
    filtered = [p for p in predictions if p[4] > conf_threshold]
    
    if not filtered:
        return []
    
    # ============ STEP 2: Sort by confidence (descending) ============
    filtered.sort(key=lambda x: x[4], reverse=True)
    
    # ============ STEP 3: Remove overlapping boxes ============
    final_detections = []
    
    while filtered:
        # Take highest confidence box
        best = filtered.pop(0)
        final_detections.append(best)
        
        # Remove boxes that overlap too much
        filtered = [box for box in filtered if compute_iou(best, box) < nms_threshold]
    
    return final_detections

def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU)
    IoU = Area of Overlap / Area of Union
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # If no overlap
    if x2 < x1 or y2 < y1:
        return 0.0
    
    # Area of intersection
    intersection = (x2 - x1) * (y2 - y1)
    
    # Area of union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union
```

---

# 8. COMPLETE WORKING CODE WITH LINE-BY-LINE EXPLANATION

## 8.1 Transfer Learning Project - Complete Implementation

```python
# ============================================
# TRANSFER LEARNING WITH RESNET - COMPLETE CODE
# EVERY LINE EXPLAINED FOR BEGINNERS
# ============================================

# ============================================
# 1. IMPORTS - Understanding Each One
# ============================================

import torch
# PyTorch - Main deep learning library

import torch.nn as nn
# Neural network modules (layers, activations, etc.)

import torch.optim as optim
# Optimization algorithms (Adam, SGD, etc.)

import torchvision
# Computer vision datasets, models, transforms

import torchvision.transforms as transforms
# Image transformations (augmentation, normalization)

from torchvision import models
# Pre-trained models (ResNet, EfficientNet, etc.)

from torch.utils.data import DataLoader, Dataset
# Data loading utilities

import matplotlib.pyplot as plt
# Plotting and visualization

import numpy as np
# Numerical computations

import os
# Operating system operations

import time
# Time measurements

from sklearn.metrics import classification_report, confusion_matrix
# Evaluation metrics

import seaborn as sns
# Enhanced visualization

# ============================================
# 2. CONFIGURATION - Settings We Can Change
# ============================================

class Config:
    """Stores all settings in one place - Easy to change!"""
    
    # ----- Data Settings -----
    DATA_ROOT = './data'        # Where to save/download data
    BATCH_SIZE = 32            # Images per batch (can change: 16, 32, 64)
    NUM_WORKERS = 4            # Parallel data loading (can change: 2, 4, 8)
    
    # ----- Model Settings -----
    MODEL_NAME = 'resnet50'    # Which model to use (can change: 'resnet18', 'efficientnet')
    NUM_CLASSES = 10           # Number of categories (change for your dataset)
    
    # ----- Training Settings -----
    EPOCHS = 25                # Training iterations (can change: 10, 50, 100)
    LEARNING_RATE = 0.001      # Learning rate (can change: 0.01, 0.0001)
    WEIGHT_DECAY = 1e-4        # Regularization (can change: 1e-5, 1e-3)
    
    # ----- Device -----
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Automatically uses GPU if available, else CPU
    
    # ----- Paths -----
    MODEL_SAVE_PATH = 'best_resnet_model.pth'  # Where to save best model
    LOG_DIR = './logs'          # Where to save logs

# Let's understand each setting:
# BATCH_SIZE = 32 means we process 32 images at once
#   - Larger = Faster but needs more memory
#   - Smaller = Slower but more stable training
#   - If you get "CUDA out of memory", reduce this

# LEARNING_RATE = 0.001 means we take small steps
#   - Higher = Faster but may overshoot
#   - Lower = Slower but more stable
#   - 0.001 is a good starting point

# EPOCHS = 25 means we go through the entire dataset 25 times
#   - More = Better performance but slower
#   - Less = Faster but may not learn enough
#   - Watch for overfitting (training accuracy high, validation low)

# ============================================
# 3. DATA PREPARATION - Loading and Transforming
# ============================================

def get_transforms():
    """
    Creates data transformations for training and validation.
    
    What are transformations?
    - They prepare images before feeding to the model
    - Training transformations include augmentation to prevent overfitting
    - Validation transformations are just basic preprocessing
    """
    
    # ----- Normalization Statistics -----
    # These are standard for ImageNet (where ResNet was trained)
    # They ensure our images have similar statistics
    mean = [0.485, 0.456, 0.406]  # Mean of each channel (RGB)
    std = [0.229, 0.224, 0.225]    # Standard deviation of each channel
    
    """
    Why normalize?
    Images have pixel values 0-255. After converting to tensor, they're 0-1.
    Normalization makes them roughly [-2, 2] which helps training.
    """
    
    # ----- Training Transformations (With Augmentation) -----
    train_transform = transforms.Compose([
        # 1. Randomly crop and resize
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        # Why? Helps model learn different positions and sizes
        
        # 2. Random horizontal flip
        transforms.RandomHorizontalFlip(p=0.5),
        # Why? Objects can appear from either side
        
        # 3. Random rotation
        transforms.RandomRotation(degrees=15),
        # Why? Objects might be tilted
        
        # 4. Color variations
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        # Why? Makes model robust to lighting changes
        
        # 5. Convert to tensor
        transforms.ToTensor(),
        # Why? PyTorch works with tensors, not PIL images
        
        # 6. Normalize
        transforms.Normalize(mean, std)
        # Why? Matches ImageNet statistics for transfer learning
    ])
    
    # ----- Validation Transformations (No Augmentation) -----
    val_transform = transforms.Compose([
        # 1. Resize to larger size
        transforms.Resize(256),
        # Why? Ensures center crop works properly
        
        # 2. Center crop to model input size
        transforms.CenterCrop(224),
        # Why? Standard evaluation practice
        
        # 3. Convert to tensor
        transforms.ToTensor(),
        
        # 4. Normalize
        transforms.Normalize(mean, std)
    ])
    
    return train_transform, val_transform

def load_dataset():
    """
    Loads CIFAR-10 dataset (can be replaced with any dataset)
    
    For your custom dataset, use:
    dataset = torchvision.datasets.ImageFolder(root='path/to/data', transform=transform)
    """
    
    # Get transformations
    train_transform, val_transform = get_transforms()
    
    # ----- Load Training Data -----
    train_dataset = torchvision.datasets.CIFAR10(
        root=Config.DATA_ROOT,    # Where to save
        train=True,               # Training split
        download=True,            # Download if not found
        transform=train_transform # Apply training transformations
    )
    
    # ----- Load Validation Data -----
    val_dataset = torchvision.datasets.CIFAR10(
        root=Config.DATA_ROOT,
        train=False,              # Test/Validation split
        download=True,
        transform=val_transform   # Apply validation transformations
    )
    
    # ----- Create DataLoaders -----
    """
    DataLoader efficiently loads data in batches.
    It handles:
    1. Batching
    2. Shuffling
    3. Parallel loading
    4. Memory pinning (for faster GPU transfer)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,  # Images per batch
        shuffle=True,                   # Randomize order each epoch
        num_workers=Config.NUM_WORKERS, # Parallel processes
        pin_memory=True                 # Faster transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,                  # No shuffle for validation
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Get class names
    classes = train_dataset.classes  # ['airplane', 'automobile', 'bird', ...]
    
    print(f"Loaded {len(train_dataset)} training images")
    print(f"Loaded {len(val_dataset)} validation images")
    print(f"Classes: {classes}")
    
    return train_loader, val_loader, classes

# ============================================
# 4. MODEL SETUP - Creating the Network
# ============================================

def setup_model(num_classes=10, freeze_layers=True):
    """
    Sets up ResNet with pre-trained weights.
    
    This is the HEART of transfer learning!
    
    Args:
        num_classes: Number of categories in your dataset
        freeze_layers: Whether to freeze pre-trained layers
    """
    
    # ----- Load Pre-trained Model -----
    model = models.resnet50(weights='IMAGENET1K_V1')
    """
    This loads ResNet50 trained on ImageNet (1.4M images, 1000 classes).
    The model has learned to detect general features like edges, textures.
    
    We're reusing this knowledge for our task!
    """
    
    # ----- Understand Model Architecture -----
    print("Original ResNet50 Architecture:")
    print(model)
    """
    Original model ends with:
    (fc): Linear(in_features=2048, out_features=1000, bias=True)
    We need to change out_features to our number of classes.
    """
    
    # ----- Replace Final Layer -----
    num_features = model.fc.in_features  # 2048
    print(f"Final layer input features: {num_features}")
    
    # Create new final layer
    model.fc = nn.Linear(num_features, num_classes)
    """
    This replaces the classification head.
    For CIFAR-10 (10 classes): 2048 → 10
    For your custom dataset: Change num_classes accordingly
    """
    
    # ----- Optionally Add More Layers -----
    # For better performance on your dataset:
    # model.fc = nn.Sequential(
    #     nn.Dropout(0.5),
    #     nn.Linear(num_features, 256),
    #     nn.ReLU(),
    #     nn.Dropout(0.3),
    #     nn.Linear(256, num_classes)
    # )
    # This adds more capacity to the final layers
    
    # ----- Freeze Pre-trained Layers -----
    if freeze_layers:
        """
        Freezing means these layers won't be updated during training.
        We only train the new final layer.
        This is called "Feature Extraction".
        
        Why freeze?
        - Faster training
        - Less data needed
        - Prevents overfitting
        
        When NOT to freeze?
        - If you have lots of data
        - If your dataset is very different from ImageNet
        """
        for param in model.parameters():
            param.requires_grad = False
            # requires_grad=False means PyTorch won't update this parameter
        
        # Unfreeze the new final layer
        for param in model.fc.parameters():
            param.requires_grad = True
            # Only the final layer will learn
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    return model

def setup_optimizer(model):
    """
    Sets up optimizer with different learning rates for different layers.
    
    Why different learning rates?
    - Pre-trained layers: Small learning rate (don't change much)
    - New layers: Higher learning rate (learn quickly)
    """
    
    # ----- Separate Parameters -----
    fc_params = []      # Parameters of the new final layer
    other_params = []   # Parameters of pre-trained layers
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'fc' in name:
                fc_params.append(param)
            else:
                other_params.append(param)
    
    print(f"FC parameters: {sum(p.numel() for p in fc_params):,}")
    print(f"Other parameters: {sum(p.numel() for p in other_params):,}")
    
    # ----- Create Optimizer -----
    """
    SGD (Stochastic Gradient Descent) with momentum.
    
    Why SGD over Adam?
    - Works better for fine-tuning
    - More stable convergence
    - Better generalization
    
    Parameters:
    - params: Parameters to optimize
    - lr: Learning rate (step size)
    - momentum: Accelerates convergence
    - weight_decay: L2 regularization
    """
    optimizer = optim.SGD([
        {'params': other_params, 'lr': Config.LEARNING_RATE},
        {'params': fc_params, 'lr': Config.LEARNING_RATE * 10}
    ], momentum=0.9, weight_decay=Config.WEIGHT_DECAY)
    """
    new layers have 10x learning rate because they need to learn faster
    """
    
    # ----- Learning Rate Scheduler -----
    """
    Reduces learning rate when validation loss plateaus.
    Helps fine-tune model at the end of training.
    """
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',     # Reduce when loss stops decreasing
        factor=0.1,     # Multiply LR by 0.1
        patience=3,     # Wait 3 epochs before reducing
        verbose=True    # Print when reducing
    )
    
    return optimizer, scheduler

# ============================================
# 5. TRAINING - Teaching the Model
# ============================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Trains for one epoch (one pass through the data).
    
    What happens in one epoch?
    1. Model processes all training images
    2. Calculates loss
    3. Updates weights
    """
    
    model.train()  # Set to training mode (enables dropout, batch norm)
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Iterate through batches
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to GPU/CPU
        images = images.to(device)   # (batch, 3, 224, 224)
        labels = labels.to(device)   # (batch)
        
        # ----- Forward Pass -----
        optimizer.zero_grad()         # Reset gradients
        outputs = model(images)       # Get predictions
        loss = criterion(outputs, labels)  # Calculate loss
        
        # ----- Backward Pass -----
        loss.backward()               # Compute gradients
        optimizer.step()              # Update weights
        
        # ----- Track Statistics -----
        running_loss += loss.item()
        _, predicted = outputs.max(1)  # Get predicted class
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print progress every 50 batches
        if batch_idx % 50 == 49:
            print(f'  Batch {batch_idx+1}: Loss: {running_loss/50:.4f}, Acc: {100*correct/total:.2f}%')
            running_loss = 0.0
    
    # Return accuracy for this epoch
    return 100 * correct / total

def validate_epoch(model, val_loader, criterion, device):
    """
    Validates the model on validation data.
    
    Difference from training:
    - No gradient computation (faster)
    - No weight updates
    - Dropout and batch norm use population statistics
    """
    
    model.eval()  # Set to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    
    # Disable gradient computation (saves memory and time)
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Return loss and accuracy
    return val_loss / len(val_loader), 100 * correct / total

def train_model(model, train_loader, val_loader, epochs=25):
    """
    Complete training pipeline.
    
    This function:
    1. Sets up optimizer and scheduler
    2. Runs training loop
    3. Saves best model
    4. Tracks history
    """
    
    device = Config.DEVICE
    model = model.to(device)
    
    # ----- Setup -----
    criterion = nn.CrossEntropyLoss()  # For classification
    optimizer, scheduler = setup_optimizer(model)
    
    best_val_acc = 0.0
    history = {
        'train_acc': [],
        'val_acc': [],
        'val_loss': []
    }
    
    print("=" * 60)
    print(f"Training {Config.MODEL_NAME.upper()}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print("=" * 60)
    
    start_time = time.time()
    
    # ----- Training Loop -----
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 40)
        
        # Train for one epoch
        train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler (uses validation loss)
        scheduler.step(val_loss)
        
        # Save history
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        # Print results
        print(f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"✓ Best model saved! (Acc: {best_val_acc:.2f}%)")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    return model, history

# ============================================
# 6. EVALUATION - Testing Model Performance
# ============================================

def evaluate_model(model, val_loader, classes, device):
    """
    Detailed evaluation with metrics.
    
    This gives us:
    1. Per-class accuracy
    2. Precision, Recall, F1-score
    3. Confusion matrix
    """
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # ----- Classification Report -----
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(all_labels, all_predictions, target_names=classes)
    print(report)
    
    # ----- Confusion Matrix -----
    cm = confusion_matrix(all_labels, all_predictions)
    
    return all_predictions, all_labels, cm

def plot_confusion_matrix(cm, classes):
    """Visualize confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()

def plot_history(history):
    """Visualize training progress"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # ----- Accuracy Plot -----
    axes[0].plot(history['train_acc'], label='Train', linewidth=2)
    axes[0].plot(history['val_acc'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ----- Loss Plot -----
    axes[1].plot(history['val_loss'], label='Validation Loss', linewidth=2, color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.show()

# ============================================
# 7. MAIN - Running Everything
# ============================================

def main():
    """
    Main function - The entire pipeline from start to finish.
    """
    
    print("=" * 60)
    print("TRANSFER LEARNING WITH RESNET")
    print("Tech Prime Pvt Limited - Advanced AI/ML Internship")
    print("=" * 60)
    
    # ----- 1. Load Data -----
    print("\n[1/5] Loading data...")
    train_loader, val_loader, classes = load_dataset()
    print(f"✓ Loaded {len(train_loader.dataset)} training images")
    print(f"✓ Loaded {len(val_loader.dataset)} validation images")
    print(f"✓ Classes: {classes}")
    
    # ----- 2. Setup Model -----
    print(f"\n[2/5] Setting up {Config.MODEL_NAME.upper()}...")
    model = setup_model(num_classes=Config.NUM_CLASSES, freeze_layers=True)
    print("✓ Model created")
    
    # ----- 3. Train -----
    print("\n[3/5] Training model...")
    model, history = train_model(model, train_loader, val_loader, Config.EPOCHS)
    print("✓ Training complete")
    
    # ----- 4. Load Best Model -----
    print("\n[4/5] Loading best model...")
    model = setup_model(num_classes=Config.NUM_CLASSES, freeze_layers=False)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model = model.to(Config.DEVICE)
    print("✓ Best model loaded")
    
    # ----- 5. Evaluate -----
    print("\n[5/5] Evaluating model...")
    predictions, labels, cm = evaluate_model(model, val_loader, classes, Config.DEVICE)
    print("✓ Evaluation complete")
    
    # ----- Visualize Results -----
    print("\nGenerating visualizations...")
    plot_confusion_matrix(cm, classes)
    plot_history(history)
    print("✓ Visualizations saved")
    
    print("\n" + "=" * 60)
    print("✓ PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nFiles saved:")
    print(f"  - {Config.MODEL_SAVE_PATH}")
    print("  - confusion_matrix.png")
    print("  - training_history.png")

if __name__ == "__main__":
    main()
```

---

# 9. COMMON ISSUES AND SOLUTIONS

## 9.1 GPU Memory Issues

```python
# ============ PROBLEM: "CUDA out of memory" ============

# SOLUTION 1: Reduce batch size
Config.BATCH_SIZE = 16  # Instead of 32

# SOLUTION 2: Use smaller model
MODEL_NAME = 'resnet18'  # Instead of 'resnet50'

# SOLUTION 3: Use gradient accumulation
accumulation_steps = 4
optimizer.zero_grad()
for i, (images, labels) in enumerate(train_loader):
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss = loss / accumulation_steps  # Normalize
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# SOLUTION 4: Clear cache
torch.cuda.empty_cache()

# SOLUTION 5: Use mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 9.2 Training Not Learning

```python
# ============ PROBLEM: Loss not decreasing ============

# SOLUTION 1: Check learning rate
# Too high: Loss explodes
# Too low: Training too slow
optimizer = optim.SGD(model.parameters(), lr=0.001)  # Try different values

# SOLUTION 2: Check data normalization
# Images should be normalized
transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# SOLUTION 3: Don't freeze too many layers
# If dataset is large, unfreeze more layers
for param in model.parameters():
    param.requires_grad = True

# SOLUTION 4: Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 9.3 Overfitting

```python
# ============ PROBLEM: Train accuracy high, Validation low ============

# SOLUTION 1: Add more augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# SOLUTION 2: Increase regularization
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-3)

# SOLUTION 3: Add dropout
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)

# SOLUTION 4: Reduce model capacity
model = models.resnet18(pretrained=True)  # Smaller than ResNet50

# SOLUTION 5: Early stopping
# Stop training when validation loss stops improving
```

---

# 10. QUICK REFERENCE - ALL CODE PATTERNS

## 10.1 Creating Models

```python
# ============ SIMPLE CNN ============
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 8 * 8, 10)  # Adjust for your input size
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ============ RESNET WITH TRANSFER LEARNING ============
model = models.resnet50(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, num_classes)

# ============ EFFICIENTNET ============
model = models.efficientnet_b0(weights='IMAGENET1K_V1')
model.classifier = nn.Linear(model.classifier[1].in_features, num_classes)
```

## 10.2 Data Loading

```python
# ============ BASIC DATALOADER ============
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# ============ CUSTOM DATASET ============
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
```

## 10.3 Training

```python
# ============ TRAINING LOOP ============
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # Calculate metrics

# ==================== EVALUATION ============
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
```

## 10.4 Save and Load

```python
# ============ SAVE MODEL ============
torch.save(model.state_dict(), 'model.pth')

# ============ LOAD MODEL ============
model = models.resnet50(weights=None)
model.fc = nn.Linear(2048, num_classes)
model.load_state_dict(torch.load('model.pth'))

# ============ SAVE CHECKPOINT ============
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# ============ LOAD CHECKPOINT ============
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

---

**End of Week 2 Notes - Complete Noob-Friendly Guide**
