# COMPLETE WEEK 3 NOTES - Advanced Computer Vision (ULTIMATE NOOB-FRIENDLY GUIDE)

## Professional Training Documentation
**Tech Prime Pvt Limited - Advanced AI/ML Internship Program**

---

# 📑 TABLE OF CONTENTS

1. [What is Image Segmentation? - For Absolute Beginners](#1-what-is-image-segmentation-for-absolute-beginners)
2. [Types of Segmentation - Classification, Detection, and Segmentation](#2-types-of-segmentation-classification-detection-and-segmentation)
3. [U-Net Architecture - The Most Important Segmentation Model](#3-u-net-architecture-the-most-important-segmentation-model)
4. [OpenCV - Your Swiss Army Knife for Images](#4-opencv-your-swiss-army-knife-for-images)
5. [Image Preprocessing - Cleaning Your Data](#5-image-preprocessing-cleaning-your-data)
6. [Medical Image Segmentation - Real-World Application](#6-medical-image-segmentation-real-world-application)
7. [Complete Working Code with Line-by-Line Explanation](#7-complete-working-code-with-line-by-line-explanation)
8. [Common Issues and Solutions](#8-common-issues-and-solutions)
9. [Quick Reference - All Code Patterns](#9-quick-reference-all-code-patterns)

---

# 1. WHAT IS IMAGE SEGMENTATION? - FOR ABSOLUTE BEGINNERS

## 1.1 The BIG Question: What is Segmentation?

**Imagine this:** You're looking at a photo of a street. Let's compare different computer vision tasks:

```
TASK 1: CLASSIFICATION
"What is in this image?"
Answer: "A car"

TASK 2: OBJECT DETECTION
"What is where in this image?"
Answer: "Car at (100, 100, 200, 150)" [Bounding Box]

TASK 3: SEGMENTATION
"What pixel belongs to what?"
Answer: "These 5,000 pixels are the car,
        These 3,000 pixels are the road,
        These 2,000 pixels are the sky..."
        [Every pixel gets a label!]
```

### Visual Comparison

```
Original Image:                Classification:                 Detection:
┌─────────────────────┐        ┌─────────────────────┐        ┌─────────────────────┐
│     🌳     🏠       │        │     🌳     🏠       │        │     🌳     🏠       │
│                     │        │                     │        │                     │
│      🚗             │  →    │ "This is a street"  │  →    │   ┌───┐              │
│                     │        │                     │        │   │🚗 │              │
│   🧑         🐕     │        │                     │        │   └───┘              │
└─────────────────────┘        └─────────────────────┘        │   ┌───┐              │
                                                              │   │🧑 │              │
                                                              │   └───┘              │
                                                              └─────────────────────┘

Segmentation:                           Semantic Segmentation:
┌─────────────────────┐                 ┌─────────────────────┐
│  🌳🌳🌳  🏠🏠🏠     │                 │  🟩🟩🟩  🟦🟦🟦     │
│  🌳🌳🌳  🏠🏠🏠     │                 │  🟩🟩🟩  🟦🟦🟦     │
│    🚗🚗🚗            │  →              │    🟥🟥🟥            │
│  🧑🧑    🐕🐕🐕     │                 │  🟨🟨    🟪🟪🟪     │
│  🧑🧑    🐕🐕🐕     │                 │  🟨🟨    🟪🟪🟪     │
└─────────────────────┘                 └─────────────────────┘
                                         Each color = Different class
```

## 1.2 Why Segmentation is Important?

### Real-World Applications

```python
# ============ APPLICATIONS OF SEGMENTATION ============

# 1. MEDICAL IMAGING
# - Find tumors in MRI scans
# - Segment organs for surgery planning
# - Detect cancer cells in microscopy images
print("Medical: Finding tumors in brain scans")

# 2. SELF-DRIVING CARS
# - Separate road from sidewalk
# - Detect pedestrians (every pixel)
# - Identify traffic signs
print("Autonomous Driving: Understanding road scenes")

# 3. AGRICULTURE
# - Detect diseased plants
# - Segment crops from weeds
# - Count fruits on trees
print("Agriculture: Detecting plant diseases")

# 4. SATELLITE IMAGERY
# - Map forests, water, cities
# - Detect flood areas
# - Monitor deforestation
print("Satellite: Mapping land use")

# 5. PHOTO EDITING
# - Remove backgrounds
# - Change object colors
# - Apply effects to specific areas
print("Photo Editing: Background removal")
```

## 1.3 Types of Segmentation - Understanding the Differences

### Semantic Segmentation

**Definition:** Label every pixel with a class (but all objects of same class get same label)

```python
# ============ SEMANTIC SEGMENTATION ============
"""
What it does:
- Every pixel gets a label (car, road, person, etc.)
- ALL cars get the same label
- ALL people get the same label

Example:
Original: [Car 1, Car 2, Person 1]
Output:   [Car, Car, Person]  ← Same label for all cars!

Visual:
┌─────────────────────────────────┐
│ 🟦 Sky (all sky pixels)         │
│ 🟩 Grass (all grass pixels)     │
│ 🟥 Car (all car pixels)         │
│ 🟨 Person (all person pixels)   │
└─────────────────────────────────┘
"""
```

### Instance Segmentation

**Definition:** Label every pixel with class AND distinguish different objects of same class

```python
# ============ INSTANCE SEGMENTATION ============
"""
What it does:
- Every pixel gets a label
- DIFFERENT objects get DIFFERENT labels
- Same type objects are separated

Example:
Original: [Car 1, Car 2, Person 1]
Output:   [Car_1, Car_2, Person_1]  ← Different labels for each!

Visual:
┌─────────────────────────────────┐
│ 🟦 Sky                          │
│ 🟩 Grass                        │
│ 🟥 Car #1                       │
│ 🟧 Car #2                       │
│ 🟨 Person #1                    │
└─────────────────────────────────┘
"""
```

### Panoptic Segmentation

**Definition:** Combines semantic AND instance segmentation

```python
# ============ PANOPTIC SEGMENTATION ============
"""
What it does:
- For "stuff" (sky, road, grass): Semantic segmentation
- For "things" (cars, people, animals): Instance segmentation

Example:
- Sky: All sky pixels get label "Sky" (semantic)
- Road: All road pixels get label "Road" (semantic)
- Car 1: Gets label "Car_1" (instance)
- Car 2: Gets label "Car_2" (instance)
- Person: Gets label "Person_1" (instance)
"""
```

### Understanding with Simple Code

```python
# ============ DEMONSTRATING DIFFERENT SEGMENTATION TYPES ============

import torch
import matplotlib.pyplot as plt
import numpy as np

def create_demo_image():
    """Create a simple image for demonstration"""
    # Create a 100x100 image with different objects
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Blue sky (top 40 rows)
    image[0:40, :, :] = [135, 206, 235]  # Sky blue
    
    # Green grass (bottom 60 rows)
    image[40:100, :, :] = [34, 139, 34]  # Forest green
    
    # Red car (center)
    image[45:70, 20:50, :] = [255, 0, 0]  # Red
    
    # Another red car (right side)
    image[45:70, 65:95, :] = [255, 0, 0]  # Red
    
    # Yellow person (left)
    image[50:80, 5:15, :] = [255, 255, 0]  # Yellow
    
    return image

def show_segmentation_types():
    """Show different segmentation outputs"""
    
    original = create_demo_image()
    
    # ============ SEMANTIC SEGMENTATION OUTPUT ============
    semantic_output = np.zeros((100, 100), dtype=np.int32)
    semantic_output[0:40, :] = 0    # Sky
    semantic_output[40:100, :] = 1  # Grass
    semantic_output[45:70, 20:50] = 2  # Car (ALL cars are same)
    semantic_output[45:70, 65:95] = 2  # Car (ALL cars are same)
    semantic_output[50:80, 5:15] = 3   # Person
    
    # ============ INSTANCE SEGMENTATION OUTPUT ============
    instance_output = np.zeros((100, 100), dtype=np.int32)
    instance_output[0:40, :] = 0     # Sky (semantic for stuff)
    instance_output[40:100, :] = 1   # Grass (semantic for stuff)
    instance_output[45:70, 20:50] = 2  # Car 1 (instance)
    instance_output[45:70, 65:95] = 3  # Car 2 (instance - DIFFERENT!)
    instance_output[50:80, 5:15] = 4   # Person (instance)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(semantic_output, cmap='tab10')
    axes[1].set_title("Semantic Segmentation\n(All cars are same color)", fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(instance_output, cmap='tab10')
    axes[2].set_title("Instance Segmentation\n(Cars are different colors)", fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("KEY DIFFERENCES")
    print("="*60)
    print("Semantic: ALL cars → Label 2")
    print("Instance: Car 1 → Label 2, Car 2 → Label 3")
    print("Panoptic: Sky → Label 0, Car 1 → Label 2, Car 2 → Label 3")

# Run it!
show_segmentation_types()
```

## 1.4 Segmentation vs. Classification vs. Detection

```python
# ============ COMPLETE COMPARISON ============

comparison = {
    "Task": ["Classification", "Detection", "Semantic Segmentation"],
    "Question": [
        "What is in this image?",
        "What is where?",
        "What pixel belongs to what?"
    ],
    "Output": [
        "Single label",
        "Bounding boxes + labels",
        "Pixel-wise labels"
    ],
    "Example": [
        '"Cat"',
        '"Cat at (100,100,200,200)"',
        '"These 50,000 pixels are cat"'
    ],
    "Use Case": [
        "Image search",
        "Self-driving cars",
        "Medical imaging"
    ]
}

print("="*70)
print("SEGMENTATION VS CLASSIFICATION VS DETECTION")
print("="*70)

for key, values in comparison.items():
    print(f"\n{key.upper()}:")
    for i, val in enumerate(values):
        print(f"  {['📌', '🔍', '🎯'][i]}: {val}")

# ============ OUTPUT EXPLANATION ============
"""
Classification: 
  ┌──────────────┐
  │  🐱          │  → "This is a cat"
  │              │
  └──────────────┘

Detection:
  ┌──────────────┐
  │  ┌──────┐    │  → "Cat at (50,50,150,150)"
  │  │ 🐱   │    │
  │  └──────┘    │
  └──────────────┘

Segmentation:
  ┌──────────────┐
  │  ████████    │  → "These pixels are cat"
  │  ██🐱████    │
  │  ████████    │
  └──────────────┘
"""
```

---

# 2. TYPES OF SEGMENTATION - CLASSIFICATION, DETECTION, AND SEGMENTATION

## 2.1 The Evolution from Classification to Segmentation

```python
# ============ EVOLUTION OF COMPUTER VISION TASKS ============

"""
LEVEL 1: CLASSIFICATION (Image-level)
↓
LEVEL 2: DETECTION (Bounding Box-level)
↓
LEVEL 3: SEMANTIC SEGMENTATION (Pixel-level)
↓
LEVEL 4: INSTANCE SEGMENTATION (Object-level)
↓
LEVEL 5: PANOPTIC SEGMENTATION (Both)
"""

# ============ VISUAL REPRESENTATION ============

def visualize_evolution():
    """Show how tasks become more detailed"""
    
    # Create a simple scene
    scene = np.zeros((200, 200, 3), dtype=np.uint8)
    scene[0:80, :] = [135, 206, 235]  # Sky
    scene[80:200, :] = [34, 139, 34]  # Grass
    scene[120:160, 70:130] = [255, 0, 0]  # Car
    scene[100:180, 140:180] = [255, 255, 0]  # Person
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Level 1: Classification
    axes[0, 0].imshow(scene)
    axes[0, 0].set_title("Classification\n" + '"This is a street scene"', fontsize=12)
    axes[0, 0].axis('off')
    
    # Level 2: Detection
    axes[0, 1].imshow(scene)
    axes[0, 1].add_patch(plt.Rectangle((70, 120), 60, 40, 
                                      fill=False, edgecolor='red', linewidth=3))
    axes[0, 1].add_patch(plt.Rectangle((140, 100), 40, 80, 
                                      fill=False, edgecolor='yellow', linewidth=3))
    axes[0, 1].set_title("Detection\n" + '"Car at (70,120), Person at (140,100)"', fontsize=12)
    axes[0, 1].axis('off')
    
    # Level 3: Semantic Segmentation
    semantic = np.zeros((200, 200, 3), dtype=np.uint8)
    semantic[0:80, :] = [135, 206, 235]  # Sky
    semantic[80:200, :] = [34, 139, 34]  # Grass
    semantic[120:160, 70:130] = [255, 0, 0]  # Car
    semantic[100:180, 140:180] = [255, 255, 0]  # Person
    
    axes[0, 2].imshow(semantic)
    axes[0, 2].set_title("Semantic Segmentation\n" + '"Every pixel labeled by class"', fontsize=12)
    axes[0, 2].axis('off')
    
    # Level 4: Instance Segmentation
    instance = np.zeros((200, 200, 3), dtype=np.uint8)
    instance[0:80, :] = [135, 206, 235]  # Sky
    instance[80:200, :] = [34, 139, 34]  # Grass
    instance[120:160, 70:130] = [200, 0, 0]  # Car 1
    instance[100:180, 140:180] = [255, 255, 0]  # Person
    
    axes[1, 0].imshow(instance)
    axes[1, 0].set_title("Instance Segmentation\n" + '"Each object has unique label"', fontsize=12)
    axes[1, 0].axis('off')
    
    # Level 5: Panoptic
    axes[1, 1].imshow(instance)
    axes[1, 1].set_title("Panoptic Segmentation\n" + '"Stuff (sky,grass) + Things (car,person)"', fontsize=12)
    axes[1, 1].axis('off')
    
    # Legend
    axes[1, 2].axis('off')
    legend_text = """
    COMPARISON:
    
    🌅 Classification: Image-level
       "What is this?"
    
    📦 Detection: Box-level
       "What is where?"
    
    🎨 Semantic Seg: Pixel-level
       "What pixel belongs to what class?"
    
    🎭 Instance Seg: Object-level
       "Which object is this pixel from?"
    
    🎪 Panoptic: Both
       Everything gets labeled!
    """
    axes[1, 2].text(0.1, 0.5, legend_text, transform=axes[1, 2].transAxes,
                   fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    plt.show()

visualize_evolution()
```

## 2.2 Key Terminology You MUST Know

```python
# ============ ESSENTIAL SEGMENTATION TERMS ============

# 1. MASK
"""
What is a mask?
- A 2D array where each pixel has a label/class
- Same size as the original image

Example:
Original Image (3×3):        Mask (3×3):
┌─────────────┐              ┌─────────────┐
│ 255 255 255 │              │ 0   0   0   │
│ 255 0   0   │      →       │ 0   1   1   │
│ 255 0   0   │              │ 0   1   1   │
└─────────────┘              └─────────────┘
                             0 = Background
                             1 = Object
"""
mask = np.array([[0, 0, 0],
                 [0, 1, 1],
                 [0, 1, 1]])

# 2. GROUND TRUTH
"""
The "correct answer" for training
- Human-annotated masks
- Used to calculate loss
- The target we want to predict
"""

# 3. IOU (Intersection over Union)
"""
Measure of how well prediction matches ground truth
IoU = Area of Overlap / Area of Union

Example:
Ground Truth:    Prediction:     Overlap:
┌─────┐          ┌─────┐         ┌─────┐
│█████│          │███  │         │███  │
│█████│          │███  │         │███  │
│█████│          │  ███│         │  █  │
└─────┘          └─────┘         └─────┘

IoU = Overlap / (GT + Pred - Overlap)
"""

def calculate_iou(gt_mask, pred_mask):
    """Calculate Intersection over Union"""
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    iou = intersection / union if union > 0 else 0
    return iou

# 4. DICE SCORE (F1 Score for Segmentation)
"""
Similar to IoU but more sensitive to overlap
Dice = 2 × Intersection / (GT_size + Pred_size)

When to use:
- Medical imaging (prefers more overlap)
- Imbalanced datasets
"""
def calculate_dice(gt_mask, pred_mask):
    """Calculate Dice Score"""
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    total = gt_mask.sum() + pred_mask.sum()
    dice = 2 * intersection / total if total > 0 else 0
    return dice

# 5. CONFUSION MATRIX FOR SEGMENTATION
"""
For each pixel:
- TP: Both say Object
- TN: Both say Background
- FP: Prediction says Object, GT says Background
- FN: Prediction says Background, GT says Object
"""

# ============ EXAMPLE CALCULATIONS ============

def example_metrics():
    """Show how metrics work"""
    
    # Create example masks
    gt = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 0]])
    
    pred1 = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 0]])  # Perfect
    
    pred2 = np.array([[1, 1, 0],
                      [1, 1, 0],
                      [1, 1, 0]])  # Missing some
    
    print("="*60)
    print("SEGMENTATION METRICS EXAMPLE")
    print("="*60)
    print(f"Ground Truth:\n{gt}\n")
    print(f"Prediction 1 (Perfect):\n{pred1}")
    print(f"IoU: {calculate_iou(gt, pred1):.2f}")
    print(f"Dice: {calculate_dice(gt, pred1):.2f}")
    print()
    print(f"Prediction 2 (Missing):\n{pred2}")
    print(f"IoU: {calculate_iou(gt, pred2):.2f}")
    print(f"Dice: {calculate_dice(gt, pred2):.2f}")

example_metrics()
```

---

# 3. U-NET ARCHITECTURE - THE MOST IMPORTANT SEGMENTATION MODEL

## 3.1 What is U-Net and Why is it Special?

**Analogy:** Think of U-Net as a specialized detective agency:

```
CONTRACTION PATH (Encoder) - "The Investigators"
┌─────────────────────────────────────────────┐
│ Layer 1: "I see edges"                     │
│ Layer 2: "I see shapes"                    │
│ Layer 3: "I see objects"                   │
│ Layer 4: "I understand the whole scene"   │
└─────────────────────────────────────────────┘
              ↓ (They summarize their findings)

EXPANSION PATH (Decoder) - "The Report Writers"
┌─────────────────────────────────────────────┐
│ Layer 4: "The overall scene is..."         │
│ Layer 3: "Here are the details..."        │
│ Layer 2: "And here are the pixel-level..." │
│ Layer 1: "Here's the final segmented image"│
└─────────────────────────────────────────────┘
```

### The U-Shaped Architecture

```
Input Image                          Output Mask
    (572×572)                           (388×388)
       │                                   ▲
       │   ┌──────────────────────────┐   │
       │   │                          │   │
       ▼   │   ┌──────────────────┐   │   │
    ┌───┐  │   │                  │   │  ┌───┐
    │C1 │  │   │   ┌──────────┐   │   │  │D1 │
    └─┬─┘  │   │   │          │   │   │  └─┬─┘
      │    │   │   │  ┌────┐  │   │   │    │
      ▼    │   │   │  │    │  │   │   │    ▼
   ┌───┐   │   │   │  │C4  │  │   │   │ ┌───┐
   │P1 │   │   │   │  │    │  │   │   │ │U1 │
   └─┬─┘   │   │   │  └────┘  │   │   │ └─┬─┘
     │     │   │   │    │      │   │   │   │
     ▼     │   │   │    ▼      │   │   │   ▼
   ┌───┐   │   │   │  ┌────┐  │   │   │ ┌───┐
   │C2 │   │   │   │  │C5  │  │   │   │ │U2 │
   └─┬─┘   │   │   │  │    │  │   │   │ └─┬─┘
     │     │   │   │  └────┘  │   │   │   │
     ▼     │   │   │    │      │   │   │   ▼
   ┌───┐   │   │   │    ▼      │   │   │ ┌───┐
   │P2 │   │   │   │  ┌────┐  │   │   │ │U3 │
   └─┬─┘   │   │   │  │C6  │  │   │   │ └─┬─┘
     │     │   │   │  │    │  │   │   │   │
     ▼     │   │   │  └────┘  │   │   │   ▼
   ┌───┐   │   │   │    │      │   │   │ ┌───┐
   │C3 │   │   │   │    ▼      │   │   │ │U4 │
   └─┬─┘   │   │   │  ┌────┐  │   │   │ └─┬─┘
     │     │   │   │  │C7  │  │   │   │   │
     ▼     │   │   │  │    │  │   │   │   ▼
   ┌───┐   │   │   │  └────┘  │   │   │ ┌───┐
   │P3 │   │   │   │    │      │   │   │ │U5 │
   └─┬─┘   │   │   │    ▼      │   │   │ └─┬─┘
     │     │   │   │  ┌────┐  │   │   │   │
     ▼     │   │   │  │C8  │  │   │   │   ▼
   ┌───┐   │   │   │  │    │  │   │   │ ┌───┐
   │C4 │   │   │   │  └────┘  │   │   │ │U6 │
   └─┬─┘   │   │   │    │      │   │   │ └─┬─┘
     │     │   │   │    ▼      │   │   │   │
     ▼     │   │   │  ┌────┐  │   │   │   ▼
   ┌───┐   │   │   │  │C9  │  │   │   │ ┌───┐
   │P4 │   │   │   │  │    │  │   │   │ │U7 │
   └─┬─┘   │   │   │  └────┘  │   │   │ └─┬─┘
     │     │   │   │    │      │   │   │   │
     ▼     │   │   │    ▼      │   │   │   ▼
   ┌───┐   │   │   │  ┌────┐  │   │   │ ┌───┐
   │C5 │   │   │   │  │C10 │  │   │   │ │U8 │
   └───┘   │   │   │  │    │  │   │   │ └───┘
           │   │   │  └────┘  │   │   │
           │   │   │    │      │   │   │
           │   │   └────┼──────┘   │   │
           │   │        │          │   │
           │   └────────┼──────────┘   │
           │            │              │
           └────────────┼──────────────┘
                        │
```

**Note:** C = Convolution, P = Pooling, U = Up-Convolution, D = Output

## 3.2 U-Net Architecture - Layer by Layer

### The Complete Architecture

```python
# ============ U-NET COMPLETE ARCHITECTURE ============

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    Double Convolution Block
    
    WHAT IT DOES:
    - Two convolution layers with ReLU activations
    - Each convolution has 3×3 kernel size
    - Padding=1 to keep size same
    
    WHY TWO CONVOLUTIONS?
    - First conv: Extract features
    - Second conv: Extract more complex features
    - Better than single conv for learning patterns
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        
        # Two sequential convolutions
        self.conv = nn.Sequential(
            # Conv 1
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Helps training
            nn.ReLU(inplace=True),         # Nonlinearity
            
            # Conv 2
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class DownBlock(nn.Module):
    """
    Down Block - Encoder Path
    
    WHAT IT DOES:
    1. Double convolution (feature extraction)
    2. Max pooling (reduce size by half)
    
    WHY MAX POOLING?
    - Reduces image size (makes computation faster)
    - Increases receptive field (sees more context)
    - Makes model robust to small translations
    """
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # Store features for skip connection
        conv_out = self.double_conv(x)
        pooled = self.pool(conv_out)
        return conv_out, pooled

class UpBlock(nn.Module):
    """
    Up Block - Decoder Path
    
    WHAT IT DOES:
    1. Upsample (increase size by 2×)
    2. Concatenate with skip connection features
    3. Double convolution
    
    WHY UPSAMPLING?
    - Restores original image size
    - Reconstructs spatial information
    
    WHY SKIP CONNECTIONS?
    - Gives access to high-resolution features
    - Helps recover fine details
    - Makes segmentation more accurate
    """
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        
        # Upsample: Transposed convolution
        self.up = nn.ConvTranspose2d(
            in_channels,           # Input channels
            in_channels // 2,      # Output channels (half)
            kernel_size=2, 
            stride=2
        )
        
        # After concatenation:
        # in_channels // 2 (from up) + in_channels // 2 (from skip)
        # = in_channels (total)
        self.double_conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        x1: Features from decoder (up path)
        x2: Features from encoder (skip connection)
        """
        # Upsample x1
        x1 = self.up(x1)
        
        # Handle size mismatch (due to cropping)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        
        # Concatenate: (batch, channels, height, width)
        x = torch.cat([x2, x1], dim=1)
        
        # Double convolution
        x = self.double_conv(x)
        return x

class UNet(nn.Module):
    """
    Complete U-Net Architecture
    
    STRUCTURE:
    ┌─────────────────────────────────────────────┐
    │ INPUT (3, 572, 572)                        │
    │    ↓                                       │
    │ ENCODER PATH (Contraction)                │
    │    ↓                                       │
    │    DoubleConv (3 → 64)                    │
    │    ↓                                       │
    │    MaxPool (64 → 64, size half)           │
    │    ↓                                       │
    │    DoubleConv (64 → 128)                  │
    │    ↓                                       │
    │    MaxPool (128 → 128, size half)         │
    │    ↓                                       │
    │    DoubleConv (128 → 256)                 │
    │    ↓                                       │
    │    MaxPool (256 → 256, size half)         │
    │    ↓                                       │
    │    DoubleConv (256 → 512)                 │
    │    ↓                                       │
    │    MaxPool (512 → 512, size half)         │
    │    ↓                                       │
    │ BOTTLENECK                                  │
    │    ↓                                       │
    │    DoubleConv (512 → 1024)                │
    │    ↓                                       │
    │ DECODER PATH (Expansion)                  │
    │    ↓                                       │
    │    UpBlock (1024 → 512) + skip (512)      │
    │    ↓                                       │
    │    UpBlock (512 → 256) + skip (256)       │
    │    ↓                                       │
    │    UpBlock (256 → 128) + skip (128)       │
    │    ↓                                       │
    │    UpBlock (128 → 64) + skip (64)         │
    │    ↓                                       │
    │ OUTPUT (1, 388, 388)                      │
    └─────────────────────────────────────────────┘
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (1 for binary)
            features: Number of features at each level
        """
        super(UNet, self).__init__()
        
        # ============ ENCODER (Contraction Path) ============
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # First block: in_channels → features[0]
        self.downs.append(DoubleConv(in_channels, features[0]))
        
        # Subsequent blocks: features[i-1] → features[i]
        for i in range(1, len(features)):
            self.downs.append(DoubleConv(features[i-1], features[i]))
        
        # ============ BOTTLENECK ============
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # ============ DECODER (Expansion Path) ============
        self.ups = nn.ModuleList()
        
        # Up blocks: features[i] → features[i-1]
        for i in range(len(features)-1, 0, -1):
            self.ups.append(UpBlock(features[i] * 2, features[i-1]))
        
        # Last up block: features[0] * 2 → features[0]
        self.ups.append(UpBlock(features[0] * 2, features[0]))
        
        # ============ FINAL CONVOLUTION ============
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through U-Net
        
        Returns:
            Segmentation mask with same spatial dimensions as input
        """
        
        # ============ ENCODER PATH ============
        skip_connections = []
        
        # Store features and downsample
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # ============ BOTTLENECK ============
        x = self.bottleneck(x)
        
        # ============ DECODER PATH ============
        skip_connections = skip_connections[::-1]  # Reverse for skip connections
        
        for idx, up in enumerate(self.ups):
            x = up(x, skip_connections[idx])
        
        # ============ FINAL OUTPUT ============
        x = self.final_conv(x)
        
        return x

# ============ TEST THE ARCHITECTURE ============

def test_unet():
    """Print U-Net architecture details"""
    
    print("="*70)
    print("U-NET ARCHITECTURE EXPLAINED")
    print("="*70)
    
    # Create U-Net
    model = UNet(in_channels=3, out_channels=1)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 572, 572)
    
    # Forward pass
    output = model(dummy_input)
    
    print("\n📐 INPUT SHAPE:")
    print(f"   {dummy_input.shape}")
    print("   ↓")
    print("   ║")
    print("   ╠═══════════════════════════════════════════════╣")
    print("   ║  ENCODER PATH (Contraction)                   ║")
    print("   ║  - DoubleConv: Extract features               ║")
    print("   ║  - MaxPool: Reduce size by half              ║")
    print("   ╠═══════════════════════════════════════════════╣")
    print("   ║  BOTTLENECK                                   ║")
    print("   ║  - DoubleConv with max channels              ║")
    print("   ╠═══════════════════════════════════════════════╣")
    print("   ║  DECODER PATH (Expansion)                    ║")
    print("   ║  - UpConv: Increase size by 2×               ║")
    print("   ║  - Concatenate with skip connections         ║")
    print("   ║  - DoubleConv: Refine features               ║")
    print("   ╠═══════════════════════════════════════════════╣")
    print("   ║  FINAL CONV: 1×1 to output channels          ║")
    print("   ↓")
    print(f"\n📐 OUTPUT SHAPE:")
    print(f"   {output.shape}")
    
    print("\n" + "="*70)
    print("KEY FEATURES:")
    print("="*70)
    print("1. 🏗️  Encoder-Decoder Structure")
    print("   - Symmetrical U-shape")
    print("   - Captures context and location")
    print()
    print("2. 🔗  Skip Connections")
    print("   - Connects encoder to decoder")
    print("   - Preserves fine details")
    print("   - Crucial for accurate segmentation")
    print()
    print("3. 🎯  Pixel-wise Output")
    print("   - Output has same size as input")
    print("   - Each pixel gets a label")
    print()
    print("4. 💪  Designed for Medical Images")
    print("   - Works with limited data")
    print("   - Handles variable input sizes")
    print("   - Accurate segmentation")

test_unet()
```

## 3.3 Understanding Each Component in Detail

### 1. Double Convolution Block

```python
# ============ DOUBLE CONVOLUTION EXPLAINED ============

class DoubleConvDetailed(nn.Module):
    """
    Double Convolution Block - EVERYTHING YOU NEED TO KNOW
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConvDetailed, self).__init__()
        
        # ============ LAYER 1 ============
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,   # Input channels
            out_channels=out_channels, # Output channels
            kernel_size=3,             # 3×3 filter
            stride=1,                  # Move 1 pixel at a time
            padding=1                  # Keep size same
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # ============ LAYER 2 ============
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        print(f"DoubleConv: {in_channels} → {out_channels}")
        print(f"  Total parameters: {(in_channels*3*3*out_channels) + out_channels + "
              f"(out_channels*3*3*out_channels) + out_channels:,}")
    
    def forward(self, x):
        # First layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Second layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x

# Let's calculate parameters
def calculate_doubleconv_params():
    """Show parameter calculation"""
    print("="*60)
    print("DOUBLE CONV PARAMETER CALCULATION")
    print("="*60)
    
    in_ch = 64
    out_ch = 128
    
    # Conv1: in_ch → out_ch
    params_conv1 = (in_ch * 3 * 3 * out_ch) + out_ch  # weights + bias
    print(f"Conv1: {in_ch}→{out_ch}: {params_conv1:,} params")
    
    # Conv2: out_ch → out_ch
    params_conv2 = (out_ch * 3 * 3 * out_ch) + out_ch
    print(f"Conv2: {out_ch}→{out_ch}: {params_conv2:,} params")
    
    # Total
    total = params_conv1 + params_conv2
    print(f"Total: {total:,} params")
    print("\n💡 Note: 3×3 convolution with {out_ch} channels")
    print(f"   = {out_ch} filters × {in_ch} input channels × 3×3 kernel")

calculate_doubleconv_params()
```

### 2. Skip Connections - The Magic of U-Net

```python
# ============ SKIP CONNECTIONS EXPLAINED ============

def explain_skip_connections():
    """
    Why are skip connections important?
    
    PROBLEM WITHOUT SKIP CONNECTIONS:
    ┌────────────────────────────────────────────────────┐
    │ Input → Conv → Conv → Pool → Conv → Conv → Pool  │
    │                                                  │
    │ After pooling: Lost fine details!               │
    │ ┌────────────────────────────────────────┐      │
    │ │ 🐱 (blurry, no whiskers!)             │      │
    │ └────────────────────────────────────────┘      │
    │                                                  │
    │ Conv → Conv → UpConv → UpConv → Output         │
    │                                                  │
    │ Output: Blurry edges, missing details           │
    └────────────────────────────────────────────────────┘
    
    SOLUTION WITH SKIP CONNECTIONS:
    ┌────────────────────────────────────────────────────┐
    │ Input → Conv → Conv → Pool                       │
    │    │                   │                         │
    │    └─── Skip ──────────┤                         │
    │                        ↓                         │
    │                   UpConv → Conv → Conv → Output │
    │                                                  │
    │ Output: Sharp edges, all details preserved!     │
    └────────────────────────────────────────────────────┘
    """
    
    print("="*60)
    print("SKIP CONNECTIONS - WHY THEY'RE AMAZING")
    print("="*60)
    print()
    print("🔍 WHAT SKIP CONNECTIONS DO:")
    print("  1. Connect encoder layer to corresponding decoder layer")
    print("  2. Pass high-resolution features directly")
    print("  3. Preserve spatial information")
    print()
    print("🎯 BENEFITS:")
    print("  1. Better localization (knows where things are)")
    print("  2. Sharper boundaries (fine details preserved)")
    print("  3. Easier to train (gradients flow better)")
    print()
    print("📊 VISUALIZATION:")
    print("  ┌─────────────────────────────────────────────┐")
    print("  │ Encoder → Pool → 64×64   │  Decoder ← Up  │")
    print("  │  (128×128)    │           │    (128×128)   │")
    print("  │     ───────────┼───────────┤               │")
    print("  │                │           │                │")
    print("  │  Encoder → 32×32  │  Decoder ← Up         │")
    print("  │  (64×64)        │     (64×64)            │")
    print("  │     ────────────┼───────────┤             │")
    print("  │                 │           │              │")
    print("  │  Encoder → 16×16    │  Decoder ← Up      │")
    print("  │  (32×32)          │     (32×32)         │")
    print("  └─────────────────────────────────────────────┘")

explain_skip_connections()
```

### 3. Upsampling (Transposed Convolution)

```python
# ============ UPSAMPLING EXPLAINED ============

class UpsampleDetailed(nn.Module):
    """
    Transposed Convolution - Increase Image Size
    """
    def __init__(self, in_channels, out_channels):
        super(UpsampleDetailed, self).__init__()
        
        # ============ TRANS CONV ============
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )
        """
        HOW TRANS CONV WORKS:
        
        Input (2×2):        Kernel (2×2):
        ┌─────────┐         ┌─────────┐
        │ a    b  │         │ w1  w2  │
        │ c    d  │         │ w3  w4  │
        └─────────┘         └─────────┘
        
        Output (4×4):
        ┌─────────────────────────┐
        │ a*w1  a*w2  b*w1  b*w2 │
        │ a*w3  a*w4  b*w3  b*w4 │
        │ c*w1  c*w2  d*w1  d*w2 │
        │ c*w3  c*w4  d*w3  d*w4 │
        └─────────────────────────┘
        
        Each input pixel becomes a 2×2 block!
        """
    
    def forward(self, x):
        return self.up(x)

def visualize_upsampling():
    """Show how upsampling works"""
    import numpy as np
    
    # Simple 2×2 input
    input_2x2 = np.array([[1, 2],
                          [3, 4]])
    
    print("="*60)
    print("UPSAMPLING VISUALIZATION")
    print("="*60)
    print("Input (2×2):")
    print(input_2x2)
    print()
    print("After upsampling (4×4):")
    print("┌─────────────────────────┐")
    print("│ 1   1   2   2          │")
    print("│ 1   1   2   2          │")
    print("│ 3   3   4   4          │")
    print("│ 3   3   4   4          │")
    print("└─────────────────────────┘")
    print()
    print("💡 Each 2×2 block comes from one input pixel")

visualize_upsampling()
```

## 3.4 U-Net Variants

```python
# ============ U-NET VARIANTS ============

"""
1. ORIGINAL U-NET (2015)
- 23 convolutional layers
- Designed for medical images
- Works with small datasets

2. RESIDUAL U-NET
- Adds residual connections
- Easier to train deeper
- Better gradient flow

3. ATTENTION U-NET
- Adds attention mechanisms
- Focuses on important regions
- Better for complex images

4. 3D U-NET
- Processes 3D volumes (MRI, CT)
- Uses 3D convolutions
- Medical imaging

5. U-NET++ (Nested U-Net)
- Dense skip connections
- More flexible architecture
- Better performance

6. TRANSFORMER U-NET
- Combines CNN with Transformers
- Captures global context
- State-of-the-art performance
"""

# ============ RESIDUAL U-NET ============

class ResidualBlock(nn.Module):
    """
    Residual Block for UNet
    Adds skip connection within each block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection for shortcut
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + self.shortcut(residual)
        x = self.relu(x)
        return x

# ============ ATTENTION U-NET ============

class AttentionGate(nn.Module):
    """
    Attention Gate - Helps focus on important regions
    
    How it works:
    1. Takes features from encoder and decoder
    2. Learns which regions are important
    3. Applies weights to focus on important regions
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # For encoder features
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # For decoder features
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Attention weights
        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, g, x):
        # g: Features from decoder
        # x: Features from encoder (skip connection)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Combine and compute attention
        psi = self.psi(torch.relu(g1 + x1))
        
        # Apply attention
        return x * psi
```

---

# 4. OPENCV - YOUR SWISS ARMY KNIFE FOR IMAGES

## 4.1 What is OpenCV and Why Do You Need It?

**OpenCV** (Open Source Computer Vision Library) is like a Swiss Army knife for images. It has hundreds of tools for:
- Reading/writing images
- Resizing, cropping, rotating
- Drawing shapes and text
- Image filtering and enhancement
- Feature detection
- And much more!

```python
# ============ OPENCV INTRODUCTION ============

import cv2
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("OPENCV - THE COMPUTER VISION SWISS ARMY KNIFE")
print("="*70)

print("\n📚 WHAT OPENCV CAN DO:")
print("  1. Read and write images (JPEG, PNG, etc.)")
print("  2. Resize, crop, rotate images")
print("  3. Convert color spaces (RGB, Grayscale, HSV)")
print("  4. Apply filters (blur, sharpen, edge detection)")
print("  5. Draw shapes and text on images")
print("  6. Detect edges, corners, blobs")
print("  7. Find contours and boundaries")
print("  8. Match templates and patterns")
print("  9. Video processing and motion detection")
print(" 10. Camera calibration and 3D reconstruction")
```

## 4.2 Basic OpenCV Operations

### Reading and Writing Images

```python
# ============ READING AND WRITING IMAGES ============

def opencv_basic_io():
    """
    How to read, display, and save images with OpenCV
    """
    
    print("="*60)
    print("OPENCV - BASIC I/O OPERATIONS")
    print("="*60)
    
    # ============ READ IMAGE ============
    # cv2.imread() reads an image from file
    # Returns: numpy array of shape (height, width, channels)
    image = cv2.imread('sample_image.jpg')
    # Note: OpenCV reads images in BGR format (not RGB!)
    
    if image is None:
        print("❌ Image not found! Creating a dummy image...")
        # Create a dummy image for demonstration
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Draw a rectangle
        cv2.rectangle(image, (50, 50), (150, 150), (0, 255, 0), 2)
        cv2.putText(image, 'OpenCV', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2)
    
    print(f"📐 Image shape: {image.shape}")  # (height, width, channels)
    print(f"💾 Image dtype: {image.dtype}")
    print(f"📊 Min value: {image.min()}, Max value: {image.max()}")
    
    # ============ DISPLAY IMAGE ============
    # cv2.imshow() displays an image in a window
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)  # Wait for key press
    cv2.destroyAllWindows()  # Close all windows
    
    # ============ SAVE IMAGE ============
    cv2.imwrite('output_image.jpg', image)
    print("💾 Image saved as 'output_image.jpg'")
    
    return image

# ============ COLOR SPACE CONVERSIONS ============

def color_space_conversions():
    """
    Convert between different color spaces
    """
    
    # Create a test image
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[50:150, 50:150] = [0, 255, 0]  # Green square (BGR format)
    
    print("="*60)
    print("COLOR SPACE CONVERSIONS")
    print("="*60)
    
    # ============ BGR TO RGB ============
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("✅ BGR → RGB (for matplotlib)")
    
    # ============ BGR TO GRAYSCALE ============
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("✅ BGR → Grayscale")
    
    # ============ BGR TO HSV ============
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    print("✅ BGR → HSV (Hue, Saturation, Value)")
    
    # ============ BGR TO LAB ============
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    print("✅ BGR → LAB (Lightness, A, B)")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gray_image, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(hsv_image)
    axes[1, 0].set_title('HSV Image')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(lab_image)
    axes[1, 1].set_title('LAB Image')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return rgb_image, gray_image, hsv_image, lab_image
```

### Image Transformations

```python
# ============ IMAGE TRANSFORMATIONS ============

def image_transformations():
    """
    Resize, crop, rotate, flip images
    """
    
    # Create a test image
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (250, 250), (0, 255, 0), -1)  # Green square
    cv2.circle(image, (150, 150), 50, (0, 0, 255), -1)  # Red circle
    
    print("="*60)
    print("IMAGE TRANSFORMATIONS")
    print("="*60)
    
    # ============ RESIZE ============
    # cv2.resize() changes image size
    # Interpolation methods:
    # - cv2.INTER_LINEAR: Bilinear (good for zoom)
    # - cv2.INTER_NEAREST: Nearest neighbor (fast)
    # - cv2.INTER_CUBIC: Bicubic (best quality)
    # - cv2.INTER_AREA: Area-based (good for shrinking)
    
    resized_linear = cv2.resize(image, (150, 150), interpolation=cv2.INTER_LINEAR)
    resized_cubic = cv2.resize(image, (150, 150), interpolation=cv2.INTER_CUBIC)
    resized_area = cv2.resize(image, (150, 150), interpolation=cv2.INTER_AREA)
    
    print("✅ Resized to 150×150")
    
    # ============ CROP ============
    # Cropping is just array indexing
    cropped = image[50:250, 50:250]  # [y_start:y_end, x_start:x_end]
    print("✅ Cropped region (50:250, 50:250)")
    
    # ============ ROTATE ============
    # Need rotation matrix
    height, width = image.shape[:2]
    center = (width//2, height//2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)  # 45 degrees
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    print("✅ Rotated 45 degrees")
    
    # ============ FLIP ============
    # cv2.flip() flips the image
    flip_horizontal = cv2.flip(image, 1)  # 1 = horizontal
    flip_vertical = cv2.flip(image, 0)    # 0 = vertical
    flip_both = cv2.flip(image, -1)       # -1 = both
    print("✅ Flipped")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(resized_linear, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Resized (Linear)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Cropped')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Rotated 45°')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(flip_horizontal, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Flipped Horizontal')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cv2.cvtColor(flip_vertical, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Flipped Vertical')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
```

### Drawing on Images

```python
# ============ DRAWING ON IMAGES ============

def draw_on_images():
    """
    Draw shapes, text, and annotations
    """
    
    # Create blank image
    image = np.ones((400, 400, 3), dtype=np.uint8) * 255  # White background
    
    print("="*60)
    print("DRAWING ON IMAGES")
    print("="*60)
    
    # ============ DRAW LINES ============
    # cv2.line(image, start_point, end_point, color, thickness)
    cv2.line(image, (50, 50), (350, 50), (0, 0, 255), 3)  # Red line
    
    # ============ DRAW RECTANGLES ============
    # cv2.rectangle(image, top_left, bottom_right, color, thickness)
    cv2.rectangle(image, (50, 100), (350, 200), (0, 255, 0), 2)  # Green rectangle
    
    # ============ DRAW CIRCLES ============
    # cv2.circle(image, center, radius, color, thickness)
    cv2.circle(image, (200, 300), 50, (255, 0, 0), -1)  # Blue filled circle
    
    # ============ DRAW ELLIPSES ============
    # cv2.ellipse(image, center, axes, angle, start_angle, end_angle, color, thickness)
    cv2.ellipse(image, (100, 300), (50, 25), 0, 0, 360, (0, 255, 255), 2)
    
    # ============ DRAW POLYGONS ============
    pts = np.array([[300, 100], [350, 100], [375, 150], [325, 150]], np.int32)
    cv2.polylines(image, [pts], True, (255, 0, 255), 2)
    
    # ============ PUT TEXT ============
    # cv2.putText(image, text, position, font, scale, color, thickness)
    cv2.putText(image, 'OpenCV Drawing', (100, 380), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    print("✅ Drew various shapes on image")
    
    # Display
    cv2.imshow('Drawing', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return image
```

## 4.3 Image Filtering and Enhancement

```python
# ============ IMAGE FILTERING ============

def image_filtering():
    """
    Apply various filters to images
    """
    
    # Create a test image with noise
    image = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (150, 150), 255, -1)  # White square
    
    # Add salt and pepper noise
    noise = np.random.random(image.shape)
    image[noise < 0.05] = 0
    image[noise > 0.95] = 255
    
    print("="*60)
    print("IMAGE FILTERING")
    print("="*60)
    
    # ============ GAUSSIAN BLUR ============
    # Smooths image, reduces noise
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    print("✅ Gaussian Blur applied")
    
    # ============ MEDIAN BLUR ============
    # Better for salt and pepper noise
    median = cv2.medianBlur(image, 5)
    print("✅ Median Blur applied")
    
    # ============ BILATERAL FILTER ============
    # Preserves edges while smoothing
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    print("✅ Bilateral Filter applied")
    
    # ============ SHARPENING ============
    # Kernel for sharpening
    kernel = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    print("✅ Sharpening applied")
    
    # ============ EDGE DETECTION ============
    # Canny edge detection
    edges = cv2.Canny(image, 100, 200)
    print("✅ Edge detection applied")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original (with noise)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gaussian, cmap='gray')
    axes[0, 1].set_title('Gaussian Blur')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(median, cmap='gray')
    axes[0, 2].set_title('Median Blur')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(bilateral, cmap='gray')
    axes[1, 0].set_title('Bilateral Filter')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(sharpened, cmap='gray')
    axes[1, 1].set_title('Sharpened')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(edges, cmap='gray')
    axes[1, 2].set_title('Edges (Canny)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return gaussian, median, bilateral, sharpened, edges
```

## 4.4 Contour Detection

```python
# ============ CONTOUR DETECTION ============

def contour_detection():
    """
    Find and draw contours in images
    """
    
    # Create a test image
    image = np.zeros((300, 300), dtype=np.uint8)
    
    # Draw some shapes
    cv2.rectangle(image, (50, 50), (100, 100), 255, -1)
    cv2.circle(image, (200, 200), 40, 255, -1)
    cv2.ellipse(image, (200, 80), (50, 30), 0, 0, 360, 255, -1)
    
    print("="*60)
    print("CONTOUR DETECTION")
    print("="*60)
    
    # ============ FIND CONTOURS ============
    # cv2.findContours() finds boundaries of objects
    # Returns: contours (list of points) and hierarchy
    contours, hierarchy = cv2.findContours(
        image, 
        cv2.RETR_EXTERNAL,  # Only external contours
        cv2.CHAIN_APPROX_SIMPLE  # Compress contour points
    )
    
    print(f"✅ Found {len(contours)} contours")
    
    # ============ DRAW CONTOURS ============
    # Create color image for visualization
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Draw contours
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    
    # ============ CONTOUR PROPERTIES ============
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        print(f"  Contour {i+1}: Area={area:.0f}, Perimeter={perimeter:.1f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Contours Detected')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return contours
```

---

# 5. IMAGE PREPROCESSING - CLEANING YOUR DATA

## 5.1 Why Preprocessing is Important

**Analogy:** You can't bake a cake with dirty ingredients. Similarly, you can't train a model with messy images.

```python
# ============ WHY PREPROCESSING ============

print("="*70)
print("WHY IMAGE PREPROCESSING IS CRUCIAL")
print("="*70)

print("""
🔴 PROBLEMS IN RAW IMAGES:
   1. Different sizes (model expects fixed size)
   2. Different brightness/contrast (inconsistent data)
   3. Noise and artifacts (distractions)
   4. Unwanted background (irrelevant information)
   5. Color variations (lighting differences)

✅ WHAT PREPROCESSING DOES:
   1. Standardizes images (same size, format)
   2. Enhances important features
   3. Removes noise and artifacts
   4. Normalizes pixel values
   5. Makes model training more stable

📊 BEFORE AND AFTER:
   Before: [Random sizes, different brightness, noise, uneven lighting]
   After:  [Same size, consistent brightness, clean, normalized]
""")
```

## 5.2 Complete Preprocessing Pipeline

```python
# ============ COMPLETE PREPROCESSING PIPELINE ============

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

class ImagePreprocessor:
    """
    Complete image preprocessing pipeline for medical images
    """
    
    def __init__(self):
        self.operations = []
    
    def load_image(self, image_path):
        """
        Load image from file
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return image
    
    def resize_image(self, image, target_size=(256, 256)):
        """
        Resize image to target size
        """
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    def convert_to_grayscale(self, image):
        """
        Convert to grayscale if not already
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def normalize_pixels(self, image):
        """
        Normalize pixel values to [0, 1]
        """
        return image.astype(np.float32) / 255.0
    
    def standardize(self, image):
        """
        Standardize to mean=0, std=1
        """
        mean = np.mean(image)
        std = np.std(image)
        return (image - mean) / (std + 1e-8)
    
    def enhance_contrast(self, image):
        """
        Enhance contrast using CLAHE
        """
        if len(image.shape) == 3:
            # Convert to LAB
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge back
            lab = cv2.merge((l, a, b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def remove_noise(self, image):
        """
        Remove noise using median filter
        """
        if len(image.shape) == 3:
            # Apply to each channel
            result = np.zeros_like(image)
            for i in range(3):
                result[:, :, i] = cv2.medianBlur(image[:, :, i], 3)
            return result
        else:
            return cv2.medianBlur(image, 3)
    
    def remove_background(self, image, threshold=0.5):
        """
        Simple background removal using thresholding
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Threshold
        _, mask = cv2.threshold(gray, int(threshold * 255), 255, cv2.THRESH_BINARY)
        
        # Apply mask
        if len(image.shape) == 3:
            result = image.copy()
            result[mask == 0] = 0
            return result
        else:
            return cv2.bitwise_and(image, mask)
    
    def apply_gaussian_blur(self, image, kernel_size=(5, 5)):
        """
        Apply Gaussian blur
        """
        return cv2.GaussianBlur(image, kernel_size, 0)
    
    def detect_edges(self, image):
        """
        Detect edges using Canny
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return cv2.Canny(gray, 50, 150)
    
    def morphological_operations(self, image, operation='open', kernel_size=3):
        """
        Apply morphological operations
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if operation == 'open':
            # Remove small objects
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'close':
            # Fill small holes
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operation == 'dilate':
            # Enlarge objects
            return cv2.dilate(image, kernel, iterations=1)
        elif operation == 'erode':
            # Shrink objects
            return cv2.erode(image, kernel, iterations=1)
    
    def preprocess_pipeline(self, image_path, visualize=True):
        """
        Complete preprocessing pipeline
        """
        print("="*60)
        print("IMAGE PREPROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Load image
        print("1. Loading image...")
        image = self.load_image(image_path)
        results = [('Original', image.copy())]
        
        # Step 2: Resize
        print("2. Resizing to 256×256...")
        image = self.resize_image(image, (256, 256))
        results.append(('Resized', image.copy()))
        
        # Step 3: Convert to grayscale
        print("3. Converting to grayscale...")
        image = self.convert_to_grayscale(image)
        results.append(('Grayscale', image.copy()))
        
        # Step 4: Remove noise
        print("4. Removing noise...")
        image = self.remove_noise(image)
        results.append(('Denoised', image.copy()))
        
        # Step 5: Enhance contrast
        print("5. Enhancing contrast...")
        image = self.enhance_contrast(image)
        results.append(('Enhanced Contrast', image.copy()))
        
        # Step 6: Normalize
        print("6. Normalizing pixels...")
        image = self.normalize_pixels(image)
        results.append(('Normalized', image.copy()))
        
        # Step 7: Standardize
        print("7. Standardizing...")
        image = self.standardize(image)
        results.append(('Standardized', image.copy()))
        
        # Step 8: Edge detection (for visualization)
        print("8. Detecting edges...")
        edges = self.detect_edges(image)
        results.append(('Edges', edges))
        
        print("✅ Preprocessing complete!")
        
        if visualize:
            self.visualize_pipeline(results)
        
        return image
    
    def visualize_pipeline(self, results):
        """
        Visualize each step of the pipeline
        """
        n = len(results)
        cols = 3
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        axes = axes.flatten()
        
        for i, (name, img) in enumerate(results):
            if len(img.shape) == 3:
                axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                axes[i].imshow(img, cmap='gray')
            axes[i].set_title(name, fontsize=12)
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

# ============ EXAMPLE USAGE ============

def demonstrate_preprocessing():
    """
    Demonstrate complete preprocessing
    """
    # Create a dummy image with artifacts
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Add a shape with noise
    cv2.rectangle(image, (50, 50), (250, 250), (0, 255, 0), -1)
    
    # Add salt and pepper noise
    noise = np.random.random((300, 300))
    image[noise < 0.05] = 0
    image[noise > 0.95] = 255
    
    # Add random brightness variation
    brightness = np.random.randint(-50, 50, (300, 300, 3))
    image = np.clip(image.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
    
    # Save temporary image
    cv2.imwrite('temp_image.jpg', image)
    
    # Preprocess
    preprocessor = ImagePreprocessor()
    processed = preprocessor.preprocess_pipeline('temp_image.jpg', visualize=True)
    
    return processed

# Run the demonstration
demonstrate_preprocessing()
```

## 5.3 Advanced Preprocessing Techniques

```python
# ============ ADVANCED PREPROCESSING ============

class AdvancedPreprocessor:
    """
    Advanced preprocessing techniques for medical images
    """
    
    def histogram_equalization(self, image):
        """
        Global histogram equalization
        """
        if len(image.shape) == 3:
            # Convert to YUV
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            return cv2.equalizeHist(image)
    
    def adaptive_thresholding(self, image):
        """
        Adaptive thresholding for varying lighting
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return binary
    
    def remove_shadows(self, image):
        """
        Remove shadows using morphological operations
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Create shadow mask
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Shadow regions
        shadow_mask = cv2.subtract(gray, opening)
        _, shadow_mask = cv2.threshold(shadow_mask, 20, 255, cv2.THRESH_BINARY)
        
        # Remove shadows
        result = gray.copy()
        result[shadow_mask > 0] = opening[shadow_mask > 0]
        
        return result
    
    def unsharp_masking(self, image, sigma=1.0, amount=1.5):
        """
        Unsharp masking for sharpening
        """
        # Blur the image
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        
        # Apply unsharp mask
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def denoise_nlm(self, image, h=10):
        """
        Non-local means denoising
        """
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, h, 7, 21)
```

---

# 6. MEDICAL IMAGE SEGMENTATION - REAL-WORLD APPLICATION

## 6.1 Understanding Medical Image Segmentation

```python
# ============ MEDICAL IMAGE SEGMENTATION ============

def medical_segmentation_intro():
    """
    Introduction to medical image segmentation
    """
    
    print("="*70)
    print("MEDICAL IMAGE SEGMENTATION")
    print("="*70)
    
    print("""
    🏥 WHAT IS MEDICAL IMAGE SEGMENTATION?
    ========================================
    - Identifying and outlining anatomical structures
    - Separating organs, tissues, or abnormalities
    - Pixel-level classification in medical scans
    
    📋 COMMON APPLICATION AREAS:
    =============================
    1. Brain MRI Segmentation
       - Separate gray matter, white matter, CSF
       - Detect tumors and lesions
       - Plan surgery
    
    2. CT Scan Segmentation
       - Identify organs (liver, kidneys, lungs)
       - Detect fractures and abnormalities
       - Radiation therapy planning
    
    3. Cell Microscopy Segmentation
       - Detect and count cells
       - Identify cell nuclei
       - Cancer detection
    
    4. X-ray Segmentation
       - Bones and joints
       - Dental imaging
       - Chest X-ray analysis
    
    5. Retinal Imaging
       - Blood vessel segmentation
       - Diabetic retinopathy detection
       - Glaucoma screening
    
    🔬 WHY IS IT CHALLENGING?
    =========================
    1. Low contrast between tissues
    2. Variability in anatomy
    3. Noise and artifacts
    4. Limited labeled data
    5. High resolution (requires memory)
    6. 3D data (volumetric analysis)
    """)

medical_segmentation_intro()
```

## 6.2 Medical Image Formats

```python
# ============ MEDICAL IMAGE FORMATS ============

def medical_image_formats():
    """
    Understanding medical image formats
    """
    
    print("="*60)
    print("MEDICAL IMAGE FORMATS")
    print("="*60)
    
    formats = {
        "DICOM": {
            "Full Name": "Digital Imaging and Communications in Medicine",
            "Extension": ".dcm, .dicom",
            "Description": "Standard for storing and transmitting medical images",
            "Features": "Contains patient info, metadata, and pixel data",
            "Common in": "MRI, CT, Ultrasound, X-ray",
            "Challenges": "Complex format, need specialized libraries"
        },
        "NIfTI": {
            "Full Name": "Neuroimaging Informatics Technology Initiative",
            "Extension": ".nii, .nii.gz",
            "Description": "Neuroimaging data format",
            "Features": "Combines image and header data",
            "Common in": "MRI, fMRI, PET",
            "Challenges": "Less widely supported"
        },
        "MHD": {
            "Full Name": "MetaImage",
            "Extension": ".mhd, .raw",
            "Description": "Medical image with separate header and data",
            "Features": "Header describes image properties",
            "Common in": "CT, MRI",
            "Challenges": "Two files to manage"
        },
        "PNG/JPG": {
            "Full Name": "Standard image formats",
            "Extension": ".png, .jpg",
            "Description": "Conventional image formats",
            "Features": "Widely supported, easy to handle",
            "Common in": "Processed images",
            "Challenges": "Loses medical metadata"
        }
    }
    
    for name, info in formats.items():
        print(f"\n📁 {name}")
        print(f"   Full Name: {info['Full Name']}")
        print(f"   Extension: {info['Extension']}")
        print(f"   Description: {info['Description']}")
        print(f"   Common in: {info['Common in']}")
        print(f"   Challenges: {info['Challenges']}")
    
    print("\n" + "="*60)
    print("💡 TIPS FOR WORKING WITH MEDICAL IMAGES:")
    print("  1. Use specialized libraries (pydicom, nibabel)")
    print("  2. Check orientation (RAS, LPS coordinates)")
    print("  3. Understand intensity units (HU for CT)")
    print("  4. Handle large volumes carefully")
    print("  5. Preserve spatial information")
```

## 6.3 Complete Medical Segmentation Project

```python
# ============ COMPLETE MEDICAL SEGMENTATION PROJECT ============

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# ============================================
# 1. DATASET CLASS FOR MEDICAL IMAGES
# ============================================

class MedicalSegmentationDataset(Dataset):
    """
    Dataset for medical image segmentation
    
    Assumes directory structure:
    data/
      images/
        image_001.png
        image_002.png
        ...
      masks/
        mask_001.png
        mask_002.png
        ...
    """
    
    def __init__(self, images_dir, masks_dir, transform=None, target_size=(256, 256)):
        """
        Args:
            images_dir: Directory containing input images
            masks_dir: Directory containing ground truth masks
            transform: Optional transforms
            target_size: Desired image size (height, width)
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_size = target_size
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(images_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Check if corresponding masks exist
        self.mask_files = []
        for img_file in self.image_files:
            mask_file = img_file  # Assuming same name
            if os.path.exists(os.path.join(masks_dir, mask_file)):
                self.mask_files.append(mask_file)
            else:
                print(f"⚠️  Mask not found for: {img_file}")
                self.mask_files.append(None)
        
        print(f"✅ Loaded {len(self.image_files)} image-mask pairs")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # (H, W, C) → (C, H, W)
        
        # Load mask if exists
        if self.mask_files[idx] is not None:
            mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)  # Binary mask
            mask = np.expand_dims(mask, axis=0)  # (H, W) → (1, H, W)
        else:
            # Dummy mask
            mask = np.zeros((1, *self.target_size), dtype=np.float32)
        
        return torch.from_numpy(image), torch.from_numpy(mask)

# ============================================
# 2. U-NET MODEL FOR MEDICAL SEGMENTATION
# ============================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNetMedical(nn.Module):
    """
    U-Net optimized for medical image segmentation
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNetMedical, self).__init__()
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        # First encoder
        self.encoders.append(DoubleConv(in_channels, features[0]))
        
        # Subsequent encoders
        for i in range(1, len(features)):
            self.encoders.append(DoubleConv(features[i-1], features[i]))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        for i in range(len(features)-1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(features[i] * 2, features[i], 
                                               kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(features[i] + features[i-1], features[i-1]))
        
        # Last decoder
        self.ups.append(nn.ConvTranspose2d(features[0] * 2, features[0], 
                                           kernel_size=2, stride=2))
        self.decoders.append(DoubleConv(features[0] + features[0], features[0]))
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Sigmoid for binary segmentation
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder path
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            skip_connections.append(x)
            if i < len(self.encoders) - 1:
                x = self.pools[i](x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        skip_connections = skip_connections[::-1]
        
        for i, (up, decoder) in enumerate(zip(self.ups, self.decoders)):
            x = up(x)
            
            # Handle size mismatch
            if x.shape[2:] != skip_connections[i].shape[2:]:
                diff_y = skip_connections[i].size(2) - x.size(2)
                diff_x = skip_connections[i].size(3) - x.size(3)
                x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                                          diff_y // 2, diff_y - diff_y // 2])
            
            x = torch.cat([skip_connections[i], x], dim=1)
            x = decoder(x)
        
        # Final convolution
        x = self.final_conv(x)
        x = self.sigmoid(x)
        
        return x

# ============================================
# 3. TRAINING FUNCTIONS
# ============================================

def dice_loss(pred, target):
    """
    Dice Loss for medical image segmentation
    """
    smooth = 1e-6
    
    # Flatten tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def combined_loss(pred, target):
    """
    Combined BCE + Dice Loss
    """
    bce_loss = nn.BCELoss()(pred, target)
    dice_loss_value = dice_loss(pred, target)
    return bce_loss + dice_loss_value

def train_epoch(model, dataloader, optimizer, device):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    total_dice = 0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        predictions = model(images)
        loss = combined_loss(predictions, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        
        # Calculate Dice score
        pred_binary = (predictions > 0.5).float()
        dice = 1 - dice_loss(pred_binary, masks)
        total_dice += dice.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    
    return avg_loss, avg_dice

def validate_epoch(model, dataloader, device):
    """
    Validate for one epoch
    """
    model.eval()
    total_loss = 0
    total_dice = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            predictions = model(images)
            loss = combined_loss(predictions, masks)
            
            total_loss += loss.item()
            
            pred_binary = (predictions > 0.5).float()
            dice = 1 - dice_loss(pred_binary, masks)
            total_dice += dice.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    
    return avg_loss, avg_dice

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-4, device='cuda'):
    """
    Complete training pipeline
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': []
    }
    
    best_dice = 0
    
    print("="*60)
    print("TRAINING MEDICAL SEGMENTATION MODEL")
    print("="*60)
    
    for epoch in range(epochs):
        # Train
        train_loss, train_dice = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss, val_dice = validate_epoch(model, val_loader, device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), 'best_medical_segmentation.pth')
            print(f"  ✅ Best model saved! (Dice: {best_dice:.4f})")
    
    print(f"\n✅ Training complete! Best Dice: {best_dice:.4f}")
    
    return model, history

# ============================================
# 4. EVALUATION AND VISUALIZATION
# ============================================

def evaluate_model(model, dataloader, device, num_samples=5):
    """
    Evaluate model and visualize results
    """
    model.eval()
    
    # Get a batch of data
    images, masks = next(iter(dataloader))
    images = images[:num_samples].to(device)
    masks = masks[:num_samples]
    
    with torch.no_grad():
        predictions = model(images)
        predictions = (predictions > 0.5).float()
    
    # Move to CPU for visualization
    images = images.cpu().numpy()
    masks = masks.numpy()
    predictions = predictions.cpu().numpy()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    for i in range(num_samples):
        # Image
        img = np.transpose(images[i], (1, 2, 0))
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Image {i+1}')
        axes[i, 0].axis('off')
        
        # Ground Truth
        mask = masks[i, 0]
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        pred = predictions[i, 0]
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('segmentation_results.png', dpi=300)
    plt.show()
    
    # Calculate metrics on entire dataset
    all_dice = []
    all_iou = []
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            predictions = model(images)
            predictions = (predictions > 0.5).float()
            
            for i in range(predictions.size(0)):
                pred_flat = predictions[i].view(-1)
                mask_flat = masks[i].view(-1)
                
                # Dice
                intersection = (pred_flat * mask_flat).sum()
                union = pred_flat.sum() + mask_flat.sum()
                dice = (2. * intersection + 1e-6) / (union + 1e-6)
                all_dice.append(dice.item())
                
                # IoU
                iou = (intersection + 1e-6) / (union - intersection + 1e-6)
                all_iou.append(iou.item())
    
    mean_dice = np.mean(all_dice)
    mean_iou = np.mean(all_iou)
    
    print(f"\n📊 Final Metrics:")
    print(f"  Mean Dice Score: {mean_dice:.4f}")
    print(f"  Mean IoU: {mean_iou:.4f}")
    
    return mean_dice, mean_iou

# ============================================
# 5. MAIN FUNCTION
# ============================================

def main():
    """
    Complete medical image segmentation pipeline
    """
    
    print("="*70)
    print("MEDICAL IMAGE SEGMENTATION SYSTEM")
    print("Tech Prime Pvt Limited - Advanced AI/ML Internship")
    print("="*70)
    
    # ============================================
    # DATA PREPARATION
    # ============================================
    
    print("\n[1/5] Loading and preparing data...")
    
    # Create dummy data for demonstration
    # In practice, you would load actual medical images
    os.makedirs('data/images', exist_ok=True)
    os.makedirs('data/masks', exist_ok=True)
    
    # Generate dummy images and masks
    for i in range(100):
        # Random image
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(f'data/images/image_{i:03d}.png', img)
        
        # Random mask (circle in center)
        mask = np.zeros((256, 256), dtype=np.uint8)
        cx, cy = np.random.randint(50, 200, 2)
        r = np.random.randint(30, 80)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        cv2.imwrite(f'data/masks/mask_{i:03d}.png', mask)
    
    print("✅ Created dummy dataset with 100 images and masks")
    
    # Create dataset
    dataset = MedicalSegmentationDataset(
        images_dir='data/images',
        masks_dir='data/masks',
        target_size=(256, 256)
    )
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    print(f"✅ Train samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    
    # ============================================
    # MODEL SETUP
    # ============================================
    
    print("\n[2/5] Setting up U-Net model...")
    
    model = UNetMedical(in_channels=3, out_channels=1)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # ============================================
    # TRAINING
    # ============================================
    
    print("\n[3/5] Training model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, history = train_model(
        model, train_loader, val_loader,
        epochs=20, lr=1e-4, device=device
    )
    
    # ============================================
    # EVALUATION
    # ============================================
    
    print("\n[4/5] Evaluating model...")
    
    # Load best model
    model.load_state_dict(torch.load('best_medical_segmentation.pth'))
    
    # Evaluate
    mean_dice, mean_iou = evaluate_model(model, val_loader, device)
    
    # ============================================
    # PLOT HISTORY
    # ============================================
    
    print("\n[5/5] Generating plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dice plot
    axes[1].plot(history['train_dice'], label='Train Dice', linewidth=2)
    axes[1].plot(history['val_dice'], label='Validation Dice', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Training and Validation Dice Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.show()
    
    print("\n" + "="*70)
    print("✅ PROJECT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\n📁 Output files:")
    print("  - best_medical_segmentation.pth (model weights)")
    print("  - segmentation_results.png (visual results)")
    print("  - training_history.png (training plots)")
    print("\n📊 Final Metrics:")
    print(f"  - Dice Score: {mean_dice:.4f}")
    print(f"  - IoU: {mean_iou:.4f}")

if __name__ == "__main__":
    main()
```

---

# 7. COMPLETE WORKING CODE WITH LINE-BY-LINE EXPLANATION

## 7.1 Complete Medical Segmentation Pipeline

```python
# ============================================
# COMPLETE MEDICAL SEGMENTATION SYSTEM
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

from torch.utils.data import Dataset, DataLoader
# Data loading utilities

import numpy as np
# Numerical computations

import cv2
# OpenCV - Image processing

import matplotlib.pyplot as plt
# Plotting and visualization

import os
# Operating system operations

from sklearn.model_selection import train_test_split
# For splitting datasets

import warnings
warnings.filterwarnings('ignore')
# Suppress warnings for cleaner output

# ============================================
# 2. CONFIGURATION - Settings We Can Change
# ============================================

class MedicalConfig:
    """
    Configuration for medical image segmentation
    
    All settings in one place for easy modification
    """
    
    # ----- Data Settings -----
    IMAGE_SIZE = (256, 256)    # Target image size (height, width)
    BATCH_SIZE = 8             # Images per batch (reduce if out of memory)
    NUM_WORKERS = 4            # Parallel data loading threads
    
    # ----- Model Settings -----
    IN_CHANNELS = 3            # Input channels (RGB)
    OUT_CHANNELS = 1           # Output channels (binary segmentation)
    FEATURES = [64, 128, 256, 512]  # Features at each U-Net level
    
    # ----- Training Settings -----
    EPOCHS = 50                # Number of training epochs
    LEARNING_RATE = 1e-4       # Learning rate for optimizer
    WEIGHT_DECAY = 1e-5        # L2 regularization strength
    
    # ----- Data Split -----
    TRAIN_SPLIT = 0.8          # Percentage of data for training
    VAL_SPLIT = 0.2            # Percentage for validation
    
    # ----- Paths -----
    DATA_DIR = './medical_data'           # Directory containing data
    IMAGES_DIR = './medical_data/images'  # Input images directory
    MASKS_DIR = './medical_data/masks'    # Ground truth masks directory
    MODEL_SAVE_PATH = 'best_medical_segmentation.pth'  # Model save path
    LOG_DIR = './logs'                     # Log directory
    
    # ----- Device -----
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Auto-select GPU if available, else use CPU
    
    # ----- Visualization -----
    NUM_VISUAL_SAMPLES = 5    # Number of images to show in visualization

# ============================================
# 3. MEDICAL SEGMENTATION DATASET
# ============================================

class MedicalSegmentationDataset(Dataset):
    """
    Custom Dataset for medical image segmentation
    
    This class handles:
    1. Loading images and masks
    2. Resizing to target size
    3. Normalization
    4. Data augmentation (optional)
    5. Converting to PyTorch tensors
    
    Directory Structure Expected:
    data/
      images/
        image_001.png
        image_002.png
        ...
      masks/
        mask_001.png
        mask_002.png
        ...
    """
    
    def __init__(self, images_dir, masks_dir, 
                 transform=None, target_size=(256, 256)):
        """
        Initialize the dataset
        
        Args:
            images_dir: Path to directory containing input images
            masks_dir: Path to directory containing ground truth masks
            transform: Optional transformations (augmentations)
            target_size: Desired image size (height, width)
        """
        super(MedicalSegmentationDataset, self).__init__()
        
        # Store parameters
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_size = target_size
        
        # Get list of image files (sorted for consistency)
        self.image_files = sorted([f for f in os.listdir(images_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Verify masks exist for each image
        self.mask_files = []
        missing_masks = 0
        
        for img_file in self.image_files:
            # Assume mask has same name as image
            mask_file = img_file
            
            # Check if mask exists
            mask_path = os.path.join(masks_dir, mask_file)
            if os.path.exists(mask_path):
                self.mask_files.append(mask_file)
            else:
                # No mask found, use None (will create dummy mask)
                self.mask_files.append(None)
                missing_masks += 1
        
        print(f"📊 Dataset Statistics:")
        print(f"  - Total images: {len(self.image_files)}")
        print(f"  - Images with masks: {len(self.image_files) - missing_masks}")
        print(f"  - Images without masks: {missing_masks}")
        
        if missing_masks > 0:
            print(f"  ⚠️  {missing_masks} images have no corresponding masks!")
            print(f"  → These will use dummy masks for demonstration.")
    
    def __len__(self):
        """
        Return total number of images in dataset
        """
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset
        
        Returns:
            image: Tensor of shape (C, H, W) normalized to [0, 1]
            mask: Tensor of shape (1, H, W) binary mask
        """
        
        # ----- Load Image -----
        # Read image using OpenCV
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(img_path)  # OpenCV reads in BGR format
        
        # Check if image loaded successfully
        if image is None:
            print(f"⚠️  Could not load image: {img_path}")
            # Return dummy data
            dummy = np.zeros((self.target_size[0], self.target_size[1], 3), 
                           dtype=np.uint8)
            return torch.zeros(3, *self.target_size), torch.zeros(1, *self.target_size)
        
        # Convert BGR to RGB (standard format)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to target size
        image = cv2.resize(image, self.target_size, 
                          interpolation=cv2.INTER_LINEAR)
        
        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Change from (H, W, C) to (C, H, W) for PyTorch
        image = np.transpose(image, (2, 0, 1))
        
        # ----- Load Mask -----
        if self.mask_files[idx] is not None:
            # Load mask
            mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is not None:
                # Resize mask (using nearest neighbor for binary data)
                mask = cv2.resize(mask, self.target_size, 
                                 interpolation=cv2.INTER_NEAREST)
                
                # Convert to binary (0 or 1)
                mask = (mask > 127).astype(np.float32)
            else:
                # Create dummy mask
                mask = np.zeros(self.target_size, dtype=np.float32)
        else:
            # No mask available, create dummy mask
            mask = np.zeros(self.target_size, dtype=np.float32)
        
        # Add channel dimension: (H, W) → (1, H, W)
        mask = np.expand_dims(mask, axis=0)
        
        # ----- Apply Transformations -----
        if self.transform is not None:
            # Apply data augmentation
            image, mask = self.transform(image, mask)
        
        # Convert to PyTorch tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        
        return image, mask

# ============================================
# 4. DATA AUGMENTATION FOR MEDICAL IMAGES
# ============================================

class MedicalAugmentation:
    """
    Data augmentation for medical image segmentation
    
    Why augmentation for medical images?
    1. Limited labeled data (hard to get medical annotations)
    2. Variations in scanning conditions
    3. Patient positioning variations
    4. Small dataset sizes
    """
    
    def __init__(self, rotation_range=30, flip_prob=0.5,
                 brightness_range=0.1, contrast_range=0.1):
        """
        Initialize augmentation parameters
        """
        self.rotation_range = rotation_range
        self.flip_prob = flip_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def __call__(self, image, mask):
        """
        Apply augmentations to both image and mask
        """
        
        # ----- Random Horizontal Flip -----
        if np.random.random() < self.flip_prob:
            # Flip image horizontally
            image = np.flip(image, axis=2)  # Flip along width axis
            mask = np.flip(mask, axis=2)
        
        # ----- Random Rotation -----
        if self.rotation_range > 0:
            # Random angle
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            
            # Get image dimensions
            h, w = image.shape[1], image.shape[2]
            
            # Rotation matrix
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Rotate image
            image = cv2.warpAffine(np.transpose(image, (1, 2, 0)), M, (w, h))
            image = np.transpose(image, (2, 0, 1))
            
            # Rotate mask
            mask = cv2.warpAffine(np.transpose(mask, (1, 2, 0)), M, (w, h))
            mask = np.transpose(mask, (2, 0, 1))
        
        # ----- Random Brightness -----
        if self.brightness_range > 0:
            # Random brightness factor
            brightness = 1 + np.random.uniform(-self.brightness_range, 
                                               self.brightness_range)
            image = np.clip(image * brightness, 0, 1)
        
        # ----- Random Contrast -----
        if self.contrast_range > 0:
            # Random contrast factor
            contrast = 1 + np.random.uniform(-self.contrast_range, 
                                             self.contrast_range)
            
            # Apply contrast
            mean = np.mean(image)
            image = np.clip((image - mean) * contrast + mean, 0, 1)
        
        return image, mask

# ============================================
# 5. U-NET MODEL (MEDICAL VERSION)
# ============================================

class DoubleConv(nn.Module):
    """
    Double Convolution Block for U-Net
    
    Two consecutive convolution layers with:
    - Batch Normalization (stabilizes training)
    - ReLU activation (nonlinearity)
    - Same padding (preserves spatial dimensions)
    """
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        
        # Sequential container for two convolutions
        self.conv = nn.Sequential(
            # First convolution
            nn.Conv2d(in_channels, out_channels, 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Second convolution
            nn.Conv2d(out_channels, out_channels, 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNetMedical(nn.Module):
    """
    U-Net architecture optimized for medical image segmentation
    
    Key Features:
    1. Symmetrical encoder-decoder structure
    2. Skip connections for detail preservation
    3. Contracting path (encoder) for context
    4. Expanding path (decoder) for localization
    5. Sigmoid output for binary segmentation
    """
    
    def __init__(self, in_channels=3, out_channels=1, 
                 features=[64, 128, 256, 512]):
        """
        Initialize U-Net
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (1 for binary)
            features: Number of feature maps at each level
        """
        super(UNetMedical, self).__init__()
        
        # ============================================
        # ENCODER (Contracting Path)
        # ============================================
        
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        # First encoder: in_channels → features[0]
        self.encoders.append(DoubleConv(in_channels, features[0]))
        
        # Subsequent encoders
        for i in range(1, len(features)):
            # Encoder: features[i-1] → features[i]
            self.encoders.append(DoubleConv(features[i-1], features[i]))
            
            # Pooling: reduce size by half
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # ============================================
        # BOTTLENECK
        # ============================================
        
        # Deepest part of U-Net
        # features[-1] → features[-1] * 2
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # ============================================
        # DECODER (Expanding Path)
        # ============================================
        
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # For each level (going up)
        for i in range(len(features)-1, 0, -1):
            # Upsample: features[i] * 2 → features[i]
            self.ups.append(
                nn.ConvTranspose2d(features[i] * 2, features[i],
                                  kernel_size=2, stride=2)
            )
            
            # Decoder: features[i] + features[i-1] → features[i-1]
            self.decoders.append(
                DoubleConv(features[i] + features[i-1], features[i-1])
            )
        
        # Last upsampling
        self.ups.append(
            nn.ConvTranspose2d(features[0] * 2, features[0],
                              kernel_size=2, stride=2)
        )
        
        # Last decoder: features[0] + features[0] → features[0]
        self.decoders.append(
            DoubleConv(features[0] + features[0], features[0])
        )
        
        # ============================================
        # FINAL LAYER
        # ============================================
        
        # 1×1 convolution to get desired output channels
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Sigmoid activation for binary segmentation
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass through U-Net
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            segmentation_mask: Shape (batch, out_channels, height, width)
        """
        
        # ============================================
        # ENCODER PATH
        # ============================================
        
        skip_connections = []
        
        for i, encoder in enumerate(self.encoders):
            # Apply encoder
            x = encoder(x)
            
            # Store for skip connection
            skip_connections.append(x)
            
            # Apply pooling (except for last encoder)
            if i < len(self.encoders) - 1:
                x = self.pools[i](x)
        
        # ============================================
        # BOTTLENECK
        # ============================================
        
        x = self.bottleneck(x)
        
        # ============================================
        # DECODER PATH
        # ============================================
        
        # Reverse skip connections (from deepest to shallowest)
        skip_connections = skip_connections[::-1]
        
        for i, (up, decoder) in enumerate(zip(self.ups, self.decoders)):
            # Upsample
            x = up(x)
            
            # Handle size mismatch (due to cropping in original U-Net)
            if x.shape[2:] != skip_connections[i].shape[2:]:
                diff_y = skip_connections[i].size(2) - x.size(2)
                diff_x = skip_connections[i].size(3) - x.size(3)
                x = nn.functional.pad(x, [
                    diff_x // 2, diff_x - diff_x // 2,
                    diff_y // 2, diff_y - diff_y // 2
                ])
            
            # Concatenate with skip connection
            x = torch.cat([skip_connections[i], x], dim=1)
            
            # Decode
            x = decoder(x)
        
        # ============================================
        # FINAL OUTPUT
        # ============================================
        
        x = self.final_conv(x)
        x = self.sigmoid(x)
        
        return x

# ============================================
# 6. LOSS FUNCTIONS FOR MEDICAL SEGMENTATION
# ============================================

def dice_loss(pred, target):
    """
    Dice Loss for medical image segmentation
    
    Why Dice Loss?
    - Handles class imbalance well
    - Directly optimizes Dice coefficient
    - Good for medical images with small objects
    
    Formula: 1 - (2 * |X ∩ Y|) / (|X| + |Y|)
    """
    smooth = 1e-6  # To avoid division by zero
    
    # Flatten tensors to 1D
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    # Dice coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # Dice loss = 1 - Dice
    return 1 - dice

def iou_loss(pred, target):
    """
    IoU (Jaccard) Loss
    
    Formula: 1 - |X ∩ Y| / |X ∪ Y|
    """
    smooth = 1e-6
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou

def combined_loss(pred, target):
    """
    Combined Binary Cross-Entropy + Dice Loss
    
    Why combine?
    - BCE: Good for per-pixel classification
    - Dice: Good for overall overlap
    - Combined: Best of both worlds
    """
    # Binary Cross-Entropy Loss
    bce_loss = nn.BCELoss()(pred, target)
    
    # Dice Loss
    dice_loss_value = dice_loss(pred, target)
    
    # Combined (equal weight)
    return bce_loss + dice_loss_value

# ============================================
# 7. TRAINING FUNCTIONS
# ============================================

def train_epoch(model, dataloader, optimizer, device):
    """
    Train model for one epoch
    
    This function:
    1. Puts model in training mode
    2. Iterates through all batches
    3. Computes loss
    4. Backpropagates and updates weights
    5. Returns average loss and Dice score
    """
    
    # Set model to training mode
    model.train()
    
    # Track metrics
    total_loss = 0
    total_dice = 0
    
    # Iterate through batches
    for batch_idx, (images, masks) in enumerate(dataloader):
        # Move data to device
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        predictions = model(images)
        loss = combined_loss(predictions, masks)
        
        # Backward pass
        optimizer.zero_grad()  # Reset gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
        
        # Track metrics
        total_loss += loss.item()
        
        # Calculate Dice score for this batch
        pred_binary = (predictions > 0.5).float()
        dice = 1 - dice_loss(pred_binary, masks)
        total_dice += dice.item()
    
    # Average metrics
    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    
    return avg_loss, avg_dice

def validate_epoch(model, dataloader, device):
    """
    Validate model for one epoch
    
    This function:
    1. Puts model in evaluation mode
    2. Iterates through all batches
    3. Computes loss (no gradients)
    4. Returns average loss and Dice score
    """
    
    # Set model to evaluation mode
    model.eval()
    
    # Track metrics
    total_loss = 0
    total_dice = 0
    
    # Disable gradient computation (saves memory)
    with torch.no_grad():
        for images, masks in dataloader:
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass (no gradients)
            predictions = model(images)
            loss = combined_loss(predictions, masks)
            
            # Track metrics
            total_loss += loss.item()
            
            # Calculate Dice score
            pred_binary = (predictions > 0.5).float()
            dice = 1 - dice_loss(pred_binary, masks)
            total_dice += dice.item()
    
    # Average metrics
    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    
    return avg_loss, avg_dice

def train_model(model, train_loader, val_loader, 
                epochs=50, lr=1e-4, device='cuda'):
    """
    Complete training pipeline
    
    This function:
    1. Sets up optimizer
    2. Runs training loop
    3. Validates each epoch
    4. Saves best model
    5. Tracks training history
    """
    
    # Move model to device
    model = model.to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # For tracking history
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': []
    }
    
    # Track best model
    best_dice = 0
    
    print("="*70)
    print("TRAINING MEDICAL SEGMENTATION MODEL")
    print("="*70)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {lr}")
    print("="*70)
    
    for epoch in range(epochs):
        # Train one epoch
        train_loss, train_dice = train_epoch(
            model, train_loader, optimizer, device
        )
        
        # Validate one epoch
        val_loss, val_dice = validate_epoch(
            model, val_loader, device
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), 'best_medical_segmentation.pth')
            print(f"  ✅ Best model saved! (Dice: {best_dice:.4f})")
    
    print(f"\n✅ Training complete! Best Dice: {best_dice:.4f}")
    
    return model, history

# ============================================
# 8. EVALUATION FUNCTIONS
# ============================================

def evaluate_model(model, dataloader, device, num_samples=5):
    """
    Evaluate model and visualize results
    
    This function:
    1. Gets a batch of data
    2. Generates predictions
    3. Visualizes images, masks, and predictions
    4. Calculates metrics on entire dataset
    """
    
    # Set model to evaluation mode
    model.eval()
    
    # Get a batch of data for visualization
    images, masks = next(iter(dataloader))
    images = images[:num_samples].to(device)
    masks = masks[:num_samples]
    
    # Generate predictions
    with torch.no_grad():
        predictions = model(images)
        predictions = (predictions > 0.5).float()
    
    # Move to CPU for visualization
    images = images.cpu().numpy()
    masks = masks.numpy()
    predictions = predictions.cpu().numpy()
    
    # ============================================
    # VISUALIZATION
    # ============================================
    
    print("\n" + "="*70)
    print("VISUALIZING SEGMENTATION RESULTS")
    print("="*70)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    for i in range(num_samples):
        # Image
        img = np.transpose(images[i], (1, 2, 0))
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Image {i+1}')
        axes[i, 0].axis('off')
        
        # Ground Truth Mask
        mask = masks[i, 0]
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        pred = predictions[i, 0]
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('segmentation_results.png', dpi=300)
    plt.show()
    print("💾 Saved visualization as 'segmentation_results.png'")
    
    # ============================================
    # METRICS ON ENTIRE DATASET
    # ============================================
    
    print("\n" + "="*70)
    print("CALCULATING METRICS")
    print("="*70)
    
    all_dice = []
    all_iou = []
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            predictions = model(images)
            predictions = (predictions > 0.5).float()
            
            for i in range(predictions.size(0)):
                pred_flat = predictions[i].view(-1)
                mask_flat = masks[i].view(-1)
                
                # Dice score
                intersection = (pred_flat * mask_flat).sum()
                union = pred_flat.sum() + mask_flat.sum()
                dice = (2. * intersection + 1e-6) / (union + 1e-6)
                all_dice.append(dice.item())
                
                # IoU
                iou = (intersection + 1e-6) / (union - intersection + 1e-6)
                all_iou.append(iou.item())
    
    # Calculate statistics
    mean_dice = np.mean(all_dice)
    std_dice = np.std(all_dice)
    mean_iou = np.mean(all_iou)
    std_iou = np.std(all_iou)
    
    print(f"\n📊 Final Metrics:")
    print(f"  Dice Score: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"  IoU: {mean_iou:.4f} ± {std_iou:.4f}")
    
    return mean_dice, mean_iou

# ============================================
# 9. MAIN FUNCTION - COMPLETE PIPELINE
# ============================================

def main():
    """
    Complete medical image segmentation pipeline
    
    Steps:
    1. Data Preparation
    2. Model Setup
    3. Training
    4. Evaluation
    5. Visualization
    """
    
    print("="*70)
    print("MEDICAL IMAGE SEGMENTATION SYSTEM")
    print("Tech Prime Pvt Limited - Advanced AI/ML Internship")
    print("="*70)
    print("\n🔬 This system performs pixel-level segmentation")
    print("   of medical images using U-Net architecture.")
    print("   It can identify and segment anatomical structures,")
    print("   tumors, and abnormalities.")
    
    # ============================================
    # STEP 1: DATA PREPARATION
    # ============================================
    
    print("\n[1/5] Loading and preparing data...")
    print("-" * 40)
    
    # Create directories if they don't exist
    os.makedirs(MedicalConfig.IMAGES_DIR, exist_ok=True)
    os.makedirs(MedicalConfig.MASKS_DIR, exist_ok=True)
    
    print("📁 Data directories created:")
    print(f"   Images: {MedicalConfig.IMAGES_DIR}")
    print(f"   Masks: {MedicalConfig.MASKS_DIR}")
    
    # For demonstration, create dummy dataset
    # In practice, you would use real medical images
    print("\n📊 Creating dummy dataset for demonstration...")
    
    for i in range(100):
        # Create random image
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(f'{MedicalConfig.IMAGES_DIR}/image_{i:03d}.png', img)
        
        # Create random mask (shape in center)
        mask = np.zeros((256, 256), dtype=np.uint8)
        cx, cy = np.random.randint(50, 200, 2)
        r = np.random.randint(30, 80)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        cv2.imwrite(f'{MedicalConfig.MASKS_DIR}/mask_{i:03d}.png', mask)
    
    print(f"✅ Created 100 dummy image-mask pairs")
    
    # Create dataset
    dataset = MedicalSegmentationDataset(
        images_dir=MedicalConfig.IMAGES_DIR,
        masks_dir=MedicalConfig.MASKS_DIR,
        transform=MedicalAugmentation(
            rotation_range=15,
            flip_prob=0.5,
            brightness_range=0.1,
            contrast_range=0.1
        ),
        target_size=MedicalConfig.IMAGE_SIZE
    )
    
    # Split into train and validation
    train_size = int(MedicalConfig.TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=MedicalConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=MedicalConfig.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=MedicalConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=MedicalConfig.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"\n✅ Data split complete:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # ============================================
    # STEP 2: MODEL SETUP
    # ============================================
    
    print("\n[2/5] Setting up U-Net model...")
    print("-" * 40)
    
    model = UNetMedical(
        in_channels=MedicalConfig.IN_CHANNELS,
        out_channels=MedicalConfig.OUT_CHANNELS,
        features=MedicalConfig.FEATURES
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("✅ Model created successfully")
    print(f"   Architecture: U-Net")
    print(f"   Input channels: {MedicalConfig.IN_CHANNELS}")
    print(f"   Output channels: {MedicalConfig.OUT_CHANNELS}")
    print(f"   Feature levels: {MedicalConfig.FEATURES}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # ============================================
    # STEP 3: TRAINING
    # ============================================
    
    print("\n[3/5] Training model...")
    print("-" * 40)
    
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=MedicalConfig.EPOCHS,
        lr=MedicalConfig.LEARNING_RATE,
        device=MedicalConfig.DEVICE
    )
    
    # ============================================
    # STEP 4: EVALUATION
    # ============================================
    
    print("\n[4/5] Evaluating model...")
    print("-" * 40)
    
    # Load best model
    model.load_state_dict(torch.load(MedicalConfig.MODEL_SAVE_PATH))
    print(f"✅ Loaded best model from: {MedicalConfig.MODEL_SAVE_PATH}")
    
    # Evaluate
    mean_dice, mean_iou = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=MedicalConfig.DEVICE,
        num_samples=MedicalConfig.NUM_VISUAL_SAMPLES
    )
    
    # ============================================
    # STEP 5: VISUALIZE TRAINING HISTORY
    # ============================================
    
    print("\n[5/5] Generating training history plots...")
    print("-" * 40)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dice plot
    axes[1].plot(history['train_dice'], label='Train Dice', linewidth=2)
    axes[1].plot(history['val_dice'], label='Validation Dice', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Training and Validation Dice Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.show()
    print("💾 Saved training history as 'training_history.png'")
    
    # ============================================
    # COMPLETION
    # ============================================
    
    print("\n" + "="*70)
    print("✅ PROJECT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\n📁 Output Files:")
    print(f"  - {MedicalConfig.MODEL_SAVE_PATH} (model weights)")
    print("  - segmentation_results.png (visual results)")
    print("  - training_history.png (training plots)")
    print("\n📊 Final Performance:")
    print(f"  - Dice Score: {mean_dice:.4f}")
    print(f"  - IoU: {mean_iou:.4f}")
    print("\n💡 Next Steps:")
    print("  1. Use real medical imaging data")
    print("  2. Experiment with different architectures")
    print("  3. Try different loss functions")
    print("  4. Implement 3D U-Net for volumetric data")
    
    return model, history

if __name__ == "__main__":
    main()
```

---

# 8. COMMON ISSUES AND SOLUTIONS

## 8.1 Medical Imaging Specific Issues

```python
# ============ COMMON ISSUES IN MEDICAL SEGMENTATION ============

def common_issues_and_solutions():
    """
    Common problems and their solutions in medical segmentation
    """
    
    print("="*70)
    print("COMMON ISSUES AND SOLUTIONS IN MEDICAL SEGMENTATION")
    print("="*70)
    
    issues = {
        "1. Class Imbalance": {
            "Problem": "Medical images often have small regions of interest (e.g., tumors occupy <1% of image)",
            "Solutions": [
                "Use Dice loss instead of Cross-Entropy",
                "Oversample minority class",
                "Use focal loss",
                "Weighted loss functions",
                "Data augmentation for minority class"
            ],
            "Code": """
# Weighted BCE
class WeightedBCE(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred, target):
        bce = nn.BCEWithLogitsLoss()(pred, target)
        return bce * self.weight

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = nn.BCEWithLogitsLoss(reduction='none')(pred, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
            """
        },
        
        "2. Limited Training Data": {
            "Problem": "Medical annotations are expensive and time-consuming",
            "Solutions": [
                "Data augmentation (rotation, flipping, elastic deformations)",
                "Transfer learning from similar datasets",
                "Semi-supervised learning",
                "Synthetic data generation",
                "Active learning"
            ],
            "Code": """
# Elastic deformation augmentation
def elastic_deformation(image, mask, alpha=50, sigma=5):
    random_state = np.random.RandomState(None)
    shape = image.shape[:2]
    
    dx = random_state.randn(*shape) * sigma
    dy = random_state.randn(*shape) * sigma
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy, x + dx)
    
    image = ndimage.map_coordinates(image, indices, order=1)
    mask = ndimage.map_coordinates(mask, indices, order=0)
    
    return image, mask
            """
        },
        
        "3. Poor Contrast / Low Quality": {
            "Problem": "Medical images often have low contrast and noise",
            "Solutions": [
                "Contrast enhancement (CLAHE)",
                "Histogram equalization",
                "Denoising filters",
                "Intensity normalization",
                "Standardization"
            ],
            "Code": """
# CLAHE contrast enhancement
def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# Z-score normalization
def z_score_normalize(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / (std + 1e-8)
            """
        },
        
        "4. Variability in Anatomy": {
            "Problem": "Different patients have different anatomies",
            "Solutions": [
                "Data augmentation with anatomical variations",
                "Registration to standard template",
                "Multi-scale architectures",
                "Ensemble methods"
            ],
            "Code": """
# Scale augmentation
def random_scale(image, mask, scale_range=(0.8, 1.2)):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    h, w = image.shape[:2]
    new_h = int(h * scale)
    new_w = int(w * scale)
    image = cv2.resize(image, (new_w, new_h))
    mask = cv2.resize(mask, (new_w, new_h))
    return image, mask
            """
        },
        
        "5. Memory Issues": {
            "Problem": "Medical images are large (e.g., 3D volumes)",
            "Solutions": [
                "Patch-based training",
                "Reduce batch size",
                "Use smaller input size",
                "Gradient accumulation",
                "Mixed precision training"
            ],
            "Code": """
# Patch-based training
def extract_patches(image, mask, patch_size=64, stride=32):
    patches_img = []
    patches_mask = []
    
    h, w = image.shape[:2]
    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            patch_img = image[y:y+patch_size, x:x+patch_size]
            patch_mask = mask[y:y+patch_size, x:x+patch_size]
            patches_img.append(patch_img)
            patches_mask.append(patch_mask)
    
    return np.array(patches_img), np.array(patches_mask)
            """
        }
    }
    
    for issue, info in issues.items():
        print(f"\n🔴 {issue}")
        print(f"   Problem: {info['Problem']}")
        print(f"   Solutions:")
        for sol in info['Solutions']:
            print(f"   - {sol}")
        if 'Code' in info:
            print(f"\n   Example Code:")
            print(f"   {info['Code']}")
        print("-" * 70)

common_issues_and_solutions()
```

## 8.2 Performance Optimization

```python
# ============ PERFORMANCE OPTIMIZATION ============

def performance_optimization():
    """
    Tips for improving medical segmentation performance
    """
    
    print("="*70)
    print("PERFORMANCE OPTIMIZATION TIPS")
    print("="*70)
    
    tips = {
        "Training Speed": [
            "Use GPU acceleration",
            "Increase batch size (if memory allows)",
            "Use mixed precision training",
            "Reduce image size",
            "Use data parallelism",
            "Optimize data loading (pre-fetch)"
        ],
        "Model Accuracy": [
            "Use deeper U-Net",
            "Add attention mechanisms",
            "Use residual connections",
            "Ensemble multiple models",
            "Use test-time augmentation",
            "Post-processing (CRF)"
        ],
        "Memory Efficiency": [
            "Use gradient checkpointing",
            "Reduce feature channels",
            "Use patch-based training",
            "Use 3D to 2D projections",
            "Use model pruning",
            "Use quantization"
        ],
        "Generalization": [
            "Use diverse training data",
            "Apply heavy augmentation",
            "Use label smoothing",
            "Use adversarial training",
            "Use self-supervised pre-training",
            "Use domain adaptation"
        ]
    }
    
    for category, items in tips.items():
        print(f"\n📊 {category.upper()}:")
        for i, item in enumerate(items, 1):
            print(f"   {i}. {item}")

performance_optimization()
```

---

# 9. QUICK REFERENCE - ALL CODE PATTERNS

## 9.1 U-Net Architecture Patterns

```python
# ============ BASIC U-NET ============

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Define layers as shown in section 3
    
    def forward(self, x):
        # Encoder + Decoder with skip connections
        return x

# ============ RESIDUAL U-NET ============

class ResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Use residual blocks instead of plain convolutions
    
    def forward(self, x):
        return x

# ============ ATTENTION U-NET ============

class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Add attention gates between encoder and decoder
    
    def forward(self, x):
        return x
```

## 9.2 Data Loading Patterns

```python
# ============ MEDICAL DATASET ============

class MedicalDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = cv2.imread(...)
        mask = cv2.imread(...)
        return image, mask

# ============ DATA LOADER ============

train_loader = DataLoader(
    dataset, batch_size=8, shuffle=True, num_workers=4
)
```

## 9.3 Training Patterns

```python
# ============ TRAINING LOOP ============

for epoch in range(epochs):
    model.train()
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        predictions = model(images)
        loss = loss_fn(predictions, masks)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            predictions = model(images)
            # Calculate metrics

# ============ DICE LOSS ============

def dice_loss(pred, target):
    smooth = 1e-6
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    return 1 - (2 * intersection + smooth) / (union + smooth)

# ============ COMBINED LOSS ============

def combined_loss(pred, target):
    return nn.BCELoss()(pred, target) + dice_loss(pred, target)
```

## 9.4 Evaluation Patterns

```python
# ============ METRICS ============

def compute_metrics(pred, target):
    pred = (pred > 0.5).float()
    
    # Dice
    dice = 2 * (pred * target).sum() / (pred.sum() + target.sum() + 1e-6)
    
    # IoU
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-6)
    
    return dice, iou

# ============ VISUALIZATION ============

def visualize_results(images, masks, predictions):
    fig, axes = plt.subplots(len(images), 3)
    for i in range(len(images)):
        axes[i, 0].imshow(images[i])
        axes[i, 1].imshow(masks[i], cmap='gray')
        axes[i, 2].imshow(predictions[i], cmap='gray')
    plt.show()
```

## 9.5 OpenCV Patterns

```python
# ============ BASIC OPERATIONS ============

# Read image
image = cv2.imread('image.jpg')

# Resize
resized = cv2.resize(image, (256, 256))

# Convert color
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Draw shapes
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.circle(image, (cx, cy), radius, (0, 0, 255), -1)
cv2.putText(image, 'Text', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Show image
cv2.imshow('Window', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save image
cv2.imwrite('output.jpg', image)

# ============ FILTERING ============

# Blur
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Edge detection
edges = cv2.Canny(image, 100, 200)

# Contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# ============ PREPROCESSING ============

# Histogram equalization
equalized = cv2.equalizeHist(gray)

# Thresholding
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Morphological operations
kernel = np.ones((3, 3), np.uint8)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
```

---

**End of Week 3 Notes - Complete Noob-Friendly Guide**

##  Key Takeaways

1. **Segmentation** = Label every pixel, not just finding objects
2. **U-Net** = The go-to architecture for medical image segmentation
3. **Skip Connections** = Preserve fine details in segmentation
4. **OpenCV** = Your Swiss Army knife for image processing
5. **Preprocessing** = Clean data is crucial for good results
6. **Medical Imaging** = Special challenges: low contrast, limited data, class imbalance
7. **Dice Loss** = Better than BCE for imbalanced segmentation
8. **Data Augmentation** = Essential for medical images with limited data

