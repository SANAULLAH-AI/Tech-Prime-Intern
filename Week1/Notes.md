# COMPLETE WEEK 1 NOTES - PyTorch Fundamentals & CIFAR-10 Classifier

## Professional Training Documentation
**Tech Prime Pvt Limited - Advanced AI/ML Internship Program**

---

# 📑 TABLE OF CONTENTS

1. [Introduction to PyTorch](#1-introduction-to-pytorch)
2. [Understanding Tensors](#2-understanding-tensors)
3. [Autograd - Automatic Differentiation](#3-autograd---automatic-differentiation)
4. [Building Neural Networks with nn.Module](#4-building-neural-networks-with-nnmodule)
5. [Loss Functions Explained](#5-loss-functions-explained)
6. [Optimizers - How Models Learn](#6-optimizers---how-models-learn)
7. [Data Loading and Preprocessing](#7-data-loading-and-preprocessing)
8. [Complete Training Pipeline](#8-complete-training-pipeline)
9. [GPU Acceleration](#9-gpu-acceleration)
10. [Model Evaluation and Testing](#10-model-evaluation-and-testing)
11. [Complete Working Code](#11-complete-working-code)
12. [Common Issues and Solutions](#12-common-issues-and-solutions)
13. [Quick Reference Guide](#13-quick-reference-guide)
14. [Final Checklist](#14-final-checklist)

---

# 1. INTRODUCTION TO PYTORCH

## 1.1 What is PyTorch?

**Definition:** PyTorch is an open-source machine learning framework developed by Facebook's AI Research lab.

**Simple Analogy:** 
- Think of PyTorch as **LEGO blocks for AI**
- Each block is a building component
- You connect blocks to build complex AI systems

## 1.2 Why PyTorch?

| Feature | Benefit |
|---------|---------|
| **Python Native** | Feels like writing normal Python |
| **Dynamic Computation Graphs** | Easy to debug and modify |
| **GPU Support** | 100x faster training |
| **Large Community** | Lots of tutorials and help available |
| **Industry Standard** | Used by Tesla, Meta, OpenAI, and more |

## 1.3 PyTorch vs Other Frameworks

```
PyTorch          = Pythonic, Dynamic, Research Friendly
TensorFlow       = Production Ready, Static, More Complex
NumPy            = Only Arrays, No GPU, No Deep Learning
Keras            = High Level, Less Control, Good for Beginners
```

---

# 2. UNDERSTANDING TENSORS

## 2.1 What is a Tensor?

**Definition:** A tensor is a multi-dimensional array of numbers.

**Think of it as:**
- **0D Tensor (Scalar)** = Single number → `5`
- **1D Tensor (Vector)** = List → `[1, 2, 3]`
- **2D Tensor (Matrix)** = Table → `[[1,2], [3,4]]`
- **3D Tensor** = Cube → Like a color image (height × width × channels)
- **4D Tensor** = Batch of images → (batch_size × height × width × channels)

## 2.2 Creating Tensors

### Method 1: From Python Lists

```python
import torch

# 0D Tensor (Scalar)
scalar = torch.tensor(5)
print(scalar)              # tensor(5)
print(scalar.shape)        # torch.Size([])

# 1D Tensor (Vector)
vector = torch.tensor([1, 2, 3, 4, 5])
print(vector)              # tensor([1, 2, 3, 4, 5])
print(vector.shape)        # torch.Size([5])

# 2D Tensor (Matrix)
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
print(matrix)              # tensor([[1, 2, 3], [4, 5, 6]])
print(matrix.shape)        # torch.Size([2, 3])

# 3D Tensor (Like RGB Image)
tensor_3d = torch.tensor([[[1, 2], [3, 4]],
                          [[5, 6], [7, 8]]])
print(tensor_3d.shape)     # torch.Size([2, 2, 2])
```

### Method 2: Using PyTorch Functions

```python
# ============ ZEROS ============
zeros = torch.zeros(3, 4)  # 3 rows, 4 columns of zeros
print(zeros)
"""
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]])
"""

# ============ ONES ============
ones = torch.ones(2, 3)    # 2 rows, 3 columns of ones
print(ones)
"""
tensor([[1., 1., 1.],
        [1., 1., 1.]])
"""

# ============ RANDOM NUMBERS ============
# Normal distribution (mean=0, std=1)
random_normal = torch.randn(3, 3)
print(random_normal)

# Uniform distribution (0 to 1)
random_uniform = torch.rand(3, 3)
print(random_uniform)

# ============ IDENTITY MATRIX ============
identity = torch.eye(4)    # 4x4 identity matrix
print(identity)
"""
tensor([[1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]])
"""

# ============ SEQUENCE ============
sequence = torch.arange(0, 10, 2)  # Start, Stop, Step
print(sequence)                    # tensor([0, 2, 4, 6, 8])

# ============ LINSPACE ============
linspace = torch.linspace(0, 10, 5)  # 5 evenly spaced numbers from 0 to 10
print(linspace)                      # tensor([0.0000, 2.5000, 5.0000, 7.5000, 10.0000])
```

### Method 3: From NumPy

```python
import numpy as np

# Convert NumPy to PyTorch Tensor
numpy_array = np.array([[1, 2], [3, 4]])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(tensor_from_numpy)

# Convert PyTorch Tensor to NumPy
tensor = torch.tensor([1, 2, 3])
numpy_from_tensor = tensor.numpy()
print(numpy_from_tensor)
```

### Method 4: With Specific Data Types

```python
# Float (default)
float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
print(float_tensor.dtype)  # torch.float32

# Integer
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
print(int_tensor.dtype)    # torch.int64

# Boolean
bool_tensor = torch.tensor([True, False, True], dtype=torch.bool)
print(bool_tensor.dtype)   # torch.bool
```

## 2.3 Tensor Properties

```python
tensor = torch.randn(3, 4, 5)  # Random 3x4x5 tensor

# ============ SHAPE ============
print(tensor.shape)        # torch.Size([3, 4, 5])
print(tensor.size())       # torch.Size([3, 4, 5])

# ============ DATA TYPE ============
print(tensor.dtype)        # torch.float32

# ============ DEVICE ============
print(tensor.device)       # cpu

# ============ NUMBER OF DIMENSIONS ============
print(tensor.ndim)         # 3

# ============ TOTAL ELEMENTS ============
print(tensor.numel())      # 60 (3 × 4 × 5)

# ============ MEMORY SIZE ============
print(tensor.element_size())  # 4 bytes (for float32)
print(tensor.numel() * tensor.element_size(), "bytes")
```

## 2.4 Tensor Operations

### Basic Arithmetic

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# ============ ADDITION ============
print(a + b)          # tensor([5, 7, 9])
print(torch.add(a, b))  # Same as above
print(a.add(b))         # Same as above

# ============ SUBTRACTION ============
print(a - b)          # tensor([-3, -3, -3])
print(a.sub(b))       # Same

# ============ MULTIPLICATION (Element-wise) ============
print(a * b)          # tensor([4, 10, 18])
print(torch.mul(a, b)) # Same

# ============ DIVISION ============
print(a / b)          # tensor([0.2500, 0.4000, 0.5000])

# ============ POWER ============
print(a ** 2)         # tensor([1, 4, 9])
print(torch.pow(a, 2)) # Same
```

### Matrix Operations

```python
A = torch.tensor([[1, 2],
                  [3, 4]])
B = torch.tensor([[5, 6],
                  [7, 8]])

# ============ MATRIX MULTIPLICATION ============
print(A @ B)  # Using @ operator
"""
tensor([[19, 22],
        [43, 50]])
"""

print(torch.matmul(A, B))  # Using torch.matmul
# Same result

# ============ TRANSPOSE ============
print(A.T)
"""
tensor([[1, 3],
        [2, 4]])
"""

# ============ DOT PRODUCT ============
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(torch.dot(a, b))  # tensor(32) = 1*4 + 2*5 + 3*6
```

### Broadcasting

Broadcasting = Smaller tensor expands to match larger tensor

```python
# ============ SCALAR BROADCASTING ============
tensor = torch.tensor([1, 2, 3])
print(tensor + 10)  # tensor([11, 12, 13]) - 10 added to each element
print(tensor * 2)   # tensor([2, 4, 6]) - each element multiplied by 2

# ============ VECTOR BROADCASTING ============
A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
B = torch.tensor([10, 20, 30])  # Broadcast to both rows
print(A + B)
"""
tensor([[11, 22, 33],
        [14, 25, 36]])
"""
```

### Reshaping and Resizing

```python
# ============ VIEW (Reshape) ============
tensor = torch.arange(12)  # [0, 1, 2, ..., 11]
print(tensor)

# Reshape to 3x4
reshaped = tensor.view(3, 4)
print(reshaped)
"""
tensor([[0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11]])
"""

# Reshape to 2x6
reshaped_2 = tensor.view(2, 6)
print(reshaped_2)
"""
tensor([[0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11]])
"""

# ============ FLATTEN ============
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
flattened = tensor.flatten()
print(flattened)  # tensor([1, 2, 3, 4, 5, 6])

# ============ SQUEEZE (Remove dimensions of size 1) ============
tensor = torch.tensor([[[1, 2, 3]]])  # Shape: (1, 1, 3)
squeezed = tensor.squeeze()           # Shape: (3,)
print(squeezed)  # tensor([1, 2, 3])

# ============ UNSQUEEZE (Add dimension of size 1) ============
tensor = torch.tensor([1, 2, 3])      # Shape: (3,)
unsqueezed = tensor.unsqueeze(0)       # Shape: (1, 3)
print(unsqueezed)  # tensor([[1, 2, 3]])
```

### Indexing and Slicing

```python
tensor = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])

# ============ SINGLE ELEMENT ============
print(tensor[0, 0])  # 1 (First row, first column)
print(tensor[1, 2])  # 7 (Second row, third column)

# ============ ROW SLICING ============
print(tensor[0, :])  # First row: [1, 2, 3, 4]
print(tensor[:, 0])  # First column: [1, 5, 9]

# ============ SUB-MATRIX ============
print(tensor[0:2, 0:2])  # First 2 rows, first 2 columns
"""
tensor([[1, 2],
        [5, 6]])
"""

# ============ STEP SLICING ============
print(tensor[::2, ::2])  # Every 2nd row and column
"""
tensor([[1, 3],
        [9, 11]])
"""

# ============ BOOLEAN INDEXING ============
mask = tensor > 5
print(tensor[mask])  # tensor([6, 7, 8, 9, 10, 11, 12])
```

### Aggregation Operations

```python
tensor = torch.tensor([1, 2, 3, 4, 5])

# ============ SUM ============
print(tensor.sum())        # 15
print(tensor.sum().item()) # 15 (as Python number)

# ============ MEAN ============
print(tensor.mean())       # 3.0
print(tensor.mean().item()) # 3.0

# ============ MAX / MIN ============
print(tensor.max())        # 5
print(tensor.min())        # 1

# ============ STANDARD DEVIATION ============
print(tensor.std())        # 1.5811

# ============ ARGMAX / ARGMIN ============
print(tensor.argmax())     # Index of max value: 4
print(tensor.argmin())     # Index of min value: 0

# ============ DIMENSION-WISE OPERATIONS ============
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])

print(matrix.sum(dim=0))  # Sum along columns: [5, 7, 9]
print(matrix.sum(dim=1))  # Sum along rows: [6, 15]
print(matrix.mean(dim=0)) # Mean along columns: [2.5, 3.5, 4.5]
```

## 2.5 In-Place vs Out-of-Place Operations

```python
x = torch.tensor([1, 2, 3])

# ============ OUT-OF-PLACE (Creates new tensor) ============
y = x.add(5)  # x unchanged, y = [6, 7, 8]
print(x)      # tensor([1, 2, 3])
print(y)      # tensor([6, 7, 8])

# ============ IN-PLACE (Modifies original) ============
x.add_(5)     # Note the underscore! x becomes [6, 7, 8]
print(x)      # tensor([6, 7, 8])

# Other in-place operations:
# add_(), sub_(), mul_(), div_(), zero_(), fill_(), etc.
```

---

# 3. AUTOGRAD - AUTOMATIC DIFFERENTIATION

## 3.1 What is Autograd?

**Definition:** Autograd is PyTorch's automatic differentiation engine that computes gradients (derivatives) of tensor operations.

**Simple Analogy:**
- You're driving down a hill blindfolded
- Autograd tells you which direction is DOWN
- This helps you find the lowest point (minimum loss)

## 3.2 How Autograd Works

```python
import torch

# ============ STEP 1: Create tensor with requires_grad=True ============
x = torch.tensor(2.0, requires_grad=True)
print(x)  # tensor(2., requires_grad=True)

# ============ STEP 2: Perform operations ============
y = x ** 2          # y = x² = 4
z = y + 3           # z = x² + 3 = 7

# ============ STEP 3: Compute gradient ============
z.backward()  # Calculates dz/dx

# ============ STEP 4: Access gradient ============
print(x.grad)  # tensor(4.)
# Why 4? Because d(z)/dx = d(x²+3)/dx = 2x, at x=2 => 4
```

## 3.3 More Complex Example

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Function: f(x) = 3x² + 2x + 1
f = 3 * x**2 + 2 * x + 1

# Sum all values
loss = f.sum()

# Compute gradients
loss.backward()

# Gradient: df/dx = 6x + 2
print(x.grad)  # tensor([8., 14., 20.])
# At x=1: 8, x=2: 14, x=3: 20
```

## 3.4 Stopping Gradient Tracking

Sometimes you don't want gradients (like during testing).

```python
x = torch.tensor(2.0, requires_grad=True)

# ============ METHOD 1: detach() ============
y = x.detach()  # New tensor without gradient tracking
print(y.requires_grad)  # False

# ============ METHOD 2: with torch.no_grad() ============
with torch.no_grad():
    z = x * 2
    print(z.requires_grad)  # False

# ============ METHOD 3: .requires_grad_(False) ============
x.requires_grad_(False)
print(x.requires_grad)  # False
```

## 3.5 Zeroing Gradients

```python
x = torch.tensor(2.0, requires_grad=True)

# ============ WRONG: Gradients accumulate! ============
for i in range(3):
    y = x ** 2
    y.backward()
    print(x.grad)  # 4.0, 8.0, 12.0 (accumulating!)
    
# ============ CORRECT: Zero gradients each time ============
for i in range(3):
    y = x ** 2
    y.backward()
    print(x.grad)  # 4.0 each time
    x.grad.zero_()  # Reset to zero
```

## 3.6 Autograd in Machine Learning

**Why we need Autograd:**

```
1. Forward Pass: model predicts
2. Calculate Loss: see how wrong
3. Backward Pass: Autograd calculates gradients
4. Update Weights: using gradients
```

**In code:**
```python
# Training step
optimizer.zero_grad()    # Reset gradients
output = model(data)     # Forward pass
loss = criterion(output, target)  # Calculate loss
loss.backward()          # Autograd computes gradients
optimizer.step()         # Update weights
```

---

# 4. BUILDING NEURAL NETWORKS WITH nn.MODULE

## 4.1 Understanding nn.Module

**Definition:** nn.Module is the base class for all neural network modules in PyTorch.

**Two Required Methods:**
1. `__init__()`: Define your layers
2. `forward()`: Define how data flows

**Think of it like:**
```
class YourModel(nn.Module):
    def __init__(self):
        # Set up the factory (layers)
        
    def forward(self, x):
        # How materials flow through factory
        return x
```

## 4.2 Basic Linear Model

```python
import torch.nn as nn

class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        # One layer: 10 inputs -> 1 output
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

# Create and use model
model = SimpleLinearModel()
x = torch.randn(32, 10)  # Batch of 32 samples, 10 features
output = model(x)
print(output.shape)  # torch.Size([32, 1])
```

## 4.3 Multi-Layer Perceptron (MLP)

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: 10 -> 32
        self.fc1 = nn.Linear(10, 32)
        
        # Layer 2: 32 -> 16
        self.fc2 = nn.Linear(32, 16)
        
        # Layer 3: 16 -> 2
        self.fc3 = nn.Linear(16, 2)
        
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)  # Activation after layer 1
        x = self.fc2(x)
        x = self.relu(x)  # Activation after layer 2
        x = self.fc3(x)   # No activation on last layer
        return x

model = MLP()
x = torch.randn(32, 10)
output = model(x)
print(output.shape)  # torch.Size([32, 2])
```

## 4.4 CNN Model for CIFAR-10

```python
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ============ CONVOLUTIONAL LAYERS ============
        # Input: (batch, 3, 32, 32)
        # Output: (batch, 32, 32, 32)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        
        # Input: (batch, 32, 32, 32)
        # Output: (batch, 64, 32, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Input: (batch, 64, 32, 32)
        # Output: (batch, 128, 32, 32)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # ============ POOLING ============
        # Reduces size by half
        self.pool = nn.MaxPool2d(2, 2)
        
        # ============ BATCH NORMALIZATION ============
        # Helps training by normalizing inputs
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # ============ FULLY CONNECTED LAYERS ============
        # After 3 pooling layers, image size: 32 -> 16 -> 8 -> 4
        # So final size: 128 * 4 * 4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)  # 10 classes
        
        # ============ REGULARIZATION ============
        self.dropout = nn.Dropout(0.5)
        
        # ============ ACTIVATIONS ============
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Input: (batch, 3, 32, 32)
        
        # Block 1: Conv + BN + ReLU + Pool
        x = self.conv1(x)          # (batch, 32, 32, 32)
        x = self.bn1(x)            # (batch, 32, 32, 32)
        x = self.relu(x)           # (batch, 32, 32, 32)
        x = self.pool(x)           # (batch, 32, 16, 16)
        
        # Block 2: Conv + BN + ReLU + Pool
        x = self.conv2(x)          # (batch, 64, 16, 16)
        x = self.bn2(x)            # (batch, 64, 16, 16)
        x = self.relu(x)           # (batch, 64, 16, 16)
        x = self.pool(x)           # (batch, 64, 8, 8)
        
        # Block 3: Conv + BN + ReLU + Pool
        x = self.conv3(x)          # (batch, 128, 8, 8)
        x = self.bn3(x)            # (batch, 128, 8, 8)
        x = self.relu(x)           # (batch, 128, 8, 8)
        x = self.pool(x)           # (batch, 128, 4, 4)
        
        # Flatten: (batch, 128, 4, 4) -> (batch, 2048)
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully Connected Layers
        x = self.fc1(x)            # (batch, 256)
        x = self.relu(x)           # (batch, 256)
        x = self.dropout(x)        # (batch, 256)
        x = self.fc2(x)            # (batch, 10)
        
        return x
```

## 4.5 Understanding Each Layer

### Conv2d (Convolutional Layer)

```python
conv = nn.Conv2d(
    in_channels=3,      # Input channels (RGB = 3)
    out_channels=32,    # Number of filters (feature maps)
    kernel_size=3,      # Size of filter (3x3)
    stride=1,           # How many pixels to shift
    padding=1           # Add zeros to keep size same
)

# Input: (batch, 3, 32, 32)
# Output: (batch, 32, 32, 32)
```

### MaxPool2d (Pooling Layer)

```python
pool = nn.MaxPool2d(
    kernel_size=2,   # Window size (2x2)
    stride=2         # How much to shift
)

# Input: (batch, 32, 32, 32)
# Output: (batch, 32, 16, 16)  # Size reduced by half
```

### BatchNorm2d (Batch Normalization)

```python
bn = nn.BatchNorm2d(
    num_features=32  # Number of channels
)

# Normalizes each batch
# Helps training be more stable
# Reduces overfitting
```

### Dropout

```python
dropout = nn.Dropout(
    p=0.5  # Probability of turning off each neuron
)

# During training: randomly turns off 50% of neurons
# During testing: uses all neurons (multiplied by 0.5)
# Prevents overfitting
```

### Linear (Fully Connected)

```python
fc = nn.Linear(
    in_features=2048,  # Input features
    out_features=256    # Output features
)

# Input: (batch, 2048)
# Output: (batch, 256)
```

## 4.6 Activation Functions

```python
import torch.nn as nn
import torch.nn.functional as F

# ============ ReLU (Most Common) ============
# Negative values become 0, positive unchanged
relu = nn.ReLU()
x = torch.tensor([-2, -1, 0, 1, 2])
print(relu(x))  # tensor([0, 0, 0, 1, 2])

# ============ Softmax (For Classification) ============
# Converts outputs to probabilities (sum to 1)
softmax = nn.Softmax(dim=1)
x = torch.tensor([[1.0, 2.0, 3.0]])
print(softmax(x))  # tensor([[0.0900, 0.2447, 0.6652]])

# ============ Sigmoid ============
# Squeezes values between 0 and 1
sigmoid = nn.Sigmoid()
x = torch.tensor([-2, -1, 0, 1, 2])
print(sigmoid(x))  # tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])

# ============ Tanh ============
# Squeezes values between -1 and 1
tanh = nn.Tanh()
x = torch.tensor([-2, -1, 0, 1, 2])
print(tanh(x))  # tensor([-0.9640, -0.7616, 0.0000, 0.7616, 0.9640])
```

## 4.7 Model Summary

```python
def model_summary(model):
    """Print model architecture and parameter count"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("\nArchitecture:")
    print(model)

# Usage
model = CIFAR10CNN()
model_summary(model)
```

---

# 5. LOSS FUNCTIONS EXPLAINED

## 5.1 What is a Loss Function?

**Definition:** A loss function measures how wrong your model's predictions are.

- **Low Loss** = Model is accurate
- **High Loss** = Model is inaccurate

**Analogy:** 
- Loss is like a score in a video game
- You want to get the LOWEST score possible
- Optimizer tries to reduce this score

## 5.2 Common Loss Functions

### CrossEntropyLoss (For Classification)

```python
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

# Example
predictions = torch.tensor([[0.5, 1.5, 0.2, 0.8],  # Logits for 4 classes
                            [2.0, 0.1, 0.3, 0.6]])
targets = torch.tensor([1, 0])  # True class indices

loss = criterion(predictions, targets)
print(loss)  # Single scalar value

# What it does:
# 1. Applies Softmax to predictions (converts to probabilities)
# 2. Takes negative log of probability of correct class
# 3. Averages over batch
```

### MSELoss (For Regression)

```python
criterion = nn.MSELoss()  # Mean Squared Error

# Example
predictions = torch.tensor([[1.0], [2.0], [3.0]])
targets = torch.tensor([[1.5], [2.5], [3.5]])

loss = criterion(predictions, targets)
print(loss)  # 0.25

# What it does:
# 1. Calculates (prediction - target)²
# 2. Averages over batch
```

### L1Loss (Mean Absolute Error)

```python
criterion = nn.L1Loss()

predictions = torch.tensor([[1.0], [2.0], [3.0]])
targets = torch.tensor([[1.5], [2.5], [3.5]])

loss = criterion(predictions, targets)
print(loss)  # 0.5

# What it does:
# 1. Calculates |prediction - target|
# 2. Averages over batch
```

### Binary Cross Entropy (For Binary Classification)

```python
criterion = nn.BCEWithLogitsLoss()  # Includes sigmoid

# Example
predictions = torch.tensor([0.2, -0.5, 1.0])
targets = torch.tensor([1.0, 0.0, 1.0])

loss = criterion(predictions, targets)
print(loss)
```

## 5.3 Choosing the Right Loss

| Task | Loss Function |
|------|---------------|
| Multi-class Classification | CrossEntropyLoss |
| Binary Classification | BCEWithLogitsLoss |
| Regression | MSELoss or L1Loss |
| Custom | Define your own |

---

# 6. OPTIMIZERS - HOW MODELS LEARN

## 6.1 What is an Optimizer?

**Definition:** An optimizer updates the model's weights to reduce the loss.

**Analogy:**
- You're blindfolded on a hill
- You want to reach the bottom (minimum loss)
- Optimizer tells you: "Take a step in this direction"

## 6.2 Common Optimizers

### Adam (Most Popular)

```python
import torch.optim as optim

optimizer = optim.Adam(
    model.parameters(),  # What to optimize
    lr=0.001            # Learning rate (step size)
)

# Benefits:
# - Adapts learning rate for each parameter
# - Works well for most problems
# - Fast convergence
```

### SGD (Stochastic Gradient Descent)

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,           # Learning rate
    momentum=0.9       # Helps escape local minima
)

# Benefits:
# - Simple and well-understood
# - Can find better minima
# - Requires more tuning
```

### AdamW (Adam with Weight Decay)

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01  # L2 regularization
)

# Benefits:
# - Like Adam but better regularization
# - Recommended for many tasks
```

### RMSprop

```python
optimizer = optim.RMSprop(
    model.parameters(),
    lr=0.001,
    alpha=0.9
)

# Benefits:
# - Good for RNNs
# - Adaptive learning rate
```

## 6.3 Learning Rate Schedulers

### ReduceLROnPlateau

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',       # Reduce when metric stops improving
    factor=0.1,       # Multiply lr by 0.1
    patience=5,       # Wait 5 epochs before reducing
    verbose=True
)

# In training loop:
scheduler.step(val_loss)  # Pass validation loss
```

### StepLR

```python
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,    # Reduce every 10 epochs
    gamma=0.1        # Multiply lr by 0.1
)

# In training loop:
scheduler.step()  # Called each epoch
```

### CosineAnnealingLR

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,        # Number of epochs for cycle
    eta_min=0        # Minimum learning rate
)
```

## 6.4 Optimizer Step-by-Step

```python
# ============ COMPLETE OPTIMIZER USAGE ============

# 1. Create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 2. In training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(batch)
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
```

---

# 7. DATA LOADING AND PREPROCESSING

## 7.1 Data Transformations

```python
from torchvision import transforms

# ============ BASIC TRANSFORMATIONS ============

# Convert image to tensor
to_tensor = transforms.ToTensor()

# Normalize (mean and std for CIFAR-10)
normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010]
)

# ============ DATA AUGMENTATION (Training) ============
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Flip images randomly
    transforms.RandomRotation(10),      # Rotate by up to 10 degrees
    transforms.ToTensor(),              # Convert to tensor
    transforms.Normalize(mean, std)     # Normalize
])

# ============ WITHOUT AUGMENTATION (Testing) ============
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

## 7.2 Loading Datasets

### CIFAR-10 Dataset

```python
from torchvision import datasets

# ============ TRAINING DATA ============
train_dataset = datasets.CIFAR10(
    root='./data',       # Where to save
    train=True,          # Training data
    download=True,       # Download if not exists
    transform=train_transform
)

# ============ TEST DATA ============
test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,         # Test data
    download=True,
    transform=test_transform
)

# ============ CLASS NAMES ============
classes = train_dataset.classes
print(classes)
# ['airplane', 'automobile', 'bird', 'cat', 'deer',
#  'dog', 'frog', 'horse', 'ship', 'truck']
```

## 7.3 DataLoaders

```python
from torch.utils.data import DataLoader

# ============ CREATE DATALOADERS ============
train_loader = DataLoader(
    train_dataset,
    batch_size=64,       # Number of samples per batch
    shuffle=True,        # Randomize order
    num_workers=2,       # Parallel loading
    pin_memory=True      # Faster GPU transfer
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,       # No shuffle for testing
    num_workers=2,
    pin_memory=True
)

# ============ ITERATE THROUGH DATALOADER ============
for batch_idx, (images, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx}:")
    print(f"  Images shape: {images.shape}")  # [64, 3, 32, 32]
    print(f"  Labels shape: {labels.shape}")  # [64]
    break
```

## 7.4 Custom Dataset (Advanced)

```python
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Load labels
        with open(labels_file, 'r') as f:
            self.samples = [line.strip().split() for line in f]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Get image path and label
        img_path, label = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_path)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Convert label to tensor
        label = torch.tensor(int(label))
        
        return image, label
```

---

# 8. COMPLETE TRAINING PIPELINE

## 8.1 Training Loop Structure

```python
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler, num_epochs, device, save_path):
    """
    Complete training pipeline
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs
        device: 'cuda' or 'cpu'
        save_path: Where to save best model
    
    Returns:
        model: Trained model
        history: Dictionary of training metrics
    """
    
    # ============ TRACKING ============
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    # ============ TRAINING LOOP ============
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # ------------ TRAINING PHASE ------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            # Print progress
            if batch_idx % 100 == 99:
                print(f"Batch {batch_idx+1}: Loss: {train_loss/100:.4f}")
                train_loss = 0.0
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100 * train_correct / train_total
        
        # ------------ VALIDATION PHASE ------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total
        
        # ------------ UPDATE SCHEDULER ------------
        if scheduler:
            scheduler.step(epoch_val_loss)
        
        # ------------ SAVE HISTORY ------------
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # ------------ PRINT RESULTS ------------
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # ------------ SAVE BEST MODEL ------------
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✓ Best model saved! (Acc: {best_val_acc:.2f}%)")
    
    print(f"\nTraining Complete! Best Val Acc: {best_val_acc:.2f}%")
    return model, history
```

## 8.2 Visualization of Training

```python
import matplotlib.pyplot as plt

def plot_training_history(history):
    """Plot loss and accuracy curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # ============ LOSS PLOT ============
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ============ ACCURACY PLOT ============
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
```

---

# 9. GPU ACCELERATION

## 9.1 Setting Up GPU

```python
import torch

# ============ CHECK GPU AVAILABILITY ============
def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("✓ Using CPU (CUDA not available)")
    return device

device = setup_device()
```

## 9.2 Moving Data to GPU

```python
# ============ MOVE MODEL ============
model = CIFAR10CNN().to(device)

# ============ MOVE DATA IN TRAINING LOOP ============
for inputs, targets in train_loader:
    inputs = inputs.to(device)
    targets = targets.to(device)
    # ... training code

# ============ MULTI-GPU TRAINING ============
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
```

## 9.3 GPU Memory Management

```python
# ============ CHECK GPU MEMORY ============
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# ============ CLEAR GPU CACHE ============
torch.cuda.empty_cache()

# ============ OPTIMIZE MEMORY USAGE ============
# Use smaller batch size if out of memory
# Use pin_memory=True in DataLoader
# Use torch.no_grad() during evaluation
```

---

# 10. MODEL EVALUATION AND TESTING

## 10.1 Evaluating on Test Set

```python
def evaluate_model(model, test_loader, device, classes):
    """Evaluate model on test data"""
    model.eval()
    correct = 0
    total = 0
    
    # Per-class accuracy
    class_correct = {c: 0 for c in range(len(classes))}
    class_total = {c: 0 for c in range(len(classes))}
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Per-class statistics
            for target, pred in zip(targets, predicted):
                if target == pred:
                    class_correct[target.item()] += 1
                class_total[target.item()] += 1
    
    # Overall accuracy
    overall_acc = 100 * correct / total
    print(f"Overall Test Accuracy: {overall_acc:.2f}%")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i in range(len(classes)):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"  {classes[i]:15s}: {acc:.2f}%")
    
    return overall_acc
```

## 10.2 Making Predictions on New Images

```python
def predict_image(model, image, device, classes):
    """Predict class for a single image"""
    model.eval()
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
    
    return classes[predicted.item()], confidence.item()
```

## 10.3 Displaying Results

```python
import matplotlib.pyplot as plt

def show_predictions(model, test_loader, device, classes, num_images=12):
    """Display predictions with images"""
    model.eval()
    
    # Get batch
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Predict
    images_gpu = images[:num_images].to(device)
    with torch.no_grad():
        outputs = model(images_gpu)
        _, predicted = outputs.max(1)
    
    # Display
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()
    
    for idx in range(num_images):
        # Denormalize image
        img = images[idx].cpu().numpy().transpose((1, 2, 0))
        img = img * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        
        axes[idx].imshow(img)
        axes[idx].set_title(
            f"True: {classes[labels[idx]]}\nPred: {classes[predicted[idx]]}",
            color='green' if predicted[idx] == labels[idx] else 'red'
        )
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
```

---

# 11. COMPLETE WORKING CODE

## 11.1 Full Implementation

```python
# ============================================
# CIFAR-10 Image Classifier - Complete Code
# Week 1 - PyTorch Fundamentals
# Author: Sanaullah
# Tech Prime Pvt Limited - Advanced AI/ML Internship
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

# ============================================
# 1. CONFIGURATION
# ============================================

class Config:
    # Data
    BATCH_SIZE = 64
    NUM_WORKERS = 2
    
    # Model
    NUM_CLASSES = 10
    
    # Training
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    
    # Paths
    DATA_ROOT = './data'
    MODEL_PATH = 'cifar10_best_model.pth'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# 2. DATA LOADING
# ============================================

def get_data():
    """Load CIFAR-10 dataset"""
    
    # Transformations
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=Config.DATA_ROOT,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=Config.DATA_ROOT,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    classes = train_dataset.classes
    
    return train_loader, test_loader, classes

# ============================================
# 3. MODEL ARCHITECTURE
# ============================================

class CIFAR10CNN(nn.Module):
    """CNN for CIFAR-10 classification"""
    
    def __init__(self):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        
        # Regularization
        self.dropout = nn.Dropout(0.5)
        
        # Activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Conv blocks
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ============================================
# 4. TRAINING FUNCTIONS
# ============================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        if batch_idx % 100 == 99:
            print(f"  Batch {batch_idx+1}: Loss: {running_loss/100:.4f}, Acc: {100*correct/total:.2f}%")
            running_loss = 0.0
    
    return 100 * correct / total

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    return val_loss / len(val_loader), 100 * correct / total

def train_full(model, train_loader, test_loader, config):
    """Complete training pipeline"""
    
    print("=" * 60)
    print("CIFAR-10 TRAINING")
    print(f"Device: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print("=" * 60)
    
    # Setup
    model = model.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Training history
    history = {'train_acc': [], 'val_acc': [], 'val_loss': []}
    best_acc = 0.0
    
    start_time = time.time()
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        print("-" * 40)
        
        # Train
        train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, config.DEVICE)
        
        # Save history
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        # Print results
        print(f"Train Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"✓ Best model saved! (Acc: {best_acc:.2f}%)")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    
    return model, history

# ============================================
# 5. EVALUATION FUNCTIONS
# ============================================

def evaluate_model(model, test_loader, classes, device):
    """Detailed evaluation"""
    model.eval()
    class_correct = {c: 0 for c in range(len(classes))}
    class_total = {c: 0 for c in range(len(classes))}
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            for target, pred in zip(targets, predicted):
                if target == pred:
                    class_correct[target.item()] += 1
                class_total[target.item()] += 1
    
    # Overall accuracy
    overall_acc = 100 * np.sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {overall_acc:.2f}%\n")
    
    print("Per-Class Accuracy:")
    for i in range(len(classes)):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"  {classes[i]:15s}: {acc:.2f}%")
    
    return overall_acc, all_preds, all_targets

# ============================================
# 6. VISUALIZATION
# ============================================

def plot_results(history):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_acc'], label='Train', linewidth=2)
    axes[0].plot(history['val_acc'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['val_loss'], label='Validation Loss', linewidth=2, color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300)
    plt.show()

def show_predictions(model, test_loader, classes, device, num_images=12):
    """Show sample predictions"""
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    images_gpu = images[:num_images].to(device)
    with torch.no_grad():
        outputs = model(images_gpu)
        _, predicted = outputs.max(1)
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()
    
    for idx in range(num_images):
        img = images[idx].cpu().numpy().transpose((1, 2, 0))
        img = img * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        
        axes[idx].imshow(img)
        axes[idx].set_title(
            f"True: {classes[labels[idx]]}\nPred: {classes[predicted[idx]]}",
            color='green' if predicted[idx] == labels[idx] else 'red'
        )
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=300)
    plt.show()

# ============================================
# 7. MAIN
# ============================================

def main():
    print("=" * 60)
    print("WEEK 1: CIFAR-10 IMAGE CLASSIFIER")
    print("Tech Prime Pvt Limited - Advanced AI/ML Internship")
    print("=" * 60)
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader, classes = get_data()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Classes: {classes}")
    
    # Create model
    print("\nCreating model...")
    model = CIFAR10CNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Train
    print("\nStarting training...")
    model, history = train_full(model, train_loader, test_loader, Config)
    
    # Load best model
    print("\nLoading best model...")
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    
    # Evaluate
    print("\nEvaluating...")
    evaluate_model(model, test_loader, classes, Config.DEVICE)
    
    # Visualize
    print("\nGenerating plots...")
    plot_results(history)
    show_predictions(model, test_loader, classes, Config.DEVICE)
    
    print("\n" + "=" * 60)
    print("✓ PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nFiles saved:")
    print(f"  - {Config.MODEL_PATH}")
    print("  - training_results.png")
    print("  - predictions.png")

if __name__ == "__main__":
    main()
```

---

# 12. COMMON ISSUES AND SOLUTIONS

## 12.1 Error: "CUDA out of memory"

```python
# SOLUTION 1: Reduce batch size
BATCH_SIZE = 32  # Instead of 64

# SOLUTION 2: Clear cache
torch.cuda.empty_cache()

# SOLUTION 3: Use gradient accumulation
accumulation_steps = 4
for i, (inputs, targets) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss = loss / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 12.2 Error: "RuntimeError: Expected object of device type cuda but got cpu"

```python
# SOLUTION: Make sure both model and data are on the same device
model = model.to(device)
inputs = inputs.to(device)
targets = targets.to(device)

# Check device
print(next(model.parameters()).device)  # Model device
print(inputs.device)                    # Data device
```

## 12.3 Error: "RuntimeError: shape mismatch"

```python
# SOLUTION: Check tensor shapes
print(inputs.shape)   # Should be [batch, channels, height, width]
print(labels.shape)   # Should be [batch]
print(outputs.shape)  # Should be [batch, num_classes]

# Fix: Reshape if needed
inputs = inputs.view(-1, 3, 32, 32)  # Reshape to correct dimensions
```

## 12.4 Model Not Learning

```python
# SOLUTION 1: Check learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Try 0.001

# SOLUTION 2: Check data normalization
# Make sure data is normalized
transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

# SOLUTION 3: Check loss function
# For classification, use CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

# SOLUTION 4: Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 12.5 Slow Training

```python
# SOLUTION 1: Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# SOLUTION 2: Increase batch size
BATCH_SIZE = 128  # If memory allows

# SOLUTION 3: Use DataLoader with num_workers
train_loader = DataLoader(..., num_workers=4, pin_memory=True)

# SOLUTION 4: Use mixed precision training (AMP)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

# 13. QUICK REFERENCE GUIDE

## 13.1 Common Code Patterns

```python
# ============ CREATE TENSOR ============
x = torch.tensor([1, 2, 3])
x = torch.zeros(3, 4)
x = torch.randn(3, 4)
x = torch.arange(10)

# ============ CHECK PROPERTIES ============
x.shape
x.dtype
x.device
x.numel()

# ============ MOVE TO DEVICE ============
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = x.to(device)
model = model.to(device)

# ============ CREATE MODEL ============
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    def forward(self, x):
        return self.fc(x)

# ============ TRAINING STEP ============
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()

# ============ EVALUATION ============
model.eval()
with torch.no_grad():
    outputs = model(inputs)
    _, predicted = outputs.max(1)

# ============ SAVE/LOAD MODEL ============
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
```

## 13.2 Common Layer Types

| Layer | Purpose | Example |
|-------|---------|---------|
| **Linear** | Fully connected | `nn.Linear(10, 5)` |
| **Conv2d** | Convolution (images) | `nn.Conv2d(3, 32, 3)` |
| **MaxPool2d** | Downsampling | `nn.MaxPool2d(2, 2)` |
| **BatchNorm2d** | Normalize batch | `nn.BatchNorm2d(32)` |
| **Dropout** | Prevent overfitting | `nn.Dropout(0.5)` |
| **ReLU** | Activation | `nn.ReLU()` |
| **Softmax** | Probabilities | `nn.Softmax(dim=1)` |

## 13.3 Common Optimizer Settings

```python
# ============ ADAM (Default) ============
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============ SGD ============
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# ============ ADAMW (With Weight Decay) ============
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

## 13.4 Common Loss Functions

```python
# Classification
criterion = nn.CrossEntropyLoss()

# Binary Classification
criterion = nn.BCEWithLogitsLoss()

# Regression
criterion = nn.MSELoss()
criterion = nn.L1Loss()
```

---

# 14. FINAL CHECKLIST

## Week 1 Completion Checklist

| Task | Status | Notes |
|------|--------|-------|
| Install PyTorch | ☐ | `pip install torch torchvision` |
| Understand Tensors | ☐ | Creating, manipulating, operations |
| Understand Autograd | ☐ | Gradients, backward pass |
| Build Model with nn.Module | ☐ | __init__, forward |
| Load CIFAR-10 Dataset | ☐ | DataLoader, transformations |
| Train Model | ☐ | Training loop, loss, optimizer |
| Evaluate Model | ☐ | Test accuracy, per-class accuracy |
| GPU Acceleration | ☐ | Move model and data to GPU |
| Save Model | ☐ | torch.save(), load |
| Visualize Results | ☐ | Plot training curves, predictions |
| Write README | ☐ | Documentation, usage, results |
| Push to GitHub | ☐ | Repository, commit, push |


## Key Concepts to Understand

- [ ] What is a tensor?
- [ ] What is autograd?
- [ ] How does a neural network work?
- [ ] What is a loss function?
- [ ] What is an optimizer?
- [ ] What is the training loop?
- [ ] How does a CNN work?
- [ ] What is GPU acceleration?

---

#  SUMMARY

## What I've Learned

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| Day 1 | PyTorch Basics | Tensors, Autograd |
| Day 2 | Neural Networks | nn.Module, Layers |
| Day 3 | Training | Loss, Optimizers |
| Day 4 | Data | Loading, Augmentation |
| Day 5 | Training Loop | Full pipeline |
| Day 6 | GPU | Acceleration |
| Day 7 | Project | CIFAR-10 Classifier |


---

**End of Week 1 Notes**

