
#  FastAPI + PyTorch: Professional Production-Ready Guide

> **Complete Reference:** From Zero to Production Deployment

---

##  Table of Contents

1. [PyTorch Fundamentals](#pytorch-fundamentals)
2. [PyTorch Core Components](#pytorch-core-components)
3. [PyTorch GPU & Device Management](#pytorch-gpu--device-management)
4. [PyTorch Data Loading & Processing](#pytorch-data-loading--processing)
5. [PyTorch Model Building](#pytorch-model-building)
6. [PyTorch Training Loop](#pytorch-training-loop)
7. [PyTorch Model Saving & Loading](#pytorch-model-saving--loading)
8. [FastAPI Fundamentals](#fastapi-fundamentals)
9. [FastAPI Core Imports & Usage](#fastapi-core-imports--usage)
10. [FastAPI Request & Response Handling](#fastapi-request--response-handling)
11. [FastAPI File Upload & Processing](#fastapi-file-upload--processing)
12. [FastAPI Middleware & CORS](#fastapi-middleware--cors)
13. [FastAPI Background Tasks](#fastapi-background-tasks)
14. [FastAPI Error Handling](#fastapi-error-handling)
15. [PyTorch + FastAPI Integration](#pytorch--fastapi-integration)
16. [Deployment & Production Checklist](#deployment--production-checklist)
17. [Complete Working Examples](#complete-working-examples)

---

##  PyTorch Fundamentals

### What is PyTorch?

PyTorch is an **open-source deep learning framework** developed by Facebook's AI Research lab. It provides:
- Dynamic computation graphs
- GPU acceleration
- Automatic differentiation
- Production-ready deployment

### Core Philosophy

PyTorch follows a **define-by-run** approach, meaning:
- The computational graph is built on the fly
- Python-native feel
- Easy debugging with standard Python tools

---

##  PyTorch Core Components

### `torch.tensor()`

**Definition:** The fundamental data structure in PyTorch. A tensor is a multi-dimensional matrix containing elements of a single data type.

**Purpose:**
- Store and manipulate data
- Perform mathematical operations
- Run on CPU or GPU

**Usage:**

```python
import torch

# Create tensors from Python lists
tensor_1d = torch.tensor([1, 2, 3, 4, 5])
tensor_2d = torch.tensor([[1, 2], [3, 4]])
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Create tensors with specific data types
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)

# Create tensors with random values
random_tensor = torch.randn(3, 4)  # Normal distribution
uniform_tensor = torch.rand(3, 4)   # Uniform distribution
zeros_tensor = torch.zeros(3, 4)    # All zeros
ones_tensor = torch.ones(3, 4)      # All ones

# Common operations
shape = tensor_2d.shape             # Get shape
size = tensor_2d.size()             # Get size
dtype = tensor_2d.dtype             # Get data type
device = tensor_2d.device           # Get device (CPU/GPU)
```

---

### `torch.cuda.is_available()`

**Definition:** Checks whether CUDA-compatible GPU is available for PyTorch operations.

**Purpose:**
- Detect GPU availability
- Enable conditional GPU usage
- Fallback to CPU when GPU is not available

**Usage:**

```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    print(f" GPU Available: {torch.cuda.get_device_name(0)}")
    print(f" GPU Count: {torch.cuda.device_count()}")
    print(f" Current CUDA Device: {torch.cuda.current_device()}")
    print(f" CUDA Version: {torch.version.cuda}")
else:
    print(" No GPU found. Running on CPU.")

# Conditional device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")
```

---

### `torch.device()`

**Definition:** Represents the device (CPU or GPU) where tensors and models will be placed.

**Purpose:**
- Specify computation device
- Standardize device selection
- Enable seamless CPU/GPU switching

**Usage:**

```python
import torch

# Create device objects
cpu_device = torch.device("cpu")
cuda_device = torch.device("cuda:0")  # First GPU
cuda_device_2 = torch.device("cuda:1")  # Second GPU

# Dynamic device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assign tensors to devices
tensor = torch.randn(3, 4)
tensor_on_device = tensor.to(device)  # Move tensor to device

# Assign models to devices
model = MyModel()
model = model.to(device)  # Move model to device

# Multiple GPUs
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)  # Parallel processing
```

---

### `var.to(device)`

**Definition:** Moves a tensor or model to a specified device (CPU or GPU).

**Purpose:**
- Enable GPU acceleration
- Ensure tensors are on the same device
- Optimize memory usage

**Usage:**

```python
import torch

# Create tensor on CPU
tensor_cpu = torch.randn(3, 4)

# Move to GPU (if available)
tensor_gpu = tensor_cpu.to(device)  # Device selection is dynamic

# Move back to CPU
tensor_cpu_again = tensor_gpu.to("cpu")

# In-place device assignment
tensor_cpu = tensor_cpu.to(device)  # Replaces the original

# Transfer model to device
model = torch.nn.Linear(10, 5)
model = model.to(device)

# Transfer model parameters
for param in model.parameters():
    param.data = param.data.to(device)
```

---

##  PyTorch Data Loading & Processing

### `torch.utils.data.Dataset`

**Definition:** An abstract class representing a dataset. Must be subclassed to create custom datasets.

**Usage:**

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label
```

---

### `torch.utils.data.DataLoader`

**Definition:** Combines a dataset and a sampler to provide iterable over the dataset.

**Purpose:**
- Batch processing
- Shuffle data
- Parallel loading
- Memory optimization

**Usage:**

```python
from torch.utils.data import DataLoader

# Create dataloader
dataloader = DataLoader(
    dataset=CustomDataset(data, labels),
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False
)

# Iterate over batches
for batch_idx, (data, labels) in enumerate(dataloader):
    data = data.to(device)
    labels = labels.to(device)
    # Process batch...
```

---

### Data Augmentation Transforms

```python
from torchvision import transforms

# Training transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Validation transforms
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

---

##  PyTorch Model Building

### `torch.nn.Module`

**Definition:** Base class for all neural network modules. Every custom model must inherit from this.

**Usage:**

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout1(x)
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x
```

### Common Layers

```python
# Linear layers
linear = nn.Linear(in_features=512, out_features=256)

# Convolutional layers
conv2d = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
conv1d = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
conv3d = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3)

# Pooling layers
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = nn.AdaptiveAvgPool2d((1, 1))

# Normalization layers
batch_norm = nn.BatchNorm2d(num_features=64)
layer_norm = nn.LayerNorm(normalized_shape=512)
dropout = nn.Dropout(p=0.5)

# Recurrent layers
lstm = nn.LSTM(input_size=100, hidden_size=256, num_layers=2)
gru = nn.GRU(input_size=100, hidden_size=256, num_layers=2)

# Activation functions
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
softmax = nn.Softmax(dim=1)
```

---

##  PyTorch Training Loop

### Complete Training Pipeline

```python
import torch
import torch.optim as optim

class TrainingPipeline:
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3
        )
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(val_loader)
        
        self.scheduler.step(avg_loss)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=100):
        best_accuracy = 0
        best_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"   Model saved (Accuracy: {best_accuracy:.2f}%)")
```

---

##  PyTorch Model Saving & Loading

### Save Methods

```python
import torch

# Method 1: Save entire model (state dict)
torch.save(model.state_dict(), 'model_weights.pth')

# Method 2: Save entire model with architecture
torch.save(model, 'full_model.pth')

# Method 3: Save checkpoint with training info
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'best_accuracy': best_accuracy
}
torch.save(checkpoint, 'checkpoint.pth')
```

### Load Methods

```python
import torch

# Method 1: Load state dict
model = SimpleCNN()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Method 2: Load full model
model = torch.load('full_model.pth')
model.eval()

# Method 3: Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
best_accuracy = checkpoint['best_accuracy']
```

### Model Export (ONNX)

```python
import torch.onnx

# Export to ONNX format
dummy_input = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

---

##  FastAPI Fundamentals

### What is FastAPI?

FastAPI is a **modern, fast web framework** for building APIs with Python. It provides:
- Automatic OpenAPI documentation
- Data validation with Pydantic
- Asynchronous support
- High performance
- Dependency injection

### Installation

```bash
# Core installation
pip install fastapi

# For production server
pip install uvicorn

# Additional dependencies
pip install python-multipart  # File uploads
pip install python-jose      # JWT tokens
pip install passlib          # Password hashing
pip install python-email     # Email validation
```

---

##  FastAPI Core Imports & Usage

### All Essential Imports

```python
# Core FastAPI imports
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder

# Request/Response handling
from fastapi import Request, Response, Cookie, Header, Query, Path, Body, Form, File, UploadFile

# HTTP status codes
from fastapi import status
from fastapi.exceptions import RequestValidationError

# Data validation
from pydantic import BaseModel, EmailStr, Field, validator

# Type hints
from typing import List, Optional, Dict, Any, Union, Callable

# Asynchronous support
import asyncio

# Context managers
from contextlib import asynccontextmanager

# Database imports
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, Session, declarative_base

# JWT authentication
from jose import JWTError, jwt

# Password hashing
from passlib.context import CryptContext

# Background tasks
from background import BackgroundTasks

# Middleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# File handling
import os
import shutil
from pathlib import Path

# Date and time
from datetime import datetime, timedelta

# JSON handling
import json

# Environment variables
import os
from dotenv import load_dotenv
```

---

##  FastAPI Application Setup

### Basic Application

```python
from fastapi import FastAPI
import uvicorn

# Create FastAPI app instance
app = FastAPI(
    title="AI API Service",
    description="Production-ready API for AI models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to AI API Service",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# Start server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
```

---

##  FastAPI Request & Response Handling

### Query Parameters

```python
from fastapi import Query

@app.get("/items/")
async def get_items(
    skip: int = Query(0, description="Number of items to skip"),
    limit: int = Query(10, description="Number of items to return"),
    category: Optional[str] = Query(None, description="Filter by category")
):
    return {"skip": skip, "limit": limit, "category": category}
```

### Path Parameters

```python
from fastapi import Path

@app.get("/items/{item_id}/")
async def get_item(
    item_id: int = Path(..., description="The ID of the item"),
    name: Optional[str] = None
):
    return {"item_id": item_id, "name": name}
```

### Request Body (Pydantic)

```python
from pydantic import BaseModel, Field

class Item(BaseModel):
    name: str = Field(..., description="Item name", min_length=1)
    price: float = Field(..., description="Item price", gt=0)
    quantity: int = Field(1, description="Item quantity", ge=1)
    tags: List[str] = Field(default=[], description="Item tags")
    
    @validator('price')
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v

@app.post("/items/")
async def create_item(item: Item):
    return {
        "message": "Item created successfully",
        "item": item.dict()
    }
```

### Response Models

```python
from pydantic import BaseModel
from typing import Optional

class ItemResponse(BaseModel):
    id: int
    name: str
    price: float
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True

@app.post("/items/", response_model=ItemResponse)
async def create_item(item: Item) -> ItemResponse:
    # Process and return response
    return ItemResponse(
        id=1,
        name=item.name,
        price=item.price,
        created_at=datetime.now()
    )
```

---

##  FastAPI File Upload & Processing

### Single File Upload

```python
from fastapi import File, UploadFile
import shutil
import os

@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(..., description="File to upload")
):
    # Check file size
    content = await file.read()
    file_size = len(content)
    
    # Check file extension
    allowed_extensions = [".jpg", ".png", ".jpeg"]
    file_extension = os.path.splitext(file.filename)[1]
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File extension {file_extension} not allowed"
        )
    
    # Save file
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "filename": file.filename,
        "size": file_size,
        "content_type": file.content_type,
        "saved_path": file_path
    }
```

### Multiple File Upload

```python
@app.post("/upload-multiple/")
async def upload_multiple_files(
    files: List[UploadFile] = File(..., description="Multiple files to upload")
):
    results = []
    for file in files:
        content = await file.read()
        results.append({
            "filename": file.filename,
            "size": len(content),
            "content_type": file.content_type
        })
    
    return {"files": results, "count": len(files)}
```

### Image Processing Upload

```python
from PIL import Image
import io

@app.post("/process-image/")
async def process_image(
    file: UploadFile = File(...)
):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Process image
    image = image.convert('RGB')
    image = image.resize((224, 224))
    
    # Save processed image
    processed_path = f"processed/{file.filename}"
    image.save(processed_path)
    
    return {
        "original_size": len(contents),
        "processed_size": os.path.getsize(processed_path),
        "dimensions": image.size
    }
```

### File Response (Send File to Client)

```python
from fastapi.responses import FileResponse

@app.get("/download/{file_path:path}")
async def download_file(file_path: str):
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type='application/octet-stream'
    )
```

---

##  FastAPI Middleware & CORS

### CORS Middleware

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (modify for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],
    max_age=3600
)
```

### Custom Middleware

```python
from fastapi import Request
import time

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    print(f"Response: {response.status_code}")
    return response
```

---

##  FastAPI Background Tasks

```python
from fastapi import BackgroundTasks

@app.post("/process-long-task/")
async def process_long_task(
    background_tasks: BackgroundTasks,
    data: Dict[str, Any]
):
    # Add task to background
    background_tasks.add_task(process_data, data)
    
    return {
        "message": "Task started in background",
        "status": "processing"
    }

def process_data(data: Dict[str, Any]):
    # Heavy processing here
    time.sleep(10)
    print(f"Processed data: {data}")
```

---

##  FastAPI Error Handling

### Custom Exception Handlers

```python
from fastapi.exceptions import HTTPException, RequestValidationError

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "path": str(request.url)
            }
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": 422,
                "message": "Validation error",
                "details": exc.errors(),
                "body": exc.body
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "detail": str(exc)
            }
        }
    )
```

### Custom Exceptions

```python
class CustomError(Exception):
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message

def raise_custom_error(code: int, message: str):
    raise HTTPException(status_code=code, detail=message)
```

---

##  FastAPI Authentication & Security

### OAuth2 with JWT

```python
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError
from passlib.context import CryptContext

# Setup
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# User model
class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

# Authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Token endpoint
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Check credentials
    if form_data.username != "admin":
        raise HTTPException(status_code=401, detail="Incorrect username")
    
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

# Protected endpoint
@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"username": username}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

---

##  PyTorch + FastAPI Integration

### Complete Working Example

```python
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import io
from PIL import Image
import numpy as np

# -------------------- PyTorch Model --------------------
class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -------------------- FastAPI App --------------------
app = FastAPI(title="AI Image Classifier API")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ImageClassifier(num_classes=10)
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model = model.to(device)
model.eval()

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Transformations
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# -------------------- API Endpoints --------------------
@app.get("/")
async def root():
    return {
        "message": "AI Image Classifier API",
        "model": "CNN Classifier",
        "classes": class_names,
        "device": str(device),
        "endpoints": {
            "classify": "/classify/ (POST)",
            "health": "/health (GET)"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Check image format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Transform image
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Move to device
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities).item() * 100
        
        # Return response
        return {
            "predicted_class": class_names[predicted_class.item()],
            "class_index": predicted_class.item(),
            "confidence": round(confidence, 2),
            "all_probabilities": {
                name: round(prob.item() * 100, 2) 
                for name, prob in zip(class_names, probabilities[0])
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-classify/")
async def batch_classify(files: List[UploadFile] = File(...)):
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
                confidence = torch.max(probabilities).item() * 100
            
            results.append({
                "filename": file.filename,
                "predicted_class": class_names[predicted_class.item()],
                "confidence": round(confidence, 2)
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results, "total": len(files)}

# -------------------- Startup Event --------------------
@app.on_event("startup")
async def startup_event():
    print(f" Model loaded on device: {device}")
    print(f" Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f" Server ready!")

# -------------------- Shutdown Event --------------------
@app.on_event("shutdown")
async def shutdown_event():
    print(" Server shutting down...")

# -------------------- Run Application --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

##  Deployment & Production Checklist

### Environment Variables

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI API Service"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    DATABASE_URL: str
    MODEL_PATH: str = "model_weights.pth"
    MODEL_DEVICE: str = "cuda"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png"]
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### Production Server

```bash
# Run with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Run with gunicorn (recommended)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app

# Run with Docker
docker build -t ai-api .
docker run -p 8000:8000 ai-api
```

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### requirements.txt

```txt
fastapi==0.104.1
uvicorn==0.24.0
torch==2.0.1
torchvision==0.15.2
pillow==10.1.0
python-multipart==0.0.6
python-dotenv==1.0.0
pydantic==2.4.2
pydantic-settings==2.0.3
python-jose==3.3.0
passlib==1.7.4
SQLAlchemy==2.0.22
```

---

##  Complete Performance Metrics

### PyTorch Performance

```python
import time

class PerformanceMonitor:
    @staticmethod
    def measure_inference_time(model, tensor, iterations=100):
        model.eval()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(tensor)
        
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / iterations
        
        return {
            "total_time": elapsed_time,
            "average_time": avg_time,
            "fps": 1 / avg_time
        }
    
    @staticmethod
    def get_model_size(model):
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        return {"size_mb": round(size_mb, 2)}
```

---

##  Quick Reference: FastAPI Essential Methods

| Method/Function | Purpose |
|-----------------|---------|
| `FastAPI()` | Create application instance |
| `@app.get()` | GET request handler |
| `@app.post()` | POST request handler |
| `@app.put()` | PUT request handler |
| `@app.delete()` | DELETE request handler |
| `@app.patch()` | PATCH request handler |
| `@app.options()` | OPTIONS request handler |
| `@app.head()` | HEAD request handler |
| `@app.middleware()` | Middleware decorator |
| `@app.exception_handler()` | Exception handler decorator |
| `@app.on_event()` | Event handler (startup/shutdown) |
| `Depends()` | Dependency injection |
| `HTTPException()` | Raise HTTP errors |
| `FileResponse()` | Return file response |
| `JSONResponse()` | Return JSON response |
| `StreamingResponse()` | Return streaming response |
| `BackgroundTasks()` | Background task management |
| `UploadFile` | File upload handling |

---

##  Quick Reference: PyTorch Essential Methods

| Method/Function | Purpose |
|-----------------|---------|
| `torch.tensor()` | Create tensor from data |
| `torch.cuda.is_available()` | Check GPU availability |
| `torch.device()` | Specify computation device |
| `tensor.to(device)` | Move tensor to device |
| `nn.Module()` | Base class for models |
| `nn.Linear()` | Fully connected layer |
| `nn.Conv2d()` | 2D convolutional layer |
| `nn.MaxPool2d()` | Max pooling layer |
| `nn.BatchNorm2d()` | Batch normalization |
| `nn.Dropout()` | Dropout regularization |
| `nn.ReLU()` | ReLU activation |
| `nn.Softmax()` | Softmax activation |
| `optim.Adam()` | Adam optimizer |
| `optim.SGD()` | SGD optimizer |
| `DataLoader()` | Data batching |
| `Dataset()` | Custom dataset creation |
| `torch.save()` | Save model |
| `torch.load()` | Load model |
| `torch.onnx.export()` | Export to ONNX format |

---

## ✅ Deployment Checklist

- [ ] ✅ Model validation and testing complete
- [ ] ✅ CORS configuration correct
- [ ] ✅ Environment variables set
- [ ] ✅ Authentication enabled (if required)
- [ ] ✅ File size limits configured
- [ ] ✅ Logging implemented
- [ ] ✅ Error handling comprehensive
- [ ] ✅ Rate limiting applied (if required)
- [ ] ✅ SSL/TLS enabled (production)
- [ ] ✅ Monitoring and metrics set up
- [ ] ✅ Documentation updated
- [ ] ✅ Performance testing done
- [ ] ✅ Security vulnerabilities checked
- [ ] ✅ Database migrations done
- [ ] ✅ Backup strategy in place

---

> 💡 **Pro Tip**: Always use environment variables for sensitive information. Implement comprehensive error handling. Use async endpoints for I/O operations. Test with real-world data before production deployment!

---

**END OF COMPLETE REFERENCE GUIDE**

