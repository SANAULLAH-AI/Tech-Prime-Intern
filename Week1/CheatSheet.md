

##  PyTorch Cheat Sheet & Notes

---

### 1. What is PyTorch?
- **PyTorch** = Open-source deep learning framework by Facebook
- **Key Features:**
  - Dynamic computation graphs (define-by-run)
  - GPU acceleration
  - Automatic differentiation
  - Python-native feel

---

### 2. Core Tensor Operations

| Function | Purpose | Example |
|----------|---------|---------|
| `torch.tensor(data)` | Create tensor from list/array | `torch.tensor([1,2,3])` |
| `torch.zeros(shape)` | Create tensor with zeros | `torch.zeros(3,4)` |
| `torch.ones(shape)` | Create tensor with ones | `torch.ones(3,4)` |
| `torch.rand(shape)` | Random values (0 to 1) | `torch.rand(3,4)` |
| `torch.randn(shape)` | Random normal distribution | `torch.randn(3,4)` |
| `torch.eye(n)` | Identity matrix | `torch.eye(3)` |
| `torch.arange(start,end)` | Range of values | `torch.arange(0,10)` |
| `torch.linspace(start,end,steps)` | Linearly spaced values | `torch.linspace(0,1,5)` |

---

### 3. Tensor Properties & Methods

| Property/Method | Description | Example |
|-----------------|-------------|---------|
| `.shape` | Get tensor dimensions | `tensor.shape` → `(3,4)` |
| `.size()` | Get tensor dimensions | `tensor.size()` → `torch.Size([3,4])` |
| `.dtype` | Get data type | `tensor.dtype` → `torch.float32` |
| `.device` | Get device (CPU/GPU) | `tensor.device` → `device(type='cuda')` |
| `.numpy()` | Convert to NumPy | `tensor.numpy()` |
| `.item()` | Get single value | `tensor.item()` → `5.0` |
| `.tolist()` | Convert to Python list | `tensor.tolist()` |
| `.to(device)` | Move to device | `tensor.to('cuda')` |
| `.clone()` | Create a copy | `tensor.clone()` |
| `.detach()` | Remove from computation graph | `tensor.detach()` |

---

### 4. GPU / Device Management

| Function | Purpose | Example |
|----------|---------|---------|
| `torch.cuda.is_available()` | Check GPU availability | `torch.cuda.is_available()` → `True/False` |
| `torch.cuda.device_count()` | Number of GPUs | `torch.cuda.device_count()` → `1` |
| `torch.cuda.current_device()` | Current GPU ID | `torch.cuda.current_device()` → `0` |
| `torch.cuda.get_device_name(0)` | GPU name | `torch.cuda.get_device_name(0)` |
| `torch.device('cuda')` | Create device object | `device = torch.device('cuda')` |
| `torch.cuda.empty_cache()` | Clear GPU memory | `torch.cuda.empty_cache()` |

**Device Setup Pattern:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = tensor.to(device)
model = model.to(device)
```

---

### 5. Neural Network Layers (`torch.nn`)

| Layer | Purpose | Example |
|-------|---------|---------|
| `nn.Linear(in, out)` | Fully connected | `nn.Linear(784, 256)` |
| `nn.Conv2d(in, out, kernel)` | 2D Convolution | `nn.Conv2d(3, 64, 3)` |
| `nn.Conv1d(in, out, kernel)` | 1D Convolution | `nn.Conv1d(1, 64, 3)` |
| `nn.MaxPool2d(kernel, stride)` | Max pooling | `nn.MaxPool2d(2, 2)` |
| `nn.AvgPool2d(kernel, stride)` | Average pooling | `nn.AvgPool2d(2, 2)` |
| `nn.AdaptiveAvgPool2d(output)` | Adaptive pooling | `nn.AdaptiveAvgPool2d((1,1))` |
| `nn.BatchNorm2d(channels)` | Batch normalization | `nn.BatchNorm2d(64)` |
| `nn.LayerNorm(shape)` | Layer normalization | `nn.LayerNorm(512)` |
| `nn.Dropout(p)` | Dropout regularization | `nn.Dropout(0.5)` |
| `nn.Flatten()` | Flatten tensor | `nn.Flatten()` |
| `nn.Embedding(vocab, dim)` | Embedding layer | `nn.Embedding(1000, 256)` |
| `nn.LSTM(input, hidden, layers)` | LSTM | `nn.LSTM(100, 256, 2)` |
| `nn.GRU(input, hidden, layers)` | GRU | `nn.GRU(100, 256, 2)` |

---

### 6. Activation Functions

| Function | Purpose | When to Use |
|----------|---------|-------------|
| `nn.ReLU()` | Max(0, x) | **Most common** in hidden layers |
| `nn.LeakyReLU()` | Max(0.01x, x) | To avoid dead neurons |
| `nn.Sigmoid()` | 1/(1+e^(-x)) | Binary classification output |
| `nn.Tanh()` | (e^x - e^(-x))/(e^x + e^(-x)) | Between -1 and 1 |
| `nn.Softmax(dim)` | Probability distribution | Multi-class output |
| `nn.LogSoftmax(dim)` | Log of Softmax | Numerically stable with NLLLoss |

**Functional versions:**
```python
import torch.nn.functional as F

F.relu(x)
F.sigmoid(x)
F.softmax(x, dim=1)
F.tanh(x)
F.log_softmax(x, dim=1)
```

---

### 7. Loss Functions (`torch.nn`)

| Loss Function | Purpose | Example |
|---------------|---------|---------|
| `nn.MSELoss()` | Mean Squared Error | Regression problems |
| `nn.L1Loss()` | Mean Absolute Error | Regression (robust) |
| `nn.CrossEntropyLoss()` | Multi-class classification | **Most common** for classification |
| `nn.BCEWithLogitsLoss()` | Binary classification | Binary classification with logits |
| `nn.BCELoss()` | Binary classification | Binary with probabilities |
| `nn.NLLLoss()` | Negative Log Likelihood | Used with LogSoftmax |
| `nn.KLDivLoss()` | KL Divergence | Distribution matching |
| `nn.HuberLoss()` | Combination of MSE & L1 | Robust regression |

---

### 8. Optimizers (`torch.optim`)

| Optimizer | Purpose | Example |
|-----------|---------|---------|
| `optim.Adam()` | Adaptive Moment Estimation | **Most common** |
| `optim.SGD()` | Stochastic Gradient Descent | Basic optimizer |
| `optim.AdamW()` | Adam with weight decay | Better regularization |
| `optim.RMSprop()` | RMSprop | Good for RNNs |
| `optim.Adagrad()` | Adaptive gradient | Sparse data |
| `optim.Adamax()` | Adam with infinity norm | Alternative to Adam |

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

---

### 9. Learning Rate Schedulers

| Scheduler | Purpose | Example |
|-----------|---------|---------|
| `optim.lr_scheduler.StepLR` | Step decay | `StepLR(optimizer, step_size=10, gamma=0.1)` |
| `optim.lr_scheduler.MultiStepLR` | Multi-step decay | `MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)` |
| `optim.lr_scheduler.ReduceLROnPlateau` | Reduce on plateau | `ReduceLROnPlateau(optimizer, patience=3)` |
| `optim.lr_scheduler.CosineAnnealingLR` | Cosine decay | `CosineAnnealingLR(optimizer, T_max=50)` |

---

### 10. Data Handling

**Dataset:**
```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

**DataLoader:**
```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

---

### 11. Model Saving & Loading

| Operation | Code |
|-----------|------|
| **Save state dict** | `torch.save(model.state_dict(), 'model.pth')` |
| **Load state dict** | `model.load_state_dict(torch.load('model.pth'))` |
| **Save full model** | `torch.save(model, 'model.pth')` |
| **Load full model** | `model = torch.load('model.pth')` |
| **Save checkpoint** | `torch.save({'epoch': e, 'state_dict': model.state_dict()}, 'ckpt.pth')` |
| **Load checkpoint** | `ckpt = torch.load('ckpt.pth')` |
| **Export ONNX** | `torch.onnx.export(model, dummy, 'model.onnx')` |

---

### 12. Training Loop Template

```python
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
```

---

##  FastAPI Cheat Sheet & Notes

---

### 1. What is FastAPI?
- **FastAPI** = Modern web framework for building APIs
- **Key Features:**
  - High performance (on par with NodeJS/Go)
  - Automatic OpenAPI documentation
  - Data validation with Pydantic
  - Async support
  - Type hints based

---

### 2. Core Imports

```python
# Core
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# Request/Response
from fastapi import Request, Response, Query, Path, Body, Form, File, UploadFile
from fastapi import Cookie, Header

# Data Validation
from pydantic import BaseModel, Field, EmailStr, validator

# Type Hints
from typing import List, Optional, Dict, Any, Union, Callable

# File/Path
import os
import shutil
from pathlib import Path

# Date/Time
from datetime import datetime, timedelta

# Async
import asyncio

# JSON
import json
```

---

### 3. Application Setup

```python
app = FastAPI(
    title="API Title",
    description="API Description",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc",    # ReDoc
)

@app.on_event("startup")
async def startup():
    print("Server starting...")

@app.on_event("shutdown")
async def shutdown():
    print("Server shutting down...")
```

---

### 4. HTTP Methods

| Method | Decorator | Purpose |
|--------|-----------|---------|
| **GET** | `@app.get()` | Retrieve data |
| **POST** | `@app.post()` | Create data |
| **PUT** | `@app.put()` | Update entire resource |
| **PATCH** | `@app.patch()` | Partial update |
| **DELETE** | `@app.delete()` | Delete data |
| **HEAD** | `@app.head()` | Headers only |
| **OPTIONS** | `@app.options()` | Allowed methods |

---

### 5. Endpoint Parameters

| Parameter Type | Decorator | Description |
|----------------|-----------|-------------|
| **Query** | `Query()` | URL parameters |
| **Path** | `Path()` | URL path parameters |
| **Body** | `Body()` | JSON request body |
| **Form** | `Form()` | Form data |
| **File** | `File()` | File upload |
| **Header** | `Header()` | Request headers |
| **Cookie** | `Cookie()` | Cookies |
| **Dependency** | `Depends()` | Dependency injection |

---

### 6. Request/Response Models

**Pydantic Models:**
```python
class ItemCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)
    description: Optional[str] = None
    
    @validator('price')
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v

class ItemResponse(BaseModel):
    id: int
    name: str
    price: float
    created_at: datetime
    
    class Config:
        orm_mode = True
```

**Response Status Codes:**
```python
from fastapi import status

@app.post("/items/", status_code=status.HTTP_201_CREATED)
async def create_item():
    return item
```

---

### 7. File Upload

**Single File:**
```python
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    # Process contents
    return {"filename": file.filename, "size": len(contents)}
```

**Multiple Files:**
```python
@app.post("/upload-multiple/")
async def upload_multiple(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        content = await file.read()
        results.append({"filename": file.filename, "size": len(content)})
    return {"results": results}
```

---

### 8. Response Types

| Response | Purpose | Example |
|----------|---------|---------|
| `JSONResponse()` | JSON data | `return JSONResponse(content={"key": "value"})` |
| `FileResponse()` | File download | `return FileResponse(path="file.pdf")` |
| `HTMLResponse()` | HTML page | `return HTMLResponse(content="<h1>Hello</h1>")` |
| `StreamingResponse()` | Stream data | `return StreamingResponse(generator())` |
| `RedirectResponse()` | Redirect | `return RedirectResponse(url="/docs")` |
| `PlainTextResponse()` | Plain text | `return PlainTextResponse(content="Hello")` |

---

### 9. Middleware

**CORS:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Custom Middleware:**
```python
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(time.time() - start)
    return response
```

---

### 10. Error Handling

**HTTP Exceptions:**
```python
raise HTTPException(
    status_code=status.HTTP_404_NOT_FOUND,
    detail="Item not found"
)
```

**Custom Exception Handlers:**
```python
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )
```

---

### 11. Authentication (JWT)

```python
from fastapi.security import OAuth2PasswordBearer
from jose import jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Validate credentials
    token = jwt.encode({"sub": form_data.username}, SECRET_KEY)
    return {"access_token": token, "token_type": "bearer"}

@app.get("/protected")
async def protected(token: str = Depends(oauth2_scheme)):
    # Verify token
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return {"user": payload["sub"]}
```

---

### 12. Background Tasks

```python
from fastapi import BackgroundTasks

@app.post("/process/")
async def process(background_tasks: BackgroundTasks):
    background_tasks.add_task(heavy_task, data)
    return {"message": "Processing started"}

def heavy_task(data):
    # Heavy processing here
    time.sleep(10)
```

---

### 13. Dependencies

```python
def common_parameters(
    skip: int = Query(0),
    limit: int = Query(10)
):
    return {"skip": skip, "limit": limit}

@app.get("/items/")
async def get_items(params: dict = Depends(common_parameters)):
    return params
```

---

### 14. Validation

| Validator | Purpose | Example |
|-----------|---------|---------|
| `min_length` | Minimum string length | `Field(..., min_length=3)` |
| `max_length` | Maximum string length | `Field(..., max_length=100)` |
| `gt` | Greater than | `Field(..., gt=0)` |
| `ge` | Greater or equal | `Field(..., ge=1)` |
| `lt` | Less than | `Field(..., lt=100)` |
| `le` | Less or equal | `Field(..., le=100)` |
| `regex` | Regular expression | `Field(..., regex="^[a-z]+$")` |
| `@validator` | Custom validation | `def validate_price(cls, v):` |

---

### 15. Running the Server

```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000

# With gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

---

##  PyTorch + FastAPI Integration Cheat Sheet

### Complete Flow

```python
# 1. Load PyTorch Model
import torch
model = MyModel().to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 2. FastAPI Endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # 3. Preprocess
    image = process_image(await file.read())
    tensor = tensor.to(device)
    
    # 4. Inference
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1)
    
    # 5. Return Response
    return {"prediction": pred.item()}
```

---

##  Quick Reference Cards

### PyTorch Quick Reference

| Category | Key Functions |
|----------|---------------|
| **Create** | `tensor()`, `zeros()`, `ones()`, `rand()`, `randn()` |
| **Move** | `.to()`, `.cuda()`, `.cpu()` |
| **Operate** | `+`, `-`, `*`, `/`, `matmul()`, `sum()`, `mean()` |
| **Reshape** | `.view()`, `.reshape()`, `.flatten()`, `.squeeze()`, `.unsqueeze()` |
| **NN** | `nn.Linear()`, `nn.Conv2d()`, `nn.ReLU()`, `nn.Dropout()` |
| **Loss** | `nn.CrossEntropyLoss()`, `nn.MSELoss()` |
| **Optim** | `optim.Adam()`, `optim.SGD()` |
| **Save/Load** | `torch.save()`, `torch.load()` |

### FastAPI Quick Reference

| Category | Key Functions |
|----------|---------------|
| **App** | `FastAPI()`, `@app.get()`, `@app.post()` |
| **Params** | `Query()`, `Path()`, `Body()`, `Form()`, `File()` |
| **Models** | `BaseModel`, `Field()`, `@validator` |
| **Response** | `JSONResponse`, `FileResponse`, `HTMLResponse` |
| **Middleware** | `CORSMiddleware`, `@app.middleware()` |
| **Errors** | `HTTPException()`, `@app.exception_handler()` |
| **Security** | `OAuth2PasswordBearer`, `Depends()` |
| **Tasks** | `BackgroundTasks` |

---
