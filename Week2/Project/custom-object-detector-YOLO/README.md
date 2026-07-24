
# Custom Object Detection System

A production-ready object detection system using YOLOv8 pre-trained on the COCO dataset. This system provides real-time object detection through a professional web interface with comprehensive analytics and visualization capabilities.

## System Overview

This project implements a complete object detection pipeline using the YOLOv8 (You Only Look Once) architecture. The system leverages transfer learning through a pre-trained model on the COCO dataset, enabling detection of 80 different object classes without requiring custom training. The solution features a responsive web interface, detailed detection analytics, and batch processing capabilities.

## Key Features

- **80 Object Classes Detection**: Supports detection of people, vehicles, animals, indoor objects, food items, sports equipment, and accessories
- **Real-time Inference**: Optimized for fast processing with GPU acceleration support
- **Interactive Web Interface**: User-friendly interface with confidence threshold adjustment
- **Comprehensive Analytics**: Detailed detection reports with class-wise breakdown, confidence statistics, and object size analysis
- **Visualization**: Professional bounding box rendering with class labels and confidence scores
- **Batch Processing**: Process multiple images simultaneously with results aggregation
- **Performance Metrics**: Real-time display of inference time, FPS, and model statistics
- **Export Capabilities**: Save annotated images and detection summaries

## Technical Architecture

### Model Specifications
- **Architecture**: YOLOv8 (Nano version)
- **Training Dataset**: COCO (Common Objects in Context)
- **Total Classes**: 80
- **Model Size**: 6.25 MB
- **Inference Device**: GPU/CPU (automatically detected)
- **Framework**: PyTorch with Ultralytics YOLO implementation

### Technology Stack
- **Core Framework**: PyTorch 2.10+
- **Model Library**: Ultralytics YOLOv8
- **Web Interface**: Gradio 3.x
- **Image Processing**: OpenCV, Pillow
- **Visualization**: Matplotlib, Seaborn
- **Data Handling**: NumPy, JSON
- **Environment**: Kaggle Notebook / Python 3.12+

## Installation

### Prerequisites
```bash
Python 3.8+
CUDA-capable GPU (optional, for faster inference)
Kaggle Notebook or Local Python Environment
```

### Dependencies
```bash
pip install -r requirements.txt
```

### requirements.txt
```txt
ultralytics>=8.0.0
opencv-python-headless>=4.5.0
gradio>=3.50.2
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
pillow>=8.0.0
```

## Quick Start

### Using the Web Interface
1. Upload an image using the file upload area
2. Adjust the confidence threshold (0.5 recommended for balanced results)
3. Click the Submit button
4. View detection results with bounding boxes
5. Access detailed detection report with statistics


### Basic Detection
```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('yolov8n.pt')

# Perform detection
results = model('image.jpg', conf=0.5)

# Extract detections
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()
```

### Batch Processing
```python
from detector import detect_objects

# Process multiple images
images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
for img_path in images:
    image = cv2.imread(img_path)
    annotated, detections, summary = detect_objects(image, conf=0.5)
    # Save or display results
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| Model Size | 6.25 MB |
| Total Classes | 80 |
| Inference Time | < 100ms (GPU) |
| FPS | 10+ (GPU) |
| mAP | 50%+ (COCO validation) |
| Memory Usage | < 1 GB |

