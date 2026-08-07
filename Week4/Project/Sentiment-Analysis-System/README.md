
# Sentiment Analysis System

A production-ready sentiment analysis system for movie reviews using deep learning.

## Overview

This system analyzes sentiment (positive/negative) of movie reviews using 5 different deep learning models. Built with PyTorch and HuggingFace Transformers, it achieves 89.4% accuracy on the IMDB dataset.

## Results and Visualizations

### Model Performance Comparison
![Model Performance Comparison](https://raw.githubusercontent.com/SANAULLAH-AI/Tech-Prime-Intern/main/Week4/Project/Sentiment-Analysis-System/model_comparison_day5.png)

### Final Ensemble Comparison
![Final Ensemble Comparison](https://raw.githubusercontent.com/SANAULLAH-AI/Tech-Prime-Intern/main/Week4/Project/Sentiment-Analysis-System/final_comparison_ensemble.png)

### Confusion Matrix (DistilBERT)
![Confusion Matrix](https://raw.githubusercontent.com/SANAULLAH-AI/Tech-Prime-Intern/main/Week4/Project/Sentiment-Analysis-System/confusion_matrix_distilbert.png)

### Dataset Audit
![Dataset Audit](https://raw.githubusercontent.com/SANAULLAH-AI/Tech-Prime-Intern/main/Week4/Project/Sentiment-Analysis-System/dataset_audit_day5.png)

## Models Performance

| Model | Accuracy | Parameters | Training Time |
|-------|----------|------------|---------------|
| DistilBERT | 89.40% | 66.9M | 14 min |
| BiGRU | 85.99% | 2.47M | 11 min |
| GRU | 85.95% | 2.18M | 10 min |
| BiLSTM | 85.83% | 2.63M | 14 min |
| LSTM | 84.91% | 2.25M | 12 min |
| Voting Ensemble | 89.03% | - | - |

## Features

- Real-time sentiment prediction
- Confidence scores for predictions
- 5 different model architectures
- Ensemble predictions (voting + weighted)
- Professional web interface
- REST API ready

## How It Works

1. User enters a movie review
2. Text is preprocessed and tokenized
3. Model predicts sentiment (Positive/Negative)
4. Confidence score is displayed

## Dataset

- IMDB Dataset of 50,000 movie reviews
- Balanced: 50% positive, 50% negative
- Average length: 231 words
- Cleaned and preprocessed for all models

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Web Interface
```python
python app.py
```

### API
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
├── app.py                 # Web interface
├── main.py               # API endpoint
├── deployment_model/     # Saved model files
├── train_data.csv        # Training data
├── test_data.csv         # Test data
└── requirements.txt      # Dependencies
```

## Technologies

- PyTorch
- HuggingFace Transformers
- Gradio
- FastAPI
- Scikit-learn
- Pandas, NumPy

## Model Selection Guide

- **Best Performance**: DistilBERT (89.4%)
- **Best Speed/Accuracy**: BiGRU (85.99%)
- **Fastest Training**: GRU (10 mins)
- **Smallest Model**: GRU (2.18M parameters)
- **Production Choice**: DistilBERT or BiGRU

## Acknowledgments

- IMDB Dataset for research use
- HuggingFace for pre-trained models
- Open-source community


