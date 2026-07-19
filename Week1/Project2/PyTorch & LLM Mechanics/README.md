# Multi-Model Token Profiler & Memory Math Engine

**Tech Prime Pvt Limited - Advanced AI/ML Internship Program**
**Week 1: PyTorch & LLM Mechanics**

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Core Components](#core-components)
6. [Key Formulas](#key-formulas)
7. [Results & Benchmarks](#results--benchmarks)
8. [Usage Guide](#usage-guide)
9. [Troubleshooting](#troubleshooting)
10. [License](#license)

---

## Overview

A Python-based utility for profiling tokenization across multiple language models, estimating VRAM requirements for model serving, and projecting API costs for various deployment scenarios.

### Key Features

- Tokenization profiling across 7+ models (GPT-2 variants, BERT, RoBERTa, T5, Mistral-7B)
- VRAM estimation with multiple precision formats (FP32, FP16, BF16, INT8, INT4)
- KV-Cache memory footprint calculation
- API cost estimation with monthly projections
- Model comparison reports

### Supported Models

| Model | Parameters | Tokenizer | Context Window |
|-------|------------|-----------|----------------|
| GPT-2 | 124M | BPE | 1024 |
| GPT-2 Medium | 354M | BPE | 1024 |
| GPT-2 Large | 774M | BPE | 1024 |
| BERT Base | 110M | WordPiece | 512 |
| RoBERTa Base | 125M | BPE | 512 |
| T5 Small | 60M | SentencePiece | 512 |
| Mistral-7B | 7B | BPE | 8192 |

---

## Project Structure

```
llm_profiler/
├── llm_profiler.py          # Main application
├── README.md                # Documentation
├── requirements.txt         # Dependencies
├── examples/
│   └── sample_usage.py      # Example scripts
└── outputs/
    ├── tokenization_report.txt
    ├── vram_comparison.csv
    └── cost_projections.json
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for actual model loading)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/llm-profiler.git
cd llm-profiler

# Install dependencies
pip install torch transformers matplotlib numpy scikit-learn

# Verify installation
python -c "import transformers; print('Installation successful')"
```

### Dependencies

```
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

---

## Quick Start

```python
from llm_profiler import LLMProfiler

# Initialize profiler
profiler = LLMProfiler()

# Sample text
text = """
The Transformer architecture has revolutionized natural language processing.
At its core, the self-attention mechanism allows the model to weigh the
importance of different parts of the input sequence.
"""

# Run profiling pipeline
profiler.profile_pipeline(text, model_key='gpt2')

# Generate comprehensive report
profiler.generate_full_report(text)
```

### Sample Output

```
================================================================================
LLM PROFILING PIPELINE
================================================================================

1. TOKENIZATION
----------------------------------------
Model: GPT-2
Token Count: 149
Vocab Size: 50257

2. MEMORY ESTIMATION
----------------------------------------
Model: GPT-2
Parameters: 0.124B
Precision: fp16
Model Weights: 0.23 GB
KV-Cache: 0.07 GB
Activations: 0.06 GB
Total VRAM: 0.36 GB

3. COST ESTIMATION
----------------------------------------
Model: gpt-3.5-turbo
Input Tokens: 149
Output Tokens: 74.5
Total Cost: $0.000186

4. MONTHLY PROJECTION
----------------------------------------
Daily Requests: 100
Daily Tokens: 14,900
Monthly Tokens: 447,000
Monthly Cost: $0.37

================================================================================
```

---

## Core Components

### 1. TokenProfiler

Profiles text tokenization across different models.

```python
profiler = TokenProfiler()

# Count tokens for a single model
result = profiler.count_tokens(text, model_key='gpt2')

# Profile across all models
results = profiler.profile_document(text)

# Generate report
report = profiler.generate_report(text)
```

**Key Methods:**
- `count_tokens()`: Token count for specific model
- `profile_document()`: Compare tokenization across models
- `estimate_cost()`: Calculate API costs
- `generate_report()`: Formatted comparison report

### 2. MemoryMathEngine

Calculates VRAM requirements for model inference.

```python
engine = MemoryMathEngine()

# Calculate total VRAM
vram = engine.calculate_total_vram(
    model_key='llama-7b',
    seq_len=2048,
    batch_size=1,
    precision='fp16'
)

# Compare deployments
engine.compare_deployments(seq_len=2048)
```

**Key Methods:**
- `calculate_model_weights()`: Memory for parameters
- `calculate_kv_cache()`: KV-cache memory
- `calculate_activations()`: Activation memory
- `calculate_total_vram()`: Complete VRAM requirements

### 3. CostEstimator

Estimates API costs for LLM inference.

```python
estimator = CostEstimator()

# Single request cost
cost = estimator.estimate_cost(
    token_count=1000,
    model='gpt-3.5-turbo'
)

# Monthly projection
monthly = estimator.monthly_cost_projection(
    daily_tokens=10000,
    model='gpt-3.5-turbo'
)
```

---

## Key Formulas

### Token Count Estimation

```
English Text: Tokens ≈ Words × 1.3
General Text: Tokens ≈ Characters / 4
```

### VRAM Calculation

```
Total VRAM = Model_Weights + KV_Cache + Activations + Gradients
```

Where:

```
Model_Weights = Parameters × Bytes_per_Parameter
KV_Cache = 2 × Batch_Size × Heads × Seq_Len × Head_Dim × Layers × Bytes_per_Parameter
Activations ≈ Model_Weights × 0.25
```

### KV-Cache Growth

```
For LLaMA-7B (FP16):
- 512 context: 0.25 GB
- 1024 context: 0.50 GB
- 2048 context: 1.00 GB
- 4096 context: 2.00 GB
```

### Cost Estimation

```
Cost = (Input_Tokens / 1,000,000) × Input_Price + (Output_Tokens / 1,000,000) × Output_Price
```

---

## Results & Benchmarks

### Tokenization Comparison

| Model | Token Count | Vocab Size | Cost per 1k tokens |
|-------|-------------|------------|-------------------|
| BERT Base | 107 | 30,522 | $0.107 |
| T5 Small | 111 | 32,100 | $0.111 |
| Mistral-7B | 130 | 32,000 | $0.130 |
| GPT-2 | 149 | 50,257 | $0.149 |
| RoBERTa Base | 151 | 50,265 | $0.151 |

### VRAM Requirements (FP16)

| Model | 512 Context | 1024 Context | 2048 Context | 4096 Context |
|-------|-------------|--------------|--------------|--------------|
| LLaMA-7B | 16.55 GB | 16.80 GB | 17.30 GB | 18.30 GB |
| LLaMA-13B | 30.66 GB | 31.05 GB | 31.83 GB | 33.39 GB |
| LLaMA-70B | 164.23 GB | 165.48 GB | 167.98 GB | 172.98 GB |
| Mistral-7B | 16.55 GB | 16.80 GB | 17.30 GB | 18.30 GB |

### Quantization Impact (LLaMA-70B, 4096 Context)

| Precision | Memory | Reduction |
|-----------|--------|-----------|
| FP16 | 172.98 GB | 0% |
| INT8 | 86.49 GB | 50% |
| INT4 | 43.25 GB | 75% |

### API Cost Comparison

| Model | Input Price (per 1M) | Output Price (per 1M) | Avg per 1k tokens |
|-------|---------------------|----------------------|-------------------|
| GPT-3.5 Turbo | $0.50 | $1.50 | $0.001 |
| Llama-2-70B | $2.00 | $2.00 | $0.002 |
| Claude 3 Sonnet | $3.00 | $15.00 | $0.009 |
| GPT-4 | $30.00 | $60.00 | $0.045 |

---

## Usage Guide

### Basic Usage

```python
from llm_profiler import LLMProfiler

profiler = LLMProfiler()

# Profile a single document
text = "Your document text here..."
profiler.profile_pipeline(text, model_key='gpt2')
```

### Advanced Usage

```python
# Custom models list
models = ['gpt2', 'bert-base', 'roberta-base']

# Custom sequence lengths
seq_lens = [512, 1024, 2048, 4096, 8192]

# Generate comprehensive report
profiler.generate_full_report(
    text=text,
    models=models,
    seq_lens=seq_lens
)

# Export results
results = profiler.profile_pipeline(text)
```

### Cost Projection

```python
# Project monthly costs
projection = profiler.cost_estimator.monthly_cost_projection(
    daily_tokens=100000,  # 100,000 tokens per day
    model='gpt-3.5-turbo',
    output_ratio=0.5      # 50% output tokens
)

print(f"Monthly cost: ${projection['monthly_cost']:.2f}")
```

### VRAM Comparison

```python
# Compare models for deployment
profiler.memory_engine.compare_deployments(
    seq_len=2048,
    batch_size=1
)
```

---

## Troubleshooting

### Tokenizer Loading Issues

**Error:** `ValueError: Model not loaded`

**Solution:**
```python
# Use a fallback model
profiler.token_profiler.count_tokens(text, model_key='gpt2')
```

**Error:** `ConnectionError` when loading tokenizers

**Solution:**
- Check internet connection
- Use local tokenizer cache
- Set `use_fast=False` for some models

### Memory Estimation Issues

**Error:** Inaccurate VRAM estimates

**Solution:**
- Verify model configuration
- Account for memory fragmentation (add 10-20% buffer)
- Consider batch size impact

### Cost Calculation Issues

**Error:** `KeyError` for pricing model

**Solution:**
```python
# Check available models
print(profiler.cost_estimator.pricing.keys())
```

---

