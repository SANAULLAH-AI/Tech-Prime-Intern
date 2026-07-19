# COMPLETE WEEK 1 NOTES - PyTorch & LLM Mechanics

## Professional Training Documentation
**Tech Prime Pvt Limited - Advanced AI/ML Internship Program**

---

# 📑 TABLE OF CONTENTS

1. [Introduction to LLM Mechanics](#1-introduction-to-llm-mechanics)
2. [Tokenization - The Foundation](#2-tokenization---the-foundation)
3. [Embeddings - From Tokens to Vectors](#3-embeddings---from-tokens-to-vectors)
4. [Self-Attention Mechanics](#4-self-attention-mechanics)
5. [KV-Cache - Memory Optimization](#5-kv-cache---memory-optimization)
6. [VRAM Requirements - Memory Math](#6-vram-requirements---memory-math)
7. [Precision Formats Explained](#7-precision-formats-explained)
8. [Context Windows and Scaling](#8-context-windows-and-scaling)
9. [Complete Working Code](#9-complete-working-code)
10. [Common Issues and Solutions](#10-common-issues-and-solutions)
11. [Quick Reference Guide](#11-quick-reference-guide)
12. [Final Checklist](#12-final-checklist)

---

# 1. INTRODUCTION TO LLM MECHANICS

## 1.1 What Makes LLMs Different?

**Definition:** Large Language Models (LLMs) are neural networks trained on massive text datasets to predict the next token in a sequence.

**Key Differences from Traditional ML:**

| Traditional ML | LLMs |
|----------------|------|
| Fixed input size | Variable context window |
| Simple features | Complex semantic representations |
| Single task | Multi-task capabilities |
| Small datasets | Billions of parameters |

## 1.2 The Transformer Architecture

```
                    ┌─────────────────────────┐
                    │     Output Probabilities │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │      Softmax            │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │      Linear             │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Add & Norm            │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Feed Forward          │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Add & Norm            │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Multi-Head Attention  │  ← Self-Attention
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Positional Encoding   │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Input Embedding       │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Input Tokens          │
                    └─────────────────────────┘
```

---

# 2. TOKENIZATION - THE FOUNDATION

## 2.1 What is Tokenization?

**Definition:** Tokenization is the process of converting raw text into discrete units (tokens) that the model can process.

**Simple Analogy:**
- Like breaking a sentence into words
- But LLMs use "subwords" for efficiency
- "unbelievable" → "un" + "believ" + "able"

## 2.2 Tokenizer Types

### Byte-Pair Encoding (BPE)

```python
# BPE Algorithm Steps:
# 1. Start with characters
# 2. Find most frequent pair
# 3. Merge into new token
# 4. Repeat

# Example:
"low" → "l" + "o" + "w"
"lowest" → "low" + "est"
```

### WordPiece (BERT-style)

```python
# WordPiece Algorithm:
# 1. Start with characters
# 2. Merge based on likelihood
# 3. Preserve whole words when possible

# Example:
"tokenization" → "token" + "ization"
```

### SentencePiece (LLaMA-style)

```python
# SentencePiece Features:
# - No pre-tokenization
# - Handles whitespace as a token
# - Works with any language
# - Uses BPE or unigram

# Example:
"Hello world" → "▁Hello" + "▁world"
```

## 2.3 Practical Tokenization

```python
from transformers import AutoTokenizer

# ============ LOAD TOKENIZERS ============
def load_tokenizers():
    """Load different tokenizers for comparison"""
    tokenizers = {
        'gpt2': AutoTokenizer.from_pretrained('gpt2'),
        'bert': AutoTokenizer.from_pretrained('bert-base-uncased'),
        'llama': AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf'),
        't5': AutoTokenizer.from_pretrained('t5-small')
    }
    return tokenizers

# ============ TOKENIZE TEXT ============
def tokenize_text(tokenizer, text):
    """Tokenize text and return tokens"""
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)
    
    return {
        'tokens': tokens,
        'ids': ids,
        'num_tokens': len(tokens),
        'num_ids': len(ids)
    }

# Example usage
text = "The quick brown fox jumps over the lazy dog."
tokenizer = AutoTokenizer.from_pretrained('gpt2')
result = tokenize_text(tokenizer, text)
print(f"Tokens: {result['tokens']}")
print(f"Token IDs: {result['ids']}")
print(f"Number of tokens: {result['num_tokens']}")
```

## 2.4 Tokenization Comparison

```python
def compare_tokenizers(text):
    """Compare tokenization across different models"""
    tokenizers = load_tokenizers()
    
    results = {}
    for name, tokenizer in tokenizers.items():
        tokens = tokenizer.tokenize(text)
        results[name] = {
            'tokens': tokens[:10],  # Show first 10
            'total': len(tokens),
            'vocab_size': tokenizer.vocab_size
        }
    
    print("Tokenization Comparison")
    print("-" * 60)
    for name, data in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Vocab size: {data['vocab_size']:,}")
        print(f"  Token count: {data['total']}")
        print(f"  First tokens: {data['tokens']}")
    
    return results

# Example
compare_tokenizers("The cat sat on the mat.")
```

## 2.5 Tokenization Patterns

```python
# ============ SPECIAL TOKENS ============
special_tokens = {
    'gpt2': {
        'PAD': None,
        'UNK': '<|endoftext|>',
        'BOS': '<|endoftext|>',
        'EOS': '<|endoftext|>',
    },
    'bert': {
        'PAD': '[PAD]',
        'UNK': '[UNK]',
        'BOS': '[CLS]',
        'EOS': '[SEP]',
    },
    'llama': {
        'PAD': None,
        'UNK': '<unk>',
        'BOS': '<s>',
        'EOS': '</s>',
    }
}

# ============ PADDING AND TRUNCATION ============
def prepare_batch(tokenizer, texts, max_length=128):
    """Prepare a batch of texts for model input"""
    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask']
    }

# ============ TOKEN COUNTING ============
def count_tokens(texts, tokenizer_name='gpt2'):
    """Count tokens for a list of texts"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    total_tokens = 0
    for text in texts:
        tokens = tokenizer.encode(text)
        total_tokens += len(tokens)
    
    return total_tokens
```

---

# 3. EMBEDDINGS - FROM TOKENS TO VECTORS

## 3.1 What are Embeddings?

**Definition:** Embeddings are dense vector representations of tokens that capture semantic meaning.

**Analogy:**
- Think of embedding space as a map
- Similar words are closer together
- "king" and "queen" near each other
- "cat" and "dog" near each other

## 3.2 Embedding Dimensions

```python
# ============ EMBEDDING SIZES ============
model_configs = {
    'gpt2': {
        'vocab_size': 50257,
        'embed_dim': 768,
        'num_heads': 12,
        'num_layers': 12,
    },
    'gpt2-medium': {
        'vocab_size': 50257,
        'embed_dim': 1024,
        'num_heads': 16,
        'num_layers': 24,
    },
    'llama-7b': {
        'vocab_size': 32000,
        'embed_dim': 4096,
        'num_heads': 32,
        'num_layers': 32,
    },
    'llama-13b': {
        'vocab_size': 32000,
        'embed_dim': 5120,
        'num_heads': 40,
        'num_layers': 40,
    },
    'llama-70b': {
        'vocab_size': 32000,
        'embed_dim': 8192,
        'num_heads': 64,
        'num_layers': 80,
    }
}

# ============ EMBEDDING MEMORY CALCULATION ============
def embedding_memory(vocab_size, embed_dim, dtype='fp32'):
    """Calculate memory required for embedding layer"""
    bytes_per_param = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'int8': 1,
        'int4': 0.5
    }
    
    memory_bytes = vocab_size * embed_dim * bytes_per_param[dtype]
    memory_mb = memory_bytes / (1024 * 1024)
    memory_gb = memory_bytes / (1024**3)
    
    return {
        'bytes': memory_bytes,
        'MB': memory_mb,
        'GB': memory_gb
    }

# Example
for model, config in model_configs.items():
    mem = embedding_memory(config['vocab_size'], config['embed_dim'])
    print(f"{model}: {mem['GB']:.2f} GB")
```

## 3.3 Positional Embeddings

```python
# ============ SINUSOIDAL POSITIONAL ENCODING ============
import torch
import math

def sinusoidal_positional_encoding(seq_len, embed_dim):
    """Generate sinusoidal positional encodings"""
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2) * 
        -(math.log(10000.0) / embed_dim)
    )
    
    pos_encoding = torch.zeros(seq_len, embed_dim)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    
    return pos_encoding

# ============ LEARNED POSITIONAL EMBEDDINGS ============
class LearnedPositionalEmbedding(torch.nn.Module):
    """Learnable positional embeddings"""
    def __init__(self, max_seq_len, embed_dim):
        super().__init__()
        self.position_embeddings = torch.nn.Embedding(max_seq_len, embed_dim)
    
    def forward(self, seq_len):
        positions = torch.arange(seq_len)
        return self.position_embeddings(positions)

# ============ ROTARY POSITIONAL EMBEDDINGS (RoPE) ============
def rotary_positional_embedding(q, k, base=10000):
    """Simplified RoPE implementation"""
    seq_len, head_dim = q.shape[-2], q.shape[-1]
    
    # Create rotation matrix
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))
    positions = torch.arange(seq_len).float()
    
    # Rotate queries and keys
    # Implementation details...
    
    return q, k
```

## 3.4 Embedding Visualization

```python
# ============ VISUALIZE EMBEDDINGS ============
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_embeddings(embeddings, words, title="Word Embeddings"):
    """Visualize word embeddings using PCA"""
    # Reduce to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    
    # Label points
    for i, word in enumerate(words):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.savefig('embeddings_visualization.png', dpi=300)
    plt.show()

# Example using random embeddings
words = ['king', 'queen', 'man', 'woman', 'cat', 'dog']
embeddings = torch.randn(len(words), 50).numpy()
visualize_embeddings(embeddings, words)
```

---

# 4. SELF-ATTENTION MECHANICS

## 4.1 The Attention Equation

**The Core Equation:**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**Where:**
- **Q** = Query matrix
- **K** = Key matrix  
- **V** = Value matrix
- **d_k** = Dimension of keys

## 4.2 Understanding Attention

```python
# ============ SIMPLE ATTENTION IMPLEMENTATION ============
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention
    
    Args:
        Q: Query matrix [batch, seq_len, dim]
        K: Key matrix [batch, seq_len, dim]
        V: Value matrix [batch, seq_len, dim]
        mask: Optional mask for padding/causal
    """
    # Calculate attention scores
    d_k = K.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

# Example
batch_size, seq_len, dim = 2, 10, 64
Q = torch.randn(batch_size, seq_len, dim)
K = torch.randn(batch_size, seq_len, dim)
V = torch.randn(batch_size, seq_len, dim)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")
print(f"Weights shape: {weights.shape}")
```

## 4.3 Multi-Head Attention

```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections
        self.W_q = torch.nn.Linear(embed_dim, embed_dim)
        self.W_k = torch.nn.Linear(embed_dim, embed_dim)
        self.W_v = torch.nn.Linear(embed_dim, embed_dim)
        self.W_o = torch.nn.Linear(embed_dim, embed_dim)
        
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections and reshape
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.W_o(output)
        
        return output

# ============ ATTENTION COMPUTATION COMPLEXITY ============
def attention_complexity(batch_size, seq_len, embed_dim, num_heads):
    """Calculate FLOPs for attention computation"""
    head_dim = embed_dim // num_heads
    
    # QK^T multiplication: batch * num_heads * seq_len^2 * head_dim
    flops_qk = batch_size * num_heads * seq_len * seq_len * head_dim
    
    # Softmax: batch * num_heads * seq_len^2
    flops_softmax = batch_size * num_heads * seq_len * seq_len
    
    # attention * V: batch * num_heads * seq_len^2 * head_dim
    flops_av = batch_size * num_heads * seq_len * seq_len * head_dim
    
    total_flops = flops_qk + flops_softmax + flops_av
    
    return {
        'QK^T': flops_qk,
        'softmax': flops_softmax,
        'attention*V': flops_av,
        'total': total_flops
    }

# Example
flops = attention_complexity(
    batch_size=1,
    seq_len=2048,
    embed_dim=4096,
    num_heads=32
)

print("Attention FLOPs:")
for name, value in flops.items():
    print(f"  {name}: {value:,}")
```

## 4.4 Attention Visualization

```python
def visualize_attention(attention_weights, tokens):
    """Visualize attention weights as a heatmap"""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    plt.imshow(attention_weights, cmap='viridis', aspect='auto')
    
    # Add labels
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.yticks(range(len(tokens)), tokens)
    
    plt.colorbar(label='Attention Weight')
    plt.title('Attention Weights Visualization')
    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=300)
    plt.show()

# Example
tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
attention_weights = torch.randn(len(tokens), len(tokens))
attention_weights = F.softmax(attention_weights, dim=-1)
visualize_attention(attention_weights.numpy(), tokens)
```

---

# 5. KV-CACHE - MEMORY OPTIMIZATION

## 5.1 What is KV-Cache?

**Definition:** KV-Cache stores keys and values from previous tokens to avoid recomputing them during autoregressive generation.

**Why KV-Cache Matters:**
- Without cache: O(n²) per token
- With cache: O(n) per token
- Critical for inference efficiency

## 5.2 KV-Cache Memory Calculation

```python
def kv_cache_memory(batch_size, seq_len, num_layers, num_heads, head_dim, dtype='fp16'):
    """
    Calculate KV-Cache memory requirements
    
    Args:
        batch_size: Number of sequences in batch
        seq_len: Sequence length
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension per head
        dtype: Precision format
    
    Returns:
        dict: Memory requirements in different units
    """
    bytes_per_param = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'int8': 1
    }
    
    # Cache stores K and V for each layer
    # Shape per layer: [batch_size, num_heads, seq_len, head_dim]
    per_layer_memory = 2 * batch_size * num_heads * seq_len * head_dim * bytes_per_param[dtype]
    
    total_memory = per_layer_memory * num_layers
    
    return {
        'per_layer_bytes': per_layer_memory,
        'total_bytes': total_memory,
        'MB': total_memory / (1024 * 1024),
        'GB': total_memory / (1024**3),
        'formula': f"2 × {batch_size} × {num_heads} × {seq_len} × {head_dim} × {num_layers} × {bytes_per_param[dtype]} bytes"
    }

# ============ KV-CACHE FOR DIFFERENT MODELS ============
def kv_cache_for_models(seq_len=2048, batch_size=1):
    """Calculate KV-cache for different models"""
    models = [
        {'name': 'GPT-2', 'layers': 12, 'heads': 12, 'head_dim': 64},
        {'name': 'LLaMA-7B', 'layers': 32, 'heads': 32, 'head_dim': 128},
        {'name': 'LLaMA-13B', 'layers': 40, 'heads': 40, 'head_dim': 128},
        {'name': 'LLaMA-70B', 'layers': 80, 'heads': 64, 'head_dim': 128},
        {'name': 'GPT-3 175B', 'layers': 96, 'heads': 96, 'head_dim': 128},
    ]
    
    print("KV-Cache Memory Requirements")
    print("=" * 70)
    print(f"Sequence Length: {seq_len}, Batch Size: {batch_size}")
    print("-" * 70)
    
    for model in models:
        mem = kv_cache_memory(
            batch_size=batch_size,
            seq_len=seq_len,
            num_layers=model['layers'],
            num_heads=model['heads'],
            head_dim=model['head_dim']
        )
        print(f"{model['name']:12s}: {mem['GB']:.2f} GB (FP16)")
    
    print("=" * 70)

kv_cache_for_models()
```

## 5.3 KV-Cache Growth Visualization

```python
def plot_kv_cache_growth():
    """Visualize how KV-cache grows with sequence length"""
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
    models = {
        'LLaMA-7B': (32, 32, 128),
        'LLaMA-13B': (40, 40, 128),
        'LLaMA-70B': (80, 64, 128),
    }
    
    plt.figure(figsize=(12, 6))
    
    for model_name, (layers, heads, head_dim) in models.items():
        memory_gb = []
        for seq_len in seq_lengths:
            mem = kv_cache_memory(1, seq_len, layers, heads, head_dim)
            memory_gb.append(mem['GB'])
        
        plt.plot(seq_lengths, memory_gb, marker='o', label=model_name)
    
    plt.xlabel('Sequence Length')
    plt.ylabel('KV-Cache Memory (GB)')
    plt.title('KV-Cache Growth with Sequence Length (Batch=1, FP16)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('kv_cache_growth.png', dpi=300)
    plt.show()

plot_kv_cache_growth()
```

## 5.4 Efficient KV-Cache Management

```python
class KVCacheManager:
    """Manage KV-cache efficiently"""
    
    def __init__(self, max_batch_size, max_seq_len, num_layers, num_heads, head_dim):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Pre-allocate cache
        self.cache = None
        self.initialize_cache()
    
    def initialize_cache(self):
        """Initialize empty cache"""
        self.cache = {
            'k': [None] * self.num_layers,
            'v': [None] * self.num_layers
        }
    
    def update_cache(self, layer_idx, new_k, new_v):
        """Update cache with new tokens"""
        if self.cache['k'][layer_idx] is None:
            self.cache['k'][layer_idx] = new_k
            self.cache['v'][layer_idx] = new_v
        else:
            # Concatenate along sequence dimension
            self.cache['k'][layer_idx] = torch.cat(
                [self.cache['k'][layer_idx], new_k], dim=-2
            )
            self.cache['v'][layer_idx] = torch.cat(
                [self.cache['v'][layer_idx], new_v], dim=-2
            )
    
    def get_cache_size(self, layer_idx=None):
        """Get current cache size"""
        if layer_idx is not None:
            return {
                'k': self.cache['k'][layer_idx].shape if self.cache['k'][layer_idx] is not None else None,
                'v': self.cache['v'][layer_idx].shape if self.cache['v'][layer_idx] is not None else None
            }
        else:
            total_k = 0
            total_v = 0
            for i in range(self.num_layers):
                if self.cache['k'][i] is not None:
                    total_k += self.cache['k'][i].numel()
                    total_v += self.cache['v'][i].numel()
            return {'k_elements': total_k, 'v_elements': total_v}
    
    def clear(self):
        """Clear cache"""
        self.cache = {'k': [None] * self.num_layers, 'v': [None] * self.num_layers}
```

---

# 6. VRAM REQUIREMENTS - MEMORY MATH

## 6.1 Model Memory Components

**Total VRAM = Model Weights + KV-Cache + Activations + Gradients**

```
┌─────────────────────────────────────────────────────────┐
│                      Total VRAM                        │
├─────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────┐ │
│  │           Model Weights (4N bytes for FP32)      │ │
│  └───────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────┐ │
│  │              KV-Cache                            │ │
│  └───────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────┐ │
│  │              Activations                         │ │
│  └───────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────┐ │
│  │              Gradients (during training)         │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 6.2 Model Memory Calculator

```python
def model_memory_calculator(model_params, dtype='fp16', seq_len=2048, batch_size=1, num_layers=None):
    """
    Calculate total VRAM requirements
    
    Args:
        model_params: Number of parameters (in billions)
        dtype: Precision format
        seq_len: Sequence length
        batch_size: Batch size
        num_layers: Number of transformer layers (required for KV-cache)
    """
    
    param_bytes = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'int8': 1,
        'int4': 0.5
    }
    
    # Calculate model weights
    weights_gb = (model_params * 1e9 * param_bytes[dtype]) / (1024**3)
    
    # Calculate KV-cache (if num_layers is provided)
    if num_layers is not None:
        # Estimate head dimension from model architecture
        # Simplified assumption: embed_dim = 4096 for 7B, 8192 for 70B
        if model_params < 10:
            embed_dim = 4096
            num_heads = 32
        elif model_params < 30:
            embed_dim = 5120
            num_heads = 40
        else:
            embed_dim = 8192
            num_heads = 64
        
        head_dim = embed_dim // num_heads
        kv_cache = kv_cache_memory(batch_size, seq_len, num_layers, num_heads, head_dim, dtype)
        kv_gb = kv_cache['GB']
    else:
        kv_gb = 0
    
    # Estimate activations (roughly 20-30% of model weights)
    activations_gb = weights_gb * 0.25
    
    # Estimate gradients (only during training, roughly same as weights)
    gradients_gb = weights_gb if 'train' else 0
    
    total_gb = weights_gb + kv_gb + activations_gb + gradients_gb
    
    return {
        'model_weights_gb': weights_gb,
        'kv_cache_gb': kv_gb,
        'activations_gb': activations_gb,
        'gradients_gb': gradients_gb,
        'total_gb': total_gb
    }

# ============ COMPARE PRECISION FORMATS ============
def compare_precision_formats(model_params, seq_len=2048):
    """Compare memory usage across precision formats"""
    formats = ['fp32', 'fp16', 'bf16', 'int8', 'int4']
    
    print(f"Model: {model_params}B parameters, Seq Len: {seq_len}")
    print("=" * 70)
    print(f"{'Format':<8} {'Weights':<10} {'KV-Cache':<10} {'Total VRAM':<10} {'Memory Savings':<15}")
    print("-" * 70)
    
    base_mem = None
    for fmt in formats:
        mem = model_memory_calculator(model_params, fmt, seq_len, batch_size=1)
        total = mem['total_gb']
        
        if base_mem is None:
            base_mem = total
            savings = "0%"
        else:
            savings = f"{((base_mem - total) / base_mem * 100):.1f}%"
        
        print(f"{fmt:<8} {mem['model_weights_gb']:<10.2f} {mem['kv_cache_gb']:<10.2f} {total:<10.2f} {savings:<15}")
    
    print("=" * 70)

compare_precision_formats(7)
compare_precision_formats(70)
```

## 6.3 Practical VRAM Estimation

```python
def estimate_vram_for_deployment(model_name, seq_len, batch_size, dtype='fp16'):
    """Estimate VRAM for specific deployment scenario"""
    
    # Model specifications
    model_specs = {
        'gpt2': {'params': 0.124, 'layers': 12},
        'gpt2-medium': {'params': 0.35, 'layers': 24},
        'gpt2-large': {'params': 0.77, 'layers': 36},
        'gpt2-xl': {'params': 1.5, 'layers': 48},
        'llama-7b': {'params': 7, 'layers': 32},
        'llama-13b': {'params': 13, 'layers': 40},
        'llama-70b': {'params': 70, 'layers': 80},
        'mistral-7b': {'params': 7, 'layers': 32},
    }
    
    if model_name not in model_specs:
        raise ValueError(f"Model {model_name} not found")
    
    spec = model_specs[model_name]
    mem = model_memory_calculator(
        spec['params'],
        dtype,
        seq_len,
        batch_size,
        spec['layers']
    )
    
    print(f"VRAM Estimation for {model_name}")
    print("=" * 60)
    print(f"Parameters: {spec['params']}B")
    print(f"Sequence Length: {seq_len}")
    print(f"Batch Size: {batch_size}")
    print(f"Precision: {dtype}")
    print("-" * 60)
    print(f"Model Weights: {mem['model_weights_gb']:.2f} GB")
    print(f"KV-Cache: {mem['kv_cache_gb']:.2f} GB")
    print(f"Activations: {mem['activations_gb']:.2f} GB")
    print(f"Total VRAM: {mem['total_gb']:.2f} GB")
    print("=" * 60)
    
    return mem

# Example usage
estimate_vram_for_deployment('llama-7b', 2048, 1, 'fp16')
estimate_vram_for_deployment('llama-70b', 4096, 1, 'int8')
```

## 6.4 VRAM Requirements Table

| Model | FP32 | FP16 | INT8 | INT4 | Context 2048 | Context 8192 |
|-------|------|------|------|------|--------------|--------------|
| LLaMA-7B | 28GB | 14GB | 7GB | 3.5GB | +2GB | +8GB |
| LLaMA-13B | 52GB | 26GB | 13GB | 6.5GB | +4GB | +16GB |
| LLaMA-70B | 280GB | 140GB | 70GB | 35GB | +20GB | +80GB |
| GPT-3 175B | 700GB | 350GB | 175GB | 87.5GB | +50GB | +200GB |

---

# 7. PRECISION FORMATS EXPLAINED

## 7.1 Understanding Precision Formats

```
┌─────────────────────────────────────────────────────────────┐
│                     FP32 (32-bit)                         │
│  ┌─────┬───────────────────────┬────────────────────────┐ │
│  │Sign │    Exponent (8 bits)  │  Mantissa (23 bits)    │ │
│  └─────┴───────────────────────┴────────────────────────┘ │
│  Range: ~1e-38 to ~3e38                                    │
│  Memory: 4 bytes per parameter                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     FP16 (16-bit)                         │
│  ┌─────┬───────────────────────┬────────────────────────┐ │
│  │Sign │    Exponent (5 bits)  │  Mantissa (10 bits)    │ │
│  └─────┴───────────────────────┴────────────────────────┘ │
│  Range: ~5.96e-8 to 65504                                 │
│  Memory: 2 bytes per parameter                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     BF16 (16-bit)                         │
│  ┌─────┬───────────────────────┬────────────────────────┐ │
│  │Sign │    Exponent (8 bits)  │  Mantissa (7 bits)     │ │
│  └─────┴───────────────────────┴────────────────────────┘ │
│  Range: ~1e-38 to ~3e38 (same as FP32)                   │
│  Memory: 2 bytes per parameter                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     INT8 (8-bit)                          │
│  ┌───────────────────────────────────────────────────────┐ │
│  │              Integer (0 to 255 or -128 to 127)       │ │
│  └───────────────────────────────────────────────────────┘ │
│  Range: -128 to 127 (signed) or 0 to 255 (unsigned)      │
│  Memory: 1 byte per parameter                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     INT4 (4-bit)                          │
│  ┌───────────────────────────────────────────────────────┐ │
│  │            Integer (0 to 15 or -8 to 7)             │ │
│  └───────────────────────────────────────────────────────┘ │
│  Range: -8 to 7 (signed) or 0 to 15 (unsigned)           │
│  Memory: 0.5 bytes per parameter                          │
└─────────────────────────────────────────────────────────────┘
```

## 7.2 Precision Comparison

```python
import numpy as np

def compare_precision_formats():
    """Compare precision formats with examples"""
    
    print("Precision Format Comparison")
    print("=" * 80)
    print(f"{'Format':<8} {'Bits':<6} {'Bytes/Param':<12} {'Range':<25} {'Use Case':<25}")
    print("-" * 80)
    
    formats = [
        ('FP32', 32, 4, '~1e-38 to ~3e38', 'Training, high precision'),
        ('FP16', 16, 2, '~5.96e-8 to 65504', 'Inference, fine-tuning'),
        ('BF16', 16, 2, '~1e-38 to ~3e38', 'Training, stable'),
        ('INT8', 8, 1, '-128 to 127', 'Efficient inference'),
        ('INT4', 4, 0.5, '-8 to 7', 'Extreme quantization'),
    ]
    
    for fmt, bits, bytes_per, range_val, use_case in formats:
        print(f"{fmt:<8} {bits:<6} {bytes_per:<12} {range_val:<25} {use_case:<25}")
    
    print("=" * 80)
    
    # Memory comparison
    print("\nMemory Comparison (for 7B parameters):")
    print("-" * 50)
    for fmt, _, bytes_per, _, _ in formats:
        memory_gb = (7 * 1e9 * bytes_per) / (1024**3)
        print(f"{fmt:8s}: {memory_gb:.2f} GB")

compare_precision_formats()
```

## 7.3 Quantization Methods

```python
# ============ QUANTIZATION TECHNIQUES ============

def explain_quantization():
    """Explain different quantization approaches"""
    
    print("Quantization Techniques")
    print("=" * 70)
    print("\n1. Post-Training Quantization (PTQ)")
    print("   - Quantize weights after training")
    print("   - No retraining needed")
    print("   - Lower accuracy loss for larger models")
    
    print("\n2. Quantization-Aware Training (QAT)")
    print("   - Train with quantization in mind")
    print("   - Simulate quantization during training")
    print("   - Better accuracy than PTQ")
    
    print("\n3. GPTQ (Generative Pre-trained Transformer Quantization)")
    print("   - State-of-the-art for LLMs")
    print("   - Uses Hessian information")
    print("   - 4-bit quantization with minimal accuracy loss")
    
    print("\n4. AWQ (Activation-aware Weight Quantization)")
    print("   - Protects important weights")
    print("   - Considers activations")
    print("   - Good for Llama models")
    
    print("\n5. GGUF (GPT-Generated Unified Format)")
    print("   - Format for CPU inference")
    print("   - Uses 4-bit quantization")
    print("   - Optimized for Llama.cpp")
    
    print("=" * 70)

explain_quantization()
```

## 7.4 Choosing the Right Precision

```python
def recommend_precision(model_size_gb, speed_required, accuracy_required):
    """Recommend precision based on constraints"""
    
    print("Precision Recommendation")
    print("=" * 60)
    print(f"Model Size: {model_size_gb} GB")
    print(f"Speed Required: {speed_required}")
    print(f"Accuracy Required: {accuracy_required}")
    print("-" * 60)
    
    recommendations = []
    
    if model_size_gb < 2:
        recommendations.append(('FP32', 'Best accuracy'))
    
    if model_size_gb < 16:
        recommendations.append(('FP16', 'Good balance'))
        recommendations.append(('BF16', 'Stable training'))
    
    if speed_required:
        recommendations.append(('INT8', 'Fast inference'))
    
    if speed_required and accuracy_required < 0.95:
        recommendations.append(('INT4', 'Extreme speed'))
    
    print("\nRecommended Formats:")
    for fmt, reason in recommendations:
        print(f"  - {fmt}: {reason}")
    
    print("=" * 60)
    return recommendations

recommend_precision(7, speed_required=True, accuracy_required=0.9)
```

---

# 8. CONTEXT WINDOWS AND SCALING

## 8.1 Context Window Basics

**Definition:** The context window is the maximum number of tokens the model can process at once.

**Why It Matters:**
- Longer context = More information
- Longer context = More memory required
- Longer context = More compute required

```python
def context_window_costs(context_lengths, model_params=7, dtype='fp16'):
    """Calculate costs for different context windows"""
    
    print(f"Context Window Costs (Model: {model_params}B, Precision: {dtype})")
    print("=" * 80)
    print(f"{'Context Length':<15} {'Memory':<12} {'Cost/Token':<12} {'Cost/Request':<12} {'KV-Cache':<12}")
    print("-" * 80)
    
    for seq_len in context_lengths:
        mem = model_memory_calculator(model_params, dtype, seq_len, batch_size=1)
        
        # Simplified cost model
        cost_per_1k_tokens = 0.001  # $0.001 per 1k tokens
        cost_request = (seq_len / 1000) * cost_per_1k_tokens
        
        kv_mem = mem['kv_cache_gb']
        
        print(f"{seq_len:<15} {mem['total_gb']:<12.2f} {cost_per_1k_tokens*1000:<12.4f} {cost_request:<12.4f} {kv_mem:<12.2f}")

context_window_costs([512, 1024, 2048, 4096, 8192, 16384])
```

## 8.2 Scaling Laws

```python
def scaling_laws():
    """Explain scaling laws for LLMs"""
    
    print("Model Scaling Laws")
    print("=" * 60)
    print("1. Parameter Scaling:")
    print("   - Performance ∝ N^0.35")
    print("   - 10x parameters = ~2x performance")
    
    print("\n2. Data Scaling:")
    print("   - Performance ∝ D^0.35")
    print("   - 10x data = ~2x performance")
    
    print("\n3. Compute Scaling:")
    print("   - Performance ∝ C^0.28")
    print("   - 10x compute = ~1.8x performance")
    
    print("\n4. Chinchilla Scaling Law:")
    print("   - Optimal: D = 20 × N")
    print("   - For N parameters, need 20N training tokens")
    
    print("\n5. Implications:")
    print("   - Bigger models need more data")
    print("   - Data is as important as model size")
    print("   - Compute is the real bottleneck")

scaling_laws()
```

## 8.3 Context Window Extensions

```python
def context_window_extensions():
    """Explain techniques for extending context windows"""
    
    print("Context Window Extension Techniques")
    print("=" * 60)
    
    techniques = [
        {
            'name': 'Rotary Positional Embeddings (RoPE)',
            'description': 'Uses rotation to encode positions',
            'benefit': 'Can extend to 32k+ tokens',
            'limitation': 'Limited by training length'
        },
        {
            'name': 'Alibi',
            'description': 'Adds linear bias to attention',
            'benefit': 'No position embeddings needed',
            'limitation': 'Performance degradation for very long sequences'
        },
        {
            'name': 'Position Interpolation',
            'description': 'Linearly scales position indices',
            'benefit': 'Doubles context window with minimal training',
            'limitation': 'Some performance loss'
        },
        {
            'name': 'YaRN (Yet another RoPE extension)',
            'description': 'Interpolation with NTK-aware scaling',
            'benefit': 'Better performance than simple interpolation',
            'limitation': 'More complex implementation'
        },
        {
            'name': 'Sliding Window Attention',
            'description': 'Only attends to local tokens',
            'benefit': 'Linear complexity',
            'limitation': 'Global dependency issues'
        }
    ]
    
    for tech in techniques:
        print(f"\n{tech['name']}:")
        print(f"  Description: {tech['description']}")
        print(f"  Benefit: {tech['benefit']}")
        print(f"  Limitation: {tech['limitation']}")

context_window_extensions()
```

---

# 9. COMPLETE WORKING CODE

## 9.1 Token Profiler

```python
# ============================================
# Multi-Model Token Profiler & Memory Math Engine
# Week 1 - PyTorch & LLM Mechanics
# Tech Prime Pvt Limited - Advanced AI/ML Internship
# ============================================

import torch
import math
from transformers import AutoTokenizer
import json
import csv
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. CONFIGURATION
# ============================================

class LLMConfig:
    """Configuration for LLM profiling"""
    
    # Supported models - using only publicly available tokenizers
    MODELS = {
        'gpt2': {
            'name': 'GPT-2',
            'tokenizer': 'gpt2',
            'params_b': 0.124,
            'layers': 12,
            'heads': 12,
            'head_dim': 64,
            'vocab_size': 50257,
            'max_ctx': 1024
        },
        'gpt2-medium': {
            'name': 'GPT-2 Medium',
            'tokenizer': 'gpt2-medium',
            'params_b': 0.35,
            'layers': 24,
            'heads': 16,
            'head_dim': 64,
            'vocab_size': 50257,
            'max_ctx': 1024
        },
        'gpt2-large': {
            'name': 'GPT-2 Large',
            'tokenizer': 'gpt2-large',
            'params_b': 0.77,
            'layers': 36,
            'heads': 20,
            'head_dim': 64,
            'vocab_size': 50257,
            'max_ctx': 1024
        },
        'bert-base': {
            'name': 'BERT Base',
            'tokenizer': 'bert-base-uncased',
            'params_b': 0.11,
            'layers': 12,
            'heads': 12,
            'head_dim': 64,
            'vocab_size': 30522,
            'max_ctx': 512
        },
        'roberta-base': {
            'name': 'RoBERTa Base',
            'tokenizer': 'roberta-base',
            'params_b': 0.125,
            'layers': 12,
            'heads': 12,
            'head_dim': 64,
            'vocab_size': 50265,
            'max_ctx': 512
        },
        't5-small': {
            'name': 'T5 Small',
            'tokenizer': 't5-small',
            'params_b': 0.06,
            'layers': 6,
            'heads': 8,
            'head_dim': 64,
            'vocab_size': 32128,
            'max_ctx': 512
        },
        'mistral-7b': {
            'name': 'Mistral-7B',
            'tokenizer': 'mistralai/Mistral-7B-v0.1',
            'params_b': 7,
            'layers': 32,
            'heads': 32,
            'head_dim': 128,
            'vocab_size': 32000,
            'max_ctx': 8192
        }
    }
    
    # LLaMA models (for VRAM calculation only, not tokenization)
    LLAMA_MODELS = {
        'llama-7b': {
            'name': 'LLaMA-7B',
            'params_b': 7,
            'layers': 32,
            'heads': 32,
            'head_dim': 128,
            'vocab_size': 32000,
            'max_ctx': 4096
        },
        'llama-13b': {
            'name': 'LLaMA-13B',
            'params_b': 13,
            'layers': 40,
            'heads': 40,
            'head_dim': 128,
            'vocab_size': 32000,
            'max_ctx': 4096
        },
        'llama-70b': {
            'name': 'LLaMA-70B',
            'params_b': 70,
            'layers': 80,
            'heads': 64,
            'head_dim': 128,
            'vocab_size': 32000,
            'max_ctx': 4096
        }
    }
    
    # Precision formats
    PRECISIONS = {
        'fp32': {'bits': 32, 'bytes': 4, 'name': 'FP32'},
        'fp16': {'bits': 16, 'bytes': 2, 'name': 'FP16'},
        'bf16': {'bits': 16, 'bytes': 2, 'name': 'BF16'},
        'int8': {'bits': 8, 'bytes': 1, 'name': 'INT8'},
        'int4': {'bits': 4, 'bytes': 0.5, 'name': 'INT4'}
    }
    
    # Default settings
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_SEQ_LEN = 2048
    DEFAULT_PRECISION = 'fp16'

# ============================================
# 2. TOKEN PROFILER
# ============================================

class TokenProfiler:
    """Profile tokenization across different models"""
    
    def __init__(self):
        self.tokenizers = {}
        self.load_tokenizers()
    
    def load_tokenizers(self):
        """Load tokenizers for all models"""
        print("Loading tokenizers...")
        for model_key, config in LLMConfig.MODELS.items():
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    config['tokenizer'],
                    trust_remote_code=True
                )
                self.tokenizers[model_key] = tokenizer
                print(f"  ✓ Loaded: {config['name']}")
            except Exception as e:
                print(f"  ✗ Failed to load {config['name']}: {e}")
    
    def count_tokens(self, text: str, model_key: str = 'gpt2') -> Dict:
        """Count tokens for a given text and model"""
        if model_key not in self.tokenizers:
            # Try fallback to a model that is loaded
            available_models = list(self.tokenizers.keys())
            if available_models:
                fallback = available_models[0]
                print(f"  ⚠ Model {model_key} not loaded, using {fallback} as fallback")
                model_key = fallback
            else:
                raise ValueError("No tokenizers loaded. Please check your internet connection.")
        
        tokenizer = self.tokenizers[model_key]
        
        try:
            tokens = tokenizer.encode(text)
            token_strings = tokenizer.convert_ids_to_tokens(tokens)
            
            return {
                'text': text[:100] + '...' if len(text) > 100 else text,
                'model': LLMConfig.MODELS.get(model_key, {}).get('name', model_key),
                'token_count': len(tokens),
                'token_ids': tokens[:10],  # First 10 tokens
                'token_strings': token_strings[:10],
                'vocab_size': tokenizer.vocab_size
            }
        except Exception as e:
            return {
                'error': str(e),
                'model': model_key
            }
    
    def profile_document(self, document: str, models: Optional[List[str]] = None) -> Dict:
        """Profile a document across multiple models"""
        if models is None:
            models = list(self.tokenizers.keys())
        
        results = {}
        for model_key in models:
            try:
                results[model_key] = self.count_tokens(document, model_key)
            except Exception as e:
                results[model_key] = {'error': str(e)}
        
        return results
    
    def estimate_cost(self, token_count: int, cost_per_1k: float = 0.001) -> float:
        """Estimate API cost based on token count"""
        return (token_count / 1000) * cost_per_1k
    
    def generate_report(self, text: str) -> str:
        """Generate a report of tokenization across models"""
        results = self.profile_document(text)
        
        report = [
            "=" * 70,
            "TOKENIZATION REPORT",
            "=" * 70,
            f"\nText Sample: {text[:200]}...\n",
            f"{'Model':<20} {'Tokens':<10} {'Vocab Size':<15} {'Est. Cost (1000x)':<15}"
        ]
        report.append("-" * 70)
        
        for model_key, result in results.items():
            if 'error' in result:
                report.append(f"{model_key:<20} {'ERROR':<10}")
            else:
                cost = self.estimate_cost(result['token_count'])
                report.append(
                    f"{result['model']:<20} "
                    f"{result['token_count']:<10} "
                    f"{result['vocab_size']:<15} "
                    f"${cost*1000:<14.4f}"
                )
        
        report.append("=" * 70)
        return "\n".join(report)

# ============================================
# 3. MEMORY MATH ENGINE
# ============================================

class MemoryMathEngine:
    """Calculate VRAM requirements for LLM inference"""
    
    def __init__(self):
        self.model_configs = {**LLMConfig.MODELS, **LLMConfig.LLAMA_MODELS}
        self.precisions = LLMConfig.PRECISIONS
    
    def calculate_model_weights(self, params_b: float, precision: str) -> Dict:
        """Calculate memory for model weights"""
        if precision not in self.precisions:
            raise ValueError(f"Precision {precision} not supported")
        
        bytes_per_param = self.precisions[precision]['bytes']
        total_bytes = params_b * 1e9 * bytes_per_param
        
        return {
            'bytes': total_bytes,
            'MB': total_bytes / (1024 * 1024),
            'GB': total_bytes / (1024**3)
        }
    
    def calculate_kv_cache(self, batch_size: int, seq_len: int, 
                           layers: int, heads: int, head_dim: int,
                           precision: str) -> Dict:
        """Calculate KV-cache memory"""
        if precision not in self.precisions:
            raise ValueError(f"Precision {precision} not supported")
        
        bytes_per_param = self.precisions[precision]['bytes']
        
        # Each layer stores K and V
        per_layer_bytes = 2 * batch_size * heads * seq_len * head_dim * bytes_per_param
        total_bytes = per_layer_bytes * layers
        
        return {
            'bytes': total_bytes,
            'MB': total_bytes / (1024 * 1024),
            'GB': total_bytes / (1024**3),
            'formula': f"2 × {batch_size} × {heads} × {seq_len} × {head_dim} × {layers} × {bytes_per_param}B"
        }
    
    def calculate_activations(self, params_b: float, seq_len: int, precision: str) -> Dict:
        """Estimate activation memory"""
        # Rough estimate: 20-30% of model weights
        factor = 0.25
        weights = self.calculate_model_weights(params_b, precision)
        
        total_bytes = weights['bytes'] * factor
        
        return {
            'bytes': total_bytes,
            'MB': total_bytes / (1024 * 1024),
            'GB': total_bytes / (1024**3),
            'factor': factor
        }
    
    def calculate_total_vram(self, model_key: str, seq_len: int, 
                           batch_size: int, precision: str = 'fp16') -> Dict:
        """Calculate total VRAM requirements"""
        if model_key not in self.model_configs:
            raise ValueError(f"Model {model_key} not found")
        
        config = self.model_configs[model_key]
        
        # Model weights
        weights = self.calculate_model_weights(config['params_b'], precision)
        
        # KV-cache
        kv_cache = self.calculate_kv_cache(
            batch_size, seq_len,
            config['layers'],
            config['heads'],
            config['head_dim'],
            precision
        )
        
        # Activations
        activations = self.calculate_activations(config['params_b'], seq_len, precision)
        
        # Gradients (during training)
        gradients = {'bytes': weights['bytes'], 'GB': weights['GB']}
        
        # Total
        total_bytes = weights['bytes'] + kv_cache['bytes'] + activations['bytes']
        total_gb = total_bytes / (1024**3)
        
        return {
            'model_name': config['name'],
            'params_b': config['params_b'],
            'seq_len': seq_len,
            'batch_size': batch_size,
            'precision': precision,
            'model_weights': weights,
            'kv_cache': kv_cache,
            'activations': activations,
            'gradients': gradients,
            'total_gb': total_gb,
            'total_bytes': total_bytes
        }
    
    def compare_deployments(self, seq_len: int, batch_size: int = 1):
        """Compare VRAM across models and precisions"""
        
        models = ['llama-7b', 'llama-13b', 'llama-70b', 'mistral-7b']
        precisions = ['fp16', 'int8', 'int4']
        
        print("VRAM Comparison for Deployment")
        print("=" * 80)
        print(f"Sequence Length: {seq_len}, Batch Size: {batch_size}")
        print("-" * 80)
        print(f"{'Model':<15} {'Precision':<10} {'Weights':<12} {'KV-Cache':<12} {'Total':<12}")
        print("-" * 80)
        
        results = []
        for model in models:
            if model not in self.model_configs:
                continue
            for prec in precisions:
                try:
                    mem = self.calculate_total_vram(model, seq_len, batch_size, prec)
                    results.append({
                        'model': mem['model_name'],
                        'precision': prec,
                        'weights_gb': mem['model_weights']['GB'],
                        'kv_cache_gb': mem['kv_cache']['GB'],
                        'total_gb': mem['total_gb']
                    })
                    print(f"{mem['model_name']:<15} {prec:<10} "
                          f"{mem['model_weights']['GB']:<12.2f} "
                          f"{mem['kv_cache']['GB']:<12.2f} "
                          f"{mem['total_gb']:<12.2f}")
                except Exception as e:
                    print(f"{model:<15} {prec:<10} {'ERROR':<12}")
        
        print("=" * 80)
        return results

# ============================================
# 4. COST ESTIMATOR
# ============================================

class CostEstimator:
    """Estimate API costs for LLM inference"""
    
    def __init__(self):
        # Standard API pricing (per 1M tokens)
        self.pricing = {
            'gpt-4': {
                'input': 30.00,
                'output': 60.00
            },
            'gpt-4-turbo': {
                'input': 10.00,
                'output': 30.00
            },
            'gpt-3.5-turbo': {
                'input': 0.50,
                'output': 1.50
            },
            'claude-3-opus': {
                'input': 15.00,
                'output': 75.00
            },
            'claude-3-sonnet': {
                'input': 3.00,
                'output': 15.00
            },
            'llama-2-7b': {
                'input': 0.20,
                'output': 0.20
            },
            'llama-2-13b': {
                'input': 0.50,
                'output': 0.50
            },
            'llama-2-70b': {
                'input': 2.00,
                'output': 2.00
            },
            'mistral-7b': {
                'input': 0.20,
                'output': 0.20
            },
            'mixtral-8x7b': {
                'input': 0.50,
                'output': 0.50
            }
        }
    
    def estimate_cost(self, token_count: int, model: str, 
                     input_tokens: int = None, output_tokens: int = None) -> Dict:
        """Estimate cost for a request"""
        
        if model not in self.pricing:
            raise ValueError(f"Model {model} not found in pricing")
        
        if input_tokens is None:
            input_tokens = token_count
            output_tokens = token_count * 0.5  # Assume output is half of input
        
        pricing = self.pricing[model]
        
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        total_cost = input_cost + output_cost
        
        return {
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost
        }
    
    def batch_cost_estimate(self, requests: List[Dict]) -> Dict:
        """Estimate cost for a batch of requests"""
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0
        
        results = []
        for req in requests:
            cost = self.estimate_cost(
                req.get('token_count', 0),
                req['model'],
                req.get('input_tokens'),
                req.get('output_tokens')
            )
            results.append(cost)
            total_input_tokens += cost['input_tokens']
            total_output_tokens += cost['output_tokens']
            total_cost += cost['total_cost']
        
        return {
            'total_requests': len(requests),
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens,
            'total_cost': total_cost,
            'per_request': results
        }
    
    def monthly_cost_projection(self, daily_tokens: int, model: str,
                               output_ratio: float = 0.5) -> Dict:
        """Project monthly costs based on daily usage"""
        monthly_tokens = daily_tokens * 30
        
        input_tokens = int(monthly_tokens / (1 + output_ratio))
        output_tokens = int(monthly_tokens - input_tokens)
        
        cost = self.estimate_cost(
            monthly_tokens,
            model,
            input_tokens,
            output_tokens
        )
        
        return {
            'daily_tokens': daily_tokens,
            'monthly_tokens': monthly_tokens,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'monthly_cost': cost['total_cost'],
            'daily_cost': cost['total_cost'] / 30,
            'model': model
        }

# ============================================
# 5. MAIN APPLICATION
# ============================================

class LLMProfiler:
    """Main application for LLM profiling"""
    
    def __init__(self):
        self.token_profiler = TokenProfiler()
        self.memory_engine = MemoryMathEngine()
        self.cost_estimator = CostEstimator()
    
    def profile_pipeline(self, text: str, model_key: str = 'gpt2',
                        seq_len: int = 2048, precision: str = 'fp16'):
        """Run complete profiling pipeline"""
        
        print("=" * 80)
        print("LLM PROFILING PIPELINE")
        print("=" * 80)
        
        # 1. Tokenization
        print("\n1. TOKENIZATION")
        print("-" * 40)
        token_result = self.token_profiler.count_tokens(text, model_key)
        if 'error' in token_result:
            print(f"Error: {token_result['error']}")
            return
        print(f"Model: {token_result['model']}")
        print(f"Token Count: {token_result['token_count']}")
        print(f"Vocab Size: {token_result['vocab_size']}")
        
        # 2. Memory Estimation
        print("\n2. MEMORY ESTIMATION")
        print("-" * 40)
        try:
            mem_result = self.memory_engine.calculate_total_vram(
                model_key, seq_len, 1, precision
            )
            print(f"Model: {mem_result['model_name']}")
            print(f"Parameters: {mem_result['params_b']}B")
            print(f"Precision: {mem_result['precision']}")
            print(f"Model Weights: {mem_result['model_weights']['GB']:.2f} GB")
            print(f"KV-Cache: {mem_result['kv_cache']['GB']:.2f} GB")
            print(f"Activations: {mem_result['activations']['GB']:.2f} GB")
            print(f"Total VRAM: {mem_result['total_gb']:.2f} GB")
        except Exception as e:
            print(f"Error in memory estimation: {e}")
            mem_result = None
        
        # 3. Cost Estimation
        print("\n3. COST ESTIMATION")
        print("-" * 40)
        try:
            cost = self.cost_estimator.estimate_cost(
                token_result['token_count'],
                'gpt-3.5-turbo'
            )
            print(f"Model: {cost['model']}")
            print(f"Input Tokens: {cost['input_tokens']:,}")
            print(f"Output Tokens: {cost['output_tokens']:,}")
            print(f"Total Cost: ${cost['total_cost']:.6f}")
        except Exception as e:
            print(f"Error in cost estimation: {e}")
            cost = None
        
        # 4. Monthly Projection
        print("\n4. MONTHLY PROJECTION")
        print("-" * 40)
        try:
            monthly = self.cost_estimator.monthly_cost_projection(
                token_result['token_count'] * 100,  # 100 requests per day
                'gpt-3.5-turbo'
            )
            print(f"Daily Requests: 100")
            print(f"Daily Tokens: {monthly['daily_tokens']:,}")
            print(f"Monthly Tokens: {monthly['monthly_tokens']:,}")
            print(f"Monthly Cost: ${monthly['monthly_cost']:.2f}")
        except Exception as e:
            print(f"Error in monthly projection: {e}")
            monthly = None
        
        print("\n" + "=" * 80)
        
        return {
            'tokenization': token_result,
            'memory': mem_result,
            'cost': cost,
            'monthly': monthly
        }
    
    def generate_full_report(self, text: str, models: List[str] = None,
                           seq_lens: List[int] = None):
        """Generate comprehensive report"""
        
        if models is None:
            models = ['llama-7b', 'llama-13b', 'llama-70b', 'mistral-7b']
        
        if seq_lens is None:
            seq_lens = [512, 1024, 2048, 4096]
        
        print("=" * 80)
        print("COMPREHENSIVE LLM REPORT")
        print("=" * 80)
        
        # Tokenization report
        print("\nTOKENIZATION ACROSS MODELS")
        print("-" * 40)
        token_report = self.token_profiler.generate_report(text)
        print(token_report)
        
        # VRAM comparison
        print("\nVRAM COMPARISON")
        print("-" * 40)
        for seq_len in seq_lens:
            print(f"\nSequence Length: {seq_len}")
            self.memory_engine.compare_deployments(seq_len)
        
        # Cost comparison
        print("\nCOST COMPARISON")
        print("-" * 40)
        models_cost = ['gpt-3.5-turbo', 'gpt-4', 'claude-3-sonnet', 'llama-2-70b']
        print(f"{'Model':<25} {'Cost per 1M tokens (avg)':<25} {'Cost per 1k tokens':<20}")
        print("-" * 70)
        for model in models_cost:
            pricing = self.cost_estimator.pricing.get(model, {})
            if pricing:
                avg_cost = (pricing['input'] + pricing['output']) / 2
                print(f"{model:<25} ${avg_cost:<24.2f} ${avg_cost/1000:<19.4f}")
        
        print("\n" + "=" * 80)

# ============================================
# 6. RUN
# ============================================

def main():
    """Main entry point"""
    
    print("=" * 80)
    print("MULTI-MODEL TOKEN PROFILER & MEMORY MATH ENGINE")
    print("Tech Prime Pvt Limited - Advanced AI/ML Internship")
    print("=" * 80)
    
    # Initialize profiler
    profiler = LLMProfiler()
    
    # Sample text
    sample_text = """
    The Transformer architecture has revolutionized natural language processing.
    At its core, the self-attention mechanism allows the model to weigh the 
    importance of different parts of the input sequence. This enables the model
    to capture long-range dependencies and contextual relationships that were
    previously difficult for traditional recurrent neural networks.
    
    The key innovation is the attention mechanism, which computes a weighted
    sum of values based on the compatibility of queries and keys. This allows
    the model to focus on relevant parts of the input when generating each
    output token.
    """
    
    # Run profiling using a working model
    profiler.profile_pipeline(sample_text, model_key='gpt2')
    
    # Generate full report
    profiler.generate_full_report(sample_text)
    
    print("\n" + "=" * 80)
    print("✓ PROFILING COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
```

---

# 10. COMMON ISSUES AND SOLUTIONS

## 10.1 CUDA Out of Memory

```python
# ============ SOLUTION 1: Reduce Batch Size ============
batch_size = 1  # Instead of 4 or 8

# ============ SOLUTION 2: Use Smaller Precision ============
precision = 'int8'  # Instead of 'fp16'

# ============ SOLUTION 3: Use Gradient Checkpointing ============
model.gradient_checkpointing_enable()

# ============ SOLUTION 4: Use Flash Attention ============
# pip install flash-attn
from flash_attn import flash_attn_func

# ============ SOLUTION 5: Clear Cache ============
torch.cuda.empty_cache()
```

## 10.2 Slow Tokenization

```python
# ============ SOLUTION 1: Batch Tokenization ============
def batch_tokenize(tokenizer, texts, batch_size=32):
    """Tokenize texts in batches"""
    all_tokens = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True)
        all_tokens.extend(tokens['input_ids'])
    return all_tokens

# ============ SOLUTION 2: Cache Tokenizer ============
class CachedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.cache = {}
    
    def encode(self, text):
        if text not in self.cache:
            self.cache[text] = self.tokenizer.encode(text)
        return self.cache[text]

# ============ SOLUTION 3: Use Faster Tokenizer ============
# Use 'fast' tokenizer when available
tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
```

## 10.3 Token Count Mismatch

```python
# ============ SOLUTION 1: Use Same Tokenizer ============
# Different models use different tokenizers
# GPT-2 tokenizer ≠ LLaMA tokenizer
def count_tokens_consistent(text, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return len(tokenizer.encode(text))

# ============ SOLUTION 2: Account for Special Tokens ============
def count_with_special_tokens(text, tokenizer):
    """Count tokens including special tokens"""
    tokens = tokenizer.tokenize(text)
    special_tokens = tokenizer.all_special_tokens
    special_count = sum(1 for t in tokens if t in special_tokens)
    return len(tokens), special_count

# ============ SOLUTION 3: Use Model's Actual Tokenizer ============
def get_accurate_token_count(text, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return len(tokenizer.encode(text, add_special_tokens=True))
```

## 10.4 VRAM Calculation Errors

```python
# ============ SOLUTION 1: Account for All Components ============
def complete_vram_calculation(model, seq_len, batch_size):
    """Calculate VRAM including all components"""
    # Model parameters
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Buffer memory (activations, gradients)
    buffer_memory = estimate_buffer_memory(model, seq_len, batch_size)
    
    # KV-cache
    kv_memory = estimate_kv_cache(model, seq_len, batch_size)
    
    # Total
    total = param_memory + buffer_memory + kv_memory
    
    return total

# ============ SOLUTION 2: Use Actual PyTorch Memory ============
def get_actual_memory_usage():
    """Get actual memory usage from PyTorch"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return allocated, reserved
    return 0, 0

# ============ SOLUTION 3: Account for Memory Fragmentation ============
def add_fragmentation_buffer(estimated_vram, factor=1.1):
    """Add buffer for memory fragmentation"""
    return estimated_vram * factor
```

## 10.5 API Cost Calculation Errors

```python
# ============ SOLUTION 1: Use Correct Token Count ============
def get_accurate_api_cost(text, model, api_pricing):
    """Calculate accurate API cost"""
    # Count tokens with the model's tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokens = tokenizer.encode(text)
    
    # Cost per 1M tokens
    cost_per_1m = api_pricing[model]
    
    return (len(tokens) / 1_000_000) * cost_per_1m

# ============ SOLUTION 2: Account for Input/Output Pricing ============
def cost_with_separate_pricing(input_text, output_text, model, pricing):
    input_tokens = len(AutoTokenizer.from_pretrained(model).encode(input_text))
    output_tokens = len(AutoTokenizer.from_pretrained(model).encode(output_text))
    
    input_cost = (input_tokens / 1_000_000) * pricing[model]['input']
    output_cost = (output_tokens / 1_000_000) * pricing[model]['output']
    
    return input_cost + output_cost

# ============ SOLUTION 3: Add Buffer for Overhead ============
def cost_with_buffer(text, model, pricing, buffer=0.1):
    """Add buffer for tokenization overhead"""
    base_cost = get_accurate_api_cost(text, model, pricing)
    return base_cost * (1 + buffer)
```

---

# 11. QUICK REFERENCE GUIDE

## 11.1 Key Formulas

```
# Token Count Estimation
Tokens ≈ Words × 1.3  (English)
Tokens ≈ Characters ÷ 4  (General)

# VRAM Calculation
VRAM = Model_Weights + KV_Cache + Activations + Gradients

# KV-Cache Formula
KV_Cache = 2 × batch_size × seq_len × embed_dim × layers × bytes_per_param

# Attention Complexity
O(n²)  for n tokens
O(n)   with KV-cache

# Cost Estimation
Cost = (Tokens / 1,000,000) × Price_per_1M
```

## 11.2 Common Commands

```python
# ============ LOAD TOKENIZER ============
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('model-name')

# ============ COUNT TOKENS ============
token_count = len(tokenizer.encode(text))

# ============ BATCH TOKENIZATION ============
tokens = tokenizer(texts, padding=True, truncation=True)

# ============ GET MODEL CONFIG ============
from transformers import AutoConfig
config = AutoConfig.from_pretrained('model-name')

# ============ CHECK GPU MEMORY ============
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()

# ============ CLEAR GPU CACHE ============
torch.cuda.empty_cache()

# ============ MIXED PRECISION ============
from torch.cuda.amp import autocast, GradScaler
with autocast():
    outputs = model(inputs)
```

## 11.3 Memory Reference Table

| Model | Parameters | FP16 VRAM | INT8 VRAM | KV-Cache (2048) | Context Limit |
|-------|------------|-----------|-----------|-----------------|---------------|
| GPT-2 | 124M | 0.5 GB | 0.25 GB | 0.1 GB | 1024 |
| GPT-2 Medium | 354M | 1.4 GB | 0.7 GB | 0.3 GB | 1024 |
| LLaMA-7B | 7B | 14 GB | 7 GB | 2 GB | 4096 |
| LLaMA-13B | 13B | 26 GB | 13 GB | 4 GB | 4096 |
| LLaMA-70B | 70B | 140 GB | 70 GB | 20 GB | 4096 |
| Mistral-7B | 7B | 14 GB | 7 GB | 4 GB | 8192 |

---

# 12. FINAL CHECKLIST

## Week 1 Completion Checklist

| Task | Status | Notes |
|------|--------|-------|
| Understand Tokenization | ☐ | BPE, WordPiece, SentencePiece |
| Understand Embeddings | ☐ | Positional, Learned, RoPE |
| Understand Self-Attention | ☐ | Q, K, V, Multi-Head |
| Understand KV-Cache | ☐ | Memory, Growth, Management |
| Calculate VRAM Requirements | ☐ | Model, Cache, Activations |
| Understand Precision Formats | ☐ | FP32, FP16, BF16, INT8, INT4 |
| Build Token Profiler | ☐ | Multiple tokenizers comparison |
| Build Memory Math Engine | ☐ | VRAM estimation formulas |
| Build Cost Estimator | ☐ | API pricing projections |
| Test with Multiple Models | ☐ | LLaMA, GPT-2, Mistral |
| Document README | ☐ | Formulas, usage, results |
| Push to GitHub | ☐ | Repository, commit, push |


## Key Concepts to Understand

- [ ] What is tokenization and why does it matter?
- [ ] How do embeddings work in LLMs?
- [ ] What is self-attention and how does it work?
- [ ] What is KV-cache and why is it important?
- [ ] How do you calculate VRAM requirements?
- [ ] What are the different precision formats?
- [ ] What is the difference between FP16 and BF16?
- [ ] How does quantization work (INT8, INT4)?
- [ ] What is a context window and how does it scale?
- [ ] How do you estimate API costs?

---

**End of Week 1 Notes**
