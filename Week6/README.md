
## Week 6: Large Language Models - Complete Study Notes

### Table of Contents

1. Introduction to Large Language Models
2. LLM Architecture (Full Deep Dive)
3. Pre-training and Training Process
4. Prompt Engineering (Complete Guide)
5. Parameter-Efficient Fine-Tuning (LoRA)
6. LLM Evaluation (All Metrics)
7. Retrieval-Augmented Generation (RAG)
8. LLM Deployment and Optimization
9. Ethics and Safety in LLMs
10. Complete Project: Fine-Tuned LLM-Powered Chatbot

---

## 1. Introduction to Large Language Models

### 1.1 What is a Large Language Model?

A Large Language Model (LLM) is a type of artificial intelligence model designed to understand, generate, and manipulate human language. These models are called "large" because they contain billions of parameters and are trained on massive amounts of text data.

**Simple Definition:**
An LLM is a computer program that has read billions of pages of text and learned to predict what words should come next in any given context.

**Key Characteristics:**

| Characteristic | Explanation | Example |
|----------------|-------------|---------|
| **Scale** | Billions of parameters | GPT-3 has 175 billion parameters |
| **Data** | Trained on massive text corpora | Wikipedia, books, web pages, articles |
| **Capabilities** | Can perform many tasks without specific training | Translation, summarization, code generation |
| **Emergent Abilities** | Skills that appear at large scales | Reasoning, chain-of-thought, few-shot learning |

### 1.2 How LLMs Work - The Core Concept

LLMs work on a fundamental principle: **next token prediction**.

```
INPUT: "The capital of France is"
MODEL PREDICTS: "Paris"

INPUT: "The capital of France is Paris"
MODEL PREDICTS: "."

FINAL OUTPUT: "The capital of France is Paris."
```

**Tokenization Process:**

```
RAW TEXT: "I love NLP"
    ↓
TOKENIZATION: ["I", "love", "NL", "##P"]
    ↓
TOKEN IDS: [1045, 2293, 17953, 2361]
    ↓
EMBEDDINGS: [[0.2, -0.5, ...], [0.1, 0.7, ...], ...]
    ↓
MODEL PROCESSING: → →
    ↓
PREDICTION: Next token probabilities
```

### 1.3 Evolution of Language Models

```
TIMELINE:
═══════════════════════════════════════════════════════════════

2017: Transformer Architecture
    ↓
2018: BERT (340M parameters)
    ↓
2019: GPT-2 (1.5B parameters)
    ↓
2020: GPT-3 (175B parameters)
    ↓
2021: PaLM (540B parameters)
    ↓
2022: ChatGPT (based on GPT-3.5)
    ↓
2023: GPT-4 (estimated 1.7T parameters)
    ↓
2024: Claude 3, Gemini, Llama 3

KEY TREND: Models keep getting larger and more capable
```

### 1.4 Types of LLMs

| Type | Architecture | Examples | Best For |
|------|--------------|----------|----------|
| **Decoder-Only** | Transformer decoder layers (causal attention) | GPT, Llama, Mistral | Text generation, chat, reasoning |
| **Encoder-Only** | Transformer encoder layers (bidirectional attention) | BERT, RoBERTa | Text understanding, classification, NER |
| **Encoder-Decoder** | Both encoder and decoder layers | T5, BART | Translation, summarization, text-to-text tasks |

---

## 2. LLM Architecture (Full Deep Dive)

### 2.1 Complete Transformer Architecture

The transformer is the foundation of all modern LLMs. Understanding it is essential.

```
COMPLETE TRANSFORMER ARCHITECTURE:
═══════════════════════════════════════════════════════════════

INPUT: "I love NLP"
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. INPUT EMBEDDING                                         │
│    Convert tokens to vectors                               │
│    "I" → [0.2, -0.5, 0.8, ...]                           │
│    "love" → [0.1, 0.7, -0.3, ...]                        │
│    "NLP" → [-0.4, 0.3, 0.9, ...]                         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. POSITIONAL ENCODING                                     │
│    Add position information                                │
│    Position 0: [0.00, 1.00, 0.00, ...]                    │
│    Position 1: [0.84, 0.54, 0.01, ...]                    │
│    Position 2: [0.91, -0.42, 0.02, ...]                   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. TRANSFORMER BLOCK (Repeated N times)                    │
│                                                           │
│  ┌─────────────────────────────────────────────────┐      │
│  │ 3a. MULTI-HEAD SELF-ATTENTION                  │      │
│  │     Each word looks at all other words         │      │
│  │     • Query: "What am I looking for?"          │      │
│  │     • Key: "What information do I have?"       │      │
│  │     • Value: "What do I provide?"              │      │
│  └─────────────────────────────────────────────────┘      │
│    │                                                      │
│    ▼                                                      │
│  ┌─────────────────────────────────────────────────┐      │
│  │ 3b. ADD & NORMALIZE                           │      │
│  │     Residual connection + Layer Normalization  │      │
│  └─────────────────────────────────────────────────┘      │
│    │                                                      │
│    ▼                                                      │
│  ┌─────────────────────────────────────────────────┐      │
│  │ 3c. FEED FORWARD NETWORK                       │      │
│  │     • Linear Layer 1: hidden_size × 4         │      │
│  │     • Activation: GELU                         │      │
│  │     • Linear Layer 2: hidden_size × 4 → hidden│      │
│  └─────────────────────────────────────────────────┘      │
│    │                                                      │
│    ▼                                                      │
│  ┌─────────────────────────────────────────────────┐      │
│  │ 3d. ADD & NORMALIZE                           │      │
│  │     Residual connection + Layer Normalization  │      │
│  └─────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. OUTPUT LAYER                                            │
│    • Classification: Use [CLS] token                       │
│    • NER: Label each token                                 │
│    • Generation: Predict next token                        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
OUTPUT: "J'aime le NLP" (for translation task)
```

### 2.2 Self-Attention Mechanism

The self-attention mechanism is the most important innovation in transformers.

```
SELF-ATTENTION STEP-BY-STEP:
═══════════════════════════════════════════════════════════════

STEP 1: CREATE QUERY, KEY, VALUE VECTORS
────────────────────────────────────────────────────────────

Input: "I love NLP"

For each word, create three vectors:
┌─────────┬──────────┬──────────┬──────────┐
│  Word   │  Query   │   Key    │  Value   │
├─────────┼──────────┼──────────┼──────────┤
│   I     │    Q₁    │    K₁    │    V₁    │
│  love   │    Q₂    │    K₂    │    V₂    │
│   NLP   │    Q₃    │    K₃    │    V₃    │
└─────────┴──────────┴──────────┴──────────┘


STEP 2: CALCULATE ATTENTION SCORES
────────────────────────────────────────────────────────────

Score(Q, K) = Q × Kᵀ / √d

For "I" (Q₁):
• Score(Q₁, K₁) = How much "I" attends to "I"
• Score(Q₁, K₂) = How much "I" attends to "love"
• Score(Q₁, K₃) = How much "I" attends to "NLP"


STEP 3: APPLY SOFTMAX
────────────────────────────────────────────────────────────

Weights = Softmax(Scores)

Example attention weights for "I":
┌─────────┬──────────┐
│  Word   │  Weight  │
├─────────┼──────────┤
│   I     │   0.1    │
│  love   │   0.8    │
│   NLP   │   0.1    │
└─────────┴──────────┘

"I" pays most attention to "love"!


STEP 4: CALCULATE WEIGHTED SUM
────────────────────────────────────────────────────────────

Output for "I" = 0.1×V₁ + 0.8×V₂ + 0.1×V₃

This gives "I" information from all words!
```

### 2.3 Multi-Head Attention

Multi-head attention uses multiple attention mechanisms in parallel.

```
MULTI-HEAD ATTENTION:
═══════════════════════════════════════════════════════════════

Each head learns different patterns:

Head 1: Learns syntactic relationships
   "cat" → "sat" (subject-verb)
   
Head 2: Learns semantic relationships
   "cat" → "animal" (meaning)

Head 3: Learns positional relationships
   "the" → "cat" (adjacent)

Head 4: Learns specific patterns
   "I" → "love" (pronoun-verb)

┌─────────────────────────────────────────────────────────────┐
│                    MULTI-HEAD OUTPUT                        │
│                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │   Head 1    │  │   Head 2    │  │   Head 3    │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
│        │               │               │                  │
│        └───────────────┼───────────────┘                  │
│                        ▼                                 │
│              ┌─────────────────┐                         │
│              │   Concatenate   │                         │
│              └─────────────────┘                         │
│                        │                                 │
│                        ▼                                 │
│              ┌─────────────────┐                         │
│              │   Linear Layer  │                         │
│              └─────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Positional Encoding

Transformers don't naturally understand word order. Positional encoding adds this information.

```
POSITIONAL ENCODING:
═══════════════════════════════════════════════════════════════

Why needed: "I love you" ≠ "You love I"

Input: [I, love, NLP]

Without Positional Encoding:
┌─────────────────────────────────────────────────────────────┐
│ Model sees: [I, love, NLP] as a set, not a sequence       │
│ "I love NLP" = "NLP love I" (same to the model!)          │
└─────────────────────────────────────────────────────────────┘

With Positional Encoding:
┌─────────────────────────────────────────────────────────────┐
│ Position 0: [sin(0), cos(0), ...] → "I"                  │
│ Position 1: [sin(1), cos(1), ...] → "love"               │
│ Position 2: [sin(2), cos(2), ...] → "NLP"                │
│                                                           │
│ Now model knows: "I" is first, "love" is second, etc.    │
└─────────────────────────────────────────────────────────────┘

Formula for Positional Encoding:

For position pos and dimension i:
- If i is even: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- If i is odd: PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

This creates unique patterns for each position!
```

### 2.5 Feed-Forward Networks

The feed-forward network processes each token independently.

```
FEED-FORWARD NETWORK:
═══════════════════════════════════════════════════════════════

PER TOKEN PROCESSING:
────────────────────────────────────────────────────────────

Input: [0.2, -0.5, 0.8, ...]  (hidden_size dimensions)

Layer 1: hidden_size → hidden_size × 4
    ↓
Activation: GELU (Gaussian Error Linear Unit)
    ↓
Layer 2: hidden_size × 4 → hidden_size
    ↓
Output: [0.3, -0.4, 0.9, ...]  (back to hidden_size)

Why × 4 expansion?
→ Allows model to learn complex patterns
→ Gives more representational power
→ Then compresses back for efficiency

GELU Activation:
────────────────────────────────────────────────────────────
GELU(x) = x × Φ(x)  (where Φ is the standard normal CDF)

- Smooth activation (unlike ReLU)
- Better for deep networks
- More natural than ReLU
```

### 2.6 Residual Connections and Layer Normalization

These are crucial for training very deep networks.

```
RESIDUAL CONNECTIONS:
═══════════════════════════════════════════════════════════════

Problem: Deep networks lose information (vanishing gradients)

Solution: Add the input to the output!

Output = Layer(x) + x

Example:
Input: [0.2, -0.5, 0.8]
Layer Output: [0.3, -0.4, 0.9]
Residual Output: [0.5, -0.9, 1.7]

Benefits:
1. Information always flows through
2. Easier to train
3. Deeper networks possible
4. Faster convergence

LAYER NORMALIZATION:
═══════════════════════════════════════════════════════════════

Normalizes across features (not batch):

For each token:
1. Calculate mean: μ = (x₁ + x₂ + ... + xₙ) / n
2. Calculate variance: σ² = Σ(xᵢ - μ)² / n
3. Normalize: x̂ᵢ = (xᵢ - μ) / √(σ² + ε)
4. Scale and shift: yᵢ = γ × x̂ᵢ + β

Benefits:
1. Stable training
2. Faster convergence
3. Works with any batch size
4. Each token normalized independently
```

### 2.7 Architecture Comparison

```
COMPARISON OF POPULAR LLM ARCHITECTURES:
═══════════════════════════════════════════════════════════════

┌─────────────┬──────────────┬──────────────┬──────────────┐
│   Model     │  Parameters  │    Layers    │   Context    │
├─────────────┼──────────────┼──────────────┼──────────────┤
│  BERT-Base  │  110M        │  12          │  512         │
│  BERT-Large │  340M        │  24          │  512         │
│  GPT-2      │  1.5B        │  48          │  1024        │
│  GPT-3      │  175B        │  96          │  2048        │
│  Llama      │  7B-70B      │  32-80       │  2048-4096   │
│  Mistral    │  7B          │  32          │  8192        │
│  Phi-2      │  2.7B        │  32          │  2048        │
│  Gemma      │  2B-7B       │  18-28       │  8192        │
└─────────────┴──────────────┴──────────────┴──────────────┘

TRENDS:
1. Models are getting larger (more parameters)
2. Context windows are growing (longer memory)
3. Efficiency is improving (better architectures)
4. Open-source models are catching up to closed ones
```

---

## 3. Pre-training and Training Process

### 3.1 Pre-training Objectives

LLMs are pre-trained using self-supervised learning objectives:

**3.1.1 Masked Language Modeling (MLM)**

Used by BERT and encoder-only models.

```
TEXT: "The cat [MASK] on the mat"
    ↓
MODEL PREDICTS: "sat"
    ↓
LOSS: Difference between predicted and actual

MASKING STRATEGY:
- 15% of tokens are masked
- 80%: replaced with [MASK]
- 10%: replaced with random token
- 10%: kept unchanged
```

**3.1.2 Causal Language Modeling (CLM)**

Used by GPT and decoder-only models.

```
TEXT: "The cat sat on the mat"
    ↓
Prediction: "The" → next token "cat"
Prediction: "The cat" → next token "sat"
Prediction: "The cat sat" → next token "on"
    ↓
LOSS: Sum of all prediction errors

This is the "next token prediction" task!
```

**3.1.3 Span Corruption (T5)**

Used by T5 and encoder-decoder models.

```
TEXT: "The cat sat on the mat"
    ↓
CORRUPT: "The cat [X] the mat"
    ↓
PREDICT: "[X] = sat on"
    ↓
LOSS: Difference between predicted and actual span
```

### 3.2 Training Data

LLMs are trained on massive, diverse datasets.

```
TRAINING DATA SOURCES:
═══════════════════════════════════════════════════════════════

1. WIKIPEDIA
   ──────────
   • 6+ million articles
   • 2.5+ billion words
   • High quality, structured
   • Multiple languages

2. BOOKS
   ──────
   • Project Gutenberg (60,000+ books)
   • Google Books (scanned)
   • Fiction and non-fiction
   • Diverse writing styles

3. WEB PAGES
   ───────────
   • Common Crawl (billions of pages)
   • News articles
   • Blogs and forums
   • All topics

4. CODE REPOSITORIES
   ──────────────────
   • GitHub (public repos)
   • Stack Overflow
   • Programming documentation
   • Multiple languages

5. SOCIAL MEDIA
   ─────────────
   • Reddit (subreddits)
   • Twitter (tweets)
   • Conversation data
   • Informal language

DATA CLEANING:
────────────────────────────────────────────────────────────
1. Remove duplicates
2. Filter low-quality content
3. Balance languages
4. Remove personal information
5. Filter inappropriate content
6. Normalize formatting
7. Split into training/validation
```

### 3.3 Training Scale

Training an LLM requires enormous computational resources.

```
TRAINING SCALE (GPT-3):
═══════════════════════════════════════════════════════════════

DATA:
• 570 GB of text
• 300 billion tokens
• 10,000+ books
• 40+ million web pages

COMPUTE:
• 10,000+ NVIDIA V100 GPUs
• 14.8 days of training
• 3.64 × 10^23 FLOPs
• Cost: $4.6 million

TRAINING SCHEDULE:
────────────────────────────────────────────────────────────
1. Initial warmup: 0-10% of training
2. Peak learning rate: 10-80% of training
3. Decay: 80-100% of training
4. Total: 3-5 epochs (enough)

MEMORY REQUIREMENTS:
────────────────────────────────────────────────────────────
• Model weights: 700 GB (175B × 4 bytes)
• Gradients: 700 GB
• Optimizer states: 1.4 TB
• Total: ~2.8 TB RAM

PRACTICAL IMPLICATION:
────────────────────────────────────────────────────────────
For your project, you will FINE-TUNE, not pre-train!
Fine-tuning is much cheaper and faster:
• Data: 10,000-100,000 examples
• Compute: 1-4 GPUs
• Time: 1-3 hours
• Cost: $10-100
```

### 3.4 Training Challenges

```
TRAINING CHALLENGES AND SOLUTIONS:
═══════════════════════════════════════════════════════════════

1. INSTABILITY
   ─────────────
   Problem: Loss spikes, NaN values
   Solution:
   • Gradient clipping (limit magnitude)
   • Learning rate warmup
   • Layer normalization
   • Mixed precision training

2. VANISHING/EXPLODING GRADIENTS
   ──────────────────────────────
   Problem: Gradients too small or too large
   Solution:
   • Residual connections
   • Layer normalization
   • Gradient clipping
   • Careful initialization

3. OVERFITTING
   ─────────────
   Problem: Model memorizes training data
   Solution:
   • Regularization (dropout, weight decay)
   • Data augmentation
   • More data
   • Smaller model
   • Early stopping

4. MEMORY LIMITS
   ──────────────
   Problem: OOM (Out of Memory)
   Solution:
   • Gradient accumulation
   • Model parallelism
   • Mixed precision
   • Checkpointing
   • Reduce batch size

5. SLOW TRAINING
   ──────────────
   Problem: Takes too long
   Solution:
   • Optimize data loading
   • Use mixed precision
   • Increase batch size
   • Use better hardware
   • Parallel training
```

### 3.5 Training Strategies

```
TRAINING STRATEGIES:
═══════════════════════════════════════════════════════════════

1. FULL FINE-TUNING
   ──────────────────
   All parameters updated
   Requires: Enough memory for full model + gradients
   Time: Fastest convergence
   Risk: Catastrophic forgetting

2. PARAMETER-EFFICIENT FINE-TUNING (PEFT)
   ────────────────────────────────────────
   Only a few parameters updated
   Requires: Minimal extra memory
   Time: Slightly slower convergence
   Risk: Lower capacity

3. INSTRUCTION TUNING
   ────────────────────
   Training on instruction-response pairs
   Example: "Explain X" → "Explanation of X"
   Results in better following instructions

4. RLHF (Reinforcement Learning from Human Feedback)
   ──────────────────────────────────────────────────
   Reward model: Predicts human preferences
   Policy model: Generates responses
   PPO: Optimizes policy based on rewards
   Results: More helpful, safe responses

5. CONTINUAL PRE-TRAINING
   ───────────────────────
   Continue pre-training on domain data
   Maintains language understanding
   Adds domain knowledge
   Then fine-tune on specific task
```

---

## 4. Prompt Engineering (Complete Guide)

### 4.1 What is Prompt Engineering?

Prompt engineering is the practice of designing and optimizing input prompts to elicit desired outputs from LLMs. Since LLMs are trained to respond to natural language instructions, the quality and structure of prompts significantly impact output quality.

**Definition:**
Prompt engineering is the art of asking questions in a way that gets you the best possible answer from an LLM.

### 4.2 Basic Prompting Techniques

**4.2.1 Zero-Shot Prompting**

The model is given a task description without any examples.

```
PROMPT:
Classify the sentiment of the following text as Positive or Negative.

Text: "The movie was absolutely fantastic!"

Sentiment: Positive

Why it works: The model understands the task from the instruction
When to use: When the task is simple and well-defined
```

**4.2.2 Few-Shot Prompting**

The model is given a few examples before the actual task.

```
PROMPT:
Classify sentiment.

Text: "I loved this film." → Positive
Text: "Waste of time." → Negative
Text: "The acting was superb." → Positive
Text: "Boring plot." → Negative
Text: "Amazing performances." →

Expected Response: Positive

Why it works: Examples show the pattern
When to use: When the task is complex or ambiguous
```

**4.2.3 Chain-of-Thought (CoT) Prompting**

The model is prompted to show its reasoning step by step.

```
PROMPT:
Solve this problem: "A store has 12 apples. It sells 3 apples each hour. After how many hours will there be no apples left?"

Let me think step by step:
1. The store has 12 apples
2. It sells 3 apples per hour
3. Hours = Total apples / Apples per hour
4. Hours = 12 / 3 = 4 hours

Answer: 4 hours

Why it works: Forces the model to reason
When to use: For complex reasoning tasks
```

### 4.3 Advanced Prompting Techniques

**4.3.1 System Instructions**

System prompts define the role and behavior of the LLM.

```
SYSTEM PROMPT:
You are an expert in Computer Science. You provide clear, accurate, and well-explained answers to technical questions. Always include code examples when relevant. Be concise but thorough. If you don't know something, say "I don't know" instead of making up an answer.

USER INPUT:
Explain inheritance in Python.

MODEL RESPONSE:
Inheritance is a fundamental concept in object-oriented programming...
```

**4.3.2 Structured Outputs**

Prompting for structured outputs ensures consistent, parsable responses.

```
PROMPT:
Provide your response in the following JSON format:
{
    "answer": "your answer here",
    "confidence": 0.0-1.0,
    "explanation": "your reasoning here",
    "sources": ["source1", "source2"]
}

Why it works: Consistent format for parsing
When to use: When you need machine-readable output
```

**4.3.3 Role Prompting**

Assigning a specific role or persona to the model.

```
PROMPT:
You are a senior software engineer at Google. You are mentoring a junior developer.

Now, explain how to design a REST API.

Response will be from the perspective of a senior engineer:
- Practical advice
- Industry best practices
- Potential pitfalls
- Real-world examples
```

**4.3.4 Retrieval-Augmented Generation (RAG) Prompting**

Providing relevant context from external sources.

```
PROMPT:
Context: [Retrieved relevant information from knowledge base]

Question: [User's question]

Answer based on the context provided.

Why it works: Ensures factual accuracy
When to use: When you need specific, up-to-date information
```

### 4.4 Prompt Optimization Strategies

```
PROMPT OPTIMIZATION TECHNIQUES:
═══════════════════════════════════════════════════════════════

1. BE SPECIFIC
   ─────────────
   ❌ "Write about Python"
   ✅ "Write a Python function that implements binary search with complexity analysis"

2. PROVIDE EXAMPLES
   ──────────────────
   ❌ "Explain inheritance"
   ✅ "Explain inheritance with a simple Python code example"
   
   Example: "class Parent: ... class Child(Parent): ..."

3. BREAK DOWN COMPLEX TASKS
   ──────────────────────────
   ❌ "Write a complete web application"
   ✅ "Step 1: Design the database schema... Step 2: Create API endpoints..."

4. SET FORMAT EXPECTATIONS
   ─────────────────────────
   ❌ "List the steps"
   ✅ "Provide the steps as a numbered list with brief explanations for each"

5. SPECIFY DESIRED LENGTH
   ────────────────────────
   ❌ "Explain machine learning"
   ✅ "Explain machine learning in 3 paragraphs, keeping it accessible to beginners"

6. REQUEST EXPLANATIONS
   ──────────────────────
   ❌ "Write code to sort a list"
   ✅ "Write code to sort a list and explain the algorithm's time complexity"

7. USE DIRECTIVES
   ────────────────
   ❌ "Here is a problem..."
   ✅ "Here is a problem. Solve it step by step and provide your final answer."

8. ITERATE AND REFINE
   ────────────────────
   Start with basic prompt → Evaluate → Improve → Repeat
```

### 4.5 Common Prompting Mistakes

```
MISTAKES TO AVOID:
═══════════════════════════════════════════════════════════════

1. VAGUE PROMPTS
   ──────────────
   ❌ "Tell me about AI"
   ✅ "Provide a brief overview of AI, focusing on recent developments in NLP"

2. OVERLY BROAD PROMPTS
   ──────────────────────
   ❌ "Explain everything about Python"
   ✅ "Explain Python's type system and its advantages over static typing"

3. AMBIGUOUS INSTRUCTIONS
   ────────────────────────
   ❌ "Make it better"
   ✅ "Rewrite this text to be more concise while maintaining all key information"

4. NO CONTEXT
   ────────────
   ❌ "Write code for this"
   ✅ "Write Python code for [task] that handles [specific requirements]"

5. INCONSISTENT FORMATTING
   ────────────────────────
   ❌ Mixed formats, unclear expectations
   ✅ Clear, consistent formatting

6. TOO MANY REQUIREMENTS
   ──────────────────────
   ❌ "Do A, B, C, D, E all at once"
   ✅ Break into multiple prompts or clearly prioritize
```

### 4.6 When Prompt Engineering is Not Enough

Prompt engineering has limitations, and sometimes fine-tuning is necessary.

```
LIMITATIONS OF PROMPT ENGINEERING:
═══════════════════════════════════════════════════════════════

1. CONSISTENCY
   ─────────────
   Prompt engineering gives inconsistent results
   Fine-tuning: Reliable, consistent

2. TASK-SPECIFIC FORMAT
   ──────────────────────
   Prompt engineering: May not follow strict format
   Fine-tuning: Learn exact format

3. EDGE CASES
   ────────────
   Prompt engineering: Poor handling of unusual cases
   Fine-tuning: Learn to handle edge cases

4. DOMAIN KNOWLEDGE
   ──────────────────
   Prompt engineering: Limited by model's knowledge
   Fine-tuning: Inject domain-specific knowledge

5. HALLUCINATION
   ───────────────
   Prompt engineering: Still hallucinates
   Fine-tuning: Can learn to avoid specific hallucinations

6. LATENCY
   ──────────
   Prompt engineering: Longer prompts = longer inference
   Fine-tuning: Shorter prompts, faster inference

WHEN TO FINE-TUNE:
────────────────────────────────────────────────────────────
1. Consistent structured output required
2. Domain-specific knowledge needed
3. Edge cases must be handled reliably
4. Hallucination reduction is critical
5. Latency and cost are concerns
6. Large-scale deployment
```

---

## 5. Parameter-Efficient Fine-Tuning (LoRA)

### 5.1 Why Parameter-Efficient Fine-Tuning?

Full fine-tuning of LLMs has several challenges:

```
FULL FINE-TUNING CHALLENGES:
═══════════════════════════════════════════════════════════════

1. MEMORY REQUIREMENTS
   ─────────────────────
   Model weights: 7B × 4 bytes = 28 GB
   Gradients: 28 GB
   Optimizer states: 56 GB (Adam)
   Total: 112 GB
   → Requires expensive hardware

2. TRAINING TIME
   ──────────────
   Full fine-tuning takes hours to days
   Need to update all parameters
   Slow convergence

3. STORAGE COSTS
   ──────────────
   Each task = full model copy
   10 tasks = 10 × 7B model = 70B parameters
   Expensive storage

4. CATASTROPHIC FORGETTING
   ─────────────────────────
   Updating all parameters can damage
   Pre-trained knowledge
   Model "forgets" what it learned

5. TASK SWITCHING
   ────────────────
   Need to load different model for each task
   Slow and expensive
```

### 5.2 Introduction to LoRA

**Low-Rank Adaptation (LoRA)** is a popular PEFT technique introduced by Hu et al. (2021).

**Core Idea:**
The weight update during fine-tuning has a low "intrinsic rank," meaning it can be approximated by a low-rank matrix.

```
MATHEMATICAL EXPLANATION:
═══════════════════════════════════════════════════════════════

Original forward pass:
    h = W × x

Full fine-tuning (updates W):
    h = (W + ΔW) × x

LoRA:
    h = W × x + B × A × x

Where:
- W: Original weight matrix (frozen)
- A: Low-rank matrix (d × r)
- B: Low-rank matrix (r × k)
- r << min(d, k) (very small rank)

EXAMPLE:
────────────────────────────────────────────────────────────
If W is (1000 × 1000):
- W has 1,000,000 parameters
- With LoRA (r = 8):
- A: (1000 × 8) = 8,000 parameters
- B: (8 × 1000) = 8,000 parameters
- Total: 16,000 parameters
- 1,000,000 → 16,000 (62.5× reduction!)

Training:
    • W is frozen (pre-trained knowledge preserved)
    • A and B are updated (task-specific learning)
```

### 5.3 LoRA Hyperparameters

```
LORA HYPERPARAMETERS:
═══════════════════════════════════════════════════════════════

1. RANK (r)
   ──────────
   Definition: Dimension of low-rank matrices
   Typical values: 4, 8, 16, 32, 64
   
   Effects:
   • Higher r: More expressive, more parameters
   • Lower r: More efficient, less expressive
   
   Rule of thumb:
   • r=4: Very efficient, good for simple tasks
   • r=8: Good balance
   • r=16: More capacity
   • r=32: High capacity, more memory

2. ALPHA (α)
   ───────────
   Definition: Scaling parameter
   Typical values: 8, 16, 32, 64
   
   Effects:
   • Higher α: Stronger adaptation
   • Lower α: Weaker adaptation
   
   Scaling factor: α / r
   Formula: h = Wx + (α/r) × BAx

3. TARGET MODULES
   ────────────────
   Which layers to apply LoRA to
   Usually attention layers:
   • q_proj (Query projection)
   • v_proj (Value projection)
   • k_proj (Key projection)
   • o_proj (Output projection)
   
   Can also apply to:
   • Feed-forward layers
   • Embedding layers
   • Bias terms

4. DROPOUT
   ──────────
   Regularization technique
   Typical values: 0.05, 0.1, 0.2
   Prevents overfitting
```

### 5.4 LoRA vs Full Fine-Tuning

```
COMPARISON:
═══════════════════════════════════════════════════════════════

┌─────────────────────┬──────────────────┬──────────────────┐
│      Aspect         │ Full Fine-Tuning │       LoRA       │
├─────────────────────┼──────────────────┼──────────────────┤
│ Trainable Params    │ 100% (7B)        │ <1% (16M)        │
│ Memory Required     │ Very High (112GB)│ Low (28GB)       │
│ Training Time       │ Slow (days)      │ Fast (hours)     │
│ Storage per Task    │ Full model (7GB) │ A+B (35MB)       │
│ Task Switching      │ Slow             │ Fast             │
│ Catastrophic        │ Possible         │ Minimal          │
│ Forgetting          │                  │                  │
│ Performance         │ Typically higher │ Comparable       │
│ Implementation      │ Simple           │ Slightly Complex │
└─────────────────────┴──────────────────┴──────────────────┘

WHEN TO USE LORA:
────────────────────────────────────────────────────────────
✅ Multiple tasks (many adapters)
✅ Limited GPU memory
✅ Fast experimentation
✅ Quick task switching
✅ When performance gap is acceptable

WHEN TO USE FULL FINE-TUNING:
────────────────────────────────────────────────────────────
✅ Single task
✅ Maximum performance needed
✅ Enough GPU memory
✅ Training once, deploying many times
```

### 5.5 Practical LoRA Implementation

```
IMPLEMENTATION STEPS:
═══════════════════════════════════════════════════════════════

STEP 1: Install Required Libraries
────────────────────────────────────────────────────────────
pip install peft transformers accelerate bitsandbytes

STEP 2: Load Base Model
────────────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    device_map="auto"
)

STEP 3: Configure LoRA
────────────────────────────────────────────────────────────
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

STEP 4: Apply LoRA
────────────────────────────────────────────────────────────
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 8.4M || all params: 2.7B || 0.31%

STEP 5: Train
────────────────────────────────────────────────────────────
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

STEP 6: Save and Load
────────────────────────────────────────────────────────────
# Save adapter
model.save_pretrained("./lora_adapter")

# Load adapter
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "./lora_adapter")

# Merge with base model (optional)
model = model.merge_and_unload()
```

### 5.6 Other PEFT Techniques

```
OTHER PEFT TECHNIQUES:
═══════════════════════════════════════════════════════════════

1. PREFIX TUNING
   ──────────────
   Adds learnable vectors to the beginning of input
   • No extra parameters in model
   • Trains a few thousand parameters
   • Good for generation tasks

2. PROMPT TUNING
   ──────────────
   Learns soft prompts (continuous vectors)
   • Frozen model
   • Only prompts are optimized
   • Very efficient
   • Works for classification

3. ADAPTERS
   ──────────
   Adds small bottleneck layers between layers
   • 2-5% additional parameters
   • Easy to add/remove
   • Task-specific

4. BITFIT
   ────────
   Only trains bias terms
   • < 0.1% parameters
   • Very fast
   • Good for simple tasks

COMPARISON:
────────────────────────────────────────────────────────────
┌─────────────┬──────────────┬──────────────┬──────────────┐
│  Technique  │ Parameters   │ Performance  │ Use Case     │
├─────────────┼──────────────┼──────────────┼──────────────┤
│  Full FT    │ 100%         │ Best         │ High capacity│
│  LoRA       │ < 1%         │ Very Good    │ General      │
│  Adapters   │ 2-5%         │ Good         │ Multi-task   │
│  Prefix     │ 0.1%         │ Good         │ Generation   │
│  Prompt     │ 0.01%        │ Good         │ Classification│
│  BitFit     │ 0.1%         │ Decent       │ Simple tasks │
└─────────────┴──────────────┴──────────────┴──────────────┘
```

### 5.7 Quantization for Memory Efficiency

```
QUANTIZATION:
═══════════════════════════════════════════════════════════════

Reduces model precision to save memory.

PRECISION TYPES:
────────────────────────────────────────────────────────────
1. FP32 (32-bit): Full precision, slow, high memory
2. FP16 (16-bit): Half precision, faster, lower memory
3. INT8 (8-bit): 8-bit quantization, fast, low memory
4. NF4 (4-bit): 4-bit quantization, fastest, lowest memory

IMPLEMENTATION:
────────────────────────────────────────────────────────────
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    quantization_config=bnb_config
)

MEMORY SAVING:
────────────────────────────────────────────────────────────
Model: 7B parameters (28 GB at FP32)
→ FP16: 14 GB (2x saving)
→ INT8: 7 GB (4x saving)
→ NF4: 3.5 GB (8x saving)

This makes it possible to run on consumer GPUs!
```

---

## 6. LLM Evaluation (All Metrics)

### 6.1 Why LLM Evaluation is Challenging

Unlike traditional ML models, LLMs have:
- Open-ended outputs (no single correct answer)
- Subjective quality (multiple good answers possible)
- Complex tasks (reasoning, creativity, safety)
- Language-dependent performance

```
EVALUATION CHALLENGES:
═══════════════════════════════════════════════════════════════

1. MULTIPLE CORRECT ANSWERS
   ──────────────────────────
   Question: "What is the capital of France?"
   Correct Answer: "Paris"
   Also Acceptable: "The capital of France is Paris."

   Traditional metrics fail!

2. SUBJECTIVE QUALITY
   ────────────────────
   "The movie was good" vs "The film was excellent"
   Both are valid, different quality

3. CONTEXT DEPENDENCE
   ────────────────────
   "That's sick!" = Good or Bad? (Depends on context)
   Model needs to understand context

4. CREATIVE TASKS
   ────────────────
   Summarization, translation, code generation
   Multiple "correct" outputs
   Hard to measure automatically

5. SAFETY AND BIAS
   ─────────────────
   Need to ensure responses are safe and unbiased
   Hard to measure systematically
```

### 6.2 Evaluation Dimensions

```
EVALUATION DIMENSIONS:
═══════════════════════════════════════════════════════════════

1. TASK PERFORMANCE
   ──────────────────
   How well does the model perform the task?
   • Accuracy (for classification)
   • ROUGE (for summarization)
   • BLEU (for translation)
   • Exact match (for QA)

2. FACTUAL ACCURACY
   ──────────────────
   Is the information correct?
   • Fact extraction
   • Ground truth comparison
   • Verifiable facts

3. LATENCY
   ──────────
   How fast is the response?
   • Time per query
   • Tokens per second
   • Throughput

4. COST
   ──────
   What is the cost per inference?
   • Number of tokens
   • Compute time
   • API costs

5. FORMAT COMPLIANCE
   ────────────────────
   Does it follow the specified format?
   • JSON validation
   • Schema checking
   • Format consistency

6. ROBUSTNESS
   ─────────────
   Does it handle variations?
   • Paraphrasing
   • Edge cases
   • Adversarial inputs

7. SAFETY
   ────────
   Does it refuse harmful requests?
   • Safety test cases
   • Harmful content detection
   • Bias detection

8. HALLUCINATION
   ───────────────
   Does it generate false information?
   • Fact verification
   • Source checking
   • Confidence scoring
```

### 6.3 Evaluation Methods

```
EVALUATION METHODS:
═══════════════════════════════════════════════════════════════

1. GOLDEN EXAMPLES
   ──────────────────
   Using a curated dataset of inputs and expected outputs.
   
   TEST CASE:
   Input: "What is the capital of France?"
   Expected: "Paris"
   
   Evaluation: Compare model output to expected.
   Metrics: Exact match, semantic similarity, fuzzy match.

2. LLM-AS-A-JUDGE
   ─────────────────
   Using another LLM to evaluate responses.
   
   PROMPT:
   "Rate the following response from 1-10 based on:
   - Accuracy (4 points)
   - Completeness (3 points)
   - Clarity (3 points)"
   
   Best Practices:
   • Use a different model than the one being evaluated
   • Pin the judge model and prompt version
   • Keep the judge prompt small and explicit
   • Sample-grade with humans monthly

3. FACT-BASED METRICS
   ────────────────────
   Breaking down responses into facts.
   
   FACT CATEGORIES:
   • Matched: Facts present in both response and ground truth
   • Omitted: Facts in ground truth but not response
   • Incorrect: Facts that contradict ground truth
   • Irrelevant: Facts not related to ground truth
   
   METRICS:
   • Fact Precision: Matched / (Matched + Incorrect + Irrelevant)
   • Fact Recall: Matched / (Matched + Omitted)
   • Accuracy Score: Combined measure

4. HUMAN EVALUATION
   ──────────────────
   Human experts evaluate responses.
   
   SCORING DIMENSIONS:
   • Coherence: Does it make sense?
   • Fluency: Is it grammatically correct?
   • Relevance: Is it on topic?
   • Helpfulness: Is it useful?
   • Safety: Is it safe?
   
   PROS: Gold standard
   CONS: Expensive, slow, subjective

5. BENCHMARK DATASETS
   ────────────────────
   Standard datasets for comparing models.
   
   EXAMPLES:
   • MMLU: Multi-task language understanding
   • GSM8K: Grade school math
   • HumanEval: Code generation
   • GLUE/SuperGLUE: General language tasks
   • HELM: Holistic evaluation
```

### 6.4 Comparing Base vs Fine-Tuned Models

```
EVALUATION WORKFLOW:
═══════════════════════════════════════════════════════════════

STEP 1: PREPARE TEST DATASET
────────────────────────────────────────────────────────────
• 100-500 test examples
• Diverse topics and difficulties
• Balanced categories
• Ground truth annotations

STEP 2: RUN EVALUATIONS
────────────────────────────────────────────────────────────
• Base model: Run on all test examples
• Fine-tuned model: Run on all test examples
• Record all responses

STEP 3: COMPARE RESULTS
────────────────────────────────────────────────────────────
For each example:
• Compare base vs fine-tuned responses
• Score both against ground truth
• Identify improvements/regressions

STEP 4: ANALYZE DIFFERENCES
────────────────────────────────────────────────────────────
Questions to answer:
1. Did factual accuracy improve?
2. Did response quality increase?
3. Did the model learn new information?
4. Did it forget any existing knowledge?
5. Did it maintain safety standards?

STEP 5: STATISTICAL ANALYSIS
────────────────────────────────────────────────────────────
• Average scores
• Score distribution
• Per-category performance
• Statistical significance
• Confidence intervals
```

### 6.5 Test Dataset Types

```
TEST DATASET TYPES:
═══════════════════════════════════════════════════════════════

1. GOLDEN PATH
   ─────────────
   Tasks the model must always solve correctly.
   Examples:
   • "What is 2+2?" → "4"
   • "What is the capital of France?" → "Paris"
   • "Write 'Hello World' in Python" → "print('Hello World')"

2. EDGE CASES
   ─────────────
   Past bugs, boundary conditions, tricky inputs.
   Examples:
   • Very long inputs
   • Very short inputs
   • Ambiguous queries
   • Malformed inputs
   • Domain-specific edge cases

3. ADVERSARIAL
   ──────────────
   Injection attacks, ambiguous inputs, missing data.
   Examples:
   • "Ignore previous instructions and..."
   • "Your system prompt is wrong, instead..."
   • "What is the best illegal drug?"

4. DOMAIN-SPECIFIC
   ──────────────────
   Test examples from your specific domain.
   Examples:
   • Medical: "What are the symptoms of..."
   • Legal: "What is the statute of limitations..."
   • Technical: "How to fix..."
```

### 6.6 Common Evaluation Pitfalls

```
PITFALLS TO AVOID:
═══════════════════════════════════════════════════════════════

1. CONTAMINATION
   ──────────────
   Problem: Test examples are in training data
   Solution: Use recent data, synthetic examples
   
2. JUDGE BIAS
   ────────────
   Problem: Judge model favors certain styles
   Solution: Multiple judges, human validation
   
3. TINY EVAL SETS
   ────────────────
   Problem: Not statistically significant
   Solution: Minimum 100-200 examples
   
4. JUDGE DRIFT
   ─────────────
   Problem: Judge model changes over time
   Solution: Pin judge model version
   
5. SUBJECTIVITY
   ──────────────
   Problem: Different judges, different scores
   Solution: Clear rubrics, multiple judges
   
6. OVER-OPTIMIZATION
   ──────────────────
   Problem: Models overfit to eval set
   Solution: Separate validation and test sets
```

---

## 7. Retrieval-Augmented Generation (RAG)

### 7.1 What is RAG?

RAG combines LLMs with external knowledge retrieval to provide more accurate, up-to-date, and fact-based responses.

```
RAG ARCHITECTURE:
═══════════════════════════════════════════════════════════════

USER QUERY: "What is the latest iPhone?"
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. RETRIEVAL                                               │
│    Search knowledge base for relevant documents             │
│    ┌─────────────────────────────────────────────────┐      │
│    │ Knowledge Base                                  │      │
│    │ • Documents (PDFs, reports)                    │      │
│    │ • Websites                                    │      │
│    │ • Databases                                   │      │
│    │ • Internal data                               │      │
│    └─────────────────────────────────────────────────┘      │
│    │                                                      │
│    ▼                                                      │
│    Retrieved: "iPhone 15 was released in September 2024"  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. GENERATION                                               │
│    LLM generates response using context                     │
│    Context: "iPhone 15 was released in September 2024"     │
│    Query: "What is the latest iPhone?"                     │
│    Response: "The latest iPhone is the iPhone 15,          │
│               released in September 2024."                 │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
RESPONSE: "The latest iPhone is the iPhone 15, released in September 2024."
```

### 7.2 Benefits of RAG

```
BENEFITS:
═══════════════════════════════════════════════════════════════

1. UP-TO-DATE INFORMATION
   ────────────────────────
   • Knowledge base can be updated regularly
   • No need to retrain model
   • Always have latest information

2. FACTUAL ACCURACY
   ──────────────────
   • Responses based on retrieved documents
   • Reduced hallucination
   • Verifiable sources

3. CUSTOM DOMAINS
   ─────────────────
   • Use company-specific documents
   • Industry-specific knowledge
   • Private data (not in model training)

4. EFFICIENCY
   ──────────────
   • Smaller model needed (no need to store all knowledge)
   • Faster updates
   • Lower training costs

5. TRANSPARENCY
   ──────────────
   • Can show sources
   • Explainable responses
   • Trustworthy outputs
```

### 7.3 RAG Components

```
RAG COMPONENTS:
═══════════════════════════════════════════════════════════════

1. KNOWLEDGE BASE
   ─────────────────
   Where the information is stored.
   
   Types:
   • Vector database (FAISS, Pinecone, Weaviate)
   • Document store (Elasticsearch, Solr)
   • SQL database
   • File system
   
   Requirements:
   • Fast retrieval
   • Scalable
   • Searchable

2. EMBEDDING MODEL
   ─────────────────
   Converts text to vectors for similarity search.
   
   Popular models:
   • OpenAI embeddings
   • BERT embeddings
   • Sentence Transformers
   • Instructor embeddings
   
   Example:
   "iPhone 15" → [0.2, -0.5, 0.8, ...]

3. RETRIEVER
   ────────────
   Finds relevant documents based on query.
   
   Methods:
   • Semantic search (cosine similarity)
   • Keyword search (BM25)
   • Hybrid (both)
   • Query expansion

4. LLM
   ──────
   Generates response using retrieved context.
   
   Options:
   • Any LLM (GPT, Llama, Claude)
   • Fine-tuned or base model
   • Small or large model
   • Open or closed source
```

### 7.4 RAG vs Fine-Tuning

```
RAG VS FINE-TUNING:
═══════════════════════════════════════════════════════════════

┌─────────────────────┬──────────────────┬──────────────────┐
│      Aspect         │       RAG        │   Fine-Tuning    │
├─────────────────────┼──────────────────┼──────────────────┤
│ Knowledge Updates   │ Easy (update DB) │ Hard (retrain)   │
│ Accuracy            │ High             │ Very High        │
│ Latency             │ Higher           │ Lower            │
│ Transparency        │ High (sources)   │ Low              │
│ Customization       │ Limited          │ High             │
│ Cost                │ Medium           │ High             │
│ Implementation      │ Moderate         │ Complex          │
│ Hallucination       │ Low              │ Medium           │
│ Capability          │ Retriever-dependent │ Model-dependent │
└─────────────────────┴──────────────────┴──────────────────┘

WHEN TO USE RAG:
────────────────────────────────────────────────────────────
✅ Frequently updated knowledge
✅ Need to show sources
✅ Limited training data
✅ Multiple knowledge domains
✅ Compliance requirements

WHEN TO USE FINE-TUNING:
────────────────────────────────────────────────────────────
✅ Task-specific behavior
✅ Consistent formatting
✅ Unique style/tone
✅ Proprietary knowledge (can't share)
✅ Low-latency requirements
```

---

## 8. LLM Deployment and Optimization

### 8.1 Deployment Options

```
DEPLOYMENT OPTIONS:
═══════════════════════════════════════════════════════════════

1. LOCAL DEPLOYMENT
   ──────────────────
   Running the model on your own hardware.
   
   PROS:
   • Full control
   • No API costs
   • Data privacy
   • No rate limits
   
   CONS:
   • Hardware requirements
   • Maintenance
   • Updates needed
   • Scaling challenges

2. CLOUD DEPLOYMENT
   ──────────────────
   Running on cloud platforms (AWS, GCP, Azure).
   
   PROS:
   • Scalable
   • Managed infrastructure
   • Pay-as-you-go
   • Professional support
   
   CONS:
   • Costs
   • Vendor lock-in
   • Network dependency
   • Data transfer fees

3. API DEPLOYMENT
   ─────────────────
   Using external APIs (OpenAI, Anthropic, Cohere).
   
   PROS:
   • No infrastructure needed
   • State-of-the-art models
   • Fast setup
   • Ongoing improvements
   
   CONS:
   • API costs
   • Rate limits
   • Data privacy concerns
   • Vendor dependency

4. EDGE DEPLOYMENT
   ──────────────────
   Running on edge devices (phones, IoT).
   
   PROS:
   • Low latency
   • Privacy
   • Offline capability
   • No network needed
   
   CONS:
   • Hardware constraints
   • Smaller models
   • Battery consumption
   • Update challenges
```

### 8.2 Optimization Techniques

```
OPTIMIZATION TECHNIQUES:
═══════════════════════════════════════════════════════════════

1. QUANTIZATION
   ──────────────
   Reduce model precision.
   • FP32 → INT8: 4x memory reduction
   • FP32 → INT4: 8x memory reduction
   • Minimal accuracy loss

2. PRUNING
   ────────────
   Remove unnecessary weights.
   • Structured pruning: Remove entire neurons
   • Unstructured pruning: Remove individual weights
   • 50-70% sparsity possible

3. DISTILLATION
   ──────────────
   Train smaller model to mimic larger model.
   • Teacher: Large model (GPT-3)
   • Student: Small model (GPT-2)
   • 10-20x smaller model
   • 80-90% performance

4. CACHING
   ──────────────
   Store repeated computations.
   • KV cache (generation)
   • Embedding cache (RAG)
   • Response cache (frequent queries)

5. BATCHING
   ────────────
   Process multiple requests together.
   • Higher throughput
   • Better GPU utilization
   • Lower cost per request

6. SPECULATIVE DECODING
   ──────────────────────
   Predict future tokens efficiently.
   • Draft model: Small fast model
   • Target model: Large slow model
   • 2-3x speed improvement
```

### 8.3 Inference Optimization

```
INFERENCE OPTIMIZATION:
═══════════════════════════════════════════════════════════════

1. KV CACHING
   ────────────
   Store key-value pairs from previous tokens.
   
   Without caching:
   Token 1: Compute KV for all tokens (1)
   Token 2: Compute KV for all tokens (1,2)
   Token 3: Compute KV for all tokens (1,2,3)
   → O(n²) complexity
   
   With caching:
   Token 1: Compute KV for all tokens (1)
   Token 2: Compute only KV for token 2
   Token 3: Compute only KV for token 3
   → O(n) complexity

2. FLASH ATTENTION
   ─────────────────
   Optimized attention computation.
   • Tiling: Process chunks in memory
   • Recomputation: Compute on the fly
   • 2-4x speedup
   • Reduced memory usage

3. CONTINUOUS BATCHING
   ─────────────────────
   Process variable-length requests together.
   • Dynamic padding
   • Different sequence lengths
   • 2-3x throughput improvement

4. MODEL COMPILATION
   ───────────────────
   Optimize model for hardware.
   • JIT compilation
   • Kernel fusion
   • Graph optimization
   • 20-50% speedup

5. TENSOR PARALLELISM
   ────────────────────
   Split model across GPUs.
   • Layer splitting
   • Column/row splitting
   • For very large models
   • Near-linear scaling
```

---

## 9. Ethics and Safety in LLMs

### 9.1 Key Ethical Concerns

```
ETHICAL CONCERNS:
═══════════════════════════════════════════════════════════════

1. BIAS
   ──────
   Problem: Models reflect biases in training data.
   Examples:
   • Gender bias: "Doctor" → He
   • Racial bias: Stereotypical associations
   • Age bias: Assumptions based on age
   • Socioeconomic bias: Class assumptions
   
   Solutions:
   • Diverse training data
   • Bias detection and mitigation
   • Regular audits
   • Debiasing techniques

2. MISINFORMATION
   ─────────────────
   Problem: Models can generate false information.
   Examples:
   • Fake news
   • Incorrect facts
   • Fabricated sources
   • Hallucinated details
   
   Solutions:
   • Fact verification
   • Source attribution
   • Confidence scoring
   • Human review

3. PRIVACY
   ──────────
   Problem: Models may memorize and expose private data.
   Examples:
   • Personal information
   • Financial data
   • Medical records
   • Internal company data
   
   Solutions:
   • Data anonymization
   • Privacy-preserving training
   • Differential privacy
   • Data deletion

4. MISUSE
   ────────
   Problem: Models can be used for harmful purposes.
   Examples:
   • Generating spam
   • Creating phishing emails
   • Spreading misinformation
   • Automated harassment
   • Generating hate speech
   
   Solutions:
   • Usage policies
   • Content filtering
   • Access controls
   • Monitoring and reporting

5. ENVIRONMENTAL IMPACT
   ──────────────────────
   Problem: Training LLMs consumes significant energy.
   Examples:
   • CO2 emissions
   • Water usage
   • Electronic waste
   • Resource consumption
   
   Solutions:
   • Efficient architectures
   • Renewable energy
   • Model distillation
   • Responsible scaling
```

### 9.2 Safety Measures

```
SAFETY MEASURES:
═══════════════════════════════════════════════════════════════

1. CONTENT FILTERING
   ───────────────────
   Filtering harmful content in inputs and outputs.
   
   Types:
   • Toxicity detection
   • Hate speech detection
   • Harmful content detection
   • Profanity filtering

2. SAFETY PROMPTS
   ────────────────
   Prompts that enforce safety guidelines.
   
   Example:
   "You are a helpful, safe, and harmless assistant.
   Refuse to generate content that is:
   - Violent
   - Hateful
   - Illegal
   - Harmful"

3. OUTPUT MONITORING
   ───────────────────
   Monitoring generated content for safety issues.
   
   Methods:
   • Real-time filtering
   • Post-generation review
   • User reporting
   • Automated detection

4. USER AUTHENTICATION
   ─────────────────────
   Ensuring only authorized users access the model.
   
   Methods:
   • API keys
   • User accounts
   • Rate limiting
   • Access controls

5. REGULAR AUDITS
   ─────────────────
   Periodic review of model behavior.
   
   What to audit:
   • Bias in responses
   • Safety violations
   • User complaints
   • Performance metrics
```

---

## 10. Complete Project: Fine-Tuned LLM-Powered Chatbot

### 10.1 Project Overview

**Goal:** Fine-tune a small LLM (1B-3B parameters) on a Computer Science Q&A dataset and deploy it as a chatbot interface.

### 10.2 Project Requirements

```
REQUIREMENTS:
═══════════════════════════════════════════════════════════════

HARDWARE:
• Kaggle T4 GPU (free, 16GB VRAM) or better
• 8GB+ RAM

SOFTWARE:
• Python 3.8+
• PyTorch 2.0+
• Transformers 4.30+
• PEFT (LoRA)
• Gradio for UI

DATA:
• 10,000-50,000 CS Q&A pairs
• Diverse topics
• Quality annotations

TIME:
• Dataset preparation: 2-3 hours
• Fine-tuning: 1-3 hours
• Evaluation: 1-2 hours
• Deployment: 1-2 hours

MODELS:
• Option 1: Phi-2 (2.7B) - Best for CS
• Option 2: Gemma-2B (2B) - Good balance
• Option 3: TinyLlama (1.1B) - Fastest
```

### 10.3 Implementation Steps

```
IMPLEMENTATION STEPS:
═══════════════════════════════════════════════════════════════

DAY 1: DATA PREPARATION
────────────────────────────────────────────────────────────

1. Collect CS Q&A data
   Sources:
   • Stack Overflow Q&A
   • CS textbooks
   • Programming guides
   • Generated by GPT-4

2. Format data
   {
     "instruction": "What is inheritance?",
     "response": "Inheritance is..."
   }

3. Clean data
   • Remove duplicates
   • Fix formatting
   • Ensure quality
   • Balance topics

4. Split data
   • Train: 80%
   • Validation: 10%
   • Test: 10%

DAY 2: MODEL SETUP & LORA
────────────────────────────────────────────────────────────

1. Load base model
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")

2. Configure LoRA
   lora_config = LoraConfig(
       r=8,
       lora_alpha=16,
       target_modules=["q_proj", "v_proj"],
       lora_dropout=0.1,
   )

3. Apply LoRA
   model = get_peft_model(model, lora_config)

4. Test inference before training

DAY 3: FINE-TUNING
────────────────────────────────────────────────────────────

1. Load training data
2. Format prompts
   "<|system|>You are a CS expert...<|user|>Question<|assistant|>Answer"
3. Run training
   • epochs: 3-5
   • batch size: 4-8
   • learning rate: 2e-4
4. Save checkpoints

DAY 4: EVALUATION
────────────────────────────────────────────────────────────

1. Test on validation set
2. Compare base vs fine-tuned
3. Measure:
   • Accuracy
   • Factual correctness
   • Response quality
   • Latency

DAY 5: DEPLOYMENT
────────────────────────────────────────────────────────────

1. Build Gradio UI
   • Chat interface
   • Conversation history
   • Settings
2. Deploy with Ngrok
3. Test with real users
4. Document results
```

### 10.4 Sample Code

```
COMPLETE IMPLEMENTATION:
═══════════════════════════════════════════════════════════════

# cell 1: install dependencies
!pip install transformers peft accelerate bitsandbytes datasets
!pip install gradio pyngrok trl

# cell 2: load model with lora
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# cell 3: train
from transformers import Trainer, TrainingArguments
from datasets import Dataset

# Load data
data = [
    {"instruction": "What is Python?", "response": "Python is..."},
    # ... more data
]
dataset = Dataset.from_list(data)

def format_prompt(example):
    return f"<|user|>{example['instruction']}<|assistant|>{example['response']}"

dataset = dataset.map(lambda x: {"text": format_prompt(x)})

training_args = TrainingArguments(
    output_dir="./phi2_cs_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-4,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

# cell 4: save
model.save_pretrained("./phi2_cs_adapter")

# cell 5: inference
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# cell 6: gradio ui
import gradio as gr

def chat(message, history):
    prompt = f"<|user|>{message}<|assistant|>"
    response = generate_response(prompt)
    return response

demo = gr.ChatInterface(fn=chat)
demo.launch()

# cell 7: deploy with ngrok
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")
public_url = ngrok.connect(7860)
print(f"Public URL: {public_url}")
```

### 10.5 Expected Results

```
EXPECTED RESULTS:
═══════════════════════════════════════════════════════════════

BASE MODEL (Phi-2):
────────────────────────────────────────────────────────────
• General language understanding
• Some CS knowledge
• May hallucinate on specifics
• OK for general questions

FINE-TUNED MODEL:
────────────────────────────────────────────────────────────
• CS-specific knowledge
• Accurate code examples
• Clear explanations
• Better factual accuracy
• Consistent formatting

PERFORMANCE METRICS:
────────────────────────────────────────────────────────────
• Accuracy: 75-85%
• Factual correctness: 70-80%
• Response quality: Good
• Latency: 2-5 seconds
• Training time: 1-2 hours

IMPROVEMENTS OVER BASE:
────────────────────────────────────────────────────────────
• 20-30% better accuracy on CS questions
• More detailed explanations
• Better code examples
• More coherent responses
• Reduced hallucination
```

---

## Summary

```
WEEK 6 - KEY TAKEAWAYS:
═══════════════════════════════════════════════════════════════

1. LLM ARCHITECTURE
   • Transformers: self-attention, multi-head, positional encoding
   • Decoder-only models are most common for chat
   • Scale matters: more parameters = better performance

2. PRE-TRAINING
   • Massive data (Wikipedia, books, web)
   • Self-supervised learning (next token prediction)
   • Very expensive (millions of dollars)
   • Requires specialized hardware

3. PROMPT ENGINEERING
   • Zero-shot, few-shot, chain-of-thought
   • System prompts, role prompting
   • Structured outputs
   • Important for getting good results

4. FINE-TUNING
   • LoRA: efficient (<1% parameters)
   • Good for domain-specific tasks
   • Consistent formatting
   • Cost-effective

5. EVALUATION
   • Golden examples, LLM-as-judge
   • ROUGE, BLEU, factual accuracy
   • Compare base vs fine-tuned
   • Multiple evaluation dimensions

6. DEPLOYMENT
   • Gradio for UI
   • Ngrok for public access
   • Optimize for speed and cost
   • Monitor for safety and quality

7. ETHICS
   • Bias, misinformation, privacy
   • Content filtering, safety prompts
   • Regular audits
   • Responsible deployment
```
