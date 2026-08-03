

#  COMPLETE WEEK 4 NOTES - NLP Fundamentals (ULTIMATE NOOB-FRIENDLY GUIDE)

## Professional Training Documentation
**Tech Prime Pvt Limited - Advanced AI/ML Internship Program**

---

#  TABLE OF CONTENTS

1. [What is NLP? - For Absolute Beginners](#1-what-is-nlp-for-absolute-beginners)
2. [Tokenization - Breaking Text into Pieces](#2-tokenization-breaking-text-into-pieces)
3. [Word Embeddings - Giving Meaning to Words](#3-word-embeddings-giving-meaning-to-words)
4. [RNNs - Remembering What Came Before](#4-rnns-remembering-what-came-before)
5. [LSTM - Solving the Memory Problem](#5-lstm-solving-the-memory-problem)
6. [Transformers - The Modern Revolution](#6-transformers-the-modern-revolution)
7. [HuggingFace Ecosystem - NLP Made Easy](#7-huggingface-ecosystem-nlp-made-easy)
8. [Complete Working Code - Sentiment Analysis System](#8-complete-working-code-sentiment-analysis-system)
9. [Common Issues and Solutions](#9-common-issues-and-solutions)
10. [Quick Reference - All Code Patterns](#10-quick-reference-all-code-patterns)

---

# 1. WHAT IS NLP? - FOR ABSOLUTE BEGINNERS

## 1.1 The BIG Question: What is NLP?

**Analogy:** Think of NLP as teaching computers to understand human language, just like you learned English in school.

```
HOW HUMANS LEARN LANGUAGE:
┌─────────────────────────────────────────────────────────────┐
│ 1. Learn letters → 2. Learn words → 3. Learn sentences   │
│ 4. Understand meaning → 5. Understand context             │
│ 6. Respond appropriately                                   │
└─────────────────────────────────────────────────────────────┘

HOW COMPUTERS LEARN LANGUAGE (NLP):
┌─────────────────────────────────────────────────────────────┐
│ 1. Tokenization (letters→tokens)                          │
│ 2. Embeddings (tokens→numbers)                            │
│ 3. Understand relationships (RNN/LSTM/Transformer)        │
│ 4. Learn meaning (training)                               │
│ 5. Make predictions (classification/translation/etc.)    │
└─────────────────────────────────────────────────────────────┘
```

### What Makes NLP So Hard?

```
PROBLEM 1: AMBIGUITY
"I saw a man with a telescope."
Who has the telescope? Me or the man?

PROBLEM 2: CONTEXT
"Bank" = Financial institution OR River edge
"I went to the bank to deposit money." → Financial
"The river bank was flooded." → River edge

PROBLEM 3: INFORMAL LANGUAGE
"OMG this movie is lit fr fr no cap" → Good or bad?
"lowkey kinda mid tbh" → 7 different ways to say "average"

PROBLEM 4: MULTIPLE LANGUAGES
"Hello" = English
"Hola" = Spanish
"Bonjour" = French
"नमस्ते" = Hindi
```

## 1.2 NLP Pipeline - The Complete Journey

```python
# ============ THE NLP PIPELINE ============

def nlp_pipeline_explained():
    """
    Every NLP project follows these steps
    """
    
    print("="*70)
    print("THE COMPLETE NLP PIPELINE")
    print("="*70)
    
    pipeline = {
        "Step 1: Data Collection": {
            "What": "Gathering text data",
            "Example": "Movie reviews, tweets, customer feedback",
            "Code": 'texts = ["I love this movie!", "This is terrible."]'
        },
        "Step 2: Preprocessing": {
            "What": "Cleaning and normalizing text",
            "Example": "Lowercasing, removing punctuation, tokenization",
            "Code": 'text = "I LOVE this!!!" → "i love this"'
        },
        "Step 3: Feature Extraction": {
            "What": "Converting text to numbers",
            "Example": "Word embeddings, TF-IDF, bag-of-words",
            "Code": '"cat" → [0.2, -0.5, 0.8, ...]'
        },
        "Step 4: Model Training": {
            "What": "Learning patterns from data",
            "Example": "RNN, LSTM, Transformer",
            "Code": 'model.fit(X_train, y_train)'
        },
        "Step 5: Evaluation": {
            "What": "Testing model performance",
            "Example": "Accuracy, F1-score, confusion matrix",
            "Code": 'accuracy = model.evaluate(X_test, y_test)'
        },
        "Step 6: Inference": {
            "What": "Making predictions on new data",
            "Example": "Is this review positive or negative?",
            "Code": 'prediction = model.predict("This movie is great!")'
        }
    }
    
    for step, info in pipeline.items():
        print(f"\n📌 {step}")
        print(f"   What: {info['What']}")
        print(f"   Example: {info['Example']}")
        print(f"   Code: {info['Code']}")

nlp_pipeline_explained()
```

### Visual: The NLP Pipeline

```
Raw Text → Preprocessing → Features → Model → Prediction
    ↓           ↓             ↓          ↓         ↓
┌─────────┐ ┌──────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│"I love  │ │"i love   │ │[0.2,   │ │ RNN/   │ │"POSI-  │
│ this    │ │ this     │ │-0.5,   │ │ LSTM/  │ │ TIVE"  │
│ movie!" │ │ movie"   │ │0.8]    │ │ TRANS  │ │        │
└─────────┘ └──────────┘ └────────┘ └────────┘ └────────┘
```

## 1.3 Types of NLP Tasks

```python
# ============ NLP TASKS CATEGORIZED ============

def nlp_task_types():
    """
    Everything you can do with NLP
    """
    
    print("="*70)
    print("NLP TASK TYPES")
    print("="*70)
    
    tasks = {
        "📝 TEXT CLASSIFICATION": {
            "Description": "Assign a category to text",
            "Examples": [
                "Sentiment Analysis: Positive/Negative",
                "Spam Detection: Spam/Not Spam",
                "Topic Classification: Sports/Politics/Tech"
            ],
            "Your Project": "SENTIMENT ANALYSIS ✅"
        },
        
        "🔄 SEQUENCE LABELING": {
            "Description": "Label each token in a sequence",
            "Examples": [
                "Named Entity Recognition: Person/Location/Organization",
                "Part-of-Speech Tagging: Noun/Verb/Adjective",
                "Chunking: Noun phrase/Verb phrase"
            ]
        },
        
        "📄 TEXT GENERATION": {
            "Description": "Generate new text",
            "Examples": [
                "Machine Translation: English→French",
                "Text Summarization: Long→Short",
                "Creative Writing: Poems, stories"
            ]
        },
        
        "❓ QUESTION ANSWERING": {
            "Description": "Answer questions from text",
            "Examples": [
                "SQuAD: Reading comprehension",
                "Chatbots: Customer service",
                "Search: Finding answers in documents"
            ]
        },
        
        "💬 CONVERSATIONAL AI": {
            "Description": "Hold conversations",
            "Examples": [
                "ChatGPT, Claude, Gemini",
                "Customer service bots",
                "Virtual assistants"
            ]
        }
    }
    
    for task_type, info in tasks.items():
        print(f"\n{task_type}")
        print(f"   {info['Description']}")
        print(f"   Examples:")
        for ex in info['Examples']:
            print(f"   • {ex}")
        if 'Your Project' in info:
            print(f"   ⭐ {info['Your Project']}")

nlp_task_types()
```

---

# 2. TOKENIZATION - BREAKING TEXT INTO PIECES

## 2.1 What is Tokenization?

**Analogy:** Tokenization is like cutting a pizza into slices. You can't eat the whole pizza at once, just like you can't process a whole sentence at once.

```
SENTENCE: "I love NLP!"
    ↓ TOKENIZATION ↓
┌─────┬─────┬─────┬─────┐
│  I  │love │ NLP │  !  │
└─────┴─────┴─────┴─────┘
  Token Token Token Token
```

### Types of Tokenization

```python
# ============ TYPES OF TOKENIZATION ============

def tokenization_types():
    """
    Different ways to break text into tokens
    """
    
    text = "I love Natural Language Processing!"
    
    print("="*70)
    print("TYPES OF TOKENIZATION")
    print("="*70)
    print(f"Original Text: '{text}'\n")
    
    # ============ TYPE 1: WORD TOKENIZATION ============
    print("📌 1. WORD TOKENIZATION")
    print("   Break by spaces and punctuation")
    print("   ['I', 'love', 'Natural', 'Language', 'Processing!']")
    print("   ✅ Easy to understand")
    print("   ❌ Huge vocabulary (100,000+ words)")
    print("   ❌ 'Natural' and 'natural' are different tokens\n")
    
    # ============ TYPE 2: CHARACTER TOKENIZATION ============
    print("📌 2. CHARACTER TOKENIZATION")
    print("   Break by each character")
    print("   ['I', ' ', 'l', 'o', 'v', 'e', ' ', 'N', 'a', ...]")
    print("   ✅ Small vocabulary (only 26 letters + symbols)")
    print("   ❌ Loses meaning of words")
    print("   ❌ Very long sequences (500+ tokens per sentence)\n")
    
    # ============ TYPE 3: SUBWORD TOKENIZATION ============
    print("📌 3. SUBWORD TOKENIZATION (BEST!)")
    print("   Break into meaningful sub-parts")
    print("   ['I', 'love', 'Nat', '##ural', 'Language', 'Processing', '!']")
    print("   ✅ Handles unknown words")
    print("   ✅ Smaller vocabulary (30,000-50,000 tokens)")
    print("   ✅ Best balance of size and meaning")
    print("   ✅ Used in BERT, GPT, etc.\n")
    
    # ============ TYPE 4: SENTENCE TOKENIZATION ============
    print("📌 4. SENTENCE TOKENIZATION")
    print("   Split into sentences")
    print("   ['I love NLP.', 'It is amazing!']")
    print("   ✅ Good for document-level tasks")
    print("   ✅ Preserves context")

tokenization_types()
```

## 2.2 The Tokenizer Family Tree

```
TOKENIZER FAMILY TREE
======================

                    TOKENIZERS
                         |
        ┌────────────────┼────────────────┐
        │                │                │
   WORD TOKENIZERS  SUBWORD TOKENIZERS  CHARACTER TOKENIZERS
        │                │                    │
        │        ┌───────┼────────┐          │
        │        │       │        │          │
    Simple    BPE    WordPiece  Unigram   Character
    Whitespace  │       │        │
                │       │        │
           GPT-2   BERT    XLNet
           (BPE)   (WordPiece)
```

### Subword Tokenization in Detail

```python
# ============ SUBWORD TOKENIZATION DEEP DIVE ============

def subword_tokenization_explained():
    """
    Understanding BPE (Byte-Pair Encoding)
    """
    
    print("="*70)
    print("BYTE-PAIR ENCODING (BPE) - HOW IT WORKS")
    print("="*70)
    
    print("""
    Step 1: Start with character-level tokens
    ──────────────────────────────────────────
    "lower" → ['l', 'o', 'w', 'e', 'r']
    
    Step 2: Count frequency of each pair
    ──────────────────────────────────────────
    Pair 'l','o' appears 100 times
    Pair 'o','w' appears 100 times
    Pair 'w','e' appears 100 times
    Pair 'e','r' appears 100 times
    
    Step 3: Merge most frequent pair
    ──────────────────────────────────────────
    'lo' + 'we' + 'er' → 'lo', 'we', 'er'
    
    Step 4: Repeat until desired vocabulary size
    ──────────────────────────────────────────
    'low' + 'er' → 'low', 'er'
    
    Final: "lower" → ['low', 'er']
    """)
    
    # Visual example
    words = ["lower", "lowest", "lowering", "low"]
    print("\n📝 EXAMPLE ON MULTIPLE WORDS:")
    print("-" * 40)
    for word in words:
        print(f"'{word}' → ['{word[:3]}', '{word[3:]}']" if len(word) > 3 
              else f"'{word}' → ['{word}']")

subword_tokenization_explained()
```

## 2.3 Implementing Tokenization with Code

```python
# ============ TOKENIZATION WITH CODE ============

import re
from collections import defaultdict

class BasicTokenizer:
    """
    Learn tokenization from scratch - no libraries!
    
    This is how tokenizers actually work under the hood.
    """
    
    def __init__(self):
        self.vocab = {}
        self.unk_token = "<UNK>"  # Unknown token
        
    def word_tokenize(self, text):
        """
        Simple word tokenization
        """
        # Convert to lowercase
        text = text.lower()
        
        # Replace punctuation with spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split by spaces
        tokens = text.split()
        return tokens
    
    def build_vocabulary(self, texts):
        """
        Build vocabulary from list of texts
        """
        vocab = set()
        for text in texts:
            tokens = self.word_tokenize(text)
            vocab.update(tokens)
        
        self.vocab = {word: idx for idx, word in enumerate(vocab)}
        self.vocab[self.unk_token] = len(self.vocab)
        
        return self.vocab
    
    def encode(self, text):
        """
        Convert text to token IDs
        """
        tokens = self.word_tokenize(text)
        ids = []
        
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab[self.unk_token])
        
        return ids
    
    def decode(self, ids):
        """
        Convert token IDs back to text
        """
        reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        tokens = [reverse_vocab.get(idx, self.unk_token) for idx in ids]
        return ' '.join(tokens)

# ============ USING THE TOKENIZER ============

def demonstrate_tokenizer():
    """
    Show how tokenization works
    """
    
    print("="*70)
    print("TOKENIZATION DEMONSTRATION")
    print("="*70)
    
    # Create tokenizer
    tokenizer = BasicTokenizer()
    
    # Example texts
    texts = [
        "I love machine learning!",
        "Natural language processing is amazing.",
        "Deep learning is a subset of machine learning.",
        "Transformers revolutionized NLP."
    ]
    
    print("📚 Training Data:")
    for i, text in enumerate(texts, 1):
        print(f"   {i}. {text}")
    
    # Build vocabulary
    vocab = tokenizer.build_vocabulary(texts)
    print(f"\n📊 Vocabulary Size: {len(vocab)} tokens")
    print(f"\n📖 Vocabulary (first 20):")
    for i, (word, idx) in enumerate(list(vocab.items())[:20], 1):
        print(f"   {idx}: '{word}'")
    
    # Encode a new text
    new_text = "I love programming!"
    print(f"\n🔤 New Text: '{new_text}'")
    
    ids = tokenizer.encode(new_text)
    print(f"   Token IDs: {ids}")
    
    # Decode back
    decoded = tokenizer.decode(ids)
    print(f"   Decoded: '{decoded}'")
    
    print("\n💡 Note: 'programming' is not in vocabulary")
    print("   So it became <UNK> (unknown token)")

demonstrate_tokenizer()
```

## 2.4 Using Pre-trained Tokenizers

```python
# ============ USING BERT TOKENIZER ============

# Note: This requires installation:
# pip install transformers

def use_bert_tokenizer():
    """
    Demonstrate using HuggingFace's pre-trained tokenizer
    """
    
    try:
        from transformers import BertTokenizer
        
        print("="*70)
        print("USING BERT TOKENIZER")
        print("="*70)
        
        # Load pre-trained tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Example text
        text = "I love Natural Language Processing!"
        
        print(f"📝 Original: {text}")
        
        # Tokenize
        tokens = tokenizer.tokenize(text)
        print(f"\n📌 Tokens: {tokens}")
        
        # Convert to IDs
        ids = tokenizer.convert_tokens_to_ids(tokens)
        print(f"📌 Token IDs: {ids}")
        
        # Encode (includes special tokens)
        encoded = tokenizer.encode(text)
        print(f"📌 Encoded (with special tokens): {encoded}")
        
        # Decode back
        decoded = tokenizer.decode(encoded)
        print(f"📌 Decoded: {decoded}")
        
        # Explain special tokens
        print("\n🔍 EXPLANATION:")
        print("   [CLS] = Start of sequence (classification token)")
        print("   [SEP] = Separator between sentences")
        print("   [PAD] = Padding to make sequences same length")
        print("   [UNK] = Unknown words")
        print("   [MASK] = Masked tokens (for pre-training)")
        
        print("\n📊 Vocabulary size: ~30,000 tokens")
        print("   ✅ Covers most English words")
        print("   ✅ Can handle subwords")
        
    except ImportError:
        print("❌ transformers not installed. Install with: pip install transformers")

# Try it!
use_bert_tokenizer()
```

---

# 3. WORD EMBEDDINGS - GIVING MEANING TO WORDS

## 3.1 Why Do We Need Embeddings?

**Analogy:** Imagine you're explaining colors to a blind person. You can't say "red looks like red." Instead, you describe it: "Red is like fire, warm, intense." That's what embeddings do - they describe words using numbers.

```
PROBLEM: COMPUTERS ONLY UNDERSTAND NUMBERS
┌─────────────────────────────────────────────────────┐
│ "cat" → ? How do we represent this as numbers?    │
└─────────────────────────────────────────────────────┘

SOLUTION 1: ONE-HOT ENCODING
┌─────────────────────────────────────────────────────┐
│ "cat" → [1, 0, 0, 0, 0, 0, 0, ...]               │
│ "dog" → [0, 1, 0, 0, 0, 0, 0, ...]               │
│ "car" → [0, 0, 1, 0, 0, 0, 0, ...]               │
│                                                   │
│ PROBLEM: 100,000 dimensions for 100,000 words!    │
│ "cat" and "dog" are equally different to "car"    │
│ No relationship between words!                    │
└─────────────────────────────────────────────────────┘

SOLUTION 2: WORD EMBEDDINGS (DENSE VECTORS)
┌─────────────────────────────────────────────────────┐
│ "cat" → [0.2, -0.5, 0.8, -0.1, 0.3, ...]        │
│ "dog" → [0.3, -0.4, 0.7, -0.2, 0.4, ...]        │
│ "car" → [0.8, -0.1, -0.3, 0.9, 0.2, ...]        │
│                                                   │
│ ✅ Small dimensions (50-300)                      │
│ ✅ Similar words = similar vectors                │
│ ✅ Captures relationships                         │
└─────────────────────────────────────────────────────┘
```

## 3.2 Word Embedding Visualized

```python
# ============ VISUALIZING WORD EMBEDDINGS ============

def visualize_embeddings():
    """
    Show how embeddings capture meaning
    """
    
    print("="*70)
    print("WORD EMBEDDINGS - HOW THEY WORK")
    print("="*70)
    
    print("""
    📊 3D Visualization of Word Embeddings:
    ┌─────────────────────────────────────────┐
    │           ANIMALS (Dimension 1)         │
    │                ▲                        │
    │         cat ●  │  dog ●                │
    │          ▲     │     ▲                  │
    │          │     │     │                  │
    │          │     │     │                  │
    │     horse●     │     │                  │
    │          │     │     │                  │
    │          └─────┼─────┘                  │
    │                │                        │
    │          ●car  │  ●truck               │
    │                │                        │
    │                │                        │
    │          VEHICLES (Dimension 2)        │
    └─────────────────────────────────────────┘
    
    Similar words are closer together!
    cat ≈ dog (both animals)
    car ≈ truck (both vehicles)
    cat is NOT close to car (different categories)
    """)
    
    print("\n🔢 ACTUAL EMBEDDINGS EXAMPLE:")
    embeddings = {
        "cat": [0.2, -0.5, 0.8, -0.1, 0.3],
        "dog": [0.3, -0.4, 0.7, -0.2, 0.4],
        "car": [0.8, -0.1, -0.3, 0.9, 0.2],
        "truck": [0.7, -0.2, -0.4, 0.8, 0.3],
        "house": [-0.5, 0.8, -0.1, -0.3, 0.6]
    }
    
    import numpy as np
    from scipy.spatial.distance import cosine
    
    print("\n   Word embeddings (5 dimensions):")
    for word, vec in embeddings.items():
        print(f"   {word}: {vec}")
    
    print("\n   Similarity between words (cosine distance):")
    print(f"   cat vs dog: {cosine(embeddings['cat'], embeddings['dog']):.3f} (close ✅)")
    print(f"   cat vs car: {cosine(embeddings['cat'], embeddings['car']):.3f} (far ✅)")
    print(f"   car vs truck: {cosine(embeddings['car'], embeddings['truck']):.3f} (close ✅)")

visualize_embeddings()
```

## 3.3 Types of Word Embeddings

```python
# ============ TYPES OF WORD EMBEDDINGS ============

def embedding_types():
    """
    Different embedding techniques
    """
    
    print("="*70)
    print("TYPES OF WORD EMBEDDINGS")
    print("="*70)
    
    print("""
    📌 1. WORD2VEC (2013)
    ──────────────────────
    "You shall know a word by the company it keeps"
    
    Two architectures:
    • CBOW: Predict word from context
      "I love ___ NLP" → "learning"
    
    • Skip-gram: Predict context from word
      "learning" → "I", "love", "NLP"
    
    ✅ Captures semantic meaning
    ✅ Fast training
    ❌ Single vector per word (can't handle polysemy)
    ❌ No context-dependent meaning
    
    Example: "bank" has same vector for river and financial
    
    
    📌 2. GLOVE (Global Vectors) (2014)
    ─────────────────────────────────────
    Combines Word2Vec with matrix factorization
    
    "Count how often words co-occur"
    
    ✅ Uses global statistics
    ✅ Better performance than Word2Vec
    ❌ Still single vector per word
    
    
    📌 3. FASTTEXT (2016)
    ─────────────────────
    "Words are made of subwords"
    
    "learning" = "lear" + "earn" + "arni" + ... 
    
    ✅ Handles out-of-vocabulary words
    ✅ Better for morphologically rich languages
    ❌ More memory intensive
    
    
    📌 4. CONTEXTUALIZED EMBEDDINGS (ELMo, BERT) (2018+)
    ──────────────────────────────────────────────────────
    "The meaning changes with context"
    
    "I went to the bank" → bank = financial
    "I sat by the bank" → bank = river edge
    
    ✅ Context-dependent meaning
    ✅ State-of-the-art performance
    ❌ Computationally expensive
    ❌ Requires large models
    """)

embedding_types()
```

## 3.4 Implementing Word Embeddings

```python
# ============ WORD EMBEDDINGS WITH CODE ============

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class SimpleEmbeddingModel:
    """
    Building word embeddings from scratch
    
    The goal: Learn vectors that capture word relationships
    """
    
    def __init__(self, vocab_size, embedding_dim=50):
        """
        Args:
            vocab_size: Number of unique words
            embedding_dim: Size of each embedding vector
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # The embedding matrix: vocab_size × embedding_dim
        # Each row is an embedding for a word
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        print(f"✅ Created embedding matrix of shape {self.embeddings.shape}")
    
    def get_embedding(self, word_id):
        """Get embedding for a word"""
        if word_id >= self.vocab_size:
            raise ValueError(f"Word ID {word_id} out of range")
        return self.embeddings[word_id]
    
    def get_similar_words(self, word_id, top_k=5):
        """
        Find most similar words using cosine similarity
        """
        word_embedding = self.embeddings[word_id]
        
        # Calculate similarity with all words
        similarities = []
        for i, emb in enumerate(self.embeddings):
            if i != word_id:
                # Cosine similarity
                sim = np.dot(word_embedding, emb) / (
                    np.linalg.norm(word_embedding) * np.linalg.norm(emb) + 1e-8
                )
                similarities.append((i, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

# ============ USING PRE-TRAINED EMBEDDINGS ============

def use_pretrained_embeddings():
    """
    Demonstrate using pre-trained embeddings
    """
    
    try:
        import gensim.downloader as api
        
        print("="*70)
        print("USING PRE-TRAINED WORD2VEC")
        print("="*70)
        
        # Download pre-trained Word2Vec (may take a moment)
        print("📥 Downloading pre-trained Word2Vec...")
        model = api.load("word2vec-google-news-300")
        
        print("✅ Loaded Word2Vec (300 dimensions)")
        print(f"📊 Vocabulary size: {len(model.key_to_index):,} words")
        
        # Get embedding for a word
        word = "king"
        embedding = model[word]
        print(f"\n🔤 '{word}' embedding shape: {embedding.shape}")
        
        # Find similar words
        print(f"\n🔍 Words similar to '{word}':")
        for similar, score in model.most_similar(word, topn=5):
            print(f"   {similar}: {score:.3f}")
        
        # Word analogies
        print("\n🧩 Word Analogies:")
        print("   king - man + woman = ?")
        result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
        print(f"   → {result[0][0]} (should be: queen)")
        
        print("\n   paris - france + germany = ?")
        result = model.most_similar(positive=['paris', 'germany'], negative=['france'], topn=1)
        print(f"   → {result[0][0]} (should be: berlin)")
        
    except ImportError:
        print("❌ gensim not installed. Install with: pip install gensim")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   (Google News model might take a while to download)")

# ============ VISUALIZING EMBEDDINGS ============

def visualize_embeddings_pca():
    """
    Visualize embeddings in 2D using PCA
    """
    
    print("="*70)
    print("VISUALIZING WORD EMBEDDINGS")
    print("="*70)
    
    try:
        import gensim.downloader as api
        
        # Load small model for speed
        model = api.load("glove-wiki-gigaword-50")
        
        # Select words to visualize
        words = [
            'king', 'queen', 'man', 'woman', 
            'paris', 'france', 'berlin', 'germany',
            'cat', 'dog', 'fish', 'bird',
            'car', 'truck', 'bicycle', 'motorcycle',
            'happy', 'sad', 'angry', 'excited'
        ]
        
        # Get embeddings
        embeddings = []
        available_words = []
        for word in words:
            if word in model:
                embeddings.append(model[word])
                available_words.append(word)
        
        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(12, 8))
        for i, word in enumerate(available_words):
            x, y = embeddings_2d[i]
            plt.scatter(x, y, s=100)
            plt.annotate(word, (x, y), fontsize=12)
        
        plt.title("Word Embeddings Visualized in 2D\n(Similar words are close together)", fontsize=14)
        plt.xlabel("PCA Dimension 1")
        plt.ylabel("PCA Dimension 2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('word_embeddings_visualization.png', dpi=300)
        plt.show()
        
        print("💾 Saved visualization as 'word_embeddings_visualization.png'")
        print("\n📊 Observations:")
        print("   • Words from same category are close")
        print("   • King/Queen are close (royalty)")
        print("   • Paris/Berlin are close (cities)")
        print("   • Cat/Dog are close (animals)")
        print("   • Happy/Sad are close (emotions)")
        
    except ImportError:
        print("❌ gensim not installed. Install with: pip install gensim")
    except Exception as e:
        print(f"❌ Error: {e}")

# Run demonstrations
use_pretrained_embeddings()
visualize_embeddings_pca()
```

---

# 4. RNNS - REMEMBERING WHAT CAME BEFORE

## 4.1 What is an RNN?

**Analogy:** Imagine reading a book. You understand each word based on what you read before. RNNs work the same way - they process text sequentially and maintain a "memory" of what came before.

```
READING A SENTENCE (Human):
┌─────────────────────────────────────────────────────────┐
│ "I" → "love" → "Natural" → "Language" → "Processing"  │
│  ↓       ↓         ↓           ↓            ↓         │
│  "I"    "I love"  "I love    "I love NLP" "I love     │
│                 Natural"               NLP!"          │
│  (Each word builds on previous understanding)          │
└─────────────────────────────────────────────────────────┘

RNN PROCESSING (Computer):
┌─────────────────────────────────────────────────────────┐
│ Word 1 → Word 2 → Word 3 → Word 4 → Word 5            │
│   ↓        ↓         ↓         ↓         ↓            │
│  h₁  ←──  h₂  ←──  h₃  ←──  h₄  ←──  h₅            │
│         (Hidden state carries memory)                  │
└─────────────────────────────────────────────────────────┘
```

### RNN Structure Visualized

```
RNN UNROLLED (Time Steps):
============================================

t=1          t=2          t=3          t=4
┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐
│     │     │     │     │     │     │     │
│  h₁ │─────│  h₂ │─────│  h₃ │─────│  h₄ │
│     │     │     │     │     │     │     │
└──┬──┘     └──┬──┘     └──┬──┘     └──┬──┘
   │            │            │            │
   │            │            │            │
┌──┴──┐     ┌──┴──┐     ┌──┴──┐     ┌──┴──┐
│  x₁ │     │  x₂ │     │  x₃ │     │  x₄ │
│ "I" │     │"love"│     │ "NLP"│     │  "!" │
└─────┘     └─────┘     └─────┘     └─────┘

h₁ = f(x₁, h₀)  where h₀ is initial memory
h₂ = f(x₂, h₁)  memory from previous step
h₃ = f(x₃, h₂)  memory from previous step
h₄ = f(x₄, h₃)  memory from previous step

Each h contains information from ALL previous words!
```

## 4.2 RNN Mathematics Made Simple

```python
# ============ RNN MATHEMATICS ============

def rnn_math_explained():
    """
    Explain RNN equations in simple terms
    """
    
    print("="*70)
    print("RNN MATHEMATICS - IN PLAIN ENGLISH")
    print("="*70)
    
    print("""
    RNN CELL AT EACH TIME STEP:
    ┌─────────────────────────────────────────────────────────┐
    │                                                       │
    │  Input:  xₜ  (current word embedding)               │
    │  Memory: hₜ₋₁ (previous hidden state)               │
    │                                                       │
    │  Step 1: Combine input and memory                    │
    │  ─────────────────────────────────────────────────    │
    │  combined = Wᵢₕ × xₜ + Wₕₕ × hₜ₋₁                 │
    │                                                       │
    │  Step 2: Apply activation (non-linearity)            │
    │  ─────────────────────────────────────────────────    │
    │  hₜ = tanh(combined)                                │
    │                                                       │
    │  Step 3: Generate output (optional)                  │
    │  ─────────────────────────────────────────────────    │
    │  yₜ = softmax(Wₕᵧ × hₜ)                             │
    │                                                       │
    └─────────────────────────────────────────────────────────┘
    
    LET'S TRANSLATE:
    ──────────────────
    • Wᵢₕ = "How important is the input?"
    • Wₕₕ = "How much should we remember?"
    • tanh = "Squash everything between -1 and 1"
    • hₜ = "New memory" (contains all past information)
    
    ANALOGY: Writing a summary of a book
    ──────────────────────────────────────
    xₜ = Current sentence
    hₜ₋₁ = Summary of previous sentences
    hₜ = Updated summary
    
    Each new word updates the summary!
    """)

rnn_math_explained()
```

## 4.3 RNN from Scratch

```python
# ============ RNN IMPLEMENTATION ============

import numpy as np
import torch
import torch.nn as nn

class RNNFromScratch:
    """
    Building an RNN from scratch to understand how it works
    
    This is the simplest RNN - it processes one token at a time
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
            input_size: Size of input vectors (e.g., embedding dimension)
            hidden_size: Size of hidden state (memory)
            output_size: Size of output (e.g., number of classes)
        """
        # Randomly initialize weights
        # Input to hidden
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        # Hidden to hidden (the "memory" connection)
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        # Hidden to output
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        
        # Biases
        self.bh = np.zeros((hidden_size, 1))  # Hidden bias
        self.by = np.zeros((output_size, 1))   # Output bias
        
        self.hidden_size = hidden_size
        
    def forward(self, inputs):
        """
        Process a sequence of inputs
        
        Args:
            inputs: List of vectors, one per time step
        
        Returns:
            outputs: List of outputs, one per time step
            h: Final hidden state
        """
        h = np.zeros((self.hidden_size, 1))  # Initial hidden state (zero)
        outputs = []
        
        for x in inputs:
            # Update hidden state
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            
            # Compute output
            y = np.dot(self.Why, h) + self.by
            
            outputs.append(y)
        
        return outputs, h
    
    def predict(self, inputs):
        """
        Predict class for a sequence
        """
        outputs, h = self.forward(inputs)
        # Use the last output for classification
        last_output = outputs[-1]
        # Apply softmax to get probabilities
        exp_y = np.exp(last_output - np.max(last_output))
        probs = exp_y / exp_y.sum()
        return probs

# ============ PYTORCH RNN ============

class PyTorchRNN(nn.Module):
    """
    RNN implemented with PyTorch
    
    Much simpler and faster than from scratch!
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(PyTorchRNN, self).__init__()
        
        # Embedding layer (converts word IDs to vectors)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,  # Input shape: (batch, sequence, features)
            nonlinearity='tanh'
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
        print(f"✅ RNN Model Created")
        print(f"   Vocabulary size: {vocab_size}")
        print(f"   Embedding dim: {embedding_dim}")
        print(f"   Hidden size: {hidden_size}")
        print(f"   Output classes: {num_classes}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input sequence of word IDs (batch, sequence_length)
        
        Returns:
            logits: Raw predictions (batch, num_classes)
        """
        # Get embeddings
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        # RNN processing
        output, hidden = self.rnn(embedded)
        # output: (batch, seq_len, hidden_size)
        # hidden: (1, batch, hidden_size)
        
        # Use the final hidden state for classification
        final_hidden = hidden[-1]  # (batch, hidden_size)
        
        # Classify
        logits = self.fc(final_hidden)  # (batch, num_classes)
        
        return logits

# ============ DEMONSTRATE RNN ============

def demonstrate_rnn():
    """
    Show how RNN works with a simple example
    """
    
    print("="*70)
    print("RNN DEMONSTRATION")
    print("="*70)
    
    # ============ FROM SCRATCH ============
    print("\n📌 RNN FROM SCRATCH:")
    print("-" * 40)
    
    # Create RNN
    input_size = 3
    hidden_size = 4
    output_size = 2
    rnn = RNNFromScratch(input_size, hidden_size, output_size)
    
    # Create dummy input sequence
    inputs = [
        np.random.randn(input_size, 1) for _ in range(5)
    ]
    
    print(f"Input sequence length: {len(inputs)}")
    print(f"Input shape: {inputs[0].shape}")
    
    # Forward pass
    outputs, final_hidden = rnn.forward(inputs)
    print(f"Outputs shape: {outputs[0].shape}")
    print(f"Final hidden state shape: {final_hidden.shape}")
    
    # ============ PYTORCH ============
    print("\n📌 PYTORCH RNN:")
    print("-" * 40)
    
    # Create model
    vocab_size = 1000
    embedding_dim = 50
    hidden_size = 64
    num_classes = 2
    
    model = PyTorchRNN(vocab_size, embedding_dim, hidden_size, num_classes)
    
    # Create dummy input
    batch_size = 4
    seq_length = 10
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    print("\n🔍 HOW RNN PROCESSES TEXT:")
    print("   Word 1 → Embedding → Hidden State 1")
    print("   Word 2 → Embedding → Hidden State 2")
    print("   Word 3 → Embedding → Hidden State 3")
    print("   ...")
    print("   Final Hidden State → Classification")

demonstrate_rnn()
```

## 4.4 RNN Problems - Vanishing Gradient

```python
# ============ VANISHING GRADIENT PROBLEM ============

def vanishing_gradient_explained():
    """
    The problem that RNNs have with long sequences
    """
    
    print("="*70)
    print("VANISHING GRADIENT PROBLEM")
    print("="*70)
    
    print("""
    THE PROBLEM:
    ─────────────────────────────────────────────────────────────
    In an RNN, information flows through many time steps.
    But the gradient (learning signal) gets smaller and smaller.
    
    VISUAL:
    ─────────────────────────────────────────────────────────────
    t=1 → t=2 → t=3 → t=4 → ... → t=100
     │     │     │     │             │
     └─────┴─────┴─────┴─────────────┘
            Each step multiplies the gradient
            by a number < 1
            
    0.9 × 0.9 × 0.9 × ... = Very small!
    
    RESULT:
    ─────────────────────────────────────────────────────────────
    "I was born in France and lived there for 20 years... I speak ___"
    
    The model forgets "France" by the end of the sentence!
    
    Why?
    • Gradient gets too small → No learning
    • Old information is "forgotten"
    • Can't learn long-range dependencies
    
    SOLUTION:
    ─────────────────────────────────────────────────────────────
    LSTM → Uses gates to control what to remember and forget
    """)
    
    # Show mathematically
    print("\n📊 MATHEMATICAL EXAMPLE:")
    print("   Hidden state update: hₜ = W × hₜ₋₁")
    print("   After 10 steps: h₁₀ = W¹⁰ × h₀")
    
    W_values = [0.9, 0.7, 0.5, 0.3]
    
    print("\n   |W| (weight magnitude) → Effect after 10 steps:")
    for w in W_values:
        effect = w ** 10
        print(f"   {w:.1f} → {effect:.6f} (gradient {'alive' if effect > 0.01 else 'dead'})")
    
    print("\n💡 THIS IS WHY WE NEED LSTM AND TRANSFORMERS!")

vanishing_gradient_explained()
```

---

# 5. LSTM - SOLVING THE MEMORY PROBLEM

## 5.1 What is LSTM?

**Analogy:** LSTM is like having a whiteboard where you can write important information and erase unimportant stuff.

```
RNN = Small notebook (can't write much, old stuff gets lost)
LSTM = Big whiteboard (can keep important info, erase unimportant)

LSTM GATES:
┌─────────────────────────────────────────────────────────────┐
│                                                           │
│  1. FORGET GATE: "What should I forget?"                  │
│     • Looks at new input and old memory                   │
│     • Decides what to erase from memory                   │
│                                                           │
│  2. INPUT GATE: "What should I remember?"                 │
│     • Looks at new input                                  │
│     • Decides what to add to memory                       │
│                                                           │
│  3. OUTPUT GATE: "What should I output?"                  │
│     • Looks at memory                                     │
│     • Decides what to output                              │
│                                                           │
└─────────────────────────────────────────────────────────────┘
```

### LSTM Architecture Visualized

```
LSTM CELL - UNROLLED:
============================================

          FORGET         INPUT          OUTPUT
           GATE           GATE           GATE
            ↓              ↓              ↓
    ┌───────────────────────────────────────────┐
    │  ┌─────┐    ┌─────┐    ┌─────┐         │
    │  │  σ  │    │  σ  │    │  σ  │         │
    │  └──┬──┘    └──┬──┘    └──┬──┘         │
    │     │           │           │            │
    │     ▼           ▼           ▼            │
    │  ┌─────┐    ┌─────┐    ┌─────┐         │
    │  │  ×  │    │  ×  │    │  ×  │         │
    │  └──┬──┘    └──┬──┘    └──┬──┘         │
    │     │           │           │            │
    │     │    ┌──────┼───────────┤            │
    │     └────┤     │           │            │
    │          │     ▼           │            │
    │          │  ┌─────┐       │            │
    │          │  │  +  │       │            │
    │          │  └─────┘       │            │
    │          │     │          │            │
    │          ▼     ▼          ▼            │
    │    ┌────────────────────────┐          │
    │    │     CELL STATE (Cₜ)    │          │
    │    │   Long-term memory     │          │
    │    └────────────────────────┘          │
    │          │                             │
    │          ▼                             │
    │    ┌──────────────┐                    │
    │    │ HIDDEN (hₜ)  │                    │
    │    └──────────────┘                    │
    └───────────────────────────────────────────┘

σ = Sigmoid activation (0 to 1) → "How much to pass"
tanh = Hyperbolic tangent (-1 to 1) → "New memory content"
× = Multiplication (gating)
+ = Addition (memory update)
```

## 5.2 LSTM Mathematics Made Simple

```python
# ============ LSTM MATHEMATICS EXPLAINED ============

def lstm_math_explained():
    """
    Explain LSTM equations in simple terms
    """
    
    print("="*70)
    print("LSTM MATHEMATICS - IN PLAIN ENGLISH")
    print("="*70)
    
    print("""
    LSTM CELL - STEP BY STEP:
    ─────────────────────────────────────────────────────────────
    
    Inputs:
    ──────
    • xₜ = Current input (word embedding)
    • hₜ₋₁ = Previous hidden state (short-term memory)
    • Cₜ₋₁ = Previous cell state (long-term memory)
    
    Step 1: FORGET GATE
    ──────────────────
    fₜ = σ(W_f · [hₜ₋₁, xₜ] + b_f)
    
    "How much of the old memory should I forget?"
    • Output is between 0 and 1
    • 0 = Forget everything
    • 1 = Remember everything
    
    Step 2: INPUT GATE
    ─────────────────
    iₜ = σ(W_i · [hₜ₋₁, xₜ] + b_i)
    C̃ₜ = tanh(W_C · [hₜ₋₁, xₜ] + b_C)
    
    "What new information should I add?"
    • iₜ = How much to add (0 to 1)
    • C̃ₜ = What to add (new candidate memory)
    
    Step 3: UPDATE CELL STATE
    ──────────────────────────
    Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ
    
    "Update my long-term memory"
    • Keep what's important (fₜ × old memory)
    • Add new information (iₜ × new candidate)
    
    Step 4: OUTPUT GATE
    ───────────────────
    oₜ = σ(W_o · [hₜ₋₁, xₜ] + b_o)
    hₜ = oₜ * tanh(Cₜ)
    
    "What should I output?"
    • oₜ = How much of memory to output
    • hₜ = Output based on memory
    
    TRANSLATION:
    ────────────
    • fₜ = "Forget these old things"
    • iₜ = "Remember these new things"
    • Cₜ = "Updated long-term memory"
    • oₜ = "Here's what I think"
    • hₜ = "My output based on memory"
    """)

lstm_math_explained()
```

## 5.3 LSTM Implementation

```python
# ============ LSTM IMPLEMENTATION ============

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM for text classification
    
    Why LSTM?
    • Handles long sequences
    • Remembers important information
    • Forgets irrelevant information
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        """
        Args:
            vocab_size: Number of unique words
            embedding_dim: Size of word embeddings
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers (stacked)
            num_classes: Number of output classes
        """
        super(LSTMModel, self).__init__()
        
        # ============ EMBEDDING LAYER ============
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # ============ LSTM LAYER ============
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input: (batch, seq_len, features)
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=False  # Use bidirectional for better performance
        )
        
        # ============ DROPOUT ============
        self.dropout = nn.Dropout(0.2)
        
        # ============ OUTPUT LAYER ============
        self.fc = nn.Linear(hidden_size, num_classes)
        
        print(f"✅ LSTM Model Created")
        print(f"   Vocabulary size: {vocab_size}")
        print(f"   Embedding dim: {embedding_dim}")
        print(f"   Hidden size: {hidden_size}")
        print(f"   Number of layers: {num_layers}")
        print(f"   Output classes: {num_classes}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input sequence (batch, seq_len)
        
        Returns:
            logits: Raw predictions (batch, num_classes)
        """
        # ============ EMBEDDING ============
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        # ============ LSTM ============
        # output: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        # c_n: (num_layers, batch, hidden_size)
        output, (h_n, c_n) = self.lstm(embedded)
        
        # ============ GET FINAL STATE ============
        # Use the hidden state of the last layer
        final_hidden = h_n[-1]  # (batch, hidden_size)
        
        # ============ DROPOUT ============
        final_hidden = self.dropout(final_hidden)
        
        # ============ CLASSIFICATION ============
        logits = self.fc(final_hidden)  # (batch, num_classes)
        
        return logits

# ============ BIDIRECTIONAL LSTM ============

class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM - Reads text both ways
    
    Why Bidirectional?
    • Context from both directions
    • Better understanding of meaning
    • Common in modern NLP models
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        super(BiLSTMModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # ⭐ This is the key difference
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(0.2)
        
        # Note: hidden_size doubled because forward + backward
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
        print(f"✅ Bidirectional LSTM Model Created")
        print(f"   (Reads text both forward AND backward)")
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (h_n, c_n) = self.lstm(embedded)
        
        # Combine forward and backward final states
        # h_n: (num_layers * 2, batch, hidden_size)
        final_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        
        final_hidden = self.dropout(final_hidden)
        logits = self.fc(final_hidden)
        
        return logits

# ============ DEMONSTRATE LSTM ============

def demonstrate_lstm():
    """
    Show how LSTM works
    """
    
    print("="*70)
    print("LSTM DEMONSTRATION")
    print("="*70)
    
    # ============ CREATE MODEL ============
    vocab_size = 10000
    embedding_dim = 100
    hidden_size = 128
    num_layers = 2
    num_classes = 2
    
    print("\n📌 CREATING LSTM MODEL:")
    print("-" * 40)
    
    model = LSTMModel(vocab_size, embedding_dim, hidden_size, num_layers, num_classes)
    
    # ============ FORWARD PASS ============
    batch_size = 4
    seq_length = 20
    
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    print(f"\n📌 FORWARD PASS:")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   (batch_size={batch_size}, seq_len={seq_length})")
    
    output = model(dummy_input)
    print(f"   Output shape: {output.shape}")
    print(f"   (batch_size={batch_size}, num_classes={num_classes})")
    
    # ============ COMPARE RNN VS LSTM ============
    print("\n📌 RNN VS LSTM COMPARISON:")
    print("-" * 40)
    
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │                    RNN                    LSTM         │
    ├─────────────────────────────────────────────────────────┤
    │ Memory           Short-term only         Short + Long │
    │ Gradients        Vanishing               Stable       │
    │ Long Sequences   ❌ Forgets              ✅ Remembers  │
    │ Training         Simple                  Complex      │
    │ Parameters       Few                     Many         │
    │ Speed            Fast                    Slower       │
    │ Use Case         Simple tasks            Complex NLP  │
    └─────────────────────────────────────────────────────────┘
    """)

demonstrate_lstm()
```

## 5.4 Bidirectional LSTM Explained

```python
# ============ BIDIRECTIONAL LSTM ============

def bidirectional_lstm_explained():
    """
    Why reading both ways matters
    """
    
    print("="*70)
    print("BIDIRECTIONAL LSTM - SEEING BOTH WAYS")
    print("="*70)
    
    print("""
    READING ONLY FORWARD (Unidirectional):
    ──────────────────────────────────────
    "I ____ NLP"
    
    Can you guess the missing word?
    • Could be "love", "hate", "study", etc.
    • Need more context!
    
    READING BOTH WAYS (Bidirectional):
    ──────────────────────────────────────
    "I ____ NLP" + "I ____ NLP is my favorite subject"
    
    Now you know: "I love NLP"
    Because context from the right tells you about the missing word!
    
    VISUAL:
    ──────────────────────────────────────
    
    Unidirectional:
    ┌─────────────────────────────────────────┐
    │ "I" → "love" → "NLP"                  │
    │  ↓       ↓        ↓                    │
    │  h₁ →    h₂ →     h₃                   │
    └─────────────────────────────────────────┘
    
    Bidirectional:
    ┌─────────────────────────────────────────┐
    │  → → → → → → → → → → → → → → →       │
    │ "I" → "love" → "NLP"                  │
    │  ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←       │
    └─────────────────────────────────────────┘
    
    BENEFITS:
    ──────────────────────────────────────────
    • More context = Better understanding
    • Works better for sentiment analysis
    • Captures dependencies in both directions
    • Standard in modern NLP models
    """)
    
    # Show example
    print("\n📝 EXAMPLE:")
    print("   Sentence: 'The movie was not good'")
    print()
    print("   Unidirectional LSTM:")
    print("   • Reads: The → movie → was → not → good")
    print("   • At 'not', doesn't know what comes after")
    print("   • Might think 'not' = negative")
    print()
    print("   Bidirectional LSTM:")
    print("   • Reads: The → movie → was → not → good")
    print("   • Also reads: good → not → was → movie → The")
    print("   • Knows 'not' modifies 'good'")
    print("   • Understands: 'not good' = negative")

bidirectional_lstm_explained()
```

---

# 6. TRANSFORMERS - THE MODERN REVOLUTION

## 6.1 Why Transformers?

**Problem:** RNNs/LSTMs process text sequentially (one word at a time). This is slow and can't see the whole sentence at once.

**Solution:** Transformers process ALL words at the same time using **Attention Mechanism**.

```
RNN/LSTM PROCESSING:
┌─────────────────────────────────────────────────────────────┐
│ Word 1 → Word 2 → Word 3 → Word 4 → Word 5              │
│  ↓        ↓        ↓        ↓        ↓                    │
│ h₁ →     h₂ →     h₃ →     h₄ →     h₅                   │
│                                                           │
│ ⏱️ Takes 5 steps (slow)                                  │
│ 😕 Can't see all words at once                           │
└─────────────────────────────────────────────────────────────┘

TRANSFORMER PROCESSING:
┌─────────────────────────────────────────────────────────────┐
│ Word 1 ─────┐                                              │
│ Word 2 ─────┼──→ ALL PAIRS OF WORDS ──→ Context         │
│ Word 3 ─────┤    at the same time!                        │
│ Word 4 ─────┤                                              │
│ Word 5 ─────┘                                              │
│                                                           │
│ ⚡ Takes 1 step (fast!)                                   │
│ ✅ Can see all words at once                              │
└─────────────────────────────────────────────────────────────┘
```

## 6.2 Attention Mechanism - The Core of Transformers

**Analogy:** Attention is like reading a document with a highlighter. You focus on the most important words.

```
ATTENTION VISUALIZED:
═══════════════════════════════════════════════════════════════

Sentence: "The cat sat on the mat"

When processing "cat":
┌─────────────────────────────────────────────────────────────┐
│ "cat" pays attention to:                                   │
│ • "The" → 0.1 (not important)                              │
│ • "cat" → 0.8 (very important)                            │
│ • "sat" → 0.5 (somewhat important)                        │
│ • "on" → 0.1 (not important)                              │
│ • "mat" → 0.3 (somewhat important)                        │
└─────────────────────────────────────────────────────────────┘

HEATMAP:
─────────────────────────────────────────────────────────────
            The   cat   sat   on    the   mat
    The     ████  ████  ████  ████  ████  ████
    cat     ████  ████████████  ████  ████  ████
    sat     ████  ████  ████████████  ████  ████
    on      ████  ████  ████  ████████████  ████
    the     ████  ████  ████  ████  ████████████
    mat     ████  ████  ████  ████  ████  ████████████

(Each word pays attention to every other word!)
```

### How Attention Works

```python
# ============ ATTENTION MECHANISM EXPLAINED ============

def attention_explained():
    """
    Step-by-step explanation of self-attention
    """
    
    print("="*70)
    print("SELF-ATTENTION - HOW IT WORKS")
    print("="*70)
    
    print("""
    STEP 1: Create Query, Key, Value for each word
    ─────────────────────────────────────────────────────────────
    For each word, create 3 vectors:
    • Query (Q): "What am I looking for?"
    • Key (K): "What do I have?"
    • Value (V): "What information do I provide?"
    
    STEP 2: Calculate Attention Scores
    ─────────────────────────────────────────────────────────────
    Score(Q, K) = Q · Kᵀ / √d
    
    "How relevant is this word to me?"
    • Higher score = More attention
    • Lower score = Less attention
    
    STEP 3: Apply Softmax
    ─────────────────────────────────────────────────────────────
    Attention Weights = Softmax(Scores)
    
    "Convert scores to probabilities"
    • All weights sum to 1
    • Higher weights = More focus
    
    STEP 4: Weighted Sum of Values
    ─────────────────────────────────────────────────────────────
    Output = Attention Weights × Values
    
    "Combine information from all words"
    • Words with high attention contribute more
    • Words with low attention contribute less
    
    VISUAL:
    ─────────────────────────────────────────────────────────────
    For each word "I" in "I love NLP":
    
    Query("I") × Key("I") = Score(I,I)  → How much I pays attention to I
    Query("I") × Key("love") = Score(I,love) → How much I pays attention to love
    Query("I") × Key("NLP") = Score(I,NLP) → How much I pays attention to NLP
    
    Softmax → Weights → New representation for "I"
    """)

attention_explained()
```

## 6.3 Transformer Architecture

```python
# ============ TRANSFORMER ARCHITECTURE ============

def transformer_architecture():
    """
    Complete Transformer architecture breakdown
    """
    
    print("="*70)
    print("TRANSFORMER ARCHITECTURE - COMPLETE BREAKDOWN")
    print("="*70)
    
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │              TRANSFORMER ARCHITECTURE                  │
    ├─────────────────────────────────────────────────────────┤
    │                                                       │
    │  INPUT: "I love NLP"                                  │
    │      ↓                                                │
    │  ┌─────────────────────────────────────────────┐      │
    │  │        INPUT EMBEDDING                       │      │
    │  │   (Convert words to vectors)                 │      │
    │  └─────────────────────────────────────────────┘      │
    │      ↓                                                │
    │  ┌─────────────────────────────────────────────┐      │
    │  │        POSITIONAL ENCODING                   │      │
    │  │   (Add word position information)            │      │
    │  └─────────────────────────────────────────────┘      │
    │      ↓                                                │
    │  ┌─────────────────────────────────────────────┐      │
    │  │       TRANSFORMER BLOCK × N                 │      │
    │  │  ┌─────────────────────────────────────┐    │      │
    │  │  │    MULTI-HEAD SELF-ATTENTION        │    │      │
    │  │  │   (Words pay attention to each other)│    │      │
    │  │  └─────────────────────────────────────┘    │      │
    │  │      ↓                                     │      │
    │  │  ┌─────────────────────────────────────┐    │      │
    │  │  │    ADD & NORMALIZE                   │    │      │
    │  │  │   (Residual connection + LayerNorm)  │    │      │
    │  │  └─────────────────────────────────────┘    │      │
    │  │      ↓                                     │      │
    │  │  ┌─────────────────────────────────────┐    │      │
    │  │  │    FEED FORWARD NETWORK              │    │      │
    │  │  │   (Process each word independently)  │    │      │
    │  │  └─────────────────────────────────────┘    │      │
    │  │      ↓                                     │      │
    │  │  ┌─────────────────────────────────────┐    │      │
    │  │  │    ADD & NORMALIZE                   │    │      │
    │  │  └─────────────────────────────────────┘    │      │
    │  └─────────────────────────────────────────────┘      │
    │      ↓                                                │
    │  ┌─────────────────────────────────────────────┐      │
    │  │        OUTPUT LAYER                         │      │
    │  │   (Classification / Translation / etc.)     │      │
    │  └─────────────────────────────────────────────┘      │
    │      ↓                                                │
    │  OUTPUT: "Positive" / "Negative" / ...               │
    └─────────────────────────────────────────────────────────┘
    
    KEY COMPONENTS EXPLAINED:
    ─────────────────────────────────────────────────────────────
    
    1. POSITIONAL ENCODING:
    "Words need to know their position"
    • sin/cos functions to encode position
    • Without this, "I love you" and "You love I" are the same!
    
    2. MULTI-HEAD ATTENTION:
    "Multiple perspectives"
    • 8-16 attention heads
    • Each head learns different patterns
    • Captures different relationships
    
    3. ADD & NORMALIZE:
    "Residual connections + Layer Normalization"
    • Residual: Input + Output (helps gradient flow)
    • LayerNorm: Normalize across features (stabilizes training)
    
    4. FEED FORWARD NETWORK:
    "Process each word independently"
    • Two linear layers with ReLU
    • Same network applied to each position
    """)

transformer_architecture()
```

## 6.4 Transformer vs LSTM vs RNN

```python
# ============ COMPARISON TABLE ============

def model_comparison():
    """
    Compare different NLP architectures
    """
    
    print("="*70)
    print("NLP ARCHITECTURES COMPARISON")
    print("="*70)
    
    comparison = {
        "Feature": [
            "Processing Type",
            "Memory",
            "Long-range Dependencies",
            "Parallelization",
            "Training Speed",
            "Memory Required",
            "Context Window",
            "Best For"
        ],
        "RNN": [
            "Sequential",
            "Short-term only",
            "❌ Poor",
            "❌ None",
            "Slow",
            "Low",
            "Short",
            "Simple sequences"
        ],
        "LSTM": [
            "Sequential",
            "Short + Long-term",
            "✅ Good",
            "❌ Limited",
            "Medium",
            "Medium",
            "Medium",
            "Complex sequences"
        ],
        "Transformer": [
            "Parallel",
            "All positions",
            "✅ Excellent",
            "✅ Full",
            "Fast",
            "High",
            "Long (entire text)",
            "State-of-the-art NLP"
        ]
    }
    
    # Print table
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    for i in range(len(comparison["Feature"])):
        feature = comparison["Feature"][i]
        rnn_val = comparison["RNN"][i]
        lstm_val = comparison["LSTM"][i]
        trans_val = comparison["Transformer"][i]
        
        print(f"│ {feature:<20} │ {rnn_val:<12} │ {lstm_val:<12} │ {trans_val:<12} │")
        
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    print("\n📊 WHEN TO USE EACH:")
    print("   • RNN: Short sequences, simple tasks, limited resources")
    print("   • LSTM: Longer sequences, complex tasks, moderate resources")
    print("   • Transformer: Large datasets, state-of-the-art performance, enough resources")

model_comparison()
```

## 6.5 BERT - The Most Important Transformer

```python
# ============ BERT EXPLAINED ============

def bert_explained():
    """
    BERT - Bidirectional Encoder Representations from Transformers
    """
    
    print("="*70)
    print("BERT - THE FOUNDATION OF MODERN NLP")
    print("="*70)
    
    print("""
    WHAT MAKES BERT SPECIAL?
    ─────────────────────────────────────────────────────────────
    
    1. BIDIRECTIONAL:
    ─────────────────
    Reads text in BOTH directions simultaneously
    • Left-to-right AND right-to-left
    • Understands context from both sides
    • Better understanding of meaning
    
    2. MASKED LANGUAGE MODEL (MLM):
    ─────────────────────────────────
    BERT was pre-trained by predicting masked words
    
    Training: "I love [MASK] Language Processing"
    Task: Predict "Natural"
    
    Why? Forces BERT to understand context!
    
    3. NEXT SENTENCE PREDICTION (NSP):
    ──────────────────────────────────
    Predicts if two sentences are consecutive
    
    Training: 
    Sentence A: "I love NLP."
    Sentence B: "It is fascinating."
    Task: Is Sentence B the next sentence? → Yes
    
    Sentence A: "I love NLP."
    Sentence B: "The sky is blue."
    Task: Is Sentence B the next sentence? → No
    
    4. HOW BERT IS USED:
    ────────────────────
    ┌─────────────────────────────────────────────────────────┐
    │  PRE-TRAINING (Large dataset)                         │
    │  • 3.3 billion words                                  │
    │  • Books, Wikipedia, news                            │
    │  • Weeks to train                                    │
    └─────────────────────────────────────────────────────────┘
                         ↓
    ┌─────────────────────────────────────────────────────────┐
    │  FINE-TUNING (Your specific task)                     │
    │  • Smaller dataset                                   │
    │  • Hours to train                                    │
    │  • Adapts to your task                              │
    └─────────────────────────────────────────────────────────┘
                         ↓
    ┌─────────────────────────────────────────────────────────┐
    │  YOUR APPLICATION                                     │
    │  • Sentiment Analysis                                 │
    │  • Question Answering                                 │
    │  • Named Entity Recognition                           │
    └─────────────────────────────────────────────────────────┘
    
    BERT VARIANTS:
    ─────────────────────────────────────────────────────────────
    • BERT-base: 110M parameters
    • BERT-large: 340M parameters
    • DistilBERT: 66M (faster, slightly less accurate)
    • RoBERTa: Improved BERT (better training)
    • ALBERT: Lite BERT (memory efficient)
    """)

bert_explained()
```

---

# 7. HUGGINGFACE ECOSYSTEM - NLP MADE EASY

## 7.1 What is HuggingFace?

**Analogy:** HuggingFace is like the App Store for NLP models. Instead of building everything from scratch, you download pre-trained models and use them immediately.

```
WITHOUT HUGGINGFACE:
┌─────────────────────────────────────────────────────────────┐
│ 1. Collect data (months)                                  │
│ 2. Preprocess (weeks)                                     │
│ 3. Build model (months)                                   │
│ 4. Train model (months, expensive)                       │
│ 5. Test and deploy (weeks)                               │
│ Total: 6-12 months                                       │
└─────────────────────────────────────────────────────────────┘

WITH HUGGINGFACE:
┌─────────────────────────────────────────────────────────────┐
│ 1. Download pre-trained model (2 minutes)                 │
│ 2. Fine-tune on your data (hours)                         │
│ 3. Deploy (hours)                                         │
│ Total: 1-2 days                                           │
└─────────────────────────────────────────────────────────────┘
```

## 7.2 HuggingFace Ecosystem Components

```python
# ============ HUGGINGFACE ECOSYSTEM ============

def huggingface_ecosystem():
    """
    Overview of HuggingFace tools
    """
    
    print("="*70)
    print("HUGGINGFACE ECOSYSTEM - COMPLETE TOOLKIT")
    print("="*70)
    
    print("""
    📦 1. TRANSFORMERS LIBRARY
    ──────────────────────────
    The main library - includes all models
    
    • 100,000+ pre-trained models
    • Supports: BERT, GPT, RoBERTa, T5, etc.
    • PyTorch, TensorFlow, JAX support
    
    pip install transformers
    
    
    🔧 2. DATASETS LIBRARY
    ──────────────────────
    Access to thousands of datasets
    
    • 500+ datasets
    • Standard formats
    • Easy loading and preprocessing
    
    pip install datasets
    
    
    💾 3. MODEL HUB
    ───────────────
    Repository of all models
    
    • Upload your own models
    • Download others' models
    • Version control for models
    
    huggingface.co/models
    
    
    🚀 4. SPACES
    ────────────
    Host and share AI apps
    
    • Deploy demos
    • Interactive apps
    • No hosting setup needed
    
    huggingface.co/spaces
    
    
    📚 5. TOKENIZERS
    ────────────────
    Fast tokenization library
    
    • BPE, WordPiece, Unigram
    • Rust implementation (fast)
    • Used by all models
    
    pip install tokenizers
    
    
    🎯 6. ACCELERATE
    ────────────────
    Easy training on multiple GPUs
    
    • Multi-GPU training
    • Mixed precision
    • Simplified code
    
    pip install accelerate
    """)

huggingface_ecosystem()
```

## 7.3 Using HuggingFace - Quick Start

```python
# ============ HUGGINGFACE QUICK START ============

def huggingface_quickstart():
    """
    Get started with HuggingFace
    """
    
    print("="*70)
    print("HUGGINGFACE - QUICK START GUIDE")
    print("="*70)
    
    print("""
    📌 STEP 1: INSTALL
    ──────────────────
    pip install transformers torch
    
    📌 STEP 2: CHOOSE A MODEL
    ─────────────────────────
    For sentiment analysis:
    • distilbert-base-uncased-finetuned-sst-2-english
    • bert-base-uncased
    • roberta-base
    
    📌 STEP 3: LOAD AND USE
    ───────────────────────
    from transformers import pipeline
    
    # Load model
    classifier = pipeline("sentiment-analysis")
    
    # Use it!
    result = classifier("I love this movie!")
    print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]
    
    📌 STEP 4: FINE-TUNE
    ────────────────────
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    
    # Train on your data!
    
    📌 STEP 5: DEPLOY
    ─────────────────
    • Save model: model.save_pretrained("./my_model")
    • Share on Hub: model.push_to_hub("my-model")
    • Deploy as API: Spaces or HuggingFace Inference API
    """)
    
    # Try it if transformers is installed
    try:
        from transformers import pipeline
        
        print("\n🔍 TRYING IT RIGHT NOW:")
        print("-" * 40)
        
        # Load sentiment analysis pipeline
        classifier = pipeline("sentiment-analysis", device=-1)  # CPU
        
        # Test texts
        texts = [
            "I absolutely love this product! It's amazing!",
            "This is the worst thing I've ever bought.",
            "The movie was okay, nothing special.",
            "Great service, will definitely come back!"
        ]
        
        print("\n📊 Sentiment Analysis Results:")
        for text in texts:
            result = classifier(text)[0]
            label = result['label']
            score = result['score']
            
            emoji = "😊" if label == "POSITIVE" else "😞"
            print(f"   {emoji} {label}: {score:.2%} | {text}")
        
    except ImportError:
        print("\n❌ transformers not installed. Install with: pip install transformers")
    except Exception as e:
        print(f"\n❌ Error: {e}")

# Try it!
huggingface_quickstart()
```

## 7.4 HuggingFace Tokenizers

```python
# ============ HUGGINGFACE TOKENIZERS ============

def huggingface_tokenizers():
    """
    Using HuggingFace tokenizers
    """
    
    print("="*70)
    print("HUGGINGFACE TOKENIZERS - DETAILED")
    print("="*70)
    
    try:
        from transformers import AutoTokenizer
        
        # ============ LOAD TOKENIZER ============
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        print(f"📌 Tokenizer: {tokenizer.__class__.__name__}")
        print(f"📊 Vocabulary size: {tokenizer.vocab_size}")
        print(f"📌 Special tokens:")
        print(f"   [CLS]: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
        print(f"   [SEP]: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
        print(f"   [PAD]: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"   [UNK]: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
        print(f"   [MASK]: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
        
        # ============ TOKENIZE ============
        text = "I love Natural Language Processing!"
        
        print(f"\n📝 Text: '{text}'")
        
        # Tokenize
        tokens = tokenizer.tokenize(text)
        print(f"\n📌 Tokens: {tokens}")
        
        # Convert to IDs
        ids = tokenizer.convert_tokens_to_ids(tokens)
        print(f"\n📌 Token IDs: {ids}")
        
        # Full encode (with special tokens)
        encoded = tokenizer.encode(text, add_special_tokens=True)
        print(f"\n📌 Encoded (with special tokens): {encoded}")
        
        # Decode
        decoded = tokenizer.decode(encoded)
        print(f"\n📌 Decoded: '{decoded}'")
        
        # ============ BATCH ENCODING ============
        texts = [
            "I love this movie!",
            "This is terrible.",
            "The film was good."
        ]
        
        print(f"\n📌 Batch Encoding:")
        print(f"   {len(texts)} texts")
        
        # Pad to same length
        encoded_batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        print(f"   Input IDs shape: {encoded_batch['input_ids'].shape}")
        print(f"   Attention mask shape: {encoded_batch['attention_mask'].shape}")
        
        print("\n   Encoded batch:")
        for i, text in enumerate(texts):
            tokens = tokenizer.decode(encoded_batch['input_ids'][i])
            print(f"   {i}: {tokens}")
        
    except ImportError:
        print("❌ transformers not installed. Install with: pip install transformers")

huggingface_tokenizers()
```

---

# 8. COMPLETE WORKING CODE - SENTIMENT ANALYSIS SYSTEM

## 8.1 Project Overview

```python
# ============================================
# COMPLETE SENTIMENT ANALYSIS SYSTEM
# EVERY LINE EXPLAINED FOR BEGINNERS
# ============================================

def project_overview():
    """
    Overview of the sentiment analysis project
    """
    
    print("="*70)
    print("SENTIMENT ANALYSIS SYSTEM - PROJECT OVERVIEW")
    print("="*70)
    
    print("""
    🎯 PROJECT GOAL:
    ───────────────
    Build a system that classifies text as POSITIVE or NEGATIVE
    
    📊 DATA:
    ────────
    • IMDB Movie Reviews (50,000 reviews)
    • 25,000 positive, 25,000 negative
    • Balanced dataset
    
    🏗️ MODELS WE'LL BUILD:
    ─────────────────────
    1. Simple RNN (baseline)
    2. LSTM (improved memory)
    3. Bidirectional LSTM (context both ways)
    4. Transformer / BERT (state-of-the-art)
    
    📈 EVALUATION:
    ─────────────
    • Accuracy
    • F1-Score
    • Confusion Matrix
    • Training/Validation Curves
    
    📁 PROJECT STRUCTURE:
    ────────────────────
    sentiment_analysis/
    ├── data/
    │   ├── train/
    │   │   ├── pos/ (12,500 reviews)
    │   │   └── neg/ (12,500 reviews)
    │   └── test/
    │       ├── pos/ (12,500 reviews)
    │       └── neg/ (12,500 reviews)
    ├── models/
    │   ├── rnn_model.pth
    │   ├── lstm_model.pth
    │   └── bert_model/
    ├── utils/
    │   ├── tokenizer.py
    │   └── data_loader.py
    └── main.py
    """)

project_overview()
```

## 8.2 Complete Code - All Models

```python
# ============================================
# SENTIMENT ANALYSIS - COMPLETE IMPLEMENTATION
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SENTIMENT ANALYSIS SYSTEM - COMPLETE IMPLEMENTATION")
print("="*70)

# ============================================
# CONFIGURATION
# ============================================

class Config:
    # Data
    MAX_SEQ_LEN = 128
    BATCH_SIZE = 32
    TRAIN_SPLIT = 0.8
    
    # Model - RNN
    RNN_HIDDEN_SIZE = 128
    RNN_NUM_LAYERS = 2
    
    # Model - LSTM
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS = 2
    
    # Model - BERT
    BERT_MODEL_NAME = "bert-base-uncased"
    
    # Training
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"📌 Device: {DEVICE}")

# ============================================
# DATA PREPARATION
# ============================================

def create_dummy_data():
    """
    Create dummy dataset for demonstration
    In practice, you would use IMDB dataset
    """
    
    print("\n📊 Creating dummy dataset...")
    
    # Sample reviews
    positive_reviews = [
        "I absolutely loved this movie! Best film of the year.",
        "Amazing performances, great story, highly recommended.",
        "A masterpiece of modern cinema, breathtaking visuals.",
        "Brilliant acting, wonderful screenplay, must watch!",
        "I was blown away by this film. Simply incredible.",
        "Outstanding movie with fantastic performances.",
        "A cinematic masterpiece that will stay with me forever.",
        "Excellent direction, stellar cast, perfect movie."
    ]
    
    negative_reviews = [
        "Terrible movie, completely wasted my time.",
        "Disappointing plot, poor acting, avoid this film.",
        "One of the worst movies I've ever seen. Boring.",
        "Bad screenplay, wooden acting, total disaster.",
        "I hated every minute of this film. Awful.",
        "Boring and predictable, completely forgettable.",
        "Waste of time and money, would not recommend.",
        "Poorly made film with no redeeming qualities."
    ]
    
    # Duplicate to create more data
    reviews = []
    labels = []
    
    for i in range(100):
        # Add some variation
        pos_idx = i % len(positive_reviews)
        neg_idx = i % len(negative_reviews)
        
        reviews.append(positive_reviews[pos_idx])
        labels.append(1)  # Positive
        
        reviews.append(negative_reviews[neg_idx])
        labels.append(0)  # Negative
    
    # Create DataFrame
    df = pd.DataFrame({'text': reviews, 'label': labels})
    
    print(f"✅ Created {len(df)} samples")
    print(f"   Positive: {df[df.label==1].shape[0]}")
    print(f"   Negative: {df[df.label==0].shape[0]}")
    
    return df

# ============================================
# SIMPLE VOCABULARY + TOKENIZER
# ============================================

class SimpleTokenizer:
    """
    Build vocabulary from text
    """
    
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.CLS_TOKEN = '<CLS>'
        self.SEP_TOKEN = '<SEP>'
    
    def build_vocab(self, texts, max_vocab=10000):
        """
        Build vocabulary from texts
        """
        # Count word frequencies
        word_counts = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create vocabulary
        vocab = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1, 
                 self.CLS_TOKEN: 2, self.SEP_TOKEN: 3}
        
        for i, (word, _) in enumerate(sorted_words[:max_vocab - 4]):
            vocab[word] = i + 4
        
        self.word_to_idx = vocab
        self.idx_to_word = {idx: word for word, idx in vocab.items()}
        self.vocab_size = len(vocab)
        
        print(f"✅ Vocabulary size: {self.vocab_size}")
        
        return self
    
    def encode(self, text, max_len=None):
        """
        Convert text to token IDs
        """
        words = text.lower().split()
        
        # Add CLS token at start
        ids = [self.word_to_idx[self.CLS_TOKEN]]
        
        # Convert words to IDs
        for word in words:
            if word in self.word_to_idx:
                ids.append(self.word_to_idx[word])
            else:
                ids.append(self.word_to_idx[self.UNK_TOKEN])
        
        # Add SEP token at end
        ids.append(self.word_to_idx[self.SEP_TOKEN])
        
        # Pad or truncate
        if max_len:
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [self.word_to_idx[self.PAD_TOKEN]] * (max_len - len(ids))
        
        return ids
    
    def decode(self, ids):
        """
        Convert IDs back to text
        """
        words = []
        for idx in ids:
            if idx in self.idx_to_word:
                word = self.idx_to_word[idx]
                if word not in [self.PAD_TOKEN, self.CLS_TOKEN, self.SEP_TOKEN]:
                    words.append(word)
        return ' '.join(words)

# ============================================
# SENTIMENT DATASET
# ============================================

class SentimentDataset(Dataset):
    """
    Dataset for sentiment analysis
    """
    
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encode text
        ids = self.tokenizer.encode(text, self.max_len)
        
        # Convert to tensor
        ids = torch.tensor(ids, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        
        return ids, label

# ============================================
# RNN MODEL
# ============================================

class RNNModel(nn.Module):
    """
    Simple RNN for sentiment analysis
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(
            embedding_dim, hidden_size, num_layers,
            batch_first=True, dropout=0.2 if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        final_hidden = hidden[-1]
        final_hidden = self.dropout(final_hidden)
        logits = self.fc(final_hidden)
        return logits

# ============================================
# LSTM MODEL
# ============================================

class LSTMModel(nn.Module):
    """
    LSTM for sentiment analysis
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers,
            batch_first=True, dropout=0.2 if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        final_hidden = hidden[-1]
        final_hidden = self.dropout(final_hidden)
        logits = self.fc(final_hidden)
        return logits

# ============================================
# BIDIRECTIONAL LSTM MODEL
# ============================================

class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM for sentiment analysis
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        super(BiLSTMModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        
        # Combine forward and backward final states
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        final_hidden = self.dropout(final_hidden)
        logits = self.fc(final_hidden)
        return logits

# ============================================
# TRAINING FUNCTIONS
# ============================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        
        # Get predictions
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1

def train_model(model, train_loader, val_loader, epochs, lr, device):
    """
    Complete training pipeline
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    best_acc = 0
    
    print(f"\n📊 Training {model.__class__.__name__}")
    print("-" * 40)
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'best_{model.__class__.__name__}.pth')
            print(f"  ✅ Best model saved! (Acc: {best_acc:.4f})")
    
    return model, history

# ============================================
# PLOTTING FUNCTIONS
# ============================================

def plot_history(history, model_name):
    """
    Plot training history
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name} - Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_history.png', dpi=300)
    plt.show()
    
    print(f"💾 Saved plot as '{model_name}_history.png'")

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """
    Complete sentiment analysis pipeline
    """
    
    print("="*70)
    print("SENTIMENT ANALYSIS SYSTEM - COMPLETE PIPELINE")
    print("="*70)
    
    # ============ DATA PREPARATION ============
    print("\n[1/6] Preparing Data...")
    print("-" * 40)
    
    df = create_dummy_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].values, df['label'].values,
        test_size=0.2, random_state=42
    )
    
    print(f"Train: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # ============ BUILD VOCABULARY ============
    print("\n[2/6] Building Vocabulary...")
    print("-" * 40)
    
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(X_train)
    
    # ============ CREATE DATALOADERS ============
    print("\n[3/6] Creating DataLoaders...")
    print("-" * 40)
    
    train_dataset = SentimentDataset(X_train, y_train, tokenizer, Config.MAX_SEQ_LEN)
    test_dataset = SentimentDataset(X_test, y_test, tokenizer, Config.MAX_SEQ_LEN)
    
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # ============ TRAIN MODELS ============
    print("\n[4/6] Training Models...")
    print("-" * 40)
    
    vocab_size = tokenizer.vocab_size
    embedding_dim = 100
    num_classes = 2
    
    models = {
        "RNN": RNNModel(vocab_size, embedding_dim, Config.RNN_HIDDEN_SIZE, 
                        Config.RNN_NUM_LAYERS, num_classes),
        "LSTM": LSTMModel(vocab_size, embedding_dim, Config.LSTM_HIDDEN_SIZE, 
                         Config.LSTM_NUM_LAYERS, num_classes),
        "BiLSTM": BiLSTMModel(vocab_size, embedding_dim, Config.LSTM_HIDDEN_SIZE, 
                             Config.LSTM_NUM_LAYERS, num_classes)
    }
    
    histories = {}
    results = {}
    
    for name, model in models.items():
        print(f"\n📌 Training {name}...")
        
        trained_model, history = train_model(
            model, train_loader, test_loader,
            epochs=Config.EPOCHS, lr=Config.LEARNING_RATE,
            device=Config.DEVICE
        )
        
        histories[name] = history
        results[name] = history['val_acc'][-1]
        
        # Plot history
        plot_history(history, name)
    
    # ============ COMPARE RESULTS ============
    print("\n[5/6] Model Comparison...")
    print("-" * 40)
    
    print("\n📊 FINAL RESULTS:")
    print("┌─────────────────────────────────────────────────────┐")
    print("│ Model              │ Validation Accuracy │ Rank   │")
    print("├─────────────────────────────────────────────────────┤")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for i, (name, acc) in enumerate(sorted_results, 1):
        medal = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"#{i}"
        print(f"│ {name:<18} │ {acc:>19.4f} │ {medal:<6} │")
    print("└─────────────────────────────────────────────────────┘")
    
    # ============ TEST BEST MODEL ============
    print("\n[6/6] Testing Best Model...")
    print("-" * 40)
    
    best_model_name = sorted_results[0][0]
    print(f"\n🏆 Best Model: {best_model_name}")
    
    # Load best model
    best_model = models[best_model_name]
    best_model.load_state_dict(torch.load(f'best_{best_model_name}.pth'))
    best_model.to(Config.DEVICE)
    
    # Test on new examples
    test_examples = [
        "This movie is absolutely amazing! I loved every minute.",
        "Terrible film, waste of money, don't watch it.",
        "The acting was good but the plot was boring.",
        "I would definitely recommend this to everyone!",
        "What a disappointment. Expected much more.",
        "An incredible masterpiece of modern cinema."
    ]
    
    print("\n📝 Testing on New Examples:")
    print("-" * 40)
    
    for text in test_examples:
        ids = tokenizer.encode(text, Config.MAX_SEQ_LEN)
        ids = torch.tensor([ids], dtype=torch.long).to(Config.DEVICE)
        
        with torch.no_grad():
            output = best_model(ids)
            _, pred = torch.max(output, 1)
            pred_label = "POSITIVE 😊" if pred.item() == 1 else "NEGATIVE 😞"
        
        print(f"   {pred_label}: {text[:50]}...")
    
    print("\n" + "="*70)
    print("✅ SENTIMENT ANALYSIS SYSTEM COMPLETE!")
    print("="*70)
    
    return models, histories, results

if __name__ == "__main__":
    main()
```

---

# 9. COMMON ISSUES AND SOLUTIONS

## 9.1 NLP-Specific Issues

```python
# ============ COMMON NLP ISSUES ============

def nlp_issues_and_solutions():
    """
    Common problems in NLP and their solutions
    """
    
    print("="*70)
    print("COMMON NLP ISSUES AND SOLUTIONS")
    print("="*70)
    
    issues = {
        "1. Vocabulary Size": {
            "Problem": "Huge vocabulary (100,000+ words)",
            "Solutions": [
                "Use subword tokenization (BPE, WordPiece)",
                "Limit vocabulary size (e.g., 50,000)",
                "Use pre-trained tokenizers",
                "Handle unknown words with <UNK>"
            ]
        },
        "2. Sequence Length": {
            "Problem": "Varying sequence lengths",
            "Solutions": [
                "Padding (fill with <PAD> tokens)",
                "Truncation (cut long sequences)",
                "Dynamic RNNs (handle variable length)",
                "Attention mechanisms"
            ]
        },
        "3. Out-of-Vocabulary (OOV)": {
            "Problem": "Words not in training vocabulary",
            "Solutions": [
                "Subword tokenization",
                "FastText (handles OOV)",
                "Contextual embeddings (BERT)",
                "Use <UNK> token"
            ]
        },
        "4. Class Imbalance": {
            "Problem": "Uneven distribution of classes",
            "Solutions": [
                "Class weights",
                "Oversampling minority class",
                "Undersampling majority class",
                "Focal Loss"
            ]
        },
        "5. Overfitting": {
            "Problem": "Model memorizes training data",
            "Solutions": [
                "Dropout (0.2-0.5)",
                "Early stopping",
                "Regularization (L1/L2)",
                "Data augmentation",
                "Less model capacity"
            ]
        },
        "6. GPU Memory": {
            "Problem": "Large models don't fit in GPU memory",
            "Solutions": [
                "Reduce batch size",
                "Use gradient accumulation",
                "Use smaller model",
                "Mixed precision training",
                "Model parallelism"
            ]
        }
    }
    
    for issue, info in issues.items():
        print(f"\n🔴 {issue}")
        print(f"   Problem: {info['Problem']}")
        print(f"   Solutions:")
        for sol in info['Solutions']:
            print(f"   • {sol}")

nlp_issues_and_solutions()
```

## 9.2 Debugging NLP Models

```python
# ============ DEBUGGING NLP MODELS ============

def debugging_tips():
    """
    Tips for debugging NLP models
    """
    
    print("="*70)
    print("NLP MODEL DEBUGGING TIPS")
    print("="*70)
    
    print("""
     1. TOKENIZATION CHECK
    ────────────────────────
    • Print tokenized text
    • Check for missing special tokens
    • Verify vocabulary size
    • Check for <UNK> tokens
    
     2. DATA CHECK
    ────────────────
    • Print one batch of data
    • Check shape of tensors
    • Verify labels are correct
    • Check data distribution
    
     3. MODEL CHECK
    ──────────────────
    • Print model architecture
    • Verify parameter count
    • Check input/output shapes
    • Test with dummy input
    
     4. TRAINING CHECK
    ─────────────────────
    • Monitor loss (should decrease)
    • Monitor accuracy (should increase)
    • Check for overfitting
    • Use gradient clipping
    
     5. INFERENCE CHECK
    ──────────────────────
    • Test on known examples
    • Test on edge cases
    • Check confidence scores
    • Test with different lengths
    """)

debugging_tips()
```

---

# 10. QUICK REFERENCE - ALL CODE PATTERNS

## 10.1 Tokenization Patterns

```python
# ============ TOKENIZATION PATTERNS ============

# Simple word tokenization
def simple_tokenize(text):
    return text.lower().split()

# BPE Tokenization
from tokenizers import ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(["text.txt"])

# HuggingFace Tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("I love NLP")
ids = tokenizer.encode("I love NLP")
```

## 10.2 Model Patterns

```python
# ============ MODEL PATTERNS ============

# RNN
class RNNModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.RNN(emb_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        _, h = self.rnn(x)
        return self.fc(h[-1])

# LSTM
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (h, c) = self.lstm(x)
        return self.fc(h[-1])

# Bidirectional LSTM
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (h, c) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(h)

# BERT
from transformers import AutoModel
class BERTModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(outputs.pooler_output)
```

## 10.3 Training Patterns

```python
# ============ TRAINING PATTERNS ============

# Basic training loop
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluation
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
    
    return total_loss / len(dataloader), predictions

# Save and load
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
```

## 10.4 HuggingFace Patterns

```python
# ============ HUGGINGFACE PATTERNS ============

# Load model and tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

# Tokenize batch
inputs = tokenizer(
    texts, 
    padding=True, 
    truncation=True, 
    max_length=128,
    return_tensors="pt"
)

# Forward pass
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=1)

# Fine-tuning
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

## 10.5 Data Patterns

```python
# ============ DATA PATTERNS ============

# Custom Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# DataLoader
dataloader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=4
)
```

---

**End of Week 4 Notes - Complete Noob-Friendly Guide**

## 📌 Key Takeaways

1. **NLP Pipeline**: Text → Tokenization → Embeddings → Model → Prediction
2. **Tokenization**: Break text into tokens (word, subword, character)
3. **Embeddings**: Convert words to vectors that capture meaning
4. **RNN**: Process text sequentially with memory
5. **LSTM**: Improved memory with gates (forget, input, output)
6. **Transformers**: Process all tokens simultaneously with attention
7. **BERT**: Bidirectional transformer - state-of-the-art NLP
8. **HuggingFace**: Pre-trained models made easy
9. **Sentiment Analysis**: Classify text as positive or negative

---
