# WEEK 5: ADVANCED NLP - COMPLETE NOTES

## Professional Training Documentation
**Tech Prime Pvt Limited - Advanced AI/ML Internship Program**

---

# TABLE OF CONTENTS

1. [Named Entity Recognition (NER)](#1-named-entity-recognition-ner)
2. [Fine-Tuning BERT](#2-fine-tuning-bert)
3. [Sequence-to-Sequence Models](#3-sequence-to-sequence-models)
4. [Text Summarization](#4-text-summarization)
5. [Evaluation Metrics for NLP](#5-evaluation-metrics-for-nlp)
6. [Advanced Transformer Concepts](#6-advanced-transformer-concepts)
7. [Transfer Learning in NLP](#7-transfer-learning-in-nlp)
8. [Complete Working Code - NER & Summarization System](#8-complete-working-code-ner--summarization-system)
9. [Common Issues and Solutions](#9-common-issues-and-solutions)
10. [Quick Reference - All Code Patterns](#10-quick-reference-all-code-patterns)

---

# 1. NAMED ENTITY RECOGNITION (NER)

## 1.1 What is NER?

**Analogy:** NER is like reading a document and highlighting all the important names, places, dates, and organizations - just like you would with a highlighter pen when studying.

```
TEXT: "Elon Musk founded SpaceX in Hawthorne, California on May 6, 2002."

NER HIGHLIGHTING:
═══════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────┐
│ Elon Musk    → PERSON                                      │
│ SpaceX       → ORGANIZATION                                │
│ Hawthorne    → LOCATION                                    │
│ California   → LOCATION                                    │
│ May 6, 2002  → DATE                                        │
└─────────────────────────────────────────────────────────────┘

VISUALIZED:
═══════════════════════════════════════════════════════════════

"Elon Musk"  founded  "SpaceX"  in  "Hawthorne, California"  on  "May 6, 2002"
    💃            ⚡       🏢            🏙️              📅
  PERSON      ACTION  ORGANIZATION     LOCATION          DATE
```

### Types of Named Entities

```python
# ============ NER ENTITY TYPES ============

def ner_entity_types():
    """
    Common entity types in NER
    """
    
    print("="*70)
    print("NAMED ENTITY RECOGNITION - ENTITY TYPES")
    print("="*70)
    
    entity_types = {
        "🔴 PERSON": {
            "Description": "People's names",
            "Examples": ["Elon Musk", "Albert Einstein", "Marie Curie"],
            "Code": f"PERSON"
        },
        "🏢 ORGANIZATION": {
            "Description": "Companies, institutions, etc.",
            "Examples": ["Google", "NASA", "Harvard University"],
            "Code": "ORG"
        },
        "🏙️ LOCATION": {
            "Description": "Geographical locations",
            "Examples": ["Paris", "Mount Everest", "Amazon River"],
            "Code": "LOC"
        },
        "📅 DATE": {
            "Description": "Dates and times",
            "Examples": ["January 1, 2024", "5pm", "Monday"],
            "Code": "DATE"
        },
        "💰 MONEY": {
            "Description": "Monetary values",
            "Examples": ["$1,000", "€500", "20 million dollars"],
            "Code": "MONEY"
        },
        "📊 PERCENT": {
            "Description": "Percentages",
            "Examples": ["25%", "10 percent", "half"],
            "Code": "PERCENT"
        },
        "📧 EMAIL": {
            "Description": "Email addresses",
            "Examples": ["john@email.com", "support@company.org"],
            "Code": "EMAIL"
        },
        "🔗 URL": {
            "Description": "Web addresses",
            "Examples": ["www.google.com", "https://example.com"],
            "Code": "URL"
        }
    }
    
    for entity, info in entity_types.items():
        print(f"\n{entity} ({info['Code']})")
        print(f"   {info['Description']}")
        print(f"   Examples: {', '.join(info['Examples'])}")

ner_entity_types()
```

## 1.2 How NER Works

```python
# ============ NER PIPELINE ============

def ner_pipeline_explained():
    """
    Complete NER pipeline explanation
    """
    
    print("="*70)
    print("NER PIPELINE - STEP BY STEP")
    print("="*70)
    
    print("""
    STEP 1: TOKENIZATION
    ─────────────────────
    Break text into tokens (words/subwords)
    
    Input: "Elon Musk founded SpaceX in California"
    ↓
    ['Elon', 'Musk', 'founded', 'SpaceX', 'in', 'California']
    
    
    STEP 2: PART-OF-SPEECH (POS) TAGGING
    ─────────────────────────────────────
    Identify grammatical role of each word
    
    ['Elon'(PROPN), 'Musk'(PROPN), 'founded'(VERB), 
     'SpaceX'(PROPN), 'in'(ADP), 'California'(PROPN)]
    
    
    STEP 3: DEPENDENCY PARSING
    ──────────────────────────
    Understand relationships between words
    
    Elon Musk ──founded──→ SpaceX ──in──→ California
    (Subject)   (Action)   (Object)       (Location)
    
    
    STEP 4: NER TAGGING
    ───────────────────
    Label each token with entity type
    
    ['Elon'(B-PER), 'Musk'(I-PER), 'founded'(O), 
     'SpaceX'(B-ORG), 'in'(O), 'California'(B-LOC)]
    
    BIO TAGGING SCHEME:
    ─────────────────────────────────────────────────────────────
    • B-XXX = Beginning of entity type XXX
    • I-XXX = Inside entity type XXX
    • O = Outside any entity
    
    Example: "Elon Musk" → B-PER, I-PER
             "SpaceX" → B-ORG
             "California" → B-LOC
             "founded" → O
    """)

ner_pipeline_explained()
```

### Visual: How NER Processing Works

```
Text Input
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ "Elon Musk founded SpaceX in Hawthorne, California"        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    TOKENIZATION                             │
│ ┌──────┬──────┬────────┬───────┬──┬───────────┬──────────┐ │
│ │ Elon │ Musk │ founded│SpaceX │in│Hawthorne,│California│ │
│ └──────┴──────┴────────┴───────┴──┴───────────┴──────────┘ │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    NER LABELING                             │
│ ┌──────┬──────┬────────┬───────┬──┬───────────┬──────────┐ │
│ │B-PER │I-PER │   O    │ B-ORG │O │  B-LOC    │  I-LOC   │ │
│ └──────┴──────┴────────┴───────┴──┴───────────┴──────────┘ │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    FINAL OUTPUT                             │
│ {"Elon Musk": "PERSON", "SpaceX": "ORG",                    │
│  "Hawthorne, California": "LOC"}                           │
└─────────────────────────────────────────────────────────────┘
```

## 1.3 Implementing NER with HuggingFace

```python
# ============ NER WITH HUGGINGFACE ============

def huggingface_ner():
    """
    Named Entity Recognition using HuggingFace
    """
    
    print("="*70)
    print("NER WITH HUGGINGFACE - COMPLETE IMPLEMENTATION")
    print("="*70)
    
    try:
        from transformers import pipeline
        
        print("\n📌 Load NER Pipeline...")
        nlp = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        
        print("\n📝 Sample Text:")
        text = """
        Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne 
        in April 1976 in Cupertino, California. The company is now worth over 
        $3 trillion and employs more than 150,000 people worldwide.
        """
        print(text)
        
        print("\n📊 NER Results:")
        print("-" * 50)
        
        # Get predictions
        entities = nlp(text)
        
        # Group tokens into entities
        current_entity = None
        entity_text = ""
        
        for entity in entities:
            word = entity['word']
            label = entity['entity']
            score = entity['score']
            
            # Skip special tokens
            if word.startswith('##'):
                entity_text += word[2:]
            else:
                if entity_text and current_entity:
                    print(f"   {current_entity}: '{entity_text}' (Score: {score:.3f})")
                entity_text = word
                current_entity = label
            
            # Check if this is the last token
            if entity['index'] == entities[-1]['index']:
                if entity_text and current_entity:
                    print(f"   {current_entity}: '{entity_text}' (Score: {score:.3f})")
        
    except ImportError:
        print("❌ Transformers not installed. Install with: pip install transformers")
    except Exception as e:
        print(f"❌ Error: {e}")

# Try it!
huggingface_ner()
```

## 1.4 Custom NER with PyTorch

```python
# ============ CUSTOM NER MODEL ============

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class NERDataset(Dataset):
    """
    Dataset for NER tasks
    """
    
    def __init__(self, sentences, labels, word_to_idx, label_to_idx):
        self.sentences = sentences
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.label_to_idx = label_to_idx
        self.max_len = max(len(s) for s in sentences)
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        words = self.sentences[idx]
        labels = self.labels[idx]
        
        # Convert words to indices
        word_ids = [self.word_to_idx.get(w, 1) for w in words]  # 1 = UNK
        
        # Convert labels to indices
        label_ids = [self.label_to_idx.get(l, 0) for l in labels]  # 0 = O
        
        # Pad sequences
        word_ids = word_ids + [0] * (self.max_len - len(word_ids))  # 0 = PAD
        label_ids = label_ids + [0] * (self.max_len - len(label_ids))
        
        return torch.tensor(word_ids, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)

class NERModel(nn.Module):
    """
    BiLSTM-CRF Model for NER
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_labels, embedding_weights=None):
        super(NERModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
        
        # BiLSTM
        self.bilstm = nn.LSTM(
            embedding_dim, 
            hidden_size // 2,  # Because bidirectional
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Linear layer for classification
        self.fc = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, lengths=None):
        # x: (batch, seq_len)
        embeddings = self.embedding(x)  # (batch, seq_len, embedding_dim)
        embeddings = self.dropout(embeddings)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(embeddings)  # (batch, seq_len, hidden_size)
        
        # Classification
        logits = self.fc(lstm_out)  # (batch, seq_len, num_labels)
        
        return logits

def train_ner_model():
    """
    Train a custom NER model
    """
    
    print("="*70)
    print("CUSTOM NER MODEL - TRAINING DEMONSTRATION")
    print("="*70)
    
    # ============ SAMPLE DATA ============
    print("\n📊 Creating sample training data...")
    
    sentences = [
        ["Elon", "Musk", "founded", "SpaceX", "in", "California"],
        ["Google", "is", "based", "in", "Mountain", "View"],
        ["Steve", "Jobs", "was", "CEO", "of", "Apple"],
        ["Microsoft", "was", "founded", "by", "Bill", "Gates"],
        ["Amazon", "headquarters", "is", "in", "Seattle"]
    ]
    
    # BIO labels
    labels = [
        ["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC"],
        ["B-ORG", "O", "O", "O", "B-LOC", "I-LOC"],
        ["B-PER", "I-PER", "O", "O", "O", "B-ORG"],
        ["B-ORG", "O", "O", "O", "B-PER", "I-PER"],
        ["B-ORG", "O", "O", "O", "B-LOC"]
    ]
    
    # Build vocabulary
    word_to_idx = {"<PAD>": 0, "<UNK>": 1}
    for sent in sentences:
        for word in sent:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
    
    # Build label vocabulary
    label_to_idx = {"O": 0}
    for label_seq in labels:
        for label in label_seq:
            if label not in label_to_idx:
                label_to_idx[label] = len(label_to_idx)
    
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    print(f"✅ Vocabulary size: {len(word_to_idx)}")
    print(f"✅ Number of label types: {len(label_to_idx)}")
    print(f"   Labels: {', '.join(label_to_idx.keys())}")
    
    # ============ CREATE DATASET ============
    dataset = NERDataset(sentences, labels, word_to_idx, label_to_idx)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # ============ CREATE MODEL ============
    vocab_size = len(word_to_idx)
    embedding_dim = 50
    hidden_size = 64
    num_labels = len(label_to_idx)
    
    model = NERModel(vocab_size, embedding_dim, hidden_size, num_labels)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD tokens
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n📌 Model Architecture:")
    print(f"   Vocabulary: {vocab_size}")
    print(f"   Embedding dim: {embedding_dim}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Number of labels: {num_labels}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ============ TRAINING ============
    print("\n📌 Training NER Model...")
    print("-" * 40)
    
    epochs = 20
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Forward pass
            outputs = model(inputs)  # (batch, seq_len, num_labels)
            
            # Reshape for loss calculation
            outputs = outputs.view(-1, num_labels)  # (batch * seq_len, num_labels)
            targets = targets.view(-1)  # (batch * seq_len)
            
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    print("✅ Training complete!")
    
    # ============ EVALUATION ============
    print("\n📌 Testing on Sample Sentence...")
    print("-" * 40)
    
    test_sentence = ["Elon", "Musk", "works", "at", "SpaceX"]
    
    # Convert to IDs
    test_ids = [word_to_idx.get(w, 1) for w in test_sentence]  # 1 = UNK
    test_ids = test_ids + [0] * (dataset.max_len - len(test_ids))
    test_tensor = torch.tensor([test_ids], dtype=torch.long)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(test_tensor)
        predictions = outputs.argmax(dim=-1)
    
    # Convert predictions to labels
    pred_labels = [idx_to_label[idx.item()] for idx in predictions[0]]
    
    # Print results
    print("\n   Results:")
    print("   " + "-" * 30)
    for word, pred in zip(test_sentence, pred_labels):
        if word != "<PAD>":
            print(f"   {word:12} → {pred}")
    
    # ============ VISUALIZE ============
    print("\n📌 Visualizing NER Tags:")
    print("-" * 40)
    
    # Color mapping
    color_map = {
        "B-PER": "\033[94m",  # Blue
        "I-PER": "\033[94m",
        "B-ORG": "\033[92m",  # Green
        "I-ORG": "\033[92m",
        "B-LOC": "\033[93m",  # Yellow
        "I-LOC": "\033[93m",
        "O": "\033[0m",       # Default
    }
    
    result_text = ""
    for i, (word, pred) in enumerate(zip(test_sentence, pred_labels)):
        if word != "<PAD>":
            color = color_map.get(pred, "\033[0m")
            if pred != "O":
                result_text += f"{color}{word}\033[0m "
            else:
                result_text += f"{word} "
    
    print(f"   {result_text}")
    print("\n   Legend:")
    print("   🔵 PERSON (Blue)")
    print("   🟢 ORGANIZATION (Green)")
    print("   🟡 LOCATION (Yellow)")

# Run training
train_ner_model()
```

---

# 2. FINE-TUNING BERT

## 2.1 What is Fine-Tuning?

**Analogy:** Fine-tuning is like taking a chef who's been trained in French cuisine and teaching them to make sushi. The chef already knows cooking basics (temperature control, knife skills, etc.) - you just need to teach the specifics of sushi.

```
PRE-TRAINING (General Knowledge):
═══════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────┐
│                                                           │
│  BERT pre-trained on:                                     │
│  • Wikipedia (2.5 billion words)                         │
│  • Books (800 million words)                             │
│  • News articles                                        │
│  • Web pages                                            │
│                                                           │
│  Learned:                                                │
│  • Language structure                                    │
│  • Word meanings                                        │
│  • Context understanding                                 │
│  • Grammar                                               │
│                                                           │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
FINE-TUNING (Task-Specific Knowledge):
═══════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────┐
│                                                           │
│  Fine-tuned on:                                           │
│  • Sentiment analysis dataset (10,000 reviews)           │
│                                                           │
│  Learned:                                                │
│  • Words that indicate positive/negative sentiment       │
│  • Specific patterns in reviews                          │
│  • Task-specific features                                │
│                                                           │
└─────────────────────────────────────────────────────────────┘
```

## 2.2 Fine-Tuning Process

```python
# ============ FINE-TUNING PROCESS ============

def fine_tuning_process():
    """
    Step-by-step fine-tuning explanation
    """
    
    print("="*70)
    print("BERT FINE-TUNING - COMPLETE PROCESS")
    print("="*70)
    
    print("""
    STEP 1: LOAD PRE-TRAINED BERT
    ──────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    
    • Downloads 110 million parameters
    • BERT base has 12 layers
    • 768 hidden size
    • 12 attention heads
    
    
    STEP 2: ADD TASK-SPECIFIC HEAD
    ───────────────────────────────
    Replace classifier head:
    • Original: MLM prediction (30,000 classes)
    • New: Sentiment analysis (2 classes)
    
    New head is randomly initialized
    Rest of BERT is pre-trained
    
    
    STEP 3: PREPARE DATA
    ─────────────────────
    • Tokenize text with BERT tokenizer
    • Add [CLS] and [SEP] tokens
    • Create attention masks
    • Create labels
    
    Input format:
    [CLS] I love this movie! [SEP]
    
    [CLS] = Classification token
    [SEP] = Separator token
    
    
    STEP 4: TRAINING
    ─────────────────
    • Use smaller learning rate (2e-5 to 5e-5)
    • Use AdamW optimizer
    • Use cross-entropy loss
    • Train for 2-4 epochs
    
    Why small learning rate?
    • BERT already knows language
    • Don't want to "forget" pre-training
    • Only need to adjust slightly
    
    
    STEP 5: EVALUATION
    ───────────────────
    • Test on validation set
    • Calculate accuracy, F1-score
    • Save best model
    
    
    VISUAL PROCESS:
    ─────────────────────────────────────────────────────────────
    
    Pre-trained BERT
         │
         ▼
    ┌─────────────────┐
    │  12 Layers      │  ← Frozen (not updated)
    │  768 Hidden     │
    │  12 Heads       │
    └─────────────────┘
         │
         ▼
    ┌─────────────────┐
    │  Classifier     │  ← New (trainable)
    │  Head           │
    └─────────────────┘
         │
         ▼
    Output (Positive/Negative)
    """)

fine_tuning_process()
```

## 2.3 Complete BERT Fine-Tuning Code

```python
# ============ BERT FINE-TUNING IMPLEMENTATION ============

def bert_finetuning_demo():
    """
    Complete BERT fine-tuning for sentiment analysis
    """
    
    print("="*70)
    print("BERT FINE-TUNING - COMPLETE IMPLEMENTATION")
    print("="*70)
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
        from datasets import Dataset
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score
        
        # ============ STEP 1: CREATE DATA ============
        print("\n📌 [1/6] Creating Sample Data...")
        
        # Sample data
        texts = [
            "I absolutely loved this movie! Best film ever!",
            "This is the worst movie I've ever seen.",
            "Amazing performances and great story.",
            "Terrible acting, boring plot.",
            "I would definitely recommend this film.",
            "Waste of time and money.",
            "Incredible cinematography and direction.",
            "Disappointing and predictable.",
            "This film is a masterpiece.",
            "I hated every minute of this movie.",
            "Beautifully crafted and emotionally powerful.",
            "Boring and completely forgettable.",
            "The acting was superb.",
            "The dialogue was cringe-worthy.",
            "A must-watch for everyone.",
            "Skip this one, it's terrible."
        ]
        
        labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        
        # Create dataset
        dataset = Dataset.from_dict({
            'text': texts,
            'label': labels
        })
        
        train_size = int(0.8 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))
        
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Evaluation samples: {len(eval_dataset)}")
        
        # ============ STEP 2: LOAD TOKENIZER ============
        print("\n📌 [2/6] Loading Tokenizer...")
        
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"   Model: {model_name}")
        print(f"   Vocabulary size: {tokenizer.vocab_size}")
        
        # ============ STEP 3: TOKENIZE DATA ============
        print("\n📌 [3/6] Tokenizing Data...")
        
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=128
            )
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        eval_dataset = eval_dataset.map(tokenize_function, batched=True)
        
        print("   ✅ Tokenization complete!")
        
        # ============ STEP 4: LOAD MODEL ============
        print("\n📌 [4/6] Loading BERT Model...")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
        
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # ============ STEP 5: SETUP TRAINING ============
        print("\n📌 [5/6] Setting up Training...")
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {
                'accuracy': accuracy_score(labels, predictions),
                'f1': f1_score(labels, predictions, average='weighted')
            }
        
        training_args = TrainingArguments(
            output_dir="./bert_finetuned",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        
        # ============ STEP 6: TRAIN ============
        print("\n📌 [6/6] Training BERT...")
        print("   (This might take a few minutes...)")
        
        trainer.train()
        
        # ============ EVALUATE ============
        print("\n📊 Training Complete! Evaluating...")
        
        eval_results = trainer.evaluate()
        print(f"   Accuracy: {eval_results['eval_accuracy']:.4f}")
        print(f"   F1-Score: {eval_results['eval_f1']:.4f}")
        
        # ============ TEST ON NEW TEXT ============
        print("\n📝 Testing on New Examples:")
        print("-" * 40)
        
        test_texts = [
            "This movie is absolutely fantastic!",
            "What a waste of time.",
            "The performances were incredible.",
            "I regret watching this film."
        ]
        
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            outputs = model(**inputs)
            prediction = outputs.logits.argmax().item()
            sentiment = "😊 POSITIVE" if prediction == 1 else "😞 NEGATIVE"
            print(f"   {sentiment}: {text[:50]}...")
        
        print("\n✅ BERT Fine-Tuning Complete!")
        print("   Model saved to: ./bert_finetuned")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install transformers datasets scikit-learn")
    except Exception as e:
        print(f"❌ Error: {e}")

# Run fine-tuning
bert_finetuning_demo()
```

## 2.4 BERT Variants Comparison

```python
# ============ BERT VARIANTS ============

def bert_variants():
    """
    Compare different BERT models
    """
    
    print("="*70)
    print("BERT VARIANTS - COMPARISON")
    print("="*70)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  Model          │ Parameters │ Speed  │ Size  │ Best For                  │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │  BERT-Base      │ 110M       │ Base   │ 1.2GB │ General purpose           │
    │  BERT-Large     │ 340M       │ Slow   │ 3.6GB │ High accuracy             │
    │  DistilBERT     │ 66M        │ Fast   │ 700MB │ Speed > Accuracy          │
    │  ALBERT         │ 12M        │ Fast   │ 500MB │ Memory efficient          │
    │  RoBERTa        │ 125M       │ Medium │ 1.6GB │ Better than BERT          │
    │  ELECTRA        │ 110M       │ Medium │ 1.3GB │ Efficient pre-training    │
    │  TinyBERT       │ 4.4M       │ Very   │ 50MB  │ Mobile/Edge devices       │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    WHEN TO USE EACH:
    ─────────────────────────────────────────────────────────────────────────────
    
    🔵 BERT-Base (Default Choice)
    • Good balance of accuracy and speed
    • Works for most tasks
    • 12 layers, 768 hidden
    
    🔴 BERT-Large (Highest Accuracy)
    • 24 layers, 1024 hidden
    • Best performance, but slowest
    • Requires powerful GPU
    
    🟢 DistilBERT (Fastest)
    • 40% smaller, 60% faster
    • 97% of BERT's performance
    • Best for production
    
    🟡 ALBERT (Memory Efficient)
    • 18x fewer parameters
    • Good for limited memory
    • Slower training
    
    🔶 RoBERTa (Better BERT)
    • Improved training
    • Better performance
    • Newer than BERT
    
    🟣 TinyBERT (Edge Devices)
    • Very small
    • Good for mobile
    • Lower accuracy
    """)

bert_variants()
```

---

# 3. SEQUENCE-TO-SEQUENCE MODELS

## 3.1 What are Seq2Seq Models?

**Analogy:** Seq2Seq is like a translator. You input a sentence in one language, and it outputs the same meaning in another language.

```
INPUT SEQUENCE (Source)    →    OUTPUT SEQUENCE (Target)
═══════════════════════════════════════════════════════════════

"I love NLP"               →    "J'aime le NLP"

"The weather is nice today" →    "Il fait beau aujourd'hui"

"Translate this"           →    "Traduis ceci"
```

### Seq2Seq Architecture

```
Seq2Seq Model - Complete Architecture
═══════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────┐
│                    ENCODER                                  │
│  ┌──────────────┐                                         │
│  │    I         │  ← Word 1                                │
│  └──────┬───────┘                                         │
│         ▼                                                 │
│  ┌──────────────┐                                         │
│  │    love      │  ← Word 2                                │
│  └──────┬───────┘                                         │
│         ▼                                                 │
│  ┌──────────────┐                                         │
│  │    NLP       │  ← Word 3                                │
│  └──────┬───────┘                                         │
│         ▼                                                 │
│     Context Vector                                         │
│     (Final Hidden State)                                   │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    DECODER                                  │
│  ┌──────────────┐                                         │
│  │    J'aime    │  ← Output Word 1                         │
│  └──────┬───────┘                                         │
│         ▼                                                 │
│  ┌──────────────┐                                         │
│  │     le       │  ← Output Word 2                         │
│  └──────┬───────┘                                         │
│         ▼                                                 │
│  ┌──────────────┐                                         │
│  │     NLP      │  ← Output Word 3                         │
│  └──────────────┘                                         │
└─────────────────────────────────────────────────────────────┘
```

## 3.2 Implementing Seq2Seq

```python
# ============ SEQ2SEQ IMPLEMENTATION ============

import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    """
    Encoder part of Seq2Seq model
    Reads input sequence and creates context vector
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Project bidirectional to unidirectional
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        # LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        
        # hidden: (num_layers * 2, batch, hidden_size)
        # Combine bidirectional states
        batch_size = hidden.size(1)
        
        # Reshape to combine directions
        hidden = hidden.view(self.num_layers, 2, batch_size, -1)
        cell = cell.view(self.num_layers, 2, batch_size, -1)
        
        # Combine forward and backward
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=2)
        
        # Project back to hidden_size
        hidden = self.fc_hidden(hidden)
        cell = self.fc_cell(cell)
        
        return hidden, cell

class Decoder(nn.Module):
    """
    Decoder part of Seq2Seq model
    Generates output sequence from context
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2):
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def forward(self, x, hidden, cell):
        # x: (batch, 1)  (single token at a time)
        embedded = self.embedding(x)  # (batch, 1, embedding_dim)
        
        # LSTM
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        
        # Predict next token
        logits = self.fc(output.squeeze(1))  # (batch, vocab_size)
        
        return logits, hidden, cell

class Seq2Seq(nn.Module):
    """
    Complete Seq2Seq model with attention
    """
    
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.size(0)
        target_len = target.size(1)
        vocab_size = self.decoder.fc.out_features
        
        # Encoder
        hidden, cell = self.encoder(source)
        
        # Decoder
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
        
        # First input is <SOS> token
        decoder_input = target[:, 0].unsqueeze(1)  # (batch, 1)
        
        for t in range(1, target_len):
            # Decoder step
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t, :] = output
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)  # (batch, 1)
            
            if teacher_force:
                decoder_input = target[:, t].unsqueeze(1)
            else:
                decoder_input = top1
        
        return outputs

def create_seq2seq_example():
    """
    Demonstrate Seq2Seq model
    """
    
    print("="*70)
    print("SEQ2SEQ MODEL - COMPLETE IMPLEMENTATION")
    print("="*70)
    
    # ============ CONFIGURATION ============
    vocab_size = 1000
    embedding_dim = 50
    hidden_size = 64
    num_layers = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"📌 Device: {device}")
    
    # ============ CREATE MODEL ============
    print("\n📌 Creating Seq2Seq Model...")
    
    encoder = Encoder(vocab_size, embedding_dim, hidden_size, num_layers)
    decoder = Decoder(vocab_size, embedding_dim, hidden_size, num_layers)
    model = Seq2Seq(encoder, decoder, device)
    model = model.to(device)
    
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ============ CREATE DUMMY DATA ============
    print("\n📌 Creating Dummy Data...")
    
    batch_size = 4
    source_len = 10
    target_len = 8
    
    source = torch.randint(0, vocab_size, (batch_size, source_len)).to(device)
    target = torch.randint(0, vocab_size, (batch_size, target_len)).to(device)
    
    print(f"   Source shape: {source.shape}")
    print(f"   Target shape: {target.shape}")
    
    # ============ FORWARD PASS ============
    print("\n📌 Forward Pass...")
    
    with torch.no_grad():
        outputs = model(source, target, teacher_forcing_ratio=0.5)
    
    print(f"   Output shape: {outputs.shape}")
    print(f"   (batch_size={batch_size}, target_len={target_len}, vocab_size={vocab_size})")
    
    # ============ EXPLANATION ============
    print("\n📖 SEQ2SEQ EXPLANATION:")
    print("-" * 40)
    
    print("""
    🎯 HOW SEQ2SEQ WORKS:
    ─────────────────────
    
    1. ENCODER:
       • Reads input sequence (source)
       • Creates a context vector (hidden state)
       • Captures meaning of entire sequence
    
    2. CONTEXT VECTOR:
       • Summary of input sequence
       • Passed to decoder
       • Contains all information needed
    
    3. DECODER:
       • Generates output sequence (target)
       • Uses context vector as initial state
       • Generates one token at a time
    
    4. TEACHER FORCING:
       • During training, use actual target token
       • Helps model learn faster
       • 50% of the time in this example
    
    5. APPLICATIONS:
       • Machine Translation
       • Text Summarization
       • Chatbots
       • Question Answering
       • Image Captioning
    """)

create_seq2seq_example()
```

## 3.3 Seq2Seq with Attention

```python
# ============ ATTENTION MECHANISM ============

class Attention(nn.Module):
    """
    Attention mechanism for Seq2Seq
    """
    
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: (batch, hidden_size)
        # encoder_outputs: (batch, seq_len, hidden_size * 2)
        
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # Repeat hidden state for each encoder output
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_size)
        
        # Calculate attention scores
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # (batch, seq_len)
        
        # Apply softmax
        attention_weights = torch.softmax(attention, dim=1)  # (batch, seq_len)
        
        # Context vector = weighted sum of encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # (batch, 1, hidden_size*2)
        context = context.squeeze(1)  # (batch, hidden_size*2)
        
        return context, attention_weights

class DecoderWithAttention(nn.Module):
    """
    Decoder with attention mechanism
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2):
        super(DecoderWithAttention, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim + hidden_size * 2,  # Input + context
            hidden_size,
            num_layers,
            batch_first=True
        )
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size + hidden_size * 2, vocab_size)
        self.hidden_size = hidden_size
        
    def forward(self, x, hidden, cell, encoder_outputs):
        # x: (batch, 1)
        # hidden: (num_layers, batch, hidden_size)
        # encoder_outputs: (batch, seq_len, hidden_size * 2)
        
        # Embed input
        embedded = self.embedding(x)  # (batch, 1, embedding_dim)
        
        # Get last layer hidden state
        hidden_last = hidden[-1]  # (batch, hidden_size)
        
        # Attention
        context, attention_weights = self.attention(hidden_last, encoder_outputs)
        
        # Combine embedded input with context
        lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        
        # LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # Combine output with context
        output = output.squeeze(1)  # (batch, hidden_size)
        combined = torch.cat((output, context), dim=1)  # (batch, hidden_size * 3)
        
        # Predict next token
        logits = self.fc(combined)  # (batch, vocab_size)
        
        return logits, hidden, cell, attention_weights

def attention_explained():
    """
    Explain attention in Seq2Seq
    """
    
    print("="*70)
    print("ATTENTION IN SEQ2SEQ")
    print("="*70)
    
    print("""
    🤔 WHY ATTENTION?
    ─────────────────────────────────────────────────────────────
    
    Problem without attention:
    • Context vector is the same for all outputs
    • First word gets same context as last word
    • Long sentences lose important information
    
    Solution with attention:
    • Decoder "pays attention" to different parts
    • Each output word uses different context
    • Works better for long sequences
    
    
    HOW ATTENTION WORKS:
    ─────────────────────────────────────────────────────────────
    
    For each output step:
    
    1. Calculate attention scores:
       Score = Query × Keyᵀ
       • Query = Decoder's current state
       • Key = Encoder's hidden states
    
    2. Apply softmax:
       Weights = Softmax(Score)
       • Sum of weights = 1
       • Higher weights = More attention
    
    3. Calculate context:
       Context = Weights × Value
       • Value = Encoder's hidden states
       • Weighted sum of all encoder states
    
    4. Use context:
       • Combined with decoder input
       • Better predictions!
    
    
    VISUAL EXAMPLE:
    ─────────────────────────────────────────────────────────────
    
    Translating: "I love NLP" → "J'aime le NLP"
    
    Output "J'aime":
    ┌─────────────────────────────────────────────────────────┐
    │ Pays attention to:                                     │
    │ • "I"    → 80%  (very important)                     │
    │ • "love" → 15%                                         │
    │ • "NLP"  → 5%                                          │
    └─────────────────────────────────────────────────────────┘
    
    Output "NLP":
    ┌─────────────────────────────────────────────────────────┐
    │ Pays attention to:                                     │
    │ • "I"    → 5%                                          │
    │ • "love" → 10%                                         │
    │ • "NLP"  → 85%  (very important)                     │
    └─────────────────────────────────────────────────────────┘
    """)

attention_explained()
```

---

# 4. TEXT SUMMARIZATION

## 4.1 What is Text Summarization?

**Analogy:** Text summarization is like writing a book summary - you read a long text and produce a shorter version that captures the main ideas.

```
LONG TEXT (Input):
═══════════════════════════════════════════════════════════════
"On Thursday, tech giant Apple announced its new product line, 
including the iPhone 15 with improved camera features. The 
company's CEO Tim Cook stated that the new device would 
revolutionize mobile photography. Additionally, Apple revealed 
the Apple Watch Series 9 with new health monitoring features. 
The event took place at Apple's headquarters in Cupertino, 
California, and was attended by thousands of people."

SUMMARY (Output):
═══════════════════════════════════════════════════════════════
"Apple announced iPhone 15 with improved cameras and Apple Watch 
Series 9 at their Cupertino headquarters on Thursday."
```

### Types of Summarization

```python
# ============ TYPES OF SUMMARIZATION ============

def summarization_types():
    """
    Different types of text summarization
    """
    
    print("="*70)
    print("TYPES OF TEXT SUMMARIZATION")
    print("="*70)
    
    print("""
    📌 1. EXTRACTIVE SUMMARIZATION
    ───────────────────────────────
    Selects important sentences from the original text
    
    Example:
    Original: "The cat sat on the mat. It was a comfortable mat. 
               The cat slept peacefully."
    Summary:  "The cat sat on the mat. The cat slept peacefully."
    
    ✅ Preserves original wording
    ✅ Grammatically correct
    ❌ Can be repetitive
    ❌ May miss connections
    
    
    📌 2. ABSTRACTIVE SUMMARIZATION
    ───────────────────────────────
    Generates new sentences that capture the meaning
    
    Example:
    Original: "The cat sat on the mat. It was a comfortable mat. 
               The cat slept peacefully."
    Summary:  "A cat slept peacefully on a comfortable mat."
    
    ✅ More natural and concise
    ✅ Captures overall meaning
    ❌ Can be grammatically incorrect
    ❌ May introduce errors
    
    
    📌 3. HYBRID SUMMARIZATION
    ──────────────────────────
    Combines extractive and abstractive approaches
    
    Uses:
    • Extract key sentences
    • Paraphrase and combine
    • Best of both worlds!
    
    
    TYPES BY LENGTH:
    ─────────────────────────────────────────────────────────────
    
    • Micro-summary: 1-2 sentences (tweet length)
    • Short summary: 1 paragraph
    • Medium summary: 2-3 paragraphs
    • Long summary: 10% of original length
    
    
    TYPES BY PURPOSE:
    ─────────────────────────────────────────────────────────────
    
    • News summarization: Key events
    • Document summarization: Main points
    • Meeting summarization: Decisions and actions
    • Medical summarization: Symptoms and diagnosis
    • Product summarization: Key features
    """)

summarization_types()
```

## 4.2 Implementing Summarization

```python
# ============ SUMMARIZATION IMPLEMENTATION ============

def summarization_with_transformers():
    """
    Text summarization using HuggingFace transformers
    """
    
    print("="*70)
    print("TEXT SUMMARIZATION - COMPLETE IMPLEMENTATION")
    print("="*70)
    
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        
        # ============ LOAD MODEL ============
        print("\n📌 Loading Summarization Model...")
        print("   (This may take a moment...)")
        
        model_name = "facebook/bart-large-cnn"
        
        summarizer = pipeline(
            "summarization",
            model=model_name,
            device=-1  # Use CPU
        )
        
        print(f"   Model: {model_name}")
        
        # ============ SAMPLE TEXT ============
        print("\n📝 Sample Text:")
        print("-" * 40)
        
        text = """
        Apple Inc. unveiled its highly anticipated iPhone 15 on Wednesday, 
        featuring a new titanium design and an advanced camera system. 
        The company's CEO Tim Cook described the device as "the best 
        iPhone we have ever created." The iPhone 15 Pro models include 
        a new action button that replaces the traditional mute switch, 
        allowing users to customize its function. The camera system 
        has been significantly upgraded, with a 48-megapixel main 
        sensor that enables unprecedented image quality. Apple also 
        announced the Apple Watch Series 9, featuring a new S9 chip 
        that enables on-device processing for Siri requests. Both 
        products will be available for preorder starting Friday and 
        will ship the following week. The event, held at Apple Park 
        in Cupertino, California, also showcased new sustainability 
        initiatives, including Apple's goal to become carbon neutral 
        by 2030. Analysts predict strong demand for the new products, 
        with iPhone sales expected to boost Apple's revenue in the 
        coming quarter.
        """
        
        print(text[:200] + "...")
        
        # ============ GENERATE SUMMARY ============
        print("\n📌 Generating Summary...")
        print("-" * 40)
        
        summary = summarizer(
            text,
            max_length=150,
            min_length=50,
            do_sample=False
        )
        
        print(f"Summary:\n{summary[0]['summary_text']}")
        
        # ============ DETAILED ANALYSIS ============
        print("\n📊 Summary Analysis:")
        print("-" * 40)
        
        original_length = len(text.split())
        summary_length = len(summary[0]['summary_text'].split())
        reduction_ratio = (1 - summary_length / original_length) * 100
        
        print(f"   Original length: {original_length} words")
        print(f"   Summary length: {summary_length} words")
        print(f"   Reduction ratio: {reduction_ratio:.1f}%")
        
        # ============ DIFFERENT LENGTHS ============
        print("\n📌 Different Summary Lengths:")
        print("-" * 40)
        
        lengths = [30, 80, 150]
        
        for max_len in lengths:
            summary = summarizer(
                text,
                max_length=max_len,
                min_length=max_len // 3,
                do_sample=False
            )
            print(f"\n   {max_len} words max:")
            print(f"   {summary[0]['summary_text']}")
        
    except ImportError:
        print("❌ Transformers not installed. Install with: pip install transformers")
    except Exception as e:
        print(f"❌ Error: {e}")

# Run summarization
summarization_with_transformers()
```

## 4.3 Custom Summarization Model

```python
# ============ CUSTOM SUMMARIZATION MODEL ============

def custom_summarizer():
    """
    Build a simple extractive summarizer from scratch
    """
    
    print("="*70)
    print("CUSTOM EXTRACTIVE SUMMARIZER")
    print("="*70)
    
    import re
    from collections import Counter
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    class SimpleSummarizer:
        """
        Simple extractive summarizer using TF-IDF
        """
        
        def __init__(self, num_sentences=3):
            self.num_sentences = num_sentences
            self.vectorizer = None
        
        def preprocess(self, text):
            """Clean text and split into sentences"""
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            # Split into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            return sentences
        
        def summarize(self, text):
            """Generate extractive summary"""
            
            # Step 1: Split into sentences
            sentences = self.preprocess(text)
            
            if len(sentences) <= self.num_sentences:
                return sentences
            
            # Step 2: Create TF-IDF vectors
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=100
            )
            
            # Calculate TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            
            # Step 3: Calculate sentence scores
            # Average TF-IDF score per sentence
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            
            # Step 4: Select top sentences
            top_indices = sentence_scores.argsort()[-self.num_sentences:][::-1]
            top_indices = sorted(top_indices)  # Keep original order
            
            # Step 5: Build summary
            summary = ' '.join([sentences[i] for i in top_indices])
            
            return summary
    
    # ============ DEMONSTRATE ============
    print("\n📌 Creating Custom Summarizer...")
    
    summarizer = SimpleSummarizer(num_sentences=4)
    
    text = """
    Machine learning is a branch of artificial intelligence that enables 
    systems to learn and improve from experience without being explicitly 
    programmed. It is a method of data analysis that automates analytical 
    model building. The process of machine learning involves feeding 
    data to algorithms, which learn from the data to make predictions 
    or decisions. There are three main types of machine learning: 
    supervised learning, unsupervised learning, and reinforcement 
    learning. Supervised learning uses labeled data to train models, 
    while unsupervised learning finds patterns in unlabeled data. 
    Reinforcement learning involves agents learning through trial 
    and error to maximize rewards. Deep learning, a subset of machine 
    learning, uses neural networks with many layers to learn complex 
    patterns in data. This technology has revolutionized fields such 
    as computer vision, natural language processing, and speech 
    recognition. Despite its success, machine learning faces challenges 
    including data privacy concerns, algorithmic bias, and interpretability 
    issues. However, researchers continue to develop new methods to 
    address these challenges and improve the technology.
    """
    
    print(f"\n📝 Original Text ({len(text.split())} words):")
    print("-" * 40)
    print(text[:200] + "...")
    
    # Generate summary
    summary = summarizer.summarize(text)
    
    print(f"\n📊 Summary ({len(summary.split())} words):")
    print("-" * 40)
    print(summary)
    
    # ============ COMPARE WITH BERT ============
    print("\n📌 Comparison with BART:")
    print("-" * 40)
    
    try:
        from transformers import pipeline
        
        bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        bart_summary = bart_summarizer(text, max_length=50, min_length=20)[0]['summary_text']
        
        print(f"\n   BART Summary:")
        print(f"   {bart_summary}")
        
        print("\n   Differences:")
        print("   • Extractive (ours): Uses original sentences")
        print("   • Abstractive (BART): Generates new sentences")
        print("   • Our summary is factually accurate but less fluent")
        print("   • BART summary is more natural but may introduce errors")
        
    except Exception as e:
        print(f"   BART comparison not available: {e}")

# Run custom summarizer
custom_summarizer()
```

---

# 5. EVALUATION METRICS FOR NLP

## 5.1 Common Metrics Overview

```python
# ============ NLP EVALUATION METRICS ============

def nlp_metrics():
    """
    All important NLP evaluation metrics explained
    """
    
    print("="*70)
    print("NLP EVALUATION METRICS - COMPLETE GUIDE")
    print("="*70)
    
    print("""
    📊 CLASSIFICATION METRICS
    ─────────────────────────────────────────────────────────────
    
    1. ACCURACY
    ────────────
    Accuracy = (True Positives + True Negatives) / Total
    
    ✅ Simple and intuitive
    ❌ Not good for imbalanced data
    
    
    2. PRECISION
    ─────────────
    Precision = True Positives / (True Positives + False Positives)
    
    "When we predict positive, how often are we right?"
    ✅ Useful when false positives are costly
    
    
    3. RECALL (SENSITIVITY)
    ────────────────────────
    Recall = True Positives / (True Positives + False Negatives)
    
    "Of all actual positives, how many did we find?"
    ✅ Useful when false negatives are costly
    
    
    4. F1-SCORE
    ────────────
    F1 = 2 × (Precision × Recall) / (Precision + Recall)
    
    ✅ Balance of precision and recall
    ✅ Best single metric for imbalanced data
    
    
    5. CONFUSION MATRIX
    ────────────────────
    Shows all predictions:
    
                 Predicted
                 Pos   Neg
    Actual Pos   TP    FN
    Actual Neg   FP    TN
    
    
    📊 GENERATION METRICS
    ─────────────────────────────────────────────────────────────
    
    6. BLEU SCORE
    ──────────────
    Measures similarity to reference text
    
    BLEU = min(1, length_penalty) × exp(∑ log precision)
    
    ✅ Used in translation
    ❌ Only considers exact matches
    
    
    7. ROUGE SCORE
    ───────────────
    Measures overlap between generated and reference
    
    ROUGE-N: N-gram overlap
    ROUGE-L: Longest common subsequence
    ROUGE-S: Skip-gram overlap
    
    ✅ Used in summarization
    ✅ Multiple variants for different needs
    ❌ Can be misleading for abstractive summaries
    
    
    8. PERPLEXITY
    ──────────────
    Measures how "surprised" model is by test data
    
    Perplexity = 2^(-log likelihood)
    
    ✅ Lower is better
    ✅ Used in language modeling
    ❌ Not intuitive for non-experts
    
    
    9. METEOR
    ──────────
    Considers synonyms and stems
    
    METEOR = F1 × (1 - penalty)
    
    ✅ Better than BLEU
    ✅ Considers meaning
    ❌ More complex to calculate
    
    
    📊 SUMMARIZATION METRICS
    ─────────────────────────────────────────────────────────────
    
    10. COMPRESSION RATIO
    ──────────────────────
    Ratio = Summary Length / Original Length
    
    ✅ Simple to calculate
    ✅ Shows reduction
    ❌ Doesn't measure quality
    
    
    11. INFORMATIVENESS
    ────────────────────
    How much information is preserved
    
    ✅ Measures content preservation
    ❌ Hard to quantify automatically
    """)

nlp_metrics()
```

## 5.2 Implementing Metrics

```python
# ============ METRICS IMPLEMENTATION ============

def metrics_implementation():
    """
    Implementing common NLP metrics
    """
    
    print("="*70)
    print("METRICS IMPLEMENTATION - WITH CODE")
    print("="*70)
    
    try:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, confusion_matrix, classification_report
        )
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import numpy as np
        
        # ============ CLASSIFICATION METRICS ============
        print("\n📌 CLASSIFICATION METRICS")
        print("-" * 40)
        
        # Sample predictions
        y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
        
        print(f"True labels: {y_true}")
        print(f"Predictions: {y_pred}")
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"\nAccuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1-Score:  {f1:.3f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  [[{cm[0,0]:>2}  {cm[0,1]:>2}]")
        print(f"   [{cm[1,0]:>2}  {cm[1,1]:>2}]]")
        
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        # ============ ROUGE SCORES ============
        print("\n📌 ROUGE SCORES (Summarization)")
        print("-" * 40)
        
        reference = "The cat sat on the mat and slept peacefully."
        hypothesis = "A cat slept peacefully on a comfortable mat."
        
        print(f"Reference: {reference}")
        print(f"Hypothesis: {hypothesis}")
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        
        for metric, score in scores.items():
            print(f"\n{metric.upper()}:")
            print(f"  Precision: {score.precision:.3f}")
            print(f"  Recall: {score.recall:.3f}")
            print(f"  F1: {score.fmeasure:.3f}")
        
        # ============ BLEU SCORE ============
        print("\n📌 BLEU SCORE (Translation)")
        print("-" * 40)
        
        reference = ["the cat sat on the mat"]
        hypothesis = ["the cat slept on the mat"]
        
        print(f"Reference: {reference[0]}")
        print(f"Hypothesis: {hypothesis[0]}")
        
        smoothing = SmoothingFunction()
        bleu = sentence_bleu(
            [reference[0].split()],
            hypothesis[0].split(),
            smoothing_function=smoothing.method1
        )
        
        print(f"\nBLEU Score: {bleu:.3f}")
        
        print("\n💡 INTERPRETATION:")
        print("   • ROUGE: Measures overlap (good for summarization)")
        print("   • BLEU: Measures similarity (good for translation)")
        print("   • Higher scores = Better quality")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install rouge-score nltk")
        print("   Then: python -c 'import nltk; nltk.download(\"punkt\")'")

# Run metrics demonstration
metrics_implementation()
```

---

# 6. ADVANCED TRANSFORMER CONCEPTS

## 6.1 Transformer Architecture Deep Dive

```python
# ============ TRANSFORMER DEEP DIVE ============

def transformer_deep_dive():
    """
    Deep dive into transformer architecture
    """
    
    print("="*70)
    print("TRANSFORMER ARCHITECTURE - DEEP DIVE")
    print("="*70)
    
    print("""
    🏗️ ENCODER BLOCK (Detailed)
    ─────────────────────────────────────────────────────────────
    
    Input: (batch, seq_len, hidden_size)
    
    1. SELF-ATTENTION LAYER
    ────────────────────────
    Q = X × W_Q
    K = X × W_K
    V = X × W_V
    
    Attention = softmax(Q × Kᵀ / √d) × V
    
    • Multiple heads (8-16)
    • Each head learns different patterns
    • Output: (batch, seq_len, hidden_size)
    
    
    2. ADD & NORMALIZE
    ──────────────────
    X = LayerNorm(X + Attention)
    
    • Residual connection (add input)
    • Layer normalization (stable training)
    • Output: (batch, seq_len, hidden_size)
    
    
    3. FEED FORWARD NETWORK
    ───────────────────────
    FFN(x) = GELU(x × W₁ + b₁) × W₂ + b₂
    
    • Two linear layers
    • Hidden size × 4
    • Applied to each token independently
    • Output: (batch, seq_len, hidden_size)
    
    
    4. ADD & NORMALIZE
    ──────────────────
    X = LayerNorm(X + FFN)
    
    • Another residual connection
    • Output: (batch, seq_len, hidden_size)
    
    
    🏗️ DECODER BLOCK (Detailed)
    ─────────────────────────────────────────────────────────────
    
    Input: (batch, seq_len, hidden_size)
    
    1. MASKED SELF-ATTENTION
    ────────────────────────
    • Same as encoder self-attention
    • With causal masking (can't see future)
    • Output: (batch, seq_len, hidden_size)
    
    
    2. ADD & NORMALIZE
    ──────────────────
    X = LayerNorm(X + Masked_Attention)
    
    
    3. CROSS-ATTENTION
    ──────────────────
    Q = X × W_Q  (from decoder)
    K = Encoder × W_K  (from encoder)
    V = Encoder × W_V  (from encoder)
    
    • Uses encoder outputs as Key and Value
    • Query from decoder
    • Output: (batch, seq_len, hidden_size)
    
    
    4. ADD & NORMALIZE
    ──────────────────
    X = LayerNorm(X + Cross_Attention)
    
    
    5. FEED FORWARD NETWORK
    ───────────────────────
    Same as encoder
    Output: (batch, seq_len, hidden_size)
    
    
    6. ADD & NORMALIZE
    ──────────────────
    X = LayerNorm(X + FFN)
    Output: (batch, seq_len, hidden_size)
    """)

transformer_deep_dive()
```

## 6.2 Positional Encoding

```python
# ============ POSITIONAL ENCODING ============

def positional_encoding_explained():
    """
    Understanding positional encoding
    """
    
    print("="*70)
    print("POSITIONAL ENCODING - HOW IT WORKS")
    print("="*70)
    
    print("""
    PROBLEM:
    ────────
    Transformers process all tokens simultaneously
    Without position information:
    "I love you" = "You love I" (same representation!)
    
    SOLUTION: POSITIONAL ENCODING
    ─────────────────────────────
    Add position information to word embeddings
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    • pos = position in sequence
    • i = dimension index
    • d_model = embedding dimension
    
    
    WHY THIS FORMULA?
    ──────────────────
    1. Different frequencies for each dimension
    2. Allows model to learn relative positions
    3. Can extrapolate to unseen lengths
    
    
    VISUAL EXAMPLE:
    ─────────────────────────────────────────────────────────────
    
    Position 0:  [0.00, 1.00, 0.00, 1.00, ...]
    Position 1:  [0.84, 0.54, 0.01, 0.99, ...]
    Position 2:  [0.91, -0.42, 0.02, 0.99, ...]
    Position 3:  [0.14, -0.99, 0.03, 0.99, ...]
    
    Each position has a unique pattern!
    
    
    LEARNED POSITIONAL ENCODING (Alternative):
    ──────────────────────────────────────────
    • Learn position embeddings like word embeddings
    • Can handle up to max_position length
    • Used in BERT, GPT
    • More flexible but less extrapolation
    
    
    COMPARISON:
    ─────────────────────────────────────────────────────────────
    
    Sinusoidal Positional Encoding:
    ✅ No parameters to learn
    ✅ Can handle any sequence length
    ❌ Fixed function
    
    Learned Positional Encoding:
    ✅ Can adapt to data
    ✅ Better for fixed lengths
    ❌ Needs parameters
    ❌ Limited to max_length
    """)

positional_encoding_explained()
```

## 6.3 Attention Visualization

```python
# ============ ATTENTION VISUALIZATION ============

def attention_visualization():
    """
    Visualizing attention patterns
    """
    
    print("="*70)
    print("ATTENTION PATTERNS - WHAT MODELS LEARN")
    print("="*70)
    
    print("""
    🔍 ATTENTION PATTERN EXAMPLES:
    ─────────────────────────────────────────────────────────────
    
    1. SYNTACTIC ATTENTION
    ──────────────────────
    Pays attention to grammatical relationships
    
    "The cat sat on the mat"
    
    "cat" attends to:
    • "The" (determiner)
    • "sat" (verb)
    • "mat" (prepositional object)
    
    
    2. SEMANTIC ATTENTION
    ─────────────────────
    Pays attention to meaning-related words
    
    "The dog is barking loudly"
    
    "barking" attends to:
    • "dog" (who is barking)
    • "loudly" (how it's barking)
    
    
    3. CONTEXT ATTENTION
    ────────────────────
    Looks at surrounding context
    
    "I went to the bank to deposit money"
    
    "bank" attends to:
    • "deposit" (context clue)
    • "money" (context clue)
    • → Financial bank (not river bank!)
    
    
    4. COREFERENCE ATTENTION
    ────────────────────────
    Links pronouns to nouns
    
    "Alice said she would come"
    
    "she" attends to "Alice"
    
    
    5. POSITIONAL ATTENTION
    ───────────────────────
    Pays attention to nearby words
    
    Usually focuses on:
    • Adjacent words (high attention)
    • Far words (low attention)
    
    VISUAL HEATMAP:
    ─────────────────────────────────────────────────────────────
    
                     The   cat   sat   on    the   mat
          The       0.1   0.2   0.1   0.2   0.3   0.1
          cat       0.1   0.4   0.3   0.1   0.0   0.1
          sat       0.1   0.2   0.2   0.2   0.1   0.2
          on        0.0   0.1   0.2   0.1   0.3   0.3
          the       0.1   0.1   0.1   0.1   0.4   0.2
          mat       0.0   0.1   0.1   0.1   0.2   0.5
    
    Each row shows what a word pays attention to
    Darker = More attention
    """)

attention_visualization()
```

---

# 7. TRANSFER LEARNING IN NLP

## 7.1 What is Transfer Learning?

```python
# ============ TRANSFER LEARNING EXPLAINED ============

def transfer_learning_explained():
    """
    Understanding transfer learning in NLP
    """
    
    print("="*70)
    print("TRANSFER LEARNING IN NLP")
    print("="*70)
    
    print("""
    🎯 WHAT IS TRANSFER LEARNING?
    ─────────────────────────────────────────────────────────────
    
    Traditional ML: Train model from scratch for each task
    Transfer Learning: Use knowledge from one task for another
    
    ANALOGY:
    ────────
    Learning French → Easier to learn Spanish
    (Share similar grammar and vocabulary)
    
    NLP Version:
    Language Understanding → Easier to do any NLP task
    (Share understanding of grammar, semantics, etc.)
    
    
    🏗️ TYPES OF TRANSFER LEARNING
    ─────────────────────────────────────────────────────────────
    
    1. FEATURE EXTRACTION (Frozen)
    ──────────────────────────────
    • Use pre-trained model as feature extractor
    • Freeze all layers
    • Only train new classifier head
    
    ✅ Fast training
    ✅ No risk of overfitting
    ❌ Less task-specific
    
    
    2. FINE-TUNING (Unfrozen)
    ──────────────────────────
    • Update all layers
    • Small learning rate
    • Train on task-specific data
    
    ✅ Better performance
    ✅ More task-specific
    ❌ Risk of overfitting
    ❌ Slower training
    
    
    3. PROMPT-BASED LEARNING
    ─────────────────────────
    • Convert task to text completion
    • Use pre-trained language model
    • No training needed (in-context learning)
    
    ✅ No training needed
    ✅ Flexible
    ❌ Less reliable
    ❌ Requires large model
    
    
    📊 PRE-TRAINING DATA SOURCES
    ─────────────────────────────────────────────────────────────
    
    For Language Understanding:
    • Wikipedia (2.5B words)
    • Books (800M words)
    • News articles
    • Web pages (CommonCrawl)
    
    For Specialized Models:
    • Medical: PubMed articles
    • Legal: Court documents
    • Scientific: Research papers
    
    
    🚀 POPULAR PRE-TRAINED MODELS
    ─────────────────────────────────────────────────────────────
    
    1. BERT: Bidirectional understanding
    2. GPT: Text generation
    3. RoBERTa: Improved BERT
    4. T5: Text-to-text framework
    5. BART: Denoising autoencoder
    6. ELECTRA: Efficient pre-training
    7. ALBERT: Lightweight BERT
    """)

transfer_learning_explained()
```

## 7.2 Transfer Learning Strategies

```python
# ============ TRANSFER LEARNING STRATEGIES ============

def transfer_learning_strategies():
    """
    Different strategies for transfer learning
    """
    
    print("="*70)
    print("TRANSFER LEARNING STRATEGIES")
    print("="*70)
    
    print("""
    📌 STRATEGY 1: ZERO-SHOT LEARNING
    ──────────────────────────────────
    Model works on tasks it wasn't trained on
    
    Example: "Classify this as positive or negative"
    ↓
    Model understands the instruction
    
    Used in: GPT-3, ChatGPT, T0
    
    
    📌 STRATEGY 2: ONE-SHOT LEARNING
    ──────────────────────────────────
    Model learns from a single example
    
    Example: "This is positive: 'Great movie!'
              Now classify: 'Terrible film.'"
    ↓
    Model learns from one example
    
    
    📌 STRATEGY 3: FEW-SHOT LEARNING
    ──────────────────────────────────
    Model learns from a few examples
    
    Example: "Positive: 'Great movie!'
              Positive: 'Amazing film!'
              Negative: 'Bad movie.'
              Now classify: 'Good film.'"
    ↓
    Model learns from 3 examples
    
    
    📌 STRATEGY 4: MULTI-TASK LEARNING
    ────────────────────────────────────
    Model learns multiple tasks at once
    
    Tasks:
    • Sentiment Analysis
    • Named Entity Recognition
    • Question Answering
    
    Benefits:
    ✅ Better generalization
    ✅ Shared representations
    ✅ More robust models
    
    
    📌 STRATEGY 5: DOMAIN ADAPTATION
    ──────────────────────────────────
    Adapt model to new domain
    
    Example: BERT (general) → BioBERT (medical)
    
    Steps:
    1. Pre-train on general data
    2. Continue pre-training on domain data
    3. Fine-tune on task
    
    
    📊 COMPARISON TABLE:
    ─────────────────────────────────────────────────────────────
    
    Strategy    │ Data Needed │ Training │ Performance
    ────────────┼─────────────┼──────────┼─────────────
    Zero-shot   │ 0           │ None     │ Medium
    One-shot    │ 1 example   │ None     │ Medium-High
    Few-shot    │ 5-10 examples│ None    │ High
    Fine-tuning │ 100-10K     │ Hours    │ Very High
    Pre-training│ Millions    │ Weeks    │ Best
    
    
    💡 WHEN TO USE EACH:
    ─────────────────────────────────────────────────────────────
    
    • Zero-shot: Quick experiments, no data
    • Few-shot: Limited data, no training
    • Fine-tuning: Good data, best performance
    • Pre-training: No pre-trained model available
    """)

transfer_learning_strategies()
```

---

# 8. COMPLETE WORKING CODE - NER & SUMMARIZATION SYSTEM

## 8.1 Project Overview

```python
# ============================================
# COMPLETE NER & SUMMARIZATION SYSTEM
# EVERY LINE EXPLAINED FOR BEGINNERS
# ============================================

def project_overview_week5():
    """
    Overview of the NER and Summarization project
    """
    
    print("="*70)
    print("NER & SUMMARIZATION SYSTEM - PROJECT OVERVIEW")
    print("="*70)
    
    print("""
    🎯 PROJECT GOALS:
    ─────────────────
    1. Named Entity Recognition (NER)
       • Identify and classify named entities in text
       • Person, Organization, Location, Date, etc.
    
    2. Text Summarization
       • Generate concise summaries of long texts
       • Both extractive and abstractive approaches
    
    
    📊 DATA:
    ────────
    For NER:
    • CoNLL-2003 dataset
    • 14,000+ sentences
    • 4 entity types: PER, ORG, LOC, MISC
    
    For Summarization:
    • CNN/DailyMail dataset
    • 300,000+ news articles
    • Article-summary pairs
    
    
    🏗️ MODELS WE'LL BUILD:
    ─────────────────────
    1. NER: BiLSTM-CRF model
    2. NER: BERT fine-tuned for NER
    3. Summarization: BART-based
    4. Summarization: Custom extractive
    
    
    📁 PROJECT STRUCTURE:
    ────────────────────
    nlp_advanced/
    ├── ner/
    │   ├── models/
    │   │   ├── bilstm_crf.py
    │   │   └── bert_ner.py
    │   ├── data/
    │   │   ├── conll2003/
    │   │   └── preprocessed/
    │   └── utils/
    │       ├── tokenizer.py
    │       └── evaluator.py
    ├── summarization/
    │   ├── models/
    │   │   ├── bart_summarizer.py
    │   │   └── extractive.py
    │   ├── data/
    │   │   └── cnn_dailymail/
    │   └── utils/
    │       ├── preprocess.py
    │       └── metrics.py
    └── main.py
    """)

project_overview_week5()
```

## 8.2 Complete Implementation

```python
# ============================================
# NER & SUMMARIZATION - COMPLETE IMPLEMENTATION
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("NER & SUMMARIZATION SYSTEM - COMPLETE IMPLEMENTATION")
print("="*70)

# ============================================
# PART 1: NAMED ENTITY RECOGNITION
# ============================================

class NERConfig:
    # Data
    MAX_SEQ_LEN = 128
    BATCH_SIZE = 16
    
    # Model
    EMBEDDING_DIM = 100
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    
    # Training
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NERDataset(Dataset):
    """
    Dataset for NER
    """
    
    def __init__(self, texts, labels, word_to_idx, label_to_idx, max_len=128):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.label_to_idx = label_to_idx
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        words = self.texts[idx]
        labels = self.labels[idx]
        
        # Convert to IDs
        word_ids = [self.word_to_idx.get(w, 1) for w in words]  # 1 = UNK
        label_ids = [self.label_to_idx.get(l, 0) for l in labels]  # 0 = O
        
        # Pad
        word_ids = word_ids[:self.max_len] + [0] * (self.max_len - len(word_ids))
        label_ids = label_ids[:self.max_len] + [0] * (self.max_len - len(label_ids))
        
        return (torch.tensor(word_ids, dtype=torch.long),
                torch.tensor(label_ids, dtype=torch.long))

class BiLSTM_CRF(nn.Module):
    """
    BiLSTM-CRF for NER
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_labels):
        super(BiLSTM_CRF, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size * 2, num_labels)
        self.dropout = nn.Dropout(0.3)
        
        # CRF parameters (simplified)
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))
        self.num_labels = num_labels
    
    def forward(self, x):
        embeddings = self.embedding(x)
        embeddings = self.dropout(embeddings)
        
        lstm_out, _ = self.lstm(embeddings)
        logits = self.fc(lstm_out)
        
        return logits

def create_ner_data():
    """
    Create sample NER data
    """
    
    print("\n📊 Creating NER Dataset...")
    
    # Sample sentences with labels (simplified)
    sentences = [
        ["Elon", "Musk", "founded", "SpaceX", "in", "California"],
        ["Google", "is", "headquartered", "in", "Mountain", "View"],
        ["Steve", "Jobs", "was", "the", "CEO", "of", "Apple"],
        ["Microsoft", "was", "founded", "by", "Bill", "Gates"],
        ["Amazon", "has", "its", "HQ", "in", "Seattle"],
        ["Tim", "Cook", "is", "the", "CEO", "of", "Apple"],
        ["Tesla", "is", "based", "in", "Austin", "Texas"],
        ["OpenAI", "was", "founded", "by", "Sam", "Altman"],
        ["NASA", "launched", "a", "rocket", "from", "Florida"],
        ["Satya", "Nadella", "runs", "Microsoft"]
    ]
    
    # BIO labels
    all_labels = [
        ["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC"],
        ["B-ORG", "O", "O", "O", "B-LOC", "I-LOC"],
        ["B-PER", "I-PER", "O", "O", "O", "O", "B-ORG"],
        ["B-ORG", "O", "O", "O", "B-PER", "I-PER"],
        ["B-ORG", "O", "O", "O", "O", "B-LOC"],
        ["B-PER", "I-PER", "O", "O", "O", "O", "B-ORG"],
        ["B-ORG", "O", "O", "O", "B-LOC", "I-LOC"],
        ["B-ORG", "O", "O", "O", "B-PER", "I-PER"],
        ["B-ORG", "O", "O", "O", "O", "B-LOC"],
        ["B-PER", "I-PER", "O", "B-ORG"]
    ]
    
    # Build vocabulary
    word_to_idx = {"<PAD>": 0, "<UNK>": 1}
    for sent in sentences:
        for word in sent:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
    
    # Build label vocabulary
    label_to_idx = {"O": 0}
    for label_seq in all_labels:
        for label in label_seq:
            if label not in label_to_idx:
                label_to_idx[label] = len(label_to_idx)
    
    print(f"   Vocabulary size: {len(word_to_idx)}")
    print(f"   Label types: {len(label_to_idx)}")
    
    return sentences, all_labels, word_to_idx, label_to_idx

def train_ner():
    """
    Train NER model
    """
    
    print("\n" + "="*70)
    print("PART 1: NAMED ENTITY RECOGNITION")
    print("="*70)
    
    # Create data
    sentences, labels, word_to_idx, label_to_idx = create_ner_data()
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    # Create dataset
    dataset = NERDataset(sentences, labels, word_to_idx, label_to_idx, NERConfig.MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=NERConfig.BATCH_SIZE, shuffle=True)
    
    # Create model
    model = BiLSTM_CRF(
        len(word_to_idx),
        NERConfig.EMBEDDING_DIM,
        NERConfig.HIDDEN_SIZE,
        NERConfig.NUM_LAYERS,
        len(label_to_idx)
    ).to(NERConfig.DEVICE)
    
    print(f"\n📌 Model: BiLSTM-CRF")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=NERConfig.LEARNING_RATE)
    
    print(f"\n📌 Training for {NERConfig.EPOCHS} epochs...")
    print("-" * 40)
    
    for epoch in range(NERConfig.EPOCHS):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(NERConfig.DEVICE), targets.to(NERConfig.DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, len(label_to_idx)), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{NERConfig.EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")
    
    # Test
    print("\n📌 Testing NER Model...")
    print("-" * 40)
    
    test_sentences = [
        ["Elon", "Musk", "is", "CEO", "of", "Tesla"],
        ["Microsoft", "is", "in", "Redmond", "Washington"]
    ]
    
    for sent in test_sentences:
        # Convert to IDs
        ids = [word_to_idx.get(w, 1) for w in sent]
        ids = ids + [0] * (NERConfig.MAX_SEQ_LEN - len(ids))
        ids_tensor = torch.tensor([ids], dtype=torch.long).to(NERConfig.DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(ids_tensor)
            preds = outputs.argmax(dim=-1)[0]
        
        print(f"\n   Sentence: {' '.join(sent)}")
        for word, pred in zip(sent, preds[:len(sent)]):
            label = idx_to_label[pred.item()]
            if label != "O":
                print(f"   • {word}: {label}")
    
    return model, idx_to_label

# ============================================
# PART 2: TEXT SUMMARIZATION
# ============================================

class SummarizationConfig:
    MAX_INPUT_LEN = 512
    MAX_SUMMARY_LEN = 150
    MIN_SUMMARY_LEN = 30

def create_summarization_data():
    """
    Create sample summarization data
    """
    
    print("\n📊 Creating Summarization Dataset...")
    
    # Sample article-summary pairs
    data = [
        {
            "article": """
            Apple Inc. unveiled its highly anticipated iPhone 15 on Wednesday, 
            featuring a new titanium design and an advanced camera system. 
            The company's CEO Tim Cook described the device as "the best 
            iPhone we have ever created." The iPhone 15 Pro models include 
            a new action button that replaces the traditional mute switch, 
            allowing users to customize its function. The camera system 
            has been significantly upgraded, with a 48-megapixel main 
            sensor that enables unprecedented image quality. Apple also 
            announced the Apple Watch Series 9, featuring a new S9 chip 
            that enables on-device processing for Siri requests.
            """,
            "summary": "Apple announced iPhone 15 with titanium design and improved camera, plus Apple Watch Series 9 with new S9 chip."
        },
        {
            "article": """
            Climate change continues to pose significant challenges worldwide, 
            with rising temperatures and extreme weather events becoming more 
            frequent. A new study from the United Nations reveals that global 
            emissions must be reduced by 45% by 2030 to avoid catastrophic 
            consequences. The report highlights the urgent need for countries 
            to transition to renewable energy sources and implement sustainable 
            practices. Several nations have already pledged to achieve net-zero 
            emissions by 2050, but experts warn that current efforts are 
            insufficient to meet the targets set by the Paris Agreement.
            """,
            "summary": "UN study warns global emissions must drop 45% by 2030 to avoid climate catastrophe, as current efforts fall short of Paris Agreement goals."
        }
    ]
    
    print(f"   Created {len(data)} examples")
    
    return data

def abstractive_summarization():
    """
    Use pre-trained model for abstractive summarization
    """
    
    print("\n📌 Abstractive Summarization (Using BART)")
    print("-" * 40)
    
    try:
        from transformers import pipeline
        
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        data = create_summarization_data()
        
        for i, item in enumerate(data, 1):
            article = item["article"].strip()
            reference = item["summary"]
            
            print(f"\n📝 Article {i}:")
            print(f"   {article[:150]}...")
            
            # Generate summary
            summary = summarizer(article, max_length=SummarizationConfig.MAX_SUMMARY_LEN)[0]['summary_text']
            
            print(f"\n📊 Generated Summary:")
            print(f"   {summary}")
            
            print(f"\n📝 Reference Summary:")
            print(f"   {reference}")
            
    except ImportError:
        print("   ❌ Transformers not installed")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def extractive_summarization():
    """
    Custom extractive summarization
    """
    
    print("\n📌 Extractive Summarization (Custom)")
    print("-" * 40)
    
    class ExtractiveSummarizer:
        """
        Simple extractive summarizer using sentence similarity
        """
        
        def __init__(self, num_sentences=3):
            self.num_sentences = num_sentences
        
        def summarize(self, text):
            # Split into sentences
            sentences = text.split('.')
            sentences = [s.strip() + '.' for s in sentences if len(s.strip()) > 20]
            
            if len(sentences) <= self.num_sentences:
                return ' '.join(sentences)
            
            # Simple scoring: longer sentences and those with important words
            important_words = set(['important', 'significant', 'major', 'key', 'new', 'first', 'latest'])
            scores = []
            
            for sent in sentences:
                score = 0
                # Length score
                score += len(sent.split()) / 100
                # Important words score
                words = set(sent.lower().split())
                score += len(words.intersection(important_words))
                scores.append(score)
            
            # Select top sentences
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:self.num_sentences]
            
            top_indices = sorted(top_indices)
            summary = ' '.join([sentences[i] for i in top_indices])
            
            return summary
    
    # Test
    summarizer = ExtractiveSummarizer(num_sentences=2)
    
    text = """
    Artificial intelligence has made remarkable progress in recent years. 
    Machine learning models can now perform tasks that were once thought impossible. 
    Deep learning has revolutionized computer vision and natural language processing. 
    However, challenges remain in areas such as interpretability and fairness. 
    Researchers are actively working on solving these problems.
    """
    
    summary = summarizer.summarize(text)
    
    print(f"\n📝 Original Text:")
    print(f"   {text}")
    
    print(f"\n📊 Extractive Summary:")
    print(f"   {summary}")

# ============================================
# PART 3: INTEGRATION & DEPLOYMENT
# ============================================

def integrated_system():
    """
    Complete integrated system
    """
    
    print("="*70)
    print("INTEGRATED NER & SUMMARIZATION SYSTEM")
    print("="*70)
    
    try:
        from transformers import pipeline
        
        print("\n📌 Loading Models...")
        
        # Load NER model
        ner_model = pipeline("ner", aggregation_strategy="simple")
        
        # Load Summarization model
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        print("   ✅ Models loaded successfully")
        
        # ============ SAMPLE TEXT ============
        text = """
        Apple Inc., led by CEO Tim Cook, announced its new iPhone 15 on Wednesday 
        at their headquarters in Cupertino, California. The company also unveiled 
        the Apple Watch Series 9 with new health features. This marks Apple's 
        first major product launch of 2024, with analysts expecting strong sales 
        in the holiday quarter. The iPhone 15 features a titanium frame and a 
        powerful A17 chip, while the Apple Watch introduces a new temperature 
        sensor and advanced health monitoring capabilities.
        """
        
        print(f"\n📝 Input Text:")
        print("-" * 40)
        print(text)
        
        # ============ NER ============
        print(f"\n🔍 Named Entity Recognition:")
        print("-" * 40)
        
        entities = ner_model(text)
        for entity in entities:
            print(f"   {entity['word']}: {entity['entity_group']} (Confidence: {entity['score']:.2f})")
        
        # ============ SUMMARIZATION ============
        print(f"\n📝 Summarization:")
        print("-" * 40)
        
        summary = summarizer(text, max_length=80, min_length=30)[0]['summary_text']
        print(f"   {summary}")
        
        # ============ STATISTICS ============
        print(f"\n📊 Statistics:")
        print("-" * 40)
        print(f"   Original length: {len(text.split())} words")
        print(f"   Summary length: {len(summary.split())} words")
        print(f"   Entities found: {len(entities)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# ============================================
# MAIN EXECUTION
# ============================================

def main_week5():
    """
    Main function for Week 5
    """
    
    print("\n" + "="*70)
    print("WEEK 5: ADVANCED NLP - COMPLETE SYSTEM")
    print("="*70)
    
    # Part 1: NER
    train_ner()
    
    # Part 2: Summarization
    print("\n" + "="*70)
    print("PART 2: TEXT SUMMARIZATION")
    print("="*70)
    
    abstractive_summarization()
    extractive_summarization()
    
    # Part 3: Integration
    integrated_system()
    
    print("\n" + "="*70)
    print("✅ WEEK 5 COMPLETE!")
    print("   NER & Summarization System Ready")
    print("="*70)

if __name__ == "__main__":
    main_week5()
```

---

# 9. COMMON ISSUES AND SOLUTIONS

## 9.1 NER-Specific Issues

```python
# ============ NER ISSUES ============

def ner_issues():
    """
    Common NER problems and solutions
    """
    
    print("="*70)
    print("NER COMMON ISSUES AND SOLUTIONS")
    print("="*70)
    
    issues = {
        "1. Entity Boundaries": {
            "Problem": "Model doesn't know where entities start/end",
            "Solutions": [
                "Use BIO tagging scheme",
                "Add CRF layer",
                "Train on more data",
                "Use longer context"
            ]
        },
        "2. Rare Entities": {
            "Problem": "Entities not seen in training",
            "Solutions": [
                "Use character-level embeddings",
                "Use subword tokenization",
                "Data augmentation",
                "Transfer learning"
            ]
        },
        "3. Ambiguous Entities": {
            "Problem": "Same word can be different entity types",
            "Solutions": [
                "Use context embeddings (BERT)",
                "Use larger context window",
                "Multi-task learning",
                "Entity disambiguation"
            ]
        }
    }
    
    for issue, info in issues.items():
        print(f"\n🔴 {issue}")
        print(f"   Problem: {info['Problem']}")
        print(f"   Solutions:")
        for sol in info['Solutions']:
            print(f"   • {sol}")

ner_issues()
```

## 9.2 Summarization Issues

```python
# ============ SUMMARIZATION ISSUES ============

def summarization_issues():
    """
    Common summarization problems and solutions
    """
    
    print("="*70)
    print("SUMMARIZATION COMMON ISSUES AND SOLUTIONS")
    print("="*70)
    
    issues = {
        "1. Factual Consistency": {
            "Problem": "Model generates incorrect facts",
            "Solutions": [
                "Use larger models",
                "Fact-checking modules",
                "Reinforcement learning",
                "Consistency constraints"
            ]
        },
        "2. Hallucination": {
            "Problem": "Model generates information not in source",
            "Solutions": [
                "Attention visualization",
                "Copy mechanisms",
                "Fact consistency loss",
                "Better training data"
            ]
        },
        "3. Informativeness": {
            "Problem": "Summary misses important information",
            "Solutions": [
                "Extractive + Abstractive hybrid",
                "Key phrase extraction",
                "Multiple reference summaries",
                "RL with informativeness reward"
            ]
        }
    }
    
    for issue, info in issues.items():
        print(f"\n🔴 {issue}")
        print(f"   Problem: {info['Problem']}")
        print(f"   Solutions:")
        for sol in info['Solutions']:
            print(f"   • {sol}")

summarization_issues()
```

---

# 10. QUICK REFERENCE - ALL CODE PATTERNS

## 10.1 NER Patterns

```python
# ============ NER PATTERNS ============

# BiLSTM-CRF
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers, num_labels):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers, 
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_labels)

# HuggingFace NER
from transformers import pipeline
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
entities = ner("Apple was founded by Steve Jobs")

# BERT for NER
from transformers import AutoTokenizer, AutoModelForTokenClassification
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=9)
```

## 10.2 Summarization Patterns

```python
# ============ SUMMARIZATION PATTERNS ============

# BART Summarization
from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer(text, max_length=150, min_length=30)

# PEGASUS
summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail")

# T5
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Extractive Summarization
def extractive_summarize(text, num_sentences):
    sentences = text.split('.')
    # Score sentences by importance
    scores = []
    for sent in sentences:
        # Add scoring logic
        scores.append(len(sent.split()))
    # Select top sentences
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_sentences]
    return ' '.join([sentences[i] for i in sorted(top_indices)])
```

## 10.3 Metrics Patterns

```python
# ============ METRICS PATTERNS ============

# ROUGE
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference, hypothesis)

# BLEU
from nltk.translate.bleu_score import sentence_bleu
bleu = sentence_bleu([reference.split()], hypothesis.split())

# Classification Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
```

## 10.4 Fine-Tuning Patterns

```python
# ============ FINE-TUNING PATTERNS ============

# BERT Fine-tuning
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
)
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()

# BART Fine-tuning for Summarization
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
)
trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=train_dataset)
```

---

**End of Week 5 Notes - Complete Advanced NLP Guide**

## 📌 Key Takeaways

1. **NER**: Identifies and classifies named entities (Person, Organization, Location, etc.)
2. **BIO Tagging**: B=Beginning, I=Inside, O=Outside
3. **Fine-Tuning**: Adapt pre-trained models to specific tasks with small learning rates
4. **Seq2Seq**: Encoder-decoder architecture for sequence generation
5. **Attention**: Mechanism that focuses on relevant parts of the input
6. **Summarization**: Extractive (select sentences) vs Abstractive (generate new text)
7. **Evaluation Metrics**: ROUGE, BLEU, Accuracy, F1-Score, etc.
8. **Transfer Learning**: Use pre-trained knowledge for new tasks

---

