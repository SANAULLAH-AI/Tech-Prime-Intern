# Week 7: Retrieval-Augmented Generation (RAG) - Complete Study Notes

## Table of Contents

1. Introduction to RAG Systems
2. Embeddings and Vector Representations
3. Vector Databases
4. Chunking Strategies
5. Retrieval Methods
6. Reranking
7. Query Understanding and Enhancement
8. RAG Pipeline Architecture
9. LangChain for RAG
10. LlamaIndex for RAG
11. Modern RAG Patterns
12. RAG Evaluation
13. RAG Optimization
14. Complete Project: RAG-Based Document Q&A System

---

## 1. Introduction to RAG Systems

### 1.1 What is Retrieval-Augmented Generation?

Retrieval-Augmented Generation (RAG) is an architecture pattern that combines:
- **Retrieval**: Finding relevant information from external knowledge sources
- **Generation**: Using LLMs to produce responses grounded in retrieved information

```text
RETRIEVAL-AUGMENTED GENERATION CORE PRINCIPLE:
═══════════════════════════════════════════════════════════════

User Question
     ↓
Retrieve relevant documents from knowledge base
     ↓
Inject retrieved content into LLM prompt as context
     ↓
LLM generates grounded answer
     ↓
Answer with citations / sources
```

### 1.2 The Need for RAG

LLMs have fundamental limitations that RAG addresses:

| Limitation | RAG Solution |
|---|---|
| Knowledge cutoff | Access to updated external sources |
| Hallucination | Grounding in retrieved documents |
| No private data access | Enterprise document retrieval |
| Black box reasoning | Source attribution and transparency |
| High retraining cost | Knowledge updates without retraining |
| Limited long-term memory | Persistent knowledge base |

### 1.3 RAG System Components

```text
RAG SYSTEM COMPONENT OVERVIEW:
═══════════════════════════════════════════════════════════════

1. KNOWLEDGE BASE
   - Raw documents (PDFs, web pages, databases)
   - Indexed for efficient retrieval
   - Updated independently of the LLM

2. CHUNKING
   - Document splitting into manageable pieces
   - Balance between context and specificity
   - Various strategies (fixed, semantic, document-structure-aware)

3. EMBEDDING MODEL
   - Converts text into dense vector representations
   - Captures semantic meaning
   - Enables similarity search

4. VECTOR INDEX
   - Stores embeddings for efficient retrieval
   - Approximate Nearest Neighbor (ANN) search
   - Scalable to millions of documents

5. RETRIEVER
   - Query processing and similarity calculation
   - Returns top-k relevant chunks
   - May combine multiple retrieval strategies

6. RERANKER (Optional)
   - Secondary scoring of retrieved candidates
   - Higher precision than initial retrieval
   - Orders by relevance for final context

7. LLM GENERATOR
   - Takes query + retrieved context as input
   - Produces final grounded response
   - May be prompted to include citations

8. GROUNDING / CITATION
   - Verification that response is supported by sources
   - Citation of specific documents/chunks
   - Hallucination detection
```

### 1.4 RAG vs Fine-Tuning

```text
RAG VS FINE-TUNING COMPARISON:
═══════════════════════════════════════════════════════════════

┌─────────────────────┬──────────────────┬──────────────────┐
│      Aspect         │       RAG        │   Fine-Tuning    │
├─────────────────────┼──────────────────┼──────────────────┤
│ Knowledge Updates   │ Instantaneous    │ Weeks / months   │
│ Factual Accuracy    │ High (grounded)  │ Variable         │
│ Hallucination       │ Lower            │ Higher           │
│ Transparency        │ High (sources)   │ Low              │
│ Latency             │ Higher           │ Lower            │
│ Implementation      │ Moderate         │ Complex          │
│ Cost (operational)  │ Variable (retrieval + gen) │ Fixed (inference) │
│ New knowledge       │ Immediate        │ Needs retraining │
│ Domain adaptation   │ Moderate         │ Strong           │
│ Behavior/format     │ Limited          │ Strong           │
│ Compute resources   │ Moderate         │ High             │
│ Privacy concerns    │ Data in external DB│ Model contains data │
└─────────────────────┴──────────────────┴──────────────────┘

WHEN TO USE RAG:
────────────────────────────────────────────────────────────
- Frequently changing knowledge
- Need for source attribution
- Access to large document collections
- Privacy requirements
- Limited training data

WHEN TO USE FINE-TUNING:
────────────────────────────────────────────────────────────
- Consistent response formatting
- Domain-specific language/style
- Behavior adaptation
- Low-latency requirements
- Proprietary knowledge
```

### 1.5 RAG System Requirements

```text
RAG SYSTEM DESIGN REQUIREMENTS:
═══════════════════════════════════════════════════════════════

ACCURACY REQUIREMENTS:
- Precision at K (P@K): Fraction of relevant documents in top-K results
- Recall at K (R@K): Fraction of relevant documents retrieved in top-K results
- Answer faithfulness: Response factually supported by sources
- Answer relevance: Response directly answers the question

PERFORMANCE REQUIREMENTS:
- Query latency (target: < 2-5 seconds)
- Ingestion throughput (documents/second)
- Query throughput (queries/second)
- Index size and memory usage

SCALABILITY REQUIREMENTS:
- Document count: 10^3 to 10^9
- Query volume: variable
- Concurrent users: variable
- Update frequency: daily to real-time

MAINTENANCE REQUIREMENTS:
- Index rebuilding strategy
- Document versioning
- Monitoring and alerts
- Quality measurement
- Model updates

COST REQUIREMENTS:
- Storage costs
- Embedding computation
- LLM invocation costs
- Infrastructure costs
- Development and maintenance
```

---

## 2. Embeddings and Vector Representations

### 2.1 What Are Embeddings?

Embeddings are dense vector representations of text that capture semantic meaning in a continuous vector space.

```text
EMBEDDING FUNDAMENTALS:
═══════════════════════════════════════════════════════════════

TEXT → EMBEDDING MODEL → VECTOR

"I love NLP" → [0.23, -0.45, 0.78, 0.12, ...]
"The cat sat" → [0.56, 0.12, -0.34, 0.89, ...]

KEY PROPERTIES:
────────────────────────────────────────────────────────────
1. Semantic Similarity: Similar texts have similar vectors
2. Dimensionality: Typically 384, 768, 1024, or 1536 dimensions
3. Normalization: Often unit-normalized (cosine similarity)
4. Fixed Size: Same length regardless of input text length
5. Dense: All dimensions contain meaningful information

SIMILARITY MEASURES:
────────────────────────────────────────────────────────────
Cosine Similarity: cos(θ) = (A·B) / (||A|| × ||B||)
- Range: [-1, 1]
- 1: Same direction, very similar
- 0: Orthogonal, unrelated
- -1: Opposite directions, dissimilar

Dot Product: A·B
- Efficient with normalized vectors
- Equivalent to cosine similarity for normalized vectors

Euclidean Distance: ||A - B||
- Smaller = more similar
- Sensitive to vector magnitude
```

### 2.2 Embedding Models

```text
POPULAR EMBEDDING MODELS:
═══════════════════════════════════════════════════════════════

OPENAI EMBEDDINGS:
- text-embedding-3-small (1536 dimensions)
- text-embedding-3-large (3072 dimensions)
- text-embedding-ada-002 (1536 dimensions)
- Good general-purpose embeddings
- API-based, not open-source

SENTENCE TRANSFORMERS (SBERT):
- all-MiniLM-L6-v2 (384 dimensions)
- all-mpnet-base-v2 (768 dimensions)
- BAAI/bge-large-en-v1.5 (1024 dimensions)
- Open-source, local deployment
- Fine-tuned for similarity tasks

INSTRUCTOR:
- Instructor-XL / Instructor-Large
- Instruction-aware embeddings
- Support task-specific behavior

E5 (EmbEddings from bidirEctional Encoder rEpresentations):
- intfloat/e5-small (384 dims)
- intfloat/e5-base (768 dims)
- intfloat/e5-large (1024 dims)
- Strong performance on MTEB benchmark
- Open-source

COHERE EMBEDDINGS:
- embed-english-v3.0
- embed-multilingual-v3.0
- Good multilingual support

VOYAGE AI:
- voyage-large-2
- voyage-code-2
- Optimized for RAG

OPEN-SOURCE LEADERS (2026):
────────────────────────────────────────────────────────────
- BGE-M3: Multilingual, mixed granularity
- GTE-Qwen2-7B-instruct: Strong for retrieval
- E5-Mistral-7B-instruct: Large, high quality
- NV-Embed-QA: Strong embedding model
```

### 2.3 Embedding Model Selection

```text
EMBEDDING MODEL SELECTION CRITERIA:
═══════════════════════════════════════════════════════════════

1. TASK ALIGNMENT
   - Information retrieval: Models fine-tuned for retrieval
   - Semantic search: General similarity models
   - Classification: Sentence embedding models
   - Multilingual: Multilingual models (e.g., BGE-M3)

2. DIMENSIONALITY
   - Higher dimensions: More expressive, more storage
   - Lower dimensions: Less expressive, less storage
   - Trade-off: Accuracy vs. storage/compute

3. PERFORMANCE
   - MTEB (Massive Text Embedding Benchmark)
   - BEIR (Benchmarking Information Retrieval)
   - Domain-specific benchmarks
   - Commercial vs. open-source

4. DEPLOYMENT CONSIDERATIONS
   - Local vs. API
   - Latency requirements
   - Throughput requirements
   - Cost constraints
   - Privacy requirements

5. PRACTICAL RULES:
────────────────────────────────────────────────────────────
- Start with all-MiniLM-L6-v2 for prototyping (small, fast)
- Use BGE or E5 for high-quality open-source RAG
- Use OpenAI embeddings for convenience and quality
- Use multilingual models when needed
- Test multiple models on your specific data

MODEL COMPARISON (GENERAL):
────────────────────────────────────────────────────────────
┌──────────────────────┬──────────┬──────────┬──────────────┐
│ Model                │ Dims     │ MTEB     │ Type         │
├──────────────────────┼──────────┼──────────┼──────────────┤
│ all-MiniLM-L6-v2     │ 384      │ 58.8     │ Open-source  │
│ all-mpnet-base-v2    │ 768      │ 63.0     │ Open-source  │
│ BAAI/bge-large-en    │ 1024     │ 63.8     │ Open-source  │
│ E5-large-v2          │ 1024     │ 64.6     │ Open-source  │
│ text-embedding-3-small│ 1536    │ 62.3     │ API          │
│ text-embedding-3-large│ 3072    │ 64.6     │ API          │
└──────────────────────┴──────────┴──────────┴──────────────┘
```

### 2.4 Embedding Generation Code

```python
# EMBEDDING GENERATION EXAMPLES:
# ============================================================

# Using Sentence Transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["I love NLP", "The cat sat"])
# Shape: (2, 384)

# Using OpenAI API
import openai

client = openai.OpenAI(api_key="YOUR_API_KEY")
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["I love NLP", "The cat sat"]
)
embeddings = [r.embedding for r in response.data]

# Using Hugging Face Transformers
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en")
model = AutoModel.from_pretrained("BAAI/bge-large-en")

def encode_bge(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

embeddings = encode_bge(["I love NLP", "The cat sat"])
```

---

## 3. Vector Databases

### 3.1 What is a Vector Database?

A vector database is a specialized database designed for storing, indexing, and searching high-dimensional vectors efficiently.

```text
VECTOR DATABASE OVERVIEW:
═══════════════════════════════════════════════════════════════

STORAGE LAYER:
- Stores vector embeddings
- May store metadata (text, source, timestamp)
- Persists data to disk

INDEXING LAYER:
- Builds efficient data structures for similarity search
- Approximate Nearest Neighbor (ANN) algorithms
- Trade-off: accuracy vs. speed vs. memory

QUERY LAYER:
- Converts query text to vector
- Searches index for nearest neighbors
- Returns top-k results with scores
- Filters by metadata

KEY OPERATIONS:
────────────────────────────────────────────────────────────
1. Insert: Add document chunks with embeddings and metadata
2. Delete: Remove documents
3. Update: Update embeddings or metadata
4. Query: Find top-k similar documents
5. Filter: Apply metadata filters
6. Get: Retrieve by ID
```

### 3.2 Vector Search Algorithms

```text
VECTOR SEARCH ALGORITHMS:
═══════════════════════════════════════════════════════════════

EXACT SEARCH (Brute Force):
- Compute similarity with all vectors
- O(n) time complexity
- 100% accuracy
- Impractical for large datasets

FLAT INDEX:
- No indexing structure
- Brute force search
- Best for small datasets (< 10,000 vectors)
- Highest accuracy

APPROXIMATE NEAREST NEIGHBOR (ANN):
────────────────────────────────────────────────────────────
- Trade accuracy for speed
- O(log n) to O(sqrt(n)) complexity
- 90-99% accuracy possible

1. IVF (Inverted File Index)
   - Cluster vectors into "inverted lists"
   - Search only relevant clusters
   - nprobe: number of clusters to search

2. HNSW (Hierarchical Navigable Small World)
   - Graph-based indexing
   - Multi-layered graph structure
   - Fast and high-quality
   - Memory-intensive

3. PQ (Product Quantization)
   - Compresses vectors by sub-vector quantization
   - Lower memory usage
   - Distance estimation

4. ScaNN (Scalable Nearest Neighbors)
   - Google's algorithm
   - Good speed-quality trade-off

5. IVFFlat
   - IVF with flat storage
   - Good baseline

6. IVFPQ
   - IVF with product quantization
   - Memory efficient

ALGORITHM COMPARISON:
────────────────────────────────────────────────────────────
┌────────────┬───────────┬────────────┬────────────┬────────────┐
│ Algorithm  │ Accuracy  │ Speed      │ Memory     │ Build Time │
├────────────┼───────────┼────────────┼────────────┼────────────┤
│ Flat       │ 100%      │ Slow       │ High       │ Instant    │
│ IVFFlat    │ 95%       │ Fast       │ High       │ Medium     │
│ HNSW       │ 95%       │ Fast       │ High       │ Slow       │
│ IVFPQ      │ 93%       │ Fast       │ Low        │ Slow       │
│ ScaNN      │ 96%       │ Very Fast  │ Medium     │ Medium     │
└────────────┴───────────┴────────────┴────────────┴────────────┘
```

### 3.3 Major Vector Databases

```text
VECTOR DATABASE COMPARISON:
═══════════════════════════════════════════════════════════════

FAISS (Facebook AI Similarity Search):
- Library, not a database
- Highly optimized
- Many index types
- Excellent performance
- No persistence built-in
- Popular for research

Pinecone:
- Managed vector database
- Serverless or dedicated
- High availability
- Automatic indexing
- Built-in filtering
- Production-ready
- Cost: $$$

Weaviate:
- Open-source
- GraphQL and REST API
- Built-in object storage
- Hybrid search (vector + keyword)
- Production-ready
- Flexible deployment
- Cost: $ (self-hosted)

Qdrant:
- Open-source
- REST API + gRPC
- Payload filtering
- Excellent performance
- Rust-based
- Production-ready
- Cost: $ (self-hosted)

Milvus / Zilliz:
- Cloud-native
- GPU acceleration
- Highly scalable
- Multiple index types
- Production-ready
- Cost: $$ (managed)

ChromaDB:
- Lightweight
- Python-native
- Built-in embeddings
- Simple API
- Not for large scale
- Best for prototyping

Elasticsearch:
- Full-text search
- Vector plugin
- Hybrid retrieval
- Mature ecosystem
- Cost: $ (self-hosted)

Pgvector:
- PostgreSQL extension
- SQL interface
- Transaction support
- Full-text integration
- Cost: $ (self-hosted)

RECOMMENDATIONS:
────────────────────────────────────────────────────────────
- Prototyping: ChromaDB
- Small/Medium self-hosted: Qdrant or Weaviate
- Large scale: Pinecone or Milvus
- If using PostgreSQL: pgvector
- If using Elasticsearch: vector plugin
- High performance: FAISS
```

### 3.4 Vector Database Implementation

```text
VECTOR DATABASE OPERATIONS:
═══════════════════════════════════════════════════════════════

# Using ChromaDB
import chromadb
from chromadb.utils import embedding_functions

# Initialize
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection(
    name="documents",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction()
)

# Insert documents
collection.add(
    documents=["Document 1 text", "Document 2 text", "Document 3 text"],
    metadatas=[{"source": "file1"}, {"source": "file2"}, {"source": "file3"}],
    ids=["doc1", "doc2", "doc3"]
)

# Query
results = collection.query(
    query_texts=["What is RAG?"],
    n_results=3,
    where={"source": "file1"}  # optional filtering
)

# Using Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(path="./qdrant_data")
collection_name = "documents"

# Create collection
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Insert points
client.upsert(
    collection_name=collection_name,
    points=[
        {
            "id": 1,
            "vector": [0.1, 0.2, ...],
            "payload": {"text": "Document 1", "source": "file1"}
        }
    ]
)

# Search
results = client.search(
    collection_name=collection_name,
    query_vector=[0.1, 0.2, ...],
    limit=3,
    query_filter={"must": [{"key": "source", "match": {"value": "file1"}}]}
)
```

### 3.5 Vector Database Index Configuration

```text
INDEX CONFIGURATION PARAMETERS:
═══════════════════════════════════════════════════════════════

IVF (Inverted File):
- nlist: Number of clusters
  - Larger = better quality, slower search
  - Smaller = faster search, lower quality
  - Typical: 100-1000 per million vectors

HNSW:
- M: Maximum connections per layer
  - Larger = better recall, more memory
  - Smaller = faster, lower recall
  - Typical: 16-64
- ef_construction: Build-time search depth
  - Larger = better quality, slower build
  - Typical: 100-500
- ef_search: Search-time search depth
  - Larger = better recall, slower
  - Typical: 100-300

PQ (Product Quantization):
- m: Number of sub-vectors
  - Larger = better compression, less accuracy
  - Typical: 8-16
- bits: Bits per sub-vector
  - Typical: 8

IVFPQ:
- nlist: Number of clusters
- m: Sub-vector count
- nprobe: Number of clusters to search
  - Larger = better recall, slower
  - Typical: 1-20

PRACTICAL GUIDELINES:
────────────────────────────────────────────────────────────
1. Start with default parameters
2. Test recall on your dataset
3. Increase ef_search or nprobe for better recall
4. Build index with balanced parameters
5. Monitor memory usage
6. Consider dynamic index rebuilding
```

---

## 4. Chunking Strategies

### 4.1 Why Chunking Matters

Chunking is the process of dividing documents into smaller segments for embedding and retrieval.

```text
CHUNKING IMPACT ON RAG:
═══════════════════════════════════════════════════════════════

TOO SMALL CHUNKS:
- Missing context
- Incomplete information
- Difficulty answering complex questions
- Higher retrieval costs
- More chunks to manage

TOO LARGE CHUNKS:
- Diluted semantic signal
- Inefficient retrieval
- Context window issues
- Contains irrelevant information
- Harder to cite precisely

OPTIMAL CHUNKS:
- Self-contained units
- Sufficient context for the question
- Appropriate for the embedding model
- Granular enough for citation
- Balanced size and semantic coherence
```

### 4.2 Chunking Methods

```text
CHUNKING STRATEGIES:
═══════════════════════════════════════════════════════════════

1. FIXED-SIZE CHUNKING
   ──────────────────────
   Split by number of tokens/characters with optional overlap.
   
   Pros:
   - Simple implementation
   - Predictable chunk sizes
   - Easy to optimize
   
   Cons:
   - May break semantic units
   - Can split sentences or paragraphs
   - No semantic awareness
   
   Parameters:
   - chunk_size: Number of tokens/characters
   - overlap: Overlap between chunks (10-25%)

2. SENTENCE-BASED CHUNKING
   ──────────────────────────
   Split at sentence boundaries.
   
   Pros:
   - Natural language units
   - Meaningful chunks
   - Better for some languages
   
   Cons:
   - Variable chunk sizes
   - May need to combine sentences
   - Sentence boundaries not always semantic

3. PARAGRAPH-BASED CHUNKING
   ────────────────────────────
   Split at paragraph boundaries.
   
   Pros:
   - Natural document structure
   - Coherent units
   - Usually good semantic units
   
   Cons:
   - Very variable sizes
   - Some paragraphs too large
   - May need post-processing

4. RECURSIVE SPLITTING
   ──────────────────────
   Try multiple strategies in order.
   
   Process:
   1. Try splitting by separators (paragraphs)
   2. If too large, try sentences
   3. If still too large, split words
   
   Pros:
   - Flexible
   - Maintains structure when possible
   - Good default choice

5. SEMANTIC CHUNKING
   ────────────────────
   Use embeddings to find natural boundaries.
   
   Process:
   1. Generate embeddings for sentences
   2. Compute similarity between adjacent sentences
   3. Split where similarity drops
   
   Pros:
   - Semantic coherence
   - Adaptive to content
   
   Cons:
   - Computationally expensive
   - Requires embedding model
   - Slower ingestion

6. DOCUMENT-STRUCTURE CHUNKING
   ──────────────────────────────
   Use document structure (headers, sections, tables).
   
   Pros:
   - Excellent for structured documents
   - Preserves hierarchy
   - Better for navigation
   
   Cons:
   - Requires structured documents
   - Implementation complexity

7. CONTEXTUAL RETRIEVAL
   ──────────────────────
   Each chunk includes preceding context.
   
   Example:
   - Chunk: "The system uses vector search"
   - Context: "RAG architecture. The system uses vector search"
   
   Pros:
   - Better context for chunks
   - Improves retrieval quality
   
   Cons:
   - Larger chunks
   - More storage
```

### 4.3 Chunk Size Selection

```text
CHUNK SIZE GUIDELINES:
═══════════════════════════════════════════════════════════════

FACTORS TO CONSIDER:
────────────────────────────────────────────────────────────
1. Embedding Model: 
   - Max tokens (e.g., 512 for BERT-based)
   - Optimal performance range

2. LLM Context Window:
   - Available space for context
   - Number of chunks that can fit

3. Document Type:
   - Short text: Larger chunks (512-1024 tokens)
   - Long documents: Smaller chunks (256-512 tokens)
   - Code: Small chunks (50-100 tokens)

4. Query Type:
   - Factual questions: Smaller chunks
   - Complex questions: Larger chunks
   - Broad questions: More chunks

5. Retrieval Strategy:
   - Parent-child: Small chunks for retrieval, large for generation

RECOMMENDED SIZES:
────────────────────────────────────────────────────────────
┌──────────────────────────────┬────────────────────────────┐
│ Use Case                     │ Chunk Size (tokens)        │
├──────────────────────────────┼────────────────────────────┤
│ General RAG                  │ 300-500                    │
│ Factual QA                   │ 150-250                    │
│ Long-form reasoning          │ 500-800                    │
│ Code documents               │ 100-200                    │
│ Scientific papers            │ 200-400                    │
│ Legal documents              │ 200-300                    │
│ News articles                │ 300-500                    │
│ Technical documentation      │ 200-400                    │
│ Social media                 │ 50-100                     │
└──────────────────────────────┴────────────────────────────┘

OVERLAP:
────────────────────────────────────────────────────────────
- Recommended: 10-25% overlap
- Overlap helps maintain context across chunk boundaries
- Too much overlap: redundant storage
- Too little overlap: missing context

RECOMMENDED DEFAULT:
────────────────────────────────────────────────────────────
Chunk Size: 512 tokens
Overlap: 128 tokens (25%)
```
### 4.4 Chunking Implementation

```python
# CHUNKING IMPLEMENTATION EXAMPLES:
# ============================================================

# 1. Fixed-size chunking with overlap
def fixed_chunk(text, chunk_size=512, overlap=128):
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = ' '.join(tokens[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# 2. Recursive character splitting (LangChain-style)
def recursive_chunk(text, chunk_size=500, overlap=50):
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    def split_by_separator(text, separator, chunk_size):
        parts = text.split(separator)
        chunks = []
        current_chunk = ""
        for part in parts:
            if len(current_chunk + part + separator) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part + separator
            else:
                current_chunk += part + separator
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    
    for separator in separators:
        chunks = split_by_separator(text, separator, chunk_size)
        if all(len(c) <= chunk_size for c in chunks):
            return chunks
    
    # Fallback: split by words
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# 3. Sentence-based chunking
from nltk.tokenize import sent_tokenize

def sentence_chunk(text, max_chunk_size=500, overlap=50):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence.split())
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            # Overlap
            overlap_size = 0
            overlap_sentences = []
            for s in reversed(current_chunk):
                if overlap_size + len(s.split()) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_size += len(s.split())
                else:
                    break
            current_chunk = overlap_sentences
            current_size = overlap_size
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# 4. Semantic chunking
from sentence_transformers import SentenceTransformer
import numpy as np

def semantic_chunk(text, similarity_threshold=0.7, max_chunk_size=500):
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return [text]
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        similarity = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
        )
        
        if similarity < similarity_threshold:
            # New chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
            
        if len(current_chunk) * 10 > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# 5. Document-structure-aware chunking (Markdown example)
import re

def markdown_chunk(text, max_chunk_size=500):
    # Split by headers
    headers = re.split(r'\n(#{1,6}\s+.*)\n', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    current_header = ""
    
    for i, section in enumerate(headers):
        if re.match(r'^#{1,6}\s+', section):
            # This is a header
            if current_chunk and current_size > 0:
                chunks.append({
                    'header': current_header,
                    'text': '\n'.join(current_chunk).strip()
                })
            current_header = section
            current_chunk = []
            current_size = 0
        else:
            # This is content
            section_words = len(section.split())
            if current_size + section_words > max_chunk_size and current_chunk:
                chunks.append({
                    'header': current_header,
                    'text': '\n'.join(current_chunk).strip()
                })
                current_chunk = []
                current_size = 0
            current_chunk.append(section)
            current_size += section_words
    
    if current_chunk:
        chunks.append({
            'header': current_header,
            'text': '\n'.join(current_chunk).strip()
        })
    
    return chunks
```

---

## 5. Retrieval Methods

### 5.1 Types of Retrieval

```text
RETRIEVAL TYPES:
═══════════════════════════════════════════════════════════════

1. DENSE RETRIEVAL
   ──────────────────
   Use embedding vectors for semantic similarity.
   
   How it works:
   1. Convert query to vector
   2. Search vector index
   3. Return top-k by cosine similarity
   
   Pros:
   - Semantic understanding
   - Handles paraphrasing
   - Works with synonyms
   - Good for complex queries
   
   Cons:
   - Needs embedding model
   - Index building required
   - May miss exact terms
   - Less interpretable

2. SPARSE RETRIEVAL (Keyword/BM25)
   ─────────────────────────────────
   Use term frequency and inverse document frequency.
   
   How it works:
   1. Tokenize query and documents
   2. Compute TF-IDF or BM25 scores
   3. Return top-k by score
   
   Pros:
   - Fast
   - Interpretable
   - Good for exact matches
   - No embedding required
   
   Cons:
   - No semantic understanding
   - Synonym handling poor
   - Vocabulary mismatch issues

3. HYBRID RETRIEVAL
   ──────────────────
   Combine dense and sparse retrieval.
   
   How it works:
   1. Run both retrievers
   2. Merge and re-rank results
   3. Return combined top-k
   
   Pros:
   - Best of both worlds
   - Robust to different queries
   - Higher overall recall
   
   Cons:
   - More complex
   - Higher latency
   - Weight tuning required

4. STRUCTURED RETRIEVAL
   ──────────────────────
   Use metadata and relationships.
   
   How it works:
   1. Filter by metadata (date, author, category)
   2. Search within filtered set
   
   Pros:
   - Precision filtering
   - Domain-specific constraints
   - Lower false positives
   
   Cons:
   - Requires good metadata
   - Limited to structured queries

5. MULTI-QUERY RETRIEVAL
   ────────────────────────
   Generate multiple query variants.
   
   How it works:
   1. Generate N query variants with LLM
   2. Retrieve for each query
   3. Merge and rank results
   
   Pros:
   - Covers multiple angles
   - Better recall
   - Good for ambiguous queries
   
   Cons:
   - Higher latency
   - Increased cost
   - Duplicate results possible

6. PARENT-CHILD RETRIEVAL
   ────────────────────────
   Retrieve small chunks, return larger context.
   
   How it works:
   1. Split document into small parent and child chunks
   2. Embed and store child chunks
   3. Search child chunks
   4. Return parent chunks as context
   
   Pros:
   - Better retrieval precision
   - More generation context
   - Balance of specificity and completeness
   
   Cons:
   - More complex pipeline
   - Storage overhead
```

### 5.2 Retrieval Implementation

```python
# RETRIEVAL IMPLEMENTATION EXAMPLES:
# ============================================================

# 1. Dense Retrieval with FAISS
import faiss
import numpy as np

class DenseRetriever:
    def __init__(self, embed_model, index_path=None):
        self.embed_model = embed_model
        self.dimension = 384
        self.index = None
        self.documents = []
        self.ids = []
        
    def add_documents(self, documents, ids=None):
        embeddings = self.embed_model.encode(documents)
        embeddings = embeddings.astype(np.float32)
        
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.documents.extend(documents)
        if ids:
            self.ids.extend(ids)
        else:
            self.ids.extend(range(len(documents)))
    
    def search(self, query, k=5):
        query_vector = self.embed_model.encode([query]).astype(np.float32)
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    'id': self.ids[idx],
                    'document': self.documents[idx],
                    'score': float(distances[0][i])
                })
        return results

# 2. BM25 Implementation (using rank-bm25)
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self):
        self.tokenizer = lambda x: x.lower().split()
        self.documents = []
        self.bm25 = None
    
    def add_documents(self, documents):
        self.documents = documents
        tokenized_docs = [self.tokenizer(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def search(self, query, k=5):
        tokenized_query = self.tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': float(scores[idx])
            })
        return results

# 3. Hybrid Retriever
class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.weight_dense = 0.5
        self.weight_sparse = 0.5
    
    def search(self, query, k=5):
        dense_results = self.dense.search(query, k=k*2)
        sparse_results = self.sparse.search(query, k=k*2)
        
        # Merge and normalize scores
        docs = {}
        for r in dense_results:
            docs[r['document']] = {
                'dense_score': r['score'],
                'sparse_score': 0,
                'combined': self.weight_dense * r['score']
            }
        
        for r in sparse_results:
            if r['document'] in docs:
                docs[r['document']]['sparse_score'] = r['score']
                docs[r['document']]['combined'] += self.weight_sparse * r['score']
            else:
                docs[r['document']] = {
                    'dense_score': 0,
                    'sparse_score': r['score'],
                    'combined': self.weight_sparse * r['score']
                }
        
        # Sort by combined score
        sorted_docs = sorted(
            docs.items(),
            key=lambda x: x[1]['combined'],
            reverse=True
        )[:k]
        
        return [{'document': d, 'combined_score': s['combined']} for d, s in sorted_docs]
```

### 5.3 Retrieval Metrics

```text
RETRIEVAL METRICS:
═══════════════════════════════════════════════════════════════

PRECISION AT K (P@K):
- Fraction of relevant documents in top-K results
- P@5 = (Number of relevant in top 5) / 5
- Higher = better precision
- Important when false positives are costly

RECALL AT K (R@K):
- Fraction of all relevant documents found in top-K
- R@5 = (Number of relevant in top 5) / (Total relevant)
- Higher = better recall
- Important when missing a relevant document is costly

MEAN AVERAGE PRECISION (MAP):
- Average of precision at each relevant document
- Accounts for ranking quality
- Single number metric

MEAN RECIPROCAL RANK (MRR):
- Reciprocal of the rank of the first relevant document
- MRR = 1/rank_of_first_relevant
- Higher = better
- Focuses on top result quality

NDCG (Normalized Discounted Cumulative Gain):
- Accounts for graded relevance
- Positions weighted by importance
- Standard metric for information retrieval

RECOMMENDED TARGETS:
────────────────────────────────────────────────────────────
- P@5: > 0.80 (80% of top 5 are relevant)
- R@5: > 0.70 (70% of all relevant in top 5)
- MRR: > 0.75
- NDCG: > 0.80
```

---

## 6. Reranking

### 6.1 Why Reranking?

Reranking is a secondary scoring step that improves retrieval results.

```text
RERANKING BENEFITS:
═══════════════════════════════════════════════════════════════

INITIAL RETRIEVAL (Stage 1):
- Fast, approximate search
- Retrieves many candidates (50-100)
- May include irrelevant results

RERANKING (Stage 2):
- Slower, more accurate scoring
- Re-orders top candidates
- Removes irrelevant documents
- Returns top 5-10

BENEFITS:
- Higher precision
- Better ordering for context
- Can use more expensive models
- Cross-encoder can consider query-document interactions
- Works with any initial retriever
```

### 6.2 Reranking Approaches

```text
RERANKING APPROACHES:
═══════════════════════════════════════════════════════════════

1. CROSS-ENCODER RERANKING
   ─────────────────────────
   Process query and document together through a transformer.
   
   How it works:
   1. Concatenate query and document
   2. Pass through cross-encoder
   3. Score relevance directly
   
   Pros:
   - High accuracy
   - Context-aware
   - Can consider interactions
   
   Cons:
   - Slower than bi-encoders
   - Can't be used for initial retrieval
   - More expensive

2. BERT RERANKER
   ───────────────
   Use BERT for query-document relevance scoring.
   
   Common models:
   - cross-encoder/ms-marco-MiniLM-L-6-v2
   - cross-encoder/ms-marco-MiniLM-L-12-v2
   - BAAI/bge-reranker-large

3. COHERE RERANK
   ───────────────
   API-based reranking service.
   
   Features:
   - High accuracy
   - Easy to use
   - Handles multiple languages
   - Cost: per-query

4. VOTING / ENSEMBLE
   ───────────────────
   Combine multiple scoring methods.
   
   Methods:
   - Simple averaging
   - Weighted voting
   - Rank aggregation

5. HYBRID RERANKING
   ──────────────────
   Combine multiple signals.
   
   Signals:
   - Semantic similarity
   - Keyword matching
   - Metadata relevance
   - Recency
   - Authority

RERANKING MODEL COMPARISON:
────────────────────────────────────────────────────────────
┌─────────────────────┬───────────┬──────────┬──────────────┐
│ Model               │ Speed     │ Accuracy │ Size         │
├─────────────────────┼───────────┼──────────┼──────────────┤
│ MiniLM-L-6-v2       │ Fast      │ Good     │ 80 MB        │
│ MiniLM-L-12-v2      │ Medium    │ Better   │ 120 MB       │
│ BGE-reranker-large  │ Slow      │ Best     │ 1.3 GB       │
│ Cohere Rerank       │ Fast      │ Best     │ API          │
└─────────────────────┴───────────┴──────────┴──────────────┘
```

### 6.3 Reranking Implementation

```python
# RERANKING IMPLEMENTATION EXAMPLES:
# ============================================================

# 1. Cross-Encoder Reranking
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query, documents, top_k=5):
        # Prepare pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Sort by score
        scored_results = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return scored_results[:top_k]

# 2. BGE Reranker
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class BGEReranker:
    def __init__(self, model_name='BAAI/bge-reranker-large'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
    
    def rerank(self, query, documents, top_k=5):
        pairs = [[query, doc] for doc in documents]
        
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze().tolist()
        
        if isinstance(scores, float):
            scores = [scores]
        
        scored_results = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return scored_results[:top_k]

# 3. Multi-stage RAG Pipeline with Reranking
class MultiStageRAG:
    def __init__(self, retriever, reranker, llm):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        
        # Configuration
        self.initial_k = 50
        self.rerank_k = 10
    
    def query(self, question, top_k=5):
        # Stage 1: Initial retrieval
        initial_results = self.retriever.search(
            question,
            k=self.initial_k
        )
        initial_docs = [r['document'] for r in initial_results]
        
        # Stage 2: Reranking
        reranked_results = self.reranker.rerank(
            question,
            initial_docs,
            top_k=self.rerank_k
        )
        
        # Stage 3: Generate with context
        context_docs = [r[0] for r in reranked_results]
        context = "\n\n".join(context_docs)
        
        prompt = f"""Based on the following context, answer the question.
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        response = self.llm.generate(prompt)
        
        return {
            'answer': response,
            'retrieved_docs': context_docs,
            'scores': [r[1] for r in reranked_results]
        }
```

---

## 7. Query Understanding and Enhancement

### 7.1 Query Analysis

Understanding the user's query is crucial for effective retrieval.

```text
QUERY ANALYSIS TECHNIQUES:
═══════════════════════════════════════════════════════════════

1. QUERY TYPE CLASSIFICATION
   ───────────────────────────
   Determine the nature of the query.
   
   Types:
   - Factual: "What is the capital of France?"
   - Explanatory: "How does neural networks work?"
   - Procedural: "How to install Python?"
   - Comparative: "What is the difference between A and B?"
   - Exploratory: "Tell me about machine learning"
   - Conversational: Follow-up questions

2. QUERY DECOMPOSITION
   ─────────────────────
   Break complex queries into sub-queries.
   
   Example:
   Original: "What are the symptoms, causes, and treatments of diabetes?"
   Decomposed:
   - "What are the symptoms of diabetes?"
   - "What are the causes of diabetes?"
   - "What are the treatments of diabetes?"

3. QUERY REWRITING
   ────────────────
   Reformulate queries to improve retrieval.
   
   Techniques:
   - Synonym expansion: "car" → "automobile vehicle auto"
   - Re-phrasing: "Why is..." → "Causes of..."
   - Spelling correction
   - Abbreviation expansion: "AI" → "Artificial Intelligence"

4. HYDE (Hypothetical Document Embeddings)
   ────────────────────────────────────────
   Generate a hypothetical document and use its embedding.
   
   Process:
   1. Query: "Explain gradient descent"
   2. Generate hypothetical answer with LLM
   3. Embed the hypothetical answer
   4. Use this embedding for retrieval

5. STEP-BACK PROMPTING
   ─────────────────────
   Ask broader abstract question first.
   
   Example:
   Query: "What are the effects of climate change on agriculture in India?"
   Step-back: "What is climate change?"
   Then use step-back answer + original query

6. MULTI-QUERY GENERATION
   ────────────────────────
   Generate multiple related queries.
   
   Example:
   Original: "How to improve model accuracy?"
   Generated:
   - "How to reduce model error"
   - "Techniques for better prediction"
   - "Ways to boost model performance"
```

### 7.2 Query Enhancement Implementation

```python
# QUERY ENHANCEMENT IMPLEMENTATION:
# ============================================================

# 1. Query Decomposition
class QueryDecomposer:
    def __init__(self, llm):
        self.llm = llm
    
    def decompose(self, query):
        prompt = f"""Decompose the following complex question into simpler sub-questions.
        Each sub-question should be specific and answerable independently.
        
        Question: {query}
        
        Sub-questions:
        1."""
        
        response = self.llm.generate(prompt)
        
        # Parse response
        sub_queries = []
        for line in response.strip().split('\n'):
            if line.strip() and line[0].isdigit():
                sub_queries.append(line.split('.', 1)[1].strip())
        
        return sub_queries

# 2. HyDE (Hypothetical Document Embeddings)
class HyDERetriever:
    def __init__(self, llm, embed_model, retriever):
        self.llm = llm
        self.embed_model = embed_model
        self.retriever = retriever
    
    def search(self, query, k=5):
        # Generate hypothetical document
        prompt = f"""Write a hypothetical document that would answer the following query.
        The document should be informative and comprehensive.
        
        Query: {query}
        
        Hypothetical document:"""
        
        hypothetical_doc = self.llm.generate(prompt)
        
        # Use hypothetical doc for retrieval
        # Option 1: Embed and search with hypothetical doc
        doc_embedding = self.embed_model.encode([hypothetical_doc])[0]
        results = self.retriever.search_by_vector(doc_embedding, k=k)
        
        # Option 2: Use both original query and hypothetical doc
        query_embedding = self.embed_model.encode([query])[0]
        combined = (query_embedding + doc_embedding) / 2
        results = self.retriever.search_by_vector(combined, k=k)
        
        return results

# 3. Step-Back Prompting
class StepBackPrompting:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def query(self, question):
        # Step 1: Generate step-back question
        prompt = f"""Generate a broader, more general question that provides context for:
        
        Specific question: {question}
        
        Broader question:"""
        
        broader_q = self.llm.generate(prompt)
        
        # Step 2: Retrieve for both
        specific_results = self.retriever.search(question, k=3)
        broader_results = self.retriever.search(broader_q, k=3)
        
        # Step 3: Combine
        all_results = specific_results + broader_results
        # Remove duplicates
        seen = set()
        unique_results = []
        for r in all_results:
            doc_key = r.get('id', r['document'])
            if doc_key not in seen:
                seen.add(doc_key)
                unique_results.append(r)
        
        return unique_results
```

### 7.3 Query Rewriting Techniques

```text
QUERY REWRITING TECHNIQUES:
═══════════════════════════════════════════════════════════════

1. SYNONYM EXPANSION
   ───────────────────
   Expand query with synonyms.
   
   Example:
   "How to repair a car" →
   "How to repair a car automobile vehicle fix maintain"

2. ABBREVIATION EXPANSION
   ────────────────────────
   Expand abbreviations.
   
   Example:
   "AI in healthcare" →
   "Artificial Intelligence in healthcare"

3. QUERY REFORMULATION
   ─────────────────────
   Rephrase for better retrieval.
   
   Example:
   "Why is the sky blue?" →
   "The sky is blue because of what reason?"

4. SPELLING CORRECTION
   ─────────────────────
   Fix spelling errors.
   
   Example:
   "how to instll python" →
   "how to install python"

5. CONTEXT-AWARE REWRITING
   ─────────────────────────
   Include conversational context.
   
   Example:
   Previous: "What is machine learning?"
   Current: "How does it work?" →
   "How does machine learning work?"

6. QUERY EXPANSION WITH LLM
   ──────────────────────────
   Use LLM to generate related queries.
   
   Example:
   "What are the best practices for testing?"
   →
   "testing best practices software quality assurance QA strategies test automation methods"
```

### 7.4 Query Understanding Implementation

```python
# COMPLETE QUERY UNDERSTANDING SYSTEM:
# ============================================================

class QueryUnderstanding:
    def __init__(self, llm):
        self.llm = llm
    
    def analyze(self, query, conversation_history=None):
        """Analyze query and return enhanced versions"""
        enhancement = {}
        
        # 1. Classify query type
        enhancement['type'] = self._classify(query)
        
        # 2. Extract entities
        enhancement['entities'] = self._extract_entities(query)
        
        # 3. Generate alternatives
        enhancement['alternatives'] = self._generate_alternatives(query)
        
        # 4. Rewrite for retrieval
        enhancement['retrieval_query'] = self._rewrite_for_retrieval(query)
        
        # 5. Add conversation context if available
        if conversation_history:
            enhancement['contextualized'] = self._add_context(
                query, conversation_history
            )
        
        return enhancement
    
    def _classify(self, query):
        prompt = f"""Classify this query type:
        Options: factual, explanatory, procedural, comparative, exploratory
        
        Query: {query}
        
        Type:"""
        
        return self.llm.generate(prompt).strip().lower()
    
    def _extract_entities(self, query):
        prompt = f"""Extract key entities from this query.
        Return as a comma-separated list.
        
        Query: {query}
        
        Entities:"""
        
        response = self.llm.generate(prompt)
        return [e.strip() for e in response.split(',')]
    
    def _generate_alternatives(self, query):
        prompt = f"""Generate 3 alternative ways to ask this question.
        
        Original: {query}
        
        Alternatives:
        1."""
        
        response = self.llm.generate(prompt)
        alternatives = []
        for line in response.strip().split('\n'):
            if line.strip() and line[0].isdigit():
                alternatives.append(line.split('.', 1)[1].strip())
        return alternatives
    
    def _rewrite_for_retrieval(self, query):
        prompt = f"""Rewrite this query for better information retrieval.
        Make it more specific and keyword-rich.
        
        Original: {query}
        
        Rewritten:"""
        
        return self.llm.generate(prompt).strip()
    
    def _add_context(self, query, history):
        prompt = f"""Contextualize this follow-up question based on the conversation.
        
        Conversation history:
        {history}
        
        Current query: {query}
        
        Contextualized query:"""
        
        return self.llm.generate(prompt).strip()
```

---

## 8. RAG Pipeline Architecture

### 8.1 Complete RAG Pipeline

```text
COMPLETE RAG PIPELINE:
═══════════════════════════════════════════════════════════════

OFFLINE PHASE (Indexing):
═══════════════════════════════════════════════════════════════

Documents
    ↓
Document Loaders
    ↓
Chunking
    ↓
Embedding Generation
    ↓
Vector Index (Chroma, Qdrant, FAISS)
    ↓
Metadata Store
    ↓
Ready for Query

ONLINE PHASE (Query):
═══════════════════════════════════════════════════════════════

User Query
    ↓
Query Understanding
    ├── Type classification
    ├── Entity extraction
    └── Rewriting
    ↓
Enhanced Query
    ↓
Retrieval
    ├── Dense retrieval
    ├── Sparse retrieval
    └── Hybrid retrieval
    ↓
Initial Documents (50-100)
    ↓
Reranking
    ↓
Top Documents (5-10)
    ↓
Context Assembly
    ├── Ordering
    ├── Compression
    └── Prompt creation
    ↓
LLM Generation
    ↓
Validation / Grounding
    ↓
Answer + Sources
```

### 8.2 Document Processing Pipeline

```text
DOCUMENT PROCESSING PIPELINE:
═══════════════════════════════════════════════════════════════

INPUT DOCUMENTS:
- PDFs, Word, Text, Markdown
- Web pages, HTML
- Code repositories
- Databases
- Emails

DOCUMENT LOADERS:
- PyPDF2, PDFPlumber
- python-docx
- BeautifulSoup
- Unstructured library
- Custom loaders

CLEANING:
- Remove headers/footers
- Remove watermarks
- Fix encoding issues
- Normalize whitespace
- Remove repeated content
- Remove boilerplate text

CHUNKING:
- Recursive splitting
- Sentence-based
- Semantic segmentation
- Document-structure-aware

METADATA EXTRACTION:
- Source filename
- Document type
- Page numbers
- Sections/headers
- Timestamps
- Authors
- Tags

EMBEDDING:
- Batch processing
- Cache embeddings
- Handle truncation
- Multi-model support

INDEXING:
- Add to vector index
- Store metadata separately
- Maintain versioning
- Track document IDs

VALIDATION:
- Verify embeddings
- Check chunk sizes
- Validate metadata
- Test retrieval
- Quality checks
```

### 8.3 Context Assembly

```text
CONTEXT ASSEMBLY STRATEGIES:
═══════════════════════════════════════════════════════════════

ORDERING DOCUMENTS:
────────────────────────────────────────────────────────────
1. By relevance score (highest first)
2. Interleaving sources
3. Chronological order
4. Logical flow

CONTEXT COMPRESSION:
────────────────────────────────────────────────────────────
1. Keep original order of evidence
2. Remove redundant information
3. Prioritize specific vs. general
4. Keep metadata/citations

PROMPT CONSTRUCTION:
────────────────────────────────────────────────────────────
System: "You are a helpful assistant. Answer based on context."
Context: [Retrieved documents with sources]
Query: [User question]
Instructions:
  - Be concise
  - Cite sources
  - Acknowledge uncertainty
  - Don't hallucinate
  - Refuse if unsupported

WINDOW MANAGEMENT:
────────────────────────────────────────────────────────────
1. Maximum tokens: Model context window - response tokens
2. Prioritize high-relevance chunks
3. Use summarization if needed
4. Consider sliding window
```

### 8.4 Pipeline Implementation

```python
# COMPLETE RAG PIPELINE IMPLEMENTATION:
# ============================================================

from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM

class RAGPipeline:
    def __init__(self):
        # Initialize components
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = None
        self.llm = None
        
        # Initialize vector database
        self.db = chromadb.PersistentClient(path="./rag_db")
        self.collection = self.db.get_or_create_collection(
            name="documents",
            embedding_function=None  # Use our own embeddings
        )
        
        # Initialize BM25 retriever
        self.bm25 = None
        self.bm25_docs = []
        
        # Configuration
        self.initial_k = 50
        self.rerank_k = 10
        
        # Initialize reranker
        from sentence_transformers import CrossEncoder
        self.reranker = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2'
        )
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the pipeline."""
        # Extract text
        texts = [doc['text'] for doc in documents]
        ids = [doc.get('id', f"doc_{i}") for i, doc in enumerate(documents)]
        metadatas = [doc.get('metadata', {}) for doc in documents]
        
        # Generate embeddings
        embeddings = self.embed_model.encode(texts).tolist()
        
        # Add to vector database
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        # Add to BM25
        tokenized_docs = [text.lower().split() for text in texts]
        if self.bm25 is None:
            self.bm25 = BM25Okapi(tokenized_docs)
            self.bm25_docs = texts
        else:
            # Extend BM25 (simplified - in practice, rebuild or use incremental)
            self.bm25 = BM25Okapi(self.bm25_docs + tokenized_docs)
            self.bm25_docs = self.bm25_docs + texts
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        # 1. Query understanding
        enhanced_query = self._enhance_query(query)
        
        # 2. Dense retrieval
        query_embedding = self.embed_model.encode([enhanced_query]).tolist()
        dense_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=self.initial_k
        )
        
        # 3. Sparse retrieval (BM25)
        tokenized_query = enhanced_query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:self.initial_k]
        sparse_results = [self.bm25_docs[i] for i in top_bm25_indices]
        
        # 4. Hybrid retrieval - merge results
        all_docs = {}
        
        # Add dense results
        for i, doc in enumerate(dense_results['documents'][0]):
            all_docs[doc] = {'dense_score': 1.0 - (i / self.initial_k), 'sparse_score': 0}
        
        # Add sparse results
        for i, doc in enumerate(sparse_results):
            if doc in all_docs:
                all_docs[doc]['sparse_score'] = 1.0 - (i / self.initial_k)
            else:
                all_docs[doc] = {'dense_score': 0, 'sparse_score': 1.0 - (i / self.initial_k)}
        
        # Calculate combined scores
        for doc, scores in all_docs.items():
            scores['combined'] = 0.5 * scores['dense_score'] + 0.5 * scores['sparse_score']
        
        # Sort by combined score
        sorted_docs = sorted(
            all_docs.items(),
            key=lambda x: x[1]['combined'],
            reverse=True
        )
        
        candidate_docs = [doc for doc, _ in sorted_docs[:self.initial_k]]
        candidate_scores = [scores['combined'] for _, scores in sorted_docs[:self.initial_k]]
        
        # 5. Reranking
        reranked = self.reranker.rerank(query, candidate_docs, top_k=k)
        
        # 6. Return results
        results = []
        for doc, score in reranked:
            # Find metadata
            metadata = {}
            if 'metadatas' in dense_results and dense_results['metadatas']:
                # Find matching document
                for i, d in enumerate(dense_results['documents'][0]):
                    if d == doc:
                        metadata = dense_results['metadatas'][i]
                        break
            
            results.append({
                'document': doc,
                'score': score,
                'metadata': metadata
            })
        
        return results
    
    def generate(self, query: str, context_docs: List[str]) -> str:
        """Generate answer using LLM."""
        # Build context
        context = "\n\n".join(context_docs)
        
        # Build prompt
        prompt = f"""Based on the following context, answer the question.
        
        Context:
        {context}
        
        Question: {query}
        
        Instructions:
        - Answer based only on the provided context
        - If the context doesn't contain the information, say so
        - Cite the relevant parts of the context
        - Be concise and clear
        
        Answer:"""
        
        # Generate response
        if self.llm is None:
            # Simple fallback - in practice, use actual LLM
            return f"Based on the context, here's the answer to: {query}"
        
        # Use actual LLM
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.llm.generate(**inputs, max_new_tokens=512)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response
    
    def query(self, question: str) -> Dict[str, Any]:
        """Full RAG query pipeline."""
        # 1. Search
        search_results = self.search(question, k=self.rerank_k)
        context_docs = [r['document'] for r in search_results]
        
        # 2. Generate
        answer = self.generate(question, context_docs)
        
        # 3. Return
        return {
            'question': question,
            'answer': answer,
            'sources': search_results,
            'context_docs': context_docs
        }
    
    def _enhance_query(self, query: str) -> str:
        """Enhance query for better retrieval."""
        # Simple enhancement - can add more
        enhanced = query.lower().strip()
        
        # Remove punctuation
        import re
        enhanced = re.sub(r'[^\w\s]', ' ', enhanced)
        
        return enhanced
    
    def initialize_llm(self, model_name: str):
        """Initialize the LLM component."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
```

---

## 9. LangChain for RAG

### 9.1 LangChain Overview

LangChain is a framework for developing applications powered by language models.

```text
LANGCHAIN FOR RAG:
═══════════════════════════════════════════════════════════════

CORE COMPONENTS:
────────────────────────────────────────────────────────────
1. Documents: Text chunks with metadata
2. Document Loaders: Load documents from various sources
3. Text Splitters: Chunking strategies
4. Embeddings: Convert text to vectors
5. Vector Stores: Vector database interfaces
6. Retrievers: Search and retrieval logic
7. LLM: Language model interface
8. Chains: Combine components into workflows

RAG IN LANGCHAIN:
────────────────────────────────────────────────────────────
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

# 1. Load documents
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# 3. Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Vector Store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

# 5. Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 6. QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)
```

### 9.2 LangChain Components

```text
LANGCHAIN COMPONENTS DETAILED:
═══════════════════════════════════════════════════════════════

DOCUMENT LOADERS:
────────────────────────────────────────────────────────────
- PyPDFLoader: PDF files
- TextLoader: Plain text
- CSVLoader: CSV files
- UnstructuredFileLoader: Various formats
- DirectoryLoader: Load all files in directory
- WebBaseLoader: Web pages
- ArxivLoader: ArXiv papers
- WikipediaLoader: Wikipedia articles

TEXT SPLITTERS:
────────────────────────────────────────────────────────────
- CharacterTextSplitter: Fixed character length
- RecursiveCharacterTextSplitter: Recursive splitting
- TokenTextSplitter: Token-based splitting
- MarkdownHeaderTextSplitter: Markdown headers
- PythonCodeTextSplitter: Code-aware
- NLTKTextSplitter: Sentence-based (NLTK)
- SpacyTextSplitter: Sentence-based (spaCy)

EMBEDDINGS:
────────────────────────────────────────────────────────────
- HuggingFaceEmbeddings: Open-source models
- OpenAIEmbeddings: OpenAI API
- CohereEmbeddings: Cohere API
- GooglePalmEmbeddings: Google PaLM
- VertexAIEmbeddings: Google Vertex AI

VECTOR STORES:
────────────────────────────────────────────────────────────
- Chroma: Local persistent store
- FAISS: Efficient similarity search
- Pinecone: Managed vector database
- Qdrant: Open-source vector DB
- Weaviate: Open-source vector DB
- Milvus: Cloud-native vector DB
- PGVector: PostgreSQL extension

RETRIEVERS:
────────────────────────────────────────────────────────────
- VectorStoreRetriever: Basic vector retrieval
- MultiQueryRetriever: Multiple query generation
- ContextualCompressionRetriever: With compression
- ParentDocumentRetriever: Parent-child retrieval
- EnsembleRetriever: Combine multiple retrievers
- SelfQueryRetriever: Self-querying with metadata

CHAINS:
────────────────────────────────────────────────────────────
- RetrievalQA: QA with retrieval
- ConversationalRetrievalChain: Chat with memory
- StuffDocumentsChain: Combine documents
- MapReduceDocumentsChain: Map-reduce for long docs
- RefineDocumentsChain: Refine answers iteratively
- LLMChain: Simple LLM chain
```

### 9.3 LangChain RAG Implementation

```python
# LANGCHAIN RAG IMPLEMENTATION:
# ============================================================

from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline, OpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

class LangChainRAG:
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
    
    def load_documents(self, path: str, file_type: str = "pdf"):
        """Load documents from path."""
        if file_type == "pdf":
            loader = PyPDFLoader(path)
        elif file_type == "text":
            loader = TextLoader(path)
        elif file_type == "directory":
            loader = DirectoryLoader(
                path,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        documents = loader.load()
        return documents
    
    def split_documents(self, documents, chunk_size=500, chunk_overlap=50):
        """Split documents into chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        return chunks
    
    def create_vectorstore(self, chunks, model_name="all-MiniLM-L6-v2"):
        """Create vector store from chunks."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        return self.vectorstore
    
    def setup_retriever(self, search_type="similarity", k=4):
        """Setup retriever."""
        self.retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
        return self.retriever
    
    def setup_advanced_retrievers(self):
        """Setup more advanced retrieval strategies."""
        # 1. Multi-Query Retriever
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.retriever,
            llm=self.llm
        )
        
        # 2. Contextual Compression
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.retriever
        )
        
        # 3. Ensemble Retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.retriever,
                multi_query_retriever
            ],
            weights=[0.5, 0.5]
        )
        
        return {
            'multi_query': multi_query_retriever,
            'compression': compression_retriever,
            'ensemble': ensemble_retriever
        }
    
    def setup_qa_chain(self, chain_type="stuff"):
        """Setup QA chain."""
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=self.retriever,
            return_source_documents=True
        )
        return self.qa_chain
    
    def setup_conversational_chain(self):
        """Setup conversational chain with memory."""
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=memory,
            return_source_documents=True
        )
        
        return chain
    
    def query(self, question: str) -> dict:
        """Query the RAG system."""
        if self.qa_chain is None:
            self.setup_qa_chain()
        
        result = self.qa_chain({"query": question})
        
        return {
            'question': question,
            'answer': result['result'],
            'source_documents': result['source_documents']
        }
    
    def initialize_llm(self, model_name="microsoft/phi-2", use_huggingface=True):
        """Initialize LLM."""
        if use_huggingface:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.7
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
        else:
            self.llm = OpenAI(temperature=0)
        
        return self.llm
```

### 9.4 LangChain Advanced Patterns

```text
ADVANCED LANGCHAIN PATTERNS:
═══════════════════════════════════════════════════════════════

1. SELF-QUERY RETRIEVER
   ──────────────────────
   Extract metadata filters from query.
   
   from langchain.retrievers.self_query.base import SelfQueryRetriever
   
   metadata_field_info = [
       AttributeInfo(
           name="source",
           description="The source of the document",
           type="string"
       ),
       AttributeInfo(
           name="date",
           description="The publication date",
           type="date"
       )
   ]
   
   retriever = SelfQueryRetriever.from_llm(
       llm=llm,
       vectorstore=vectorstore,
       document_contents="...",
       metadata_field_info=metadata_field_info
   )

2. PARENT DOCUMENT RETRIEVER
   ───────────────────────────
   Retrieve small chunks, return parent documents.
   
   from langchain.retrievers import ParentDocumentRetriever
   
   parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
   child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
   
   retriever = ParentDocumentRetriever(
       vectorstore=vectorstore,
       docstore=docstore,
       child_splitter=child_splitter,
       parent_splitter=parent_splitter
   )

3. MULTI-VECTOR RETRIEVER
   ────────────────────────
   Multiple representations for each document.
   
   from langchain.retrievers.multi_vector import MultiVectorRetriever
   
   retriever = MultiVectorRetriever(
       vectorstore=vectorstore,
       docstore=docstore,
       id_key="doc_id"
   )

4. RERANKER INTEGRATION
   ──────────────────────
   Add reranking to retrieval.
   
   from langchain.retrievers.document_compressors import CrossEncoderReranker
   from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
   from sentence_transformers import CrossEncoder
   
   model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
   compressor = CrossEncoderReranker(model=model, top_n=5)
   retriever = ContextualCompressionRetriever(
       base_compressor=compressor,
       base_retriever=base_retriever
   )

5. HYBRID RETRIEVER WITH LANGCHAIN
   ────────────────────────────────
   Combine dense and sparse retrieval.
   
   from langchain.retrievers import EnsembleRetriever
   
   dense_retriever = vectorstore.as_retriever()
   bm25_retriever = BM25Retriever.from_documents(documents)
   
   ensemble_retriever = EnsembleRetriever(
       retrievers=[dense_retriever, bm25_retriever],
       weights=[0.5, 0.5]
   )
```

---

## 10. LlamaIndex for RAG

### 10.1 LlamaIndex Overview

LlamaIndex is a data framework for building RAG applications.

```text
LLAMAINDEX FOR RAG:
═══════════════════════════════════════════════════════════════

CORE CONCEPTS:
────────────────────────────────────────────────────────────
1. Document: Text data with metadata
2. Node: Processed document chunk
3. Index: Data structure for retrieval
4. Retriever: Query interface
5. Query Engine: End-to-end RAG
6. Agent: Multi-step reasoning

KEY ADVANTAGES:
────────────────────────────────────────────────────────────
- Rich document parsing
- Multiple indexing structures
- Customizable retrieval
- Integration with many LLMs
- Agentic workflows
- Structured data support

BASIC USAGE:
────────────────────────────────────────────────────────────
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM

# 1. Load documents
documents = SimpleDirectoryReader("data").load_data()

# 2. Create index
index = VectorStoreIndex.from_documents(documents)

# 3. Query
query_engine = index.as_query_engine()
response = query_engine.query("What is the document about?")
```

### 10.2 LlamaIndex Components

```text
LLAMAINDEX COMPONENTS:
═══════════════════════════════════════════════════════════════

DOCUMENT LOADERS:
────────────────────────────────────────────────────────────
- SimpleDirectoryReader: Load from directory
- PDFReader: PDF files
- MarkdownReader: Markdown files
- CSVReader: CSV files
- JSONReader: JSON files
- GoogleDocsReader: Google Docs
- NotionReader: Notion pages
- WebPageReader: Web pages

NODE PARSERS:
────────────────────────────────────────────────────────────
- SentenceSplitter: Sentence-based
- TokenTextSplitter: Token-based
- HierarchicalNodeParser: Hierarchical chunks
- CodeSplitter: Code-aware
- MarkdownNodeParser: Markdown structure

INDICES:
────────────────────────────────────────────────────────────
- VectorStoreIndex: Dense retrieval
- SummaryIndex: Summarization
- KeywordTableIndex: Keyword-based
- TreeIndex: Tree-structured
- DocumentSummaryIndex: Document summaries
- ComposableGraph: Combine indices

RETRIEVERS:
────────────────────────────────────────────────────────────
- VectorIndexRetriever: Vector retrieval
- QueryFusionRetriever: Combine queries
- BM25Retriever: Keyword retrieval
- RouterRetriever: Route to specific index
- AutoMergingRetriever: Merge chunks
- RecursiveRetriever: Recursive retrieval

QUERY ENGINES:
────────────────────────────────────────────────────────────
- RetrieverQueryEngine: Basic RAG
- CitationQueryEngine: With citations
- SubQuestionQueryEngine: Decompose queries
- MultiStepQueryEngine: Multi-step reasoning
- RouterQueryEngine: Route queries
- TransformQueryEngine: Query transformation

POST-PROCESSORS:
────────────────────────────────────────────────────────────
- MetadataReplacementPostProcessor: Replace metadata
- SimilarityPostprocessor: Filter by similarity
- KeywordNodePostprocessor: Filter by keyword
- CohereRerank: Rerank with Cohere
- SentenceTransformerRerank: Cross-encoder rerank
- LLMRerank: LLM-based rerank

RESPONSE SYNTHESIZERS:
────────────────────────────────────────────────────────────
- CompactAndRefine: Concise with refinement
- TreeSummarize: Tree-based summarization
- Refine: Iterative refinement
- SimpleSummarize: Simple summarization
- NoText: No generation, just retrieval
- Custom: Custom synthesis
```

### 10.3 LlamaIndex RAG Implementation

```python
# LLAMAINDEX RAG IMPLEMENTATION:
# ============================================================

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex,
    KeywordTableIndex,
    DocumentSummaryIndex
)
from llama_index.core.node_parser import (
    SentenceSplitter,
    HierarchicalNodeParser,
    SimpleNodeParser
)
from llama_index.core.postprocessor import (
    SentenceTransformerRerank,
    SimilarityPostprocessor,
    LLMRerank
)
from llama_index.core.retrievers import (
    BM25Retriever,
    RouterRetriever,
    QueryFusionRetriever
)
from llama_index.core.query_engine import (
    RouterQueryEngine,
    SubQuestionQueryEngine,
    CitationQueryEngine
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
import chromadb

class LlamaIndexRAG:
    def __init__(self):
        self.documents = None
        self.nodes = None
        self.index = None
        self.query_engine = None
        
        # Initialize embedding
        self.embed_model = HuggingFaceEmbedding(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize LLM (set later)
        self.llm = None
        
        # Chroma setup
        self.db = chromadb.PersistentClient(path="./llama_chroma")
        self.chroma_collection = self.db.get_or_create_collection("documents")
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
    
    def load_documents(self, directory_path: str):
        """Load documents from directory."""
        self.documents = SimpleDirectoryReader(directory_path).load_data()
        return self.documents
    
    def process_documents(self, chunk_size=512, chunk_overlap=50):
        """Process documents into nodes."""
        parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.nodes = parser.get_nodes_from_documents(self.documents)
        return self.nodes
    
    def create_vector_index(self):
        """Create vector store index."""
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        self.index = VectorStoreIndex(
            nodes=self.nodes,
            storage_context=storage_context,
            embed_model=self.embed_model
        )
        
        return self.index
    
    def create_summary_index(self):
        """Create summary index."""
        return SummaryIndex(nodes=self.nodes)
    
    def create_keyword_index(self):
        """Create keyword table index."""
        return KeywordTableIndex(nodes=self.nodes)
    
    def setup_query_engine(self, similarity_top_k=4, rerank_top_n=2):
        """Setup query engine."""
        # Basic retriever
        retriever = self.index.as_retriever(
            similarity_top_k=similarity_top_k
        )
        
        # Add reranker
        reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=rerank_top_n
        )
        
        # Create query engine
        self.query_engine = self.index.as_query_engine(
            retriever=retriever,
            node_postprocessors=[reranker],
            similarity_top_k=similarity_top_k,
            response_mode="compact"
        )
        
        return self.query_engine
    
    def setup_citation_engine(self):
        """Setup citation query engine."""
        self.query_engine = CitationQueryEngine.from_args(
            index=self.index,
            similarity_top_k=3,
            citation_chunk_size=512
        )
        
        return self.query_engine
    
    def setup_subquestion_engine(self):
        """Setup sub-question query engine."""
        if self.llm is None:
            raise ValueError("LLM must be initialized first")
        
        self.query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=[
                QueryEngineTool(
                    query_engine=self.index.as_query_engine(),
                    metadata=ToolMetadata(
                        name="document_store",
                        description="Main document store"
                    )
                )
            ],
            llm=self.llm
        )
        
        return self.query_engine
    
    def setup_router_engine(self):
        """Setup router query engine."""
        vector_tool = QueryEngineTool(
            query_engine=self.index.as_query_engine(),
            metadata=ToolMetadata(
                name="vector_search",
                description="Semantic search over documents"
            )
        )
        
        summary_index = self.create_summary_index()
        summary_tool = QueryEngineTool(
            query_engine=summary_index.as_query_engine(),
            metadata=ToolMetadata(
                name="summary",
                description="Document summaries"
            )
        )
        
        self.query_engine = RouterQueryEngine.from_defaults(
            query_engine_tools=[vector_tool, summary_tool],
            llm=self.llm
        )
        
        return self.query_engine
    
    def setup_agent(self):
        """Setup ReAct agent."""
        tools = [
            QueryEngineTool(
                query_engine=self.index.as_query_engine(),
                metadata=ToolMetadata(
                    name="rag_tool",
                    description="Search documents for answers"
                )
            )
        ]
        
        self.agent = ReActAgent.from_tools(
            tools=tools,
            llm=self.llm,
            verbose=True
        )
        
        return self.agent
    
    def query(self, question: str) -> dict:
        """Query the RAG system."""
        if self.query_engine is None:
            self.setup_query_engine()
        
        response = self.query_engine.query(question)
        
        return {
            'question': question,
            'answer': str(response),
            'source_nodes': [n.node.text for n in response.source_nodes]
        }
    
    def agent_query(self, question: str) -> str:
        """Query using agent."""
        if self.agent is None:
            self.setup_agent()
        
        response = self.agent.chat(question)
        return str(response)
    
    def initialize_llm(self, model_name="microsoft/phi-2"):
        """Initialize LLM."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7
        )
        
        self.llm = HuggingFaceLLM(
            pipeline=pipe,
            tokenizer=tokenizer
        )
        
        return self.llm
```

### 10.4 LlamaIndex Advanced Features

```text
LLAMAINDEX ADVANCED FEATURES:
═══════════════════════════════════════════════════════════════

1. QUERY TRANSFORMATIONS
   ──────────────────────
   Transform queries before retrieval.
   
   from llama_index.core.indices.query.query_transform import HyDEQueryTransform
   
   hyde = HyDEQueryTransform(llm=llm)
   transformed_query = hyde.run(original_query)

2. STRUCTURED DATA
   ──────────────────
   Handle structured data sources.
   
   from llama_index.core import SQLDatabase
   from llama_index.core.indices.struct_store.sql_query import NLSQLTableQueryEngine
   
   sql_database = SQLDatabase(engine)
   query_engine = NLSQLTableQueryEngine(
       sql_database=sql_database,
       tables=["users", "orders"]
   )

3. OBJECT INDEXING
   ──────────────────
   Index objects with properties.
   
   from llama_index.core import ObjectIndex
   from llama_index.core.objects import ObjectNodeMapping
   
   object_index = ObjectIndex.from_objects(
       objects=my_objects,
       object_mapping=ObjectNodeMapping()
   )

4. AUTO MERGING
   ──────────────
   Auto-merge related nodes.
   
   from llama_index.core.retrievers import AutoMergingRetriever
   
   retriever = AutoMergingRetriever(
       index=index,
       similarity_top_k=10
   )

5. RECURSIVE RETRIEVAL
   ──────────────────────
   Recursively retrieve and query.
   
   from llama_index.core.retrievers import RecursiveRetriever
   
   retriever = RecursiveRetriever(
       root_id="root",
       retriever_dict={"node": node_retriever},
       query_engine_dict={"node": query_engine}
   )

6. LLAMAINDEX + LANGCHAIN INTEGRATION
   ────────────────────────────────────
   Use both frameworks together.
   
   from langchain.retrievers import BaseRetriever
   from llama_index.core import VectorStoreIndex
   
   class LlamaIndexRetriever(BaseRetriever):
       def _get_relevant_documents(self, query):
           engine = index.as_query_engine()
           response = engine.query(query)
           return response.source_nodes
```

---

## 11. Modern RAG Patterns

### 11.1 RAG Evolution

```text
RAG EVOLUTION PATH:
═══════════════════════════════════════════════════════════════

NAIVE RAG (2023):
────────────────────────────────────────────────────────────
Query → Retrieval → Context → LLM → Answer

SIMPLE RAG:
────────────────────────────────────────────────────────────
Query → Query Understanding → Retrieval → Rerank → Context → LLM → Answer

ADVANCED RAG:
────────────────────────────────────────────────────────────
Query → Query Enhancement → Hybrid Retrieval → Rerank → 
Context Compression → LLM → Grounding → Answer

AGENTIC RAG (2025-2026):
────────────────────────────────────────────────────────────
User → Agent → Plan → Retrieve → Evaluate → 
Refine → Retrieve Again → Synthesize → Verify → Answer

GRAPH RAG (2025-2026):
────────────────────────────────────────────────────────────
Query → Entity Extraction → Graph Retrieval → Context → LLM → Answer

MODERN 2026 RAG:
────────────────────────────────────────────────────────────
Multi-modal query processing
Agentic retrieval loops
Graph-based context
Structured + unstructured data
Self-correction
Verification
```

### 11.2 Graph RAG

```text
GRAPH RAG OVERVIEW:
═══════════════════════════════════════════════════════════════

ARCHITECTURE:
────────────────────────────────────────────────────────────
Documents
    ↓
Entity Extraction
    ↓
Relationship Extraction
    ↓
Knowledge Graph
    ↓
Graph Retrieval
    ↓
Context Assembly
    ↓
LLM Generation

ADVANTAGES:
────────────────────────────────────────────────────────────
- Handles complex relationships
- Better for multi-hop questions
- Redundant information reduced
- Structured reasoning
- Relationship-centric queries

IMPLEMENTATION:
────────────────────────────────────────────────────────────
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.storage.storage_context import StorageContext

graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
    max_triplets_per_chunk=10
)

query_engine = kg_index.as_query_engine()
response = query_engine.query("How does entity A relate to entity B?")
```

### 11.3 Agentic RAG

```text
AGENTIC RAG ARCHITECTURE:
═══════════════════════════════════════════════════════════════

USER QUERY
    ↓
AGENT
    ↓
Plan: What information is needed?
    ↓
Search Loop:
    1. Retrieve
    2. Analyze
    3. Missing information? → Go to 1
    4. Complete? → Continue
    ↓
Synthesize
    ↓
Verify against sources
    ↓
Answer

AGENT RAG COMPONENTS:
────────────────────────────────────────────────────────────
1. Planner: Decides what to retrieve
2. Retriever: Executes retrieval
3. Evaluator: Assesses retrieved information
4. Refiner: Refines queries based on results
5. Synthesizer: Generates final answer
6. Verifier: Checks grounding and accuracy

IMPLEMENTATION:
────────────────────────────────────────────────────────────
# Using LlamaIndex Agent
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool

tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="document_search",
            description="Search documents for information"
        )
    ),
    QueryEngineTool(
        query_engine=wiki_engine,
        metadata=ToolMetadata(
            name="wiki_search",
            description="Search Wikipedia for additional context"
        )
    )
]

agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    max_iterations=5
)

response = agent.chat("What is the latest research on RAG?")
```

### 11.4 Self-RAG and Corrective RAG

```text
SELF-RAG:
═══════════════════════════════════════════════════════════════

Self-RAG enables the model to reflect on its own retrieval and generation.

PROCESS:
────────────────────────────────────────────────────────────
1. Initial retrieval
2. Generate response
3. Self-assessment:
   - Is this accurate?
   - Is this relevant?
   - Is this complete?
4. If not, retrieve more
5. Revise response
6. Repeat if needed

REFLECTION TYPES:
────────────────────────────────────────────────────────────
- Is retrieval relevant?
- Is generation faithful to sources?
- Are there contradictions?
- Is information complete?

CORRECTIVE RAG:
────────────────────────────────────────────────────────────
1. Retrieve documents
2. Evaluate relevance
3. If insufficient → retrieve more
4. If contradictory → reconcile
5. Generate corrected answer
6. Verify accuracy

IMPLEMENTATION:
────────────────────────────────────────────────────────────
class SelfRAG:
    def __init__(self, retriever, llm, evaluator):
        self.retriever = retriever
        self.llm = llm
        self.evaluator = evaluator
        self.max_iterations = 3
    
    def query(self, question):
        iteration = 0
        context = []
        answer = None
        
        while iteration < self.max_iterations:
            # Retrieve
            docs = self.retriever.search(question)
            context.extend(docs)
            
            # Generate
            answer = self.llm.generate(question, context)
            
            # Evaluate
            evaluation = self.evaluator.evaluate(question, answer, context)
            
            if evaluation['complete'] and evaluation['accurate']:
                break
            
            # Refine question if needed
            if evaluation['missing_info']:
                question = self._refine_question(question, evaluation)
            
            iteration += 1
        
        return answer
```

### 11.5 Multimodal RAG

```text
MULTIMODAL RAG:
═══════════════════════════════════════════════════════════════

TEXT + IMAGE RAG:
────────────────────────────────────────────────────────────
1. Documents contain text and images
2. Text chunks: Standard embedding retrieval
3. Images: Image embedding retrieval
4. Combined retrieval results
5. Multimodal context for LLM

TEXT + TABLE RAG:
────────────────────────────────────────────────────────────
1. Extract tables as structured data
2. Index table rows with embeddings
3. Retrieve relevant rows
4. Include in context as structured format

FULL MULTIMODAL:
────────────────────────────────────────────────────────────
- Text
- Images (with captions)
- Tables (structured)
- Charts (with descriptions)
- Audio (transcribed)
- Video (frame descriptions)

IMPLEMENTATION:
────────────────────────────────────────────────────────────
# Using LangChain with multimodal embeddings
from langchain.embeddings import OpenCLIPEmbeddings
from langchain.vectorstores import MultiVectorRetriever

# Store text and image embeddings
embeddings_text = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings_image = OpenCLIPEmbeddings()

# Multi-vector retriever
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key="doc_id"
)

# Add text chunks
retriever.add_documents(text_chunks)

# Add image chunks
retriever.add_documents(image_chunks)
```

---

## 12. RAG Evaluation

### 12.1 Evaluation Dimensions

```text
RAG EVALUATION DIMENSIONS:
═══════════════════════════════════════════════════════════════

1. RETRIEVAL EVALUATION
   ──────────────────────
   - Precision@K: Relevant documents in top-K
   - Recall@K: Relevant documents found in top-K
   - MRR: Mean Reciprocal Rank of first relevant
   - NDCG: Normalized Discounted Cumulative Gain
   - MAP: Mean Average Precision

2. GENERATION EVALUATION
   ──────────────────────
   - Faithfulness: Factual support from sources
   - Answer Relevance: Relevance to question
   - Answer Completeness: Coverage of information
   - Coherence: Logical flow and readability
   - Hallucination Rate: False information

3. END-TO-END EVALUATION
   ──────────────────────
   - Correctness: Answer matches ground truth
   - Completeness: All expected information present
   - Helpfulness: Usefulness for user
   - Safety: No harmful content

4. PERFORMANCE EVALUATION
   ──────────────────────
   - Latency: Time to answer
   - Throughput: Queries per second
   - Cost: Per-query cost
   - Scalability: Performance with load

5. ROBUSTNESS EVALUATION
   ──────────────────────
   - Edge cases: Unusual queries
   - Adversarial: Attempts to break system
   - Distribution shift: Different query types
   - Graceful degradation: Partial failures
```

### 12.2 Retrieval Evaluation Metrics

```text
RETRIEVAL METRICS DETAILED:
═══════════════════════════════════════════════════════════════

PRECISION AT K (P@K):
────────────────────────────────────────────────────────────
P@K = (Relevant documents in top-K) / K

Example: Top-5 has 3 relevant → P@5 = 0.6
Interpretation: How many of top-K are relevant

RECALL AT K (R@K):
────────────────────────────────────────────────────────────
R@K = (Relevant documents in top-K) / (Total relevant documents)

Example: 3 relevant found out of 10 total → R@5 = 0.3
Interpretation: How many relevant documents were found

MEAN RECIPROCAL RANK (MRR):
────────────────────────────────────────────────────────────
MRR = 1/N × Σ(1/rank_i)

Example: First relevant at rank 2 → 1/2 = 0.5
Interpretation: How good is the top result

NDCG (Normalized Discounted Cumulative Gain):
────────────────────────────────────────────────────────────
DCG = Σ relevance_i / log2(i + 1)
NDCG = DCG / IDCG (Ideal DCG)

Example: Binary relevance with positions [1,0,1,0,1]
NDCG at 5 = 0.78
Interpretation: Quality of ranking with position weighting

MAP (Mean Average Precision):
────────────────────────────────────────────────────────────
AP = Average of precision at each relevant document
MAP = Mean AP across queries

Example: AP = (1/3 + 2/4 + 3/5) / 3 = 0.73
Interpretation: Overall retrieval quality

RECOMMENDED TARGETS:
────────────────────────────────────────────────────────────
- P@5: > 0.70
- R@5: > 0.60
- MRR: > 0.70
- NDCG: > 0.75
- MAP: > 0.70
```

### 12.3 Generation Evaluation Metrics

```text
GENERATION METRICS:
═══════════════════════════════════════════════════════════════

FAITHFULNESS:
────────────────────────────────────────────────────────────
Definition: Answer is factually supported by context

Measurement:
- LLM-as-judge: "Is this answer supported by the context?"
- Fact extraction: Compare facts in answer vs context
- Citation checking: Each claim has a source

Target: > 0.90

ANSWER RELEVANCE:
────────────────────────────────────────────────────────────
Definition: Answer addresses the question

Measurement:
- Semantic similarity to ideal answer
- LLM-as-judge: "Does this answer the question?"
- Contains all expected information

Target: > 0.85

ANSWER COMPLETENESS:
────────────────────────────────────────────────────────────
Definition: All needed information is provided

Measurement:
- Compare to reference answer
- Count key points covered
- LLM-as-judge: "What information is missing?"

Target: > 0.80

HALLUCINATION RATE:
────────────────────────────────────────────────────────────
Definition: Percentage of claims not supported by context

Measurement:
- Claim extraction from answer
- Cross-reference with context
- LLM-as-judge: "Is this claim supported?"

Target: < 0.05

COHERENCE:
────────────────────────────────────────────────────────────
Definition: Answer is well-structured and flows logically

Measurement:
- LLM-as-judge: "Is this answer coherent?"
- Grammar and readability scores

Target: > 0.90
```

### 12.4 Evaluation Implementation

```python
# RAG EVALUATION IMPLEMENTATION:
# ============================================================

from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

class RAGEvaluator:
    def __init__(self):
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def precision_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate Precision@K."""
        top_k = retrieved[:k]
        relevant_set = set(relevant)
        relevant_retrieved = sum(1 for doc in top_k if doc in relevant_set)
        return relevant_retrieved / k
    
    def recall_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate Recall@K."""
        top_k = retrieved[:k]
        relevant_set = set(relevant)
        relevant_retrieved = sum(1 for doc in top_k if doc in relevant_set)
        return relevant_retrieved / len(relevant_set) if relevant_set else 0
    
    def mrr(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        relevant_set = set(relevant)
        for i, doc in enumerate(retrieved):
            if doc in relevant_set:
                return 1.0 / (i + 1)
        return 0.0
    
    def ndcg_at_k(self, retrieved: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """Calculate NDCG@K."""
        # Get scores for top-K
        top_k_scores = [relevance_scores.get(doc, 0) for doc in retrieved[:k]]
        
        # Calculate DCG
        dcg = sum(score / np.log2(i + 2) for i, score in enumerate(top_k_scores))
        
        # Calculate IDCG (ideal ordering)
        sorted_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(score / np.log2(i + 2) for i, score in enumerate(sorted_scores))
        
        return dcg / idcg if idcg > 0 else 0
    
    def faithfulness(self, answer: str, context: List[str], llm) -> float:
        """Evaluate faithfulness using LLM."""
        prompt = f"""On a scale of 0 to 1, rate how faithful this answer is to the context.
        
        Context: {context}
        
        Answer: {answer}
        
        Faithfulness score (0-1):"""
        
        response = llm.generate(prompt)
        try:
            score = float(response.strip())
            return min(1.0, max(0.0, score))
        except:
            return 0.5
    
    def answer_relevance(self, question: str, answer: str, llm) -> float:
        """Evaluate answer relevance."""
        prompt = f"""Rate how well this answer addresses the question (0-1).
        
        Question: {question}
        
        Answer: {answer}
        
        Relevance score (0-1):"""
        
        response = llm.generate(prompt)
        try:
            score = float(response.strip())
            return min(1.0, max(0.0, score))
        except:
            return 0.5
    
    def evaluate_retrieval(self, query: str, retrieved: List[str], relevant: List[str]) -> Dict[str, float]:
        """Comprehensive retrieval evaluation."""
        return {
            'precision_at_5': self.precision_at_k(retrieved, relevant, 5),
            'precision_at_10': self.precision_at_k(retrieved, relevant, 10),
            'recall_at_5': self.recall_at_k(retrieved, relevant, 5),
            'recall_at_10': self.recall_at_k(retrieved, relevant, 10),
            'mrr': self.mrr(retrieved, relevant)
        }
    
    def evaluate_generation(self, question: str, answer: str, context: List[str], llm) -> Dict[str, float]:
        """Comprehensive generation evaluation."""
        return {
            'faithfulness': self.faithfulness(answer, context, llm),
            'answer_relevance': self.answer_relevance(question, answer, llm),
        }
    
    def evaluate_end_to_end(self, 
                           questions: List[str],
                           expected_answers: List[str],
                           rag_system,
                           llm) -> Dict[str, Any]:
        """End-to-end RAG system evaluation."""
        results = []
        
        for q, expected in zip(questions, expected_answers):
            # Get RAG response
            response = rag_system.query(q)
            answer = response['answer']
            context = response.get('context_docs', [])
            
            # Evaluate retrieval
            relevant_docs = self._get_relevant_docs(q, expected)
            retrieved_docs = response.get('retrieved_docs', [])
            
            # Evaluate
            retrieval_metrics = self.evaluate_retrieval(q, retrieved_docs, relevant_docs)
            generation_metrics = self.evaluate_generation(q, answer, context, llm)
            
            results.append({
                'question': q,
                'answer': answer,
                'expected': expected,
                'retrieval': retrieval_metrics,
                'generation': generation_metrics
            })
        
        # Aggregate metrics
        aggregated = self._aggregate_results(results)
        
        return {
            'results': results,
            'aggregated': aggregated
        }
    
    def _get_relevant_docs(self, question: str, expected_answer: str) -> List[str]:
        """Get relevant documents for a question."""
        # In practice, this would come from annotations
        # Here we use a placeholder
        return []
    
    def _aggregate_results(self, results: List[Dict]) -> Dict[str, float]:
        """Aggregate results across queries."""
        metrics = {}
        
        for key in ['retrieval', 'generation']:
            for subkey in results[0][key].keys():
                values = [r[key][subkey] for r in results if subkey in r[key]]
                metrics[f"{key}_{subkey}"] = np.mean(values)
        
        return metrics
```

---

## 13. RAG Optimization

### 13.1 Optimization Dimensions

```text
RAG OPTIMIZATION DIMENSIONS:
═══════════════════════════════════════════════════════════════

1. RETRIEVAL OPTIMIZATION
   ──────────────────────
   - Better chunking strategies
   - Improved embedding models
   - Hybrid retrieval
   - Reranking
   - Query enhancement
   - Metadata filtering

2. INFRASTRUCTURE OPTIMIZATION
   ───────────────────────────
   - Index tuning (IVF, HNSW parameters)
   - Caching
   - Batch processing
   - GPU acceleration
   - Distributed retrieval

3. PROMPT OPTIMIZATION
   ──────────────────────
   - Better system prompts
   - Context formatting
   - Instruction clarity
   - Few-shot examples
   - Citation formats

4. QUALITY OPTIMIZATION
   ──────────────────────
   - Data cleaning
   - Duplicate removal
   - Metadata enrichment
   - Quality filtering
   - Relevance feedback

5. PERFORMANCE OPTIMIZATION
   ─────────────────────────
   - Parallel processing
   - Asynchronous operations
   - Connection pooling
   - Resource scaling
   - Load balancing

6. COST OPTIMIZATION
   ────────────────────
   - Smaller embedding models
   - Caching frequently used queries
   - Batch embedding generation
   - Model distillation
   - Efficient indexing
```

### 13.2 Retrieval Optimization Techniques

```text
RETRIEVAL OPTIMIZATION DETAILED:
═══════════════════════════════════════════════════════════════

CHUNKING OPTIMIZATION:
────────────────────────────────────────────────────────────
1. Adaptive chunking based on document structure
2. Semantic chunking using embeddings
3. Parent-child chunking
4. Fine-tune chunk size per document type
5. Add overlap for context preservation

QUERY ENHANCEMENT:
────────────────────────────────────────────────────────────
1. Query expansion with synonyms
2. Multi-query generation
3. HyDE (Hypothetical Document Embeddings)
4. Query decomposition for complex questions
5. Step-back prompting

INDEX OPTIMIZATION:
────────────────────────────────────────────────────────────
1. HNSW: Increase M (16-64) and ef_construction (100-500)
2. IVF: Adjust nlist (100 per million vectors)
3. PQ: Optimize sub-vector count (8-16)
4. Monitor memory usage
5. Rebuild indices periodically

HYBRID RETRIEVAL:
────────────────────────────────────────────────────────────
1. Combine dense and sparse retrieval
2. Weight tuning (0.3-0.7)
3. Rank fusion methods
4. Reciprocal Rank Fusion (RRF)
5. Dynamic weighting based on query

MULTI-STAGE RETRIEVAL:
────────────────────────────────────────────────────────────
1. Stage 1: Fast retrieval (IVF)
2. Stage 2: Reranking (Cross-encoder)
3. Stage 3: Filtering
4. Stage 4: Context compression
```

### 13.3 Quality Optimization

```text
QUALITY OPTIMIZATION TECHNIQUES:
═══════════════════════════════════════════════════════════════

DATA QUALITY:
────────────────────────────────────────────────────────────
1. Remove duplicate documents
2. Filter low-quality content
3. Fix broken formatting
4. Enrich metadata
5. Add document summaries

GROUNDING IMPROVEMENT:
────────────────────────────────────────────────────────────
1. Better citation prompts
2. Source verification
3. Fact extraction and validation
4. Confidence scoring
5. Uncertainty expression

HALLUCINATION REDUCTION:
────────────────────────────────────────────────────────────
1. Clear instructions to only use context
2. Citation requirements
3. Fact verification
4. Multi-source cross-checking
5. Confidence thresholds

HANDLING EDGE CASES:
────────────────────────────────────────────────────────────
1. No relevant context found → Refuse
2. Contradictory context → Acknowledge
3. Insufficient context → Ask clarifying question
4. Out-of-scope queries → Redirect
5. Ambiguous queries → Seek clarification

CONTEXT OPTIMIZATION:
────────────────────────────────────────────────────────────
1. Remove redundant information
2. Prioritize most relevant chunks
3. Format for readability
4. Include metadata with context
5. Compress long contexts
```

### 13.4 Performance Optimization

```text
PERFORMANCE OPTIMIZATION TECHNIQUES:
═══════════════════════════════════════════════════════════════

INDEX PERFORMANCE:
────────────────────────────────────────────────────────────
1. Use GPU for embedding generation
2. Use GPU for FAISS search
3. Optimize HNSW parameters
4. Use quantization for memory
5. Batch index operations

CACHING:
────────────────────────────────────────────────────────────
1. Query-level caching
2. Embedding caching
3. Result caching
4. KV cache for generation
5. Prefix caching

PARALLELIZATION:
────────────────────────────────────────────────────────────
1. Parallel document processing
2. Parallel embedding generation
3. Parallel retrieval
4. Parallel reranking
5. Parallel generation

BATCHING:
────────────────────────────────────────────────────────────
1. Batch embedding generation
2. Batch retrieval
3. Batch reranking
4. Batch generation
5. Batch indexing

ASYNC OPERATIONS:
────────────────────────────────────────────────────────────
1. Async embedding generation
2. Async retrieval
3. Async LLM calls
4. Non-blocking operations
5. Streaming responses

INFRASTRUCTURE:
────────────────────────────────────────────────────────────
1. Load balancing
2. Auto-scaling
3. Connection pooling
4. Resource monitoring
5. Read replicas for vector DB
```

### 13.5 Cost Optimization

```text
COST OPTIMIZATION TECHNIQUES:
═══════════════════════════════════════════════════════════════

MODEL SELECTION:
────────────────────────────────────────────────────────────
1. Use smaller embedding models (MiniLM vs large)
2. Use smaller LLM for generation
3. Distill models when possible
4. Consider open-source vs API
5. Use quantization

EMBEDDING OPTIMIZATION:
────────────────────────────────────────────────────────────
1. Cache embeddings
2. Batch generation
3. Use lower precision (FP16/INT8)
4. Compress embeddings (PCA, PQ)
5. Reuse embeddings

RETRIEVAL OPTIMIZATION:
────────────────────────────────────────────────────────────
1. Efficient index (IVFPQ)
2. Reduce nprobe
3. Reduce search K
4. Use filtering to reduce candidates
5. Implement query routing

LLM OPTIMIZATION:
────────────────────────────────────────────────────────────
1. Use smaller model for simple queries
2. Implement query routing
3. Use caching for repeated queries
4. Stream responses
5. Optimize prompt length

OPERATIONAL OPTIMIZATION:
────────────────────────────────────────────────────────────
1. Batch operations
2. Schedule expensive operations
3. Use spot instances for batch
4. Monitor and optimize resource usage
5. Implement auto-scaling
```

---

## 14. Complete Project: RAG-Based Document Q&A System

### 14.1 Project Overview

**Goal:** Build a complete RAG-based document Q&A system for Computer Science documents.

```text
PROJECT REQUIREMENTS:
═══════════════════════════════════════════════════════════════

HARDWARE:
- T4 GPU (Kaggle/Colab) or better
- 16GB+ RAM
- 50GB+ storage for documents

SOFTWARE:
- Python 3.9+
- PyTorch 2.0+
- Sentence Transformers
- ChromaDB or Qdrant
- LangChain or LlamaIndex
- Gradio for UI
- Hugging Face Transformers

DATA:
- CS research papers (PDF)
- Documentation
- Books (CS)
- Stack Overflow data

MODELS:
- Embedding: all-MiniLM-L6-v2 or BGE
- LLM: Phi-2 or Mistral-7B-Instruct
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2

TIME:
- Setup: 1-2 hours
- Indexing: 2-3 hours
- Development: 3-4 hours
- Testing: 2-3 hours
- Deployment: 1-2 hours
```

### 14.2 Project Architecture

```text
PROJECT ARCHITECTURE:
═══════════════════════════════════════════════════════════════

DOCUMENT INGESTION:
────────────────────────────────────────────────────────────
PDF Papers
    ↓
PDF Loader
    ↓
Text Extraction
    ↓
Cleaning
    ↓
Chunking (512 tokens, 25% overlap)
    ↓
Embedding (all-MiniLM-L6-v2)
    ↓
Vector Store (ChromaDB)
    ↓
BM25 Index (for hybrid)

QUERY PIPELINE:
────────────────────────────────────────────────────────────
User Question
    ↓
Query Understanding
    ├── Query type
    ├── Entities
    └── Enhancement
    ↓
Hybrid Retrieval
    ├── Dense (vector)
    └── Sparse (BM25)
    ↓
Reranking (Cross-encoder)
    ↓
Context Assembly
    ├── Merge
    ├── Compress
    └── Format
    ↓
LLM Generation (Phi-2)
    ↓
Citation & Sources
    ↓
Answer

UI:
────────────────────────────────────────────────────────────
Gradio Chat Interface
- Question input
- Answer display
- Source display
- Confidence
```

### 14.3 Implementation Steps

```text
IMPLEMENTATION STEPS:
═══════════════════════════════════════════════════════════════

DAY 1: SETUP AND DATA PREPARATION
────────────────────────────────────────────────────────────

1. Install dependencies
2. Download CS research papers
3. Set up document loader
4. Extract text from PDFs
5. Clean extracted text
6. Prepare metadata
7. Test on sample documents

DAY 2: INDEXING PIPELINE
────────────────────────────────────────────────────────────

1. Set up embedding model
2. Configure chunking
3. Set up vector database
4. Batch process documents
5. Build BM25 index
6. Validate indexing
7. Test retrieval

DAY 3: RETRIEVAL PIPELINE
────────────────────────────────────────────────────────────

1. Implement query understanding
2. Implement dense retrieval
3. Implement sparse retrieval
4. Implement hybrid retrieval
5. Add reranking
6. Test retrieval quality
7. Optimize parameters

DAY 4: GENERATION PIPELINE
────────────────────────────────────────────────────────────

1. Load LLM (Phi-2)
2. Set up generation
3. Implement context assembly
4. Add prompt engineering
5. Add citation support
6. Test generation quality
7. Validate grounding

DAY 5: UI AND DEPLOYMENT
────────────────────────────────────────────────────────────

1. Build Gradio UI
2. Add chat interface
3. Show sources
4. Add confidence indicators
5. Deploy with Ngrok
6. Test end-to-end
7. Document results
```

### 14.4 Complete Implementation

```python
# COMPLETE RAG DOCUMENT Q&A SYSTEM:
# ============================================================

# PART 1: DEPENDENCIES
# ============================================================
"""
pip install transformers torch sentence-transformers chromadb
pip install langchain llama-index gradio pdfplumber
pip install rank-bm25 qdrant-client
pip install accelerate bitsandbytes
"""

# PART 2: CONFIGURATION
# ============================================================
class Config:
    # Paths
    DATA_DIR = "./data"
    VECTOR_DB_DIR = "./chroma_db"
    
    # Models
    EMBED_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "microsoft/phi-2"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Chunking
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 128
    
    # Retrieval
    INITIAL_K = 50
    RERANK_K = 10
    FINAL_K = 5
    
    # Generation
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.7

# PART 3: DOCUMENT PROCESSING
# ============================================================
import os
import glob
import pdfplumber
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Document:
    text: str
    metadata: Dict[str, Any]
    doc_id: str

class DocumentProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.documents = []
    
    def load_pdf(self, pdf_path: str) -> Document:
        """Load and extract text from PDF."""
        text = ""
        metadata = {
            'source': os.path.basename(pdf_path),
            'path': pdf_path,
            'type': 'pdf'
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            text = ""
        
        doc_id = os.path.basename(pdf_path).replace('.pdf', '')
        
        return Document(
            text=text.strip(),
            metadata=metadata,
            doc_id=doc_id
        )
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """Load all PDFs from directory."""
        pdf_files = glob.glob(os.path.join(directory_path, "**/*.pdf"), recursive=True)
        
        documents = []
        for pdf_file in pdf_files:
            doc = self.load_pdf(pdf_file)
            if doc.text:
                documents.append(doc)
        
        self.documents = documents
        return documents
    
    def add_text_document(self, text: str, metadata: Dict[str, Any]) -> Document:
        """Add a text document."""
        doc_id = metadata.get('id', f"doc_{len(self.documents)}")
        doc = Document(text=text, metadata=metadata, doc_id=doc_id)
        self.documents.append(doc)
        return doc

# PART 4: CHUNKING
# ============================================================
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Chunker:
    def __init__(self, config: Config):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk_document(self, document: Document) -> List[Dict[str, Any]]:
        """Chunk a single document."""
        chunks = self.splitter.split_text(document.text)
        
        return [
            {
                'text': chunk,
                'metadata': {
                    **document.metadata,
                    'doc_id': document.doc_id,
                    'chunk_id': f"{document.doc_id}_{i}",
                    'chunk_index': i
                }
            }
            for i, chunk in enumerate(chunks)
        ]
    
    def chunk_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Chunk all documents."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

# PART 5: EMBEDDING
# ============================================================
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, config: Config):
        self.config = config
        self.model = SentenceTransformer(config.EMBED_MODEL)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings in batches."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings = self.embed(batch)
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

# PART 6: VECTOR STORE
# ============================================================
import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self, config: Config):
        self.config = config
        self.client = chromadb.PersistentClient(
            path=config.VECTOR_DB_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = None
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add chunks to vector store."""
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [chunk['metadata']['chunk_id'] for chunk in chunks]
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, query_embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        return [
            {
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'id': results['ids'][0][i]
            }
            for i in range(len(results['documents'][0]))
        ]

# PART 7: BM25 INDEX
# ============================================================
from rank_bm25 import BM25Okapi

class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.documents = []
    
    def build_index(self, chunks: List[Dict[str, Any]]):
        """Build BM25 index from chunks."""
        self.documents = [chunk['text'] for chunk in chunks]
        tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Search BM25 index."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-k:][::-1]
        
        return [
            {
                'text': self.documents[idx],
                'score': float(scores[idx]),
                'index': idx
            }
            for idx in top_indices
        ]

# PART 8: HYBRID RETRIEVAL
# ============================================================
class HybridRetriever:
    def __init__(self, config: Config, vector_store: VectorStore, bm25_index: BM25Index):
        self.config = config
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedder = Embedder(config)
        self.weight_dense = 0.5
        self.weight_sparse = 0.5
    
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Hybrid search combining dense and sparse retrieval."""
        # Dense retrieval
        query_embedding = self.embedder.embed([query])[0]
        dense_results = self.vector_store.search(query_embedding, k=50)
        
        # Sparse retrieval
        sparse_results = self.bm25_index.search(query, k=50)
        
        # Normalize and combine
        dense_map = {}
        for r in dense_results:
            dense_map[r['text']] = {
                'dense_score': 1.0 - (r['distance'] / 2),  # Convert distance to similarity
                'metadata': r['metadata']
            }
        
        sparse_map = {}
        for r in sparse_results:
            # Normalize BM25 score (max score = 1)
            normalized_score = r['score'] / (max(sparse_results, key=lambda x: x['score'])['score'] + 1e-8)
            sparse_map[r['text']] = {
                'sparse_score': normalized_score
            }
        
        # Combine
        combined = {}
        all_texts = set(dense_map.keys()) | set(sparse_map.keys())
        
        for text in all_texts:
            combined_score = 0
            metadata = {}
            
            if text in dense_map:
                combined_score += self.weight_dense * dense_map[text]['dense_score']
                metadata = dense_map[text]['metadata']
            
            if text in sparse_map:
                combined_score += self.weight_sparse * sparse_map[text]['sparse_score']
            
            combined[text] = {
                'combined_score': combined_score,
                'metadata': metadata
            }
        
        # Sort and return top-k
        sorted_results = sorted(
            combined.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )[:k]
        
        return [
            {
                'text': text,
                'score': data['combined_score'],
                'metadata': data['metadata']
            }
            for text, data in sorted_results
        ]

# PART 9: RERANKING
# ============================================================
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, config: Config):
        self.config = config
        self.model = CrossEncoder(config.RERANKER_MODEL)
    
    def rerank(self, query: str, documents: List[str], top_k: int) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder."""
        if not documents:
            return []
        
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        
        # Sort by score
        sorted_results = sorted(
            zip(documents, scores.tolist() if hasattr(scores, 'tolist') else scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {'text': doc, 'score': score}
            for doc, score in sorted_results[:top_k]
        ]

# PART 10: GENERATION
# ============================================================
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class Generator:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the LLM."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.LLM_MODEL)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.LLM_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                temperature=self.config.TEMPERATURE,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def generate(self, context: str, question: str) -> str:
        """Generate answer using LLM."""
        if self.model is None:
            return self._fallback_generation(context, question)
        
        prompt = self._build_prompt(context, question)
        
        try:
            response = self.pipeline(prompt)[0]['generated_text']
            # Extract only the new generation
            answer = response[len(prompt):].strip()
            return answer
        except Exception as e:
            print(f"Generation error: {e}")
            return self._fallback_generation(context, question)
    
    def _build_prompt(self, context: str, question: str) -> str:
        """Build the generation prompt."""
        return f"""You are a helpful assistant. Answer the question based only on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the context
- If the context doesn't contain the information, say so
- Cite the relevant parts of the context
- Be concise and clear
- Include sources when possible

Answer:"""
    
    def _fallback_generation(self, context: str, question: str) -> str:
        """Fallback generation without LLM."""
        # Simple fallback that extracts the most relevant sentence
        # In production, this would be improved
        return f"Based on the provided context, here is information related to: {question}"

# PART 11: RAG SYSTEM
# ============================================================
class RAGSystem:
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize components
        self.processor = DocumentProcessor(config)
        self.chunker = Chunker(config)
        self.embedder = Embedder(config)
        self.vector_store = VectorStore(config)
        self.bm25_index = BM25Index()
        self.hybrid_retriever = None
        self.reranker = Reranker(config)
        self.generator = Generator(config)
        
        # State
        self.is_indexed = False
    
    def index_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Index documents into the system."""
        # Chunk documents
        chunks = self.chunker.chunk_documents(documents)
        
        # Generate embeddings
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.embed_batch(texts)
        
        # Add to vector store
        self.vector_store.add_chunks(chunks, embeddings)
        
        # Build BM25 index
        self.bm25_index.build_index(chunks)
        
        # Initialize hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            self.config,
            self.vector_store,
            self.bm25_index
        )
        
        self.is_indexed = True
        
        return {
            'num_documents': len(documents),
            'num_chunks': len(chunks),
            'embedding_dimension': self.embedder.dimension
        }
    
    def index_directory(self, directory_path: str) -> Dict[str, Any]:
        """Index all documents in a directory."""
        documents = self.processor.load_directory(directory_path)
        return self.index_documents(documents)
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system."""
        if not self.is_indexed:
            return {
                'error': 'No documents indexed. Please index documents first.'
            }
        
        # 1. Retrieve
        initial_results = self.hybrid_retriever.search(
            question,
            k=self.config.INITIAL_K
        )
        initial_docs = [r['text'] for r in initial_results]
        
        # 2. Rerank
        reranked_results = self.reranker.rerank(
            question,
            initial_docs,
            top_k=self.config.RERANK_K
        )
        
        # 3. Context assembly
        context_docs = [r['text'] for r in reranked_results]
        context = "\n\n".join(context_docs)
        
        # 4. Generate answer
        answer = self.generator.generate(context, question)
        
        # 5. Prepare response
        return {
            'question': question,
            'answer': answer,
            'sources': reranked_results,
            'context': context_docs,
            'num_retrieved': len(initial_results),
            'num_reranked': len(reranked_results)
        }

# PART 12: UI
# ============================================================
import gradio as gr

def create_ui(rag_system: RAGSystem):
    """Create Gradio UI for the RAG system."""
    
    def respond(message, history):
        """Handle user message."""
        if not message:
            return "", history
        
        # Query the RAG system
        response = rag_system.query(message)
        
        if 'error' in response:
            answer = f"Error: {response['error']}"
            sources = []
        else:
            answer = response['answer']
            sources = [f"Source {i+1}: {s['text'][:200]}..." for i, s in enumerate(response['sources'])]
        
        # Update history
        history.append((message, answer))
        
        # Build source display
        source_text = "\n\n---\n\n**Sources:**\n" + "\n\n".join(sources) if sources else ""
        
        return "", history, source_text
    
    with gr.Blocks(title="RAG Document Q&A System") as demo:
        gr.Markdown("""
        # 📚 RAG Document Q&A System
        
        Ask questions about your CS documents. The system will search through indexed documents 
        and provide answers with sources.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=400
                )
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask a question about the documents...",
                    lines=2
                )
                send_btn = gr.Button("Send", variant="primary")
            
            with gr.Column(scale=1):
                sources_display = gr.Textbox(
                    label="Sources",
                    lines=10,
                    interactive=False
                )
        
        # Status
        status_text = gr.Markdown(
            f"System Status: {'Ready' if rag_system.is_indexed else 'Not Indexed'}"
        )
        
        # Event handlers
        send_btn.click(
            respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, sources_display]
        )
        msg.submit(
            respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, sources_display]
        )
    
    return demo

# PART 13: MAIN APPLICATION
# ============================================================
def main():
    # Initialize configuration
    config = Config()
    
    # Create RAG system
    rag_system = RAGSystem(config)
    
    # Index documents if available
    if os.path.exists(config.DATA_DIR):
        print(f"Indexing documents from {config.DATA_DIR}...")
        result = rag_system.index_directory(config.DATA_DIR)
        print(f"Indexed {result['num_documents']} documents, {result['num_chunks']} chunks")
    
    # Create UI
    demo = create_ui(rag_system)
    
    # Launch
    demo.launch(share=True)
    
    # For production, use:
    # demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
```

### 14.5 Testing and Evaluation

```python
# TESTING AND EVALUATION SCRIPT:
# ============================================================

class RAGTestSuite:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.test_queries = [
            {
                'question': "What is the transformer architecture?",
                'expected_topics': ['attention', 'encoder', 'decoder', 'neural network']
            },
            {
                'question': "How does gradient descent work?",
                'expected_topics': ['gradient', 'loss', 'update', 'learning rate']
            },
            {
                'question': "What is overfitting in machine learning?",
                'expected_topics': ['overfitting', 'generalization', 'variance', 'bias']
            }
        ]
    
    def test_query(self, question: str) -> Dict[str, Any]:
        """Test a single query."""
        response = self.rag_system.query(question)
        
        # Evaluate metrics
        metrics = {
            'has_answer': bool(response.get('answer', '')),
            'answer_length': len(response.get('answer', '')),
            'num_sources': len(response.get('sources', [])),
            'success': 'error' not in response
        }
        
        return {
            'question': question,
            'response': response,
            'metrics': metrics
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests."""
        results = []
        for test in self.test_queries:
            result = self.test_query(test['question'])
            results.append(result)
        
        # Aggregate metrics
        total_success = sum(1 for r in results if r['metrics']['success'])
        avg_sources = sum(r['metrics']['num_sources'] for r in results) / len(results)
        
        return {
            'results': results,
            'summary': {
                'total_queries': len(results),
                'successful': total_success,
                'success_rate': total_success / len(results),
                'avg_sources': avg_sources
            }
        }

# Usage
if __name__ == "__main__":
    config = Config()
    rag_system = RAGSystem(config)
    
    # Index documents if available
    if os.path.exists(config.DATA_DIR):
        rag_system.index_directory(config.DATA_DIR)
    
    # Run tests
    test_suite = RAGTestSuite(rag_system)
    results = test_suite.run_all_tests()
    print(f"Test Results: {results['summary']}")
```

### 14.6 Expected Results

```text
EXPECTED RESULTS:
═══════════════════════════════════════════════════════════════

PERFORMANCE METRICS:
────────────────────────────────────────────────────────────
- Retrieval Precision@5: > 0.70
- Retrieval Recall@5: > 0.60
- MRR: > 0.70
- Faithfulness: > 0.85
- Answer Relevance: > 0.80
- Latency: 3-5 seconds per query

INDEXING SCALABILITY:
────────────────────────────────────────────────────────────
- 100 documents: 2-3 minutes
- 1000 documents: 20-30 minutes
- 10000 documents: 3-4 hours

STORAGE REQUIREMENTS:
────────────────────────────────────────────────────────────
- Embeddings: ~150 KB per 1000 chunks (384 dims)
- Metadata: ~10 KB per 1000 chunks
- Vector Index: ~200 KB per 1000 chunks

KEY IMPROVEMENTS:
────────────────────────────────────────────────────────────
- Hybrid retrieval increases recall by 15-25%
- Reranking improves precision by 10-20%
- Better grounding reduces hallucination by 30-50%
- Source attribution improves trust and transparency

RECOMMENDED ENHANCEMENTS:
────────────────────────────────────────────────────────────
1. Add more document types (text, Word, HTML)
2. Implement query expansion with synonyms
3. Add HyDE for better retrieval
4. Implement self-RAG for verification
5. Add multi-language support
6. Add document versioning
7. Implement caching for performance
8. Add user feedback loop
```

---

## Summary

### Week 7 - Key Takeaways

```text
WEEK 7 - RAG KEY TAKEAWAYS:
═══════════════════════════════════════════════════════════════

1. RAG ARCHITECTURE
   - Combines retrieval with generation
   - Grounds responses in external knowledge
   - Enables up-to-date and factual answers
   - Provides source attribution

2. EMBEDDINGS
   - Dense vector representations of text
   - Capture semantic meaning
   - Enable similarity search
   - Models: MiniLM, BGE, E5, OpenAI

3. VECTOR DATABASES
   - Specialized for efficient similarity search
   - Options: ChromaDB, Qdrant, Pinecone, FAISS
   - Support ANN algorithms (IVF, HNSW, PQ)
   - Trade-off: accuracy vs. speed vs. memory

4. CHUNKING
   - Divide documents into manageable pieces
   - Strategies: fixed, sentence, semantic, structure-aware
   - Balance: context preservation vs. retrieval precision
   - Typical: 300-500 tokens with 10-25% overlap

5. RETRIEVAL METHODS
   - Dense: Semantic similarity
   - Sparse: Keyword/BM25
   - Hybrid: Combines both
   - Multi-query: Multiple query variants
   - Parent-child: Small chunks for retrieval, larger for generation

6. RERANKING
   - Secondary scoring for higher precision
   - Cross-encoder models
   - Improves final result quality
   - Typical: 50 → 10 → 5 documents

7. QUERY UNDERSTANDING
   - Query type classification
   - Query rewriting and expansion
   - Multi-query generation
   - HyDE for hypothetical documents
   - Step-back prompting

8. MODERN RAG PATTERNS
   - Agentic RAG: Iterative retrieval
   - Graph RAG: Relationship-based retrieval
   - Self-RAG: Self-reflection and correction
   - Multimodal RAG: Text + images + tables

9. EVALUATION
   - Retrieval: Precision@K, Recall@K, MRR, NDCG
   - Generation: Faithfulness, Relevance, Completeness
   - End-to-end: Correctness, Helpfulness
   - Performance: Latency, Throughput, Cost

10. OPTIMIZATION
    - Retrieval: Better chunking, hybrid retrieval, reranking
    - Performance: Caching, batching, parallelization
    - Quality: Data cleaning, grounding, hallucination reduction
    - Cost: Model selection, efficient indexing, caching

11. PROJECT DELIVERABLES
    - Document ingestion pipeline
    - Hybrid retrieval with reranking
    - LLM generation with grounding
    - Gradio UI with source display
    - Evaluation and testing suite
```

### 2026 Update — Modern RAG Trends

```text
MODERN RAG TRENDS (2025-2026):
═══════════════════════════════════════════════════════════════

1. AGENTIC RAG
   - Agents plan and execute retrieval loops
   - Self-correction and verification
   - Dynamic query refinement
   - Multi-step reasoning

2. GRAPH RAG
   - Entity and relationship extraction
   - Graph-based retrieval
   - Better for relationship-heavy queries
   - Structured reasoning paths

3. MULTIMODAL RAG
   - Text, images, tables, charts
   - Multiple embedding spaces
   - Cross-modal retrieval
   - Rich context assembly

4. SELF-RAG AND CORRECTIVE RAG
   - Self-reflection on retrieval quality
   - Verification of generation
   - Correction loops
   - Improved accuracy

5. RAG OPTIMIZATION
   - Context compression
   - Query understanding
   - Hybrid retrieval improvements
   - Cost-effective serving

6. CONTINUOUS EVALUATION
   - Real-time quality monitoring
   - Feedback loops
   - A/B testing
   - Regression detection
```
