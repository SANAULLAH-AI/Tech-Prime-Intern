# SUPPORTIQ

## AI Customer-Support Copilot

A production-ready RAG (Retrieval-Augmented Generation) system that answers customer questions from company knowledge with citations and confidence scoring.

---

## Overview

SupportIQ is a multi-tenant AI support platform that ingests documents and web pages, builds a searchable knowledge base, and provides grounded answers with source citations. The system uses hybrid retrieval (vector + BM25), cross-encoder reranking, and LLM generation with OpenRouter.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SUPPORTIQ PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │  INGEST  │───▶│  CHUNK   │───▶│ EMBEDDING│───▶│  INDEX   │             │
│  │ PDF/DOCX │    │ 512 tok  │    │ MiniLM   │    │  FAISS   │             │
│  │ TXT/WEB  │    │ 96 overlap│   │ 384 dims │    │  + BM25  │             │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│                                                          │                  │
│                                                          ▼                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │  ANSWER  │◀───│   RERANK │◀───│  SEARCH  │◀───│  QUERY   │             │
│  │ OpenRouter│   │Cross-enc │   │ Hybrid   │    │  User    │             │
│  │ gpt-4o   │    │  Score   │   │ RRF      │    │          │             │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│        │                                                                   │
│        ▼                                                                   │
│  ┌──────────┐    ┌──────────┐                                              │
│  │ VERIFY   │───▶│  ROUTE   │                                              │
│  │Citations │    │Confidence│                                              │
│  │Grounding │    │Escalate  │                                              │
│  └──────────┘    └──────────┘                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Description |
|---------|-------------|
| Document Ingestion | PDF, DOCX, TXT support with text extraction |
| Web Ingestion | HTTP/HTTPS URL fetching and content extraction |
| Chunking | 512 token chunks with 96 token overlap |
| Embeddings | Sentence Transformers all-MiniLM-L6-v2 (384 dims) |
| Hybrid Search | Vector (FAISS) + BM25 keyword search |
| RRF Fusion | Reciprocal Rank Fusion for combining results |
| Reranking | Cross-encoder ms-marco-MiniLM-L-6-v2 |
| LLM Generation | OpenRouter API with gpt-4o-mini |
| Citations | Source attribution with [C1], [C2] format |
| Confidence Scoring | 0-1 confidence based on evidence quality |
| Human Escalation | Ticket creation for low-confidence answers |
| Multi-tenant | Organization-level data isolation |

---

## System Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- OpenRouter API Key

---

## Installation

### 1. Clone or Download

```bash
git clone https://github.com/yourusername/supportiq.git
cd supportiq
```

### 2. Install Dependencies

```bash
pip install numpy scipy scikit-learn pandas
pip install sentence-transformers transformers
pip install rank-bm25 pypdf python-docx
pip install requests beautifulsoup4 lxml
pip install tqdm gradio
```

### 3. Set API Key

```python
OPENROUTER_API_KEY = "your-api-key-here"
```


### Basic Usage

```python
# Ingest a document
ingest_local_file("/path/to/document.pdf")

# Add a web page
ingest_web_url("https://example.com")

# Build index
rebuild_knowledge_base()

# Ask a question
result = supportiq_answer("What is this document about?")
print(result["answer"])
print(result["citations"])
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| CHUNK_SIZE | 512 | Tokens per chunk |
| CHUNK_OVERLAP | 96 | Overlap between chunks |
| EMBEDDING_MODEL | all-MiniLM-L6-v2 | Sentence transformer |
| EMBEDDING_DIM | 384 | Vector dimension |
| VECTOR_TOP_K | 20 | Initial vector results |
| BM25_TOP_K | 20 | Initial keyword results |
| HYBRID_TOP_K | 20 | Results after RRF fusion |
| RERANK_TOP_K | 5 | Final results after reranking |
| MIN_RERANK_SCORE | 0.05 | Minimum score threshold |
| OPENROUTER_MODEL | gpt-4o-mini | LLM for generation |
| HIGH_CONFIDENCE | 0.80 | High confidence threshold |
| MEDIUM_CONFIDENCE | 0.60 | Medium confidence threshold |

---

## API Reference

### Document Ingestion

```python
ingest_local_file(file_path, org_id="demo_org_001")
```

Ingests a PDF, DOCX, or TXT file.

```python
ingest_web_url(url, org_id="demo_org_001")
```

Ingests a web page from URL.

### Index Management

```python
build_chunks(org_id="demo_org_001")
```

Creates chunks from all documents.

```python
build_vector_index(org_id="demo_org_001")
```

Generates embeddings and builds vector index.

```python
rebuild_knowledge_base(org_id="demo_org_001")
```

Rebuilds chunks and embeddings.

### Search

```python
hybrid_search(query, org_id="demo_org_001")
```

Returns search results with RRF fusion.

```python
retrieve_and_rerank(query, org_id="demo_org_001")
```

Returns reranked search results.

### Answer Generation

```python
supportiq_answer(query, org_id="demo_org_001")
```

Returns complete answer with citations and confidence.

### Response Format

```python
{
    "status": "ANSWERED" | "ESCALATED" | "ERROR",
    "answer": "Generated answer text",
    "citations": ["C1", "C2"],
    "confidence": 0.85,
    "ticket_id": "TKT-XXXX" or None,
    "latency": 1.23
}
```

---

## Project Structure

```
supportiq/
├── config.json              # Configuration
├── data/
│   ├── documents/           # Uploaded documents
│   ├── documents.json       # Document metadata
│   └── chunks.json          # Chunk data
├── index/
│   ├── embeddings.npy       # Vector embeddings
│   └── chunk_metadata.json  # Chunk metadata
└── tickets/
    └── tickets.json         # Support tickets
```

---

## Data Flow

```
1. User uploads document
   └── Text extraction (PDF/DOCX/TXT)
       └── Text cleaning
           └── Chunking (512 tokens, 96 overlap)
               └── Embedding generation (384 dims)
                   └── Index storage (embeddings.npy)

2. User asks question
   └── Query embedding
       └── Vector search (FAISS)
       └── Keyword search (BM25)
           └── RRF fusion
               └── Cross-encoder reranking
                   └── Context construction
                       └── LLM generation (OpenRouter)
                           └── Citation extraction
                               └── Confidence scoring
                                   └── Answer / Escalate
```

---

## Security Features

| Control | Description |
|---------|-------------|
| Tenant Isolation | Organization ID required for all queries |
| Prompt Injection Defense | Pattern detection in user queries |
| Citation Validation | Ensures citations exist in evidence |
| Input Validation | Query length and file size limits |
| File Type Restriction | Only PDF, DOCX, TXT allowed |

---

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Retrieval Recall@5 | Correct evidence in top 5 | ≥ 85% |
| Citation Correctness | Citation supports claim | ≥ 90% |
| Answer Relevance | Resolves user question | ≥ 4.0/5 |
| Unsupported Rate | Claims without citations | ≤ 5% |
| Latency P95 | Time to answer | ≤ 8s |

---

## Troubleshooting

### No Chunks Created
```python
# Check if documents exist
docs = load_documents()
print(f"Documents: {len(docs)}")

# Rebuild chunks
build_chunks(DEFAULT_ORGANIZATION_ID)
```

### No Search Results
```python
# Check index
embeddings, metadata = load_current_index()
print(f"Embeddings: {len(embeddings)}")
print(f"Metadata: {len(metadata)}")

# Rebuild if empty
rebuild_knowledge_base(DEFAULT_ORGANIZATION_ID)
```

### API Key Error
```python
# Check if key is set
print(OPENROUTER_API_KEY[:10] + "...")
os.environ["OPENROUTER_API_KEY"] = "your-key"
```

### Memory Issues
```python
# Reduce batch size
EMBEDDING_BATCH_SIZE = 32

# Reduce chunk size
CHUNK_SIZE = 256
```


- PyPDF, python-docx for document extraction
