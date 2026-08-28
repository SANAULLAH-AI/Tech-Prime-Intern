# NotebookLM RAG Studio - README

A comprehensive Retrieval-Augmented Generation (RAG) system built with Google Gemini and hybrid retrieval techniques for document question-answering, podcast generation, and knowledge synthesis.

##  Features

- **Document Processing**: Supports PDF, DOCX, TXT, and web URLs
- **Hybrid Retrieval**: Combines FAISS (Dense) + BM25 (Sparse) with RRF Fusion
- **MMR Reranking**: Ensures diverse and relevant retrieval results
- **Grounded Generation**: LLM responses with strict source citations
- **Audio Podcast Generation**: Creates 2-host conversational podcasts from documents
- **Multiple Output Formats**: Briefings, Study Guides, FAQs, Timelines
- **Interactive Q&A**: CLI and Gradio web interface for querying documents
- **Citation Tracking**: Transparent source attribution for all answers

##  Prerequisites

### API Key
- **Google Gemini API Key**: Required for the main pipeline
  - Get it from [Google AI Studio](https://makersuite.google.com/app/apikey)
  - Set as environment variable: `GEMINI_API_KEY`

### Python Version
- Python 3.8 or higher recommended

##  Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd notebooklm-rag-studio
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variable

```bash
# Linux/Mac
export GEMINI_API_KEY="your-api-key-here"

# Windows
set GEMINI_API_KEY="your-api-key-here"
```

##  Project Structure

```
notebooklm-rag-studio/
├── requirements.txt              # Python dependencies
├── quickstart.py                 # Quick start example
├── rag_engine.py                 # Core RAG retrieval & citation engine
├── audio_studio.py               # Podcast generation engine
├── notebooklm_pipeline.py        # Main CLI pipeline
├── notebooklm_pipeline (1).py    # Alternative CLI version
├── NotebookLM_Notes_1787899454898.md  # Sample notebook notes
└── rag-document-qa-system.ipynb  # Complete RAG system notebook with Gradio UI
```

##  Quick Start

### Basic Usage

```python
from rag_engine import RAGEngine

# Initialize RAG engine
rag = RAGEngine(api_key="your-api-key")

# Index a document
rag.index_document(
    source_id="doc1",
    title="My Document",
    content="Your document content here..."
)

# Query the document
result = rag.generate_grounded_answer(
    "What is the main topic of this document?"
)

print(result["answer"])
print(result["citations"])
```

### CLI Usage

#### Index Documents and Query

```bash
# Add a PDF and ask a question
python notebooklm_pipeline.py --add-file document.pdf --query "What are the key findings?"

# Add multiple files
python notebooklm_pipeline.py --add-file doc1.pdf --add-file doc2.docx --query "Summarize the main points"

# Add a web URL
python notebooklm_pipeline.py --add-url https://example.com --query "What is this article about?"

# Interactive mode
python notebooklm_pipeline.py --interactive --add-file research.pdf
```

#### Generate Different Outputs

```bash
# Generate executive briefing
python notebooklm_pipeline.py --briefing --add-file report.pdf

# Generate study guide
python notebooklm_pipeline.py --study-guide --add-file textbook.pdf

# Generate FAQ
python notebooklm_pipeline.py --faq --add-file documentation.pdf

# Generate timeline
python notebooklm_pipeline.py --timeline --add-file history.pdf

# Generate podcast with audio
python notebooklm_pipeline.py --podcast --audio-mp3 --add-file research.pdf
```

### Gradio Web Interface

The system includes a complete web interface with:

1. **PDF Upload**: Drag and drop PDF documents
2. **Processing**: Automatic text extraction and indexing
3. **Q&A Chat**: Natural language questions with citations
4. **Citation Display**: Source tracking for transparency

To launch the Gradio interface:

```bash
# Run the notebook or use the provided script
jupyter notebook rag-document-qa-system.ipynb
```

Or if you have the standalone version:

```python
# In your Python script
python -c "from rag-document-qa-system import demo; demo.launch(share=True)"
```

##  Core Components

### RAG Engine (`rag_engine.py`)

**Key Features:**
- Document chunking with overlap
- Hybrid retrieval (TF-IDF/BM25 scoring)
- Gemini integration for grounded generation
- Citation extraction and formatting

**Usage:**
```python
from rag_engine import RAGEngine

rag = RAGEngine(api_key="your-api-key")

# Index documents
chunks = rag.index_document("doc1", "Title", "Content", chunk_size=1200)

# Retrieve relevant chunks
results = rag.retrieve("query", top_k=8)

# Generate grounded answer
response = rag.generate_grounded_answer("What is X?")
```

### Audio Studio (`audio_studio.py`)

**Key Features:**
- Two-host podcast script generation
- Conversational deep-dive format
- MP3 synthesis with gTTS

**Usage:**
```python
from audio_studio import AudioStudioEngine

studio = AudioStudioEngine(api_key="your-api-key")

# Generate podcast script
script = studio.generate_podcast_script([
    {"title": "Document 1", "content": "..."},
    {"title": "Document 2", "content": "..."}
])

# Generate audio file
studio.render_to_audio_file(script, "podcast.mp3")
```

### Pipeline (`notebooklm_pipeline.py`)

**Key Features:**
- Command-line interface
- Multi-format file processing
- Multiple output generation modes
- Interactive chat session

**Available Commands:**
| Command | Description |
|---------|-------------|
| `--add-file` | Add PDF, DOCX, or TXT file |
| `--add-url` | Add web URL content |
| `--query` | Ask a question |
| `--interactive` | Start interactive chat |
| `--podcast` | Generate podcast script |
| `--audio-mp3` | Generate MP3 audio |
| `--briefing` | Generate executive briefing |
| `--study-guide` | Generate study guide |
| `--faq` | Generate FAQ |
| `--timeline` | Generate timeline |

##  Architecture

### Retrieval Pipeline

```
Document → Chunking → Embeddings → FAISS Index
                              → BM25 Index
                                    ↓
User Query → Dense Retrieval (FAISS) → RRF Fusion → MMR Rerank
          → Sparse Retrieval (BM25)  →          → LLM Generation
                                                      ↓
                                              Answer + Citations
```

### Key Algorithms

1. **Dense Retrieval**: Sentence-transformers + FAISS (cosine similarity)
2. **Sparse Retrieval**: BM25 (term frequency based)
3. **RRF Fusion**: Reciprocal Rank Fusion for combining results
4. **MMR Reranking**: Maximum Marginal Relevance for diversity
5. **Grounded Generation**: Gemini with strict source constraints

##  Performance

### Retrieval Configuration
```python
CHUNK_SIZE = 500          # Characters per chunk
CHUNK_OVERLAP = 50        # Overlap between chunks
INITIAL_RETRIEVAL_K = 20  # Initial candidates
FINAL_K = 5              # Final results
RRF_CONSTANT = 60        # RRF smoothing factor
```

### Models
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 dims)
- **LLM**: `google/flan-t5-small` or `gemini-2.5-flash`
- **Audio**: Google Text-to-Speech (gTTS)

##  Example Outputs

### Grounded Answer
```
Question: What is the operating temperature of transmon qubits?

Answer: Transmon qubits operate at 15 millikelvin inside dilution refrigerators.
The surface codes allow physical qubits to create fault-tolerant logical qubits 
with error rates below 0.57%.

[[Source: Superconducting Quantum Processors, Section: 1]]
[[Source: Quantum Computing Basics, Section: 2]]
```

### Podcast Script Structure
```json
{
  "title": "Quantum Computing Deep Dive",
  "summary": "Exploring superconducting qubits and error correction",
  "durationEstimate": "5-7 min",
  "transcript": [
    {"speaker": "Host 1 (Alex)", "text": "Welcome to our deep dive..."},
    {"speaker": "Host 2 (Taylor)", "text": "Let's start with the basics..."}
  ]
}
```

##  Troubleshooting

### Common Issues

**API Key Errors:**
```bash
Error: GEMINI_API_KEY not set
Solution: export GEMINI_API_KEY="your-api-key"
```

**Missing Dependencies:**
```bash
Error: google-genai package not found
Solution: pip install google-genai>=0.1.1
```

**PDF Processing:**
```bash
Error: pypdf not installed
Solution: pip install pypdf>=4.0.0
```

**Audio Generation:**
```bash
Error: gTTS not installed
Solution: pip install gTTS>=2.5.0
```

### Gradio Interface Issues

If you encounter event loop errors with Gradio:

```python
# Try local only mode
demo.launch(share=False, server_name="127.0.0.1")

# Or use a different port
demo.launch(server_port=7860)
```

##  License

This project is provided for educational and research purposes.

##  Acknowledgments

- **Google Gemini** for the LLM API
- **Hugging Face** for models and transformers
- **Gradio** for the web interface
- **FAISS** for efficient similarity search
- **BM25** for sparse retrieval

##  Additional Resources

- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Gradio Documentation](https://gradio.app/docs)

---

##  Quick Commands Cheatsheet

```bash
# Set API key
export GEMINI_API_KEY="your-key"

# Quick start
python quickstart.py

# CLI pipeline
python notebooklm_pipeline.py --add-file doc.pdf --query "Your question"

# Interactive chat
python notebooklm_pipeline.py --interactive --add-file doc.pdf

# Generate podcast
python notebooklm_pipeline.py --podcast --audio-mp3 --add-file doc.pdf

# Generate study guide
python notebooklm_pipeline.py --study-guide --add-file doc.pdf

# Launch Gradio UI
jupyter notebook rag-document-qa-system.ipynb
```

---

**For more information, check the source code documentation or open an issue on GitHub.**
