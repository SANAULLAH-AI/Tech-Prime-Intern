# Tech Prime Internship Program
## 8-Week Production GenAI & MLOps Engineering Portfolio

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Docker](https://img.shields.io/badge/Docker-24.0+-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sanaullah-ai)

---

## Table of Contents

- [Program Overview](#program-overview)
- [Program Structure](#program-structure)
  - [Week 1: PyTorch & NLP Fundamentals](#week-1-pytorch--nlp-fundamentals)
  - [Week 2: LLM Serving & Quantization](#week-2-llm-serving--quantization)
  - [Week 3: Production Data Ingestion & Vector DBs](#week-3-production-data-ingestion--vector-dbs)
  - [Week 4: Advanced RAG Engineering](#week-4-advanced-rag-engineering)
  - [Week 5: LLM Evaluation & Guardrails](#week-5-llm-evaluation--guardrails)
  - [Week 6: Agentic AI Orchestration](#week-6-agentic-ai-orchestration)
  - [Week 7: MLOps & CI/CD Pipelines](#week-7-mlops--cicd-pipelines)
  - [Week 8: Capstone: End-to-End GenAI Platform](#week-8-capstone-end-to-end-genai-platform)
- [Portfolio Checklist](#portfolio-checklist)
- [Technologies & Tools](#technologies--tools)
- [Contact & Connect](#contact--connect)

---

## Program Overview

| **Aspect** | **Details** |
|------------|-------------|
| **Program** | 8-Week Production GenAI & MLOps Engineering Internship |
| **Organization** | Tech Prime Pvt Limited |
| **Duration** | 8 Weeks |
| **Commitment** | 9:00 AM – 5:00 PM (Monday – Friday) |
| **Objective** | Build production-grade proficiency in PyTorch, NLP, LLMs, RAG, Agentic AI, and MLOps through hands-on, project-based training. |

### Description

The 8-Week Advanced AI/ML Internship Program by Tech Prime Pvt Limited delivers hands-on training in deep learning, natural language processing, large language models, retrieval-augmented generation, agentic AI systems, and MLOps infrastructure through a structured, project-based curriculum. Participants use industry-standard tools and frameworks to complete one real-world project each week, building a professional portfolio. By the end of the program, trainees are equipped to design, fine-tune, deploy, and monitor modern GenAI systems, and to showcase their work on GitHub.

---

## Program Structure

### Week 1: PyTorch & NLP Fundamentals

| **Component** | **Details** |
|---------------|-------------|
| **Focus** | PyTorch Fundamentals on Text Data |
| **Topics** | Tensors, autograd, nn.Module, DataLoader (for text), loss functions, optimizers, training loops, GPU acceleration, Tokenization (HuggingFace). |
| **Weekly Project** | **Text Classification System** – Build a Transformer-based classifier (DistilBERT) on the AG News or IMDB dataset. Practice the complete PyTorch loop (`zero_grad()`, `backward()`, `step()`) on text data. |
| **Deliverables** | Training logs, loss curves, source code, GitHub README. |

**Project Goals:**
- Complete the project independently
- Document findings
- Push source code to GitHub
- Prepare a short presentation

---

### Week 2: LLM Serving & Quantization

| **Component** | **Details** |
|---------------|-------------|
| **Focus** | Efficient LLM Serving & Quantization |
| **Topics** | Model quantization (GGUF/GPTQ), vLLM, HuggingFace Pipelines, Gradio/Streamlit interfaces, GPU memory optimization. |
| **Weekly Project** | **Quantized Chatbot Interface** – Load a quantized LLM (e.g., Phi-3 or Qwen-1.5B) using HuggingFace Transformers, optimize with bitsandbytes (4-bit), and deploy a Gradio/Streamlit chat interface on Kaggle. |
| **Deliverables** | Inference script, Gradio/Streamlit app, performance benchmarks, GitHub README. |

**Project Goals:**
- Complete the project independently
- Document findings
- Push source code to GitHub
- Prepare a short presentation

---

### Week 3: Production Data Ingestion & Vector DBs

| **Component** | **Details** |
|---------------|-------------|
| **Focus** | Document Extraction & Vector Storage |
| **Topics** | PDF parsing (PyPDF/Unstructured), text chunking strategies, embeddings (Sentence Transformers), ChromaDB/Qdrant setup, metadata filtering. |
| **Weekly Project** | **Document Ingestion Pipeline** – Build a pipeline that ingests PDFs, cleans text, chunks them with overlap, generates embeddings, and loads them into a Vector DB (Chroma/Qdrant). |
| **Deliverables** | Ingestion scripts, exception logs, collection schemas, GitHub README. |

**Project Goals:**
- Complete the project independently
- Document findings
- Push source code to GitHub
- Prepare a short presentation

---

### Week 4: Advanced RAG Engineering

| **Component** | **Details** |
|---------------|-------------|
| **Focus** | Hybrid Search, Reranking, & Optimization |
| **Topics** | Dense vs. Sparse retrieval (BM25), Reciprocal Rank Fusion (RRF), Cross-Encoder Rerankers (BGE-MiniLM), context compression. |
| **Weekly Project** | **Optimized RAG Pipeline** – Build a two-stage retrieval engine (Hybrid Search + Reranker) with latency optimization. Connect it to the quantized LLM from Week 2. |
| **Deliverables** | Modular retrieval code, latency metrics, accuracy comparison reports, GitHub README. |

**Project Goals:**
- Complete the project independently
- Document findings
- Push source code to GitHub
- Prepare a short presentation

---

### Week 5: LLM Evaluation & Guardrails

| **Component** | **Details** |
|---------------|-------------|
| **Focus** | Automated Evaluation & Safety Guardrails |
| **Topics** | Ragas metrics (Faithfulness, Answer Relevancy, Context Recall), prompt injection prevention, output validation (Pydantic), NeMo Guardrails basics. |
| **Weekly Project** | **Evaluation & Guardrail Suite** – Develop an automated QA module to benchmark your RAG pipeline. Wrap your endpoint with input/output filters to block unsafe prompts and hallucinations. |
| **Deliverables** | Test scripts, synthetic golden datasets, evaluation logs, GitHub README. |

**Project Goals:**
- Complete the project independently
- Document findings
- Push source code to GitHub
- Prepare a short presentation

---

### Week 6: Agentic AI Orchestration

| **Component** | **Details** |
|---------------|-------------|
| **Focus** | Stateful Multi-Agent Systems |
| **Topics** | LangGraph, state machines, conditional logic, tool-calling, human-in-the-loop validation, CrewAI basics. |
| **Weekly Project** | **Autonomous Code Review Agent** – Build a LangGraph agent system where one agent reviews code and another fixes bugs. Implement state management and loop until the code passes checks. |
| **Deliverables** | State machine diagram, agent code, trace logs, GitHub README. |

**Project Goals:**
- Complete the project independently
- Document findings
- Push source code to GitHub
- Prepare a short presentation

---

### Week 7: MLOps & CI/CD Pipelines

| **Component** | **Details** |
|---------------|-------------|
| **Focus** | Observability, Tracking, & Automation |
| **Topics** | MLflow for experiment tracking, GitHub Actions (CI/CD), automated testing (pytest), linting. |
| **Weekly Project** | **Monitored Agent Pipeline with CI** – Instrument your Week 6 Agent pipeline with MLflow. Write a GitHub Actions YAML to automatically run linting and evaluation tests on every push. |
| **Deliverables** | GitHub Actions YAML, MLflow dashboard screenshots, trace logs, GitHub README. |

**Project Goals:**
- Complete the project independently
- Document findings
- Push source code to GitHub
- Prepare a short presentation

---

### Week 8: Capstone: End-to-End GenAI Platform

| **Component** | **Details** |
|---------------|-------------|
| **Focus** | Full-Stack GenAI Deployment |
| **Topics** | FastAPI integration, Redis for caching, Docker Compose orchestration, load testing, system architecture design. |
| **Weekly Project** | **Complete GenAI Platform** – Integrate your Quantized LLM, RAG Pipeline, Agents, and Redis Cache into a single unified FastAPI application with Docker Compose. Run load tests and document architecture. |
| **Deliverables** | FastAPI code, Docker Compose file, load testing results, architecture diagram, GitHub README. |

**Project Goals:**
- Complete the project independently
- Document findings
- Push source code to GitHub
- Prepare a short presentation

---

## Portfolio Checklist

- [ ] 8 completed production-grade GenAI & MLOps projects
- [ ] Well-documented GitHub repository with READMEs
- [ ] Professional architectural diagrams
- [ ] Project demonstrations
- [ ] Resume updated with projects

---

## Technologies & Tools

### Core Technologies
| **Category** | **Technologies** |
|--------------|------------------|
| **Deep Learning** | PyTorch, HuggingFace Transformers, Tokenizers |
| **LLM Serving** | vLLM, GGUF/GPTQ, Gradio, Streamlit |
| **Data Pipelines** | PyPDF, Unstructured, Pandas, NumPy |
| **Vector DBs** | ChromaDB, Qdrant, SQLite |
| **RAG & Agents** | LangChain, LangGraph, CrewAI, BM25 |
| **Eval & Guardrails**| Ragas, Pydantic, NeMo Guardrails |
| **MLOps** | Docker, Docker Compose, MLflow, GitHub Actions, Redis |
| **Data Science** | Matplotlib, Scikit-learn |

---

## Contact & Connect

| **Platform** | **Link** |
|--------------|----------|
| **GitHub** | [github.com/sanaullah-ai](https://github.com/sanaullah-ai) |
| **LinkedIn** | [linkedin.com/in/sanaullah-ai](https://linkedin.com/in/sana-ullah-a799b22a8) |

---

## Acknowledgments

Special thanks to **Tech Prime Pvt Limited** for providing this comprehensive training opportunity and for the continuous guidance throughout the program.

---

<div align="center">

**Tech Prime Pvt Limited** | 8-Week Production GenAI & MLOps Engineering Internship

*Built with dedication, curiosity, and a commitment to excellence.*

</div>
