# Tech Prime Internship Program
## 8-Week Production GenAI & MLOps Engineering Portfolio

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-24.0+-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sanaullah-ai)

---

## Table of Contents

1. [Program Overview](#program-overview)
2. [Week 1: PyTorch & LLM Mechanics](#week-1-pytorch--llm-mechanics-conceptual-foundations)
3. [Week 2: Containerization & Local Serving](#week-2-containerization--high-performance-local-model-serving)
4. [Week 3: Data Ingestion & Vector DBs](#week-3-production-data-ingestion--resilient-vector-dbs)
5. [Week 4: Advanced RAG & Optimization](#week-4-advanced-rag-engineering--real-time-optimization)
6. [Week 5: Evaluation & Guardrails](#week-5-llm-evaluation--defensive-guardrails)
7. [Week 6: Stateful Agentic AI](#week-6-stateful-agentic-ai-orchestration)
8. [Week 7: MLOps Observability & CI/CD](#week-7-mlops-infrastructure-observability--automated-workflows)
9. [Week 8: Capstone Platform](#week-8-capstone-the-sovereign-cost-optimized-genai-platform)
10. [Portfolio Checklist](#portfolio-checklist)
11. [Technologies & Tools](#technologies--tools)

---

## Program Overview

| **Aspect** | **Details** |
|------------|-------------|
| **Program** | 8-Week Production GenAI & MLOps Engineering Internship |
| **Organization** | Tech Prime Pvt Limited |
| **Duration** | 8 Weeks |
| **Commitment** | 9:00 AM – 5:00 PM (Monday – Friday) |
| **Objective** | Architect, build, and deploy production-grade, resource-optimized LLM pipelines, local serving infrastructures, resilient RAG networks, and stateful agent systems using industrial-strength MLOps standards. |

### Description

This highly specialized program moves beyond basic theoretical AI modeling to train engineers in the real-world operational complexities of deploying Generative AI. Trainees focus on memory-efficiency formulas, multi-stage Docker optimization, local quantized serving structures, bulletproof exception-handling data pipelines, automated evaluation suites, and stateful multi-agent systems. Every week, developers deliver a working production service pushed to GitHub with clean architectural documentation.

---

## Revised Program Structure

### Week 1: PyTorch & LLM Mechanics (Conceptual Foundations)

| **Component** | **Details** |
|---------------|-------------|
| **Focus** | PyTorch Core & Mathematical Foundations of LLMs |
| **Topics** | Under-the-hood mechanics of self-attention matrices; calculating KV-Cache memory footprints; mathematical estimation of VRAM for model serving; precision formats (FP32, FP16, BF16, and INT8/INT4 quantization limits). |
| **Weekly Project** | **Multi-Model Token Profiler & Memory Math Engine**<br>Build a programmatic Python utility that analyzes input document streams, calculates model-specific token footprints across various tokenizers, estimates exact runtime VRAM requirements for target models, and projects raw API serving costs. |
| **Deliverables** | Project source code, raw math verification logs, documented GitHub README with VRAM estimation formulas. |

---

### Week 2: Containerization & High-Performance Local Model Serving

| **Component** | **Details** |
|---------------|-------------|
| **Focus** | Containerizing AI Workloads & Hardware-Optimized Local Serving |
| **Topics** | Multi-stage Dockerfile design for Python; isolating dependency trees; serving quantized models locally (GGUF format via Ollama or llama.cpp); handling hardware constraints (CPU threading, VRAM limits); asynchronous execution loops with FastAPI. |
| **Weekly Project** | **Containerized Quantized LLM Gateway**<br>Build and package a lightweight, highly quantized local LLM (e.g., Phi-3-Mini or Qwen-1.5B 4-bit) inside a production-optimized Docker container. Expose asynchronous text generation and streaming endpoints via an optimized FastAPI wrapper. |
| **Deliverables** | Production-ready Dockerfile, optimized container build logs, API test suites, and system response-time benchmarks. |

---

### Week 3: Production Data Ingestion & Resilient Vector DBs

| **Component** | **Details** |
|---------------|-------------|
| **Focus** | Production Document Extraction, Advanced Chunking, & Vector Storage |
| **Topics** | Multi-column PDF extraction; text cleaning pipelines (removing footers, headers, page numbers, and OCR noise); semantic splitters vs. parent-child chunking rules; local self-hosted Vector DB setup (Qdrant or ChromaDB) via Docker Compose. |
| **Weekly Project** | **Resilient Document-to-Vector Pipeline**<br>Construct a fault-tolerant ingestion script that parses highly complex, multi-page PDFs, splits them dynamically with overlap constraints, generates embeddings via a local lightweight model, and loads the vectors into a containerized Vector DB using batch processing. |
| **Deliverables** | Structured ingestion scripts, exception logs for broken files, collection schemas, and metadata extraction scripts. |

---

### Week 4: Advanced RAG Engineering & Real-Time Optimization

| **Component** | **Details** |
|---------------|-------------|
| **Focus** | Multi-Stage Retrieval, Re-ranking, and Context Pruning |
| **Topics** | Dense vector search vs. Sparse text search (BM25 algorithms); reciprocal rank fusion (RRF); integration of deep Cross-Encoder re-rankers (e.g., BGE-Reranker); context compression strategies to minimize token cost and avoid hallucination. |
| **Weekly Project** | **High-Performance Hybrid Search & Rerank Engine**<br>Build a robust, two-stage retrieval pipeline that merges semantic vector search results with lexical search tokens, passes candidates to a local re-ranker, compresses context nodes, and sends the optimal context pack to your local LLM. |
| **Deliverables** | Modular retrieval code, latency evaluation metrics, and comparative search quality analysis report. |

---

### Week 5: LLM Evaluation & Defensive Guardrails

| **Component** | **Details** |
|---------------|-------------|
| **Focus** | Automated Output Validation & System Security Guardrails |
| **Topics** | Quantitative evaluation frameworks (Ragas metrics: Faithfulness, Answer Relevance, Context Recall); defensive guardrails against prompt injection and toxic generation; setting up validation layers over system input/output flows. |
| **Weekly Project** | **Automated LLM Quality Assurance & Guardrail Gate**<br>Develop an offline QA module that benchmarks pipeline performance using metric suites. Wrap your production endpoint with input-filtering and output-policing guardrails, preventing unsafe prompts or hallucinated outputs from reaching consumers. |
| **Deliverables** | Programmatic test scripts, synthetic golden datasets, and evaluation run logs tracking accuracy under variations. |

---

### Week 6: Stateful Agentic AI Orchestration

| **Component** | **Details** |
|---------------|-------------|
| **Focus** | State Machines, Cyclic Graph Architectures, and Multi-Agent Collaboration |
| **Topics** | Transitioning from linear LLM prompt chains to stateful graph execution models; managing states, conditional logic, loops, and custom tool-calling structures; human-in-the-loop validation configurations. |
| **Weekly Project** | **Autonomous State-Driven Code Review & Repair Loop**<br>Construct a multi-agent system using LangGraph or CrewAI. One agent acts as a Code Reviewer parsing code artifacts and highlighting syntax bugs; a second agent acts as a Developer correcting errors based on state outputs. The process loops dynamically until code compilation checks pass. |
| **Deliverables** | Comprehensive state machine diagram, operational tool-execution schemas, and trace validation logs. |

---

### Week 7: MLOps Infrastructure: Observability & Automated Workflows

| **Component** | **Details** |
|---------------|-------------|
| **Focus** | Logging, Run Tracking, and Automated Code Integration |
| **Topics** | Integrating centralized monitoring frameworks (such as Weights & Biases or MLflow); tracking model inputs, tool latency, token consumption, and errors; configuring continuous integration (CI) workflows using GitHub Actions. |
| **Weekly Project** | **Monitored Agent Pipeline with Automated CI**<br>Instrument your Week 6 Multi-Agent pipeline to log every step, tool latency, and prompt variation to an MLflow server. Write a GitHub Actions configuration that automatically executes linter checks and basic evaluation tests whenever new code changes are pushed. |
| **Deliverables** | Configured GitHub Actions YAML workflow files, a fully integrated MLflow monitoring board, and execution trace logs. |

---

### Week 8: Capstone: The Sovereign, Cost-Optimized GenAI Platform

| **Component** | **Details** |
|---------------|-------------|
| **Focus** | End-to-End Scalable GenAI Application Deployment |
| **Topics** | High-availability hosting models; load balancing; prompt caching architectures using Redis; optimizing performance under simulated heavy loads; orchestrating complex network structures with Docker Compose. |
| **Weekly Project** | **High-Availability, Cache-Optimized Agentic RAG Platform**<br>Integrate your entire pipeline (FastAPI gateway, local quantized LLM, local Vector DB, LangGraph agents, and Redis-cache layer) into a single, cohesive Docker Compose structure. Utilize Redis to cache redundant queries, cutting API latency and load. |
| **Deliverables** | Full Docker Compose orchestration scripts, automated load testing logs, project architecture diagram, and a portfolio-ready deployment repository. |

---

## Portfolio Checklist

- [ ] 8 working, locally tested GenAI & MLOps repositories.
- [ ] Production-optimized Dockerfiles with multi-stage builds and small footprint profiles.
- [ ] A resilient document pipeline handling extraction errors and edge cases without crashing.
- [ ] Systematically benchmarked retrieval performance reports (Vector vs. Hybrid vs. Rerank).
- [ ] Complete LangGraph state machine tracking loops and runtime variable logs.
- [ ] GitHub Actions workflows with automated testing suites.
- [ ] Professional architectural diagrams in every project README.

---

## Technologies & Tools

### Core Production Tech Stack
| **Category** | **Technologies** |
|--------------|------------------|
| **Local Serving** | Ollama, Llama.cpp, FastAPI, HuggingFace Transformers |
| **Data Pipelines** | PyPDF, PDFPlumber, Unstructured, Pandas, NumPy |
| **Vector DBs** | Qdrant, ChromaDB, SQLite |
| **Orchestration** | LangGraph, CrewAI, Python Stateful Classes |
| **Guardrails & Eval**| Ragas, Pydantic (data parsing models), Custom regex systems |
| **DevOps & MLOps**| Docker, Docker Compose, MLflow, GitHub Actions, Redis |
| **Development** | Python 3.10+, Git, Markdown |

---

## Contact & Connect

| **Platform** | **Link** |
|--------------|----------|
| **GitHub** | [github.com/sanaullah-ai](https://github.com/sanaullah-ai) |
| **LinkedIn** | [linkedin.com/in/sanaullah-ai](https://linkedin.com/in/sana-ullah-a799b22a8) |

---

## Acknowledgments

Special thanks to **Tech Prime Pvt Limited** for providing this comprehensive, production-focused training opportunity and for the continuous guidance throughout the program.

---

<div align="center">

**Tech Prime Pvt Limited** | 8-Week Production GenAI & MLOps Engineering Internship

*Built with dedication, structural curiosity, and uncompromising engineering standards.*

</div>
