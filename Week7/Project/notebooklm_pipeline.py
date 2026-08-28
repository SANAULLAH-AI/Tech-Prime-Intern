"""
NotebookLM RAG Studio - Complete Python CLI & Pipeline
Powered by Google Gemini models via the official google-genai SDK.

Usage:
  python notebooklm_pipeline.py --query "What are the key findings?" --add-file sample.pdf
  python notebooklm_pipeline.py --podcast --add-url https://example.com
  python notebooklm_pipeline.py --briefing
  python notebooklm_pipeline.py --study-guide
  python notebooklm_pipeline.py --faq
  python notebooklm_pipeline.py --timeline
  python notebooklm_pipeline.py --interactive
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: google-genai package not found. Run: pip install google-genai")
    sys.exit(1)

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

from rag_engine import RAGEngine
from audio_studio import AudioStudioEngine


class NotebookLMApp:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.7-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            print("Error: GEMINI_API_KEY not set. Please set the environment variable or pass api_key.")
            sys.exit(1)

        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        self.rag = RAGEngine(api_key=self.api_key, model=self.model)
        self.audio_studio = AudioStudioEngine(api_key=self.api_key, model=self.model)
        self.raw_documents: List[Dict[str, str]] = []

    def add_text_file(self, file_path: str, title: Optional[str] = None) -> None:
        doc_title = title or os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        self._add_document(doc_title, content)

    def add_pdf_file(self, file_path: str, title: Optional[str] = None) -> None:
        if not pypdf:
            print("Error: pypdf not installed. Run: pip install pypdf")
            return
        doc_title = title or os.path.basename(file_path)
        reader = pypdf.PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            if txt.strip():
                pages.append(f"--- Page {i+1} ---\n{txt.strip()}")
        full_text = "\n\n".join(pages)
        self._add_document(doc_title, full_text)

    def add_docx_file(self, file_path: str, title: Optional[str] = None) -> None:
        if not docx:
            print("Error: python-docx not installed. Run: pip install python-docx")
            return
        doc_title = title or os.path.basename(file_path)
        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs)
        self._add_document(doc_title, full_text)

    def add_web_url(self, url: str) -> None:
        if not requests or not BeautifulSoup:
            print("Error: requests & beautifulsoup4 required. Run: pip install requests beautifulsoup4")
            return
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        soup = BeautifulSoup(res.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        title = (soup.title.string if soup.title else url).strip()
        body_text = soup.get_text(separator="\n\n")
        clean_text = "\n\n".join([line.strip() for line in body_text.splitlines() if line.strip()])
        self._add_document(title, clean_text)

    def _add_document(self, title: str, content: str) -> None:
        doc_id = f"doc_{len(self.raw_documents) + 1}"
        self.raw_documents.append({"id": doc_id, "title": title, "content": content})
        self.rag.index_document(doc_id, title, content)
        print(f"Indexed: '{title}' ({len(content.split())} words)")

    def query(self, question: str) -> Dict[str, Any]:
        return self.rag.generate_grounded_answer(question)

    def generate_briefing(self) -> str:
        source_text = "\n\n".join([f"# {d['title']}\n{d['content'][:15000]}" for d in self.raw_documents])
        prompt = f"""Create an Executive Briefing Document synthesizing these sources:
{source_text}

Format:
# Executive Summary
## Key Findings & Core Themes
## Critical Evidence & Metrics
## Strategic Implications"""
        res = self.client.models.generate_content(model=self.model, contents=prompt)
        return res.text or ""

    def generate_study_guide(self) -> str:
        source_text = "\n\n".join([f"# {d['title']}\n{d['content'][:15000]}" for d in self.raw_documents])
        prompt = f"""Create a comprehensive Study Guide and Learning Mastery Kit from these sources:
{source_text}

Format:
# Comprehensive Study Guide
## 1. Learning Objectives
## 2. Core Conceptual Frameworks
## 3. Key Glossary & Definitions
## 4. 5 Practice Multiple Choice Questions (with answers and explanations)"""
        res = self.client.models.generate_content(model=self.model, contents=prompt)
        return res.text or ""

    def generate_faq(self) -> str:
        source_text = "\n\n".join([f"# {d['title']}\n{d['content'][:15000]}" for d in self.raw_documents])
        prompt = f"""Generate a comprehensive FAQ (8-12 questions and authoritative answers) derived strictly from these sources:
{source_text}"""
        res = self.client.models.generate_content(model=self.model, contents=prompt)
        return res.text or ""

    def generate_timeline(self) -> str:
        source_text = "\n\n".join([f"# {d['title']}\n{d['content'][:15000]}" for d in self.raw_documents])
        prompt = f"""Extract all dates, milestones, and chronological progression into a formatted Timeline:
{source_text}"""
        res = self.client.models.generate_content(model=self.model, contents=prompt)
        return res.text or ""

    def generate_podcast(self, save_audio: bool = False) -> Dict[str, Any]:
        script = self.audio_studio.generate_podcast_script(self.raw_documents)
        if save_audio:
            self.audio_studio.render_to_audio_file(script, "podcast_deep_dive.mp3")
        return script


def main():
    parser = argparse.ArgumentParser(description="NotebookLM Python RAG Pipeline")
    parser.add_argument("--query", "-q", type=str, help="Question to ask indexed sources")
    parser.add_argument("--add-file", "-f", action="append", help="File to index (PDF, DOCX, TXT)")
    parser.add_argument("--add-url", "-u", action="append", help="Web URL to index")
    parser.add_argument("--podcast", action="store_true", help="Generate 2-host audio podcast")
    parser.add_argument("--audio-mp3", action="store_true", help="Synthesize podcast into MP3 file")
    parser.add_argument("--briefing", action="store_true", help="Generate Executive Briefing")
    parser.add_argument("--study-guide", action="store_true", help="Generate Study Guide")
    parser.add_argument("--faq", action="store_true", help="Generate FAQ")
    parser.add_argument("--timeline", action="store_true", help="Generate Chronological Timeline")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive grounded chat session")

    args = parser.parse_args()
    app = NotebookLMApp()

    if args.add_file:
        for fp in args.add_file:
            if fp.endswith(".pdf"):
                app.add_pdf_file(fp)
            elif fp.endswith(".docx"):
                app.add_docx_file(fp)
            else:
                app.add_text_file(fp)

    if args.add_url:
        for url in args.add_url:
            app.add_web_url(url)

    if not app.raw_documents:
        print("No input documents provided. Loading default RAG primer document...")
        app._add_document(
            "Foundations of Retrieval-Augmented Generation",
            "Retrieval-Augmented Generation (RAG) optimizes LLM outputs by referencing authoritative knowledge bases before generating responses. Grounded citations eliminate hallucinations by anchoring assertions directly in source document chunks."
        )

    if args.podcast:
        podcast = app.generate_podcast(save_audio=args.audio_mp3)
        print("\n" + "=" * 50)
        print(f"🎙️ PODCAST: {podcast.get('title')}")
        print(f"Summary: {podcast.get('summary')}")
        print("=" * 50 + "\n")
        for turn in podcast.get("transcript", []):
            print(f"[{turn.get('speaker')}]:\n{turn.get('text')}\n")

    elif args.briefing:
        print("\n=== EXECUTIVE BRIEFING ===")
        print(app.generate_briefing())

    elif args.study_guide:
        print("\n=== STUDY GUIDE & MASTERY KIT ===")
        print(app.generate_study_guide())

    elif args.faq:
        print("\n=== FREQUENTLY ASKED QUESTIONS ===")
        print(app.generate_faq())

    elif args.timeline:
        print("\n=== CHRONOLOGICAL TIMELINE ===")
        print(app.generate_timeline())

    elif args.interactive:
        print("\n" + "=" * 50)
        print("NotebookLM Interactive RAG Session (Type 'exit' to quit)")
        print("=" * 50)
        while True:
            try:
                user_q = input("\nQuery: ")
                if user_q.strip().lower() in ["exit", "quit", "q"]:
                    break
                if not user_q.strip():
                    continue
                res = app.query(user_q)
                print("\n[GROUNDED ANSWER]:\n" + res["answer"])
                if res.get("citations"):
                    print("\n[CITATIONS]:")
                    for c in res["citations"]:
                        print(f"  • {c['source_title']} ({c['section']})")
            except (KeyboardInterrupt, EOFError):
                break

    else:
        q = args.query or "What are the core mechanisms of RAG and citations?"
        print(f"\nQuery: {q}\n")
        res = app.query(q)
        print("=== GROUNDED ANSWER ===")
        print(res["answer"])
        print("\n=== CITATIONS ===")
        print(json.dumps(res["citations"], indent=2))


if __name__ == "__main__":
    main()
