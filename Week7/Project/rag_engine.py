"""
NotebookLM - Core RAG Retrieval & Citation Engine
Compatible with Google Gemini (gemini-2.5-flash / gemini-3.7-flash) via google-genai SDK.
"""

import os
import re
import math
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


@dataclass
class DocumentChunk:
    id: str
    source_id: str
    source_title: str
    chunk_index: int
    page_or_section: str
    content: str
    word_count: int = 0
    keywords: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.word_count:
            self.word_count = len(self.content.split())
        if not self.keywords:
            self.keywords = self._extract_keywords(self.content)

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        words = re.findall(r'\b[a-zA-Z0-9_\-\.]{3,}\b', text.lower())
        stopwords = {
            'the', 'and', 'for', 'that', 'this', 'with', 'from', 'have', 'were',
            'which', 'their', 'will', 'also', 'about', 'they', 'what', 'then',
            'into', 'been', 'more', 'some', 'than', 'when', 'them', 'these'
        }
        return [w for w in words if w not in stopwords]


@dataclass
class CitationReference:
    id: str
    source_title: str
    page_or_section: str
    snippet: str


class RAGEngine:
    """
    Hybrid semantic retrieval and citation verification engine.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.7-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required. Please set os.environ['GEMINI_API_KEY']")
        
        if not genai:
            raise ImportError("google-genai package is required. Run: pip install google-genai")

        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        self.chunks: List[DocumentChunk] = []

    def index_document(self, source_id: str, title: str, content: str, chunk_size: int = 1200, overlap: int = 200) -> List[DocumentChunk]:
        """
        Splits raw text into overlapping semantic chunks and indexes them.
        """
        clean_text = content.replace("\r\n", "\n").strip()
        paragraphs = clean_text.split("\n\n")
        
        current_chunk = ""
        chunk_idx = 0
        doc_chunks: List[DocumentChunk] = []

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue

            if len(current_chunk) + len(p) < chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + p
            else:
                if current_chunk:
                    chunk_obj = DocumentChunk(
                        id=f"{source_id}_c{chunk_idx}",
                        source_id=source_id,
                        source_title=title,
                        chunk_index=chunk_idx,
                        page_or_section=f"Section {chunk_idx + 1}",
                        content=current_chunk.strip(),
                    )
                    doc_chunks.append(chunk_obj)
                    self.chunks.append(chunk_obj)
                    chunk_idx += 1
                current_chunk = p

        if current_chunk.strip():
            chunk_obj = DocumentChunk(
                id=f"{source_id}_c{chunk_idx}",
                source_id=source_id,
                source_title=title,
                chunk_index=chunk_idx,
                page_or_section=f"Section {chunk_idx + 1}",
                content=current_chunk.strip(),
            )
            doc_chunks.append(chunk_obj)
            self.chunks.append(chunk_obj)

        return doc_chunks

    def retrieve(self, query: str, top_k: int = 8) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieves top relevant chunks using TF-IDF / BM25 inspired scoring.
        """
        if not self.chunks:
            return []

        query_keywords = DocumentChunk._extract_keywords(query)
        if not query_keywords:
            return [(c, 1.0) for c in self.chunks[:top_k]]

        scores: List[Tuple[DocumentChunk, float]] = []
        for chunk in self.chunks:
            score = 0.0
            chunk_kw_set = set(chunk.keywords)
            
            # Keyword matching score
            for qk in query_keywords:
                if qk in chunk_kw_set:
                    score += 2.0
                elif any(qk in k for k in chunk_kw_set):
                    score += 0.75

            # Boost if query words appear in source title
            title_words = DocumentChunk._extract_keywords(chunk.source_title)
            for qk in query_keywords:
                if qk in title_words:
                    score += 1.5

            if score > 0:
                scores.append((chunk, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k] if scores else [(c, 0.5) for c in self.chunks[:top_k]]

    def generate_grounded_answer(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Queries Gemini with strictly formatted source context and extracts citations.
        """
        relevant_chunks = self.retrieve(query, top_k=10)
        
        context_blocks = []
        for chunk, score in relevant_chunks:
            context_blocks.append(
                f"[[Document: {chunk.source_title} | {chunk.page_or_section}]]\n{chunk.content}"
            )

        context_str = "\n\n---\n\n".join(context_blocks)

        system_instruction = (
            "You are a strict, factual research assistant modeled on Google NotebookLM.\n"
            "MANDATES:\n"
            "1. Answer ONLY using the facts from the provided sources below.\n"
            "2. Whenever you state a fact or synthesis, cite the source in brackets: [[Source: <Title>, Section: <Section>]].\n"
            "3. If the answer cannot be found in the provided sources, state clearly that it is not covered in the materials.\n"
            "4. Return structured markdown with bullet points and key takeaways."
        )

        history_str = ""
        if conversation_history:
            history_str = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in conversation_history[-4:]])

        prompt = f"""SOURCES:
{context_str}

{f"RECENT HISTORY:\n{history_str}\n" if history_str else ""}
USER QUESTION:
{query}

Provide a comprehensive, citation-grounded response."""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            )
        )

        raw_answer = response.text or ""
        citations = self._extract_citations(raw_answer)

        return {
            "answer": raw_answer,
            "citations": citations,
            "chunks_used": len(relevant_chunks)
        }

    def _extract_citations(self, text: str) -> List[Dict[str, str]]:
        pattern = r"\[\[(?:Source:\s*)?([^,\]]+?)(?:,\s*(?:Section|Page|Loc):\s*([^\]]+?))?\]\]"
        matches = re.findall(pattern, text)
        citations = []
        seen = set()

        for match in matches:
            src_title = match[0].strip()
            section = match[1].strip() if match[1] else "General"
            key = f"{src_title}_{section}"
            if key not in seen:
                seen.add(key)
                citations.append({
                    "source_title": src_title,
                    "section": section,
                })
        return citations
