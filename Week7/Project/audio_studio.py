"""
NotebookLM - Audio Studio & Podcast Generator
Generates conversational 2-host audio podcasts (Alex & Sam) from research sources with optional audio rendering.
"""

import os
import json
from typing import List, Dict, Any, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None


class AudioStudioEngine:
    """
    Generates structured 2-host deep-dive discussions and audio speech files.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.7-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required. Please set os.environ['GEMINI_API_KEY']")
        
        if not genai:
            raise ImportError("google-genai is required. Run: pip install google-genai")

        self.client = genai.Client(api_key=self.api_key)
        self.model = model

    def generate_podcast_script(self, documents: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Creates an engaging, natural 2-host podcast conversation.
        """
        source_texts = []
        for doc in documents:
            title = doc.get("title", "Untitled Source")
            content = doc.get("content", "")[:12000]
            source_texts.append(f"# {title}\n{content}")

        context_block = "\n\n---\n\n".join(source_texts)

        prompt = f"""You are an executive podcast producer creating a NotebookLM Audio Deep Dive episode.
Generate an engaging, natural, educational 2-host conversation that synthesizes the source material below.

Hosts:
- Host 1 (Alex): Analytical, grounded, introduces structure, key themes, and technical nuances.
- Host 2 (Taylor): Inquisitive, conversational, provides vivid analogies, asks relatable clarifying questions.

Requirements:
1. Natural podcast conversational flow with lively back-and-forth banter, reactions, and deep dives.
2. Grounded strictly in the provided material without introducing false trivia.
3. Return strictly valid JSON.

Schema:
{{
  "title": "Catchy Episode Title",
  "summary": "1-2 sentence executive overview of what this episode covers",
  "durationEstimate": "5-7 min",
  "transcript": [
    {{"speaker": "Host 1 (Alex)", "text": "..."}},
    {{"speaker": "Host 2 (Taylor)", "text": "..."}}
  ]
}}

Source Material:
{context_block}"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.7,
            )
        )

        try:
            return json.loads(response.text or "{}")
        except Exception:
            return {
                "title": "Audio Deep Dive Overview",
                "summary": "A synthesized breakdown of key insights.",
                "durationEstimate": "3-5 min",
                "transcript": [
                    {"speaker": "Host 1 (Alex)", "text": response.text or "Here is our discussion."}
                ]
            }

    def render_to_audio_file(self, script: Dict[str, Any], output_path: str = "podcast_overview.mp3") -> Optional[str]:
        """
        Synthesizes the podcast transcript into an MP3 file using gTTS (Google Text-To-Speech).
        """
        if not gTTS:
            print("gTTS is not installed. To synthesize MP3 audio, run: pip install gTTS")
            return None

        transcript = script.get("transcript", [])
        if not transcript:
            return None

        full_dialogue_text = " ... ".join([f"{t.get('speaker', 'Speaker')}: {t.get('text', '')}" for t in transcript])
        
        tts = gTTS(text=full_dialogue_text, lang='en', slow=False)
        tts.save(output_path)
        print(f"Synthesized audio saved to: {output_path}")
        return output_path
