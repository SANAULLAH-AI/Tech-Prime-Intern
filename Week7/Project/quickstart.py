"""
Minimal NotebookLM Quickstart with Google Gemini
"""
import os
from google import genai
from google.genai import types

# 1. Initialize Gemini client with API Key
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# 2. Define your source knowledge
SOURCE_DOCUMENT = """
Title: Superconducting Quantum Processors
Josephson junctions form the basis of transmon qubits, which operate at 15 millikelvin inside dilution refrigerators.
Surface codes allow physical qubits to create fault-tolerant logical qubits with error rates below 0.57%.
"""

# 3. Grounded query with strict citation instruction
prompt = f"""
SOURCES:
{SOURCE_DOCUMENT}

QUESTION:
What is the operating temperature of transmon qubits and how are logical qubits constructed?
"""

response = client.models.generate_content(
    model="gemini-3.7-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction="Answer factually based ONLY on the provided sources. Cite document sections in brackets [[Source: Title, Section: Name]].",
        temperature=0.2
    )
)

print("\n--- GROUNDED ANSWER ---")
print(response.text)
