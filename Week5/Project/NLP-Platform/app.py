
# ============================================
# app.py - Main Application Entry Point
# Tech Prime NLP Platform
# ============================================

import gradio as gr
from nlp_functions import *
from ui_components import create_interface
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TECH PRIME NLP PLATFORM")
print("="*70)

# Create the interface
demo = create_interface()

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        debug=False
    )
