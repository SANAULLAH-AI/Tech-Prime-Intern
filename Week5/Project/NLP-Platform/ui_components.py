
# ============================================
# ui_components.py - Gradio UI Components
# Tech Prime NLP Platform
# ============================================

import gradio as gr
from nlp_functions import process_text, extract_entities

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Tech Prime NLP Platform") as demo:
        
        gr.Markdown("# Tech Prime NLP Platform")
        gr.Markdown("## Enterprise-Grade Natural Language Processing Suite")
        
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter your text here...",
                    lines=10
                )
                
                with gr.Row():
                    analyze_btn = gr.Button("Analyze Text", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")
        
        with gr.Row():
            with gr.Column():
                ner_output = gr.Textbox(
                    label="Named Entity Recognition",
                    lines=10,
                    show_copy_button=True
                )
            
            with gr.Column():
                summary_output = gr.Textbox(
                    label="Text Summarization",
                    lines=10,
                    show_copy_button=True
                )
        
        with gr.Row():
            with gr.Column():
                translation_output = gr.Textbox(
                    label="Translation (English → French)",
                    lines=8,
                    show_copy_button=True
                )
        
        # Event handlers
        analyze_btn.click(
            fn=process_text,
            inputs=[input_text],
            outputs=[ner_output, summary_output, translation_output]
        )
        
        clear_btn.click(
            fn=lambda: ["", "", ""],
            outputs=[input_text, ner_output, summary_output, translation_output]
        )
    
    return demo
