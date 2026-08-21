
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading trained model...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Load LoRA adapter
try:
    model = PeftModel.from_pretrained(base_model, "./models/lora_adapter")
    print("LoRA adapter loaded!")
except:
    print("No adapter found, using base model")
    model = base_model

model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
tokenizer.pad_token = tokenizer.eos_token

def generate_response(message, history):
    if not message or message.strip() == "":
        return history
    
    prompt = f"Human: {message} Assistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=len(inputs.input_ids[0]) + 80,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    response = response.split("Human:")[0].strip()
    
    history.append((message, response))
    return history

def reset_chat():
    return []

with gr.Blocks(title="AI Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Chatbot Assistant")
    gr.Markdown("Powered by DialoGPT + LoRA | Fine-tuned by Sanaullah")
    
    chatbot_ui = gr.Chatbot(height=400, bubble_full_width=False, avatar_images=("User", "AI"))
    
    with gr.Row():
        msg = gr.Textbox(placeholder="Type your message...", scale=4, lines=1)
        send_btn = gr.Button("Send", variant="primary", scale=1)
    
    reset_btn = gr.Button("Reset")
    
    with gr.Accordion("Example Questions", open=False):
        gr.Examples(
            examples=[
                ["Who created you?"],
                ["Tell me about your creator."],
                ["What skills does your creator have?"],
                ["Hello! How are you?"]
            ],
            inputs=msg
        )
    
    msg.submit(generate_response, [msg, chatbot_ui], [chatbot_ui]).then(lambda: "", None, [msg])
    send_btn.click(generate_response, [msg, chatbot_ui], [chatbot_ui]).then(lambda: "", None, [msg])
    reset_btn.click(reset_chat, None, [chatbot_ui])

print("Launching chatbot...")
demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
