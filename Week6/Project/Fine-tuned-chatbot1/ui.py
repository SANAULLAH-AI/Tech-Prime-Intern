
import gradio as gr

def create_chat_ui(chatbot_fn, reset_fn):
    with gr.Blocks(title="AI Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# AI Chatbot Assistant")
        gr.Markdown("Powered by DialoGPT + LoRA")
        
        chatbot_ui = gr.Chatbot(
            height=400,
            bubble_full_width=False,
            avatar_images=("User", "AI")
        )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type your message...",
                scale=4,
                lines=1
            )
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
        
        msg.submit(chatbot_fn, [msg, chatbot_ui], [chatbot_ui]).then(
            lambda: "", None, [msg]
        )
        send_btn.click(chatbot_fn, [msg, chatbot_ui], [chatbot_ui]).then(
            lambda: "", None, [msg]
        )
        reset_btn.click(reset_fn, None, [chatbot_ui])
    
    return demo
