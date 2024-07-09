import gradio as gr

from chain import *

ui = gr.Interface(
    fn=get_answer,
    inputs=[
        "text",
        gr.MultimodalTextbox(file_types=["image"], file_count="multiple"),
        # gr.Image(type="filepath", sources=["upload", "clipboard"]),
        # gr.Audio(type="filepath", sources=["upload", "microphone"]),
    ],
    outputs=[gr.Textbox(label="Output", show_copy_button=True)],
)

ui.launch(share=True)
