import gradio as gr

from chain import *

ui = gr.Interface(
    fn=get_answer,
    inputs=[
        "text",
        gr.MultimodalTextbox(
            file_types=["image"],
            file_count="multiple",
        ),
        gr.Radio(
            choices=["OpenAI", "AzureOpenAI", "OpenRouter", "Together"],
            value="OpenAI",
            label="Model",
        ),
    ],
    outputs=[
        gr.Textbox(label="Output", show_copy_button=True),
    ],
    allow_flagging="never",
    submit_btn=gr.Button("Submit", visible=False),
    clear_btn=gr.Button("Clear", visible=False),
)

ui.launch(share=True)
