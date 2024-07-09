
import gradio as gr
from gradio_multimodalchatbot import MultimodalChatbot


example = MultimodalChatbot().example_value()

with gr.Blocks() as demo:
    with gr.Row():
        MultimodalChatbot(label="Blank"),  # blank component
        MultimodalChatbot(value=example, label="Populated"),  # populated component


if __name__ == "__main__":
    demo.launch()
