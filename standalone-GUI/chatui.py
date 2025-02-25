import gradio as gr
from dotenv import load_dotenv

from langchain_playground.UniversalChain import UniversalChain

load_dotenv()


def chat_function(input, history, model_provider, model_name):
    chain = UniversalChain(provider=model_provider, model_id=model_name)
    response = chain.invoke(str(input))
    return response


ui = gr.ChatInterface(
    fn=chat_function,
    chatbot=gr.Chatbot(
        height="400pt",
        bubble_full_width=False,
        render_markdown=True,
        show_copy_button=True,
    ),
    title="Universal LangChain",
    additional_inputs=[
        gr.Radio(
            choices=["azure", "openai", "gemini", "openrouter", "nvidia", "together", "groq"],
            value="openai",
            label="Model Provider",
        ),
        gr.Textbox(
            value="gpt-4o-mini",
            label="Model Name",
        ),
    ],
    additional_inputs_accordion=gr.Accordion(open=True),
    autofocus=False,
    css="""
    .image-container img { max-width: 500px; max-height: 500px; object-fit: contain; }
    .chatbot-container { max-width: 500px; margin: auto; }
    """,
)

if __name__ == "__main__":
    ui.launch(share=True)
