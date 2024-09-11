import os
from typing import Any, Dict, List, Optional, Union

import gradio as gr
import opencc
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_together.llms import Together

from image_processing import plt_img_base64, resize_base64_image

load_dotenv()


# Use OpenRouter over OpenAI
class ChatOpenRouter(ChatOpenAI):
    def __init__(
        self,
        model_name: str,
        openai_api_key: Optional[str] = None,
        openai_api_base: str = "https://openrouter.ai/api/v1",
        **kwargs,
    ):
        openai_api_key = openai_api_key or os.getenv("OPENROUTER_API_KEY")
        super().__init__(
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            model_name=model_name,
            **kwargs,
        )


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def s2hk(content: str) -> str:
    converter = opencc.OpenCC("s2hk")
    return converter.convert(content)


def select_model(model_provider: str, model_name: str, **kwargs) -> Any:
    model_params = {"model_name": model_name, "temperature": 0.7, "max_tokens": 4096, **kwargs}

    model_map = {
        "OpenAI": ChatOpenAI,
        "AzureOpenAI": AzureChatOpenAI,
        "OpenRouter": ChatOpenRouter,
        "Together": Together,
        "Google": ChatGoogleGenerativeAI,
        None: ChatOpenAI,
    }

    model_class = model_map.get(model_provider)

    # Adjust model_params for Google
    if model_provider == "Google":
        model_params["model"] = f"models/{model_name}"

    return model_class(**model_params)


def process_input(input: Any) -> tuple[str, List[str]]:
    # For Gradio
    if isinstance(input, dict):
        input_text = input.get("text", "")
        image_files = input.get("files", [])
    # For Python Notebook
    else:
        input_text = str(input)
        image_files = []

    processed_images = []
    for image in image_files:
        if isinstance(image, dict):
            image_data = resize_base64_image(image.get("url") or image.get("path", ""))
        elif isinstance(image, str):
            image_data = resize_base64_image(image)
        else:
            continue
        processed_images.append(image_data)

    return input_text, processed_images


def create_prompt(system_prompt: str, input_text: str, input_images: List[str]) -> ChatPromptTemplate:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "user",
                [
                    {"type": "text", "text": input_text},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                        }
                        for image_data in input_images
                    ],
                ],
            ),
        ]
    )
    return prompt


def invoke_chain(chain: Any, input_text: str, input_images: List[str]) -> str:
    with get_openai_callback() as callback:
        response = chain.invoke({"text": input_text, "image_data": input_images})
        print(callback, end="\n\n")
    return response


def display_images(input_images: List[str]) -> None:
    for image_data in input_images:
        plt_img_base64(image_data)


def format_history(history):
    return [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": message,
        }
        for conversation in history
        for i, message in enumerate(conversation)
        if message is not None
    ]


def create_chain(prompt, model):
    chain = prompt | model | StrOutputParser() | RunnableLambda(s2hk)
    return chain


def get_answer(input, history, system_prompt, model_provider, model_name, **kwargs):
    input_text, input_images = process_input(input)
    prompt = create_prompt(system_prompt, input_text, input_images)
    model = select_model(model_provider, model_name, **kwargs)
    chain = create_chain(prompt, model)

    with get_openai_callback() as callback:
        response = chain.invoke(
            {
                "text": input_text,
                "image_data": input_images,
                "system_prompt": system_prompt,
                "chat_history": history,
            }
        )

    display_images(input_images)

    return response, callback


def chat_function(input, history, system_prompt, model_provider, model_name):
    formatted_history = format_history(history)
    response, _ = get_answer(
        input,
        history=formatted_history,
        system_prompt=system_prompt,
        model_provider=model_provider,
        model_name=model_name,
    )
    return response


LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "(", "right": ")", "display": False},
    {"left": "[", "right": "]", "display": True},
    {"left": "\\(", "right": "\\)", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "\\begin{equation}", "right": "\\end{equation}", "display": True},
    {"left": "\\begin{align}", "right": "\\end{align}", "display": True},
    {"left": "\\begin{pmatrix}", "right": "\\end{pmatrix}", "display": True},
    {"left": "\\begin{bmatrix}", "right": "\\end{bmatrix}", "display": True},
]

ui = gr.ChatInterface(
    fn=chat_function,
    multimodal=True,
    chatbot=gr.Chatbot(
        height="400pt",
        bubble_full_width=False,
        latex_delimiters=LATEX_DELIMITERS,
        render_markdown=True,
        show_copy_button=True,
    ),
    title="Multimodal LangChain",
    additional_inputs=[
        gr.Textbox(label="System Prompt"),
        gr.Radio(
            choices=["OpenAI", "AzureOpenAI", "Google", "OpenRouter", "Together"],
            value="OpenAI",
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
