import os
from typing import Any, Dict, List, Optional, Union

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
