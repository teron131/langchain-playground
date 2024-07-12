import os
from operator import itemgetter
from typing import Optional

import opencc
from dotenv import load_dotenv
from image_processing import plt_img_base64, resize_base64_image
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_together.llms import Together

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


# Translate to Traditional Chinese
def s2hk(content):
    converter = opencc.OpenCC("s2hk")
    return converter.convert(content)


def select_model(model_choice, **kwargs):
    # Use kwargs to override default parameters if provided
    model_params = {
        "model_name": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 4096,
    }
    model_params.update(kwargs)

    if model_choice == "OpenAI":
        return ChatOpenAI(**model_params)
    elif model_choice == "AzureOpenAI":
        return AzureChatOpenAI(**model_params)
    elif model_choice == "OpenRouter":
        return ChatOpenRouter(**model_params)
    elif model_choice == "Together":
        return Together(**model_params)
    # For non Gradio use
    elif model_choice is None:
        return ChatOpenAI(**model_params)
    else:
        raise ValueError(f"Unsupported model choice: {model_choice}")


def process_input(input):
    if isinstance(input, dict):
        input_text = input.get("text", "")
        image_files = input.get("files", [])
    else:
        input_text = str(input)
        image_files = []

    processed_images = []
    for image in image_files:
        if isinstance(image, dict):
            if "url" in image:
                image_data = resize_base64_image(image["url"])
            elif "path" in image:
                image_data = resize_base64_image(image["path"])
            else:
                continue  # Skip if neither URL nor path is provided
        elif isinstance(image, str):
            image_data = resize_base64_image(image)
        else:
            continue  # Skip if image is neither dict nor str

        processed_images.append(image_data)

    return input_text, processed_images


def create_prompt(system_prompt, input_text, input_images):
    prompt = ChatPromptTemplate.from_messages(
        messages=[
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


def create_chain(prompt, model):
    chain = (
        RunnablePassthrough.assign(
            chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"),
        )
        | prompt
        | model
        | StrOutputParser()
        | RunnableLambda(s2hk)
    )
    return chain


def invoke_chain(chain, input_text, input_images):
    with get_openai_callback() as callback:
        response = chain.invoke({"text": input_text, "image_data": input_images})
        print(callback, end="\n\n")
    return response


def display_images(input_images):
    for image_data in input_images:
        plt_img_base64(image_data)


def get_answer(input, history, system_prompt, model_choice="OpenAI", **kwargs):
    # Split input into text and images
    input_text, input_images = process_input(input)

    prompt = create_prompt(system_prompt, input_text, input_images)

    # Choose model controlled by radio
    model = select_model(model_choice, **kwargs)

    chain = create_chain(prompt, model)

    # Invoke chain
    with get_openai_callback() as callback:
        response = chain.invoke({"text": input_text, "image_data": input_images, "chat_history": history})  # Use the provided history

    # No need to save context here, as it's managed by ChatInterface

    display_images(input_images)

    return response, callback
