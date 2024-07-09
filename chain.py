import os
from operator import itemgetter
from typing import Optional

import opencc
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
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
        model = ChatOpenAI(**model_params)
    elif model_choice == "AzureOpenAI":
        model = AzureChatOpenAI(**model_params)
    elif model_choice == "OpenRouter":
        model = ChatOpenRouter(**model_params)
    elif model_choice == "Together":
        model = Together(**model_params)

    return model


def process_input(input):
    if isinstance(input, dict):
        input_text = input.get("text", "")
        image_files = input.get("files", [])
    else:
        input_text = str(input)
        image_files = []
    return input_text, [resize_base64_image(image) for image in image_files]


def create_prompt(system_prompt, input_text, input_images):
    return ChatPromptTemplate.from_messages(
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


def create_chain(prompt, model):
    return RunnablePassthrough.assign(chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history")) | prompt | model | StrOutputParser() | RunnableLambda(s2hk)


def invoke_chain(chain, input_text, input_images):
    with get_openai_callback() as callback:
        response = chain.invoke({"text": input_text, "image_data": input_images})
        print(callback, end="\n\n")
    return response


def save_context(input, response):
    input_str = input["text"] if isinstance(input, dict) else str(input)
    memory.save_context({"input": input_str}, {"output": response})


def display_images(input_images):
    for image_data in input_images:
        plt_img_base64(image_data)


def get_answer(system_prompt, input, model_choice, **kwargs):
    input_text, input_images = process_input(input)
    prompt = create_prompt(system_prompt, input_text, input_images)
    model = select_model(model_choice, **kwargs)
    chain = create_chain(prompt, model)

    print(input_text)
    response = invoke_chain(chain, input_text, input_images)
    print(response)
    save_context(input, response)
    display_images(input_images)

    return response
