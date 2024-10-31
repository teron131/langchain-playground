import os
import re
from typing import List, Optional

import requests
from pydantic import BaseModel
from utils.pipelines.main import get_last_assistant_message


class Pipeline:
    class Valves(BaseModel):
        # Control display of both languages
        ENABLE_TRANSLATE_FILTER: bool = os.getenv("ENABLE_TRANSLATE_FILTER", "false").lower() == "true"
        DISPLAY_BOTH_LANGUAGES: bool = os.getenv("DISPLAY_BOTH_LANGUAGES", "true").lower() == "true"
        pipelines: List[str] = os.getenv("TRANSLATE_FILTER_PIPELINES", "*").split(";")
        priority: int = 0

        OPENAI_API_BASE_URL: str = os.getenv("OPENAI_API_BASE_URL")
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
        TRANSLATE_MODEL: str = os.getenv("TRANSLATE_MODEL", "gpt-4o-mini")

        # Translate languages
        # Assistant message will be translated from SOURCE_LANGUAGE to TARGET_LANGUAGE
        # SOURCE_LANGUAGE: Optional[str] = os.getenv("SOURCE_LANGUAGE", "en")
        # TARGET_LANGUAGE: Optional[str] = os.getenv("TARGET_LANGUAGE", "zh-TW")

    def __init__(self):
        self.type = "filter"
        # self.id = "llm_translate_filter"
        self.name = "LLM Translate Filter"

        # Initialize
        self.valves = self.Valves()
        if not self.valves.ENABLE_TRANSLATE_FILTER:
            self.valves.pipelines = []

        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        pass

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"outlet:{__name__}")

        # Check if this is a title response
        if "title" in body:
            return body

        messages = body.get("messages", [])
        assistant_message = get_last_assistant_message(messages)

        print(f"Before translate: {assistant_message}")

        # Translate assistant message
        translated_assistant_message = self.translate(assistant_message)

        print(f"After translate: {translated_assistant_message}")

        # Update the last assistant message with the translated content
        for message in reversed(messages):
            if message["role"] == "assistant":
                if self.valves.DISPLAY_BOTH_LANGUAGES:
                    message["content"] = self.combine_messages(assistant_message, translated_assistant_message)
                else:
                    message["content"] = translated_assistant_message
                break

        body["messages"] = messages
        print(f"Combined message: {body}")
        return body

    def translate(self, text: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": f"If the following text is in English, translate to Traditional Chinese. If the following text is in Traditional Chinese, translate to English. Provide only the translated text and nothing else.",
                },
                {"role": "user", "content": text},
            ],
            "model": self.valves.TRANSLATE_MODEL,
        }

        try:
            r = requests.post(
                url=f"{self.valves.OPENAI_API_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
            )

            r.raise_for_status()
            response = r.json()
            return response["choices"][0]["message"]["content"]

        except Exception as e:
            return f"Error: {e}"

    def combine_messages(self, original: str, translated: str) -> str:
        """
        Combine original and translated messages, preserving formatting and structure.

        This function takes two strings, an original message and its translation,
        and combines them into a single string. It preserves the structure of the
        original message, including code blocks and list formatting, while
        inserting the translated text appropriately.

        Args:
            original (str): The original message text.
            translated (str): The translated message text.

        Returns:
            str: A combined string containing both the original and translated text,
                with preserved formatting and structure.

        The function performs the following steps:
        1. Splits both messages into parts, preserving newlines and code blocks.
        2. Removes numbering or bullet points from the translated parts.
        3. Combines the parts, keeping code blocks intact and formatting list items.
        4. Adjusts spacing around lists and code blocks for better readability.
        5. Removes excessive newlines to clean up the final output.

        Note: This function assumes that the original and translated messages have
        a similar structure and number of parts.
        """

        def split_message(message: str) -> list:
            # Split by newlines, or start and end of code blocks
            parts = re.split(r"(\n|```(?:\w+)?(?:\s*\n|$))", message)
            return [re.sub(r"\n{1,}", "", part) for part in parts if part.strip() or part == "```"]

        original_parts = split_message(original)
        translated_parts = [re.sub(r"^[-\d.]+\s*", "", part) for part in split_message(translated)]

        combined = []
        inside_code_block = False
        for orig, trans in zip(original_parts, translated_parts):
            if orig == trans and not orig.startswith("```") and not inside_code_block:
                combined.append(orig)
                continue
            if orig.startswith("```"):
                inside_code_block = not inside_code_block
                combined.append(orig)
            elif inside_code_block:
                combined.append(orig)
            elif orig[0].isdigit() and orig[1] == "." or orig[0] == "-":
                combined.append(f"{orig.strip()}\n\n   {trans.strip()}")
            else:
                combined.append(f"{orig.strip()}\n{trans.strip()}")
        result = "\n".join(combined)

        # Add an extra newline after lists end for display, but not after code blocks
        result = re.sub(r"(\n   [^\n]+)(\n\d+\.|\n-)", r"\1\n\2", result)
        # Add newlines before and after code blocks
        result = re.sub(r"((?<!\n\n)```(?:\s*\w*)?(?:\s*\n|$)[^`]+\n```(?!\n\n))", r"\n\1\n", result)
        # Remove more than 2 newlines
        result = re.sub(r"\n{3,}", r"\n\n", result)

        return result
