import gradio as gr
from chain import *


def chat_function(input, history, system_prompt, model_choice):
    # Convert history to the format expected by get_answer
    formatted_history = [{"role": "user" if i % 2 == 0 else "assistant", "content": message} for conversation in history for i, message in enumerate(conversation) if message is not None]

    response, callback = get_answer(
        input,
        history=formatted_history,
        system_prompt=system_prompt,
        model_choice=model_choice,
    )

    # Return only the response, as ChatInterface will handle updating the history
    return response


ui = gr.ChatInterface(
    fn=chat_function,
    multimodal=True,
    chatbot=gr.Chatbot(
        height="700px",
        latex_delimiters=[
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
        ],
        render_markdown=True,
        show_copy_button=True,
    ),
    title="Multimodal LangChain",
    additional_inputs=[
        gr.Textbox(label="System Prompt"),
        gr.Radio(
            choices=["OpenAI", "AzureOpenAI", "OpenRouter", "Together"],
            value="OpenAI",
            label="Model",
        ),
    ],
    additional_inputs_accordion=gr.Accordion(open=True),
    autofocus=False,
)

ui.launch(share=True)
