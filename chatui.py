import gradio as gr

from chain import get_answer


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
            value="gpt-4o",
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

ui.launch(share=True)
