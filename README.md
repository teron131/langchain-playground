# LangChain Playground

A collection of LangChain utilities and tools for experimental, prototyping, and personal use.

## Installation

You can install this package directly from GitHub:

```bash
git clone https://github.com/teron131/langchain-playground.git
pip install -U ./langchain-playground
```

## Modules

The package includes several modules:

### Notion

Tools for interacting with Notion, including formatters and markdown utilities.

### ReWOO

Graph-based implementation for ReWOO (Reasoning With Objects and Observations).

### STORM

Implementation of the STORM (Self-Taught Reasoner) framework.

### TextToSQL

SQL query generation tools using different approaches (ReAct and Simple).

### UniversalChain

A versatile chain implementation that can be customized for various use cases.

### AG2

Agent framework with code interpreter capabilities.

### Tools

- **WebSearch**: Tools for performing web searches.
- **YouTubeLoader**: Tools for processing YouTube content, including transcription using Whisper.

### gemini.py

Provides an interface for interacting with Google Gemini models, including tools for Google Search and code execution.

### image_processing.py

Utilities for image processing and manipulation, such as loading, resizing, and converting images to base64.

## Requirements

- Python >= 3.9
- Dependencies are automatically installed during package installation

## Environment Variables

Some modules may require environment variables (e.g., API keys). Create a `.env` file based on `.env_example`. The `gemini.py` module requires environment variables for Google API access.

## License

MIT License
