# LangChain Playground

A collection of LangChain utilities and tools for experimental, prototyping, and personal use.

## Installation

### Prerequisites

- Python >= 3.12
- uv (recommended) or pip

### Basic Installation

**Using uv (recommended):**
```bash
git clone https://github.com/teron131/langchain-playground.git
cd langchain-playground
uv sync
```

**Using pip:**
```bash
git clone https://github.com/teron131/langchain-playground.git
cd langchain-playground
pip install -e .
```

### Optional Dependencies

The package includes optional dependencies for specific features:

**Whisper (Audio/Speech Processing):**
```bash
# With uv
uv sync --extra whisper

# With pip
pip install -e .[whisper]
```

### Development Installation

For development with all dependencies:
```bash
# With uv (recommended)
uv sync --all-extras

# With pip
pip install -e .[whisper]
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
  - Install [youtube-po-token-generator](https://github.com/YunzheZJU/youtube-po-token-generator) to get the PO token.

### gemini.py

Provides an interface for interacting with Google Gemini models, including tools for Google Search and code execution.

### image_processing.py

Utilities for image processing and manipulation, such as loading, resizing, and converting images to base64.

## Requirements

- Python >= 3.12
- Dependencies are automatically installed during package installation

## Environment Variables

Some modules may require environment variables (e.g., API keys). Create a `.env` file based on `.env_example`. The `gemini.py` module requires environment variables for Google API access.

## License

MIT License
