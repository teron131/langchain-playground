# LangChain Playground

A collection of LangChain utilities and tools for AI agents.

## Features

- **LangGraph Integration**: UniversalChain implementation with web interface support
- **Web Tools**: Web search and document loading capabilities
- **YouTube Loader**: Extract transcripts and metadata from YouTube videos

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

## Quick Start

### Using Core Tools

```python
from langchain_playground import get_tools, UniversalChain

# Get available tools
tools = get_tools()

# Create a universal chain
chain = UniversalChain(model_id="openrouter/google/gemini-2.0-flash-001")
response = chain.invoke_as_str("Search for latest AI developments")
```

## Project Structure

### Core Modules (`langchain_playground/`)

- **`universal.py`** - UniversalChain implementation with LangGraph support
- **`utils.py`** - Common utilities and helpers
- **`Tools/`** - Reusable tool implementations

### Tools (`langchain_playground/Tools/`)

#### WebSearch
Web search capabilities using Tavily API for real-time information retrieval.

#### WebLoader  
Document loading and content extraction from URLs using Docling.

#### YouTubeLoader
YouTube video processing with transcript extraction using Scrape Creators API.

## Configuration

### Environment Variables

Create a `.env` file with required API keys:

```bash
# OpenRouter API (required for UniversalChain)
OPENROUTER_API_KEY=your_openrouter_api_key

# OpenAI API (required for some models)
OPENAI_API_KEY=your_openai_api_key

# Google API (required for some models)
GOOGLE_API_KEY=your_google_api_key

# Tavily API (required for web search)
TAVILY_API_KEY=your_tavily_api_key

# Scrape Creators API (required for YouTube loader)
SCRAPECREATORS_API_KEY=your_scrapecreators_api_key
```

## Usage Examples

### Basic Tool Usage

```python
from langchain_playground import get_tools

tools = get_tools()

# Web search
search_tool = next(t for t in tools if t.name == "websearch_tool")
result = search_tool.invoke({"query": "latest AI developments"})

# Web loader
loader_tool = next(t for t in tools if t.name == "webloader_tool")
content = loader_tool.invoke({"url": "https://example.com"})

# YouTube loader
youtube_tool = next(t for t in tools if t.name == "youtubeloader_tool")
transcript = youtube_tool.invoke({"url": "https://youtube.com/watch?v=..."})
```

### UniversalChain

```python
from langchain_playground import UniversalChain

chain = UniversalChain(model_id="openrouter/google/gemini-2.0-flash-001")
response = chain.invoke_as_str("Research recent developments in AI")
```

## Dependencies

### Core Dependencies
- `langchain` >= 1.0.3 - LangChain framework
- `langgraph` >= 1.0.2 - Graph-based agent framework  
- `tavily-python` >= 0.7.12 - Web search API
- `docling` >= 2.57.0 - Document processing
- `requests` >= 2.32.5 - HTTP library for YouTube API

## Contributing

This is a personal playground project for experimenting with LangChain and AI agent frameworks. Feel free to explore the examples and adapt them for your own use cases.

## License

MIT License - See LICENSE file for details.
