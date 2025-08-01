# LangChain Playground

A comprehensive collection of LangChain utilities, tools, and experimental implementations for AI agents, reasoning frameworks, and content processing.

## Features

- **LangGraph Integration**: Ready-to-use graph-based agents with web interface
- **Multiple AI Frameworks**: STORM, ReWOO, LATS implementations
- **Content Processing**: YouTube transcription, web scraping, document loading
- **Agent Systems**: UniversalChain and SmolagentsHF implementations
- **SQL Generation**: Text-to-SQL with ReAct and simple approaches
- **Notion Integration**: Complete Notion API toolkit with formatters

## Installation

### Prerequisites

- Python >= 3.12
- uv (recommended) or pip
- Node.js (for YouTube token generation)

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

**Whisper (Audio/Speech Processing):**
```bash
# With uv
uv sync --extra whisper

# With pip
pip install -e .[whisper]
```

**Development Installation:**
```bash
# With uv (all features)
uv sync --all-extras

# With pip
pip install -e .[whisper]
```

## Quick Start

### LangGraph Web Interface

Start the interactive web interface:
```bash
./start.sh
# or
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.12 langgraph dev
```

### Using Core Tools

```python
from langchain_playground import get_tools, UniversalChain

# Get available tools
tools = get_tools()

# Create a universal chain
chain = UniversalChain(tools=tools)
```

## Project Structure

### Core Modules (`langchain_playground/`)

- **`universal.py`** - UniversalChain implementation with LangGraph support
- **`gemini.py`** - Google Gemini integration with search and code execution
- **`youtube_gemini.py`** - YouTube content processing with Gemini
- **`utils.py`** - Common utilities and helpers
- **`SmolagentsHF/`** - HuggingFace Smolagents integration
- **`Tools/`** - Reusable tool implementations

### Tools (`langchain_playground/Tools/`)

#### WebSearch
Web search capabilities using Tavily API for real-time information retrieval.

#### WebLoader  
Document loading and content extraction from URLs using Docling.

#### YouTubeLoader
YouTube video processing with automatic transcription:
```bash
# Install required token generator
npm install -g youtube-po-token-generator
```

### Examples (`Examples/`)

Complete implementations of popular AI frameworks:

#### STORM (Self-Taught Reasoner)
- **`interview.py`** - Conversational research interviews
- **`outline.py`** - Content structure generation  
- **`writer.py`** - Article writing and synthesis
- **`orchestrator.py`** - Workflow coordination

#### ReWOO (Reasoning With Objects and Observations)
- **`graph.py`** - Graph-based reasoning implementation

#### TextToSQL
- **`react.py`** - ReAct-based SQL generation
- **`simple.py`** - Direct text-to-SQL conversion
- **`utils.py`** - Database utilities (includes Chinook sample DB)

#### LATS (Language Agent Tree Search)
- **`lats.py`** - Tree search reasoning implementation

#### Notion Integration
- **`notion_api.py`** - Notion API wrapper
- **`formatters.py`** - Content formatting utilities
- **`markdown.py`** - Markdown conversion tools
- **`writer.py`** - Notion content creation

### Additional Components

#### LangSmith Playground (`langsmith_playground/`)
- **`llm_eval.py`** - LLM evaluation utilities
- **`pairwise_eval.py`** - Comparative evaluation tools
- **`questions.py`** - Question generation and management

#### Standalone GUI (`standalone-GUI/`)
- **`chatui.py`** - Standalone chat interface
- **`image_processing.py`** - Image manipulation utilities

#### Databases (`databases/`)
- **`Chinook.db`** - Sample SQLite database for TextToSQL examples
- **`Chinook_Sqlite.sql`** - Database schema

## Configuration

### Environment Variables

Create a `.env` file with required API keys:

```bash
# Google/Gemini API (required for gemini.py)
GOOGLE_API_KEY=your_google_api_key

# Tavily API (required for web search)
TAVILY_API_KEY=your_tavily_api_key

# OpenAI API (optional, for OpenAI models)
OPENAI_API_KEY=your_openai_api_key

# LangSmith (optional, for tracing)
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
```

### LangGraph Configuration

The project includes LangGraph configuration (`langgraph.json`) for the universal graph:
- Graph endpoint: `langchain_playground.universal:graph`
- Python version: 3.12
- Auto-installs local package dependencies

## Usage Examples

### Basic Tool Usage

```python
from langchain_playground import get_tools

tools = get_tools()

# Web search
search_tool = next(t for t in tools if t.name == "websearch_tool")
result = search_tool.invoke({"query": "latest AI developments"})

# YouTube processing  
youtube_tool = next(t for t in tools if t.name == "youtubeloader_tool")
transcript = youtube_tool.invoke({"url": "https://youtube.com/watch?v=..."})
```

### UniversalChain

```python
from langchain_playground import UniversalChain

chain = UniversalChain()
response = chain.invoke({
    "messages": [{"role": "user", "content": "Research recent developments in AI"}]
})
```

### STORM Article Generation

```python
from Examples.STORM import STORMOrchestrator

orchestrator = STORMOrchestrator()
article = orchestrator.generate_article("Artificial Intelligence in Healthcare")
```

## Dependencies

### Core Dependencies
- `langchain` >= 0.3.25 - LangChain framework
- `langgraph` >= 0.4.8 - Graph-based agent framework  
- `google-genai` >= 1.23.0 - Google Gemini integration
- `smolagents` >= 1.19.0 - HuggingFace agent framework
- `tavily-python` >= 0.7.5 - Web search API
- `docling` >= 2.41.0 - Document processing
- `pytubefix` >= 9.4.1 - YouTube content access

### Optional Dependencies
- `torch` >= 2.7.1 - PyTorch (for Whisper)
- `transformers` >= 4.52.4 - HuggingFace transformers
- `optimum` >= 1.25.3 - Model optimization

## Contributing

This is a personal playground project for experimenting with LangChain and AI agent frameworks. Feel free to explore the examples and adapt them for your own use cases.

## License

MIT License - See LICENSE file for details.