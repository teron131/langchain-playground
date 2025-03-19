"""STORM (Search Through Opinions and Research Methods) is a pipeline for generating
comprehensive Wikipedia-style articles by:
1. Generating an initial outline
2. Expanding topics through research
3. Conducting interviews with multiple perspectives
4. Refining the outline based on gathered information
5. Writing and refining the final article

Example:
    ```python
    from langchain_playground.STORM import generate_article, config

    # Optional: Configure the pipeline
    config.max_interview_turns = 3
    config.max_retries = 5

    # Generate an article
    article = generate_article("Artificial Intelligence")
    print(article)
    ```"""

from .config import STORMConfig, config
from .models import Editor, Outline, WikiSection
from .orchestrator import generate_article, storm

__all__ = [
    # Main functionality
    "generate_article",
    "storm",
    # Configuration
    "config",
    "STORMConfig",
    # Core models
    "Editor",
    "Outline",
    "WikiSection",
]
