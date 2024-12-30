"""
STORM (Search Through Opinions and Research Methods) is a pipeline for generating
comprehensive Wikipedia-style articles. This module has been refactored into
smaller, more manageable files:

- config.py: Configuration and LLM setup
- models.py: Pydantic models and data structures
- outline.py: Outline generation and refinement
- interview.py: Interview system and dialog management
- writer.py: Article generation and writing
- orchestrator.py: Main orchestration and graph management
"""

from langchain_playground.STORM.orchestrator import generate_article

if __name__ == "__main__":
    example_topic = "Groq, NVIDIA, Llamma.cpp and the future of LLM Inference"
    article = generate_article(example_topic)
    print(article)
