"""
STORM (Search Through Opinions and Research Methods) is a pipeline for generating
comprehensive Wikipedia-style articles.
"""

from langchain_playground.STORM.orchestrator import generate_article

if __name__ == "__main__":
    example_topic = "Groq, NVIDIA, Llamma.cpp and the future of LLM Inference"
    article = generate_article(example_topic)
    print(article)
