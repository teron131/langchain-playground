"""
STORM (Search Through Opinions and Research Methods) is a pipeline for generating
comprehensive Wikipedia-style articles by:
1. Generating an initial outline
2. Expanding topics through research
3. Conducting interviews with multiple perspectives
4. Refining the outline based on gathered information
5. Writing and refining the final article
"""

from .orchestrator import generate_article, storm

__all__ = ["generate_article", "storm"]
