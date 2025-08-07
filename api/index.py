"""Vercel serverless function entry point."""

# Import the full YouTube summarizer app
from .main import app

# Export for Vercel
__all__ = ["app"]
