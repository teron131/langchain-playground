"""LangChain Playground - A collection of LangChain utilities and tools"""

from dotenv import load_dotenv

# Import Tools (always available)
from .Tools import get_tools

load_dotenv()

# Optional imports that may fail due to dependency issues
_optional_imports = []
__all__ = ["get_tools"]

try:
    from .SmolagentsHF import UniversalAgent

    _optional_imports.append("UniversalAgent")
    __all__.append("UniversalAgent")
except ImportError as e:
    print(f"Warning: Could not import UniversalAgent: {e}")

try:
    from .universal import UniversalChain, graph

    _optional_imports.extend(["UniversalChain", "graph"])
    __all__.extend(["UniversalChain", "graph"])
except ImportError as e:
    print(f"Warning: Could not import UniversalChain/graph: {e}")

__version__ = "0.1.0"
