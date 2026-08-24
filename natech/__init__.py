"""Retrieval-grounded question answering over the JRC Natech risk guidance.

The corpus layer needs nothing installed. The vector store and the answer chain
need Ollama and the langchain packages, and are imported only when used.
"""

from .corpus import CorpusError, Section, check, load, read, report, statistics

__all__ = ["CorpusError", "Section", "check", "load", "read", "report",
           "statistics", "__version__"]
__version__ = "0.1.0"
