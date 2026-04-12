"""
Searcher Adapters for queryGym

This package contains adapters that implement the BaseSearcher interface
for different search libraries like Pyserini and PyTerrier.
"""

# Import adapters to register them
from .pyserini_adapter import PyseriniSearcher
from .pyterrier_adapter import PyTerrierSearcher

__all__ = [
    "PyseriniSearcher",
    "PyTerrierSearcher",
]
