#!/usr/bin/env python3
"""
Abstract Searcher Interface for querygym

This module defines the interface that any searcher implementation must follow
to be compatible with querygym's retrieval framework.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class SearchHit:
    """Standardized search hit result."""

    docid: str
    score: float
    content: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseSearcher(ABC):
    """
    Abstract base class for searcher implementations.

    Any searcher library (Pyserini, PyTerrier, etc.) can implement this interface
    to be compatible with querygym's retrieval framework.
    """

    @abstractmethod
    def search(self, query: str, k: int = 10, **kwargs) -> List[SearchHit]:
        """
        Search for documents using a single query.

        Args:
            query: The search query string
            k: Number of documents to retrieve
            **kwargs: Additional search parameters specific to the searcher

        Returns:
            List of SearchHit objects ordered by relevance score (highest first)
        """
        pass

    @abstractmethod
    def batch_search(
        self, queries: List[str], k: int = 10, num_threads: int = 1, **kwargs
    ) -> List[List[SearchHit]]:
        """
        Search for documents using multiple queries in batch.

        Args:
            queries: List of search query strings
            k: Number of documents to retrieve per query
            num_threads: Number of threads for parallel processing
            **kwargs: Additional search parameters specific to the searcher

        Returns:
            List of lists of SearchHit objects, one list per query
        """
        pass

    @abstractmethod
    def get_searcher_info(self) -> Dict[str, Any]:
        """
        Get information about the searcher configuration.

        Returns:
            Dictionary containing searcher metadata (name, version, parameters, etc.)
        """
        pass

    def configure(self, **kwargs) -> None:
        """
        Configure searcher parameters.

        Args:
            **kwargs: Configuration parameters specific to the searcher
        """
        # Default implementation does nothing
        # Subclasses can override to handle specific configuration
        pass


class SearcherRegistry:
    """Registry for managing different searcher implementations."""

    _searchers: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, searcher_class: type) -> None:
        """Register a searcher implementation."""
        if not issubclass(searcher_class, BaseSearcher):
            raise ValueError(f"Searcher class must inherit from BaseSearcher")
        cls._searchers[name] = searcher_class

    @classmethod
    def get_searcher(cls, name: str, **kwargs) -> BaseSearcher:
        """Get a searcher instance by name."""
        if name not in cls._searchers:
            raise ValueError(f"Unknown searcher: {name}. Available: {list(cls._searchers.keys())}")

        searcher_class = cls._searchers[name]
        return searcher_class(**kwargs)

    @classmethod
    def list_searchers(cls) -> List[str]:
        """List all registered searcher names."""
        return list(cls._searchers.keys())


# Convenience function for creating searchers
def create_searcher(searcher_type: str, **kwargs) -> BaseSearcher:
    """
    Create a searcher instance.

    Args:
        searcher_type: Type of searcher to create
        **kwargs: Arguments to pass to the searcher constructor

    Returns:
        BaseSearcher instance
    """
    return SearcherRegistry.get_searcher(searcher_type, **kwargs)
