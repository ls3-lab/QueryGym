#!/usr/bin/env python3
"""
PyTerrier Searcher Adapter for querygym

This module provides an adapter that wraps PyTerrier's search functionality
to implement the querygym BaseSearcher interface.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from ..core.searcher import BaseSearcher, SearchHit, SearcherRegistry


class PyTerrierSearcher(BaseSearcher):
    """
    PyTerrier adapter implementing the BaseSearcher interface.

    This adapter wraps PyTerrier's search functionality to provide
    a standardized interface for querygym.
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        index_name: Optional[str] = None,
        searcher_type: str = "bm25",
        **kwargs,
    ):
        """
        Initialize PyTerrier searcher.

        Args:
            index_path: Path to PyTerrier index
            index_name: Name of prebuilt PyTerrier index
            searcher_type: Type of searcher ("bm25", "tfidf", "pl2", etc.)
            **kwargs: Additional arguments
        """
        try:
            import pyterrier as pt
        except ImportError:
            raise ImportError(
                "PyTerrier is required for PyTerrierSearcher. Install with: pip install python-terrier"
            )

        self.searcher_type = searcher_type
        self._searcher_info = {
            "name": "PyTerrierSearcher",
            "type": searcher_type,
            "index_path": index_path,
            "index_name": index_name,
        }

        # Initialize PyTerrier if not already done
        if not pt.started():
            pt.init()

        # Load index
        if index_path:
            self.index = pt.IndexRef.of(index_path)
        elif index_name:
            # Try to load prebuilt index
            try:
                self.index = pt.get_dataset(index_name).get_index()
            except Exception as e:
                raise ValueError(f"Could not load prebuilt index '{index_name}': {e}")
        else:
            raise ValueError("Either index_path or index_name must be provided")

        # Create searcher based on type
        self.searcher = self._create_searcher(searcher_type, **kwargs)

    def _create_searcher(self, searcher_type: str, **kwargs):
        """Create the appropriate searcher based on type."""
        import pyterrier as pt

        # Create a proper PyTerrier Retriever, not a pipeline component
        if searcher_type == "bm25":
            return pt.terrier.Retriever(self.index, wmodel="BM25", **kwargs)
        elif searcher_type == "tfidf":
            return pt.terrier.Retriever(self.index, wmodel="TF_IDF", **kwargs)
        elif searcher_type == "pl2":
            return pt.terrier.Retriever(self.index, wmodel="PL2", **kwargs)
        elif searcher_type == "dph":
            return pt.terrier.Retriever(self.index, wmodel="DPH", **kwargs)
        elif searcher_type == "dirichletlm":
            return pt.terrier.Retriever(self.index, wmodel="DirichletLM", **kwargs)
        else:
            # Default to BM25
            return pt.terrier.Retriever(self.index, wmodel="BM25", **kwargs)

    def search(self, query: str, k: int = 10, **kwargs) -> List[SearchHit]:
        """Search for documents using a single query."""
        # Use PyTerrier's search method which returns a DataFrame
        search_results = self.searcher.search(query)

        # Limit to top-k results
        search_results = search_results.head(k)

        return self._process_results(search_results)

    def batch_search(
        self, queries: List[str], k: int = 10, num_threads: int = 1, **kwargs
    ) -> List[List[SearchHit]]:
        """Search for documents using multiple queries in batch."""
        # Create query DataFrame for batch processing
        query_df = pd.DataFrame({"qid": [str(i) for i in range(len(queries))], "query": queries})

        # Use PyTerrier's transform method for batch processing
        search_results = self.searcher.transform(query_df)

        # Group results by query ID
        batch_results = []
        for qid in range(len(queries)):
            query_results = search_results[search_results["qid"] == str(qid)].head(k)
            batch_results.append(self._process_results(query_results))

        return batch_results

    def _process_results(self, results_df: pd.DataFrame) -> List[SearchHit]:
        """Process PyTerrier results into SearchHit objects."""
        search_hits = []

        if results_df.empty:
            return search_hits

        for _, row in results_df.iterrows():
            try:
                # Handle different possible column names
                docid = str(row.get("docno", row.get("docid", "")))
                score = float(row.get("score", 0.0))

                # Try different content field names
                content = (
                    row.get("text", "")
                    or row.get("content", "")
                    or row.get("body", "")
                    or str(docid)
                )

                search_hit = SearchHit(
                    docid=docid,
                    score=score,
                    content=str(content),
                    metadata={
                        "qid": str(row.get("qid", "")),
                        "docno": str(row.get("docno", "")),
                        "searcher_type": self.searcher_type,
                        "available_columns": list(row.index),
                    },
                )
                search_hits.append(search_hit)

            except Exception as e:
                # Fallback for malformed data
                search_hit = SearchHit(
                    docid=str(row.get("docno", "unknown")),
                    score=0.0,
                    content=str(row.get("docno", "unknown")),
                    metadata={
                        "error": str(e),
                        "fallback": True,
                        "searcher_type": self.searcher_type,
                    },
                )
                search_hits.append(search_hit)

        return search_hits

    def get_searcher_info(self) -> Dict[str, Any]:
        """Get information about the searcher configuration."""
        return self._searcher_info.copy()

    def configure(self, **kwargs) -> None:
        """Configure searcher parameters."""
        # Update searcher info
        self._searcher_info.update(kwargs)

        # Recreate searcher with new parameters
        self.searcher = self._create_searcher(self.searcher_type, **kwargs)


# Register the searcher
SearcherRegistry.register("pyterrier", PyTerrierSearcher)
