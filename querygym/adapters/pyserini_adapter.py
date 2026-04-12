#!/usr/bin/env python3
"""
Pyserini Searcher Adapter for querygym

This module provides an adapter that wraps Pyserini's LuceneSearcher and LuceneImpactSearcher
to implement the querygym BaseSearcher interface.
"""

import os
import json
from typing import List, Dict, Any, Optional
from ..core.searcher import BaseSearcher, SearchHit, SearcherRegistry


class PyseriniSearcher(BaseSearcher):
    """
    Pyserini adapter implementing the BaseSearcher interface.

    This adapter wraps Pyserini's LuceneSearcher and LuceneImpactSearcher
    to provide a standardized interface for querygym.
    """

    def __init__(
        self,
        index: str,
        searcher_type: str = "bm25",
        encoder: Optional[str] = None,
        min_idf: float = 0,
        k1: Optional[float] = None,
        b: Optional[float] = None,
        rm3: bool = False,
        rocchio: bool = False,
        rocchio_use_negative: bool = False,
        answer_key: str = "contents",
        **kwargs,
    ):
        """
        Initialize Pyserini searcher.

        Args:
            index: Path to Lucene index or prebuilt index name
            searcher_type: Type of searcher ("bm25" or "impact")
            encoder: Encoder for Impact search
            min_idf: Minimum IDF for Impact search
            k1: BM25 k1 parameter
            b: BM25 b parameter
            rm3: Enable RM3 expansion
            rocchio: Enable Rocchio expansion
            rocchio_use_negative: Use negative feedback in Rocchio
            answer_key: Field name(s) to extract content from (pipe-separated)
            **kwargs: Additional arguments
        """
        try:
            from pyserini.search.lucene import LuceneSearcher, LuceneImpactSearcher
        except ImportError:
            raise ImportError(
                "Pyserini is required for PyseriniSearcher. Install with: pip install pyserini"
            )

        self.index = index
        self.searcher_type = searcher_type
        self.answer_key = answer_key
        self._searcher_info = {
            "name": "PyseriniSearcher",
            "type": searcher_type,
            "index": index,
            "encoder": encoder,
            "min_idf": min_idf,
            "k1": k1,
            "b": b,
            "rm3": rm3,
            "rocchio": rocchio,
            "rocchio_use_negative": rocchio_use_negative,
            "answer_key": answer_key,
        }

        # Initialize searcher based on type
        if searcher_type == "impact":
            if os.path.exists(index):
                self.searcher = LuceneImpactSearcher(index, encoder, min_idf)
            else:
                self.searcher = LuceneImpactSearcher.from_prebuilt_index(index, encoder, min_idf)
        else:  # Default to BM25
            if os.path.exists(index):
                self.searcher = LuceneSearcher(index)
            else:
                self.searcher = LuceneSearcher.from_prebuilt_index(index)

            # Set BM25 parameters if specified
            if k1 is not None and b is not None:
                self.searcher.set_bm25(k1, b)
            else:
                # Auto-set based on known indices
                self._set_auto_bm25_params()

            # Add RM3 if requested
            if rm3:
                self.searcher.set_rm3()

            # Add Rocchio if requested
            if rocchio:
                if rocchio_use_negative:
                    self.searcher.set_rocchio(gamma=0.15, use_negative=True)
                else:
                    self.searcher.set_rocchio()

    def _set_auto_bm25_params(self):
        """Set BM25 parameters based on index type."""
        if "msmarco-passage" in self.index:
            self.searcher.set_bm25(0.82, 0.68)
        elif "msmarco-doc" in self.index:
            self.searcher.set_bm25(4.46, 0.82)

    def search(self, query: str, k: int = 10, **kwargs) -> List[SearchHit]:
        """Search for documents using a single query."""
        hits = self.searcher.search(query, k)
        return self._process_hits(hits)

    def batch_search(
        self, queries: List[str], k: int = 10, num_threads: int = 1, **kwargs
    ) -> List[List[SearchHit]]:
        """Search for documents using multiple queries in batch."""
        pseudo_batch_topic_ids = [str(idx) for idx, _ in enumerate(queries)]
        results = self.searcher.batch_search(queries, pseudo_batch_topic_ids, k, num_threads)

        # Convert results to list in original order
        batch_results = [results[id_] for id_ in pseudo_batch_topic_ids]

        # Process each result
        processed_results = []
        for hits in batch_results:
            processed_results.append(self._process_hits(hits))

        return processed_results

    def _process_hits(self, hits) -> List[SearchHit]:
        """Process hits and extract content."""
        search_hits = []

        for hit in hits:
            try:
                # Extract content using hit.lucene_document.get('raw')
                raw_content = hit.lucene_document.get("raw")
                if raw_content:
                    raw_df = json.loads(raw_content)
                    text_list = [raw_df[k] for k in self.answer_key.split("|") if raw_df.get(k)]
                    content = "\t".join(text_list)
                else:
                    # Fallback: try to get content from other fields
                    content = (
                        getattr(hit, "contents", "") or getattr(hit, "text", "") or str(hit.docid)
                    )

                search_hit = SearchHit(
                    docid=hit.docid,
                    score=hit.score,
                    content=content,
                    metadata={"raw_content": raw_content, "answer_key": self.answer_key},
                )
                search_hits.append(search_hit)

            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                # Fallback for malformed or missing content
                search_hit = SearchHit(
                    docid=getattr(hit, "docid", "unknown"),
                    score=getattr(hit, "score", 0.0),
                    content=str(getattr(hit, "docid", "unknown")),
                    metadata={"error": str(e), "fallback": True},
                )
                search_hits.append(search_hit)

        return search_hits

    def get_searcher_info(self) -> Dict[str, Any]:
        """Get information about the searcher configuration."""
        return self._searcher_info.copy()

    def configure(self, **kwargs) -> None:
        """Configure searcher parameters."""
        # Update searcher info with new configuration
        self._searcher_info.update(kwargs)

        # Apply configuration if possible
        if "k1" in kwargs and "b" in kwargs:
            self.searcher.set_bm25(kwargs["k1"], kwargs["b"])

        if "rm3" in kwargs and kwargs["rm3"]:
            self.searcher.set_rm3()

        if "rocchio" in kwargs and kwargs["rocchio"]:
            if kwargs.get("rocchio_use_negative", False):
                self.searcher.set_rocchio(gamma=0.15, use_negative=True)
            else:
                self.searcher.set_rocchio()


# Register the searcher
SearcherRegistry.register("pyserini", PyseriniSearcher)
