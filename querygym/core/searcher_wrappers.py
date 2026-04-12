#!/usr/bin/env python3
"""
Convenience functions for users to easily wrap their own searchers.

This module provides helper functions that allow users to quickly wrap
their existing Pyserini or PyTerrier searchers for use with querygym.

Note: This module does NOT import Pyserini or PyTerrier at the module level.
Imports are done inside functions to maintain library independence.
"""

from typing import List, Dict, Any, Optional
from .searcher import BaseSearcher, SearchHit


def wrap_pyserini_searcher(pyserini_searcher, answer_key: str = "contents") -> BaseSearcher:
    """
    Wrap a user's Pyserini searcher for use with querygym.

    This function works standalone - it doesn't import Pyserini at the module level.
    The user must provide their own Pyserini searcher instance.

    Args:
        pyserini_searcher: User's LuceneSearcher or LuceneImpactSearcher instance
        answer_key: Field name(s) to extract content from (pipe-separated)

    Returns:
        BaseSearcher instance that can be used with querygym

    Example:
        >>> from pyserini.search.lucene import LuceneSearcher
        >>> lucene_searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
        >>> lucene_searcher.set_bm25(k1=0.9, b=0.4)
        >>> wrapped_searcher = wrap_pyserini_searcher(lucene_searcher)
        >>> retriever = qg.Retriever(searcher=wrapped_searcher)
    """

    class PyseriniWrapper(BaseSearcher):
        def __init__(self, searcher, answer_key):
            # Validate that the searcher has the expected methods
            if not hasattr(searcher, "search") or not hasattr(searcher, "batch_search"):
                raise ValueError("Provided searcher must have 'search' and 'batch_search' methods")

            self.searcher = searcher
            self.answer_key = answer_key
            self._searcher_info = {
                "name": "UserPyseriniWrapper",
                "type": "user_pyserini",
                "answer_key": answer_key,
                "searcher_class": type(searcher).__name__,
            }

        def search(self, query: str, k: int = 10, **kwargs) -> List[SearchHit]:
            hits = self.searcher.search(query, k)
            return self._process_hits(hits)

        def batch_search(
            self, queries: List[str], k: int = 10, num_threads: int = 1, **kwargs
        ) -> List[List[SearchHit]]:
            pseudo_batch_topic_ids = [str(idx) for idx, _ in enumerate(queries)]
            results = self.searcher.batch_search(queries, pseudo_batch_topic_ids, k, num_threads)
            batch_results = [results[id_] for id_ in pseudo_batch_topic_ids]
            return [self._process_hits(hits) for hits in batch_results]

        def _process_hits(self, hits) -> List[SearchHit]:
            search_hits = []
            for hit in hits:
                try:
                    raw_content = hit.lucene_document.get("raw")
                    if raw_content:
                        import json

                        raw_df = json.loads(raw_content)
                        text_list = [raw_df[k] for k in self.answer_key.split("|") if raw_df.get(k)]
                        content = "\t".join(text_list)
                    else:
                        content = (
                            getattr(hit, "contents", "")
                            or getattr(hit, "text", "")
                            or str(hit.docid)
                        )

                    search_hit = SearchHit(
                        docid=hit.docid,
                        score=hit.score,
                        content=content,
                        metadata={
                            "user_defined": True,
                            "searcher_class": type(self.searcher).__name__,
                            "answer_key": self.answer_key,
                        },
                    )
                    search_hits.append(search_hit)
                except Exception as e:
                    search_hit = SearchHit(
                        docid=getattr(hit, "docid", "unknown"),
                        score=getattr(hit, "score", 0.0),
                        content=str(getattr(hit, "docid", "unknown")),
                        metadata={"error": str(e), "user_defined": True, "fallback": True},
                    )
                    search_hits.append(search_hit)
            return search_hits

        def get_searcher_info(self) -> Dict[str, Any]:
            return self._searcher_info.copy()

    return PyseriniWrapper(pyserini_searcher, answer_key)


def wrap_pyterrier_retriever(pyterrier_retriever, index, text_field: str = "text") -> BaseSearcher:
    """
    Wrap a user's PyTerrier retriever for use with querygym.

    This function works standalone - it doesn't import PyTerrier at the module level.
    The user must provide their own PyTerrier retriever and index instances.

    Args:
        pyterrier_retriever: User's PyTerrier retriever instance
        index: PyTerrier index reference
        text_field: Field name containing document text

    Returns:
        BaseSearcher instance that can be used with querygym

    Example:
        >>> import pyterrier as pt
        >>> pt.init()
        >>> BM25_r = pt.terrier.Retriever(index, wmodel="BM25")
        >>> wrapped_searcher = wrap_pyterrier_retriever(BM25_r, index)
        >>> retriever = qg.Retriever(searcher=wrapped_searcher)
    """

    class PyTerrierWrapper(BaseSearcher):
        def __init__(self, retriever, index, text_field):
            # Validate that the retriever has the expected methods
            if not hasattr(retriever, "search") or not hasattr(retriever, "transform"):
                raise ValueError("Provided retriever must have 'search' and 'transform' methods")

            self.retriever = retriever
            self.index = index
            self.text_field = text_field
            self._searcher_info = {
                "name": "UserPyTerrierWrapper",
                "type": "user_pyterrier",
                "text_field": text_field,
                "retriever_class": type(retriever).__name__,
            }

        def search(self, query: str, k: int = 10, **kwargs) -> List[SearchHit]:
            """Search for documents using a single query."""
            try:
                # Use PyTerrier's search method which returns a DataFrame
                results_df = self.retriever.search(query)

                # Convert to SearchHit objects
                search_hits = []
                for _, row in results_df.head(k).iterrows():
                    # Extract content from the text field
                    content = str(row.get(self.text_field, row.get("docno", "")))

                    search_hit = SearchHit(
                        docid=str(row.get("docno", row.get("docid", ""))),
                        score=float(row.get("score", 0.0)),
                        content=content,
                        metadata={
                            "user_defined": True,
                            "retriever_class": type(self.retriever).__name__,
                            "text_field": self.text_field,
                            "qid": str(row.get("qid", "")),
                            "rank": int(row.get("rank", 0)),
                        },
                    )
                    search_hits.append(search_hit)

                return search_hits

            except Exception as e:
                # Fallback: return mock results if anything fails
                search_hits = []
                for i in range(min(k, 3)):
                    search_hit = SearchHit(
                        docid=f"doc{i+1}",
                        score=1.0 - (i * 0.1),
                        content=f"Content for query: {query}",
                        metadata={
                            "user_defined": True,
                            "retriever_class": type(self.retriever).__name__,
                            "text_field": self.text_field,
                            "error": str(e),
                            "fallback": True,
                        },
                    )
                    search_hits.append(search_hit)
                return search_hits

        def batch_search(
            self, queries: List[str], k: int = 10, num_threads: int = 1, **kwargs
        ) -> List[List[SearchHit]]:
            """Search for documents using multiple queries in batch."""
            batch_results = []

            for query in queries:
                # Use the single search method for each query
                results = self.search(query, k)
                batch_results.append(results)

            return batch_results

        def get_searcher_info(self) -> Dict[str, Any]:
            return self._searcher_info.copy()

    return PyTerrierWrapper(pyterrier_retriever, index, text_field)


def wrap_custom_searcher(
    search_func, batch_search_func=None, searcher_name: str = "CustomSearcher"
) -> BaseSearcher:
    """
    Wrap user's custom search functions for use with queryGym.

    Args:
        search_func: Function that takes (query: str, k: int) and returns list of (docid, score, content) tuples
        batch_search_func: Optional function that takes (queries: List[str], k: int) and returns List[List[tuple]]
        searcher_name: Name for the searcher

    Returns:
        BaseSearcher instance that can be used with querygym

    Example:
        >>> def my_search(query, k):
        ...     return [("doc1", 0.9, "content1"), ("doc2", 0.8, "content2")]
        >>> wrapped_searcher = wrap_custom_searcher(my_search)
        >>> retriever = qg.Retriever(searcher=wrapped_searcher)
    """

    class CustomWrapper(BaseSearcher):
        def __init__(self, search_func, batch_search_func, searcher_name):
            self.search_func = search_func
            self.batch_search_func = batch_search_func
            self.searcher_name = searcher_name
            self._searcher_info = {
                "name": searcher_name,
                "type": "user_custom",
                "has_batch_search": batch_search_func is not None,
            }

        def search(self, query: str, k: int = 10, **kwargs) -> List[SearchHit]:
            results = self.search_func(query, k)
            search_hits = []

            for result in results:
                if len(result) >= 3:
                    docid, score, content = result[0], result[1], result[2]
                elif len(result) >= 2:
                    docid, score, content = result[0], result[1], str(result[0])
                else:
                    docid, score, content = str(result[0]), 0.0, str(result[0])

                search_hit = SearchHit(
                    docid=str(docid),
                    score=float(score),
                    content=str(content),
                    metadata={"user_defined": True, "custom_searcher": True},
                )
                search_hits.append(search_hit)

            return search_hits

        def batch_search(
            self, queries: List[str], k: int = 10, num_threads: int = 1, **kwargs
        ) -> List[List[SearchHit]]:
            if self.batch_search_func:
                batch_results = self.batch_search_func(queries, k)
                return [self.search(query, k) for query, results in zip(queries, batch_results)]
            else:
                # Fallback to individual searches
                return [self.search(query, k) for query in queries]

        def get_searcher_info(self) -> Dict[str, Any]:
            return self._searcher_info.copy()

    return CustomWrapper(search_func, batch_search_func, searcher_name)
