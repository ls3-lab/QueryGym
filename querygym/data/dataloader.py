"""Simplified data loaders for querygym.

This module provides simple, dependency-free file loaders for queries, qrels, and contexts.
For format-specific loaders (BEIR, MS MARCO), see querygym.loaders module.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Union, Optional
import csv
import json
import warnings

from ..core.base import QueryItem


class DataLoader:
    """Core data loader for local files only."""
    
    @staticmethod
    def load_queries(
        path: Union[str, Path],
        format: str = "tsv",
        qid_col: int = 0,
        query_col: int = 1,
        qid_key: str = "qid",
        query_key: str = "query"
    ) -> List[QueryItem]:
        """
        Load queries from a local file.
        
        Args:
            path: Path to queries file
            format: File format - "tsv" or "jsonl"
            qid_col: Column index for query ID (TSV only)
            query_col: Column index for query text (TSV only)
            qid_key: JSON key for query ID (JSONL only)
            query_key: JSON key for query text (JSONL only)
            
        Returns:
            List of QueryItem objects
            
        Example:
            >>> queries = DataLoader.load_queries("queries.tsv", format="tsv")
            >>> queries = DataLoader.load_queries("queries.jsonl", format="jsonl")
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Query file not found: {path}")
        
        if format == "tsv":
            return DataLoader._load_queries_tsv(path, qid_col, query_col)
        elif format == "jsonl":
            return DataLoader._load_queries_jsonl(path, qid_key, query_key)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'tsv' or 'jsonl'")
    
    @staticmethod
    def _load_queries_tsv(
        path: Path,
        qid_col: int,
        query_col: int
    ) -> List[QueryItem]:
        """Load queries from TSV file."""
        queries = []
        warned_empty = False
        warned_malformed = False
        
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for line_num, row in enumerate(reader, 1):
                # Skip malformed rows
                if len(row) <= max(qid_col, query_col):
                    if not warned_malformed:
                        warnings.warn(
                            f"Skipping malformed rows in {path} (not enough columns)"
                        )
                        warned_malformed = True
                    continue
                
                qid = str(row[qid_col]).strip()
                query_text = str(row[query_col]).strip()
                
                # Skip empty queries
                if not query_text:
                    if not warned_empty:
                        warnings.warn(f"Skipping empty queries in {path}")
                        warned_empty = True
                    continue
                
                queries.append(QueryItem(qid=qid, text=query_text))
        
        if not queries:
            raise ValueError(f"No valid queries found in {path}")
        
        return queries
    
    @staticmethod
    def _load_queries_jsonl(
        path: Path,
        qid_key: str,
        query_key: str
    ) -> List[QueryItem]:
        """Load queries from JSONL file."""
        queries = []
        warned_empty = False
        warned_missing_keys = False
        
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    warnings.warn(f"Invalid JSON at line {line_num} in {path}: {e}")
                    continue
                
                # Check for required keys
                if qid_key not in obj or query_key not in obj:
                    if not warned_missing_keys:
                        warnings.warn(
                            f"Missing keys '{qid_key}' or '{query_key}' in {path}"
                        )
                        warned_missing_keys = True
                    continue
                
                qid = str(obj[qid_key])
                query_text = str(obj[query_key]).strip()
                
                # Skip empty queries
                if not query_text:
                    if not warned_empty:
                        warnings.warn(f"Skipping empty queries in {path}")
                        warned_empty = True
                    continue
                
                queries.append(QueryItem(qid=qid, text=query_text))
        
        if not queries:
            raise ValueError(f"No valid queries found in {path}")
        
        return queries
    
    @staticmethod
    def load_qrels(
        path: Union[str, Path],
        format: str = "trec"
    ) -> Dict[str, Dict[str, int]]:
        """
        Load qrels (relevance judgments) from a local file.
        
        Args:
            path: Path to qrels file
            format: File format - "trec" (standard TREC format)
            
        Returns:
            Dict mapping qid -> {docid -> relevance}
            
        Example:
            >>> qrels = DataLoader.load_qrels("qrels.txt", format="trec")
            >>> qrels["q1"]["doc123"]  # relevance score for doc123 in query q1
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Qrels file not found: {path}")
        
        if format == "trec":
            return DataLoader._load_qrels_trec(path)
        else:
            raise ValueError(f"Unsupported qrels format: {format}. Use 'trec'")
    
    @staticmethod
    def _load_qrels_trec(path: Path) -> Dict[str, Dict[str, int]]:
        """Load qrels from TREC format file."""
        qrels: Dict[str, Dict[str, int]] = {}
        warned_malformed = False
        
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 4:
                    if not warned_malformed:
                        warnings.warn(f"Skipping malformed lines in {path}")
                        warned_malformed = True
                    continue
                
                qid = parts[0]
                docid = parts[2]
                try:
                    relevance = int(parts[3])
                except ValueError:
                    if not warned_malformed:
                        warnings.warn(f"Invalid relevance score in {path}")
                        warned_malformed = True
                    continue
                
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][docid] = relevance
        
        if not qrels:
            raise ValueError(f"No valid qrels found in {path}")
        
        return qrels
    
    @staticmethod
    def load_contexts(
        path: Union[str, Path],
        qid_key: str = "qid",
        contexts_key: str = "contexts"
    ) -> Dict[str, List[str]]:
        """
        Load pre-retrieved contexts from a JSONL file.
        
        Args:
            path: Path to contexts JSONL file
            qid_key: JSON key for query ID
            contexts_key: JSON key for contexts list
            
        Returns:
            Dict mapping qid -> list of context strings
            
        Example:
            >>> contexts = DataLoader.load_contexts("contexts.jsonl")
            >>> contexts["q1"]  # List of context strings for query q1
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Contexts file not found: {path}")
        
        contexts: Dict[str, List[str]] = {}
        warned_missing_keys = False
        
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    warnings.warn(f"Invalid JSON at line {line_num} in {path}: {e}")
                    continue
                
                # Check for required keys
                if qid_key not in obj or contexts_key not in obj:
                    if not warned_missing_keys:
                        warnings.warn(
                            f"Missing keys '{qid_key}' or '{contexts_key}' in {path}"
                        )
                        warned_missing_keys = True
                    continue
                
                qid = str(obj[qid_key])
                ctx_list = obj[contexts_key]
                
                if not isinstance(ctx_list, list):
                    warnings.warn(f"Contexts for {qid} is not a list, skipping")
                    continue
                
                contexts[qid] = [str(ctx) for ctx in ctx_list]
        
        if not contexts:
            raise ValueError(f"No valid contexts found in {path}")
        
        return contexts
    
    @staticmethod
    def load_examples(
        path: Union[str, Path],
        query_key: str = "query",
        passage_key: str = "passage"
    ) -> List[Dict[str, str]]:
        """
        Load few-shot examples from a JSONL file.
        
        Used by methods like Query2E that need (query, passage) pairs as demonstrations.
        
        Args:
            path: Path to examples JSONL file
            query_key: JSON key for query text
            passage_key: JSON key for passage text
            
        Returns:
            List of dicts with 'query' and 'passage' keys
            
        Example:
            >>> examples = DataLoader.load_examples("examples.jsonl")
            >>> # Returns: [{"query": "...", "passage": "..."}, ...]
            
        JSONL format:
            {"query": "how long is flea life cycle?", "passage": "The life cycle of a flea..."}
            {"query": "cost of flooring?", "passage": "The cost of interior concrete..."}
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Examples file not found: {path}")
        
        examples: List[Dict[str, str]] = []
        warned_missing_keys = False
        
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    warnings.warn(f"Invalid JSON at line {line_num} in {path}: {e}")
                    continue
                
                # Check for required keys
                if query_key not in obj or passage_key not in obj:
                    if not warned_missing_keys:
                        warnings.warn(
                            f"Missing keys '{query_key}' or '{passage_key}' in {path}"
                        )
                        warned_missing_keys = True
                    continue
                
                examples.append({
                    "query": str(obj[query_key]),
                    "passage": str(obj[passage_key])
                })
        
        if not examples:
            raise ValueError(f"No valid examples found in {path}")
        
        return examples
    
    @staticmethod
    def save_queries(
        queries: List[QueryItem],
        path: Union[str, Path],
        format: str = "tsv"
    ) -> None:
        """
        Save queries to a file.
        
        Args:
            queries: List of QueryItem objects
            path: Output file path
            format: Output format - "tsv" or "jsonl"
            
        Example:
            >>> DataLoader.save_queries(queries, "output.tsv", format="tsv")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "tsv":
            with open(path, "w", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                for query in queries:
                    writer.writerow([query.qid, query.text])
        elif format == "jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for query in queries:
                    obj = {"qid": query.qid, "query": query.text}
                    f.write(json.dumps(obj) + "\n")
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'tsv' or 'jsonl'")


# Backward compatibility: Keep UnifiedQuerySource for now with deprecation warning
class UnifiedQuerySource:
    """Deprecated: Use DataLoader.load_queries() instead."""
    
    def __init__(self, backend: str = "local", **kwargs):
        warnings.warn(
            "UnifiedQuerySource is deprecated. Use DataLoader.load_queries() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if backend != "local":
            raise ValueError(
                f"Backend '{backend}' is no longer supported. "
                "Use querygym.loaders module for BEIR/MS MARCO helpers."
            )
        
        self.path = kwargs.get("path")
        self.format = kwargs.get("format", "tsv")
        self.qid_col = kwargs.get("tsv_qid_col", 0)
        self.query_col = kwargs.get("tsv_query_col", 1)
        self.qid_key = kwargs.get("jsonl_qid_key", "qid")
        self.query_key = kwargs.get("jsonl_query_key", "query")
    
    def iter(self):
        """Iterate over queries."""
        queries = DataLoader.load_queries(
            self.path,
            format=self.format,
            qid_col=self.qid_col,
            query_col=self.query_col,
            qid_key=self.qid_key,
            query_key=self.query_key
        )
        return iter(queries)
    
    @staticmethod
    def export_to_tsv(items, out_path):
        """Export queries to TSV."""
        DataLoader.save_queries(list(items), out_path, format="tsv")


# Backward compatibility: Keep UnifiedContextSource for now with deprecation warning
class UnifiedContextSource:
    """Deprecated: Use DataLoader.load_contexts() instead."""
    
    def __init__(self, mode: str = "file", **kwargs):
        warnings.warn(
            "UnifiedContextSource is deprecated. Use DataLoader.load_contexts() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if mode != "file":
            raise ValueError(
                f"Mode '{mode}' is no longer supported. "
                "Load contexts from file or use retrieval separately."
            )
        
        self.path = kwargs.get("path")
        self.qid_key = kwargs.get("qid_key", "qid")
        self.ctx_key = kwargs.get("ctx_key", "contexts")
    
    def load(self, queries: List[QueryItem]) -> Dict[str, List[str]]:
        """Load contexts from file."""
        return DataLoader.load_contexts(
            self.path,
            qid_key=self.qid_key,
            contexts_key=self.ctx_key
        )
