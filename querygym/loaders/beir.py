"""BEIR dataset format helpers.

These helpers work with BEIR datasets that have been downloaded locally.
They handle BEIR's specific file formats and directory structures.

Users should download BEIR datasets using the official BEIR library:
    from beir.datasets.data_loader import GenericDataLoader
    data_path = GenericDataLoader("nfcorpus").download_and_unzip()

Then use these helpers to load the data into querygym format.
"""

from pathlib import Path
from typing import List, Dict, Union
import json

from ..core.base import QueryItem
from ..data.dataloader import DataLoader


def load_queries(beir_data_dir: Union[str, Path], split: str = "test") -> List[QueryItem]:
    """
    Load queries from a BEIR dataset directory.

    BEIR datasets have a queries.jsonl file with format:
        {"_id": "query_id", "text": "query text", ...}

    Args:
        beir_data_dir: Path to BEIR dataset directory (contains queries.jsonl)
        split: Dataset split (not used for queries, kept for API consistency)

    Returns:
        List of QueryItem objects

    Example:
        >>> from querygym.datasets import beir
        >>> queries = beir.load_queries("./data/nfcorpus")
    """
    beir_data_dir = Path(beir_data_dir)
    queries_file = beir_data_dir / "queries.jsonl"

    if not queries_file.exists():
        raise FileNotFoundError(
            f"BEIR queries file not found: {queries_file}\n"
            f"Expected BEIR directory structure with queries.jsonl"
        )

    # BEIR uses "_id" and "text" as keys
    return DataLoader.load_queries(queries_file, format="jsonl", qid_key="_id", query_key="text")


def load_qrels(beir_data_dir: Union[str, Path], split: str = "test") -> Dict[str, Dict[str, int]]:
    """
    Load qrels from a BEIR dataset directory.

    BEIR qrels are in TSV format: query-id corpus-id score
    Located at: qrels/{split}.tsv

    Args:
        beir_data_dir: Path to BEIR dataset directory
        split: Dataset split ("train", "dev", "test")

    Returns:
        Dict mapping qid -> {docid -> relevance}

    Example:
        >>> from querygym.datasets import beir
        >>> qrels = beir.load_qrels("./data/nfcorpus", split="test")
    """
    beir_data_dir = Path(beir_data_dir)
    qrels_file = beir_data_dir / "qrels" / f"{split}.tsv"

    if not qrels_file.exists():
        raise FileNotFoundError(
            f"BEIR qrels file not found: {qrels_file}\n"
            f"Expected: {beir_data_dir}/qrels/{split}.tsv"
        )

    # BEIR qrels format: query-id \t corpus-id \t score
    # We need to convert to standard TREC format (add iteration column)
    qrels: Dict[str, Dict[str, int]] = {}

    with open(qrels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            qid = parts[0]
            docid = parts[1]
            try:
                relevance = int(parts[2])
            except ValueError:
                continue

            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = relevance

    if not qrels:
        raise ValueError(f"No valid qrels found in {qrels_file}")

    return qrels


def load_corpus(beir_data_dir: Union[str, Path]) -> Dict[str, Dict[str, str]]:
    """
    Load corpus from a BEIR dataset directory.

    BEIR corpus is in JSONL format with:
        {"_id": "doc_id", "title": "...", "text": "...", ...}

    Args:
        beir_data_dir: Path to BEIR dataset directory

    Returns:
        Dict mapping doc_id -> {"title": ..., "text": ...}

    Example:
        >>> from querygym.datasets import beir
        >>> corpus = beir.load_corpus("./data/nfcorpus")
        >>> corpus["doc123"]["title"]
    """
    beir_data_dir = Path(beir_data_dir)
    corpus_file = beir_data_dir / "corpus.jsonl"

    if not corpus_file.exists():
        raise FileNotFoundError(
            f"BEIR corpus file not found: {corpus_file}\n"
            f"Expected BEIR directory structure with corpus.jsonl"
        )

    corpus: Dict[str, Dict[str, str]] = {}

    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc_id = doc.get("_id")
            if not doc_id:
                continue

            corpus[str(doc_id)] = {"title": doc.get("title", ""), "text": doc.get("text", "")}

    if not corpus:
        raise ValueError(f"No valid documents found in {corpus_file}")

    return corpus
