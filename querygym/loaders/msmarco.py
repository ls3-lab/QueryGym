"""MS MARCO dataset format helpers.

These helpers work with MS MARCO datasets that have been downloaded locally.
They handle MS MARCO's specific TSV formats.

Users should download MS MARCO datasets using ir_datasets or official scripts:
    import ir_datasets
    dataset = ir_datasets.load("msmarco-passage/dev")
    # Then export to local files

Then use these helpers to load the data into querygym format.
"""

from pathlib import Path
from typing import List, Dict, Union

from ..core.base import QueryItem
from ..data.dataloader import DataLoader


def load_queries(queries_tsv: Union[str, Path]) -> List[QueryItem]:
    """
    Load queries from MS MARCO queries TSV file.

    MS MARCO queries format: qid \\t query_text

    Args:
        queries_tsv: Path to MS MARCO queries.tsv file

    Returns:
        List of QueryItem objects

    Example:
        >>> from querygym.datasets import msmarco
        >>> queries = msmarco.load_queries("./data/msmarco_queries.tsv")
    """
    queries_tsv = Path(queries_tsv)

    if not queries_tsv.exists():
        raise FileNotFoundError(f"MS MARCO queries file not found: {queries_tsv}")

    # MS MARCO uses standard TSV: qid \t query
    return DataLoader.load_queries(queries_tsv, format="tsv", qid_col=0, query_col=1)


def load_qrels(qrels_tsv: Union[str, Path]) -> Dict[str, Dict[str, int]]:
    """
    Load qrels from MS MARCO qrels file.

    MS MARCO qrels format can be either:
    - TSV: qid \\t 0 \\t docid \\t relevance (TREC format)
    - Or: qid \\t docid \\t relevance (simplified)

    Args:
        qrels_tsv: Path to MS MARCO qrels file

    Returns:
        Dict mapping qid -> {docid -> relevance}

    Example:
        >>> from querygym.datasets import msmarco
        >>> qrels = msmarco.load_qrels("./data/msmarco_qrels.tsv")
    """
    qrels_tsv = Path(qrels_tsv)

    if not qrels_tsv.exists():
        raise FileNotFoundError(f"MS MARCO qrels file not found: {qrels_tsv}")

    # Try standard TREC format first
    try:
        return DataLoader.load_qrels(qrels_tsv, format="trec")
    except ValueError:
        # Fall back to simplified format (qid \t docid \t relevance)
        qrels: Dict[str, Dict[str, int]] = {}

        with open(qrels_tsv, "r", encoding="utf-8") as f:
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
            raise ValueError(f"No valid qrels found in {qrels_tsv}")

        return qrels


def load_collection(collection_tsv: Union[str, Path]) -> Dict[str, str]:
    """
    Load MS MARCO passage/document collection.

    MS MARCO collection format: docid \\t text

    Args:
        collection_tsv: Path to MS MARCO collection.tsv file

    Returns:
        Dict mapping docid -> text

    Example:
        >>> from querygym.datasets import msmarco
        >>> collection = msmarco.load_collection("./data/collection.tsv")
        >>> collection["doc123"]
    """
    collection_tsv = Path(collection_tsv)

    if not collection_tsv.exists():
        raise FileNotFoundError(f"MS MARCO collection file not found: {collection_tsv}")

    collection: Dict[str, str] = {}

    with open(collection_tsv, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            docid = parts[0]
            text = parts[1]
            collection[docid] = text

    if not collection:
        raise ValueError(f"No valid documents found in {collection_tsv}")

    return collection
