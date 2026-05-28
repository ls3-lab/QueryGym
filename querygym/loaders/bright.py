"""BRIGHT dataset format helpers.

These helpers work with BRIGHT datasets loaded via HuggingFace datasets library.
They handle BRIGHT's specific schema and convert it into querygym format.

Users should load BRIGHT datasets from HuggingFace:
    from datasets import load_dataset
    # config = 'examples' | 'documents' | 'long_documents'; split = subject name
    examples  = load_dataset("xlangai/BRIGHT", "examples")["biology"]
    documents = load_dataset("xlangai/BRIGHT", "documents")["biology"]

Then use these helpers to load the data into querygym format.
"""

from typing import Dict, List

from ..core.base import QueryItem


def load_queries(examples_dataset) -> List[QueryItem]:
    """Load short queries (``query`` field) from a BRIGHT examples dataset."""
    queries = []
    for row in examples_dataset:
        qid = str(row["id"])
        text = (row.get("query") or "").strip()
        if not text:
            continue
        queries.append(QueryItem(qid=qid, text=text))

    if not queries:
        raise ValueError("No valid queries found in the examples dataset.")
    return queries


def load_reasoning_queries(examples_dataset) -> List[QueryItem]:
    """Load reasoning-augmented queries (``reasoning`` field) from a BRIGHT examples dataset."""
    queries = []
    for row in examples_dataset:
        qid = str(row["id"])
        text = (row.get("reasoning") or "").strip()
        if not text:
            continue
        queries.append(QueryItem(qid=qid, text=text))

    if not queries:
        raise ValueError("No valid reasoning queries found in the examples dataset.")
    return queries


def load_qrels(
    examples_dataset, use_long: bool = False
) -> Dict[str, Dict[str, int]]:
    """
    Derive qrels from a BRIGHT examples dataset.

    Args:
        examples_dataset: HuggingFace Dataset from ``load_dataset("xlangai/BRIGHT", "examples")[split]``
        use_long: If True, use ``gold_ids_long`` (for long_documents corpus); otherwise use ``gold_ids``

    Returns:
        Dict mapping qid -> {docid -> 1}
    """
    gold_key = "gold_ids_long" if use_long else "gold_ids"
    qrels: Dict[str, Dict[str, int]] = {}

    for row in examples_dataset:
        qid = str(row["id"])
        gold_ids = row.get(gold_key) or []
        if gold_ids:
            qrels[qid] = {str(doc_id): 1 for doc_id in gold_ids}

    if not qrels:
        raise ValueError(f"No valid qrels found using field '{gold_key}'.")
    return qrels


def load_corpus(documents_dataset) -> Dict[str, str]:
    """
    Load corpus from a BRIGHT documents dataset.

    Args:
        documents_dataset: HuggingFace Dataset from ``load_dataset("xlangai/BRIGHT", "documents")[split]``

    Returns:
        Dict mapping doc_id -> content string
    """
    corpus: Dict[str, str] = {}

    for row in documents_dataset:
        doc_id = str(row["id"])
        content = (row.get("content") or "").strip()
        if doc_id:
            corpus[doc_id] = content

    if not corpus:
        raise ValueError("No valid documents found in the documents dataset.")
    return corpus
