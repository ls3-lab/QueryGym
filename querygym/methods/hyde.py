from __future__ import annotations

from typing import Dict, List, Optional

from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method

DEFAULT_TASK = "web search"

# Task-specific prompts from the original HyDE implementation (promptor.py).
# Mr. TyDi (multilingual) is not included.
TASK_PROMPTS: Dict[str, str] = {
    "web search": "hyde.web_search.v1",
    "scifact": "hyde.scifact.v1",
    "arguana": "hyde.arguana.v1",
    "trec-covid": "hyde.trec_covid.v1",
    "fiqa": "hyde.fiqa.v1",
    "dbpedia-entity": "hyde.dbpedia_entity.v1",
    "trec-news": "hyde.trec_news.v1",
}

# Datasets with a dedicated HyDE prompt. Every other dataset (MS MARCO, DL-HARD,
# BRIGHT, custom query files) uses the web search prompt.
DATASET_TASKS: Dict[str, str] = {
    "beir-v1.0.0-scifact": "scifact",
    "beir-v1.0.0-arguana": "arguana",
    "beir-v1.0.0-trec-covid": "trec-covid",
    "beir-v1.0.0-fiqa": "fiqa",
    "beir-v1.0.0-dbpedia-entity": "dbpedia-entity",
    "beir-v1.0.0-trec-news": "trec-news",
}


def task_for_dataset(dataset_id: Optional[str]) -> str:
    """Return the HyDE task for a dataset id, defaulting to web search.

    Args:
        dataset_id: Dataset id from dataset_registry.yaml, or None for custom query files

    Returns:
        HyDE task name (a key of TASK_PROMPTS)

    Example:
        >>> task_for_dataset("beir-v1.0.0-scifact")
        'scifact'
        >>> task_for_dataset("bright-biology")
        'web search'
    """
    return DATASET_TASKS.get(dataset_id or "", DEFAULT_TASK)


@register_method("hyde")
class HyDE(BaseReformulator):
    """
    HyDE: Hypothetical Document Embeddings, from "Precise Zero-Shot Dense Retrieval
    without Relevance Labels" (Gao et al., 2022).

    Text adaptation of HyDE: instead of averaging embeddings, the hypothetical
    documents are concatenated with the original query, so the output works with
    any retriever (BM25, SPLADE++, dense).

    Pipeline:
        1. Build the task-specific prompt (params["task"], default "web search")
        2. Call LLM n_docs times (default 8) to generate hypothetical documents
        3. Concatenate: query + doc_1 + doc_2 + ... + doc_n
    """

    VERSION = "1.0"
    REQUIRES_CONTEXT = False
    CONCATENATION_STRATEGY = "query_repeat_plus_generated"
    # HyDE weights the query as one of n_docs + 1 inputs, so it appears once.
    DEFAULT_QUERY_REPEATS = 1

    def __init__(self, cfg, llm, pb):
        super().__init__(cfg, llm, pb)
        task = self.cfg.params.get("task", DEFAULT_TASK)
        if task not in TASK_PROMPTS:
            raise ValueError(f"Unknown HyDE task: {task!r}. Available: {list(TASK_PROMPTS.keys())}")
        self.task = task

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        n_docs = int(self.cfg.params.get("n_docs", 8))
        temperature = float(self.cfg.llm.get("temperature", 0.7))
        max_tokens = int(self.cfg.llm.get("max_tokens", 512))
        llm_kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        # top_p and stop are sent only when configured; not every backend accepts them
        if self.cfg.llm.get("top_p") is not None:
            llm_kwargs["top_p"] = float(self.cfg.llm["top_p"])
        if self.cfg.llm.get("stop") is not None:
            llm_kwargs["stop"] = self.cfg.llm["stop"]

        msgs = self.prompts.render(TASK_PROMPTS[self.task], query=q.text)

        hypothetical_docs: List[str] = []
        for _ in range(n_docs):
            out = self.llm.chat(msgs, **llm_kwargs)
            doc = out.strip()
            if doc:
                hypothetical_docs.append(doc)

        if hypothetical_docs:
            reformulated = self.concatenate_result(q.text, hypothetical_docs)
        else:
            reformulated = q.text

        return ReformulationResult(
            q.qid,
            q.text,
            reformulated,
            metadata={
                "task": self.task,
                "n_docs": n_docs,
                "hypothetical_docs": hypothetical_docs,
            },
        )
