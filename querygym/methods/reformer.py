from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method

# Default pattern library extracted from the Diamond dataset.
# Can be overridden at runtime via params["patterns_path"].
DEFAULT_PATTERNS: List[Dict[str, Any]] = [
    {
        "pattern_name": "Semantic Clarification",
        "description": "Clarifies ambiguous terms or actions to make the query more specific and understandable.",
        "transformation_rule": "Replace vague or ambiguous terms with more precise and contextually relevant terms.",
        "examples": [
            ["weather in prague in may", "temperature in prague in may"],
            ["ponds brand is which company", "who is pond's parent company"],
        ],
    },
    {
        "pattern_name": "Contextual Expansion",
        "description": "Expands the query to include additional context or details that can help refine the search results.",
        "transformation_rule": "Add relevant contextual information to the original query to make it more specific.",
        "examples": [
            ["weather in quito ecuador", "average temperature in quito mexico"],
            ["elmhurst ny is what county", "what is elmhurst ny's tax"],
        ],
    },
    {
        "pattern_name": "Generalization",
        "description": "Broadens the query to cover a wider range of related topics or scenarios.",
        "transformation_rule": "Replace specific terms with more general ones that encompass a broader category.",
        "examples": [
            ["name four countries that have a desert biome", "which kind of desert is found in the nevada desert"],
        ],
    },
    {
        "pattern_name": "Location Specification",
        "description": "Clarifies the location or geographical context of the query.",
        "transformation_rule": "Specify the exact location or region to make the query more targeted.",
        "examples": [
            ["weather in quito ecuador", "average temperature in quito mexico"],
            ["elmhurst ny is what county", "what is elmhurst ny's tax"],
        ],
    },
    {
        "pattern_name": "Purpose Specification",
        "description": "Clarifies the purpose or intended use of a product, service, or concept.",
        "transformation_rule": "Specify the intended use or application to make the query more focused.",
        "examples": [
            ["how do i format a timeline in ms-project 2013", "what is the timeline format in microsoft project"],
        ],
    },
    {
        "pattern_name": "Temporal Adjustment",
        "description": "Adjusts the temporal aspect of the query to reflect a more accurate or relevant time frame.",
        "transformation_rule": "Modify the time-related terms to better align with the user's intent.",
        "examples": [
            ["when kids start teething", "when should baby start teething"],
        ],
    },
    {
        "pattern_name": "Conceptual Shift",
        "description": "Transforms the query by shifting the focus from one concept to a related but different concept.",
        "transformation_rule": "Replace the main subject or action with a closely related concept that better aligns with the user's intent.",
        "examples": [
            ["what is sociometric popularity", "what is sociometric status psychology"],
        ],
    },
    {
        "pattern_name": "Clarify Intent",
        "description": "Clarifies the underlying intent or goal of the query to make it more direct and actionable.",
        "transformation_rule": "Rephrase the query to directly address the user's intended action or information need.",
        "examples": [
            ["why goji berries?", "how much do goji berries cost a pound"],
            ["is the thyroid affected by a hysterectomy", "after thyroid removal do i develop thyroid problems"],
        ],
    },
    {
        "pattern_name": "Contextual Restriction",
        "description": "Narrows down the context of the query to a more specific scenario or condition.",
        "transformation_rule": "Limit the scope of the query by adding constraints or conditions that refine the search results.",
        "examples": [
            ["weather in quito ecuador", "average temperature in quito mexico"],
        ],
    },
    {
        "pattern_name": "Clarify Subject",
        "description": "Clarifies the main subject of the query to ensure it is accurately represented.",
        "transformation_rule": "Replace or refine the main subject to better reflect the user's intent.",
        "examples": [
            ["ponds brand is which company", "who is pond's parent company"],
        ],
    },
]


@register_method("reformer")
class ReFormeR(BaseReformulator):
    """
    Pattern-based, document-conditioned query reformulation.

    For each query, selects the best-fitting reformulation pattern from a
    pre-learned library (two LLM calls: selection + application), using
    top-k retrieved documents as context for pattern selection.
    """

    VERSION = "1.0"
    REQUIRES_CONTEXT = True
    # The LLM produces a replacement query, not an expansion — output is used directly.
    CONCATENATION_STRATEGY = "generated_only"

    def __init__(self, cfg, llm, pb):
        super().__init__(cfg, llm, pb)
        patterns_path = self.cfg.params.get("patterns_path")
        if patterns_path:
            with open(patterns_path) as f:
                self._patterns: List[Dict[str, Any]] = json.load(f)
        else:
            self._patterns = DEFAULT_PATTERNS

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        ctxs = contexts or []
        context_docs = int(self.cfg.params.get("context_docs", 3))
        temperature = float(self.cfg.llm.get("temperature", 0.1))
        max_tokens = int(self.cfg.llm.get("max_tokens", 500))

        # Step 1 — select the best-fitting pattern given query + top docs
        documents_text = "\n\n".join(ctxs[:context_docs])
        selection_msgs = self.prompts.render(
            "reformer.pattern_selection.v1",
            query=q.text,
            documents_text=documents_text,
            patterns_json=json.dumps(self._patterns, indent=2),
        )
        raw_selection = self.llm.chat(
            selection_msgs, temperature=temperature, max_tokens=max_tokens
        )

        fallback = False
        try:
            selected_pattern: Dict[str, Any] = json.loads(raw_selection)
        except (json.JSONDecodeError, ValueError):
            selected_pattern = self._patterns[0]
            fallback = True

        # Step 2 — apply the selected pattern's transformation rule
        application_msgs = self.prompts.render(
            "reformer.pattern_application.v1",
            query=q.text,
            transformation_rule=selected_pattern.get("transformation_rule", ""),
            examples_json=json.dumps(selected_pattern.get("examples", []), indent=2),
        )
        reformulated = self.llm.chat(
            application_msgs, temperature=temperature, max_tokens=max_tokens
        ).strip()

        if not reformulated:
            reformulated = q.text

        return ReformulationResult(
            qid=q.qid,
            original=q.text,
            reformulated=reformulated,
            metadata={
                "selected_pattern": selected_pattern.get("pattern_name", "Unknown"),
                "pattern_fallback": fallback,
                "used_ctx": len(ctxs[:context_docs]),
            },
        )

    def _get_retrieval_params(self) -> Optional[Dict[str, Any]]:
        if "searcher" in self.cfg.params:
            return {
                "searcher": self.cfg.params["searcher"],
                "k": int(self.cfg.params.get("retrieval_k", 10)),
                "threads": int(self.cfg.params.get("threads", 16)),
            }

        if "searcher_type" in self.cfg.params:
            searcher_type = self.cfg.params.get("searcher_type", "pyserini")
            searcher_kwargs = self.cfg.params.get("searcher_kwargs", {})
            if "index" in self.cfg.params and "index" not in searcher_kwargs:
                searcher_kwargs["index"] = self.cfg.params["index"]
            return {
                "searcher_type": searcher_type,
                "searcher_kwargs": searcher_kwargs,
                "k": int(self.cfg.params.get("retrieval_k", 10)),
                "threads": int(self.cfg.params.get("threads", 16)),
            }

        index = self.cfg.params.get("index")
        if not index:
            return None

        searcher_kwargs: Dict[str, Any] = {
            "index": index,
            "searcher_type": "impact" if self.cfg.params.get("impact", False) else "bm25",
            "answer_key": self.cfg.params.get("answer_key", "contents"),
        }
        k1 = self.cfg.params.get("k1")
        b = self.cfg.params.get("b")
        if k1 is not None and b is not None:
            searcher_kwargs["k1"] = k1
            searcher_kwargs["b"] = b

        return {
            "searcher_type": "pyserini",
            "searcher_kwargs": searcher_kwargs,
            "k": int(self.cfg.params.get("retrieval_k", 10)),
            "threads": int(self.cfg.params.get("threads", 16)),
        }
