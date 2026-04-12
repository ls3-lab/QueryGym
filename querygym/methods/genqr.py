from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method
from typing import List


@register_method("genqr")
class GENQR(BaseReformulator):
    """
    GenQR: Generate reformulations N times and concatenate.

    Pipeline:
        1. Call LLM N times (default 5) to generate reformulations
        2. Concatenate: query + reformulation1 + reformulation2 + ... + reformulationN
    """

    VERSION = "1.0"

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        # Get parameters
        n_generations = int(self.cfg.params.get("n_generations", 5))
        temperature = float(self.cfg.llm.get("temperature", 0.8))
        max_tokens = int(self.cfg.llm.get("max_tokens", 256))

        # Generate reformulations N times (5 by default)
        reformulations: List[str] = []

        for _ in range(n_generations):
            msgs = self.prompts.render("genqr.keywords.v1", query=q.text)
            out = self.llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)
            # Clean up the reformulation (remove quotes, extra whitespace)
            reformulation = out.strip().strip('"').strip("'").replace("\n", " ")
            reformulations.append(reformulation)

        # Concatenate: query + reformulation1 + reformulation2 + ... + reformulationN
        reformulated = q.text + " " + " ".join(reformulations)

        return ReformulationResult(
            q.qid,
            q.text,
            reformulated,
            metadata={"n_generations": n_generations, "reformulations": reformulations},
        )
