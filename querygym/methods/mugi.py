from __future__ import annotations
from typing import List
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method


@register_method("mugi")
class MuGI(BaseReformulator):
    """
    MuGI (Multi-Text Generation Integration) Method

    Generates multiple diverse pseudo-documents per query and uses adaptive
    concatenation to balance term frequencies between short queries and long pseudo-docs.

    Key Parameters:
        - num_docs: Number of pseudo-documents to generate per query (default: 5)
        - adaptive_times: Divisor for adaptive repetition ratio (default: 5)
        - max_tokens: Maximum tokens per pseudo-document (default: 1024)
        - temperature: Sampling temperature for diversity (default: 1.0)
        - mode: "zs" for zero-shot or "fs" for few-shot (default: "zs")
        - prompt_id: (Advanced) Direct prompt ID override ("mugi.zeroshot.v1" or "mugi.fewshot.v1")
        - parallel: If True, generate all pseudo-docs in parallel; if False, sequential (default: False)

    Formula:
        repetition_times = (len(all_pseudo_docs) // len(query)) // adaptive_times
        enhanced_query = (query + ' ') * repetition_times + all_pseudo_docs
    """

    VERSION = "1.0"
    REQUIRES_CONTEXT = False
    CONCATENATION_STRATEGY = "adaptive_query_repeat_plus_generated"

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        # Get configurable parameters from config
        num_docs = int(self.cfg.params.get("num_docs", 5))
        adaptive_times = int(self.cfg.params.get("adaptive_times", 6))
        max_tokens = int(self.cfg.params.get("max_tokens", self.cfg.llm.get("max_tokens", 1024)))
        temperature = float(
            self.cfg.params.get("temperature", self.cfg.llm.get("temperature", 1.0))
        )
        parallel = bool(self.cfg.params.get("parallel", False))

        # Mode selection: "zs" (zero-shot) or "fs" (few-shot)
        mode = str(self.cfg.params.get("mode", "zs"))

        # Map mode to prompt_id (or allow direct prompt_id override for advanced users)
        if "prompt_id" in self.cfg.params:
            # Advanced: allow direct prompt ID specification
            prompt_id = str(self.cfg.params.get("prompt_id"))
        else:
            # Simple: use mode to select prompt
            if mode in ["fs", "fewshot"]:
                prompt_id = "mugi.fewshot.v1"
            elif mode in ["zs", "zeroshot"]:
                prompt_id = "mugi.zeroshot.v1"
            else:
                raise ValueError(
                    f"Invalid mode '{mode}' for MuGI. Must be 'zs' (zero-shot) or 'fs' (few-shot)."
                )

        # Generate multiple pseudo-documents for diversity and coverage
        pseudo_docs: List[str] = []

        if parallel:
            # Parallel generation using ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def generate_single_doc(doc_idx: int) -> str:
                msgs = self.prompts.render(prompt_id, query=q.text)
                pseudo_doc = self.llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)
                return pseudo_doc.strip().strip('"').strip("'")

            # Generate all pseudo-docs in parallel
            with ThreadPoolExecutor(max_workers=num_docs) as executor:
                futures = [executor.submit(generate_single_doc, i) for i in range(num_docs)]
                for future in as_completed(futures):
                    pseudo_docs.append(future.result())
        else:
            # Sequential generation (original behavior)
            for i in range(num_docs):
                # Render prompt (mugi.zeroshot.v1 or mugi.fewshot.v1)
                msgs = self.prompts.render(prompt_id, query=q.text)

                # Generate pseudo-document with high temperature for diversity
                pseudo_doc = self.llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)

                # Clean up the generated pseudo-document
                pseudo_doc = pseudo_doc.strip().strip('"').strip("'")
                pseudo_docs.append(pseudo_doc)

        # Join all pseudo-documents into single text
        all_pseudo_docs = " ".join(pseudo_docs)

        # Use base class concatenation with adaptive_query_repeat_plus_generated strategy
        # This automatically handles the MuGI formula and cleaning
        reformulated = self.concatenate_result(q.text, all_pseudo_docs)

        # Calculate metrics for metadata (after concatenation)
        query_len = len(q.text)
        docs_len = len(all_pseudo_docs)
        if query_len > 0:
            repetition_times = max(1, (docs_len // query_len) // adaptive_times)
        else:
            repetition_times = 1

        # Return result with metadata
        return ReformulationResult(
            q.qid,
            q.text,
            reformulated,
            metadata={
                "pseudo_docs": pseudo_docs,
                "num_docs": num_docs,
                "adaptive_times": adaptive_times,
                "repetition_times": repetition_times,
                "query_len": query_len,
                "docs_len": docs_len,
                "mode": mode,
                "prompt_id": prompt_id,
                "parallel": parallel,
                # Store individual pseudo-docs for analysis
                "pseudo_doc_1": pseudo_docs[0] if len(pseudo_docs) > 0 else "",
                "pseudo_doc_2": pseudo_docs[1] if len(pseudo_docs) > 1 else "",
                "pseudo_doc_3": pseudo_docs[2] if len(pseudo_docs) > 2 else "",
                "pseudo_doc_4": pseudo_docs[3] if len(pseudo_docs) > 3 else "",
                "pseudo_doc_5": pseudo_docs[4] if len(pseudo_docs) > 4 else "",
            },
        )
