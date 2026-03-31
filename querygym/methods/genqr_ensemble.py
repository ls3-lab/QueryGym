from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method
from typing import List

# 10 instruction variants from GenQREnsemble paper
VARIANT_IDS = [
    "genqr_ensemble.inst1.v1",
    "genqr_ensemble.inst2.v1",
    "genqr_ensemble.inst3.v1",
    "genqr_ensemble.inst4.v1",
    "genqr_ensemble.inst5.v1",
    "genqr_ensemble.inst6.v1",
    "genqr_ensemble.inst7.v1",
    "genqr_ensemble.inst8.v1",
    "genqr_ensemble.inst9.v1",
    "genqr_ensemble.inst10.v1",
]


@register_method("genqr_ensemble")
class GenQREnsemble(BaseReformulator):
    """
    GenQREnsemble: Generative Query Reformulation with Ensemble of Instructions.

    Uses 10 instruction variants to generate diverse keyword expansions.
    Each variant generates keywords independently, then all are merged.

    Pipeline:
        1. For each of 10 instruction variants, generate keywords (10 LLM calls)
        2. Parse and merge all keyword lists
        3. Expand query: (Q × 5) + K1 + K2 + ... + K10

    Sampling Parameters (from paper):
        - temperature: 0.92 (nucleus sampling)
        - max_tokens: 256
        - parallel: false (can be enabled for concurrent generation)

    Config Parameters:
        - variant_ids: List of prompt IDs (defaults to all 10 variants)
        - parallel: Enable parallel generation of all variants (default: false)
    """

    VERSION = "1.0"
    CONCATENATION_STRATEGY = "query_repeat_plus_generated"  # (Q × 5) + K1 + K2 + ... + K10
    DEFAULT_QUERY_REPEATS = 5  # GenQREnsemble uses 5 repetitions

    def _parse_keywords(self, keywords_text: str) -> List[str]:
        """
        Parse keywords from LLM output (comma-separated or bullet list).

        Args:
            keywords_text: Raw LLM output

        Returns:
            List of individual keywords
        """
        keywords = []

        # Check if it's a bullet list (contains dashes/newlines)
        if "\n" in keywords_text or keywords_text.strip().startswith("-"):
            # Split by newlines and extract after dashes
            for line in keywords_text.split("\n"):
                line = line.strip()
                # Remove leading dash/bullet
                if line.startswith("-"):
                    line = line[1:].strip()
                elif line.startswith("•"):
                    line = line[1:].strip()
                elif line.startswith("*"):
                    line = line[1:].strip()

                if line:
                    keywords.append(line)
        else:
            # Split by comma (standard format)
            keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]

        return keywords

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        """Generate keywords using 10 instruction variants and merge."""
        # Get parameters (matching baseline sampling params)
        temperature = float(self.cfg.llm.get("temperature", 0.92))
        max_tokens = int(self.cfg.llm.get("max_tokens", 256))
        parallel = bool(self.cfg.params.get("parallel", False))

        # Get variant IDs (configurable, defaults to all 10)
        variant_ids = self.cfg.params.get("variant_ids", VARIANT_IDS)

        # Generate keywords for each instruction variant
        all_keywords = []
        variant_outputs = {}

        if parallel:
            # Parallel generation using ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def generate_keywords_for_variant(idx: int, prompt_id: str):
                msgs = self.prompts.render(prompt_id, query=q.text)
                output = self.llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)
                keywords = self._parse_keywords(output)
                return idx, prompt_id, output, keywords

            # Generate all variants in parallel
            with ThreadPoolExecutor(max_workers=len(variant_ids)) as executor:
                futures = [
                    executor.submit(generate_keywords_for_variant, i, prompt_id)
                    for i, prompt_id in enumerate(variant_ids, 1)
                ]
                for future in as_completed(futures):
                    i, prompt_id, output, keywords = future.result()
                    all_keywords.extend(keywords)
                    variant_outputs[f"variant_{i}"] = {
                        "prompt_id": prompt_id,
                        "raw_output": output,
                        "keywords": keywords,
                        "count": len(keywords),
                    }
        else:
            # Sequential generation (default)
            for i, prompt_id in enumerate(variant_ids, 1):
                # Render prompt and generate keywords
                msgs = self.prompts.render(prompt_id, query=q.text)
                output = self.llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)

                # Parse keywords from output
                keywords = self._parse_keywords(output)
                all_keywords.extend(keywords)

                # Store per-variant output for metadata
                variant_outputs[f"variant_{i}"] = {
                    "prompt_id": prompt_id,
                    "raw_output": output,
                    "keywords": keywords,
                    "count": len(keywords),
                }

        # Merge all keywords into single string
        generated_content = " ".join(all_keywords)

        # Use base class concatenation: (Q × 5) + generated_content
        reformulated = self.concatenate_result(q.text, generated_content)

        return ReformulationResult(
            q.qid,
            q.text,
            reformulated,
            metadata={
                "num_variants": len(variant_ids),
                "total_keywords": len(all_keywords),
                "keywords": all_keywords,
                "variant_outputs": variant_outputs,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "parallel": parallel,
            },
        )
