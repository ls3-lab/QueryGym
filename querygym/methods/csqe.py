from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method
from typing import List, Optional, Dict, Any
import re


@register_method("csqe")
class CSQE(BaseReformulator):
    """
    CSQE (Context-based Sentence-level Query Expansion) method.

    Combines KEQE (knowledge-based) and CSQE (context-based) expansions:
    1. Generates N KEQE passages from LLM knowledge (no contexts) - default N=2
    2. Generates N CSQE responses with retrieved contexts (few-shot) - default N=2
    3. Extracts key sentences from CSQE responses
    4. Concatenates: query × N + keqe_passages + csqe_sentences (newline-separated, lowercased)

    Default: N=2 for both, totaling 4 generations (2 KEQE + 2 CSQE).
    Query is repeated N times (equal to number of KEQE expansions).
    """

    VERSION = "1.0"
    REQUIRES_CONTEXT = True  # Needs retrieved contexts for CSQE expansion

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        """Reformulate query using CSQE method (KEQE + CSQE)."""
        # Use provided contexts (from batch retrieval)
        ctxs = contexts or []

        # Prepare contexts for CSQE prompting
        retrieval_k = int(self.cfg.params.get("retrieval_k", 10))
        prompt_ctxs = ctxs[:retrieval_k]
        contexts_blob = "\n".join([f"{i+1}. {psg}" for i, psg in enumerate(prompt_ctxs)])

        # Get generation parameters - N=2 for both by default (paper specification)
        n_expansions = int(
            self.cfg.params.get("gen_num", 2)
        )  # Number of expansions for both KEQE and CSQE (default: 2)
        max_tokens = int(self.cfg.llm.get("max_tokens", 1024))
        temperature = float(self.cfg.llm.get("temperature", 1.0))

        # Step 1: KEQE expansion - Generate N passages from knowledge (no contexts) in one call
        msgs_keqe = self.prompts.render("keqe.v1", query=q.text)
        # Use n parameter to generate all passages in one call (matching new_baselines.py)
        resp_keqe = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=msgs_keqe,
            n=n_expansions,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        keqe_passages = [
            choice.message.content.strip().strip('"').strip("'") or ""
            for choice in resp_keqe.choices
        ]

        # Step 2: CSQE expansion - Generate N responses with contexts (few-shot) in one call
        msgs_csqe = self.prompts.render("csqe.v1", query=q.text, contexts=contexts_blob)
        # Use n parameter to generate all responses in one call (matching new_baselines.py)
        resp_csqe = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=msgs_csqe,
            n=n_expansions,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        csqe_responses = [choice.message.content or "" for choice in resp_csqe.choices]
        # Step 3: Extract key sentences from CSQE responses
        # Pattern matches sentences in double quotes (from new_baselines.py)
        csqe_sentences: List[str] = []
        for response in csqe_responses:
            extracted = self._extract_key_sentences(response)
            csqe_sentences.append(extracted)

        # Step 4: Concatenate directly (query × N + keqe_passages + csqe_sentences)
        # Format: query repeated N times (N = number of KEQE passages),
        # then all KEQE passages, then all extracted CSQE sentences, space-separated and lowercased
        parts = [q.text] * n_expansions + keqe_passages + csqe_sentences
        reformulated = " ".join(parts).lower().strip()
        # Clean up quotes
        reformulated = reformulated.strip('"').strip("'")

        return ReformulationResult(
            q.qid,
            q.text,
            reformulated,
            metadata={
                "keqe_passages": keqe_passages,
                "csqe_responses": csqe_responses,
                "csqe_sentences": csqe_sentences,
                "gen_num": n_expansions,
                "total_generations": n_expansions * 2,  # KEQE + CSQE
                "used_ctx": len(prompt_ctxs),
            },
        )

    def _extract_key_sentences(self, response: str) -> str:
        """
        Extract key sentences from CSQE response.

        Primary: Uses regex to find sentences in double quotes (expected format).
        Fallback: If no quotes found, extracts content after numbered document markers (1., 2., etc.)
        """
        # Primary strategy: Extract sentences in double quotes
        pattern = r'"([^"]*)"'
        sentences = re.findall(pattern, response)

        if sentences:
            # Found quoted sentences - join and return
            joint_sentence = " ".join(sentences)
            return joint_sentence

        # Fallback: LLM didn't use quotes, extract from numbered document format
        # Pattern: Look for content after "1.", "2.", etc. (with optional trailing colon)
        # Remove "Relevant Documents:" header first
        cleaned = re.sub(
            r"^Relevant Documents?:?\s*\n?", "", response, flags=re.IGNORECASE | re.MULTILINE
        )

        # Extract content after numbered markers (1., 2., etc.)
        doc_pattern = r"\d+[\.:]\s*(.+?)(?=\d+[\.:]|$)"
        doc_content = re.findall(doc_pattern, cleaned, re.DOTALL)

        if doc_content:
            # Clean up each extracted piece
            extracted = []
            for content in doc_content:
                # Remove extra whitespace and newlines, limit length
                cleaned_content = " ".join(content.split())
                if cleaned_content:
                    extracted.append(cleaned_content)
            if extracted:
                return " ".join(extracted)

        # If still nothing found, return empty string
        return ""

    def _get_retrieval_params(self) -> Optional[Dict[str, Any]]:
        """Get CSQE-specific retrieval parameters for batch retrieval.

        Returns retrieval parameters compatible with BaseSearcher interface.
        Supports both searcher_type/searcher_kwargs format and legacy format.
        """
        # Check if searcher is provided directly (wrappers, adapter instances, etc.)
        if "searcher" in self.cfg.params:
            # Direct searcher instance provided
            return {
                "searcher": self.cfg.params["searcher"],
                "k": int(self.cfg.params.get("retrieval_k", 10)),
                "threads": int(self.cfg.params.get("threads", 16)),
            }

        # Check if using new searcher interface format with searcher_type
        if "searcher_type" in self.cfg.params:
            # New format: use searcher_type and searcher_kwargs
            searcher_type = self.cfg.params.get("searcher_type", "pyserini")
            searcher_kwargs = self.cfg.params.get("searcher_kwargs", {})

            # If index is provided in params but not in searcher_kwargs, add it
            if "index" in self.cfg.params and "index" not in searcher_kwargs:
                searcher_kwargs["index"] = self.cfg.params["index"]

            return {
                "searcher_type": searcher_type,
                "searcher_kwargs": searcher_kwargs,
                "k": int(self.cfg.params.get("retrieval_k", 10)),
                "threads": int(self.cfg.params.get("threads", 16)),
            }

        # Legacy format: convert old-style params to new format
        index = self.cfg.params.get("index")
        if not index:
            return None

        # Build searcher_kwargs from legacy params
        searcher_kwargs = {
            "index": index,
            "searcher_type": "impact" if self.cfg.params.get("impact", False) else "bm25",
            "answer_key": self.cfg.params.get("answer_key", "contents"),
        }

        # Add BM25 parameters if specified
        k1 = self.cfg.params.get("k1")
        b = self.cfg.params.get("b")
        if k1 is not None and b is not None:
            searcher_kwargs["k1"] = k1
            searcher_kwargs["b"] = b

        # Add RM3 if requested
        if self.cfg.params.get("rm3", False):
            searcher_kwargs["rm3"] = True

        # Add Rocchio if requested
        if self.cfg.params.get("rocchio", False):
            searcher_kwargs["rocchio"] = True
            if self.cfg.params.get("rocchio_use_negative", False):
                searcher_kwargs["rocchio_use_negative"] = True

        return {
            "searcher_type": "pyserini",  # Default to pyserini for legacy format
            "searcher_kwargs": searcher_kwargs,
            "k": int(self.cfg.params.get("retrieval_k", 10)),
            "threads": int(self.cfg.params.get("threads", 16)),
        }
