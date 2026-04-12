from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method


@register_method("thinkqe")
class ThinkQE(BaseReformulator):
    """ThinkQE: query expansion via iterative corpus feedback.

    This implementation follows the original `archive/Think_QE/thinkqe.py`
    loop while fitting QueryGym's `BaseReformulator` interface:

    1. Round 0 retrieves passages with the original query.
    2. Each later round prompts the LLM with the original query plus the
       previous round's retrieved passages.
    3. The generated answering passage(s) are appended to the query.
    4. The updated query is used to retrieve again.

    The method strips any `<think>...</think>` trace and keeps only the
    answer passage for retrieval.
    """

    VERSION = "1.0"
    REQUIRES_CONTEXT = True

    def __init__(self, cfg, llm_client, prompt_resolver):
        super().__init__(cfg, llm_client, prompt_resolver)
        self._resolved_searcher = None
        self._searcher_initialized = False

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Return only the answer text after the thinking trace."""
        if "</think>\n" in text:
            return text.split("</think>\n")[-1].strip()
        if "</think>" in text:
            return text.split("</think>")[-1].strip()
        return text.strip()

    @staticmethod
    def _truncate_passage(passage: str, max_words: int) -> str:
        """Trim a passage by whitespace tokens."""
        words = passage.split()
        if len(words) <= max_words:
            return passage
        return " ".join(words[:max_words])

    @staticmethod
    def _build_query(
        original_query: str,
        expansions: List[str],
        repeat_weight: float,
        lowercase: bool,
    ) -> Tuple[str, int]:
        """Build the retrieval query using ThinkQE's repetition heuristic."""
        query_words = len(original_query.split())
        expansion_words = len("\n".join(expansions).split())

        if query_words > 0 and repeat_weight > 0:
            q_repeat = max(1, int(expansion_words / (query_words * repeat_weight)))
        else:
            q_repeat = 1

        reformulated = "\n".join([original_query] * q_repeat + expansions)
        if lowercase:
            reformulated = reformulated.lower()
        return reformulated, q_repeat

    @staticmethod
    def _filter_passages(
        passages: List[str],
        keep_k: int,
        seen: Set[str],
        prev_prev_top: Set[str],
    ) -> Tuple[List[str], Set[str]]:
        """Apply the archive-style novelty filter over retrieved passages."""
        filtered: List[str] = []
        for passage in passages:
            if passage in seen:
                continue
            if passage in prev_prev_top:
                seen.add(passage)
                continue
            filtered.append(passage)
            if len(filtered) >= keep_k:
                break
        return filtered, seen

    def _get_prompt_id(self) -> str:
        return str(self.cfg.params.get("prompt_id", "thinkqe.v1"))

    def _get_keep_passage_num(self) -> int:
        value = self.cfg.params.get(
            "keep_passage_num",
            self.cfg.params.get("retrieval_k", 5),
        )
        return int(value)

    def _get_gen_num(self) -> int:
        value = self.cfg.params.get(
            "gen_num",
            self.cfg.params.get("n_generations", 2),
        )
        return int(value)

    def _get_repeat_weight(self) -> float:
        value = self.cfg.params.get(
            "repeat_weight",
            self.cfg.params.get("reqeat_weight", 3),
        )
        return float(value)

    def _get_search_k(self) -> int:
        keep_passage_num = self._get_keep_passage_num()
        value = self.cfg.params.get(
            "search_k",
            self.cfg.params.get("hits", keep_passage_num),
        )
        return max(int(value), keep_passage_num)

    def _generate_expansions(
        self,
        query_text: str,
        contexts: List[str],
        gen_num: int,
        max_demo_len: Optional[int],
        temperature: float,
        max_tokens: int,
        no_thinking: bool,
    ) -> Tuple[List[str], List[str]]:
        """Generate `gen_num` expansions using repeated chat calls."""
        top_passages = contexts
        if max_demo_len is not None:
            top_passages = [
                self._truncate_passage(passage, int(max_demo_len)) for passage in top_passages
            ]

        contexts_blob = "\n".join(
            f"{index + 1}. {passage}" for index, passage in enumerate(top_passages)
        )
        messages = self.prompts.render(
            self._get_prompt_id(),
            query=query_text,
            contexts=contexts_blob,
        )

        if no_thinking:
            messages = list(messages)
            messages.append(
                {
                    "role": "assistant",
                    "content": "Okay, I think I have finished thinking.\n</think>\n",
                }
            )

        raw_responses: List[str] = []
        for _ in range(gen_num):
            response = self.llm.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_responses.append(response)

        expansions = [self._extract_answer(response) for response in raw_responses]
        return expansions, raw_responses

    def _build_searcher(self):
        """Resolve a searcher from config if multi-round retrieval is enabled."""
        if self._searcher_initialized:
            return self._resolved_searcher

        searcher = self.cfg.params.get("searcher")
        if searcher is not None:
            self._resolved_searcher = searcher
            self._searcher_initialized = True
            return self._resolved_searcher

        if "searcher_type" in self.cfg.params:
            from ..core.searcher import create_searcher

            self._resolved_searcher = create_searcher(
                self.cfg.params["searcher_type"],
                **dict(self.cfg.params.get("searcher_kwargs", {})),
            )
            self._searcher_initialized = True
            return self._resolved_searcher

        index = self.cfg.params.get("index")
        if not index:
            self._searcher_initialized = True
            return None

        from ..core.searcher import create_searcher

        searcher_kwargs: Dict[str, Any] = {
            "index": index,
            "answer_key": self.cfg.params.get("answer_key", "contents"),
        }

        k1 = self.cfg.params.get("k1")
        b = self.cfg.params.get("b")
        if k1 is not None and b is not None:
            searcher_kwargs["k1"] = k1
            searcher_kwargs["b"] = b

        self._resolved_searcher = create_searcher("pyserini", **searcher_kwargs)
        self._searcher_initialized = True
        return self._resolved_searcher

    def _build_retriever(self):
        """Create a callable that retrieves passage text for a query string."""
        searcher = self._build_searcher()
        if searcher is None:
            return None

        search_k = self._get_search_k()

        def _retriever(query_text: str) -> List[str]:
            hits = searcher.search(query_text, k=search_k)
            return [hit.content for hit in hits]

        return _retriever

    def _multi_round_enabled(self) -> bool:
        """Return True when ThinkQE can run its full iterative loop."""
        num_interaction = int(self.cfg.params.get("num_interaction", 3))
        return num_interaction > 0 and self._build_searcher() is not None

    @staticmethod
    def _summarize_round(result: ReformulationResult) -> Dict[str, Any]:
        """Create a compact per-round metadata summary."""
        meta = dict(result.metadata or {})
        raw_responses = meta.pop("raw_responses", None)
        summary = {
            "round": meta.pop("round", None),
            "type": meta.pop("type", "expanded"),
            "reformulated": result.reformulated,
            "metadata": meta,
        }
        if raw_responses is not None:
            summary["raw_response_count"] = len(raw_responses)
        return summary

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        """Run single-step ThinkQE using provided contexts."""
        ctxs: List[str] = contexts or []

        keep_passage_num = self._get_keep_passage_num()
        gen_num = self._get_gen_num()
        repeat_weight = self._get_repeat_weight()
        max_demo_len = self.cfg.params.get("max_demo_len")
        lowercase = bool(self.cfg.params.get("lowercase", True))
        no_thinking = bool(self.cfg.params.get("no_thinking", False))
        temperature = float(self.cfg.llm.get("temperature", 0.7))
        max_tokens = int(self.cfg.llm.get("max_tokens", 32768))

        top_passages = ctxs[:keep_passage_num]
        expansions, raw_responses = self._generate_expansions(
            q.text,
            top_passages,
            gen_num,
            max_demo_len,
            temperature,
            max_tokens,
            no_thinking,
        )
        reformulated, q_repeat = self._build_query(
            q.text,
            expansions,
            repeat_weight,
            lowercase,
        )

        return ReformulationResult(
            q.qid,
            q.text,
            reformulated,
            metadata={
                "prompt_id": self._get_prompt_id(),
                "gen_num": gen_num,
                "q_repeat": q_repeat,
                "repeat_weight": repeat_weight,
                "keep_passage_num": keep_passage_num,
                "used_ctx": len(top_passages),
                "expansions": expansions,
                "raw_responses": raw_responses,
            },
        )

    def reformulate_multi_round(
        self,
        q: QueryItem,
        retriever,
        *,
        num_interaction: Optional[int] = None,
        accumulate: Optional[bool] = None,
        use_passage_filter: Optional[bool] = None,
    ) -> List[ReformulationResult]:
        """Run the full ThinkQE loop and return per-round results."""
        if num_interaction is None:
            num_interaction = int(self.cfg.params.get("num_interaction", 3))
        if accumulate is None:
            accumulate = bool(self.cfg.params.get("accumulate", True))
        if use_passage_filter is None:
            use_passage_filter = bool(self.cfg.params.get("use_passage_filter", True))

        keep_passage_num = self._get_keep_passage_num()
        gen_num = self._get_gen_num()
        repeat_weight = self._get_repeat_weight()
        max_demo_len = self.cfg.params.get("max_demo_len")
        lowercase = bool(self.cfg.params.get("lowercase", True))
        no_thinking = bool(self.cfg.params.get("no_thinking", False))
        temperature = float(self.cfg.llm.get("temperature", 0.7))
        max_tokens = int(self.cfg.llm.get("max_tokens", 32768))

        accumulated_expansions: List[str] = []
        seen_passages: Set[str] = set()
        last_top_k: Set[str] = set()
        prev_prev_top_k: Set[str] = set()

        original_query_text = q.text
        current_query_text = q.text
        results: List[ReformulationResult] = []

        last_top_passages = retriever(current_query_text)
        last_top_k = set(last_top_passages[:keep_passage_num])
        results.append(
            ReformulationResult(
                q.qid,
                q.text,
                current_query_text,
                metadata={
                    "round": 0,
                    "type": "baseline",
                    "retrieved_count": len(last_top_passages),
                },
            )
        )

        for ridx in range(1, num_interaction + 1):
            all_passages = last_top_passages
            if use_passage_filter:
                top_passages, seen_passages = self._filter_passages(
                    all_passages,
                    keep_passage_num,
                    seen_passages,
                    prev_prev_top_k,
                )
            else:
                top_passages = all_passages[:keep_passage_num]

            expansions, raw_responses = self._generate_expansions(
                original_query_text,
                top_passages,
                gen_num,
                max_demo_len,
                temperature,
                max_tokens,
                no_thinking,
            )

            if accumulate:
                accumulated_expansions.extend(expansions)
                active_expansions = list(accumulated_expansions)
            else:
                active_expansions = expansions

            reformulated, q_repeat = self._build_query(
                original_query_text,
                active_expansions,
                repeat_weight,
                lowercase,
            )
            current_query_text = reformulated
            retrieved_passages = retriever(current_query_text)

            prev_prev_top_k = last_top_k
            last_top_k = set(retrieved_passages[:keep_passage_num])
            last_top_passages = retrieved_passages

            results.append(
                ReformulationResult(
                    q.qid,
                    q.text,
                    reformulated,
                    metadata={
                        "round": ridx,
                        "prompt_id": self._get_prompt_id(),
                        "gen_num": gen_num,
                        "q_repeat": q_repeat,
                        "repeat_weight": repeat_weight,
                        "keep_passage_num": keep_passage_num,
                        "used_ctx": len(top_passages),
                        "search_k": self._get_search_k(),
                        "accumulate": accumulate,
                        "accumulated_count": len(active_expansions),
                        "use_passage_filter": use_passage_filter,
                        "expansions": expansions,
                        "raw_responses": raw_responses,
                        "retrieved_count": len(retrieved_passages),
                    },
                )
            )

        return results

    def reformulate_with_round_history(
        self,
        q: QueryItem,
        contexts=None,
    ) -> Tuple[ReformulationResult, List[ReformulationResult]]:
        """Return the final result together with all round outputs."""
        retriever = self._build_retriever()
        if int(self.cfg.params.get("num_interaction", 3)) > 0 and retriever is not None:
            round_results = self.reformulate_multi_round(q, retriever)
            final_result = round_results[-1]
            final_metadata = dict(final_result.metadata or {})
            final_metadata["round_history"] = [
                self._summarize_round(result) for result in round_results
            ]
            final_result = ReformulationResult(
                final_result.qid,
                final_result.original,
                final_result.reformulated,
                metadata=final_metadata,
            )
            return final_result, round_results

        single_result = self.reformulate(q, contexts)
        single_metadata = dict(single_result.metadata or {})
        single_metadata["round_history"] = [self._summarize_round(single_result)]
        single_result = ReformulationResult(
            single_result.qid,
            single_result.original,
            single_result.reformulated,
            metadata=single_metadata,
        )
        return single_result, [single_result]

    def reformulate_batch(
        self,
        queries: List[QueryItem],
        ctx_map=None,
    ) -> List[ReformulationResult]:
        """Auto-dispatch to the multi-round loop when retrieval is available."""
        if self._multi_round_enabled():
            results: List[ReformulationResult] = []
            for query in queries:
                final_result, _ = self.reformulate_with_round_history(query)
                results.append(final_result)
            return results
        return super().reformulate_batch(queries, ctx_map)

    def _get_retrieval_params(self) -> Optional[Dict[str, Any]]:
        """Return retrieval settings for single-step batch reformulation."""
        keep_passage_num = self._get_keep_passage_num()

        if "searcher" in self.cfg.params:
            return {
                "searcher": self.cfg.params["searcher"],
                "k": keep_passage_num,
                "threads": int(self.cfg.params.get("threads", 16)),
            }

        if "searcher_type" in self.cfg.params:
            searcher_kwargs = dict(self.cfg.params.get("searcher_kwargs", {}))
            if "index" in self.cfg.params and "index" not in searcher_kwargs:
                searcher_kwargs["index"] = self.cfg.params["index"]
            return {
                "searcher_type": self.cfg.params["searcher_type"],
                "searcher_kwargs": searcher_kwargs,
                "k": keep_passage_num,
                "threads": int(self.cfg.params.get("threads", 16)),
            }

        index = self.cfg.params.get("index")
        if not index:
            return None

        searcher_kwargs: Dict[str, Any] = {
            "index": index,
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
            "k": keep_passage_num,
            "threads": int(self.cfg.params.get("threads", 16)),
        }
