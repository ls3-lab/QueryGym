from pathlib import Path

from querygym.core.base import MethodConfig, QueryItem
from querygym.core.prompts import PromptBank
from querygym.core.searcher import BaseSearcher, SearchHit
from querygym.methods.thinkqe import ThinkQE


class DummyLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def chat(self, messages, **kwargs):
        self.messages.append(messages)
        return self.responses.pop(0)


class DummySearcher(BaseSearcher):
    def __init__(self):
        self.queries = []

    def search(self, query: str, k: int = 10, **kwargs):
        self.queries.append(query)
        if "expansion one" in query:
            passages = [
                "Fresh support passage one",
                "Fresh support passage two",
                "Initial evidence passage one",
            ]
        else:
            passages = [
                "Initial evidence passage one",
                "Initial evidence passage two",
                "Initial evidence passage three",
            ]
        return [
            SearchHit(docid=f"d{idx}", score=1.0 / (idx + 1), content=passage)
            for idx, passage in enumerate(passages[:k])
        ]

    def batch_search(self, queries, k: int = 10, num_threads: int = 1, **kwargs):
        return [self.search(query, k=k, **kwargs) for query in queries]

    def get_searcher_info(self):
        return {"name": "dummy"}


def _prompt_bank():
    return PromptBank(Path(__file__).parents[1] / "querygym" / "prompt_bank.yaml")


def test_thinkqe_single_round_strips_thinking_trace():
    cfg = MethodConfig(
        name="thinkqe",
        params={"gen_num": 2, "repeat_weight": 3, "keep_passage_num": 2},
        llm={"model": "dummy", "temperature": 0.7, "max_tokens": 256},
    )
    llm = DummyLLM(
        [
            "<think>\nreasoning\n</think>\nAnswer Passage One",
            "<think>\nmore reasoning\n</think>\nAnswer Passage Two",
        ]
    )
    method = ThinkQE(cfg, llm, _prompt_bank())

    result = method.reformulate(
        QueryItem("q1", "test query"),
        ["Context one", "Context two", "Context three"],
    )

    assert result.metadata["expansions"] == [
        "Answer Passage One",
        "Answer Passage Two",
    ]
    assert result.metadata["used_ctx"] == 2
    assert "answer passage one" in result.reformulated
    assert "answer passage two" in result.reformulated


def test_thinkqe_multi_round_uses_searcher_and_records_history():
    cfg = MethodConfig(
        name="thinkqe",
        params={
            "searcher": DummySearcher(),
            "num_interaction": 2,
            "accumulate": True,
            "use_passage_filter": True,
            "gen_num": 1,
            "keep_passage_num": 2,
            "search_k": 3,
        },
        llm={"model": "dummy", "temperature": 0.7, "max_tokens": 256},
    )
    llm = DummyLLM(
        [
            "<think>\ntrace\n</think>\nExpansion One",
            "<think>\ntrace\n</think>\nExpansion Two",
        ]
    )
    method = ThinkQE(cfg, llm, _prompt_bank())

    results = method.reformulate_batch([QueryItem("q1", "original query")])
    result = results[0]
    round_history = result.metadata["round_history"]

    assert len(round_history) == 3
    assert round_history[0]["metadata"]["retrieved_count"] == 3
    assert round_history[-1]["metadata"]["accumulated_count"] == 2
    assert "expansion one" in result.reformulated
    assert "expansion two" in result.reformulated
    assert "original query" in llm.messages[0][0]["content"]
    assert "Initial evidence passage one" in llm.messages[0][0]["content"]
    assert "Fresh support passage one" in llm.messages[1][0]["content"]
