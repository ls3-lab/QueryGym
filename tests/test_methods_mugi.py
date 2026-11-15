from pathlib import Path

from querygym.core.base import MethodConfig, QueryItem
from querygym.core.prompts import PromptBank
from querygym.methods.mugi import MuGI

PROMPT_BANK = Path(__file__).parents[1] / "querygym" / "prompt_bank.yaml"
QUERY = "abcd"
# Five 23-char docs joined by spaces: 119 chars, 119 // 4 = 29 -> 29 // 6 = 4, 29 // 5 = 5
DOC = "a" * 23
DOCS = " ".join([DOC] * 5)


class DummyLLM:
    def __init__(self, response):
        self.response = response

    def chat(self, messages, **kwargs):
        return self.response


def make_mugi(**params):
    cfg = MethodConfig(name="mugi", params=params, llm={"model": "dummy"})
    return MuGI(cfg, DummyLLM(DOC), PromptBank(PROMPT_BANK))


def test_default_adaptive_times_is_six():
    res = make_mugi().reformulate(QueryItem("q1", QUERY))

    assert res.reformulated == (QUERY + " ") * 4 + DOCS
    assert res.metadata["adaptive_times"] == 6
    assert res.metadata["repetition_times"] == 4


def test_adaptive_times_param():
    res = make_mugi(adaptive_times=5).reformulate(QueryItem("q1", QUERY))

    assert res.reformulated == (QUERY + " ") * 5 + DOCS
    assert res.metadata["adaptive_times"] == 5
    assert res.metadata["repetition_times"] == 5


def test_query_dropped_when_docs_are_short():
    # 29 // 30 = 0: the expanded query is the pseudo-documents alone
    res = make_mugi(adaptive_times=30).reformulate(QueryItem("q1", QUERY))

    assert res.reformulated == DOCS
    assert res.metadata["repetition_times"] == 0


def test_empty_query():
    res = make_mugi().reformulate(QueryItem("q1", ""))

    assert res.reformulated == DOCS
    assert res.metadata["repetition_times"] == 0
