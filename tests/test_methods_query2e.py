from querygym.core.base import MethodConfig, QueryItem
from querygym.core.prompts import PromptBank
from querygym.core.utils import strip_think_trace
from querygym.methods.query2e import Query2E
from pathlib import Path
import pytest

PROMPT_BANK = Path(__file__).parents[1] / "querygym" / "prompt_bank.yaml"
QUERY = "what is prime rate in canada"

CATEGORIZED = """Here is a list of relevant keywords and phrases for the query "what is prime rate in canada":

**Core Concepts**
* Prime Rate
* Prime Lending Rate

**Canadian Context**
* Bank of Canada
* Overnight rate
"""


class DummyLLM:
    def __init__(self, response):
        self.response = response

    def chat(self, messages, **kwargs):
        return self.response


def make_q2e(response="", **params):
    cfg = MethodConfig(name="query2e", params=params, llm={"model": "dummy"})
    return Query2E(cfg, DummyLLM(response), PromptBank(PROMPT_BANK))


def test_default_removes_chat_framing():
    res = make_q2e(CATEGORIZED).reformulate(QueryItem("q1", QUERY))

    assert res.metadata["keywords"] == [
        "Prime Rate",
        "Prime Lending Rate",
        "Bank of Canada",
        "Overnight rate",
    ]
    assert res.metadata["clean_output"] is True
    assert "Here is" not in res.reformulated


def test_clean_output_false_parses_raw_output():
    meth = make_q2e(CATEGORIZED, clean_output=False)
    res = meth.reformulate(QueryItem("q1", QUERY))

    terms = meth._parse_keywords(CATEGORIZED)
    assert res.metadata == {
        "mode": "zs",
        "prompt_id": "q2e.zs.v1",
        "keywords": terms,
        "temperature": 0.3,
        "max_tokens": 256,
    }
    assert res.reformulated == meth.concatenate_result(QUERY, " ".join(terms))


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            'Sure! Here is a list of keywords for the query **"what is prime rate in canada":**\n\n'
            "- prime rate\n- Bank of Canada\n\nLet me know if you need a more targeted list!",
            ["prime rate", "Bank of Canada"],
        ),
        (
            "### Core Medical Procedures\n- Sclerotherapy\n- Endovenous laser ablation (EVLA)",
            ["Sclerotherapy", "Endovenous laser ablation (EVLA)"],
        ),
        (
            "**Primary keywords:** prime rate, Canada\n**Related terms:** overnight rate",
            ["prime rate", "Canada", "overnight rate"],
        ),
        (
            "1. **Prime rate** – the base lending rate\n2. **Bank of Canada** – central bank",
            ["Prime rate", "Bank of Canada"],
        ),
        (
            "Okay, here's a breakdown:\n\n1. Core Concepts\n   - Prime rate\n   - Lending rate",
            ["Prime rate", "Lending rate"],
        ),
        (
            "For this query, useful keywords include prime rate, Bank of Canada, and lending rate.",
            ["prime rate", "Bank of Canada", "lending rate"],
        ),
        (
            "Keywords: prime rate, Bank of Canada, lending rate",
            ["prime rate", "Bank of Canada", "lending rate"],
        ),
        (
            "| Category | Keywords |\n|---|---|\n| Core | prime rate, lending rate |",
            ["prime rate", "lending rate"],
        ),
        ('```json\n["prime rate", "bank of canada"]\n```', ["prime rate", "bank of canada"]),
        (
            "prime rate, Bank of Canada\n\nWould you like me to add related terms?",
            ["prime rate", "Bank of Canada"],
        ),
        (
            "okay google, voice assistant, smart speaker",
            ["okay google", "voice assistant", "smart speaker"],
        ),
        ("- 2024\n- 401k\n- 5-year fixed", ["2024", "401k", "5-year fixed"]),
        (
            "Here is a list of keywords you can use: prime rate, bank of canada, lending rate, "
            "overnight rate, mortgage rate, interest rate.",
            [
                "prime rate",
                "bank of canada",
                "lending rate",
                "overnight rate",
                "mortgage rate",
                "interest rate",
            ],
        ),
        ("| Keyword | Why |\n|---|---|\n| prime rate | the base rate |", ["prime rate"]),
        (
            "<ul>\n<li>prime rate</li>\n<li>bank of canada</li>\n</ul>",
            ["prime rate", "bank of canada"],
        ),
        (
            "Okay, the user asks about rates.\n</think>\n\nprime rate, bank of canada",
            ["prime rate", "bank of canada"],
        ),
        ("- __init__ method\n- snake_case", ["__init__ method", "snake_case"]),
    ],
)
def test_cleanup_keeps_only_keywords(raw, expected):
    terms, _ = make_q2e()._extract_keywords(raw)
    assert terms == expected


def test_think_trace_is_removed():
    terms, meta = make_q2e()._extract_keywords(
        "<think>\nThe user wants keywords.\n</think>\n\nprime rate, Bank of Canada"
    )
    assert terms == ["prime rate", "Bank of Canada"]
    assert meta == {}


def test_unclosed_think_trace_yields_no_keywords():
    terms, meta = make_q2e()._extract_keywords(
        "<think>\nOkay, the user is asking about the prime rate. Related terms: lending,"
    )
    assert terms == []
    assert meta == {"unclosed_think": True}


def test_few_shot_continuation_is_truncated():
    terms, meta = make_q2e()._extract_keywords(
        "prime rate, bank of canada\nQuery: how tall is mount everest\nKeywords: everest, nepal"
    )
    assert terms == ["prime rate", "bank of canada"]
    assert meta == {"truncated_at_query": True}


@pytest.mark.parametrize(
    "raw",
    [
        "Here are the keywords for your query.",
        "<think>\nkeywords: prime rate, bank of canada\n</think>\n",
    ],
)
def test_output_without_keywords_is_flagged(raw):
    terms, meta = make_q2e()._extract_keywords(raw)
    assert terms == []
    assert meta == {"cleanup_empty": True}


def test_cleanup_falls_back_to_plain_keyword_list():
    terms, meta = make_q2e()._extract_keywords("note, musical notation, pitch")
    assert terms == ["note", "musical notation", "pitch"]
    assert meta == {"cleanup_fallback": True}


@pytest.mark.parametrize("raw", [None, "", "  \n"])
def test_empty_output_expands_with_query_only(raw):
    res = make_q2e(raw).reformulate(QueryItem("q1", QUERY))
    assert res.metadata["keywords"] == []
    assert "error" not in res.metadata


@pytest.mark.parametrize(
    "params, expected",
    [
        ({}, {"clean_output": True}),
        ({"clean_output": True, "mode": "zs"}, {"clean_output": True, "mode": "zs"}),
        ({"clean_output": False}, {}),
        ({"clean_output": False, "num_examples": 4}, {"num_examples": 4}),
    ],
)
def test_effective_params_records_clean_output_only_when_enabled(params, expected):
    assert make_q2e(**params).effective_params() == expected


def test_effective_params_drops_runtime_objects():
    meth = make_q2e(clean_output=False, searcher=object(), retrieval_k=10)
    assert meth.effective_params() == {"retrieval_k": 10}


def test_strip_think_trace():
    assert strip_think_trace("<think>a</think>b") == ("b", False)
    assert strip_think_trace("x<think>unfinished") == ("x", True)
    assert strip_think_trace("x</think>y") == ("y", False)
    assert strip_think_trace("plain") == ("plain", False)
