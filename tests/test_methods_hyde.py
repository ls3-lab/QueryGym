from pathlib import Path

import pytest

import querygym as qg
from querygym.core.base import MethodConfig, QueryItem
from querygym.core.prompts import PromptBank
from querygym.core.registry import METHODS
from querygym.methods.hyde import DATASET_TASKS, TASK_PROMPTS, HyDE, task_for_dataset

PROMPT_BANK = Path(__file__).parents[1] / "querygym" / "prompt_bank.yaml"


class RecordingLLM:
    """Returns the queued outputs in order and records every call."""

    def __init__(self, outputs=None):
        self.outputs = list(outputs or [])
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.outputs:
            return self.outputs.pop(0)
        return f"doc {len(self.calls)}"


def make_hyde(params=None, llm_cfg=None, llm=None):
    cfg = MethodConfig(name="hyde", params=params or {}, llm={"model": "dummy", **(llm_cfg or {})})
    llm = llm or RecordingLLM()
    return HyDE(cfg, llm, PromptBank(PROMPT_BANK)), llm


def test_generates_eight_docs_and_concatenates_query_once():
    meth, llm = make_hyde()
    res = meth.reformulate(QueryItem("q1", "what causes diabetes"))

    assert len(llm.calls) == 8
    expected_docs = [f"doc {i}" for i in range(1, 9)]
    assert res.reformulated == "what causes diabetes " + " ".join(expected_docs)
    assert res.qid == "q1"
    assert res.original == "what causes diabetes"
    assert res.metadata == {
        "task": "web search",
        "n_docs": 8,
        "hypothetical_docs": expected_docs,
    }


def test_n_docs_param():
    meth, llm = make_hyde(params={"n_docs": 3})
    res = meth.reformulate(QueryItem("q1", "query"))
    assert len(llm.calls) == 3
    assert res.reformulated == "query doc 1 doc 2 doc 3"


def test_repeat_query_weight():
    meth, _ = make_hyde(params={"n_docs": 1, "repeat_query_weight": 2})
    res = meth.reformulate(QueryItem("q1", "query"))
    assert res.reformulated == "query query doc 1"


def test_default_prompt_is_web_search():
    meth, llm = make_hyde(params={"n_docs": 1})
    meth.reformulate(QueryItem("q1", "how long is a flea life cycle"))
    messages = llm.calls[0][0]
    assert messages == [
        {
            "role": "user",
            "content": "Please write a passage to answer the question.\n"
            "Question: how long is a flea life cycle\nPassage:",
        }
    ]


@pytest.mark.parametrize(
    "task, expected_start",
    [
        ("web search", "Please write a passage to answer the question."),
        ("scifact", "Please write a scientific paper passage to support/refute the claim."),
        ("arguana", "Please write a counter argument for the passage."),
        ("trec-covid", "Please write a scientific paper passage to answer the question."),
        ("fiqa", "Please write a financial article passage to answer the question."),
        ("dbpedia-entity", "Please write a passage to answer the question."),
        ("trec-news", "Please write a news passage about the topic."),
    ],
)
def test_task_selects_prompt(task, expected_start):
    meth, llm = make_hyde(params={"task": task, "n_docs": 1})
    res = meth.reformulate(QueryItem("q1", "QUERY TEXT"))
    content = llm.calls[0][0][0]["content"]
    assert content.startswith(expected_start)
    assert "QUERY TEXT" in content
    assert res.metadata["task"] == task


def test_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown HyDE task: 'scifct'"):
        make_hyde(params={"task": "scifct"})


def test_sampling_params_passed_to_llm():
    meth, llm = make_hyde(params={"n_docs": 1})
    meth.reformulate(QueryItem("q1", "query"))
    # top_p and stop are only sent when configured
    assert llm.calls[0][1] == {"temperature": 0.7, "max_tokens": 512}


def test_null_sampling_params_are_not_sent():
    meth, llm = make_hyde(params={"n_docs": 1}, llm_cfg={"top_p": None, "stop": None})
    meth.reformulate(QueryItem("q1", "query"))
    assert llm.calls[0][1] == {"temperature": 0.7, "max_tokens": 512}


def test_llm_config_overrides_sampling_params():
    meth, llm = make_hyde(
        params={"n_docs": 1},
        llm_cfg={"temperature": 0.2, "max_tokens": 64, "top_p": 0.9, "stop": ["END"]},
    )
    meth.reformulate(QueryItem("q1", "query"))
    assert llm.calls[0][1] == {"temperature": 0.2, "max_tokens": 64, "top_p": 0.9, "stop": ["END"]}


def test_empty_generations_are_dropped():
    llm = RecordingLLM(["first", "   ", "", "second"])
    meth, _ = make_hyde(params={"n_docs": 4}, llm=llm)
    res = meth.reformulate(QueryItem("q1", "query"))
    assert res.reformulated == "query first second"
    assert res.metadata["hypothetical_docs"] == ["first", "second"]


def test_all_empty_generations_fall_back_to_query():
    llm = RecordingLLM(["", " ", "\n"])
    meth, _ = make_hyde(params={"n_docs": 3}, llm=llm)
    res = meth.reformulate(QueryItem("q1", "query"))
    assert res.reformulated == "query"
    assert res.metadata["hypothetical_docs"] == []


def test_newlines_flattened():
    llm = RecordingLLM(["line one\nline two"])
    meth, _ = make_hyde(params={"n_docs": 1}, llm=llm)
    res = meth.reformulate(QueryItem("q1", "query"))
    assert res.reformulated == "query line one line two"


@pytest.mark.parametrize(
    "dataset_id, task",
    [
        ("beir-v1.0.0-scifact", "scifact"),
        ("beir-v1.0.0-arguana", "arguana"),
        ("beir-v1.0.0-trec-covid", "trec-covid"),
        ("beir-v1.0.0-fiqa", "fiqa"),
        ("beir-v1.0.0-dbpedia-entity", "dbpedia-entity"),
        ("beir-v1.0.0-trec-news", "trec-news"),
        ("msmarco-v1-passage.trecdl2019", "web search"),
        ("msmarco-v1-passage.trecdl2020", "web search"),
        ("msmarco-v1-passage.dlhard", "web search"),
        ("bright-biology", "web search"),
        ("bright-stackoverflow", "web search"),
        ("my-custom-dataset", "web search"),
        (None, "web search"),
        ("", "web search"),
    ],
)
def test_task_for_dataset(dataset_id, task):
    assert task_for_dataset(dataset_id) == task


def test_mapped_tasks_have_prompts():
    pb = PromptBank(PROMPT_BANK)
    assert set(DATASET_TASKS.values()) <= set(TASK_PROMPTS)
    for prompt_id in TASK_PROMPTS.values():
        assert pb.get(prompt_id).method_family == "hyde"


def test_registered_and_created_via_factory():
    assert METHODS["hyde"] is HyDE
    reformulator = qg.create_reformulator("hyde", model="dummy", llm_config={"api_key": "x"})
    assert isinstance(reformulator, HyDE)


def test_cli_bundled_defaults():
    from querygym import cli

    cfg = cli.resolve_config("hyde")
    assert cfg["llm"]["temperature"] == 0.7
    assert cfg["llm"]["max_tokens"] == 512
    assert cfg["llm"]["top_p"] == 1.0
    assert cfg["llm"]["stop"] == ["\n\n\n"]
    assert cfg["params"]["task"] == "web search"
    assert cfg["params"]["n_docs"] == 8


def test_cli_run_writes_concat_and_plain(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from querygym import cli
    from querygym.core import runner

    llm = RecordingLLM()
    monkeypatch.setattr(runner, "build_llm", lambda cfg: llm)
    queries = tmp_path / "queries.tsv"
    queries.write_text("q1\tprime rate\n")

    result = CliRunner().invoke(
        cli.app,
        [
            "run",
            "--method",
            "hyde",
            "--queries-tsv",
            str(queries),
            "--output-tsv",
            str(tmp_path / "out.tsv"),
        ],
    )

    assert result.exit_code == 0, result.output
    docs = [f"doc {i}" for i in range(1, 9)]
    concat = (tmp_path / "out_concat.tsv").read_text().splitlines()
    assert concat == ["q1\tprime rate " + " ".join(docs)]
    plain = (tmp_path / "out_plain.tsv").read_text().splitlines()
    assert plain[0] == "\t".join(["qid"] + [f"passage_{i}" for i in range(1, 9)])
    assert plain[1] == "\t".join(["q1"] + docs)


def test_cli_plain_rows_padded_when_generations_are_empty(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from querygym import cli
    from querygym.core import runner

    # q1: first generation empty; q2: every generation empty
    llm = RecordingLLM([""] + [f"doc {i}" for i in range(1, 8)] + [""] * 8)
    monkeypatch.setattr(runner, "build_llm", lambda cfg: llm)
    queries = tmp_path / "queries.tsv"
    queries.write_text("q1\tprime rate\nq2\tinflation\n")

    result = CliRunner().invoke(
        cli.app,
        [
            "run",
            "--method",
            "hyde",
            "--queries-tsv",
            str(queries),
            "--output-tsv",
            str(tmp_path / "out.tsv"),
        ],
    )

    assert result.exit_code == 0, result.output
    plain = [line.split("\t") for line in (tmp_path / "out_plain.tsv").read_text().splitlines()]
    assert plain[0] == ["qid"] + [f"passage_{i}" for i in range(1, 9)]
    assert plain[1] == ["q1"] + [f"doc {i}" for i in range(1, 8)] + [""]
    assert plain[2] == ["q2"] + [""] * 8
