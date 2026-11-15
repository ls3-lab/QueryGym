from querygym.core.base import QueryItem
from querygym.data.dataloader import DataLoader
from pathlib import Path
import tempfile

import pytest


def test_local_tsv_loading():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tsv") as f:
        f.write("1\tapple\n2\tbanana\n")
        path = f.name
    queries = DataLoader.load_queries(path, format="tsv")
    assert len(queries) == 2 and queries[0].qid == "1"


def write_jsonl(*lines):
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        f.write("\n".join(lines) + "\n")
        return f.name


@pytest.mark.parametrize("record", ['{"qid": null, "query": "banana"}', '{"qid": "2", "query": null}'])
def test_jsonl_queries_skip_null_values(record):
    path = write_jsonl('{"qid": "1", "query": "apple"}', record)
    with pytest.warns(UserWarning, match="Missing or null keys"):
        queries = DataLoader.load_queries(path, format="jsonl")
    assert [(q.qid, q.text) for q in queries] == [("1", "apple")]


@pytest.mark.parametrize("record", ['{"qid": null, "contexts": ["b"]}', '{"qid": "2", "contexts": null}'])
def test_contexts_skip_null_values(record):
    path = write_jsonl('{"qid": "1", "contexts": ["a"]}', record)
    with pytest.warns(UserWarning, match="Missing or null keys"):
        contexts = DataLoader.load_contexts(path)
    assert contexts == {"1": ["a"]}


def test_contexts_drop_null_entries():
    path = write_jsonl('{"qid": "1", "contexts": ["a", null, "b"]}')
    assert DataLoader.load_contexts(path) == {"1": ["a", "b"]}


@pytest.mark.parametrize("record", ['{"query": null, "passage": "p"}', '{"query": "q", "passage": null}'])
def test_examples_skip_null_values(record):
    path = write_jsonl('{"query": "apple", "passage": "fruit"}', record)
    with pytest.warns(UserWarning, match="Missing or null keys"):
        examples = DataLoader.load_examples(path)
    assert examples == [{"query": "apple", "passage": "fruit"}]


@pytest.mark.parametrize("text", [None, 42])
def test_query_item_rejects_non_str_text(text):
    with pytest.raises(TypeError, match="QueryItem text must be a str"):
        QueryItem("q1", text)
