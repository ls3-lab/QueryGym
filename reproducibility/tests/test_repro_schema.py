"""Tests for reproducibility.lib (emit + validate)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from reproducibility.lib import (
    SCHEMA_VERSION,
    ValidationError,
    build_run_summary,
    compute_params_hash,
    compute_run_id,
    validate,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_run.json"


def _load_fixture() -> dict:
    with FIXTURE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_kwargs() -> dict:
    """Inputs that produce the canonical fixture, modulo volatile fields."""
    return dict(
        dataset_id="msmarco-v1-passage.trecdl2019",
        method_id="query2e",
        model="gpt-4.1-mini",
        method_params={"mode": "zs"},
        llm_config={"temperature": 1.0, "max_tokens": 128, "top_p": 1.0},
        retrieval={
            "retriever_id": "bm25",
            "paradigm": "lexical",
            "params": {"k1": 0.9, "b": 0.4},
            "implementation": "pyserini:LuceneSearcher",
        },
        dataset_config={
            "topics": "dl19-passage",
            "index": "msmarco-v1-passage",
            "num_queries": 43,
        },
        metrics={"map": 0.3709, "ndcg_cut_10": 0.5679, "recall_1000": 0.8384},
        timing={
            "reformulation_seconds": 65.24,
            "retrieval_seconds": 3.01,
            "evaluation_seconds": 10.53,
        },
        steps_completed=["reformulate", "retrieve", "evaluate"],
        total_time_seconds=89.37,
        # Pin volatile fields for determinism in tests.
        submitted_at="2026-04-29T10:14:22Z",
        environment={
            "python_version": "3.10.13",
            "platform": "Linux-5.15.0-x86_64",
            "git_commit": "5c46a51",
        },
        querygym_version="0.3.0",
    )


# ---------- Hash properties --------------------------------------------------


def test_params_hash_is_8_hex():
    h = compute_params_hash("query2e", "gpt-4.1-mini", {"mode": "zs"}, {"t": 1.0})
    assert len(h) == 8
    assert all(c in "0123456789abcdef" for c in h)


def test_params_hash_is_stable():
    a = compute_params_hash("query2e", "gpt-4.1-mini", {"mode": "zs"}, {"temperature": 1.0})
    b = compute_params_hash("query2e", "gpt-4.1-mini", {"mode": "zs"}, {"temperature": 1.0})
    assert a == b


def test_params_hash_changes_on_temperature():
    a = compute_params_hash("query2e", "gpt-4.1-mini", {"mode": "zs"}, {"temperature": 1.0})
    b = compute_params_hash("query2e", "gpt-4.1-mini", {"mode": "zs"}, {"temperature": 0.5})
    assert a != b


def test_params_hash_invariant_to_key_order():
    a = compute_params_hash("query2e", "gpt-4.1-mini", {"a": 1, "b": 2}, {"x": 1, "y": 2})
    b = compute_params_hash("query2e", "gpt-4.1-mini", {"b": 2, "a": 1}, {"y": 2, "x": 1})
    assert a == b


def test_run_id_excludes_volatile_fields():
    payload = _load_fixture()
    rid_a = compute_run_id(payload)
    payload2 = copy.deepcopy(payload)
    payload2["submitted_at"] = "2099-12-31T23:59:59Z"
    payload2["environment"] = {"python_version": "9.9", "platform": "any", "git_commit": None}
    rid_b = compute_run_id(payload2)
    assert rid_a == rid_b


def test_run_id_changes_on_metric_change():
    payload = _load_fixture()
    rid_a = compute_run_id(payload)
    payload2 = copy.deepcopy(payload)
    payload2["metrics"]["map"] = 0.9999
    rid_b = compute_run_id(payload2)
    assert rid_a != rid_b


# ---------- build_run_summary ------------------------------------------------


def test_build_run_summary_matches_fixture():
    """The fixture should be exactly what build_run_summary produces from canonical inputs."""
    built = build_run_summary(**_build_kwargs())
    fixture = _load_fixture()
    assert built == fixture


def test_build_run_summary_validates_clean():
    payload = build_run_summary(**_build_kwargs())
    validate(payload)


def test_build_run_summary_artifact_filenames_use_params_hash():
    payload = build_run_summary(**_build_kwargs())
    h = payload["params_hash"]
    assert payload["artifacts"]["run_file"] == f"{h}.run.txt"
    assert payload["artifacts"]["reformulated_queries"] == f"{h}.queries.tsv"


# ---------- Validator: schema-level rejections -------------------------------


def test_validator_rejects_missing_schema_version():
    payload = _load_fixture()
    del payload["schema_version"]
    with pytest.raises(ValidationError, match="schema_version"):
        validate(payload)


def test_validator_rejects_wrong_schema_version():
    payload = _load_fixture()
    payload["schema_version"] = 2
    with pytest.raises(ValidationError):
        validate(payload)


def test_validator_rejects_extra_top_level_field():
    payload = _load_fixture()
    payload["whoops"] = "extra"
    with pytest.raises(ValidationError):
        validate(payload)


def test_validator_rejects_malformed_artifact_filename():
    payload = _load_fixture()
    payload["artifacts"]["run_file"] = "not-a-hash.run.txt"
    with pytest.raises(ValidationError, match="artifacts"):
        validate(payload)


# ---------- Validator: registry-level rejections -----------------------------


def test_validator_rejects_unknown_dataset():
    payload = _load_fixture()
    payload["pipeline"]["dataset_id"] = "fake-dataset"
    # Recompute hashes so we hit the registry check, not the hash check.
    payload["run_id"] = compute_run_id(payload)
    with pytest.raises(ValidationError, match="dataset_id 'fake-dataset'"):
        validate(payload)


def test_validator_rejects_metric_outside_eval_metrics():
    payload = _load_fixture()
    payload["metrics"]["bleu"] = 0.5
    payload["run_id"] = compute_run_id(payload)
    with pytest.raises(ValidationError, match="not in eval_metrics"):
        validate(payload)


def test_validator_rejects_metric_above_one():
    """A corrupt score (e.g. 23.0 from a dropped decimal point) must be caught —
    every ranking metric is normalized to [0, 1]."""
    payload = _load_fixture()
    payload["metrics"]["ndcg_cut_10"] = 23.0
    payload["run_id"] = compute_run_id(payload)
    with pytest.raises(ValidationError, match=r"outside \[0, 1\]"):
        validate(payload)


def test_validator_rejects_metric_below_zero():
    payload = _load_fixture()
    payload["metrics"]["ndcg_cut_10"] = -0.1
    payload["run_id"] = compute_run_id(payload)
    with pytest.raises(ValidationError, match=r"outside \[0, 1\]"):
        validate(payload)


def test_validator_accepts_metric_at_bounds():
    """The bounds themselves are valid (a perfect or zero score)."""
    payload = _load_fixture()
    payload["metrics"]["ndcg_cut_10"] = 1.0
    payload["metrics"]["recall_1000"] = 0.0
    payload["run_id"] = compute_run_id(payload)
    validate(payload)  # must not raise


# ---------- Validator: hash-level rejections ---------------------------------


def test_validator_rejects_tampered_params_hash():
    payload = _load_fixture()
    payload["params_hash"] = "deadbeef"
    payload["run_id"] = compute_run_id(payload)
    with pytest.raises(ValidationError, match="params_hash mismatch"):
        validate(payload)


def test_validator_rejects_tampered_run_id():
    payload = _load_fixture()
    payload["run_id"] = "0" * 16
    with pytest.raises(ValidationError, match="run_id mismatch"):
        validate(payload)


def test_validator_rejects_silent_metric_edit():
    """Hand-editing a metric value without recomputing run_id must be caught."""
    payload = _load_fixture()
    payload["metrics"]["map"] = 0.9999  # leave run_id alone
    with pytest.raises(ValidationError, match="run_id mismatch"):
        validate(payload)


# ---------- Skip-registry-checks escape hatch (for tests with synthetic ids) ---


def test_validator_skip_registry_checks_allows_unknown_ids():
    """build_run_summary with an unknown dataset still validates if registry checks are off."""
    kwargs = _build_kwargs()
    kwargs["dataset_id"] = "synthetic-test-dataset"
    payload = build_run_summary(**kwargs)
    # Schema/hash checks still pass; registry check is skipped.
    validate(payload, skip_registry_checks=True)


# ---------- Retriever registry ----------------------------------------------


def test_retriever_registry_has_exactly_the_three_published_blocks():
    import yaml

    path = Path(__file__).resolve().parents[2] / "reproducibility" / "retriever_registry.yaml"
    reg = yaml.safe_load(path.read_text(encoding="utf-8"))["retrievers"]
    assert set(reg) == {"bm25", "splade-pp", "bge-base-en-v1.5"}
    assert reg["bm25"] == {"display_name": "BM25", "paradigm": "lexical"}
    assert reg["splade-pp"] == {"display_name": "SPLADE++", "paradigm": "learned_sparse"}
    assert reg["bge-base-en-v1.5"] == {
        "display_name": "BGE-base-en-v1.5",
        "paradigm": "dense",
    }


# ---------- Schema: config.retrieval shape ----------------------------------


def _minimal_payload_for_schema_only() -> dict:
    """A structurally-valid (new-shape) payload for raw JSON-Schema checks."""
    return {
        "schema_version": 1,
        "run_id": "0" * 16,
        "params_hash": "0" * 8,
        "submitted_at": "2026-05-19T00:00:00Z",
        "querygym_version": "0.3.0",
        "environment": {"python_version": "3.12.0", "platform": "x"},
        "pipeline": {
            "dataset_id": "d",
            "method_id": "m",
            "model": "x",
            "steps_completed": ["reformulate"],
            "total_time_seconds": 1.0,
        },
        "config": {
            "method_params": {},
            "llm_config": {"temperature": 1.0, "max_tokens": 1},
            "dataset_config": {"topics": "t", "index": "i", "num_queries": 1},
            "retrieval": {
                "retriever_id": "bm25",
                "paradigm": "lexical",
                "params": {"k1": 0.9, "b": 0.4},
            },
        },
        "metrics": {"ndcg_cut_10": 0.5},
        "timing": {},
        "artifacts": {
            "run_file": "00000000.run.txt",
            "reformulated_queries": "00000000.queries.tsv",
        },
    }


def _raw_schema_validate(payload: dict) -> None:
    import json as _json

    import jsonschema

    schema_path = (
        Path(__file__).resolve().parents[2] / "reproducibility" / "schema.json"
    )
    schema = _json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=payload, schema=schema)


def test_schema_accepts_lexical_retrieval():
    _raw_schema_validate(_minimal_payload_for_schema_only())


def test_schema_accepts_learned_sparse_retrieval():
    p = _minimal_payload_for_schema_only()
    p["config"]["retrieval"] = {
        "retriever_id": "splade-pp",
        "paradigm": "learned_sparse",
        "params": {"model": "naver/splade-cocondenser-ensembledistil"},
    }
    _raw_schema_validate(p)


def test_schema_accepts_dense_retrieval_with_implementation():
    p = _minimal_payload_for_schema_only()
    p["config"]["retrieval"] = {
        "retriever_id": "bge-base-en-v1.5",
        "paradigm": "dense",
        "params": {"encoder": "BAAI/bge-base-en-v1.5"},
        "implementation": "pyserini:FaissSearcher",
    }
    _raw_schema_validate(p)


def test_schema_rejects_legacy_searcher_block():
    import jsonschema

    p = _minimal_payload_for_schema_only()
    p["config"]["searcher"] = {"name": "x", "type": "y"}
    with pytest.raises(jsonschema.ValidationError):
        _raw_schema_validate(p)


def test_schema_rejects_dataset_config_bm25_weights():
    import jsonschema

    p = _minimal_payload_for_schema_only()
    p["config"]["dataset_config"]["bm25_weights"] = {"k1": 0.9, "b": 0.4}
    with pytest.raises(jsonschema.ValidationError):
        _raw_schema_validate(p)


def test_schema_rejects_lexical_with_model_params():
    import jsonschema

    p = _minimal_payload_for_schema_only()
    p["config"]["retrieval"]["params"] = {"model": "naver/splade"}
    with pytest.raises(jsonschema.ValidationError):
        _raw_schema_validate(p)


def test_schema_rejects_dense_missing_encoder():
    import jsonschema

    p = _minimal_payload_for_schema_only()
    p["config"]["retrieval"] = {
        "retriever_id": "bge-base-en-v1.5",
        "paradigm": "dense",
        "params": {"wrong": "x"},
    }
    with pytest.raises(jsonschema.ValidationError):
        _raw_schema_validate(p)


# ---------- build_run_summary: retrieval block -------------------------------


def test_build_run_summary_emits_retrieval_not_searcher():
    payload = build_run_summary(
        dataset_id="msmarco-v1-passage.trecdl2019",
        method_id="query2e",
        model="openai/gpt-4.1",
        method_params={"mode": "zs"},
        llm_config={"temperature": 1.0, "max_tokens": 128},
        retrieval={
            "retriever_id": "bm25",
            "paradigm": "lexical",
            "params": {"k1": 0.9, "b": 0.4},
            "implementation": "pyserini:LuceneSearcher",
        },
        dataset_config={"topics": "dl19-passage", "index": "msmarco-v1-passage", "num_queries": 43},
        metrics={"ndcg_cut_10": 0.5},
        timing={},
        steps_completed=["reformulate", "retrieve", "evaluate"],
        total_time_seconds=1.0,
    )
    assert "searcher" not in payload["config"]
    assert "bm25_weights" not in payload["config"]["dataset_config"]
    assert payload["config"]["retrieval"] == {
        "retriever_id": "bm25",
        "paradigm": "lexical",
        "params": {"k1": 0.9, "b": 0.4},
        "implementation": "pyserini:LuceneSearcher",
    }


def test_build_run_summary_retrieval_implementation_is_optional():
    payload = build_run_summary(
        dataset_id="msmarco-v1-passage.trecdl2019",
        method_id="query2e",
        model="openai/gpt-4.1",
        method_params={"mode": "zs"},
        llm_config={"temperature": 1.0, "max_tokens": 128},
        retrieval={"retriever_id": "bm25", "paradigm": "lexical", "params": {"k1": 0.9, "b": 0.4}},
        dataset_config={"topics": "dl19-passage", "index": "msmarco-v1-passage", "num_queries": 43},
        metrics={"ndcg_cut_10": 0.5},
        timing={},
        steps_completed=["reformulate"],
        total_time_seconds=1.0,
    )
    assert "implementation" not in payload["config"]["retrieval"]


# ---------- Validator: retriever registry -----------------------------------


def test_validator_rejects_unknown_retriever_id():
    payload = _load_fixture()
    payload["config"]["retrieval"]["retriever_id"] = "totally-fake"
    payload["run_id"] = compute_run_id(payload)
    with pytest.raises(ValidationError, match="retriever_id 'totally-fake'"):
        validate(payload)


def test_validator_rejects_paradigm_registry_mismatch():
    payload = _load_fixture()
    # bm25 is 'lexical' in the registry; claim 'dense' (still schema-valid).
    payload["config"]["retrieval"]["paradigm"] = "dense"
    payload["config"]["retrieval"]["params"] = {"encoder": "x"}
    payload["run_id"] = compute_run_id(payload)
    with pytest.raises(ValidationError, match="paradigm mismatch"):
        validate(payload)


def test_validator_accepts_known_retriever():
    # The migrated fixture uses bm25/lexical — must validate cleanly.
    validate(_load_fixture())


def test_params_hash_unchanged_by_multi_retriever_migration():
    """Approach 1 must NOT perturb the paper-pinned params_hash.

    The hash is a function only of (method_id, model, method_params, llm_config).
    It was 'ddb15ccf' before the multi-retriever migration and must stay so.
    """
    assert (
        compute_params_hash(
            "query2e", "gpt-4.1-mini", {"mode": "zs"},
            {"temperature": 1.0, "max_tokens": 128, "top_p": 1.0},
        )
        == "ddb15ccf"
    )
    assert _load_fixture()["params_hash"] == "ddb15ccf"


# ---------- submit_run canonical path ---------------------------------------


def test_canonical_dir_includes_retriever_segment(tmp_path):
    import importlib

    submit_run = importlib.import_module("reproducibility.scripts.submit_run")
    payload = _load_fixture()
    d = submit_run._canonical_dir(tmp_path, payload)
    assert d == (
        tmp_path
        / "msmarco-v1-passage.trecdl2019"
        / "query2e"
        / "gpt-4.1-mini"
        / "bm25"
    )


# ---------- aggregator: retriever columns -----------------------------------


def test_aggregator_emits_retriever_columns():
    import importlib

    agg = importlib.import_module("reproducibility.scripts.aggregate_runs")
    assert agg.CSV_COLUMNS[:7] == [
        "schema_version",
        "run_id",
        "dataset_id",
        "method_id",
        "model",
        "retriever_id",
        "retriever",
    ]

    payload = _load_fixture()
    fake_path = agg._REPO_ROOT / "reproducibility" / "data" / "runs" / "x.json"
    rows = agg._payload_to_rows(payload, fake_path)
    row0 = dict(zip(agg.CSV_COLUMNS, rows[0]))
    assert row0["retriever_id"] == "bm25"
    assert row0["retriever"] == "BM25"


# ---------- pipeline forward-compat -----------------------------------------


def test_run_summary_emits_lexical_retrieval_block():
    import importlib

    rs = importlib.import_module("examples.querygym_pyserini.run_summary")
    results = {
        "reformulation": {
            "dataset": {
                "topics": "dl19-passage",
                "index": "msmarco-v1-passage",
                "num_queries": 43,
                "bm25_weights": {"k1": 0.9, "b": 0.4},
            },
            "reformulation": {
                "method_params": {"mode": "zs"},
                "llm_config": {"temperature": 1.0, "max_tokens": 128},
                "searcher": {"name": "UserPyseriniWrapper", "type": "user_pyserini",
                             "searcher_class": "LuceneSearcher"},
            },
            "timing": {"total_time_seconds": 1.0},
        },
        "retrieval": {"timing": {"total_time_seconds": 1.0}},
        "evaluation": {"timing": {"eval_time_seconds": 1.0},
                        "results": {"ndcg_cut_10": 0.5}},
    }
    payload = rs._build_v1_summary(
        results=results,
        dataset_name="msmarco-v1-passage.trecdl2019",
        method="query2e",
        model="openai/gpt-4.1",
        method_params={"mode": "zs"},
        llm_config={"temperature": 1.0, "max_tokens": 128},
        steps=["reformulate", "retrieve", "evaluate"],
        pipeline_time=3.0,
        registry_path="dataset_registry.yaml",
        queries_file=None,
        index_name="msmarco-v1-passage",
    )
    r = payload["config"]["retrieval"]
    assert r["retriever_id"] == "bm25"
    assert r["paradigm"] == "lexical"
    assert r["params"] == {"k1": 0.9, "b": 0.4}
    assert r["implementation"] == "pyserini:LuceneSearcher"
    assert "searcher" not in payload["config"]
    assert "bm25_weights" not in payload["config"]["dataset_config"]


# ---------- Optional artifacts ----------------------------------------------


def test_schema_accepts_empty_artifacts():
    p = _minimal_payload_for_schema_only()
    p["artifacts"] = {}
    _raw_schema_validate(p)


def test_schema_accepts_artifacts_with_only_run_file():
    p = _minimal_payload_for_schema_only()
    p["artifacts"] = {"run_file": "00000000.run.txt"}
    _raw_schema_validate(p)


def test_build_run_summary_artifacts_present_none_defaults_to_both():
    p = build_run_summary(**_build_kwargs())
    assert set(p["artifacts"]) == {"run_file", "reformulated_queries"}


def test_build_run_summary_artifacts_present_empty_emits_no_artifacts():
    kw = _build_kwargs()
    kw["artifacts_present"] = set()
    p = build_run_summary(**kw)
    assert p["artifacts"] == {}
    # validate (schema + registry + hash) passes
    validate(p)


def test_build_run_summary_artifacts_present_only_run_file():
    kw = _build_kwargs()
    kw["artifacts_present"] = {"run_file"}
    p = build_run_summary(**kw)
    assert set(p["artifacts"]) == {"run_file"}
    assert p["artifacts"]["run_file"] == f"{p['params_hash']}.run.txt"
    validate(p)


def test_build_run_summary_rejects_unknown_artifact_key():
    kw = _build_kwargs()
    kw["artifacts_present"] = {"run_file", "bogus"}
    with pytest.raises(ValueError, match="unknown artifact keys"):
        build_run_summary(**kw)


# ---------- DL-HARD registry entry ------------------------------------------


def test_dataset_registry_has_dlhard_entry():
    import yaml

    p = Path(__file__).resolve().parents[2] / "dataset_registry.yaml"
    reg = yaml.safe_load(p.read_text(encoding="utf-8"))["datasets"]
    assert "msmarco-v1-passage.dlhard" in reg
    entry = reg["msmarco-v1-passage.dlhard"]
    assert entry["index"]["name"] == "msmarco-v1-passage"
    assert set(entry["output"]["eval_metrics"]) >= {"ndcg_cut.10", "recall.1000"}


def test_validator_accepts_dlhard_run():
    kw = _build_kwargs()
    kw["dataset_id"] = "msmarco-v1-passage.dlhard"
    # DL-HARD whitelists {ndcg_cut.10, recall.1000}.
    kw["metrics"] = {"ndcg_cut_10": 0.4038, "recall_1000": 0.8415}
    p = build_run_summary(**kw)
    validate(p)


# ---------- Portable paths (no machine-specific absolute paths) --------------


def test_portable_path_normalizes_absolute_paths():
    from reproducibility.lib.emit import _portable_path

    assert (
        _portable_path("/mnt/data/son/data/msmarco/collection.tsv")
        == "msmarco/collection.tsv"
    )
    assert (
        _portable_path("/mnt/data/son/Thesis/t5/data/dlhard/neutral_queries.tsv")
        == "dlhard/neutral_queries.tsv"
    )
    assert _portable_path(r"C:\Users\bob\data\msmarco\collection.tsv") == "msmarco/collection.tsv"


def test_portable_path_leaves_identifiers_and_is_idempotent():
    from reproducibility.lib.emit import _portable_path

    # registry keys / short identifiers are not paths — untouched
    assert _portable_path("dl19-passage") == "dl19-passage"
    assert _portable_path("msmarco-v1-passage") == "msmarco-v1-passage"
    assert _portable_path("beir-v1.0.0-arguana-test") == "beir-v1.0.0-arguana-test"
    # non-strings pass through
    assert _portable_path(4) == 4
    # already-relative / already-normalized — idempotent
    assert _portable_path("msmarco/collection.tsv") == "msmarco/collection.tsv"


def test_build_run_summary_strips_absolute_method_params():
    kw = _build_kwargs()
    kw["method_id"] = "query2doc"
    kw["method_params"] = {
        "mode": "fs",
        "collection_path": "/mnt/data/son/data/msmarco/collection.tsv",
        "train_queries_path": "/mnt/data/son/data/msmarco/queries.train.tsv",
    }
    p = build_run_summary(**kw)
    mp = p["config"]["method_params"]
    assert mp["collection_path"] == "msmarco/collection.tsv"
    assert mp["train_queries_path"] == "msmarco/queries.train.tsv"
    # nothing host-specific survives anywhere in the payload
    assert "/mnt/" not in json.dumps(p)


def test_build_run_summary_strips_absolute_topics():
    kw = _build_kwargs()
    kw["dataset_id"] = "msmarco-v1-passage.dlhard"
    kw["metrics"] = {"ndcg_cut_10": 0.4038, "recall_1000": 0.8415}
    kw["dataset_config"] = {
        "topics": "/mnt/data/son/Thesis/t5/data/dlhard/neutral_queries.tsv",
        "index": "msmarco-v1-passage",
        "num_queries": 343,
    }
    p = build_run_summary(**kw)
    assert p["config"]["dataset_config"]["topics"] == "dlhard/neutral_queries.tsv"
    assert p["config"]["dataset_config"]["index"] == "msmarco-v1-passage"


def test_upstream_emit_reproduces_committed_legacy_hash():
    """Normalizing an absolute few-shot path at emit time must reproduce the
    params_hash already committed for that logical config (PR #32/#33), so a
    fresh run of the same experiment keeps its canonical identity."""
    h = compute_params_hash(
        "query2doc",
        "openai/gpt-4.1",
        {
            "mode": "fs",
            "num_examples": 4,
            "dataset_type": "msmarco",
            "collection_path": "msmarco/collection.tsv",
            "train_queries_path": "msmarco/queries.train.tsv",
            "train_qrels_path": "msmarco/qrels.train.tsv",
            "train_split": "train",
        },
        {"temperature": 1.0, "max_tokens": 128},
    )
    assert h == "97128a62"

    # and building from the ABSOLUTE paths must arrive at the same hash
    kw = _build_kwargs()
    kw.update(
        dataset_id="beir-v1.0.0-arguana",
        method_id="query2doc",
        model="openai/gpt-4.1",
        method_params={
            "mode": "fs",
            "num_examples": 4,
            "dataset_type": "msmarco",
            "collection_path": "/mnt/data/son/data/msmarco/collection.tsv",
            "train_queries_path": "/mnt/data/son/data/msmarco/queries.train.tsv",
            "train_qrels_path": "/mnt/data/son/data/msmarco/qrels.train.tsv",
            "train_split": "train",
        },
        llm_config={"temperature": 1.0, "max_tokens": 128},
        metrics={"ndcg_cut_10": 0.4012, "recall_100": 0.941},
    )
    assert build_run_summary(**kw)["params_hash"] == "97128a62"


def test_validator_rejects_absolute_path_in_method_params():
    payload = _load_fixture()
    payload["config"]["method_params"]["collection_path"] = "/mnt/data/son/data/msmarco/collection.tsv"
    with pytest.raises(ValidationError, match="absolute path"):
        validate(payload, skip_registry_checks=True)


def test_validator_rejects_absolute_topics():
    payload = _load_fixture()
    payload["config"]["dataset_config"]["topics"] = "/abs/path/to/topics.tsv"
    with pytest.raises(ValidationError, match="absolute path"):
        validate(payload, skip_registry_checks=True)
