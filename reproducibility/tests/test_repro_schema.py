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
        searcher={"name": "UserPyseriniWrapper", "type": "user_pyserini"},
        dataset_config={
            "topics": "dl19-passage",
            "index": "msmarco-v1-passage",
            "num_queries": 43,
            "bm25_weights": {"k1": 0.9, "b": 0.4},
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
