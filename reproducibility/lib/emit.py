"""Build canonical run-summary JSON payloads conformant to schema v1."""

from __future__ import annotations

import hashlib
import json
import platform as _platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Mapping

SCHEMA_VERSION = 1


def _stable_json(payload: Any) -> str:
    """Serialize with sorted keys and no whitespace — deterministic across runs."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_params_hash(
    method_id: str,
    model: str,
    method_params: Mapping[str, Any],
    llm_config: Mapping[str, Any],
) -> str:
    """8-char hex hash over the tuning surface.

    Same config -> same hash (re-run replaces the previous file).
    Different temperature -> different hash (no collision).
    """
    payload = {
        "method_id": method_id,
        "model": model,
        "method_params": dict(method_params),
        "llm_config": dict(llm_config),
    }
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return digest[:8]


# Fields excluded from run_id hash because they would make identical executions
# produce different ids. Keep this list in sync with schema.json's volatile fields.
_RUN_ID_EXCLUDED_FIELDS = ("run_id", "submitted_at", "environment")


def compute_run_id(payload: Mapping[str, Any]) -> str:
    """16-char hex hash over the payload minus volatile fields.

    Two distinct executions of the same logical experiment that produce identical
    metrics -> same run_id. Two executions whose results differ -> different run_ids.
    """
    stripped = {k: v for k, v in payload.items() if k not in _RUN_ID_EXCLUDED_FIELDS}
    digest = hashlib.sha256(_stable_json(stripped).encode("utf-8")).hexdigest()
    return digest[:16]


def _detect_environment() -> dict:
    """Best-effort capture of the runner's environment. Never raises."""
    env = {
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "platform": _platform.platform(),
        "git_commit": None,
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if commit.returncode == 0 and commit.stdout.strip():
            env["git_commit"] = commit.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return env


_ARTIFACT_KEYS = ("run_file", "reformulated_queries")
_ARTIFACT_SUFFIX = {"run_file": "run.txt", "reformulated_queries": "queries.tsv"}


def _build_artifacts(params_hash: str, present: set[str] | None) -> dict:
    """Build the artifacts dict.

    `present` selects which artifact filenames to include. None → both (the
    default for the standard pipeline). Empty set → no artifacts (when neither
    the ranking nor the reformulated-queries file is attached to the run).
    Unknown keys raise.
    """
    if present is None:
        keys = _ARTIFACT_KEYS
    else:
        unknown = set(present) - set(_ARTIFACT_KEYS)
        if unknown:
            raise ValueError(f"unknown artifact keys: {sorted(unknown)}")
        keys = tuple(k for k in _ARTIFACT_KEYS if k in present)
    return {k: f"{params_hash}.{_ARTIFACT_SUFFIX[k]}" for k in keys}


def _querygym_version() -> str:
    """Read querygym.__version__ if available, else 'unknown'."""
    try:
        import querygym  # type: ignore
        return getattr(querygym, "__version__", "unknown")
    except ImportError:
        return "unknown"


def build_run_summary(
    *,
    dataset_id: str,
    method_id: str,
    model: str,
    method_params: Mapping[str, Any],
    llm_config: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    dataset_config: Mapping[str, Any],
    metrics: Mapping[str, float],
    timing: Mapping[str, float],
    steps_completed: list,
    total_time_seconds: float,
    submitted_at: str | None = None,
    environment: Mapping[str, Any] | None = None,
    querygym_version: str | None = None,
    artifacts_present: set[str] | None = None,
) -> dict:
    """Assemble a schema-v1 run summary dict.

    Computes params_hash and run_id internally so callers can't get them wrong.
    The optional submitted_at / environment / querygym_version overrides exist
    for tests that need deterministic output; in normal use, leave them None.

    `retrieval` carries the retriever dimension: retriever_id (a path-safe slug
    in retriever_registry.yaml), paradigm, params (shape depends on paradigm),
    and an optional `implementation` provenance string. params_hash is NOT a
    function of retrieval (it stays the reformulation tuning surface); run_id
    differs per retriever because the retrieval block is in the payload.

    The returned dict validates against reproducibility/schema.json by
    construction, but callers should still pass it through validate(...) before
    writing — that adds runtime checks against the dataset/method/retriever
    registries.
    """
    params_hash = compute_params_hash(method_id, model, method_params, llm_config)

    retrieval_block: dict = {
        "retriever_id": retrieval["retriever_id"],
        "paradigm": retrieval["paradigm"],
        "params": dict(retrieval["params"]),
    }
    if retrieval.get("implementation") is not None:
        retrieval_block["implementation"] = retrieval["implementation"]

    payload: dict = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "",  # filled in below
        "params_hash": params_hash,
        "submitted_at": submitted_at
            or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "querygym_version": querygym_version or _querygym_version(),
        "environment": dict(environment) if environment is not None else _detect_environment(),
        "pipeline": {
            "dataset_id": dataset_id,
            "method_id": method_id,
            "model": model,
            "steps_completed": list(steps_completed),
            "total_time_seconds": float(total_time_seconds),
        },
        "config": {
            "method_params": dict(method_params),
            "llm_config": dict(llm_config),
            "dataset_config": {
                "topics": dataset_config["topics"],
                "index": dataset_config["index"],
                "num_queries": int(dataset_config["num_queries"]),
            },
            "retrieval": retrieval_block,
        },
        "metrics": {k: float(v) for k, v in metrics.items()},
        "timing": {k: float(v) for k, v in timing.items()},
        "artifacts": _build_artifacts(params_hash, artifacts_present),
    }

    payload["run_id"] = compute_run_id(payload)
    return payload
