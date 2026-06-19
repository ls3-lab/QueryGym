"""Validate run-summary payloads against schema.json + runtime registries."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

from .emit import compute_params_hash, compute_run_id, _is_abs_path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.json"
_DEFAULT_DATASET_REGISTRY_PATH = _REPO_ROOT / "dataset_registry.yaml"
_RETRIEVER_REGISTRY_PATH = _REPO_ROOT / "reproducibility" / "retriever_registry.yaml"


class ValidationError(ValueError):
    """Raised when a payload fails schema or registry validation."""


@lru_cache(maxsize=1)
def _load_schema() -> dict:
    with _SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_dataset_registry(path: Path | str | None) -> dict:
    """Load dataset_registry.yaml. Returns the raw 'datasets' mapping."""
    import yaml  # pyyaml is a main dep

    p = Path(path) if path else _DEFAULT_DATASET_REGISTRY_PATH
    with p.open("r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)
    return registry.get("datasets", {})


def _load_method_registry() -> Iterable[str]:
    """Return registered method ids from querygym.core.registry.METHODS."""
    try:
        import querygym  # noqa: F401  # ensure methods register on import
        from querygym.core.registry import METHODS

        return list(METHODS.keys())
    except ImportError:
        return []


@lru_cache(maxsize=1)
def _load_retriever_registry() -> dict:
    """Load retriever_registry.yaml. Returns the raw 'retrievers' mapping."""
    import yaml  # pyyaml is a main dep

    with _RETRIEVER_REGISTRY_PATH.open("r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)
    return registry.get("retrievers", {})


def _normalize_metric_key(name: str) -> str:
    """trec_eval reports use underscores; dataset_registry uses dot notation."""
    return name.replace(".", "_")


def _jsonschema_validate(payload: Mapping[str, Any]) -> None:
    """Run the static JSON Schema validator. Raises ValidationError on drift."""
    try:
        import jsonschema
    except ImportError as e:
        raise ValidationError(
            "jsonschema is required for validation. "
            "Install with: pip install querygym[repro]"
        ) from e

    schema = _load_schema()
    try:
        jsonschema.validate(instance=payload, schema=schema)
    except jsonschema.ValidationError as e:
        # Translate the field path into something readable.
        path = "/".join(str(p) for p in e.absolute_path) or "<root>"
        raise ValidationError(f"schema violation at '{path}': {e.message}") from e


def validate(
    payload: Mapping[str, Any],
    *,
    dataset_registry: Mapping[str, Any] | None = None,
    method_registry: Iterable[str] | None = None,
    dataset_registry_path: Path | str | None = None,
    skip_registry_checks: bool = False,
) -> None:
    """Validate a run-summary payload.

    Three layers:
    1. JSON Schema (reproducibility/schema.json) — types, enums, required fields.
    2. Registry checks — dataset_id, method_id, metric whitelist.
    3. Hash checks — recompute params_hash and run_id, compare to stored values.

    Pass skip_registry_checks=True only in tests that intentionally use unknown ids.
    """
    _jsonschema_validate(payload)
    _validate_no_absolute_paths(payload)
    _validate_metric_ranges(payload)

    if not skip_registry_checks:
        if dataset_registry is None:
            dataset_registry = _load_dataset_registry(dataset_registry_path)
        if method_registry is None:
            method_registry = _load_method_registry()

        _validate_registries(payload, dataset_registry, method_registry)

    _validate_hashes(payload)


def _validate_registries(
    payload: Mapping[str, Any],
    dataset_registry: Mapping[str, Any],
    method_registry: Iterable[str],
) -> None:
    pipeline = payload["pipeline"]
    dataset_id = pipeline["dataset_id"]
    method_id = pipeline["method_id"]

    if dataset_id not in dataset_registry:
        # Provide closest-match hint to catch typos.
        candidates = sorted(dataset_registry.keys())
        hint = _closest(dataset_id, candidates)
        suffix = f" (did you mean '{hint}'?)" if hint else ""
        raise ValidationError(
            f"dataset_id '{dataset_id}' not in dataset_registry.yaml{suffix}"
        )

    method_set = set(method_registry)
    if method_set and method_id not in method_set:
        hint = _closest(method_id, sorted(method_set))
        suffix = f" (did you mean '{hint}'?)" if hint else ""
        raise ValidationError(
            f"method_id '{method_id}' not in registered methods{suffix}"
        )

    retrieval = payload["config"]["retrieval"]
    retriever_id = retrieval["retriever_id"]
    retriever_registry = _load_retriever_registry()
    if retriever_id not in retriever_registry:
        candidates = sorted(retriever_registry.keys())
        hint = _closest(retriever_id, candidates)
        suffix = f" (did you mean '{hint}'?)" if hint else ""
        raise ValidationError(
            f"retriever_id '{retriever_id}' not in retriever_registry.yaml{suffix}"
        )
    expected_paradigm = retriever_registry[retriever_id].get("paradigm")
    if retrieval["paradigm"] != expected_paradigm:
        raise ValidationError(
            f"paradigm mismatch for retriever_id '{retriever_id}': payload says "
            f"'{retrieval['paradigm']}', registry says '{expected_paradigm}'"
        )

    # Metric whitelist comes from dataset_registry; normalize dot/underscore.
    allowed_raw = (
        dataset_registry[dataset_id].get("output", {}).get("eval_metrics") or []
    )
    allowed = {_normalize_metric_key(m) for m in allowed_raw}
    if not allowed:
        # No whitelist configured for this dataset; skip the check.
        return

    metrics = payload["metrics"]
    unknown = [m for m in metrics.keys() if m not in allowed]
    if unknown:
        raise ValidationError(
            f"metric(s) {sorted(unknown)} not in eval_metrics for dataset "
            f"'{dataset_id}' (allowed: {sorted(allowed)})"
        )


def _validate_metric_ranges(payload: Mapping[str, Any]) -> None:
    """Reject metric values outside [0, 1].

    Every metric on the leaderboard (nDCG, recall, MAP, …) is a normalized
    ranking score bounded in [0, 1]. The JSON Schema only types these as
    `number`, so a corrupt value (e.g. a missing decimal point producing 23.0)
    would otherwise pass schema validation and silently inflate aggregates. This
    guard catches it at validate time so the aggregator and CI block any run that
    carries an impossible score."""
    offenders = [
        f"{name}={value}"
        for name, value in payload["metrics"].items()
        if not (0.0 <= float(value) <= 1.0)
    ]
    if offenders:
        raise ValidationError(
            f"metric value(s) outside [0, 1]: {sorted(offenders)}. "
            "All ranking metrics are normalized scores; a value out of range "
            "indicates a corrupt or mis-scaled result."
        )


def _validate_no_absolute_paths(payload: Mapping[str, Any]) -> None:
    """Reject machine-specific absolute paths in the config — run identities
    must be host-independent. build_run_summary normalizes these automatically;
    this guard catches summaries produced by other (or hand-rolled) emitters so
    CI blocks any run that would reintroduce a host path."""
    config = payload["config"]
    offenders = [
        f"method_params.{k}"
        for k, v in config.get("method_params", {}).items()
        if _is_abs_path(v)
    ]
    dataset_config = config.get("dataset_config", {})
    offenders += [
        f"dataset_config.{k}"
        for k in ("topics", "index")
        if _is_abs_path(dataset_config.get(k))
    ]
    if offenders:
        raise ValidationError(
            f"machine-specific absolute path(s) in config: {offenders}. "
            "Use portable relative paths (build_run_summary normalizes these "
            "automatically)."
        )


def _validate_hashes(payload: Mapping[str, Any]) -> None:
    config = payload["config"]
    pipeline = payload["pipeline"]

    expected_params = compute_params_hash(
        method_id=pipeline["method_id"],
        model=pipeline["model"],
        method_params=config["method_params"],
        llm_config=config["llm_config"],
    )
    if payload["params_hash"] != expected_params:
        raise ValidationError(
            f"params_hash mismatch: stored '{payload['params_hash']}', "
            f"recomputed '{expected_params}'. The JSON has been edited or was "
            f"generated by an inconsistent emitter."
        )

    expected_run = compute_run_id(payload)
    if payload["run_id"] != expected_run:
        raise ValidationError(
            f"run_id mismatch: stored '{payload['run_id']}', "
            f"recomputed '{expected_run}'."
        )


def _closest(name: str, candidates: list[str]) -> str | None:
    """Return the closest candidate by simple character-set overlap."""
    if not candidates:
        return None
    name_chars = set(name.lower())
    scored = sorted(
        candidates,
        key=lambda c: -len(name_chars & set(c.lower())),
    )
    best = scored[0]
    overlap = len(name_chars & set(best.lower()))
    return best if overlap >= max(3, len(name_chars) // 3) else None
