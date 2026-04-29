# Run Summary Schema (v1)

This document mirrors `reproducibility/schema.json` in human-readable form. Both files are kept in sync via `reproducibility/tests/test_repro_schema.py`, which embeds the same canonical fixture used here.

## Top-level fields

| Field | Type | Required | Description |
|---|---|---|---|
| `schema_version` | `int` | yes | Always `1`. Bumping is a breaking change. |
| `run_id` | `string` (16-char hex) | yes | SHA-256 prefix over the payload minus volatile fields. Identifies a specific execution. |
| `params_hash` | `string` (8-char hex) | yes | SHA-256 prefix over `(method_id, model, method_params, llm_config)`. Doubles as the on-disk filename. |
| `submitted_at` | ISO 8601 UTC | yes | Wall-clock time the JSON was generated. Excluded from `run_id`. |
| `querygym_version` | `string` | yes | `querygym.__version__` at emit time. |
| `environment` | object | yes | Python version, platform, optional git commit. |
| `pipeline` | object | yes | dataset_id, method_id, model, steps_completed, total_time_seconds. |
| `config` | object | yes | method_params, llm_config, searcher, dataset_config. |
| `metrics` | object | yes | Flat `metric_name -> float`. Must have ≥1 entry. |
| `timing` | object | yes | Per-step seconds. |
| `artifacts` | object | yes | Sibling `run_file` and `reformulated_queries` filenames. |

## Validation rules (beyond the static schema)

These are enforced by `reproducibility.lib.validate(...)` at runtime:

1. `pipeline.dataset_id` must be a key in `dataset_registry.yaml`.
2. `pipeline.method_id` must be registered via `@register_method(...)` in `querygym/methods/`.
3. Each `metrics` key must be in the dataset's `output.eval_metrics` (after normalizing dots to underscores: `ndcg_cut.10` → `ndcg_cut_10`).
4. `params_hash` is recomputed from `(method_id, model, method_params, llm_config)` and must equal the stored value.
5. `run_id` is recomputed from the payload (minus `run_id`, `submitted_at`, `environment`) and must equal the stored value.
6. `artifacts.run_file` must equal `{params_hash}.run.txt`; `artifacts.reformulated_queries` must equal `{params_hash}.queries.tsv`.

Hand-editing a metric value without re-running the emitter will fail validation (rule 5). This catches silent tampering.

## Hashing details

```python
def compute_params_hash(method_id, model, method_params, llm_config) -> str:
    payload = {"method_id": ..., "model": ..., "method_params": ..., "llm_config": ...}
    return sha256(json.dumps(payload, sort_keys=True, separators=(",",":"))).hexdigest()[:8]

def compute_run_id(payload) -> str:
    stripped = {k: v for k, v in payload.items() if k not in ("run_id", "submitted_at", "environment")}
    return sha256(json.dumps(stripped, sort_keys=True, separators=(",",":"))).hexdigest()[:16]
```

`json.dumps(..., sort_keys=True)` makes hashes invariant to key ordering. Hashes change when any field they cover changes.

## Canonical example

This is `reproducibility/tests/fixtures/sample_run.json` — used by tests, embedded here, and produced from the inputs in `test_repro_schema._build_kwargs()`:

```json
{
  "schema_version": 1,
  "run_id": "cabe83ca1236a3bb",
  "params_hash": "ddb15ccf",
  "submitted_at": "2026-04-29T10:14:22Z",
  "querygym_version": "0.3.0",
  "environment": {
    "python_version": "3.10.13",
    "platform": "Linux-5.15.0-x86_64",
    "git_commit": "5c46a51"
  },
  "pipeline": {
    "dataset_id": "msmarco-v1-passage.trecdl2019",
    "method_id": "query2e",
    "model": "gpt-4.1-mini",
    "steps_completed": ["reformulate", "retrieve", "evaluate"],
    "total_time_seconds": 89.37
  },
  "config": {
    "method_params": {"mode": "zs"},
    "llm_config": {"temperature": 1.0, "max_tokens": 128, "top_p": 1.0},
    "searcher": {"name": "UserPyseriniWrapper", "type": "user_pyserini"},
    "dataset_config": {
      "topics": "dl19-passage",
      "index": "msmarco-v1-passage",
      "num_queries": 43,
      "bm25_weights": {"k1": 0.9, "b": 0.4}
    }
  },
  "metrics": {
    "map": 0.3709,
    "ndcg_cut_10": 0.5679,
    "recall_1000": 0.8384
  },
  "timing": {
    "reformulation_seconds": 65.24,
    "retrieval_seconds": 3.01,
    "evaluation_seconds": 10.53
  },
  "artifacts": {
    "run_file": "ddb15ccf.run.txt",
    "reformulated_queries": "ddb15ccf.queries.tsv"
  }
}
```

## On-disk layout

A run lives under:

```
reproducibility/data/runs/{dataset_id}/{method_id}/{model}/{params_hash}.{json,run.txt,queries.tsv}
```

The three sibling files together describe one run completely. The `.run.txt` is a TREC-format retrieval run that allows independent re-evaluation with `pytrec_eval`; the `.queries.tsv` lets reviewers spot-check reformulations.

## Bumping the schema

Future schema changes require:
1. Bumping `SCHEMA_VERSION` in `reproducibility/lib/emit.py` to 2.
2. Updating `schema.json`'s `schema_version.const` to 2.
3. Re-emitting all existing JSONs under v2 (one bulk PR).
4. Updating the dashboard product to consume v2.

The schema is intentionally hard to change so that the leaderboard's history stays comparable.
