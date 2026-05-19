"""Summary assembly for the QueryGym+Pyserini pipeline.

Extracted from pipeline.py so it can be unit-tested without importing
querygym/pyserini. Imports only reproducibility.lib + stdlib + yaml.
The shipped pipeline is lexical BM25/Pyserini only, so this emits a
`retrieval` block with paradigm 'lexical'. Non-lexical retrievers reach
the schema via reproducibility.lib.build_run_summary directly (researcher
runs) or the legacy backfill adapter (sub-project B).
"""

from __future__ import annotations

from pathlib import Path

from reproducibility.lib import build_run_summary


def _load_dataset_config_from_registry(dataset_id: str, registry_path: str) -> dict | None:
    """Pull dataset_config fields from dataset_registry.yaml. None if not registered."""
    try:
        import yaml
        with open(registry_path, 'r') as f:
            registry = yaml.safe_load(f) or {}
    except Exception:
        return None
    entry = registry.get('datasets', {}).get(dataset_id)
    if not entry:
        return None
    return {
        'topics': entry.get('topics', {}).get('name', ''),
        'index': entry.get('index', {}).get('name', ''),
        'num_queries': 0,  # filled by reformulation metadata when available
        'bm25_weights': entry.get('bm25_weights', {'k1': 0.0, 'b': 0.0}),
    }


def _build_v1_summary(
    *, results, dataset_name, method, model, method_params, llm_config,
    steps, pipeline_time, registry_path, queries_file, index_name,
) -> dict:
    """Pull fields from per-step metadata and call reproducibility.lib.build_run_summary."""
    reform = results.get('reformulation', {})
    retrieval_step = results.get('retrieval', {})
    evaluation = results.get('evaluation', {})

    reform_inner = reform.get('reformulation', {}) if isinstance(reform, dict) else {}
    dataset_inner = reform.get('dataset', {}) if isinstance(reform, dict) else {}

    # Resolve dataset_config: prefer reformulation metadata, fall back to registry.
    if dataset_inner.get('topics') or dataset_inner.get('index'):
        dataset_config = {
            'topics': dataset_inner.get('topics') or '',
            'index': dataset_inner.get('index') or (index_name or ''),
            'num_queries': int(dataset_inner.get('num_queries') or 0),
            'bm25_weights': dataset_inner.get('bm25_weights') or {'k1': 0.0, 'b': 0.0},
        }
    else:
        dataset_config = _load_dataset_config_from_registry(dataset_name, registry_path) or {
            'topics': '',
            'index': index_name or '',
            'num_queries': 0,
            'bm25_weights': {'k1': 0.0, 'b': 0.0},
        }
        if dataset_inner.get('num_queries'):
            dataset_config['num_queries'] = int(dataset_inner['num_queries'])

    # Retrieval block: the shipped pipeline is lexical BM25/Pyserini only.
    searcher_info = reform_inner.get('searcher') or {}
    bm25_weights = dataset_config.get('bm25_weights') or {'k1': 0.0, 'b': 0.0}
    searcher_class = searcher_info.get('searcher_class') or 'LuceneSearcher'
    retrieval = {
        'retriever_id': 'bm25',
        'paradigm': 'lexical',
        'params': {'k1': float(bm25_weights.get('k1', 0.0)),
                   'b': float(bm25_weights.get('b', 0.0))},
        'implementation': f'pyserini:{searcher_class}',
    }

    eff_method_params = reform_inner.get('method_params') or dict(method_params or {})
    eff_method_params = {
        k: v for k, v in eff_method_params.items()
        if not callable(v) and not hasattr(v, '__dict__') or isinstance(v, (dict, list, str, int, float, bool, type(None)))
    }
    eff_llm_config = reform_inner.get('llm_config') or dict(llm_config or {})
    if 'temperature' not in eff_llm_config:
        eff_llm_config['temperature'] = (llm_config or {}).get('temperature', 0.0)
    if 'max_tokens' not in eff_llm_config:
        eff_llm_config['max_tokens'] = (llm_config or {}).get('max_tokens', 1)

    timing = {
        'reformulation_seconds': float(reform.get('timing', {}).get('total_time_seconds', 0.0)),
        'retrieval_seconds': float(retrieval_step.get('timing', {}).get('total_time_seconds', 0.0)),
        'evaluation_seconds': float(evaluation.get('timing', {}).get('eval_time_seconds', 0.0)),
    }

    return build_run_summary(
        dataset_id=dataset_name or queries_file.stem if queries_file else (dataset_name or 'unknown'),
        method_id=method,
        model=model,
        method_params=eff_method_params,
        llm_config=eff_llm_config,
        retrieval=retrieval,
        dataset_config=dataset_config,
        metrics=evaluation.get('results', {}),
        timing=timing,
        steps_completed=steps,
        total_time_seconds=pipeline_time,
    )
