/**
 * Build the three-step reproduce commands (reformulate → retrieve → evaluate)
 * from a run summary. Shared by the home-matrix expand panel and the
 * `/runs/[run_id]` detail page so the commands stay in sync.
 */

export interface RunLike {
  run_id?: string;
  dataset_id: string;
  method_id: string;
  model: string;
  retriever_id?: string;
  metrics?: Record<string, number>;
  config?: {
    method_params?: Record<string, unknown>;
    llm_config?: { temperature?: number; max_tokens?: number; [k: string]: unknown };
    dataset_config?: { topics?: string; index?: string; [k: string]: unknown };
    retrieval?: {
      paradigm?: string;
      retriever_id?: string;
      params?: Record<string, unknown>;
    };
  };
}

export interface ReproduceCmds {
  reformulate: string;
  retrieve: string | null;
  evaluate: string;
  paradigm: string;
  qrels: string;
}

// method_params we surface in the reproduce snippet — strip locally-pathy keys
// that won't apply on a fresh checkout.
const PARAM_KEYS_TO_DROP = new Set([
  "judge_rel_mode",
  "collection_path",
  "train_queries_path",
  "train_qrels_path",
  "dataset_type",
]);

export function buildReproduceCmds(run: RunLike): ReproduceCmds {
  const cfg = run.config ?? {};
  const retrieval = cfg.retrieval ?? {};
  const dsCfg = cfg.dataset_config ?? {};
  const llm = cfg.llm_config ?? {};
  const methodParams = (cfg.method_params ?? {}) as Record<string, unknown>;

  const cleanParams: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(methodParams)) {
    if (!PARAM_KEYS_TO_DROP.has(k)) cleanParams[k] = v;
  }
  const paramsJson = Object.keys(cleanParams).length
    ? JSON.stringify(cleanParams)
    : null;

  const reformulate = `python examples/querygym_pyserini/pipeline.py \\
    --dataset ${run.dataset_id} \\
    --method ${run.method_id} \\
    --model ${run.model} \\
    --steps reformulate \\
    --temperature ${llm.temperature ?? 1.0} \\
    --max-tokens ${llm.max_tokens ?? 128} \\${paramsJson ? `
    --method-params '${paramsJson}' \\` : ""}
    --output-dir outputs/reproduce`;

  const paradigm = retrieval.paradigm ?? "";
  const params = (retrieval.params ?? {}) as Record<string, unknown>;
  // BEIR BM25 indexes carry a `.flat` suffix; SPLADE/BGE variants drop it.
  const baseIndex = String(dsCfg.index ?? "").replace(/\.flat$/, "");

  let retrieve: string | null = null;
  if (paradigm === "lexical") {
    retrieve = `python -m pyserini.search.lucene \\
  --threads 16 --batch-size 128 \\
  --index ${dsCfg.index ?? "<pyserini-index>"} \\
  --topics outputs/reproduce/queries/reformulated_queries.tsv \\
  --bm25 --k1 ${params.k1 ?? 0.9} --b ${params.b ?? 0.4} \\
  --output run.txt \\
  --hits 1000`;
  } else if (paradigm === "learned_sparse") {
    retrieve = `python -m pyserini.search.lucene \\
  --threads 16 --batch-size 128 \\
  --index ${baseIndex || "<pyserini-index>"}.splade-pp-ed \\
  --topics outputs/reproduce/queries/reformulated_queries.tsv \\
  --encoder ${params.model ?? "naver/splade-cocondenser-ensembledistil"} \\
  --output run.txt \\
  --hits 1000 --impact`;
  } else if (paradigm === "dense") {
    retrieve = `python -m pyserini.search.faiss \\
  --threads 16 --batch-size 128 \\
  --index ${baseIndex || "<pyserini-index>"}.bge-base-en-v1.5 \\
  --topics outputs/reproduce/queries/reformulated_queries.tsv \\
  --encoder ${params.encoder ?? "BAAI/bge-base-en-v1.5"} \\
  --output run.txt \\
  --hits 1000`;
  }

  const trecMetrics = Object.keys(run.metrics ?? {})
    .map((m) => m.replace(/_/g, "."))
    .join(" -m ");
  const qrels = dsCfg.topics ?? "<qrels>";
  const evaluate = `python -m pyserini.eval.trec_eval -c -m ${trecMetrics || "ndcg_cut.10"} \\
  ${qrels} run.txt`;

  return { reformulate, retrieve, evaluate, paradigm, qrels };
}

/** Pretty hint for the retrieve step header. */
export function retrieveHint(retrieverDisplay: string, paradigm: string): string {
  return `pyserini · ${retrieverDisplay}${paradigm ? ` (${paradigm})` : ""}`;
}

/** Pretty hint for the evaluate step header. */
export function evaluateHint(metricKeys: string[]): string {
  if (!metricKeys.length) return "trec_eval";
  const pretty = metricKeys.map((k) =>
    k === "ndcg_cut_10" ? "nDCG@10"
    : k === "recall_100" ? "R@100"
    : k === "recall_1000" ? "R@1k"
    : k === "map" ? "MAP"
    : k
  );
  return `trec_eval · ${pretty.join(" + ")}`;
}
