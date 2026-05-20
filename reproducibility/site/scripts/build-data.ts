/**
 * build-data.ts — turn reproducibility/data + dataset_registry.yaml +
 * retriever_registry.yaml into per-view JSON shards consumed by Astro pages.
 *
 * Inputs:
 *   - ../data/results.csv            (canonical aggregate; one row per (run, metric))
 *   - ../data/manifest.json          (run/row counts + content hash)
 *   - ../data/runs/.../*.json         (per-run detail used by /runs/[run_id])
 *   - ../../dataset_registry.yaml    (full dataset list)
 *   - ../../reproducibility/retriever_registry.yaml  (retriever slug → display + paradigm)
 *
 * Outputs (gitignored, written to src/data/):
 *   - overview.json    summary used on /
 *   - datasets.json    [{id, name, run_count, eval_metrics}]
 *   - methods.json     [{id, run_count}]
 *   - models.json      [{id, run_count}]
 *   - retrievers.json  [{id, display_name, paradigm, run_count}]
 *   - matrix.json      flat matrix: rows = (method, model, retriever), values per dataset
 *   - runs.json        full run index, keyed by run_id
 *   - views/dataset-{id}.json   (per-dataset table — used by /datasets/{id})
 *   - views/method-{id}.json    (per-method matrix — used by /methods/{id})
 *   - views/model-{id}.json     (per-model matrix — used by /models/{id})
 *   - views/retriever-{id}.json (per-retriever matrix — used by /retrievers/{id})
 *
 * Empty input is fine — empty shards still emit so pages render the empty state.
 * Run as: pnpm -F @qg/leaderboard build:data
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import Papa from "papaparse";
import yaml from "js-yaml";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SITE_ROOT = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(SITE_ROOT, "..", "..");
const DATA_DIR = path.join(REPO_ROOT, "reproducibility", "data");
const RUNS_DIR = path.join(DATA_DIR, "runs");
const RESULTS_CSV = path.join(DATA_DIR, "results.csv");
const MANIFEST_JSON = path.join(DATA_DIR, "manifest.json");
const REGISTRY_YAML = path.join(REPO_ROOT, "dataset_registry.yaml");
const RETRIEVER_REGISTRY_YAML = path.join(
  REPO_ROOT,
  "reproducibility",
  "retriever_registry.yaml",
);

const OUT_DIR = path.join(SITE_ROOT, "src", "data");
const VIEWS_DIR = path.join(OUT_DIR, "views");

const GH_RAW = "https://raw.githubusercontent.com/ls3-lab/QueryGym/main";
const GH_BLOB = "https://github.com/ls3-lab/QueryGym/blob/main";

interface ResultRow {
  schema_version: number;
  run_id: string;
  dataset_id: string;
  method_id: string;
  model: string;
  retriever_id: string;
  retriever: string; // display name from registry
  params_hash: string;
  method_params_json: string;
  llm_temperature: number;
  llm_max_tokens: number;
  metric: string;
  value: number;
  num_queries: number;
  total_time_seconds: number;
  querygym_version: string;
  run_file_path: string;
}

interface DatasetEntry {
  id: string;
  name: string;
  topics?: string;
  index?: string;
  bm25_weights?: { k1: number; b: number };
  eval_metrics: string[];
  run_count: number;
}

interface RetrieverEntry {
  id: string;
  display_name: string;
  paradigm: string;
  run_count: number;
}

interface RunDetail {
  run_id: string;
  params_hash: string;
  dataset_id: string;
  method_id: string;
  model: string;
  retriever_id: string;
  retriever_display: string;
  paradigm: string;
  model_display: string;
  metrics: Record<string, number>;
  config: Record<string, unknown>;
  timing: Record<string, number>;
  total_time_seconds: number;
  num_queries: number;
  querygym_version: string;
  submitted_at: string;
  // Artifacts present in the payload (may be empty for legacy rows).
  artifacts: {
    json_url: string;
    run_file_url: string | null;
    queries_url: string | null;
  };
}

function ensureDir(p: string) {
  fs.mkdirSync(p, { recursive: true });
}

function writeJSON(filePath: string, data: unknown) {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2) + "\n", "utf-8");
}

function readCSV(): ResultRow[] {
  if (!fs.existsSync(RESULTS_CSV)) return [];
  const text = fs.readFileSync(RESULTS_CSV, "utf-8");
  const parsed = Papa.parse<ResultRow>(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });
  return parsed.data.filter((r) => r.run_id);
}

function readManifest(): { row_count: number; run_count: number; content_hash: string } {
  if (!fs.existsSync(MANIFEST_JSON)) {
    return { row_count: 0, run_count: 0, content_hash: "" };
  }
  return JSON.parse(fs.readFileSync(MANIFEST_JSON, "utf-8"));
}

function normalizeMetric(name: string): string {
  return name.replace(/\./g, "_");
}

function readDatasetRegistry(): Record<string, Omit<DatasetEntry, "run_count">> {
  if (!fs.existsSync(REGISTRY_YAML)) return {};
  const doc = yaml.load(fs.readFileSync(REGISTRY_YAML, "utf-8")) as {
    datasets?: Record<string, any>;
  };
  const out: Record<string, Omit<DatasetEntry, "run_count">> = {};
  for (const [id, entry] of Object.entries(doc.datasets ?? {})) {
    out[id] = {
      id,
      name: entry.name ?? id,
      topics: entry.topics?.name,
      index: entry.index?.name,
      bm25_weights: entry.bm25_weights,
      eval_metrics: (entry.output?.eval_metrics ?? []).map(normalizeMetric),
    };
  }
  return out;
}

function readRetrieverRegistry(): Record<string, { display_name: string; paradigm: string }> {
  if (!fs.existsSync(RETRIEVER_REGISTRY_YAML)) return {};
  const doc = yaml.load(fs.readFileSync(RETRIEVER_REGISTRY_YAML, "utf-8")) as {
    retrievers?: Record<string, { display_name: string; paradigm: string }>;
  };
  return doc.retrievers ?? {};
}

function* iterRunFiles(): Generator<string> {
  if (!fs.existsSync(RUNS_DIR)) return;
  const stack = [RUNS_DIR];
  while (stack.length) {
    const dir = stack.pop()!;
    for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, ent.name);
      if (ent.isDirectory()) stack.push(full);
      else if (ent.isFile() && ent.name.endsWith(".json")) yield full;
    }
  }
}

function readRunDetails(retrievers: Record<string, { display_name: string; paradigm: string }>): Record<string, RunDetail> {
  const out: Record<string, RunDetail> = {};
  for (const filePath of iterRunFiles()) {
    const payload = JSON.parse(fs.readFileSync(filePath, "utf-8"));
    const rel = path.relative(REPO_ROOT, filePath).split(path.sep).join("/");
    const dirOnGitHub = path.dirname(rel);
    const hash = payload.params_hash;
    const retr = payload.config?.retrieval ?? {};
    const retrId = retr.retriever_id ?? "";
    const artifacts = payload.artifacts ?? {};
    out[payload.run_id] = {
      run_id: payload.run_id,
      params_hash: hash,
      dataset_id: payload.pipeline.dataset_id,
      method_id: payload.pipeline.method_id,
      model: payload.pipeline.model,
      model_display: displayModel(payload.pipeline.model),
      retriever_id: retrId,
      retriever_display: retrievers[retrId]?.display_name ?? retrId,
      paradigm: retrievers[retrId]?.paradigm ?? retr.paradigm ?? "",
      metrics: payload.metrics,
      config: payload.config,
      timing: payload.timing,
      total_time_seconds: payload.pipeline.total_time_seconds,
      num_queries: payload.config.dataset_config.num_queries,
      querygym_version: payload.querygym_version,
      submitted_at: payload.submitted_at,
      artifacts: {
        json_url: `${GH_BLOB}/${rel}`,
        run_file_url: artifacts.run_file ? `${GH_RAW}/${dirOnGitHub}/${artifacts.run_file}` : null,
        queries_url: artifacts.reformulated_queries ? `${GH_RAW}/${dirOnGitHub}/${artifacts.reformulated_queries}` : null,
      },
    };
  }
  return out;
}

// ---------- view builders ---------------------------------------------------

function buildPerDatasetViews(
  rows: ResultRow[],
  datasets: Record<string, Omit<DatasetEntry, "run_count">>,
) {
  const byDataset = new Map<string, ResultRow[]>();
  for (const r of rows) {
    if (!byDataset.has(r.dataset_id)) byDataset.set(r.dataset_id, []);
    byDataset.get(r.dataset_id)!.push(r);
  }

  for (const [datasetId, dsRows] of byDataset) {
    const allowed = datasets[datasetId]?.eval_metrics ?? [];

    // Pivot to one row per (logical_method, model, retriever). Variants are
    // folded by max value per metric — matches the home matrix.
    const map = new Map<string, any>();
    for (const r of dsRows) {
      const lm = logicalMethod(r.method_id, r.method_params_json);
      const key = `${lm.id}|${r.model}|${r.retriever_id}`;
      if (!map.has(key)) {
        map.set(key, {
          method_id: lm.id,
          method_display: lm.display,
          model: r.model,
          model_display: displayModel(r.model),
          retriever_id: r.retriever_id,
          retriever_display: r.retriever,
          run_id: r.run_id,            // populated/overwritten by the best cell
          metrics: {} as Record<string, number>,
          best_for: {} as Record<string, boolean>,
        });
      }
      const row = map.get(key);
      if (row.metrics[r.metric] === undefined || r.value > row.metrics[r.metric]) {
        row.metrics[r.metric] = r.value;
        row.run_id = r.run_id;
      }
    }

    // best_for flags relative to the rows above.
    const list = Array.from(map.values());
    for (const m of allowed) {
      let best = -Infinity;
      let bestRow: any = null;
      for (const row of list) {
        const v = row.metrics[m];
        if (v !== undefined && v > best) { best = v; bestRow = row; }
      }
      if (bestRow) bestRow.best_for[m] = true;
    }

    writeJSON(path.join(VIEWS_DIR, `dataset-${datasetId}.json`), {
      dataset_id: datasetId,
      dataset: datasets[datasetId] ?? { id: datasetId, name: datasetId, eval_metrics: allowed },
      metric_columns: allowed,
      runs: list,
    });
  }
}

/**
 * Logical method = the leaderboard-display unit.
 *
 * - For `query2doc`, the three `mode` variants (zs/fs/cot) are surfaced as
 *   separate logical methods (`query2doc-zs`, `query2doc-fs`, `query2doc-cot`),
 *   matching how the SIGIR paper presents them.
 * - For every other method_id, identity.
 *
 * This collapses accidental param noise in legacy runs (e.g. an extra
 * `judge_rel_mode` field, machine-specific paths in `query2doc-fs`) into the
 * same display row, while keeping genuinely different variants distinct.
 */
function logicalMethod(
  method_id: string,
  paramsJson: string,
): { id: string; display: string } {
  if (method_id === "query2doc") {
    try {
      const p = JSON.parse(paramsJson || "{}");
      const mode = String(p.mode ?? "zs").toLowerCase();
      return { id: `query2doc-${mode}`, display: `Q2D (${mode.toUpperCase()})` };
    } catch {
      return { id: method_id, display: method_id };
    }
  }
  return { id: method_id, display: method_id };
}

/**
 * One matrix-style row keyed by an "axis" tuple (e.g. (logical_method, model,
 * retriever) for the home page). Columns are dataset × metric. Each cell
 * carries value + best flag + run_id.
 *
 * Within each axis-group, when multiple legacy variants exist for the same
 * cell, the one with the highest metric value wins (display-side dedup).
 */
function buildMatrix(
  rows: ResultRow[],
  axisKey: (r: ResultRow, lm: { id: string; display: string }) => string,
  axisExtract: (r: ResultRow, lm: { id: string; display: string }) => Record<string, string>,
) {
  const map = new Map<string, any>();

  for (const r of rows) {
    const lm = logicalMethod(r.method_id, r.method_params_json);
    const key = axisKey(r, lm);
    if (!map.has(key)) {
      map.set(key, {
        ...axisExtract(r, lm),
        run_ids: {} as Record<string, string>,    // dataset_id → run_id of best cell
        values: {} as Record<string, Record<string, { value: number; best: boolean }>>,
      });
    }
    const row = map.get(key);
    if (!row.values[r.dataset_id]) row.values[r.dataset_id] = {};
    const cur = row.values[r.dataset_id][r.metric];
    if (!cur || r.value > cur.value) {
      row.values[r.dataset_id][r.metric] = { value: r.value, best: false };
      // The "winning" run for this cell (used for the row → run_detail link).
      // If different metrics inside the same cell come from different legacy
      // variants, keep the most recent assignment (any is correct).
      row.run_ids[r.dataset_id] = r.run_id;
    }
  }

  // "Best in column" across all axis rows.
  const list = Array.from(map.values());
  const bestPerCol = new Map<string, { value: number; rowIdx: number }>();
  list.forEach((row, idx) => {
    for (const [ds, metrics] of Object.entries(row.values)) {
      for (const [m, cell] of Object.entries(metrics as Record<string, { value: number; best: boolean }>)) {
        const colKey = `${ds}|${m}`;
        const cur = bestPerCol.get(colKey);
        if (!cur || cell.value > cur.value) bestPerCol.set(colKey, { value: cell.value, rowIdx: idx });
      }
    }
  });
  bestPerCol.forEach(({ rowIdx }, colKey) => {
    const [ds, m] = colKey.split("|");
    list[rowIdx].values[ds][m].best = true;
  });
  return list;
}

function buildHomeMatrix(
  rows: ResultRow[],
  datasets: Record<string, Omit<DatasetEntry, "run_count">>,
) {
  const matrixRows = buildMatrix(
    rows,
    (_r, lm) => `${lm.id}|${_r.model}|${_r.retriever_id}`,
    (r, lm) => ({
      method_id: lm.id,
      method_display: lm.display,
      model: r.model,
      model_display: displayModel(r.model),
      retriever_id: r.retriever_id,
      retriever_display: r.retriever,
    }),
  );

  // Dataset columns: derive primary/secondary metric from what's ACTUALLY in
  // the matrix data (not from the registry whitelist, which may over-specify).
  // primary  = ndcg_cut_10 if present, else the first metric found.
  // secondary = recall_1000 if present, else recall_100, else any other metric.
  const datasetCols = Object.values(datasets)
    .sort((a, b) => a.id.localeCompare(b.id))
    .map((d) => {
      const present = new Set<string>();
      for (const row of matrixRows) {
        for (const m of Object.keys(row.values?.[d.id] ?? {})) present.add(m);
      }
      const arr = Array.from(present);
      const primary =
        present.has("ndcg_cut_10") ? "ndcg_cut_10" : arr[0] ?? null;
      const secondary =
        present.has("recall_1000")
          ? "recall_1000"
          : present.has("recall_100")
            ? "recall_100"
            : arr.find((m) => m !== primary) ?? null;
      return {
        id: d.id,
        name: d.name,
        primary_metric: primary,
        secondary_metric: secondary,
        all_metrics: arr.sort(),
      };
    });

  return {
    dataset_columns: datasetCols,
    rows: matrixRows,
  };
}

function buildPerModelViews(rows: ResultRow[]) {
  const byModel = new Map<string, ResultRow[]>();
  for (const r of rows) {
    if (!byModel.has(r.model)) byModel.set(r.model, []);
    byModel.get(r.model)!.push(r);
  }
  for (const [model, mRows] of byModel) {
    const matrix = buildMatrix(
      mRows,
      (r, lm) => `${lm.id}|${r.retriever_id}`,
      (r, lm) => ({
        method_id: lm.id,
        method_display: lm.display,
        retriever_id: r.retriever_id,
        retriever_display: r.retriever,
      }),
    );
    writeJSON(path.join(VIEWS_DIR, `model-${encodePathSegment(model)}.json`), {
      model,
      rows: matrix,
    });
  }
}

function buildPerMethodViews(rows: ResultRow[]) {
  // Group by *logical* method (so q2d-zs / q2d-fs / q2d-cot each get their
  // own /methods/{id} page, mirroring how the SIGIR paper presents them).
  const byLogicalMethod = new Map<string, { display: string; rows: ResultRow[] }>();
  for (const r of rows) {
    const lm = logicalMethod(r.method_id, r.method_params_json);
    if (!byLogicalMethod.has(lm.id)) {
      byLogicalMethod.set(lm.id, { display: lm.display, rows: [] });
    }
    byLogicalMethod.get(lm.id)!.rows.push(r);
  }
  for (const [method_id, { display, rows: mRows }] of byLogicalMethod) {
    const matrix = buildMatrix(
      mRows,
      (r) => `${r.model}|${r.retriever_id}`,
      (r) => ({
        model: r.model,
        model_display: displayModel(r.model),
        retriever_id: r.retriever_id,
        retriever_display: r.retriever,
      }),
    );
    writeJSON(path.join(VIEWS_DIR, `method-${method_id}.json`), {
      method_id,
      method_display: display,
      rows: matrix,
    });
  }
}

function buildPerRetrieverViews(rows: ResultRow[]) {
  const byRetriever = new Map<string, ResultRow[]>();
  for (const r of rows) {
    if (!byRetriever.has(r.retriever_id)) byRetriever.set(r.retriever_id, []);
    byRetriever.get(r.retriever_id)!.push(r);
  }
  for (const [retriever_id, mRows] of byRetriever) {
    const matrix = buildMatrix(
      mRows,
      (r, lm) => `${lm.id}|${r.model}`,
      (r, lm) => ({
        method_id: lm.id,
        method_display: lm.display,
        model: r.model,
        model_display: displayModel(r.model),
      }),
    );
    writeJSON(path.join(VIEWS_DIR, `retriever-${retriever_id}.json`), {
      retriever_id,
      rows: matrix,
    });
  }
}

// Some model ids contain "/" (e.g. "openai/gpt-4.1"); URL-encode for file paths.
function encodePathSegment(s: string): string {
  return s.replace(/\//g, "__");
}

// Strip provider prefix from a model id for display: "openai/gpt-4.1" → "gpt-4.1".
// The canonical id stays in the data; this is purely cosmetic.
function displayModel(s: string): string {
  const i = s.indexOf("/");
  return i >= 0 ? s.slice(i + 1) : s;
}

// ---------- main ------------------------------------------------------------

function main() {
  ensureDir(OUT_DIR);
  ensureDir(VIEWS_DIR);

  const rows = readCSV();
  const manifest = readManifest();
  const datasets = readDatasetRegistry();
  const retrieverReg = readRetrieverRegistry();
  const runDetails = readRunDetails(retrieverReg);

  // Counts.
  const datasetCounts = new Map<string, number>();
  const methodCounts = new Map<string, number>();
  const modelCounts = new Map<string, number>();
  const retrieverCounts = new Map<string, number>();
  const seen = new Set<string>();
  const methodDisplay = new Map<string, string>();
  for (const r of rows) {
    if (seen.has(r.run_id)) continue;
    seen.add(r.run_id);
    const lm = logicalMethod(r.method_id, r.method_params_json);
    methodDisplay.set(lm.id, lm.display);
    datasetCounts.set(r.dataset_id, (datasetCounts.get(r.dataset_id) ?? 0) + 1);
    methodCounts.set(lm.id, (methodCounts.get(lm.id) ?? 0) + 1);
    modelCounts.set(r.model, (modelCounts.get(r.model) ?? 0) + 1);
    retrieverCounts.set(r.retriever_id, (retrieverCounts.get(r.retriever_id) ?? 0) + 1);
  }

  // Per-X views (dataset, model, method, retriever).
  buildPerDatasetViews(rows, datasets);
  buildPerModelViews(rows);
  buildPerMethodViews(rows);
  buildPerRetrieverViews(rows);

  // Home matrix.
  writeJSON(path.join(OUT_DIR, "matrix.json"), buildHomeMatrix(rows, datasets));

  // Top-level shards.
  writeJSON(path.join(OUT_DIR, "overview.json"), {
    run_count: manifest.run_count,
    row_count: manifest.row_count,
    content_hash: manifest.content_hash,
    dataset_count: Object.keys(datasets).length,
    method_count: methodCounts.size,
    model_count: modelCounts.size,
    retriever_count: retrieverCounts.size,
  });

  const datasetList: DatasetEntry[] = Object.values(datasets).map((d) => ({
    ...d,
    run_count: datasetCounts.get(d.id) ?? 0,
  }));
  datasetList.sort((a, b) => a.id.localeCompare(b.id));
  writeJSON(path.join(OUT_DIR, "datasets.json"), datasetList);

  const methodList = Array.from(methodCounts.entries())
    .map(([id, run_count]) => ({ id, run_count, display: methodDisplay.get(id) ?? id }))
    .sort((a, b) => a.id.localeCompare(b.id));
  writeJSON(path.join(OUT_DIR, "methods.json"), methodList);

  const modelList = Array.from(modelCounts.entries())
    .map(([id, run_count]) => ({
      id,
      display: displayModel(id),
      run_count,
      slug: encodePathSegment(id),
    }))
    .sort((a, b) => a.id.localeCompare(b.id));
  writeJSON(path.join(OUT_DIR, "models.json"), modelList);

  const retrieverList: RetrieverEntry[] = Array.from(retrieverCounts.entries())
    .map(([id, run_count]) => ({
      id,
      display_name: retrieverReg[id]?.display_name ?? id,
      paradigm: retrieverReg[id]?.paradigm ?? "",
      run_count,
    }))
    .sort((a, b) => a.id.localeCompare(b.id));
  writeJSON(path.join(OUT_DIR, "retrievers.json"), retrieverList);

  writeJSON(path.join(OUT_DIR, "runs.json"), runDetails);

  console.log(
    `build-data: ${rows.length} rows, ${Object.keys(runDetails).length} runs, ` +
      `${Object.keys(datasets).length} datasets, ${methodList.length} methods, ` +
      `${modelList.length} models, ${retrieverList.length} retrievers`,
  );
}

main();
