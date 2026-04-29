/**
 * build-data.ts — turn reproducibility/data + dataset_registry.yaml + method
 * registry into per-view JSON shards consumed by Astro pages.
 *
 * Inputs:
 *   - ../data/results.csv            (canonical aggregate; one row per (run, metric))
 *   - ../data/manifest.json          (run/row counts + content hash)
 *   - ../data/runs/.../*.json         (per-run detail used by /runs/[run_id])
 *   - ../../dataset_registry.yaml    (full dataset list, even when csv is empty)
 *
 * Outputs (gitignored, written to src/data/):
 *   - overview.json    summary used on /
 *   - datasets.json    [{id, name, run_count}]
 *   - methods.json     [{id, run_count}]
 *   - models.json      [{id, run_count}]
 *   - runs.json        full run index, keyed by run_id
 *   - views/dataset-{id}.json
 *   - views/method-{id}.json
 *   - views/model-{id}.json
 *
 * Empty input is fine — empty shards still emit so pages render the empty
 * state. Run as: pnpm -F @qg/leaderboard build:data
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

interface RunDetail {
  run_id: string;
  params_hash: string;
  dataset_id: string;
  method_id: string;
  model: string;
  metrics: Record<string, number>;
  config: Record<string, unknown>;
  timing: Record<string, number>;
  total_time_seconds: number;
  num_queries: number;
  querygym_version: string;
  submitted_at: string;
  artifacts: {
    json_url: string;
    run_file_url: string;
    queries_url: string;
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

function readRunDetails(): Record<string, RunDetail> {
  const out: Record<string, RunDetail> = {};
  for (const filePath of iterRunFiles()) {
    const payload = JSON.parse(fs.readFileSync(filePath, "utf-8"));
    const rel = path.relative(REPO_ROOT, filePath).split(path.sep).join("/");
    const dirOnGitHub = path.dirname(rel);
    const hash = payload.params_hash;
    out[payload.run_id] = {
      run_id: payload.run_id,
      params_hash: hash,
      dataset_id: payload.pipeline.dataset_id,
      method_id: payload.pipeline.method_id,
      model: payload.pipeline.model,
      metrics: payload.metrics,
      config: payload.config,
      timing: payload.timing,
      total_time_seconds: payload.pipeline.total_time_seconds,
      num_queries: payload.config.dataset_config.num_queries,
      querygym_version: payload.querygym_version,
      submitted_at: payload.submitted_at,
      artifacts: {
        json_url: `${GH_BLOB}/${rel}`,
        run_file_url: `${GH_RAW}/${dirOnGitHub}/${hash}.run.txt`,
        queries_url: `${GH_RAW}/${dirOnGitHub}/${hash}.queries.tsv`,
      },
    };
  }
  return out;
}

function buildViews(
  rows: ResultRow[],
  datasets: Record<string, Omit<DatasetEntry, "run_count">>,
) {
  // Index rows by composite keys so we can pivot per view.
  const byDataset = new Map<string, ResultRow[]>();
  const byMethod = new Map<string, ResultRow[]>();
  const byModel = new Map<string, ResultRow[]>();

  for (const r of rows) {
    if (!byDataset.has(r.dataset_id)) byDataset.set(r.dataset_id, []);
    byDataset.get(r.dataset_id)!.push(r);
    if (!byMethod.has(r.method_id)) byMethod.set(r.method_id, []);
    byMethod.get(r.method_id)!.push(r);
    if (!byModel.has(r.model)) byModel.set(r.model, []);
    byModel.get(r.model)!.push(r);
  }

  // Run counts (unique run_ids per facet).
  const datasetCounts = new Map<string, number>();
  const methodCounts = new Map<string, number>();
  const modelCounts = new Map<string, number>();
  const seen = new Set<string>();
  for (const r of rows) {
    const key = r.run_id;
    if (seen.has(key)) continue;
    seen.add(key);
    datasetCounts.set(r.dataset_id, (datasetCounts.get(r.dataset_id) ?? 0) + 1);
    methodCounts.set(r.method_id, (methodCounts.get(r.method_id) ?? 0) + 1);
    modelCounts.set(r.model, (modelCounts.get(r.model) ?? 0) + 1);
  }

  // Per-dataset views: pivot rows so each (method, model, params_hash) is a
  // single object with metric columns, and stamp best_for_metric flags.
  for (const [datasetId, dsRows] of byDataset) {
    const allowed = datasets[datasetId]?.eval_metrics ?? [];
    const bestPerMetric = new Map<string, { value: number; runId: string }>();
    for (const r of dsRows) {
      const cur = bestPerMetric.get(r.metric);
      if (!cur || r.value > cur.value) {
        bestPerMetric.set(r.metric, { value: r.value, runId: r.run_id });
      }
    }

    // Group rows into runs.
    const runs = new Map<string, any>();
    for (const r of dsRows) {
      if (!runs.has(r.run_id)) {
        runs.set(r.run_id, {
          run_id: r.run_id,
          method_id: r.method_id,
          model: r.model,
          params_hash: r.params_hash,
          method_params_json: r.method_params_json,
          llm_temperature: r.llm_temperature,
          llm_max_tokens: r.llm_max_tokens,
          num_queries: r.num_queries,
          total_time_seconds: r.total_time_seconds,
          querygym_version: r.querygym_version,
          run_file_path: r.run_file_path,
          metrics: {} as Record<string, number>,
          best_for: {} as Record<string, boolean>,
        });
      }
      const run = runs.get(r.run_id);
      run.metrics[r.metric] = r.value;
      run.best_for[r.metric] = bestPerMetric.get(r.metric)?.runId === r.run_id;
    }

    writeJSON(path.join(VIEWS_DIR, `dataset-${datasetId}.json`), {
      dataset_id: datasetId,
      dataset: datasets[datasetId] ?? { id: datasetId, name: datasetId, eval_metrics: allowed },
      metric_columns: allowed,
      runs: Array.from(runs.values()),
    });
  }

  // Method and model views: similar, but the "row" identity is dataset rather
  // than method/model.
  function writePivotView(
    dir: string,
    keyId: string,
    sourceRows: ResultRow[],
    rowKey: keyof ResultRow,
  ) {
    const items = new Map<string, any>();
    for (const r of sourceRows) {
      const id = String(r[rowKey]);
      if (!items.has(id)) {
        items.set(id, {
          [rowKey]: id,
          run_id: r.run_id,
          dataset_id: r.dataset_id,
          method_id: r.method_id,
          model: r.model,
          params_hash: r.params_hash,
          metrics: {} as Record<string, number>,
        });
      }
      items.get(id).metrics[r.metric] = r.value;
    }
    writeJSON(path.join(VIEWS_DIR, `${dir}-${keyId}.json`), {
      [`${dir}_id`]: keyId,
      items: Array.from(items.values()),
    });
  }

  for (const [methodId, mRows] of byMethod) {
    writePivotView("method", methodId, mRows, "run_id");
  }
  for (const [model, mRows] of byModel) {
    writePivotView("model", model, mRows, "run_id");
  }

  return { datasetCounts, methodCounts, modelCounts };
}

function main() {
  ensureDir(OUT_DIR);
  ensureDir(VIEWS_DIR);

  const rows = readCSV();
  const manifest = readManifest();
  const datasets = readDatasetRegistry();
  const runDetails = readRunDetails();

  const { datasetCounts, methodCounts, modelCounts } = buildViews(rows, datasets);

  // Top-level shards.
  writeJSON(path.join(OUT_DIR, "overview.json"), {
    run_count: manifest.run_count,
    row_count: manifest.row_count,
    content_hash: manifest.content_hash,
    dataset_count: Object.keys(datasets).length,
    method_count: methodCounts.size,
    model_count: modelCounts.size,
  });

  const datasetList: DatasetEntry[] = Object.values(datasets).map((d) => ({
    ...d,
    run_count: datasetCounts.get(d.id) ?? 0,
  }));
  datasetList.sort((a, b) => a.id.localeCompare(b.id));
  writeJSON(path.join(OUT_DIR, "datasets.json"), datasetList);

  const methodList = Array.from(methodCounts.entries())
    .map(([id, run_count]) => ({ id, run_count }))
    .sort((a, b) => a.id.localeCompare(b.id));
  writeJSON(path.join(OUT_DIR, "methods.json"), methodList);

  const modelList = Array.from(modelCounts.entries())
    .map(([id, run_count]) => ({ id, run_count }))
    .sort((a, b) => a.id.localeCompare(b.id));
  writeJSON(path.join(OUT_DIR, "models.json"), modelList);

  writeJSON(path.join(OUT_DIR, "runs.json"), runDetails);

  console.log(
    `build-data: ${rows.length} rows, ${Object.keys(runDetails).length} runs, ` +
      `${Object.keys(datasets).length} datasets, ${methodList.length} methods, ` +
      `${modelList.length} models`,
  );
}

main();
