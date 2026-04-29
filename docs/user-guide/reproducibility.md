# Reproducibility & Leaderboard Submissions

QueryGym ships with a reproducibility pipeline that powers `leaderboard.querygym.com` and the SIGIR 2026 reproducibility paper. This page explains how to submit a result.

The full schema lives at `reproducibility/schema.md` (human-readable) and `reproducibility/schema.json` (machine-readable). All submitted JSONs are validated against it three times: at emit time, at submit time, and at aggregate time in CI.

## Trusted contributor flow

If you have commit access:

```bash
# 1. Run the example pipeline.
python examples/querygym_pyserini/pipeline.py \
    --dataset msmarco-v1-passage.trecdl2019 \
    --method query2e \
    --model gpt-4.1-mini \
    --output-dir outputs/dl19_query2e_zs

# 2. Copy the output into the canonical layout.
python -m reproducibility.scripts.submit_run --from-dir outputs/dl19_query2e_zs

# 3. Regenerate the aggregate CSV + manifest.
make repro-aggregate

# 4. Commit and open a PR.
git add reproducibility/data/
git commit -m "add query2e/gpt-4.1-mini result on dl19-passage"
git push
gh pr create
```

CI runs the schema/validator tests and `aggregate_runs.py --check`. If everything is green, the leaderboard rebuilds on merge.

### Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `aggregator --check failed: results.csv is out of date` | You forgot step 3. | Run `make repro-aggregate`, commit the diff. |
| `dataset_id 'foo' not in dataset_registry.yaml` | Typo or new dataset not registered. | Add the dataset to `dataset_registry.yaml` first, then re-submit. |
| `method_id 'foo' not in registered methods` | Method not registered or name typo'd. | Register via `@register_method("foo")` in `querygym/methods/`. |
| `params_hash mismatch` | The JSON was hand-edited. | Don't hand-edit run JSONs — re-run the emitter or use `submit_run` instead. |
| `metric(s) ['bleu'] not in eval_metrics for dataset 'X'` | Unsupported metric for that dataset. | Either drop the metric or add it to the dataset's `output.eval_metrics` in the registry. |

## External (fork) contributor flow

If you don't have commit access:

1. Fork `ls3-lab/QueryGym` on GitHub and clone your fork.
2. Run steps 1–3 from the trusted flow above.
3. Push to your fork and open a PR against `ls3-lab/QueryGym:main`.

CI runs the same schema/validator/aggregator checks against your PR — no LLM keys or Pyserini are needed for these checks, so fork PRs get fast feedback.

A maintainer will additionally **re-verify your numbers locally** before merging:

- **Cheap pre-check (~30s):** the maintainer runs `pytrec_eval` against your submitted `run.txt` using the dataset's qrels and confirms the reported metrics match.
- **Full re-run (only if needed):** if the cheap check is suspicious, the maintainer runs the example pipeline with your `config` block as inputs and compares reformulated queries + run file.

This is why every submission must include `run.txt` and `reformulated_queries.tsv` alongside the JSON — they make verification cheap.

## Verifying a published number (paper readers)

Each leaderboard row links to the canonical files at a paper-release tag. To verify independently:

```bash
git clone --depth=1 --branch=paper-sigir2026 https://github.com/ls3-lab/QueryGym.git
cd QueryGym

# Pick a run.
RUN_DIR=reproducibility/data/runs/msmarco-v1-passage.trecdl2019/query2e/gpt-4.1-mini

# Re-run trec_eval against the public qrels (Pyserini ships them).
python -m pyserini.eval.trec_eval -m ndcg_cut.10 dl19-passage "${RUN_DIR}"/*.run.txt
```

The number from `pyserini.eval.trec_eval` should match `metrics.ndcg_cut_10` in the corresponding JSON.

## External tools (dashboard, third parties)

The contract is `reproducibility/schema.json` — a Draft 2020-12 JSON Schema document. Any tool that emits a conformant JSON can submit (subject to the trusted vs. fork flows above). You don't need to import any Python from QueryGym; just read the schema file and validate locally with whatever JSON Schema library your stack provides (`Ajv` for JS, `jsonschema` for Python, `everit-org/json-schema` for Java).

`schema_version` is `"const": 1` today. Bumping it to 2 will be a breaking change announced ahead of time.
