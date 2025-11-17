# Datasets

Use **UnifiedQuerySource** to load Local / MS MARCO / BEIR.

- Local TSV: `qid<TAB>query`
- Local JSONL: `{"qid": "...", "query": "..."}`

MS MARCO:
- `backend=msmarco`, `source=file`, `--msmarco-queries-tsv ...`
- or `source=hf`, `--hf-name ms_marco`, `--hf-config passage|document`, `--split dev|train|...`

BEIR:
- `backend=beir`, `source=beir`, `--beir-root /path/to/fiqa` (expects BEIR layout)
- or `source=hf`, `--hf-name beir/fiqa`, `--split test`
