[![Publish to PyPI](https://github.com/radinhamidi/queryGym/actions/workflows/publish.yml/badge.svg)](https://github.com/radinhamidi/queryGym/actions/workflows/publish.yml)

# querygym

A lightweight, reproducible toolkit for **LLM-based query reformulation**.

- Single **Prompt Bank** (YAML) with metadata.
- **Simple DataLoader**: Dependency-free file loading for queries, qrels, and contexts.
- **Format Loaders**: Optional BEIR and MS MARCO format loaders in `querygym.loaders`.
- **OpenAI-compatible** LLM client (works with any OpenAI API–compatible endpoint).
- **Pyserini** optional: either pass contexts (JSONL) or pass a retriever instance to build contexts.
- Export-only: emits reformulated queries; optionally generates a **bash** script for Pyserini + `trec_eval`.

## Quickstart

### Python API (Recommended)
```python
import querygym as qg

# Load data
queries = qg.load_queries("queries.tsv")
qrels = qg.load_qrels("qrels.txt")
contexts = qg.load_contexts("contexts.jsonl")

# Create reformulator
reformulator = qg.create_reformulator("genqr_ensemble", model="gpt-4")

# Reformulate
results = reformulator.reformulate_batch(queries)

# Save
qg.DataLoader.save_queries(
    [qg.QueryItem(r.qid, r.reformulated) for r in results],
    "reformulated.tsv"
)
```

### CLI
```bash
pip install -e .[hf,beir,dev]
export OPENAI_API_KEY=sk-...

# Run a method (e.g., genqr_ensemble)
querygym run --method genqr_ensemble \
  --queries-tsv queries.tsv \
  --output-tsv reformulated.tsv \
  --cfg-path querygym/config/defaults.yaml
```

### Loading Datasets

**BEIR:**
```python
import querygym as qg

# Download with BEIR library
from beir.datasets.data_loader import GenericDataLoader
data_path = GenericDataLoader("nfcorpus").download_and_unzip()

# Load with querygym
queries = qg.loaders.beir.load_queries(data_path)
qrels = qg.loaders.beir.load_qrels(data_path)
```

**MS MARCO:**
```python
import querygym as qg

# Load from local files (download with ir_datasets)
queries = qg.loaders.msmarco.load_queries("queries.tsv")
qrels = qg.loaders.msmarco.load_qrels("qrels.tsv")
```

See [example scripts](scripts/README.md) for complete workflows.
