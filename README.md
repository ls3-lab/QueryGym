[![Publish to PyPI](https://github.com/radinhamidi/QueryGym/actions/workflows/publish.yml/badge.svg)](https://github.com/radinhamidi/QueryGym/actions/workflows/publish.yml)
[![PyPI version](https://badge.fury.io/py/querygym.svg)](https://pypi.org/project/querygym/)
![PyPI - Downloads](https://img.shields.io/pypi/dw/querygym?color=blueviolet&label=downloads)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">
  <img src="docs/querygym-logo.png" alt="QueryGym Logo" width="600">
</div>

# QueryGym

A lightweight, reproducible toolkit for **LLM-based query reformulation**.

- Single **Prompt Bank** (YAML) with metadata.
- **Simple DataLoader**: Dependency-free file loading for queries, qrels, and contexts.
- **Format Loaders**: Optional BEIR and MS MARCO format loaders in `querygym.loaders`.
- **OpenAI-compatible** LLM client (works with any OpenAI API–compatible endpoint).
- **Pyserini** optional: either pass contexts (JSONL) or pass a retriever instance to build contexts.
- Export-only: emits reformulated queries; optionally generates a **bash** script for Pyserini + `trec_eval`.

## Installation

### Option 1: Install from PyPI
```bash
pip install querygym
```

### Option 2: Use Docker (Recommended for Quick Start)
```bash
# Pull and run pre-built image
docker pull ghcr.io/radinhamidi/querygym:latest
docker run -it --gpus all ghcr.io/radinhamidi/querygym:latest

# Or use docker-compose
docker-compose run --rm querygym
```

📖 **Docker Setup:** See [DOCKER_SETUP.md](DOCKER_SETUP.md) for quick start or the [full Docker guide](https://querygym.readthedocs.io/en/latest/user-guide/docker/) for detailed usage.

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

## Examples

See the [examples/](examples/) directory for:
- **[Code snippets](examples/snippets/)** - Quick reference examples
- **[Docker examples](examples/docker/)** - Containerized workflows with Jupyter notebooks
- **[QueryGym + Pyserini](examples/querygym_pyserini/)** - Complete retrieval pipelines

Check [examples/README.md](examples/README.md) for the full guide.
