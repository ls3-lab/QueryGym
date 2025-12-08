[![Publish to PyPI](https://github.com/ls3-lab/QueryGym/actions/workflows/publish.yml/badge.svg)](https://github.com/ls3-lab/QueryGym/actions/workflows/publish.yml)
[![Build and Push Docker Images](https://github.com/ls3-lab/QueryGym/actions/workflows/docker.yml/badge.svg)](https://github.com/ls3-lab/QueryGym/actions/workflows/docker.yml)
[![PyPI version](https://badge.fury.io/py/querygym.svg)](https://pypi.org/project/querygym/)
![PyPI - Downloads](https://img.shields.io/pypi/dw/querygym?color=blueviolet&label=downloads)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


<div align="center">
  <img src="https://raw.githubusercontent.com/ls3-lab/QueryGym/main/docs/querygym-logo.png" alt="QueryGym Logo" width="600">
</div>

<p align="center">
  <strong>A lightweight, reproducible toolkit for LLM-based query reformulation</strong>
</p>

<p align="center">
  <a href="https://querygym.readthedocs.io/">📚 Documentation</a> •
  <a href="https://ls3-lab.github.io/QueryGym/leaderboard.html">📊 Leaderboard</a> •
  <a href="https://pypi.org/project/querygym/">📦 PyPI</a> •
  <a href="https://arxiv.org/abs/2511.15996">📄 Paper</a>
</p>

---

## Features

- Single **Prompt Bank** (YAML) with metadata
- **Simple DataLoader**: Dependency-free file loading for queries, qrels, and contexts
- **Format Loaders**: Optional BEIR and MS MARCO format loaders in `querygym.loaders`
- **OpenAI-compatible** LLM client (works with any OpenAI API–compatible endpoint)
- **Pyserini** optional: either pass contexts (JSONL) or pass a retriever instance to build contexts
- Export-only: emits reformulated queries; optionally generates a **bash** script for Pyserini + `trec_eval`

## Installation

### Option 1: Install from PyPI
```bash
pip install querygym
```

### Option 2: Use Docker (Recommended for Quick Start)
```bash
# GPU version (default)
docker pull ghcr.io/ls3-lab/querygym:latest
docker run -it --gpus all ghcr.io/ls3-lab/querygym:latest

# CPU version (lightweight)
docker pull ghcr.io/ls3-lab/querygym:cpu
docker run -it ghcr.io/ls3-lab/querygym:cpu

# Or use Docker Compose
docker compose run --rm querygym
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

See the [examples](examples/) directory for:
- **[Code snippets](examples/snippets/)** - Quick reference examples
- **[Docker examples](examples/docker/)** - Containerized workflows with Jupyter notebooks
- **[QueryGym + Pyserini](examples/querygym_pyserini/)** - Complete retrieval pipelines
- **[Methods Reference](docs/user-guide/methods-reference.md)** - Complete guide to all query reformulation methods

Check [examples/README.md](examples/README.md) for the full guide.

## Contributing

We welcome contributions! Here's how you can help:

### Adding a New Prompt
1. Edit `querygym/prompt_bank.yaml`
2. Add an entry with fields: `id`, `method_family`, `version`, `introduced_by`, `license`, `authors`, `tags`, `template:{system,user}`, `notes`

### Adding a New Method
1. Create a class under `querygym/methods/*.py`
2. Subclass `BaseReformulator`, annotate `VERSION`, and register with `@register_method("name")`
3. Pull templates via `PromptBank.render(prompt_id, query=...)`

### Reporting Issues
- Found a bug? [Open an issue](https://github.com/ls3-lab/QueryGym/issues)
- Have a feature request? We'd love to hear it!

For detailed development guidelines, see the [Contributing Guide](https://querygym.readthedocs.io/en/latest/development/contributing/) in our documentation.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

### Citation

If you use QueryGym in your research, please cite:

```bibtex
@misc{bigdeli2025querygymtoolkitreproduciblellmbased,
      title={QueryGym: A Toolkit for Reproducible LLM-Based Query Reformulation}, 
      author={Amin Bigdeli and Radin Hamidi Rad and Mert Incesu and Negar Arabzadeh and Charles L. A. Clarke and Ebrahim Bagheri},
      year={2025},
      eprint={2511.15996},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2511.15996}, 
}
```
