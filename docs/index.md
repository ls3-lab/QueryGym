<div align="center">
  <img src="querygym-logo.png" alt="QueryGym Logo" width="600">
</div>

# QueryGym

A lightweight, reproducible toolkit for **LLM-based query reformulation**.


[![Publish to PyPI](https://github.com/ls3-lab/QueryGym/actions/workflows/publish.yml/badge.svg)](https://github.com/ls3-lab/QueryGym/actions/workflows/publish.yml)
[![Build and Push Docker Images](https://github.com/ls3-lab/QueryGym/actions/workflows/docker.yml/badge.svg)](https://github.com/ls3-lab/QueryGym/actions/workflows/docker.yml)
[![PyPI version](https://badge.fury.io/py/querygym.svg)](https://pypi.org/project/querygym/)
![PyPI - Downloads](https://img.shields.io/pypi/dw/querygym?color=blueviolet&label=downloads)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Features

- **Single Prompt Bank** (YAML) with metadata
- **Simple DataLoader**: Dependency-free file loading for queries, qrels, and contexts
- **Format Loaders**: Optional BEIR and MS MARCO format loaders
- **OpenAI-compatible** LLM client (works with any OpenAI API–compatible endpoint)
- **Pyserini** optional: either pass contexts (JSONL) or pass a retriever instance to build contexts
- Export-only: emits reformulated queries; optionally generates a **bash** script for Pyserini + `trec_eval`

## Quick Example

```python
import querygym as qg

# Load data
queries = qg.load_queries("queries.tsv")
qrels = qg.load_qrels("qrels.txt")

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

## Installation

### Install from PyPI
```bash
pip install querygym
```

### Use Docker (Quick Start)
```bash
# Pull pre-built image
docker pull ghcr.io/ls3-lab/querygym:latest

# Run with Docker Compose
docker compose run --rm querygym
```

See the [Docker Guide](user-guide/docker.md) for detailed setup and usage.

For optional features:

```bash
# With HuggingFace datasets support
pip install querygym[hf]

# With BEIR format support
pip install querygym[beir]

# With Pyserini adapter
pip install querygym[pyserini]

# All optional features
pip install querygym[all]

# Development dependencies
pip install querygym[dev]
```

## Documentation

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [CLI Usage](user-guide/cli.md)
- [API Reference](api/core.md)

> 📊 **Looking for benchmarks?** Visit the [Leaderboard](https://ls3-lab.github.io/QueryGym/leaderboard.html) on our main site.

## Citation

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

## License

Apache License 2.0 - see [LICENSE](about/license.md) for details.
