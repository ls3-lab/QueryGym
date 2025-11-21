# querygym

A lightweight, reproducible toolkit for **LLM-based query reformulation**.

[![PyPI version](https://badge.fury.io/py/querygym.svg)](https://pypi.org/project/querygym/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
docker pull ghcr.io/radinhamidi/querygym:latest

# Run with docker-compose
docker-compose run --rm querygym
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

## Citation

If you use querygym in your research, please cite:

```bibtex
@software{querygym2025,
  title = {querygym: LLM-based Query Reformulation Toolkit},
  author = {Bigdeli, Amin and Hamidi Rad, Radin and Incesu, Mert and Arabzadeh, Negar and Clarke, Charles L. A. and Bagheri, Ebrahim},
  year = {2025},
  url = {https://github.com/radinhamidi/QueryGym}
}
```

## License

MIT License - see [LICENSE](about/license.md) for details.
