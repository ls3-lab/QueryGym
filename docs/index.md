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

> 📊 **Looking for benchmarks?** Visit the [Leaderboard](https://leaderboard.querygym.com).

## Citation

If you use QueryGym in your research, please cite the toolkit paper:

```bibtex
@inproceedings{DBLP:conf/www/BigdeliRIACB26,
  author={Amin Bigdeli and Radin {Hamidi Rad} and Mert Incesu and Negar Arabzadeh and Charles L. A. Clarke and Ebrahim Bagheri},
  title={QueryGym: {A} Toolkit for Reproducible LLM-Based Query Reformulation},
  booktitle={Companion Proceedings of the ACM Web Conference 2026, WWW Companion 2026, Dubai, United Arab Emirates, 29 June 2026 - 3 July 2026},
  pages={196--199},
  publisher={ACM},
  year={2026},
  url={https://doi.org/10.1145/3774905.3793135},
  doi={10.1145/3774905.3793135},
}
```

If you refer to the SIGIR 2026 reproducibility results or the leaderboard, please also cite:

```bibtex
@inproceedings{DBLP:conf/sigir/BigdeliRLIACB26,
  author={Amin Bigdeli and Radin {Hamidi Rad} and Hai Son Le and Mert Incesu and Negar Arabzadeh and Charles L. A. Clarke and Ebrahim Bagheri},
  title={A Reproducibility Study of LLM-Based Query Reformulation},
  booktitle={Proceedings of the 49th International {ACM} {SIGIR} Conference on Research and Development in Information Retrieval, {SIGIR} 2026, Melbourne, VIC, Australia, July 20-24, 2026},
  pages={2824--2834},
  publisher={ACM},
  year={2026},
  url={https://doi.org/10.1145/3805712.3808560},
  doi={10.1145/3805712.3808560},
}
```

## License

Apache License 2.0 - see [LICENSE](about/license.md) for details.
