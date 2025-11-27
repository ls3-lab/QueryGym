# Installation

## Requirements

- Python 3.9 or higher
- pip package manager

## Basic Installation

Install querygym from PyPI:

```bash
pip install querygym
```

This installs the core package with minimal dependencies:
- `typer` - CLI interface
- `pyyaml` - Prompt bank parsing
- `openai` - LLM client
- `tqdm` - Progress bars

## Optional Dependencies

### HuggingFace Datasets

For loading datasets from HuggingFace:

```bash
pip install querygym[hf]
```

### BEIR Format Support

For working with BEIR benchmark datasets:

```bash
pip install querygym[beir]
```

### Pyserini Integration

For using Pyserini search adapters:

```bash
pip install querygym[pyserini]
```

### All Optional Features

Install everything:

```bash
pip install querygym[all]
```

### Development Installation

For contributing to querygym:

```bash
# Clone the repository
git clone https://github.com/ls3-lab/QueryGym.git
cd QueryGym

# Install in editable mode with dev dependencies
pip install -e .[dev]
```

This includes:
- Testing tools (pytest, pytest-cov, pytest-mock)
- Code quality tools (black, ruff, mypy)
- Documentation tools (mkdocs, mkdocs-material, mkdocstrings)

## Verify Installation

Check that querygym is installed correctly:

```bash
# Check version
pip show querygym

# Test import
python -c "import querygym; print(querygym.__version__)"

# Test CLI
querygym --help
```

## Environment Setup

### API Keys

querygym uses OpenAI-compatible APIs. Set your API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

For custom endpoints:

```bash
export OPENAI_BASE_URL="https://your-custom-endpoint.com/v1"
```

### Configuration File

Create a `.env` file in your project:

```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional
```

## Troubleshooting

### Import Errors

If you get import errors for optional dependencies:

```bash
# Install the specific extra you need
pip install querygym[beir]
```

### Version Conflicts

If you have dependency conflicts:

```bash
# Create a fresh virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install querygym
```

### Development Installation Issues

If editable install fails:

```bash
# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Try again
pip install -e .[dev]
```

## Next Steps

- [Quick Start Tutorial](quickstart.md)
- [CLI Usage Guide](../user-guide/cli.md)
- [API Reference](../api/core.md)
