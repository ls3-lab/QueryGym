# Quick Start

This guide will help you get started with querygym in minutes.

## Installation

```bash
pip install querygym
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Basic Usage

### 1. Load Queries

```python
import querygym as qg

# Load from TSV file (qid<TAB>query format)
queries = qg.load_queries("queries.tsv")

# Or load from JSONL
queries = qg.load_queries("queries.jsonl", format="jsonl")
```

### 2. Create a Reformulator

```python
# Create a reformulator with a specific method
reformulator = qg.create_reformulator(
    method_name="genqr_ensemble",
    model="gpt-4",
    params={"repeat_query_weight": 3}
)
```

Available methods:
- `genqr` - Generic keyword expansion
- `genqr_ensemble` - Ensemble of keyword expansion prompts
- `query2doc` - Pseudo-document generation
- `qa_expand` - Question-answer expansion
- `mugi` - MuGI-style passage generation
- `lamer` - Context-based passage synthesis
- `query2e` - Query to entity expansion
- `csqe` - Context-based sentence extraction

### 3. Reformulate Queries

```python
# Reformulate a single query
query = qg.QueryItem("q1", "what causes diabetes")
result = reformulator.reformulate(query)

print(f"Original: {result.original}")
print(f"Reformulated: {result.reformulated}")

# Reformulate multiple queries
results = reformulator.reformulate_batch(queries)
```

### 4. Save Results

```python
# Save reformulated queries
qg.DataLoader.save_queries(
    [qg.QueryItem(r.qid, r.reformulated) for r in results],
    "reformulated.tsv"
)
```

## Complete Example

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

print(f"Reformulated {len(results)} queries")
```

## Using the CLI

querygym also provides a command-line interface:

```bash
# Run query reformulation
querygym run \
  --method genqr_ensemble \
  --queries-tsv queries.tsv \
  --output-tsv reformulated.tsv \
  --cfg-path querygym/config/defaults.yaml
```

See [CLI Usage](../user-guide/cli.md) for more details.

## Loading Datasets

### BEIR Format

```python
import querygym as qg

# Assuming you've downloaded a BEIR dataset
queries = qg.loaders.beir.load_queries("./data/nfcorpus")
qrels = qg.loaders.beir.load_qrels("./data/nfcorpus", split="test")
```

### MS MARCO Format

```python
import querygym as qg

queries = qg.loaders.msmarco.load_queries("./data/queries.tsv")
qrels = qg.loaders.msmarco.load_qrels("./data/qrels.tsv")
```

See [Loading Datasets](../user-guide/datasets.md) for more details.

## Context-Based Reformulation

Some methods (like `lamer`, `csqe`) use retrieved contexts:

```python
import querygym as qg

# Load queries and contexts
queries = qg.load_queries("queries.tsv")
contexts = qg.load_contexts("contexts.jsonl")

# Create context-based reformulator
reformulator = qg.create_reformulator("lamer", model="gpt-4")

# Reformulate with contexts
results = reformulator.reformulate_batch(queries, contexts=contexts)
```

## Using Custom LLM Endpoints

querygym works with any OpenAI-compatible API:

```python
import os

# Set custom endpoint
os.environ["OPENAI_BASE_URL"] = "https://your-endpoint.com/v1"
os.environ["OPENAI_API_KEY"] = "your-key"

# Use as normal
reformulator = qg.create_reformulator("genqr", model="your-model")
```

## Next Steps

- [CLI Usage Guide](../user-guide/cli.md)
- [Dataset Loading](../user-guide/datasets.md)
- [Query Reformulation Methods](../user-guide/reformulation.md)
- [API Reference](../api/core.md)
