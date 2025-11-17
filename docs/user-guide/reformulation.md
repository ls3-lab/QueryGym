# Query Reformulation Methods

querygym provides several state-of-the-art query reformulation methods.

## Available Methods

### GenQR (Generic Query Reformulation)

Simple keyword expansion using LLM.

```python
import querygym as qg

reformulator = qg.create_reformulator("genqr", model="gpt-4")
result = reformulator.reformulate(qg.QueryItem("q1", "neural networks"))
```

### GenQR Ensemble

Ensemble of multiple keyword expansion prompts for better coverage.

```python
reformulator = qg.create_reformulator(
    "genqr_ensemble",
    model="gpt-4",
    params={"repeat_query_weight": 3}
)
```

**Parameters:**
- `repeat_query_weight` (int): Number of times to repeat original query (default: 3)

### Query2Doc

Generates pseudo-documents relevant to the query.

```python
reformulator = qg.create_reformulator("query2doc", model="gpt-4")
```

Supports both zero-shot and chain-of-thought variants.

### QA Expand

Decomposes query into sub-questions, generates answers, and refines.

```python
reformulator = qg.create_reformulator("qa_expand", model="gpt-4")
```

### MuGI

Multi-granularity information expansion.

```python
reformulator = qg.create_reformulator("mugi", model="gpt-4")
```

### LameR

Context-based passage synthesis using retrieved documents.

```python
# Load contexts
contexts = qg.load_contexts("contexts.jsonl")

# Create reformulator
reformulator = qg.create_reformulator("lamer", model="gpt-4")

# Reformulate with contexts
results = reformulator.reformulate_batch(queries, contexts=contexts)
```

**Note:** LameR requires contexts from initial retrieval.

### Query2E

Query to entity expansion.

```python
reformulator = qg.create_reformulator("query2e", model="gpt-4")
```

### CSQE

Context-based sentence extraction from retrieved documents.

```python
# Requires contexts
contexts = qg.load_contexts("contexts.jsonl")

reformulator = qg.create_reformulator("csqe", model="gpt-4")
results = reformulator.reformulate_batch(queries, contexts=contexts)
```

## Method Comparison

| Method | Requires Context | Type | Best For |
|--------|-----------------|------|----------|
| genqr | No | Keyword expansion | General queries |
| genqr_ensemble | No | Keyword expansion | Robust expansion |
| query2doc | No | Pseudo-document | Dense retrieval |
| qa_expand | No | QA-based | Complex queries |
| mugi | No | Multi-granular | Diverse expansion |
| lamer | Yes | Context synthesis | Re-ranking |
| query2e | No | Entity expansion | Entity queries |
| csqe | Yes | Sentence extraction | Precision-focused |

## Custom Parameters

All methods support custom parameters:

```python
reformulator = qg.create_reformulator(
    "genqr_ensemble",
    model="gpt-4",
    params={
        "repeat_query_weight": 5,
        "temperature": 0.7
    },
    llm_config={
        "temperature": 0.8,
        "max_tokens": 512
    }
)
```

## Batch Processing

Process multiple queries efficiently:

```python
queries = qg.load_queries("queries.tsv")
reformulator = qg.create_reformulator("genqr", model="gpt-4")

# Batch reformulation with progress bar
results = reformulator.reformulate_batch(queries)
```

## Using Custom Prompts

See [Prompt Bank](prompts.md) for details on customizing prompts.

## Next Steps

- [Prompt Bank Documentation](prompts.md)
- [API Reference](../api/methods.md)
