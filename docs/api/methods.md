# Methods API Reference

All methods inherit from `BaseReformulator` and provide the same interface.

## Common Interface

All methods support:

```python
# Single query reformulation
result = method.reformulate(query, contexts=None)

# Batch reformulation
results = method.reformulate_batch(queries, contexts=None)
```

## Available Methods

### GenQR

Generic keyword expansion using LLM.

```python
reformulator = qg.create_reformulator("genqr", model="gpt-4")
```

**Requires Context:** No

### GenQR Ensemble

Ensemble of multiple keyword expansion prompts.

```python
reformulator = qg.create_reformulator(
    "genqr_ensemble",
    model="gpt-4",
    params={"repeat_query_weight": 3}
)
```

**Requires Context:** No  
**Parameters:**
- `repeat_query_weight` (int): Number of query repetitions (default: 3)

### Query2Doc

Generates pseudo-documents for the query.

```python
reformulator = qg.create_reformulator("query2doc", model="gpt-4")
```

**Requires Context:** No

### QA Expand

Question-answer based expansion.

```python
reformulator = qg.create_reformulator("qa_expand", model="gpt-4")
```

**Requires Context:** No

### MuGI

Multi-granularity information expansion.

```python
reformulator = qg.create_reformulator("mugi", model="gpt-4")
```

**Requires Context:** No

### LameR

Context-based passage synthesis.

```python
reformulator = qg.create_reformulator("lamer", model="gpt-4")
```

**Requires Context:** Yes

### Query2E

Query to entity expansion.

```python
reformulator = qg.create_reformulator("query2e", model="gpt-4")
```

**Requires Context:** No

### CSQE

Context-based sentence extraction.

```python
reformulator = qg.create_reformulator("csqe", model="gpt-4")
```

**Requires Context:** Yes
