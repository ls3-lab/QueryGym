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

### ThinkQE

Multi-round query expansion with retrieved passage feedback. Each round uses the original query plus newly retrieved passages to generate pseudo-passages via a reasoning LLM, appends them to the retrieval query, and retrieves again.

```python
reformulator = qg.create_reformulator(
    "thinkqe",
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    params={
        "searcher": searcher,
        "num_interaction": 3,
        "keep_passage_num": 5,
        "gen_num": 2,
        "accumulate": True,
        "use_passage_filter": True,
        "search_k": 1000,
    },
    llm_config={"temperature": 0.7, "max_tokens": 32768}
)
```

**Requires Context:** Yes  
**Parameters:**
- `keep_passage_num` (int): Passages kept for prompting per round (default: 5)
- `gen_num` (int): Expansions generated per round (default: 2)
- `num_interaction` (int): Expansion rounds after baseline retrieval (default: 3)
- `accumulate` (bool): Accumulate all round expansions into later rounds (default: True)
- `use_passage_filter` (bool): Blacklist passages repeated from two rounds ago (default: True)
- `repeat_weight` (float): Divisor for adaptive query repetition heuristic (default: 3)
- `search_k` (int): Retrieval depth per round; use 1000 to mirror original paper runs (default: keep_passage_num)
- `max_demo_len` (int): Optional word truncation for each passage before prompting (default: None)
- `no_thinking` (bool): Prefill `</think>` to disable reasoning traces (default: False)
- `searcher`: Pre-configured searcher instance (recommended)
