# QueryGym Methods Reference

Complete reference guide for all query reformulation methods in QueryGym, including parameters, defaults, and usage examples.

## Table of Contents

- [Common Interface](#common-interface)
- [Method Parameters Overview](#method-parameters-overview)
- [Methods](#methods)
  - [GenQR](#genqr)
  - [GenQR Ensemble](#genqr-ensemble)
  - [Query2Doc](#query2doc)
  - [QA Expand](#qa-expand)
  - [MuGI](#mugi)
  - [LameR](#lamer)
  - [Query2E](#query2e)
  - [CSQE](#csqe)

---

## Common Interface

All methods inherit from `BaseReformulator` and provide the same interface:

```python
import querygym as qg

# Create reformulator
reformulator = qg.create_reformulator(
    method_name="method_name",
    model="your-model-name",
    params={...},  # Method-specific parameters
    llm_config={...}  # LLM configuration (temperature, max_tokens, base_url, api_key, etc.)
)

# Single query reformulation
result = reformulator.reformulate(qg.QueryItem("q1", "your query"))

# Batch reformulation
results = reformulator.reformulate_batch(queries)
```

---

## Method Parameters Overview

### LLM Configuration (`llm_config`)

All methods accept these LLM configuration parameters:

- `base_url` (str): LLM API endpoint URL (e.g., `"http://127.0.0.1:11434/v1"` for Ollama)
- `api_key` (str): API key for authentication (use `"ollama"` for Ollama, `"EMPTY"` for vLLM)
- `temperature` (float): Sampling temperature (default varies by method)
- `max_tokens` (int): Maximum tokens per generation (default varies by method)

### Method Parameters (`params`)

Each method has specific parameters documented below. Common parameters include:

- `retrieval_k` (int): Number of documents to retrieve for context-based methods (default: 10)
- `threads` (int): Number of threads for batch retrieval (default: 16)
- `searcher`: Pre-configured searcher instance (for methods requiring context)
- `searcher_type` (str): Type of searcher to create (`"pyserini"`, `"pyterrier"`, etc.)
- `searcher_kwargs` (dict): Keyword arguments for searcher initialization

---

## Methods

### GenQR

**Method Name:** `"genqr"`  
**Requires Context:** No  
**Description:** Generic keyword expansion using LLM. Generates reformulations N times and concatenates them.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_generations` | int | `5` | Number of times to generate reformulations |
| `temperature` | float | `0.8` | Sampling temperature (via `llm_config`) |
| `max_tokens` | int | `256` | Maximum tokens per generation (via `llm_config`) |

#### Usage Example

```python
import querygym as qg

# Basic usage
reformulator = qg.create_reformulator(
    "genqr",
    model="gpt-4",
    params={"n_generations": 5},
    llm_config={"temperature": 0.8, "max_tokens": 256}
)

# With custom LLM endpoint
reformulator = qg.create_reformulator(
    "genqr",
    model="qwen2.5:7b",
    params={"n_generations": 3},
    llm_config={
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "temperature": 0.7
    }
)

result = reformulator.reformulate(qg.QueryItem("q1", "neural networks"))
```

#### Output Format

- **Concatenation:** `query + reformulation1 + reformulation2 + ... + reformulationN`
- **Metadata:** Includes `n_generations` and list of all `reformulations`

---

### GenQR Ensemble

**Method Name:** `"genqr_ensemble"`  
**Requires Context:** No  
**Description:** Ensemble of 10 instruction variants to generate diverse keyword expansions. Each variant generates keywords independently, then all are merged.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repeat_query_weight` | int | `5` | Number of query repetitions in final output |
| `variant_ids` | list | `[all 10 variants]` | List of prompt IDs to use (advanced) |
| `parallel` | bool | `False` | Enable parallel generation of all variants |
| `temperature` | float | `0.92` | Sampling temperature (via `llm_config`) |
| `max_tokens` | int | `256` | Maximum tokens per generation (via `llm_config`) |

#### Usage Example

```python
import querygym as qg

# Basic usage
reformulator = qg.create_reformulator(
    "genqr_ensemble",
    model="gpt-4",
    params={"repeat_query_weight": 5}
)

# With parallel generation
reformulator = qg.create_reformulator(
    "genqr_ensemble",
    model="gpt-4",
    params={
        "repeat_query_weight": 3,
        "parallel": True
    },
    llm_config={"temperature": 0.92}
)

result = reformulator.reformulate(qg.QueryItem("q1", "machine learning"))
```

#### Output Format

- **Concatenation:** `(query × repeat_query_weight) + keyword1 + keyword2 + ... + keywordN`
- **Metadata:** Includes `num_variants`, `total_keywords`, `keywords` list, and per-variant outputs

---

### Query2Doc

**Method Name:** `"query2doc"`  
**Requires Context:** No  
**Description:** Generates pseudo-documents for the query using LLM knowledge. Supports zero-shot, chain-of-thought, and few-shot modes.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | `"zs"` | Mode: `"zs"` (zero-shot), `"cot"` (chain-of-thought), `"fs"`/`"fewshot"`/`"few-shot"` (few-shot) |
| `num_examples` | int | `4` | Number of few-shot examples (only for `mode="fs"`) |
| `dataset_type` | str | `None` | Dataset type for few-shot: `"msmarco"`, `"beir"`, or `"generic"` |
| `collection_path` | str | `None` | Path to collection file (for MS MARCO/generic) |
| `train_queries_path` | str | `None` | Path to training queries file |
| `train_qrels_path` | str | `None` | Path to training qrels file |
| `beir_data_dir` | str | `None` | Path to BEIR dataset directory (for `dataset_type="beir"`) |
| `train_split` | str | `"train"` | BEIR split to use: `"train"` or `"dev"` |
| `temperature` | float | `0.7` | Sampling temperature (via `llm_config`) |
| `max_tokens` | int | `256` | Maximum tokens per generation (via `llm_config`) |

**Environment Variables (Alternative to `params`):**

For few-shot mode, you can also use environment variables:
- `COLLECTION_PATH` or `MSMARCO_COLLECTION`
- `TRAIN_QUERIES_PATH` or `MSMARCO_TRAIN_QUERIES`
- `TRAIN_QRELS_PATH` or `MSMARCO_TRAIN_QRELS`
- `BEIR_DATA_DIR`

#### Usage Examples

```python
import querygym as qg

# Zero-shot mode (default)
reformulator = qg.create_reformulator(
    "query2doc",
    model="gpt-4",
    params={"mode": "zs"}
)

# Chain-of-thought mode
reformulator = qg.create_reformulator(
    "query2doc",
    model="gpt-4",
    params={"mode": "cot"}
)

# Few-shot mode with MS MARCO
reformulator = qg.create_reformulator(
    "query2doc",
    model="gpt-4",
    params={
        "mode": "fs",
        "num_examples": 4,
        "dataset_type": "msmarco",
        "collection_path": "path/to/collection.tsv",
        "train_queries_path": "path/to/queries.tsv",
        "train_qrels_path": "path/to/qrels.tsv"
    }
)

# Few-shot mode with BEIR
reformulator = qg.create_reformulator(
    "query2doc",
    model="gpt-4",
    params={
        "mode": "fewshot",
        "num_examples": 6,
        "dataset_type": "beir",
        "beir_data_dir": "path/to/beir/dataset",
        "train_split": "train"
    }
)

result = reformulator.reformulate(qg.QueryItem("q1", "what causes diabetes"))
```

#### Output Format

- **Concatenation:** Uses `query_repeat_plus_generated` strategy (default: query × 3 + generated content)
- **Metadata:** Includes `mode`, `prompt_id`, `pseudo_doc`, `num_examples` (for few-shot)

---

### QA Expand

**Method Name:** `"qa_expand"`  
**Requires Context:** No  
**Description:** Question-answer based expansion. Generates sub-questions, pseudo-answers, and refines them.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | int | `256` | Maximum tokens per generation (via `llm_config` or `params`) |
| `temperature_subq` | float | `0.8` | Temperature for sub-question generation |
| `temperature_answer` | float | `0.8` | Temperature for answer generation |
| `temperature_refine` | float | `0.8` | Temperature for answer refinement |
| `prompt_subq` | str | `"qa_expand.subq.v1"` | Prompt ID for sub-question generation (advanced) |
| `prompt_answer` | str | `"qa_expand.answer.v1"` | Prompt ID for answer generation (advanced) |
| `prompt_refine` | str | `"qa_expand.refine.v1"` | Prompt ID for refinement (advanced) |

#### Usage Example

```python
import querygym as qg

# Basic usage
reformulator = qg.create_reformulator(
    "qa_expand",
    model="gpt-4",
    llm_config={"temperature": 0.8, "max_tokens": 256}
)

# With custom temperatures per step
reformulator = qg.create_reformulator(
    "qa_expand",
    model="gpt-4",
    params={
        "temperature_subq": 0.7,
        "temperature_answer": 0.9,
        "temperature_refine": 0.6,
        "max_tokens": 512
    }
)

result = reformulator.reformulate(qg.QueryItem("q1", "how does photosynthesis work"))
```

#### Output Format

- **Concatenation:** `(query × 3) + refined_answers`
- **Metadata:** Includes `subquestions_raw`, `questions_json`, `answers_json`, `refined_answers_json`, `refined_text`, and `prompts_used`

---

### MuGI

**Method Name:** `"mugi"`  
**Requires Context:** No  
**Description:** Multi-granularity information expansion. Generates multiple diverse pseudo-documents per query with adaptive concatenation.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_docs` | int | `5` | Number of pseudo-documents to generate per query |
| `adaptive_times` | int | `6` | Divisor for adaptive repetition ratio |
| `max_tokens` | int | `1024` | Maximum tokens per pseudo-document |
| `temperature` | float | `1.0` | Sampling temperature for diversity |
| `mode` | str | `"zs"` | Mode: `"zs"` (zero-shot) or `"fs"`/`"fewshot"` (few-shot) |
| `prompt_id` | str | `None` | Direct prompt ID override (advanced) |
| `parallel` | bool | `False` | Generate all pseudo-docs in parallel |

#### Usage Example

```python
import querygym as qg

# Basic usage
reformulator = qg.create_reformulator(
    "mugi",
    model="gpt-4",
    params={
        "num_docs": 5,
        "adaptive_times": 6,
        "temperature": 1.0
    },
    llm_config={"max_tokens": 1024}
)

# With parallel generation
reformulator = qg.create_reformulator(
    "mugi",
    model="gpt-4",
    params={
        "num_docs": 3,
        "parallel": True,
        "mode": "zs"
    }
)

result = reformulator.reformulate(qg.QueryItem("q1", "artificial intelligence"))
```

#### Output Format

- **Concatenation:** Adaptive formula: `(query + ' ') * repetition_times + all_pseudo_docs`
  - `repetition_times = (len(all_pseudo_docs) // len(query)) // adaptive_times`
- **Metadata:** Includes `pseudo_docs`, `num_docs`, `adaptive_times`, `repetition_times`, `query_len`, `docs_len`, `mode`, `prompt_id`, `parallel`, and individual pseudo-docs

---

### LameR

**Method Name:** `"lamer"`  
**Requires Context:** Yes  
**Description:** Context-based passage synthesis using retrieved documents. Generates multiple passages from contexts and interleaves them with the query.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retrieval_k` | int | `10` | Number of documents to retrieve for context |
| `gen_passages` | int | `5` | Number of passages to generate |
| `threads` | int | `16` | Number of threads for batch retrieval |
| `searcher` | object | `None` | Pre-configured searcher instance (recommended) |
| `searcher_type` | str | `"pyserini"` | Type of searcher to create |
| `searcher_kwargs` | dict | `{}` | Keyword arguments for searcher initialization |
| `index` | str | `None` | Pyserini index name (legacy format) |
| `k1` | float | `None` | BM25 k1 parameter (legacy format) |
| `b` | float | `None` | BM25 b parameter (legacy format) |
| `temperature` | float | `1.0` | Sampling temperature (via `llm_config`) |
| `max_tokens` | int | `128` | Maximum tokens per generation (via `llm_config`) |

#### Usage Example

```python
import querygym as qg
from pyserini.search.lucene import LuceneSearcher

# Using wrapped Pyserini searcher (recommended)
pyserini_searcher = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")
pyserini_searcher.set_bm25(k1=0.82, b=0.68)
searcher = qg.wrap_pyserini_searcher(pyserini_searcher, answer_key="contents")

reformulator = qg.create_reformulator(
    "lamer",
    model="gpt-4",
    params={
        "searcher": searcher,
        "retrieval_k": 10,
        "gen_passages": 5,
        "threads": 16
    },
    llm_config={"temperature": 1.0, "max_tokens": 128}
)

# Or using searcher_type format
reformulator = qg.create_reformulator(
    "lamer",
    model="gpt-4",
    params={
        "searcher_type": "pyserini",
        "searcher_kwargs": {
            "index": "msmarco-v1-passage",
            "k1": 0.82,
            "b": 0.68
        },
        "retrieval_k": 10,
        "gen_passages": 5
    }
)

result = reformulator.reformulate(qg.QueryItem("q1", "machine learning"))
```

#### Output Format

- **Concatenation:** `q + passage1 + q + passage2 + q + passage3 + ...` (interleaved)
- **Metadata:** Includes `generated_passages`, `generated_passages_count`, `used_ctx`

---

### Query2E

**Method Name:** `"query2e"`  
**Requires Context:** No  
**Description:** Query to entity/keyword expansion. Generates keywords from the query, optionally using few-shot examples.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | `"zs"` | Mode: `"zs"`/`"zeroshot"` (zero-shot) or `"fs"`/`"fewshot"` (few-shot) |
| `num_examples` | int | `4` | Number of few-shot examples (only for `mode="fs"`) |
| `max_keywords` | int | `20` | Maximum number of keywords to extract |
| `dataset_type` | str | `None` | Dataset type for few-shot: `"msmarco"`, `"beir"`, or `"generic"` |
| `collection_path` | str | `None` | Path to collection file (for MS MARCO/generic) |
| `train_queries_path` | str | `None` | Path to training queries file |
| `train_qrels_path` | str | `None` | Path to training qrels file |
| `beir_data_dir` | str | `None` | Path to BEIR dataset directory (for `dataset_type="beir"`) |
| `train_split` | str | `"train"` | BEIR split to use: `"train"` or `"dev"` |
| `temperature` | float | `0.3` | Sampling temperature (via `llm_config`) |
| `max_tokens` | int | `256` | Maximum tokens per generation (via `llm_config`) |

**Environment Variables (Alternative to `params`):**

Same as Query2Doc for few-shot mode.

#### Usage Example

```python
import querygym as qg

# Zero-shot mode (default)
reformulator = qg.create_reformulator(
    "query2e",
    model="gpt-4",
    params={"mode": "zs", "max_keywords": 20}
)

# Few-shot mode
reformulator = qg.create_reformulator(
    "query2e",
    model="gpt-4",
    params={
        "mode": "fs",
        "num_examples": 4,
        "max_keywords": 15,
        "dataset_type": "msmarco",
        "collection_path": "path/to/collection.tsv",
        "train_queries_path": "path/to/queries.tsv",
        "train_qrels_path": "path/to/qrels.tsv"
    }
)

result = reformulator.reformulate(qg.QueryItem("q1", "deep learning"))
```

#### Output Format

- **Concatenation:** `(query × 5) + keywords`
- **Metadata:** Includes `mode`, `keywords` list, `prompt_id`, `num_examples` (for few-shot)

---

### CSQE

**Method Name:** `"csqe"`  
**Requires Context:** Yes  
**Description:** Context-based sentence-level query expansion. Combines KEQE (knowledge-based) and CSQE (context-based) expansions.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retrieval_k` | int | `10` | Number of documents to retrieve for context |
| `gen_num` | int | `2` | Number of expansions for both KEQE and CSQE (total: 2×gen_num) |
| `threads` | int | `16` | Number of threads for batch retrieval |
| `searcher` | object | `None` | Pre-configured searcher instance (recommended) |
| `searcher_type` | str | `"pyserini"` | Type of searcher to create |
| `searcher_kwargs` | dict | `{}` | Keyword arguments for searcher initialization |
| `index` | str | `None` | Pyserini index name (legacy format) |
| `k1` | float | `None` | BM25 k1 parameter (legacy format) |
| `b` | float | `None` | BM25 b parameter (legacy format) |
| `temperature` | float | `1.0` | Sampling temperature (via `llm_config`) |
| `max_tokens` | int | `1024` | Maximum tokens per generation (via `llm_config`) |

#### Usage Example

```python
import querygym as qg
from pyserini.search.lucene import LuceneSearcher

# Using wrapped Pyserini searcher (recommended)
pyserini_searcher = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")
pyserini_searcher.set_bm25(k1=0.82, b=0.68)
searcher = qg.wrap_pyserini_searcher(pyserini_searcher, answer_key="contents")

reformulator = qg.create_reformulator(
    "csqe",
    model="gpt-4",
    params={
        "searcher": searcher,
        "retrieval_k": 10,
        "gen_num": 2
    },
    llm_config={"temperature": 1.0, "max_tokens": 1024}
)

# Or using searcher_type format
reformulator = qg.create_reformulator(
    "csqe",
    model="gpt-4",
    params={
        "searcher_type": "pyserini",
        "searcher_kwargs": {
            "index": "msmarco-v1-passage"
        },
        "retrieval_k": 10,
        "gen_num": 2
    }
)

result = reformulator.reformulate(qg.QueryItem("q1", "quantum computing"))
```

#### Output Format

- **Concatenation:** `(query × gen_num) + keqe_passages + csqe_sentences` (lowercased, space-separated)
- **Metadata:** Includes `keqe_passages`, `csqe_responses`, `csqe_sentences`, `gen_num`, `total_generations`, `used_ctx`

---

## Quick Reference Table

| Method | Requires Context | Key Parameters | Default LLM Config |
|--------|-----------------|----------------|-------------------|
| **GenQR** | No | `n_generations` | temp=0.8, max_tokens=256 |
| **GenQR Ensemble** | No | `repeat_query_weight`, `parallel` | temp=0.92, max_tokens=256 |
| **Query2Doc** | No | `mode`, `num_examples` (fs) | temp=0.7, max_tokens=256 |
| **QA Expand** | No | `temperature_subq/answer/refine` | temp=0.8, max_tokens=256 |
| **MuGI** | No | `num_docs`, `adaptive_times`, `parallel` | temp=1.0, max_tokens=1024 |
| **LameR** | Yes | `retrieval_k`, `gen_passages`, `searcher` | temp=1.0, max_tokens=128 |
| **Query2E** | No | `mode`, `max_keywords`, `num_examples` (fs) | temp=0.3, max_tokens=256 |
| **CSQE** | Yes | `retrieval_k`, `gen_num`, `searcher` | temp=1.0, max_tokens=1024 |

---

## Tips and Best Practices

1. **Context-Based Methods (LameR, CSQE):**
   - Always provide a `searcher` instance or configure `searcher_type`/`searcher_kwargs`
   - Use `qg.wrap_pyserini_searcher()` for easy integration with Pyserini
   - Set appropriate `retrieval_k` based on your needs (default: 10)

2. **Few-Shot Methods (Query2Doc, Query2E):**
   - Ensure training data paths are correct
   - Use environment variables for cleaner configuration
   - Start with `num_examples=4` and adjust based on results

3. **LLM Configuration:**
   - For local LLMs (Ollama, vLLM), set `base_url` and `api_key` in `llm_config`
   - Adjust `temperature` based on desired diversity (higher = more diverse)
   - Set `max_tokens` based on expected output length

4. **Performance:**
   - Enable `parallel=True` for GenQR Ensemble and MuGI when using multiple generations
   - Use appropriate `threads` for batch retrieval in context-based methods

---

## See Also

- [API Reference](../api/methods.md) - Technical API documentation
- [Query Reformulation Guide](reformulation.md) - Usage tutorials
- [Examples](https://github.com/ls3-lab/QueryGym/tree/main/examples) - Complete workflow examples

