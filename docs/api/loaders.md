# Data Loaders API Reference

## Overview

querygym provides loaders for common IR dataset formats. All loaders return standardized `QueryItem` objects and dictionaries for easy integration.

## Core DataLoader

The `DataLoader` class provides format-agnostic loading utilities.

### load_queries

Load queries from TSV or JSONL files.

```python
from querygym.data.dataloader import DataLoader

# Load from TSV (qid<TAB>text format)
queries = DataLoader.load_queries("queries.tsv")

# Load from JSONL with custom keys
queries = DataLoader.load_queries(
    "queries.jsonl",
    format="jsonl",
    qid_key="id",
    query_key="text"
)
```

**Parameters:**
- `path` (str): Path to queries file
- `format` (str): File format - "tsv" or "jsonl" (default: "tsv")
- `qid_col` (int): Column index for query ID in TSV (default: 0)
- `query_col` (int): Column index for query text in TSV (default: 1)
- `qid_key` (str): JSON key for query ID in JSONL (default: "qid")
- `query_key` (str): JSON key for query text in JSONL (default: "query")

**Returns:** `List[QueryItem]`

### load_qrels

Load relevance judgments (qrels).

```python
# Load TREC format qrels (qid iter docid relevance)
qrels = DataLoader.load_qrels("qrels.txt", format="trec")
```

**Parameters:**
- `path` (str): Path to qrels file
- `format` (str): File format - "trec" (default: "trec")

**Returns:** `Dict[str, Dict[str, int]]` - Nested dict: qid → docid → relevance

### load_contexts

Load retrieved contexts for query reformulation.

```python
# Load contexts from JSONL
contexts = DataLoader.load_contexts("contexts.jsonl")
```

**Parameters:**
- `path` (str): Path to contexts JSONL file

**Returns:** `Dict[str, List[str]]` - Dict: qid → list of context strings

### save_queries

Save queries to file.

```python
# Save to TSV
DataLoader.save_queries(queries, "output.tsv")

# Save to JSONL
DataLoader.save_queries(queries, "output.jsonl", format="jsonl")
```

**Parameters:**
- `queries` (List[QueryItem]): Queries to save
- `path` (str): Output file path
- `format` (str): Output format - "tsv" or "jsonl" (default: "tsv")

---

## BEIR Format Loaders

Loaders for [BEIR](https://github.com/beir-cellar/beir) benchmark datasets.

### Installation

```bash
pip install querygym[beir]
```

### Download BEIR Datasets

```python
from beir.datasets.data_loader import GenericDataLoader

# Download and extract a BEIR dataset
data_path = GenericDataLoader("nfcorpus").download_and_unzip()
```

### querygym.loaders.beir.load_queries

Load queries from a BEIR dataset directory.

```python
from querygym.loaders.beir import load_queries

# Load queries (uses queries.jsonl)
queries = load_queries("./data/nfcorpus")
```

**Parameters:**
- `beir_data_dir` (str | Path): Path to BEIR dataset directory
- `split` (str): Dataset split (default: "test", kept for API consistency)

**Returns:** `List[QueryItem]`

**File Format:** BEIR uses `queries.jsonl` with `{"_id": "...", "text": "..."}`

### querygym.loaders.beir.load_qrels

Load qrels from a BEIR dataset.

```python
from querygym.loaders.beir import load_qrels

# Load test split qrels
qrels = load_qrels("./data/nfcorpus", split="test")

# Load dev split
qrels = load_qrels("./data/nfcorpus", split="dev")
```

**Parameters:**
- `beir_data_dir` (str | Path): Path to BEIR dataset directory
- `split` (str): Dataset split - "train", "dev", or "test" (default: "test")

**Returns:** `Dict[str, Dict[str, int]]`

**File Location:** `{beir_data_dir}/qrels/{split}.tsv`

### querygym.loaders.beir.load_corpus

Load document corpus from a BEIR dataset.

```python
from querygym.loaders.beir import load_corpus

# Load corpus
corpus = load_corpus("./data/nfcorpus")

# Access documents
doc = corpus["doc_id_123"]
print(doc["title"], doc["text"])
```

**Parameters:**
- `beir_data_dir` (str | Path): Path to BEIR dataset directory

**Returns:** `Dict[str, Dict[str, str]]` - Dict: doc_id → {"title": ..., "text": ...}

**File Format:** BEIR uses `corpus.jsonl` with `{"_id": "...", "title": "...", "text": "..."}`

---

## MS MARCO Format Loaders

Loaders for [MS MARCO](https://microsoft.github.io/msmarco/) datasets.

### Installation

```bash
pip install querygym  # No extra dependencies needed
```

### Download MS MARCO Datasets

```python
import ir_datasets

# Download via ir_datasets
dataset = ir_datasets.load("msmarco-passage/dev")

# Or download from official sources
# https://microsoft.github.io/msmarco/
```

### querygym.loaders.msmarco.load_queries

Load queries from MS MARCO TSV file.

```python
from querygym.loaders.msmarco import load_queries

# Load queries
queries = load_queries("./data/queries.dev.tsv")
```

**Parameters:**
- `queries_tsv` (str | Path): Path to MS MARCO queries TSV file

**Returns:** `List[QueryItem]`

**File Format:** TSV with `qid<TAB>query_text`

### querygym.loaders.msmarco.load_qrels

Load qrels from MS MARCO file.

```python
from querygym.loaders.msmarco import load_qrels

# Load qrels (handles both TREC and simplified formats)
qrels = load_qrels("./data/qrels.dev.tsv")
```

**Parameters:**
- `qrels_tsv` (str | Path): Path to MS MARCO qrels file

**Returns:** `Dict[str, Dict[str, int]]`

**File Formats Supported:**
- TREC format: `qid<TAB>0<TAB>docid<TAB>relevance`
- Simplified: `qid<TAB>docid<TAB>relevance`

### querygym.loaders.msmarco.load_collection

Load document collection from MS MARCO TSV file.

```python
from querygym.loaders.msmarco import load_collection

# Load collection
collection = load_collection("./data/collection.tsv")

# Access documents
doc_text = collection["doc_id_123"]
```

**Parameters:**
- `collection_tsv` (str | Path): Path to MS MARCO collection TSV file

**Returns:** `Dict[str, str]` - Dict: doc_id → document_text

**File Format:** TSV with `docid<TAB>text`

---

## Complete Example

```python
import querygym as qg

# Load BEIR dataset
queries = qg.loaders.beir.load_queries("./data/nfcorpus")
qrels = qg.loaders.beir.load_qrels("./data/nfcorpus", split="test")
corpus = qg.loaders.beir.load_corpus("./data/nfcorpus")

# Or load MS MARCO dataset
queries = qg.loaders.msmarco.load_queries("./data/queries.tsv")
qrels = qg.loaders.msmarco.load_qrels("./data/qrels.tsv")
collection = qg.loaders.msmarco.load_collection("./data/collection.tsv")

# Use with reformulation
reformulator = qg.create_reformulator("genqr_ensemble", model="gpt-4")
results = reformulator.reformulate_batch(queries)

# Save reformulated queries
qg.DataLoader.save_queries(
    [qg.QueryItem(r.qid, r.reformulated) for r in results],
    "reformulated.tsv"
)
```
