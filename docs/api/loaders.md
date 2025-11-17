# Data Loaders API Reference

## DataLoader

Core data loading utilities.

### load_queries

```python
from querygym.data.dataloader import DataLoader

# Load from TSV (qid<TAB>text)
queries = DataLoader.load_queries("queries.tsv")

# Load from JSONL
queries = DataLoader.load_queries("queries.jsonl", format="jsonl")
```

### load_qrels

```python
# Load qrels (qid docid relevance format)
qrels = DataLoader.load_qrels("qrels.txt")
```

### load_contexts

```python
# Load contexts from JSONL
contexts = DataLoader.load_contexts("contexts.jsonl")
```

### save_queries

```python
# Save queries to TSV
DataLoader.save_queries(queries, "output.tsv")

# Save to JSONL
DataLoader.save_queries(queries, "output.jsonl", format="jsonl")
```

## BEIR Loaders

Loaders for BEIR dataset format.

### load_queries

```python
from querygym.loaders.beir import load_queries

# Load queries from BEIR dataset
queries = load_queries("./data/nfcorpus")
```

### load_qrels

```python
from querygym.loaders.beir import load_qrels

# Load qrels for specific split
qrels = load_qrels("./data/nfcorpus", split="test")
```

### load_corpus

```python
from querygym.loaders.beir import load_corpus

# Load document corpus
corpus = load_corpus("./data/nfcorpus")
```

## MS MARCO Loaders

Loaders for MS MARCO dataset format.

### load_queries

```python
from querygym.loaders.msmarco import load_queries

# Load MS MARCO queries
queries = load_queries("./data/queries.tsv")
```

### load_qrels

```python
from querygym.loaders.msmarco import load_qrels

# Load MS MARCO qrels
qrels = load_qrels("./data/qrels.tsv")
```

### load_collection

```python
from querygym.loaders.msmarco import load_collection

# Load document collection
collection = load_collection("./data/collection.tsv")
```
