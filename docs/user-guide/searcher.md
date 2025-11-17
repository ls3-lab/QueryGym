# 🔍 Searcher Interface Documentation

The queryGym searcher interface provides a **library-agnostic** way to perform document retrieval. Instead of being tied to specific libraries like Pyserini or PyTerrier, queryGym now accepts any searcher that implements the `BaseSearcher` interface.

## 🎯 Key Benefits

- **🔌 Library Agnostic**: Works with Pyserini, PyTerrier, or any custom searcher
- **🎯 Simple API**: Just specify `searcher_type` and `searcher_kwargs`
- **🔧 Flexible**: Pass custom searcher instances for advanced use cases
- **📊 Consistent**: Same interface regardless of underlying search library
- **🚀 Extensible**: Easy to add new searcher implementations

## 🏗️ Architecture

### Core Components

1. **`BaseSearcher`** - Abstract interface that all searchers must implement
2. **`SearchHit`** - Standardized search result format
3. **`SearcherRegistry`** - Registry for managing searcher implementations
4. **`Retriever`** - High-level retriever that works with any BaseSearcher

### Available Searchers

- **`PyseriniSearcher`** - Wraps Pyserini's LuceneSearcher and LuceneImpactSearcher
- **`PyTerrierSearcher`** - Wraps PyTerrier's search functionality
- **Custom Searchers** - Implement BaseSearcher for your own search library

## 📖 Usage Examples

### Method 1: Using searcher_type parameter

```python
import queryGym as qg

# Create reformulator with Pyserini searcher
reformulator = qg.create_reformulator(
    method_name="lamer",
    model="gpt-4",
    params={
        "searcher_type": "pyserini",
        "searcher_kwargs": {
            "index": "msmarco-v1-passage",
            "searcher_type": "bm25",
            "k1": 0.9,
            "b": 0.4,
            "rm3": False,
            "rocchio": False,
        },
        "retrieval_k": 10,
        "gen_passages": 5,
    }
)

# This will automatically retrieve contexts using Pyserini
query = qg.QueryItem(qid="q1", text="what causes diabetes")
result = reformulator.reformulate(query)
```

### Method 2: Using custom searcher instance

```python
# Create a custom searcher
custom_searcher = qg.create_searcher(
    "pyserini",
    index="msmarco-v1-passage",
    searcher_type="bm25",
    k1=0.9,
    b=0.4,
    rm3=True  # Enable RM3
)

# Create retriever with custom searcher
retriever = qg.Retriever(searcher=custom_searcher)

# Use retriever directly
results = retriever.retrieve("machine learning algorithms", k=5)
```

### Method 3: Direct retriever usage

```python
# Create retriever with searcher configuration
retriever = qg.Retriever(
    searcher_type="pyserini",
    searcher_kwargs={
        "index": "msmarco-v1-passage",
        "searcher_type": "bm25",
    }
)

# Batch retrieval
queries = ["query 1", "query 2", "query 3"]
batch_results = retriever.retrieve_batch(queries, k=10, num_threads=4)
```

## 🔧 Searcher Configuration

### Pyserini Configuration

```python
searcher_kwargs = {
    "index": "msmarco-v1-passage",  # Path or prebuilt index name
    "searcher_type": "bm25",        # "bm25" or "impact"
    "encoder": None,                # Encoder for Impact search
    "min_idf": 0,                  # Minimum IDF for Impact search
    "k1": 0.9,                     # BM25 k1 parameter
    "b": 0.4,                      # BM25 b parameter
    "rm3": False,                  # Enable RM3 expansion
    "rocchio": False,              # Enable Rocchio expansion
    "rocchio_use_negative": False,  # Use negative feedback in Rocchio
    "answer_key": "contents",      # Field to extract content from
}
```

### PyTerrier Configuration

```python
searcher_kwargs = {
    "index_path": "/path/to/index",  # Path to PyTerrier index
    "index_name": "msmarco_passage", # Name of prebuilt index
    "searcher_type": "bm25",         # "bm25", "tfidf", "pl2", etc.
}
```

## 🛠️ Creating Custom Searchers

To create a custom searcher, implement the `BaseSearcher` interface:

```python
from queryGym.core.searcher import BaseSearcher, SearchHit
from typing import List, Dict, Any

class MyCustomSearcher(BaseSearcher):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._searcher_info = {
            "name": "MyCustomSearcher",
            "type": "custom",
            **kwargs
        }
    
    def search(self, query: str, k: int = 10, **kwargs) -> List[SearchHit]:
        """Search for documents using a single query."""
        # Your search implementation here
        hits = []
        for i in range(k):
            hit = SearchHit(
                docid=f"doc_{i}",
                score=1.0 - (i * 0.1),
                content=f"Content for query: {query}",
                metadata={"custom": True}
            )
            hits.append(hit)
        return hits
    
    def batch_search(self, queries: List[str], k: int = 10, 
                    num_threads: int = 1, **kwargs) -> List[List[SearchHit]]:
        """Search for documents using multiple queries in batch."""
        return [self.search(query, k, **kwargs) for query in queries]
    
    def get_searcher_info(self) -> Dict[str, Any]:
        """Get information about the searcher configuration."""
        return self._searcher_info.copy()

# Register the custom searcher
from queryGym.core.searcher import SearcherRegistry
SearcherRegistry.register("my_custom", MyCustomSearcher)

# Use it
retriever = qg.Retriever(searcher_type="my_custom", searcher_kwargs={})
```

## 🔄 Migration from Old API

### Before (Pyserini-specific)

```python
# Old way - tied to Pyserini
from pyserini.search.lucene import LuceneSearcher
searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
hits = searcher.search(query, k=10)
```

### After (Library-agnostic)

```python
# New way - library-agnostic
retriever = qg.Retriever(searcher_type="pyserini", searcher_kwargs={
    "index": "msmarco-v1-passage"
})
results = retriever.retrieve(query, k=10)
```

## 🚀 Advanced Usage

### Comparing Different Searchers

```python
query = "machine learning applications"
k = 5

# Test Pyserini
pyserini_retriever = qg.Retriever(
    searcher_type="pyserini",
    searcher_kwargs={"index": "msmarco-v1-passage"}
)
pyserini_results = pyserini_retriever.retrieve(query, k)

# Test PyTerrier
pyterrier_retriever = qg.Retriever(
    searcher_type="pyterrier",
    searcher_kwargs={"index_name": "msmarco_passage"}
)
pyterrier_results = pyterrier_retriever.retrieve(query, k)

# Compare results
print(f"Pyserini: {len(pyserini_results)} results")
print(f"PyTerrier: {len(pyterrier_results)} results")
```

### Integration with Query Reformulation Methods

```python
# LameR with Pyserini
lamer_pyserini = qg.create_reformulator(
    "lamer",
    params={
        "searcher_type": "pyserini",
        "searcher_kwargs": {"index": "msmarco-v1-passage"},
        "retrieval_k": 10
    }
)

# LameR with PyTerrier
lamer_pyterrier = qg.create_reformulator(
    "lamer",
    params={
        "searcher_type": "pyterrier",
        "searcher_kwargs": {"index_name": "msmarco_passage"},
        "retrieval_k": 10
    }
)

# Both will work the same way
query = qg.QueryItem("q1", "what causes diabetes")
result1 = lamer_pyserini.reformulate(query)
result2 = lamer_pyterrier.reformulate(query)
```

## 📋 CLI Usage

The CLI also supports the new searcher interface:

```bash
# Using Pyserini
python -m queryGym.retrieve_context \
  --searcher-type pyserini \
  --index msmarco-v1-passage \
  --queries-tsv queries.tsv \
  --output results.tsv \
  --k 10

# Using PyTerrier
python -m queryGym.retrieve_context \
  --searcher-type pyterrier \
  --index-name msmarco_passage \
  --queries-tsv queries.tsv \
  --output results.tsv \
  --k 10
```

## 🎯 Best Practices

1. **Use searcher_type for simple cases**: When you just need basic functionality
2. **Use custom searcher instances for advanced cases**: When you need fine-grained control
3. **Implement BaseSearcher for new libraries**: Easy to add support for new search libraries
4. **Use batch operations**: More efficient for multiple queries
5. **Configure searcher parameters**: Tune parameters for your specific use case

## 🔍 Troubleshooting

### Common Issues

1. **ImportError**: Make sure the required search library is installed
   ```bash
   pip install pyserini  # For Pyserini
   pip install python-terrier  # For PyTerrier
   ```

2. **Index not found**: Check that the index path or name is correct

3. **Searcher not registered**: Make sure to import the adapter modules
   ```python
   import queryGym.adapters.pyserini_adapter  # Registers PyseriniSearcher
   import queryGym.adapters.pyterrier_adapter  # Registers PyTerrierSearcher
   ```

### Getting Help

- Check the example scripts in `examples/searcher_interface_example.py`
- Look at the adapter implementations in `queryGym/adapters/`
- Review the BaseSearcher interface in `queryGym/core/searcher.py`
