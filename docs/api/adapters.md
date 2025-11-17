# Adapters API Reference

## Searcher Interface

The searcher interface provides a unified API for different search backends.

### BaseSearcher

Abstract base class for all searcher implementations.

**Methods:**
- `search(query: str, k: int = 10) -> List[SearchHit]` - Search for documents
- `batch_search(queries: List[str], k: int = 10) -> Dict[str, List[SearchHit]]` - Batch search

### SearchHit

Represents a single search result.

**Attributes:**
- `docid: str` - Document ID
- `score: float` - Relevance score
- `text: Optional[str]` - Document text (if available)

## Pyserini Adapter

### PyseriniSearcher

Adapter for Pyserini's LuceneSearcher and LuceneImpactSearcher.

```python
from querygym.adapters import PyseriniSearcher

searcher = PyseriniSearcher(index_path="path/to/index")
results = searcher.search("query text", k=10)
```

## PyTerrier Adapter

### PyTerrierSearcher

Adapter for PyTerrier retrievers.

```python
from querygym.adapters import PyTerrierSearcher
import pyterrier as pt

pt.init()
retriever = pt.BatchRetrieve(index, wmodel="BM25")
searcher = PyTerrierSearcher(retriever)
results = searcher.search("query text", k=10)
```

## Wrapper Functions

### wrap_pyserini_searcher

Wrap a Pyserini searcher instance.

```python
from querygym import wrap_pyserini_searcher
from pyserini.search.lucene import LuceneSearcher

lucene_searcher = LuceneSearcher("path/to/index")
searcher = wrap_pyserini_searcher(lucene_searcher)
```

### wrap_pyterrier_retriever

Wrap a PyTerrier retriever instance.

```python
from querygym import wrap_pyterrier_retriever
import pyterrier as pt

pt.init()
retriever = pt.BatchRetrieve(index, wmodel="BM25")
searcher = wrap_pyterrier_retriever(retriever)
```

### wrap_custom_searcher

Wrap a custom search function.

```python
from querygym import wrap_custom_searcher

def my_search(query: str, k: int) -> List[dict]:
    # Your custom search logic
    return [{"docid": "doc1", "score": 0.9, "text": "..."}]

searcher = wrap_custom_searcher(my_search)
```
