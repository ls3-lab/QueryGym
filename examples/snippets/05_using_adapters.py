"""Use Pyserini and PyTerrier adapters"""
import querygym as qg

# Pyserini adapter
from pyserini.search.lucene import LuceneSearcher
searcher = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")
pyserini_adapter = qg.wrap_pyserini_searcher(searcher)

# Search
query = qg.QueryItem("1", "what is neural IR")
hits = pyserini_adapter.search(query.text, k=10)
print(f"Pyserini: {len(hits)} hits")

# PyTerrier adapter (if available)
# import pyterrier as pt
# bm25 = pt.BatchRetrieve.from_dataset("msmarco_passage", "terrier_stemmed")
# pyterrier_adapter = qg.wrap_pyterrier_retriever(bm25)
