"""Reformulate queries using retrieved contexts"""
import querygym as qg
from pathlib import Path

example_dir = Path(__file__).parent

# Load queries and contexts
queries = qg.load_queries(example_dir / "tiny_queries.tsv")
contexts = qg.load_contexts(example_dir / "tiny_contexts.jsonl")

# Create context-based reformulator (Query2Doc uses contexts)
reformulator = qg.create_reformulator("query2doc", model="gpt-4")

# Reformulate with contexts
results = reformulator.reformulate_batch(queries, contexts=contexts)

# Show results
for r in results:
    print(f"{r.qid}: {r.reformulated[:80]}...")
