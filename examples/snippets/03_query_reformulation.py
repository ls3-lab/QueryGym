"""Reformulate queries with different methods"""
import querygym as qg
from pathlib import Path

example_dir = Path(__file__).parent

# Load queries
queries = qg.load_queries(example_dir / "tiny_queries.tsv")

# Create reformulator
reformulator = qg.create_reformulator("genqr_ensemble", model="gpt-4")

# Reformulate
results = reformulator.reformulate_batch(queries)

# Show results
for r in results:
    print(f"{r.qid}:")
    print(f"  Original: {r.original}")
    print(f"  Reformed: {r.reformulated}\n")
