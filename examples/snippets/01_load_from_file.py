"""Load queries from local files"""
import querygym as qg
from pathlib import Path

example_dir = Path(__file__).parent

# Load from TSV
queries = qg.load_queries(example_dir / "tiny_queries.tsv")
print(f"Loaded {len(queries)} queries")
for q in queries[:2]:
    print(f"  {q.qid}: {q.text}")

# Load qrels
qrels = qg.load_qrels(example_dir / "tiny_qrels.txt")
print(f"\nLoaded qrels for {len(qrels)} queries")

# Save queries
qg.DataLoader.save_queries(queries, example_dir / "output.tsv", format="tsv")
print("Saved to output.tsv")
