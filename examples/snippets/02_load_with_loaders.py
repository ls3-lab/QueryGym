"""Load BEIR and MS MARCO datasets"""
import querygym as qg

# BEIR dataset (assumes already downloaded)
beir_path = "./data/beir/nfcorpus"
queries = qg.loaders.beir.load_queries(beir_path, split="test")
qrels = qg.loaders.beir.load_qrels(beir_path, split="test")
print(f"BEIR: {len(queries)} queries, {len(qrels)} qrels")

# MS MARCO dataset
queries = qg.loaders.msmarco.load_queries("./data/msmarco/queries.tsv")
qrels = qg.loaders.msmarco.load_qrels("./data/msmarco/qrels.tsv")
print(f"MS MARCO: {len(queries)} queries, {len(qrels)} qrels")
