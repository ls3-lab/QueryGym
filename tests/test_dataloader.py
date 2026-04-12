from querygym.data.dataloader import DataLoader
from pathlib import Path
import tempfile


def test_local_tsv_loading():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tsv") as f:
        f.write("1\tapple\n2\tbanana\n")
        path = f.name
    queries = DataLoader.load_queries(path, format="tsv")
    assert len(queries) == 2 and queries[0].qid == "1"
