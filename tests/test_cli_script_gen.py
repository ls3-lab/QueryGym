from querygym.cli import build_script_lines


def test_script_lines():
    lines = build_script_lines(
        index_path="/idx", topics="/q.tsv", run="/run.txt", qrels="/qrels.txt"
    )
    assert any("--index /idx" in ln for ln in lines)
