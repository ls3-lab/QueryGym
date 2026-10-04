from querygym import cli
from typer.testing import CliRunner


def write_config(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_bundled_method_block_applies():
    cfg = cli.resolve_config("query2e")

    assert cfg["llm"]["temperature"] == 0.7
    assert cfg["llm"]["max_tokens"] == 256
    assert cfg["params"]["mode"] == "fs"
    assert cfg["params"]["index"] == "msmarco-v1-passage"
    assert cfg["seed"] == 17


def test_bundled_top_level_applies_without_method_block():
    cfg = cli.resolve_config("unknown_method")

    assert cfg["llm"]["temperature"] == 1.0
    assert cfg["llm"]["max_tokens"] == 1024


def test_user_config_overrides_bundled_method_block(tmp_path):
    path = write_config(
        tmp_path,
        "llm:\n  temperature: 0.5\nparams:\n  retrieval_k: 3\n"
        "query2e:\n  params:\n    max_keywords: 10\n",
    )
    cfg = cli.resolve_config("query2e", path)

    assert cfg["llm"]["temperature"] == 0.5
    assert cfg["llm"]["max_tokens"] == 256
    assert cfg["params"] == {"mode": "fs", "retrieval_k": 3, "max_keywords": 10}
    assert cfg["seed"] == 42


def test_user_method_block_overrides_user_top_level(tmp_path):
    path = write_config(
        tmp_path, "llm:\n  temperature: 0.5\nquery2e:\n  llm:\n    temperature: 0.1\n"
    )
    assert cli.resolve_config("query2e", path)["llm"]["temperature"] == 0.1


def test_run_passes_resolved_config(tmp_path, monkeypatch):
    queries = tmp_path / "queries.tsv"
    queries.write_text("q1\tprime rate\n")
    captured = {}

    def fake_run_method(method_name, cfg, queries, **kwargs):
        captured["cfg"] = cfg
        return []

    monkeypatch.setattr(cli, "run_method", fake_run_method)
    result = CliRunner().invoke(
        cli.app,
        [
            "run",
            "--method",
            "query2e",
            "--queries-tsv",
            str(queries),
            "--output-tsv",
            str(tmp_path / "out.tsv"),
            "--num-examples",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["cfg"].llm["temperature"] == 0.7
    assert captured["cfg"].params["mode"] == "fs"
    assert captured["cfg"].params["num_examples"] == 2
