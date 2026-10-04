# CLI

- `run` — produce reformulated TSV. For Query2E, `--no-clean-output` parses the raw LLM output instead of removing chat-model framing first.
- `data-to-tsv` — convert any backend to `qid<TAB>query`.
- `prompts-list` / `prompts-show` — inspect prompt bank.
- `script-gen` — create a Pyserini + trec_eval bash script.

## Configuration

`querygym run` uses the bundled defaults (`querygym/config/defaults.yaml`). They contain top-level `llm` and `params` settings plus a block per method with that method's defaults:

```yaml
query2e:
  llm:
    temperature: 0.7
    max_tokens: 256
  params:
    mode: "fs"
```

Pass `--cfg-path` to override them with your own YAML file. Settings are resolved in this order (highest first):

1. CLI flags (`--mode`, `--num-examples`, ...)
2. The method's block in your config file
3. Top-level `llm` / `params` in your config file
4. The method's block in the bundled defaults
5. Top-level `llm` in the bundled defaults

Top-level `params` of the bundled defaults (index and retrieval settings) only apply when no `--cfg-path` is given. `seed` and `retries` come from your config file when given (default `42` / `2`).

A config file only needs the settings you want to change:

```yaml
llm:
  model: "qwen2.5:7b"
  base_url: "http://localhost:11434/v1"
  api_key: "ollama"
query2e:
  params:
    max_keywords: 10
```
