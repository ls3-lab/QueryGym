# CLI

- `run` — produce reformulated TSV. For Query2E, `--no-clean-output` parses the raw LLM output instead of removing chat-model framing first.
- `data-to-tsv` — convert any backend to `qid<TAB>query`.
- `prompts-list` / `prompts-show` — inspect prompt bank.
- `script-gen` — create a Pyserini + trec_eval bash script.
