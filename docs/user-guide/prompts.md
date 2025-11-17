# Prompt Bank

Single YAML file: `queryGym/prompt_bank.yaml`

Each entry:
- `id`: Unique string (e.g., `genqr.keywords.v1`)
- `method_family`: logical grouping
- `version`: integer
- `introduced_by`: citation-like text
- `license`, `authors`, `tags`
- `template.system`, `template.user`: may contain `{query}`
- `notes`: free text
