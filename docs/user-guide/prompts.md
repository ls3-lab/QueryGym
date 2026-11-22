# Prompt Bank

Single YAML file: `querygym/prompt_bank.yaml`

Each entry:
- `id`: Unique string (e.g., `genqr.keywords.v1`)
- `method_family`: logical grouping
- `version`: integer
- `introduced_by`: citation-like text
- `license`, `authors`, `tags`
- `template`: the messages, in one of the two forms below; text may contain `{query}` and other variables passed to `PromptBank.render`
- `notes`: free text

## Template forms

**Role keys.** At most one message per role, rendered as system, user, then assistant (each only if present):

```yaml
template:
  system: |
    You output only comma-separated keywords. No sentences.
  user: |
    Suggest keywords to improve retrieval for this query:
    "{query}"
```

**Message list.** Any number of messages, rendered in the order written. Use it when a prompt has more than one message per role, such as a few-shot example written as conversation turns:

```yaml
template:
  - role: system
    content: |
      You extract key sentences from retrieved documents.
  - role: user
    content: |
      Query: "example query"
      Retrieved documents: ...
  - role: assistant
    content: |
      Example answer ...
  - role: user
    content: |
      Query: "{query}"
      Retrieved documents:
      {contexts}
```

Each message needs a `role` (`system`, `user` or `assistant`) and a `content`. Loading the bank fails if a key appears twice in the same mapping, for example two `user` keys under one `template`.
