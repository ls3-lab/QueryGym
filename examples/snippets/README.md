# queryGym Code Snippets

Quick, focused examples showing core functionality.

## Examples

1. **`01_load_from_file.py`** - Load queries and qrels from local files
2. **`02_load_with_loaders.py`** - Load BEIR and MS MARCO datasets
3. **`03_query_reformulation.py`** - Reformulate queries with LLMs
4. **`04_context_reformulation.py`** - Reformulate using retrieved contexts
5. **`05_using_adapters.py`** - Use Pyserini and PyTerrier adapters

## Run Examples

```bash
cd examples
conda activate querygym

# Set API key for reformulation
export OPENAI_API_KEY='sk-...'

# Run any example
python 01_load_from_file.py
python 03_query_reformulation.py
```

## Data Files

Examples use:
- `tiny_queries.tsv` - Sample queries
- `tiny_qrels.txt` - Sample relevance judgments
- `tiny_contexts.jsonl` - Sample retrieved contexts
- BEIR/MS MARCO data (download separately)
