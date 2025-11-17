# Development Guide

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,all]"
```

The `-e` flag installs in editable mode: code changes take effect immediately without reinstalling.

## Testing

```bash
pytest                    # Run all tests
pytest --cov=queryGym    # With coverage
black queryGym/ tests/   # Format code
ruff check --fix queryGym/  # Lint
```

## Adding a New Method

1. Create `queryGym/methods/my_method.py`:
```python
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method

@register_method("my_method")
class MyMethod(BaseReformulator):
    def reformulate(self, q: QueryItem, contexts=None):
        msgs = self.prompts.render("my_method.v1", query=q.text)
        result = self.llm.chat(msgs, temperature=0.7)
        return ReformulationResult(q.qid, q.text, result)
```

2. Add prompt to `prompt_bank.yaml`
3. Export in `methods/__init__.py` and main `__init__.py`
4. Write tests in `tests/test_my_method.py`

## Publishing

```bash
python -m build
twine upload --repository testpypi dist/*  # Test first
twine upload dist/*                         # Then publish
