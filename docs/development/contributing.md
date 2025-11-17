# Contributing to querygym

Thank you for your interest in contributing to querygym! This guide will help you get started.

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/queryGym.git
cd queryGym
```

### 2. Set Up Development Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e .[dev]
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=querygym

# Run specific test file
pytest tests/test_prompts.py

# Run with verbose output
pytest -v
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code with black
black querygym/ tests/

# Lint with ruff
ruff check querygym/ tests/

# Type checking with mypy
mypy querygym/
```

### Pre-commit Checks

Before committing, ensure:

1. All tests pass: `pytest`
2. Code is formatted: `black querygym/ tests/`
3. No linting errors: `ruff check querygym/ tests/`
4. Type hints are correct: `mypy querygym/`

## Adding a New Reformulation Method

### 1. Create Method File

Create a new file in `querygym/methods/`:

```python
# querygym/methods/my_method.py
from __future__ import annotations
from typing import Optional, List
from ..core.base import BaseReformulator, QueryItem, ReformulationResult, MethodConfig

class MyMethod(BaseReformulator):
    VERSION = "1.0"
    REQUIRES_CONTEXT = False  # Set to True if method needs contexts
    
    def reformulate(self, q: QueryItem, contexts: Optional[List[str]] = None) -> ReformulationResult:
        # Get prompt
        prompt = self.prompts.get("my_method_prompt_id")
        
        # Build messages
        messages = [
            {"role": "system", "content": prompt.template["system"]},
            {"role": "user", "content": prompt.template["user"].format(query=q.text)}
        ]
        
        # Call LLM
        response = self.llm.chat(messages)
        
        # Concatenate with original query
        reformulated = self.concatenate_result(q.text, response)
        
        return ReformulationResult(
            qid=q.qid,
            original=q.text,
            reformulated=reformulated
        )
```

### 2. Add Prompt to Prompt Bank

Add your prompt to `querygym/prompt_bank.yaml`:

```yaml
my_method_prompt_id:
  meta:
    version: 1
    introduced_by: "Your Name"
    license: "CC-BY-4.0"
    authors: ["Your Name"]
  method_family: "my_method"
  template:
    system: "You are a helpful assistant for query reformulation."
    user: "Reformulate this query: {query}"
```

### 3. Register the Method

Add to `querygym/methods/__init__.py`:

```python
from .my_method import MyMethod

__all__ = [
    # ... existing methods
    "MyMethod",
]
```

Register in `querygym/core/registry.py`:

```python
from ..methods.my_method import MyMethod

register_method("my_method", MyMethod)
```

### 4. Add Tests

Create `tests/test_my_method.py`:

```python
from querygym.methods.my_method import MyMethod
from querygym.core.base import QueryItem, MethodConfig
from pathlib import Path

def test_my_method():
    # Setup
    cfg = MethodConfig(
        name="my_method",
        params={},
        llm={"model": "gpt-4"}
    )
    
    # Mock LLM client
    class MockLLM:
        def chat(self, messages):
            return "expanded query"
    
    # Mock prompt resolver
    class MockPrompts:
        def get(self, prompt_id):
            class Prompt:
                template = {
                    "system": "System prompt",
                    "user": "User: {query}"
                }
            return Prompt()
    
    method = MyMethod(cfg, MockLLM(), MockPrompts())
    
    # Test
    query = QueryItem("q1", "test query")
    result = method.reformulate(query)
    
    assert result.qid == "q1"
    assert result.original == "test query"
    assert "expanded query" in result.reformulated
```

### 5. Update Documentation

Add documentation in `docs/user-guide/reformulation.md`.

## Pull Request Process

1. **Update your branch**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:
   ```bash
   pytest
   black querygym/ tests/
   ruff check querygym/ tests/
   mypy querygym/
   ```

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request** on GitHub

### PR Guidelines

- Write clear, descriptive commit messages
- Include tests for new features
- Update documentation as needed
- Keep PRs focused on a single feature/fix
- Respond to review feedback promptly

## Code Style

- Follow PEP 8
- Use type hints for all functions
- Write docstrings in Google style
- Keep functions small and focused
- Use meaningful variable names

## Questions?

- Open an issue on GitHub
- Join discussions in pull requests
- Check existing documentation

Thank you for contributing! 🎉
