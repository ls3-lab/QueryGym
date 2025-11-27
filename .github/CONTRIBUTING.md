# Contributing to QueryGym

Thank you for your interest in contributing to QueryGym! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/QueryGym.git
   cd QueryGym
   ```
3. **Set up the development environment**:
   ```bash
   pip install -e ".[dev]"
   ```

## How to Contribute

### Reporting Bugs

- Check if the issue already exists in [GitHub Issues](https://github.com/ls3-lab/QueryGym/issues)
- If not, create a new issue using the **Bug Report** template
- Include as much detail as possible: steps to reproduce, expected vs actual behavior, environment info

### Suggesting Features

- Open an issue using the **Feature Request** template
- Describe the feature and its use case
- Discuss the implementation approach if you have ideas

### Adding a New Prompt

1. Edit `querygym/prompt_bank.yaml`
2. Add an entry with required fields:
   - `id`: Unique identifier (e.g., `method.variant.v1`)
   - `method_family`: Parent method name
   - `version`: Version string
   - `introduced_by`: Paper/source reference
   - `license`: License type
   - `authors`: List of authors
   - `tags`: Relevant tags
   - `template`: Contains `system` and `user` templates
   - `notes`: Additional notes

### Adding a New Method

1. Create a new file under `querygym/methods/`
2. Subclass `BaseReformulator`
3. Implement required methods:
   - `reformulate(self, query, contexts=None)`
4. Register with `@register_method("method_name")`
5. Add tests in `tests/`
6. Update documentation

## Development Setup

### Prerequisites

- Python 3.9+
- Git

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/QueryGym.git
cd QueryGym

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest
```

### Using Docker

```bash
# Build development image
make build

# Run tests in container
make test
```

## Pull Request Process

1. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the style guidelines

3. **Add tests** for new functionality

4. **Run tests locally**:
   ```bash
   pytest
   ```

5. **Commit your changes** with clear messages:
   ```bash
   git commit -m "Add: description of your changes"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** on GitHub using the PR template

8. **Address review feedback** if requested

### Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Prefix with type: `Add:`, `Fix:`, `Update:`, `Remove:`, `Refactor:`

## Style Guidelines

### Python Code

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints where appropriate
- Write docstrings for public functions and classes
- Keep functions focused and small

### Documentation

- Update relevant documentation for any changes
- Use clear, concise language
- Include code examples where helpful

### Testing

- Write tests for new features
- Maintain or improve code coverage
- Tests should be fast and isolated

## Questions?

- Open a [Discussion](https://github.com/ls3-lab/QueryGym/discussions) for general questions
- Check existing issues and discussions first
- Join our community!

---

Thank you for contributing to QueryGym! 🎉
