# QueryGym Examples

This directory contains practical examples and tutorials for using QueryGym in different scenarios.

## 📁 Directory Structure

### 1. [snippets](snippets/)
**Quick code snippets** demonstrating core QueryGym functionality.

Perfect for:
- Learning the basics
- Quick reference
- Copy-paste solutions

Examples include:
- Loading data from files
- Using format loaders (BEIR, MS MARCO)
- Query reformulation
- Context-based reformulation
- Using search adapters

### 2. [docker](docker/)
**Docker-based examples** for containerized workflows.

Perfect for:
- Quick start without installation
- Reproducible environments
- Jupyter notebook development

Includes:
- Ready-to-use `docker-compose.yml` (Compose V2 compatible)
- Interactive Jupyter notebook
- Quick setup guide

### 3. [querygym_pyserini](querygym_pyserini/)
**End-to-end retrieval workflows** integrating QueryGym with Pyserini.

Perfect for:
- Complete IR experiments
- Indexing and retrieval pipelines
- Evaluation workflows

Includes:
- Building indexes
- Query reformulation + retrieval
- Evaluation with trec_eval
- Complete experiment scripts

## 🚀 Quick Start

### Option 1: Run with Docker (Easiest)
```bash
cd docker/
docker compose run --rm querygym
```

### Option 2: Run Python Snippets
```bash
# Install QueryGym first
pip install querygym

# Run any snippet
cd snippets/
python 01_load_from_file.py
```

### Option 3: Full Retrieval Pipeline
```bash
cd querygym_pyserini/
# Follow the README for complete setup
```

## 📚 Learning Path

**Beginner:**
- Start with [snippets](snippets/) to learn the basics
- Try [docker](docker/) for a containerized environment

**Intermediate:**
- Explore different reformulation methods in snippets
- Run Jupyter notebooks in Docker

**Advanced:**
- Build complete pipelines with [querygym_pyserini](querygym_pyserini/)
- Integrate with your own retrieval systems

## 🔗 Additional Resources

- **Documentation:** [https://querygym.readthedocs.io](https://querygym.readthedocs.io/)
- **Docker Setup:** [DOCKER_SETUP.md](../../DOCKER_SETUP.md) in the root directory
- **API Reference:** [https://querygym.readthedocs.io/en/latest/api/core/](https://querygym.readthedocs.io/en/latest/api/core/)
- **GitHub:** [https://github.com/ls3-lab/QueryGym](https://github.com/ls3-lab/QueryGym)

## 💡 Tips

- **Start simple:** Begin with basic snippets before complex workflows
- **Use Docker:** For quick experimentation without setup hassles
- **Check READMEs:** Each subdirectory has its own detailed README
- **Modify examples:** All examples are meant to be adapted to your needs

## 🤝 Contributing Examples

Have a useful example? We'd love to include it! Please:
1. Add it to the appropriate subdirectory
2. Include clear comments and documentation
3. Test it works with the latest QueryGym version
4. Submit a pull request

See the [Contributing section](../README.md#contributing) in the main README for general contribution guidelines.

## ❓ Need Help?

- Check the [documentation](https://querygym.readthedocs.io/)
- Open an [issue](https://github.com/ls3-lab/QueryGym/issues)
- Join discussions on GitHub
