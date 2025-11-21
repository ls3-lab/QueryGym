# Docker Examples for QueryGym

This directory contains practical examples for using QueryGym with Docker.

> 📖 **Quick Reference:** See [DOCKER_SETUP.md](../../DOCKER_SETUP.md) in the root directory for a quick setup guide.

## 🚀 Quick Start

### Prerequisites
- Docker installed
- For GPU support: NVIDIA GPU + nvidia-container-toolkit

### Setup

1. **Create directories:**
   ```bash
   mkdir -p data outputs notebooks
   ```

2. **Set your API key:**
   ```bash
   export OPENAI_API_KEY="sk-..."
   # Or create a .env file
   echo "OPENAI_API_KEY=sk-..." > .env
   ```

3. **Run with Docker Compose:**
   ```bash
   # GPU version (interactive shell)
   docker-compose run --rm querygym
   
   # CPU version (interactive shell)
   docker-compose run --rm querygym-cpu
   
   # Jupyter notebook server
   docker-compose up jupyter
   # Then open http://localhost:8888
   ```

## 📁 What's Included

- **`docker-compose.yml`** - Pre-configured services using GHCR images (symlink to root)
- **`quickstart.ipynb`** - Example Jupyter notebook
- **`data/`** - Mount your datasets here
- **`outputs/`** - Reformulated queries saved here
- **`notebooks/`** - Your Jupyter notebooks

## 💡 Usage Examples

### Example 1: Interactive Python Session

```bash
docker-compose run --rm querygym

# Inside container:
python
>>> import querygym as qg
>>> queries = qg.load_queries("data/queries.tsv")
>>> reformulator = qg.create_reformulator("genqr")
>>> result = reformulator.reformulate(queries[0])
>>> print(result.reformulated)
```

### Example 2: Run a Script

```bash
# Create your script
cat > my_experiment.py << 'EOF'
import querygym as qg

queries = qg.load_queries("data/queries.tsv")
reformulator = qg.create_reformulator("genqr_ensemble", model="gpt-4")
results = reformulator.reformulate_batch(queries)

qg.DataLoader.save_queries(
    [qg.QueryItem(r.qid, r.reformulated) for r in results],
    "outputs/reformulated.tsv"
)
print(f"Reformulated {len(results)} queries!")
EOF

# Run it
docker-compose run --rm querygym python my_experiment.py
```

### Example 3: Jupyter Notebook

```bash
# Start Jupyter server
docker-compose up jupyter

# Open http://localhost:8888 in your browser
# Try the included quickstart.ipynb notebook
```

### Example 4: CPU-Only (No GPU)

```bash
# Use the CPU version if you don't have a GPU
docker-compose run --rm querygym-cpu
```

## 🔧 Configuration

### Environment Variables

Set these in a `.env` file or export them:

```bash
OPENAI_API_KEY=sk-...              # Required for LLM-based methods
OPENAI_BASE_URL=https://...        # Optional: custom API endpoint
```

### Volume Mounts

The compose file mounts these directories:
- `./data` → `/workspace/data` (your datasets)
- `./outputs` → `/workspace/outputs` (results)
- `./notebooks` → `/workspace/notebooks` (Jupyter notebooks)

## 📚 More Information

- [Full Docker Guide](https://querygym.readthedocs.io/en/latest/user-guide/docker/)
- [QueryGym Documentation](https://querygym.readthedocs.io/)
- [GitHub Repository](https://github.com/radinhamidi/QueryGym)

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Verify nvidia-docker is working
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi

# If not working, install nvidia-container-toolkit:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Permission Issues

```bash
# Run as your user
docker-compose run --rm -u $(id -u):$(id -g) querygym
```

### Pull Latest Images

```bash
# Update to latest version
docker-compose pull
```
