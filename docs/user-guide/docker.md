# Docker Guide for QueryGym

Pre-built container images are available on GitHub Container Registry. No building required - just pull and run!

## 🐳 Why Use Our Container Images?

- ✅ **Zero setup** - Everything pre-installed and configured
- ✅ **Reproducible** - Same environment on any machine
- ✅ **Instant start** - Pull and run in seconds
- ✅ **No dependency hell** - All packages pre-configured
- ✅ **GPU ready** - CUDA support included by default

## 📦 Available Images

All images are published to GitHub Container Registry: `ghcr.io/ls3-lab/querygym`

### GPU-Enabled Image (Default - Recommended)
```bash
docker pull ghcr.io/ls3-lab/querygym:latest
```
- **Size:** ~6 GB
- **Base:** NVIDIA CUDA 12.1 on Ubuntu 22.04
- **Includes:** querygym, PySerini, PyTorch (CUDA), Jupyter, Transformers, Sentence-Transformers
- **Use case:** GPU-accelerated experiments, neural retrieval, production workloads
- **Requirements:** NVIDIA GPU + nvidia-container-toolkit

### CPU-Only Image (Lightweight Alternative)
```bash
docker pull ghcr.io/ls3-lab/querygym:cpu
```
- **Size:** ~2.5 GB
- **Base:** Python 3.10 on Debian Slim
- **Includes:** querygym, PySerini, PyTorch (CPU), Jupyter
- **Use case:** Testing, development without GPU, resource-constrained environments
- **Requirements:** Docker only (no GPU needed)

## 🚀 Quick Start

### GPU Version (Recommended)

```bash
# Pull the image
docker pull ghcr.io/ls3-lab/querygym:latest

# Run interactive shell
docker run -it --rm \
  --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  ghcr.io/ls3-lab/querygym:latest

# Or start Jupyter notebook
docker run -it --rm \
  --gpus all \
  -p 8888:8888 \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/notebooks:/workspace/notebooks \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  ghcr.io/ls3-lab/querygym:latest \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
# Open http://localhost:8888
```

### CPU Version (Lightweight)

```bash
# Pull the image
docker pull ghcr.io/ls3-lab/querygym:cpu

# Run interactive shell (no --gpus flag needed)
docker run -it --rm \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  ghcr.io/ls3-lab/querygym:cpu
```

## 📁 Directory Structure

The container has the following structure:

```
/workspace/
├── data/          # Mount your datasets here
├── outputs/       # Reformulated queries and results
├── notebooks/     # Jupyter notebooks
└── examples/      # Example scripts (read-only)
```

## 💡 Usage Examples

### Example 1: Run a Query Reformulation Script

```bash
# Create a script on your host
cat > my_experiment.py << 'EOF'
import querygym as qg

# Load queries
queries = qg.load_queries("data/queries.tsv")

# Create reformulator
reformulator = qg.create_reformulator("genqr_ensemble", model="gpt-4")

# Reformulate
results = reformulator.reformulate_batch(queries)

# Save results
qg.DataLoader.save_queries(
    [qg.QueryItem(r.qid, r.reformulated) for r in results],
    "outputs/reformulated.tsv"
)
EOF

# Run in container
docker run -it --rm \
  --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/my_experiment.py:/workspace/my_experiment.py \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  ghcr.io/ls3-lab/querygym:latest \
  python my_experiment.py
```

### Example 2: Interactive Development

```bash
# Start interactive shell
docker run -it --rm \
  --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  ghcr.io/ls3-lab/querygym:latest

# Inside container
python
>>> import querygym as qg
>>> queries = qg.load_queries("data/queries.tsv")
>>> reformulator = qg.create_reformulator("genqr")
>>> result = reformulator.reformulate(queries[0])
>>> print(result.reformulated)
```

### Example 3: Using PySerini for Retrieval

```bash
docker run -it --rm \
  --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/indexes:/workspace/indexes \
  ghcr.io/ls3-lab/querygym:latest

# Inside container
python -c "
from pyserini.search.lucene import LuceneSearcher
searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
hits = searcher.search('what is covid-19', k=10)
for hit in hits:
    print(f'{hit.docid}: {hit.score}')
"
```

### Example 4: Jupyter Notebook Development

```bash
# Start Jupyter server with GPU support
docker run -it --rm \
  --gpus all \
  -p 8888:8888 \
  -v $(pwd)/notebooks:/workspace/notebooks \
  -v $(pwd)/data:/workspace/data \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  ghcr.io/ls3-lab/querygym:latest \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''

# Open http://localhost:8888 in your browser
```

### Example 5: Verify GPU Support

```bash
# Run with GPU support
docker run -it --rm \
  --gpus all \
  ghcr.io/ls3-lab/querygym:latest \
  python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'GPU count: {torch.cuda.device_count()}')
"
```

## 🔄 Image Updates

Images are automatically built and published when:
- **New releases** are published on GitHub
- **Dockerfiles are updated** on the main branch
- **Manual workflow dispatch** is triggered

You can always get the latest version:
```bash
docker pull ghcr.io/ls3-lab/querygym:latest
docker pull ghcr.io/ls3-lab/querygym:cpu
```

## 🔧 Environment Variables

Set these environment variables for API access:

```bash
# OpenAI API
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional

# Pass to container
docker run -it --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e OPENAI_BASE_URL=$OPENAI_BASE_URL \
  querygym:latest
```

Or use a `.env` file with Docker Compose:

```bash
# Create .env file
cat > .env << EOF
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
EOF

# Docker Compose will automatically load it
docker compose run --rm querygym
```

## 📦 Installed Packages

### GPU Image (Default - `querygym:latest`)
- Python 3.10
- querygym (with all optional dependencies)
- PySerini
- PyTorch with CUDA 12.1
- Jupyter/IPython
- Transformers
- Sentence-Transformers
- OpenJDK 21 (for PySerini)
- NVIDIA CUDA runtime

### CPU Image (Lightweight - `querygym:cpu`)
- Python 3.10
- querygym (with all optional dependencies)
- PySerini
- PyTorch (CPU-only)
- Jupyter/IPython
- OpenJDK 21 (for PySerini)

## 🛠️ Customization

### Add Additional Dependencies

Create a custom Dockerfile:

```dockerfile
FROM querygym:latest

# Install additional packages
RUN pip install --no-cache-dir \
    pandas \
    matplotlib \
    seaborn

# Or install from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

## 🐛 Troubleshooting

### Issue: Permission Denied

```bash
# Run as current user
docker run -it --rm \
  -u $(id -u):$(id -g) \
  -v $(pwd)/data:/workspace/data \
  querygym:latest
```

### Issue: Out of Memory

```bash
# Increase Docker memory limit
docker run -it --rm \
  --memory=8g \
  --memory-swap=8g \
  querygym:latest
```

### Issue: GPU Not Detected

```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If not working, install nvidia-container-toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Issue: Slow Build

```bash
# Use BuildKit for faster builds (already enabled in Makefile)
DOCKER_BUILDKIT=1 docker build -t querygym:latest .

# Use cache from previous builds
docker build --cache-from querygym:latest -t querygym:latest .
```

## 📊 Image Sizes

| Image | Tag | Compressed | Uncompressed | Build Time |
|-------|-----|-----------|--------------|------------|
| GPU (default) | `latest` | ~2 GB | ~6 GB | ~10-15 min |
| CPU (lightweight) | `cpu` | ~800 MB | ~2.5 GB | ~5-10 min |

## 🔒 Security Best Practices

1. **Don't run as root in production:**
   ```bash
   docker run -u 1000:1000 querygym:latest
   ```

2. **Use read-only volumes when possible:**
   ```bash
   docker run -v $(pwd)/examples:/workspace/examples:ro querygym:latest
   ```

3. **Don't hardcode API keys:**
   ```bash
   # Use environment variables or .env files
   docker run -e OPENAI_API_KEY=$OPENAI_API_KEY querygym:latest
   ```

4. **Scan images for vulnerabilities:**
   ```bash
   docker scan querygym:latest
   ```

## 📝 Complete Workflow Example

```bash
# 1. Prepare your data
mkdir -p data outputs notebooks
cp your_queries.tsv data/

# 2. Set API key
export OPENAI_API_KEY="sk-..."

# 3. Build image
make build

# 4. Run experiment
docker run -it --rm \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  querygym:latest \
  python -c "
import querygym as qg

queries = qg.load_queries('data/your_queries.tsv')
reformulator = qg.create_reformulator('genqr_ensemble', model='gpt-4')
results = reformulator.reformulate_batch(queries)

qg.DataLoader.save_queries(
    [qg.QueryItem(r.qid, r.reformulated) for r in results],
    'outputs/reformulated.tsv'
)
print(f'Reformulated {len(results)} queries!')
"

# 5. Results are now in outputs/reformulated.tsv on your host
cat outputs/reformulated.tsv
```

## 🤝 Contributing

To contribute Docker improvements:

1. Test your changes locally
2. Update this documentation
3. Submit a pull request

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [QueryGym Documentation](https://querygym.readthedocs.io/)
- [Python Docker Best Practices](https://docs.docker.com/language/python/build-images/)
