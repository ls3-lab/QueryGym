# Docker Setup Guide

## For End Users (Quick Start)

**Just use the pre-built images - no building required!**

### Option 1: Docker Compose (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/ls3-lab/QueryGym.git
cd QueryGym

# 2. Set your API key
export OPENAI_API_KEY="sk-..."

# 3. Run with Docker Compose
docker compose run --rm querygym

# Or start Jupyter
docker compose up jupyter
```

### Option 2: Direct Docker Run

```bash
docker pull ghcr.io/ls3-lab/querygym:latest
docker run -it --rm --gpus all ghcr.io/ls3-lab/querygym:latest
```

See the [full Docker guide](https://querygym.readthedocs.io/en/latest/user-guide/docker/) for more examples.

---

## For Developers (Building Locally)

If you need to build images locally for development:

```bash
# Build GPU image
make build

# Build CPU image  
make build-cpu

# Test builds
make test

# Clean up
make clean
```

The `Makefile` is for **local development only**. End users should use pre-built images.

---

## Files Overview

- **`compose.yml`** / **`docker-compose.yml`** - Uses pre-built GHCR images (for end users)
- **`Dockerfile`** - GPU image definition (for building)
- **`Dockerfile.cpu`** - CPU image definition (for building)
- **`.dockerignore`** - Build optimization
- **`Makefile`** - Development commands (for building locally)
- **`examples/docker/`** - Ready-to-use examples with notebooks

---

## Images

- **GPU (default):** `ghcr.io/ls3-lab/querygym:latest`
- **CPU (lightweight):** `ghcr.io/ls3-lab/querygym:cpu`

Images are automatically built and published on every release.
