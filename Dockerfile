# Default Dockerfile for querygym (GPU-enabled)
# This is the recommended version with GPU support
# For CPU-only (lightweight), use Dockerfile.cpu
# Base: NVIDIA CUDA official image for GPU support

# Use NVIDIA CUDA base image (Ubuntu-based)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

LABEL maintainer="querygym team <radin.h@gmail.com>"
LABEL description="querygym: LLM-based Query Reformulation Toolkit with GPU support"
LABEL org.opencontainers.image.source="https://github.com/ls3-lab/QueryGym"

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    openjdk-21-jre-headless \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
# CUDA 12.1 compatible wheels
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install querygym with all dependencies
RUN pip install --no-cache-dir \
    querygym[all] \
    pyserini \
    jupyter \
    ipython \
    transformers \
    sentence-transformers

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
    CUDA_VISIBLE_DEVICES=0

# Create working directory
WORKDIR /workspace

# Create directories
RUN mkdir -p /workspace/data /workspace/outputs /workspace/notebooks

# Set up non-root user
RUN useradd -m -u 1000 querygym && \
    chown -R querygym:querygym /workspace

USER querygym

# Expose Jupyter port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import querygym; import torch; assert torch.cuda.is_available()" || exit 1

CMD ["/bin/bash"]
