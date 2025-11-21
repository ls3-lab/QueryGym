.PHONY: help build build-cpu build-all test clean

# Makefile for QueryGym Docker Development
# This is for developers/contributors who need to build images locally
# 
# End users should use pre-built images:
#   docker pull ghcr.io/radinhamidi/querygym:latest
#   docker-compose up

# Default target
help:
	@echo "QueryGym Docker Development Commands"
	@echo "====================================="
	@echo ""
	@echo "⚠️  NOTE: This is for developers building images locally"
	@echo "   End users should use: docker-compose up"
	@echo ""
	@echo "Build Commands:"
	@echo "  make build          Build GPU image locally"
	@echo "  make build-cpu      Build CPU image locally"
	@echo "  make build-all      Build both images"
	@echo "  make test           Test the built images"
	@echo "  make clean          Remove built images"
	@echo ""

# Build targets
build:
	@echo "Building GPU image locally..."
	@echo "Note: This may take 10-15 minutes"
	DOCKER_BUILDKIT=1 docker build -t querygym-dev:latest -f Dockerfile .
	@echo "✓ Built: querygym-dev:latest"

build-cpu:
	@echo "Building CPU image locally..."
	@echo "Note: This may take 5-10 minutes"
	DOCKER_BUILDKIT=1 docker build -t querygym-dev:cpu -f Dockerfile.cpu .
	@echo "✓ Built: querygym-dev:cpu"

build-all: build build-cpu
	@echo "✓ All images built successfully"

# Test the built images
test:
	@echo "Testing locally built images..."
	@echo ""
	@echo "Testing querygym-dev:latest (GPU)..."
	docker run --rm querygym-dev:latest python -c "import querygym; print(f'✓ querygym {querygym.__version__}')"
	docker run --rm querygym-dev:latest python -c "import pyserini; print('✓ PySerini OK')"
	docker run --rm querygym-dev:latest python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
	@echo ""
	@echo "Testing querygym-dev:cpu..."
	docker run --rm querygym-dev:cpu python -c "import querygym; print(f'✓ querygym {querygym.__version__}')"
	docker run --rm querygym-dev:cpu python -c "import pyserini; print('✓ PySerini OK')"
	docker run --rm querygym-dev:cpu python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
	@echo ""
	@echo "✓ All tests passed!"

# Clean up locally built images
clean:
	@echo "Removing locally built images..."
	docker rmi querygym-dev:latest querygym-dev:cpu 2>/dev/null || true
	@echo "✓ Cleanup complete!"
