# Changelog

All notable changes to querygym will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] - 2025-11-21

### Added
- Docker support with pre-built images on GitHub Container Registry (GHCR)
  - GPU-enabled image (`ghcr.io/ls3-lab/querygym:latest`)
  - CPU-only image (`ghcr.io/ls3-lab/querygym:cpu`)
  - Multi-platform support (linux/amd64, linux/arm64 for CPU)
- Docker Compose configuration for easy setup
- Comprehensive Docker documentation and examples
- GitHub Actions workflow for automated Docker image building and publishing
- Example notebooks and scripts in `examples/` directory
  - Code snippets for quick reference
  - Docker examples with Jupyter notebooks
  - QueryGym + Pyserini integration examples
- DOCKER_SETUP.md quick reference guide
- Makefile for local Docker development

### Changed
- **License changed from MIT to Apache License 2.0**
- Reorganized examples into three categories: snippets, docker, querygym_pyserini
- Updated to OpenJDK 21 (from 17) in Docker images
- Migrated to Docker Compose V2 syntax (`docker compose` instead of `docker-compose`)
- Improved Docker image tagging strategy (separate tags for GPU and CPU)
- Enhanced README with Docker installation options and citation information
- Updated all documentation to use modern Docker Compose commands

### Fixed
- Docker build issues with OpenJDK availability in Debian Trixie
- Docker Compose tag conflicts between GPU and CPU images
- Inconsistent command examples across documentation

## [0.1.3] - 2025-11-19

### Added
- Apache License 2.0
- Citation information with arXiv paper reference in README

### Changed
- Package metadata updated with Apache 2.0 license

## [0.1.2] - 2025-11-18

### Changed
- Updated author information with email addresses for Mert Incesu and Negar Arabzadeh
- Version bump for metadata updates

## [0.1.1] - 2025-11-18

### Added
- Added `tqdm` as core dependency for progress bars
- Complete Read the Docs documentation setup
- MkDocs configuration with Material theme
- Comprehensive API reference documentation
- GitHub Actions workflow for automated PyPI publishing

### Changed
- Updated package name from `queryGym` to `querygym` (lowercase)
- Updated author information in package metadata
- Improved dependency management

### Fixed
- Fixed missing `tqdm` dependency in core package

## [0.1.0] - 2025-11-17

### Added
- Initial release of querygym
- Core reformulation framework
- Eight query reformulation methods:
  - GenQR (generic keyword expansion)
  - GenQR Ensemble
  - Query2Doc
  - QA Expand
  - MuGI
  - LameR
  - Query2E
  - CSQE
- Prompt bank system with YAML configuration
- DataLoader for queries, qrels, and contexts
- BEIR and MS MARCO format loaders
- Pyserini and PyTerrier adapters
- CLI interface with `querygym` command
- OpenAI-compatible LLM client
- Comprehensive test suite
- Example scripts and documentation

### Features
- Dependency-free core data loading
- Optional integrations (BEIR, Pyserini, HuggingFace)
- Flexible LLM client supporting any OpenAI-compatible API
- Batch processing with progress bars
- Context-based reformulation support
- Export to TSV/JSONL formats
- Script generation for Pyserini + trec_eval

[Unreleased]: https://github.com/ls3-lab/QueryGym/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/ls3-lab/QueryGym/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/ls3-lab/QueryGym/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/ls3-lab/QueryGym/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ls3-lab/QueryGym/releases/tag/v0.1.0
