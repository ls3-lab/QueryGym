# Changelog

All notable changes to querygym will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2025-11-17

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

## [0.1.0] - 2025-01-XX

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

[0.1.1]: https://github.com/radinhamidi/queryGym/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/radinhamidi/queryGym/releases/tag/v0.1.0
