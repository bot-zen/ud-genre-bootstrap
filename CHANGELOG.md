# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- Core module skeletons for embeddings, clustering, bootstrapping, and evaluation
- CLI interface with Typer
- Configuration system with YAML support
- HuggingFace integration for datasets and embeddings
- Apache 2.0 license
- Documentation and README

### Changed
- Modernized from 2021 paper implementation
- Switched to HuggingFace Datasets ecosystem
- Updated to Python 3.12
- Added support for multiple embedding models

## [0.1.0] - 2026-01-17

### Added
- Initial release
- Bootstrap labeling algorithm
- GMM clustering at treebank and language levels
- Cross-validation evaluation metrics
- Parquet output format mirroring UD structure
