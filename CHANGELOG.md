# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CUDA-accelerated visualization support via `viz-cuda` optional dependency group
  - Includes cuML for GPU-accelerated UMAP
  - Includes CuPy for GPU-accelerated NumPy operations
  - Configured NVIDIA PyPI index in `[tool.uv]`
- Sentence-level genre assignment visualization
  - Loads `all_genres.parquet` for actual bootstrap-assigned genres
  - `--color-by` parameter to switch between genre, cluster, or treebank coloring
  - Interactive tooltips showing all metadata
- Enhanced diagnostic logging and reporting
  - Cross-lingual genre assignment report showing genres spanning multiple languages
  - Bootstrap schedule summary table showing progression
  - Label assignment logging with top-3 similarity scores
  - Low-confidence warnings for uncertain assignments
- Configuration support for excluding specific treebanks via `exclude_treebanks`
- Helper function `apply_treebank_exclusions()` for treebank filtering
- Export of `all_genres.parquet` from `label` command
- Display functions for bootstrap schedule and evaluation results
- `--overwrite` flag for `embed` command to force regeneration of cached embeddings

### Changed
- Visualization now uses sentence-level genres from `all_genres.parquet` instead of treebank metadata
- Evaluation uses ALL available splits (train, dev, test) for maximum genre coverage
  - Removed train-only treebank restrictions
  - Processes all splits per treebank to maximize data usage
- Enhanced scheduler logging with genre combination distribution
- Improved bootstrapper logging with similarity scores and confidence reporting
- Updated genre mappings and metadata patterns
- Improved configuration files (default.yaml, 2.17-local.yaml)

### Fixed
- Missing `json` import in CLI visualization command
- Visualization showing comma-separated genres instead of single-genre assignments

## [0.1.0] - 2026-01-17

### Added
- Initial release
- Bootstrap labeling algorithm
- GMM clustering at treebank and language levels
- Cross-validation evaluation metrics
- Parquet output format mirroring UD structure
