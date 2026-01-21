# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Incremental re-clustering for selective treebank updates**
  - `cluster` command now accepts `--clusters` option to load existing state
  - Combined with `--treebank`, allows re-clustering specific treebanks only
  - Keeps all other clusters unchanged and merges results
  - Useful for updating clusters after changing genre mappings
  - Example: `cluster --clusters output/ --treebank en_ewt` (updates only en_ewt)
- **K-Means clustering with GPU acceleration via cuML**
  - New `KMeansClusterer` class with GPU support
  - Configurable via `clustering.method: "kmeans"` in config
  - Automatic GPU detection and CPU fallback
  - Significantly faster than GMM on GPU (2-10x speedup)
  - Uses distance-based soft assignments (similar to GMM probabilities)
  - Recommended for large datasets and GPU environments
- Cluster state save/load functionality for efficient label command execution
  - `cluster` command now saves full cluster state to `cluster_state.pkl`
  - `label` command accepts `--clusters` option to load pre-computed state
  - Skips expensive embedding generation and re-clustering when loading state
  - Significantly speeds up iterative workflow (cluster once, label multiple times)
- Progress output for clustering pipeline
  - Shows [X/Y] progress for embedding generation, clustering, and cluster embeddings
  - Displays sentence counts, genre counts, and genre names during processing
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
- Confusion matrix visualization in `evaluate` command
  - Terminal display as Rich table with highlighted diagonal
  - PNG heatmap saved to `evaluation/confusion_matrix.png`
  - Aggregates predictions across all cross-validation folds
- Configurable canonical genre set via `genre_extraction.canonical_genres`
  - Override default UD genre taxonomy with custom genre set
  - Falls back to default set if not specified
- GPU clustering infrastructure
  - `--use-gpu` flag for `cluster` command
  - `device` configuration in `clustering` section ("auto", "cuda", "cpu")
  - K-Means now fully GPU-accelerated via cuML
  - GMM remains CPU-only (cuML limitation)

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
