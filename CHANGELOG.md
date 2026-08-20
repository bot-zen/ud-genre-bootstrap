# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Release identity, registry, and Git-backed publishing workflow for versioned
  UD genre artifacts.
- Shared `full-ud` release profile and `full-ud-v1.0.0` release matrix for UD
  2.7 through UD 2.18, replacing committed full configs per UD version.
- **Cluster quality metrics computation**
  - Silhouette score, Calinski-Harabasz index, Davies-Bouldin index now computed during clustering
  - Metrics saved in `cluster_statistics.json` for each treebank
  - Available for multi-genre treebanks (single-genre treebanks skipped as they have only 1 cluster)
  - Useful for evaluating clustering quality independently of bootstrap labeling
- **Pairwise cluster separation metrics**
  - Computes distances between all pairs of cluster centroids within each treebank
  - Shows which clusters are well-separated vs. similar
  - Includes mean, min, and max pairwise distances
  - Saved in `cluster_statistics.json` under `metrics.pairwise_distances`
- **Genre separation analysis**
  - New step in bootstrap pipeline: analyzes how separable different genres are in embedding space
  - Computes pairwise distances between genre centroids (e.g., "how far is 'news' from 'social'?")
  - Displays distance matrix showing all genre pairs
  - Reports closest and furthest genre pairs
  - Saves to `genre_separation_metrics.json` in output directory
  - Helps understand which genres are easily distinguishable vs. confusable
- **X-GENRE classifier evaluation**
  - New `evaluate-xgenre` command to compare bootstrap labels against X-GENRE predictions
  - Uses pre-trained multilingual genre classifier as independent ground truth
  - Only evaluates bootstrap-labeled sentences (excludes pre-existing metadata and single-genre treebanks)
  - Provides accuracy, per-genre precision/recall/F1, and confusion matrix
  - Customizable X-GENRE → UD genre mapping via `xgenre_evaluation.genre_mapping` config
  - Respects `include_treebanks` filtering for consistent comparisons
  - Saves detailed predictions, metrics (JSON), and confusion matrix visualization (PNG)
- **Treebank filtering via config file**
  - New `include_treebanks` config option to specify which treebanks to process
  - Works for cluster, label, and evaluate commands
  - CLI `--treebank` flag takes precedence over config
  - Useful for reproducible experiments on specific treebanks
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
- Default and release UD source repository moved from `commul/universal_dependencies`
  to `universal-dependencies/universal_dependencies`.
- UD 2.18 is the current default target in the synchronized `full-ud-v1.0.0`
  release train for `commul/ud_genre`.
- Transitional per-UD source refs such as `source/ud2.*`, `release/v1`, and
  `ud2.X-full-ud-v...` artifact identities have been replaced by
  `release/full-ud-v1`, `source/full-ud-v1.0.0`, and per-UD HF tags such as
  `artifact/full-ud-v1.0.0/ud2.18`.
- Visualization now uses sentence-level genres from `all_genres.parquet` instead of treebank metadata
- Evaluation uses ALL available splits (train, dev, test) for maximum genre coverage
  - Removed train-only treebank restrictions
  - Processes all splits per treebank to maximize data usage
- Enhanced scheduler logging with genre combination distribution
- Improved bootstrapper logging with similarity scores and confidence reporting
- Updated genre mappings and metadata patterns
- Improved configuration files (default.yaml, 2.17-local.yaml)

### Changed
- Refactored cluster results saving into shared `_save_cluster_results()` helper function
  - Both `cluster` and `run` commands now use the same code path
  - Eliminates code duplication and ensures consistent behavior
- `visualize-clusters` command now takes default paths from `--config`
  - `--clusters` is now optional and defaults to `{output.genres_path}/clusters/`
  - `--embeddings` is now optional and defaults to `{embeddings.cache_dir}`
  - `--use-gpu` is now optional and respects `clustering.device` from config (CLI flag overrides)
  - Makes visualization workflow simpler: just specify `--config` instead of all paths and flags

### Fixed
- Missing `json` import in CLI visualization command
- Visualization showing comma-separated genres instead of single-genre assignments
- `export_metadata_genres.py` script not respecting `include_treebanks` config option
- `run` command not saving `cluster_assignments.parquet` file (now saves to `output/clusters/`)
- Misleading documentation for `clustering.level` config option
  - Clarified that only `"treebank"` is currently implemented
  - Marked `"language"` and `"all"` as future options (not yet implemented)
- GMM clustering failure on single-genre treebanks with few sentences
  - Now skips clustering for single-genre treebanks (creates trivial single-cluster assignment)
  - Prevents numerical errors when n_components=1 with small datasets
  - More efficient: single-genre treebanks don't need clustering anyway

## [0.1.0] - 2026-01-17

### Added
- Initial release
- Bootstrap labeling algorithm
- GMM clustering at treebank and language levels
- Cross-validation evaluation metrics
- Parquet output format mirroring UD structure
