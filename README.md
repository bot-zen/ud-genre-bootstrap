# UD Genre Bootstrap

Modern re-implementation of genre classification for Universal Dependencies using bootstrapping and clustering.

## Attribution

This project is a modern re-implementation of the genre classification methodology from:

**"How Universal is Genre in Universal Dependencies?"**
Max Müller-Eberstein, Rob van der Goot, and Barbara Plank
SyntaxFest 2021
[Paper (ACL Anthology)](https://aclanthology.org/2021.tlt-1.7.pdf) | [Original Code](https://github.com/personads/ud-genre)

The core bootstrapping algorithm and clustering approach are based on their work. This implementation modernizes the codebase with:
- HuggingFace Datasets integration
- Modern embedding models (XLM-RoBERTa, mBERT, etc.)
- Public dataset releases
- Improved evaluation metrics
- Better configurability

## Overview

This tool automatically classifies Universal Dependencies sentences into genres using:

1. **Embedding**: Convert sentences to semantic vectors using multilingual transformers
2. **Clustering**: Group sentences within each treebank using GMM
3. **Bootstrapping**: Progressively label clusters by comparing to known single-genre treebanks
4. **Evaluation**: Cross-validate results against metadata

## Installation

### Production Use

```bash
# Using uv (recommended)
uv pip install git+https://github.com/bot-zen/ud-genre-bootstrap

# Or using pip
pip install git+https://github.com/bot-zen/ud-genre-bootstrap
```

### Development Setup

```bash
# Clone both repos side-by-side
cd /your/workspace/
git clone https://github.com/bot-zen/ud-hf-parquet-tools
git clone https://github.com/bot-zen/ud-genre-bootstrap
cd ud-genre-bootstrap/

# Create virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install ud-hf-parquet-tools in editable mode
uv pip install -e ../ud-hf-parquet-tools

# Install ud-genre-bootstrap in editable mode with dev dependencies
uv pip install -e ".[dev]"
```

### Optional: CUDA-Accelerated Visualization

For GPU-accelerated UMAP visualization (requires NVIDIA GPU + CUDA):

```bash
# Install with CUDA support
uv pip install -e ".[viz-cuda]"

# Or for CPU-only visualization
uv pip install -e ".[viz]"
```

The `viz-cuda` extra includes cuML and CuPy for significantly faster dimensionality reduction on large datasets.

## Quick Start

### Using the CLI

```bash
# Run full pipeline with default config
ud-genre-bootstrap run --config configs/default.yaml

# Or specify UD version and model
ud-genre-bootstrap run \
    --ud-version 2.15 \
    --model xlm-roberta-base \
    --output output/v2.15/

# Run individual steps
ud-genre-bootstrap embed --model xlm-roberta-base --output embeddings/
ud-genre-bootstrap cluster --embeddings embeddings/ --output clusters/
ud-genre-bootstrap label --clusters clusters/ --output genres/
ud-genre-bootstrap evaluate --genres genres/ --output evaluation/

# Efficient workflow: cluster command saves state, label command loads it
ud-genre-bootstrap cluster --config config.yaml --treebank en_ewt
# Creates: output/clusters/cluster_state.pkl (contains clusters + embeddings)
ud-genre-bootstrap label --config config.yaml --clusters output/clusters/
# Loads pre-computed state, skips expensive re-clustering

# Incremental re-clustering: update only specific treebanks
# Useful when you change genre mappings for specific treebanks
ud-genre-bootstrap cluster --config config.yaml --clusters output/clusters/ --treebank en_ewt
# Loads existing clusters, re-clusters only en_ewt, keeps others unchanged

# Treebank filtering via config or CLI
# Option 1: Set include_treebanks in config.yaml
ud-genre-bootstrap cluster --config config.yaml
# Uses treebanks specified in config

# Option 2: Override with CLI flag
ud-genre-bootstrap cluster --config config.yaml --treebank fr_gsd,es_gsd
# CLI flag overrides config

# Force regenerate embeddings (overwrite cache)
ud-genre-bootstrap embed --config config.yaml --overwrite

# GPU-accelerated K-Means clustering (recommended for GPU)
# Set method: "kmeans" in config.yaml for GPU acceleration
ud-genre-bootstrap cluster --config config.yaml --use-gpu

# Note: GMM is CPU-only (cuML doesn't support Gaussian Mixture Models)
# Use K-Means for GPU acceleration
```

### Using Python API

```python
from ud_genre_bootstrap import GenreBootstrapper

# Initialize with configuration
bootstrapper = GenreBootstrapper(
    ud_version="2.15",
    model="xlm-roberta-base",
    clustering_level="treebank"
)

# Run full pipeline
results = bootstrapper.fit()

# Access results
print(f"Resolved: {results['resolution_rate']:.2%}")
print(f"Accuracy: {results['accuracy']:.2%}")

# Export to HuggingFace
bootstrapper.push_to_hub("commul/ud-genres", revision="2.15")
```

### Visualization

Visualize cluster assignments and genre labels in 2D using UMAP or t-SNE:

```bash
# Visualize with sentence-level genre labels (after running 'label' command)
ud-genre-bootstrap visualize-clusters \
    --clusters output/ud-v2.15/genres/clusters \
    --config configs/default.yaml \
    --color-by genre

# Color by cluster assignments instead
ud-genre-bootstrap visualize-clusters \
    --clusters output/ud-v2.15/genres/clusters \
    --config configs/default.yaml \
    --color-by cluster

# Use GPU-accelerated UMAP (requires viz-cuda installation)
ud-genre-bootstrap visualize-clusters \
    --clusters output/ud-v2.15/genres/clusters \
    --config configs/default.yaml \
    --use-gpu

# Filter to specific treebanks
ud-genre-bootstrap visualize-clusters \
    --clusters output/ud-v2.15/genres/clusters \
    --config configs/default.yaml \
    --treebank en_ewt,de_gsd
```

The visualization creates an interactive HTML plot showing:
- **Genre coloring**: Each point colored by its assigned genre (default)
- **Cluster coloring**: Points colored by cluster ID
- **Treebank coloring**: Points colored by source treebank
- **Hover information**: Shows sentence ID, genre, cluster, and treebank for each point

**Note**: The visualization uses sentence-level genre assignments from `all_genres.parquet` (created by the `label` command). If this file doesn't exist, it falls back to treebank-level metadata.

## Configuration

### Data Sources

The `ud_source` configuration accepts:
- **HuggingFace**: `hf://commul/universal_dependencies` - Load from HuggingFace datasets
- **Local files**: `local:///absolute/path/to/UD_repos/` - Load from local CoNLL-U files (absolute path)
- **Local files**: `local://../relative/path/to/UD_repos/` - Load from local CoNLL-U files (relative path)

Relative paths are resolved relative to the current working directory.

Example `config.yaml`:

```yaml
ud_version: "2.15"
ud_source: "hf://commul/universal_dependencies"

# Optional: Only process specific treebanks
include_treebanks:
  - "en_ewt"
  - "de_gsd"
  - "fr_gsd"

embeddings:
  model: "xlm-roberta-base"
  pooling: "mean"
  batch_size: 64
  device: "cuda"

clustering:
  method: "kmeans"  # "kmeans" (GPU-accelerated) or "gmm" (CPU-only)
  level: "treebank"
  seed: 42
  device: "auto"  # Use GPU if available, or "cuda"/"cpu" to force
  # Note: K-Means supports GPU via cuML, GMM is CPU-only

bootstrapping:
  min_confidence: 0.8
  max_iterations: 10
  fail_on_incomplete: false

genre_extraction:
  # Optional: custom genre mappings and patterns
  mapping_path: "configs/genre_mappings.json"
  patterns_path: "configs/metadata_patterns.json"
  # Optional: override default canonical genre set
  canonical_genres: ["news", "wiki", "fiction", "blog", "legal", "spoken"]

evaluation:
  enabled: true
  metadata_validation:
    method: "kfold"
    k: 5
    stratify_by: "genre"
    group_by: "language"
    min_genre_sentences: 100  # Minimum sentences per genre for evaluation

output:
  genres_path: "output/ud-v2.15/genres/"
  embeddings_hf_repo: "commul/ud-embeddings-xlm-roberta-base"
  push_to_hub: true

# Optional: Exclude specific treebanks from processing
exclude_treebanks:
  - "en_lines"  # Example: exclude problematic treebanks
  - "ar_nyuad"
```

**Genre Extraction Configuration**: See [Genre Pattern Configuration](docs/GENRE_PATTERNS.md) for detailed documentation on pattern-based genre extraction from sentence metadata.

## Output Format

### Genre Classifications

The bootstrap labeling produces two main output files:

#### 1. Sentence-Level Genre Assignments (`all_genres.parquet`)

Contains one genre per sentence after GMM+L bootstrap labeling:

```python
import pandas as pd
df = pd.read_parquet("output/ud-v2.15/genres/all_genres.parquet")

# Columns:
# - sent_id: Sentence identifier (e.g., "en_ewt-ud-train-00001")
# - genre: Single genre label assigned by bootstrap (e.g., "news", "wiki")
# - confidence: Cosine similarity to genre centroid [0-1]
# - method: "bootstrap-labeled" or "bootstrap-inferred" (low confidence)

# Example usage
print(df.head())
#                    sent_id  genre  confidence          method
# 0  en_ewt-ud-train-00001   news      0.8523  bootstrap-labeled
# 1  en_ewt-ud-train-00002   news      0.7891  bootstrap-labeled
# 2  de_gsd-ud-test-00042    wiki      0.4321  bootstrap-inferred
```

#### 2. Cluster Assignments (`clusters/cluster_assignments.parquet`)

Contains cluster IDs before genre labeling:

```python
df_clusters = pd.read_parquet("output/ud-v2.15/genres/clusters/cluster_assignments.parquet")

# Columns:
# - treebank: Treebank code (e.g., "en_ewt")
# - split: train/dev/test
# - sent_id: Sentence identifier
# - cluster_id: Cluster number (0 to n_genres-1)
# - confidence: GMM assignment confidence

# Join with genre assignments
df_combined = df_clusters.merge(df, on='sent_id')
```

#### 3. Join with UD Data

```python
# Load genre predictions with original UD data
from datasets import load_dataset
ud = load_dataset("commul/universal_dependencies", "en_ewt", split="train", revision="2.15")
ud_with_genres = ud.to_pandas().merge(df, on="sent_id")
```

### Embeddings

HuggingFace datasets mirroring UD structure:

```python
from datasets import load_dataset

# Load embeddings for specific treebank
embeddings = load_dataset(
    "commul/ud-embeddings-xlm-roberta-base",
    "en_ewt",
    revision="2.15",
    split="train"
)
# Returns: {sent_id, embedding}
```

## Evaluation

The tool provides multiple evaluation metrics:

- **Metadata Validation**: k-fold cross-validation against known genres
- **Cluster Quality**: Silhouette score, Calinski-Harabasz index
- **Bootstrap Statistics**: Resolution rate, convergence metrics, confidence distribution

Results are saved in `output/evaluation/`:
- `cv_results.json` - Cross-validation accuracy, F1, precision, recall
- `confusion_matrix.png` - Visual confusion matrix
- `cluster_quality.json` - Unsupervised cluster metrics
- `convergence_report.json` - Bootstrap statistics

### Cross-Validation

The `evaluate` command performs k-fold cross-validation using **all available splits** (train, dev, test) from treebanks with sentence-level genre metadata:

```bash
ud-genre-bootstrap evaluate \
    --config configs/default.yaml \
    --n-folds 5 \
    --group-by language
```

**Key features**:
- Uses all splits since we only use text content and genre metadata (not UD annotations)
- Creates virtual single-genre splits from each treebank split
- Supports stratification by genre and grouping by language/treebank
- Reports cross-lingual genre identification accuracy

## Diagnostic Output

The tool provides detailed diagnostic logging to help verify the GMM+L algorithm is working correctly:

### Bootstrap Schedule Summary

Shows the progression of genre discovery during bootstrapping:

```
Environment 1: 5 known, 12 predictable, 8 disjunct
  ✓ New known genres: fiction, legal
  → Can predict these combinations: ('news', 'blog'), ('wiki', 'reviews'), ...
  ✗ Still disjunct: ('spoken', 'medical'), ...
```

### Cross-Lingual Genre Assignment Report

Critical diagnostic showing whether genres are identified across languages:

```
================================================================================
CROSS-LINGUAL GENRE ASSIGNMENT REPORT
================================================================================

Genre: NEWS
  Found in 5 language(s), 12 cluster(s), 450 sentence(s)
    en: 3 cluster(s), 120 sent(s), avg_conf=0.850
    de: 2 cluster(s), 80 sent(s), avg_conf=0.820
    fr: 4 cluster(s), 150 sent(s), avg_conf=0.840

CROSS-LINGUAL CONSISTENCY CHECK:
✓ Found 8 genre(s) spanning multiple languages:
  - news: en, de, fr, es, it
  - fiction: en, fr, de
  - wiki: en, de, fr, it, es, fi
```

**Warning signs**:
- If no genres span multiple languages, clustering may be separating by language rather than genre
- Low confidence scores indicate weak genre signals
- Many "bootstrap-inferred" assignments suggest the model is uncertain

### Label Assignment Details

Shows similarity scores for each cluster assignment:

```
Cluster c2 (150 sents) → news (conf=0.850, top3: news:0.850, blog:0.720, reviews:0.650)
⚠ Cluster c5 in de_gsd:test → wiki (LOW conf=0.432, top3: wiki:0.432, news:0.428, fiction:0.401)
```

This helps identify:
- Which genres are easily distinguishable
- Which genres are confusable
- Where the model lacks confidence

## Project Structure

```
ud-genre-bootstrap/
├── src/ud_genre_bootstrap/
│   ├── embeddings/       # Sentence embedding generation
│   ├── clustering/       # GMM clustering
│   ├── bootstrapping/    # Bootstrap labeling algorithm
│   ├── evaluation/       # Evaluation metrics
│   ├── utils/           # Helpers and data loading
│   └── cli.py           # Command-line interface
├── tests/
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── scripts/             # Utility scripts
├── configs/             # Configuration examples
└── docs/               # Documentation
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ud_genre_bootstrap --cov-report=html

# Run specific test file
pytest tests/unit/test_embeddings.py
```

### Code Quality

```bash
# Format code
black src/ tests/
ruff check --fix src/ tests/

# Type checking
mypy src/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## Citation

If you use this tool in your research, please cite both this implementation and the original paper:

```bibtex
@inproceedings{muller-eberstein-etal-2021-universal,
    title = "How Universal is Genre in {U}niversal {D}ependencies?",
    author = "M{\"u}ller-Eberstein, Max  and
      van der Goot, Rob  and
      Plank, Barbara",
    booktitle = "Proceedings of the 20th International Workshop on Treebanks and Linguistic Theories (TLT, SyntaxFest 2021)",
    month = dec,
    year = "2021",
    address = "Sofia, Bulgaria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.tlt-1.7",
    pages = "59--72",
}
```

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

For questions or issues, please open an issue on GitHub.
