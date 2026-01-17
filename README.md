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

## Configuration

Example `config.yaml`:

```yaml
ud_version: "2.15"
ud_source: "hf://commul/universal_dependencies"

embeddings:
  model: "xlm-roberta-base"
  pooling: "mean"
  batch_size: 64
  device: "cuda"

clustering:
  method: "gmm"
  level: "treebank"
  seed: 42

bootstrapping:
  min_confidence: 0.8
  max_iterations: 10
  fail_on_incomplete: false

evaluation:
  enabled: true
  metadata_validation:
    method: "kfold"
    k: 5
    stratify_by: "genre"
    group_by: "language"

output:
  genres_path: "output/ud-v2.15/genres/"
  embeddings_hf_repo: "commul/ud-embeddings-xlm-roberta-base"
  push_to_hub: true
```

## Output Format

### Genre Classifications

Parquet files mirror the UD structure:

```python
# Load genre predictions
import pandas as pd
df = pd.read_parquet("output/ud-v2.15/genres/UD_English-EWT/en_ewt-ud-train.parquet")

# Columns:
# - sent_id: Join key (e.g., "en_ewt-ud-train#1")
# - genre: Predicted genre label or None
# - confidence: Bootstrap confidence [0-1]
# - method: "metadata", "bootstrap-labeled", "bootstrap-inferred", "bootstrap-failed"

# Join with original UD data
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
