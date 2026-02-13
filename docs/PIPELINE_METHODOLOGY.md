# Pipeline Methodology: Bootstrap Genre Classification for Universal Dependencies

This document provides a comprehensive description of the bootstrap genre classification pipeline, including both the functional overview of the approach and implementation details suitable for reproducibility.

## 1. Functional Overview

### 1.1 Overall Goal

The pipeline addresses the problem of assigning genre labels to sentences in multilingual treebanks where genre metadata is either absent, inconsistent, or available only at the document level. The approach leverages treebanks with known single-genre content as reference points to progressively label multi-genre treebanks through an iterative bootstrapping procedure.

The core assumption is that sentences from the same genre exhibit similar distributional properties in a multilingual embedding space, regardless of their source language. This enables cross-lingual transfer of genre information from resource-rich to resource-poor treebanks.

### 1.2 Pipeline Stages

The pipeline consists of seven primary stages:

#### Stage 1: Embedding Generation

All sentences across all treebanks are encoded into a shared multilingual semantic space using a pre-trained transformer encoder. This stage produces dense vector representations that capture both linguistic and topical properties of the text.

**Input:** Raw sentence text from Universal Dependencies treebanks
**Output:** Dense embeddings (typically 768 or 1024 dimensions) for each sentence
**Purpose:** Transform heterogeneous text into a comparable geometric space

#### Stage 2: Clustering

Each treebank is clustered based on the expected number of genres indicated by its metadata. Crucially, when a treebank has multiple splits (train/dev/test), all splits are combined before clustering to ensure consistent genre structure across the entire treebank. The clustering results are then distributed back to individual splits.

For treebanks with sentence-level genre annotations, virtual splits are created by separating sentences according to their genre labels, effectively treating each genre subset as a distinct single-genre treebank. Virtual splits span all available splits of the treebank.

**Input:** Sentence embeddings grouped by treebank (combining all splits)
**Output:** Cluster assignments for each sentence
**Purpose:** Group sentences with similar distributional properties within each treebank, ensuring consistency across train/dev/test splits

#### Stage 3: Cluster Embedding Computation

For each cluster identified in Stage 2, a representative embedding is computed by averaging the embeddings of all sentences assigned to that cluster. These cluster embeddings serve as centroids in the semantic space.

**Input:** Cluster assignments and sentence embeddings
**Output:** Cluster centroid embeddings
**Purpose:** Create condensed representations for genre similarity computation

#### Stage 4: Bootstrap Schedule Creation

The scheduler analyzes the genre structure of the dataset and creates an ordered sequence of labeling environments. Each environment specifies which genres can be used as known references and which multi-genre combinations can be labeled in that iteration.

**Input:** Genre combinations from all treebanks
**Output:** Ordered sequence of labeling environments
**Purpose:** Determine the order in which to progressively resolve genre ambiguities

The scheduling algorithm follows Müller-Eberstein et al. (2021):
1. Initially, only single-genre treebanks (and virtual splits) are considered "known"
2. In each iteration, multi-genre treebanks containing at least one known genre become predictable
3. Cluster assignments refine the known genre set for subsequent iterations
4. The process continues until no further progress can be made

#### Stage 5: Single-Genre Labeling

All sentences from treebanks (or virtual splits) containing only a single genre are trivially labeled with that genre at 100% confidence. This establishes the initial set of reference data for the bootstrapping procedure.

**Input:** Single-genre treebank/split metadata
**Output:** Genre labels for all single-genre sentences
**Purpose:** Establish ground truth references for bootstrap initialization

#### Stage 6: Bootstrap Labeling

Following the schedule from Stage 4, multi-genre clusters are labeled by comparing their centroid embeddings to reference genre embeddings. Reference embeddings are computed from cluster centroids of all known single-genre sources, using configurable weighting (`sentence_count` by default, or `uniform`).

For each unlabeled cluster:
1. Compute cosine similarity to all known genre embeddings
2. Assign the genre with highest similarity
3. Apply uncertainty thresholds to set method flags (`bootstrap-labeled` vs `bootstrap-inferred`)
4. All sentences in the cluster receive the assigned genre label, confidence, and method tag

**Input:** Cluster embeddings, reference genre embeddings, bootstrap schedule
**Output:** Genre labels and confidence scores for all sentences
**Purpose:** Progressively label multi-genre content using iterative refinement

#### Stage 7: Export

The final labeled dataset is exported in formats compatible with the Universal Dependencies ecosystem, including sentence-level genre annotations and confidence metadata.

**Input:** Final genre labels and confidence scores
**Output:** Parquet files with sentence-level genre annotations
**Purpose:** Package results for downstream use

### 1.3 Key Design Principles

1. **Language Independence:** The multilingual embedding model ensures that genre signals transfer across languages without requiring language-specific resources.

2. **Progressive Resolution:** The bootstrap schedule ensures that simpler labeling decisions (e.g., treebanks with one known and one unknown genre) are resolved before more complex cases.

3. **Treebank-Level Consistency:** When a treebank has multiple splits (train/dev/test), all splits are combined before clustering to ensure consistent genre structure across the entire treebank, rather than clustering each split independently.

4. **Virtual Split Innovation:** When sentence-level metadata exists in multi-genre treebanks, the pipeline creates virtual single-genre splits that span all available splits, effectively increasing the available reference data for bootstrap initialization.

5. **High Coverage with Uncertainty Flags:** All clusters are labeled for maximum coverage. Configurable uncertainty thresholds mark assignments as high-confidence (`bootstrap-labeled`) or low-confidence (`bootstrap-inferred`) so downstream consumers can filter by method when stricter precision is needed.

## 2. Implementation Details

### 2.0 Shared Clustering Operations

To ensure consistency between production and evaluation, core clustering operations are implemented in a shared utility module (`clustering/clustering_utils.py`). This guarantees that both pipelines use identical logic for:

- **Grouping splits by treebank**: Combining train/dev/test splits of the same treebank
- **Combining embeddings**: Stacking embeddings from multiple splits with proper tracking
- **Creating virtual splits**: Extracting single-genre subsets from multi-genre treebanks
- **Computing cluster centroids**: Averaging embeddings within clusters
- **Building reference embeddings**: Constructing genre references from virtual splits
- **Labeling clusters**: Assigning genres with confidence + margin thresholds

This architectural pattern ensures that any improvements or bug fixes automatically apply to both production and evaluation, preventing divergence.

### 2.1 Embedding Generation

**Model Architecture:**
The default embedding model is `intfloat/multilingual-e5-large` (Wang et al., 2024), a 560M parameter multilingual transformer encoder trained on 1 billion text pairs across 100+ languages.

**Technical Implementation:**
- **Framework:** HuggingFace Transformers (Wolf et al., 2020)
- **Tokenizer:** `AutoTokenizer` with model-specific vocabulary
- **Model:** `AutoModel` (encoder-only architecture)
- **Device:** Automatic CUDA/CPU detection with configurable override

**Pooling Strategy:**
Two pooling methods are supported:
1. **Mean pooling** (default): Average of all token embeddings weighted by attention mask
   ```
   sentence_emb = Σ(token_emb × attention_mask) / Σ(attention_mask)
   ```
2. **CLS pooling**: Use the first token ([CLS]) representation

**Configuration Parameters:**
- `model`: HuggingFace model identifier (default: `intfloat/multilingual-e5-large`)
- `pooling`: Pooling strategy (`mean` or `cls`)
- `layer`: Transformer layer to extract embeddings from (default: `-1` = final layer)
- `batch_size`: Number of sentences per batch (default: `8`)
- `device`: Compute device (`auto`, `cuda`, or `cpu`)
- `cache_dir`: Directory for caching computed embeddings to avoid recomputation

**Caching Mechanism:**
Embeddings are cached on disk using a deterministic naming scheme:
```
{cache_dir}/{treebank_code}_{split}_{model_hash}.npz
```
This enables efficient re-use across multiple pipeline runs.

**Output Format:**
For each treebank split:
```python
{
    'sent_id': List[str],      # UD sentence identifiers
    'embedding': np.ndarray    # [n_sentences, embedding_dim]
}
```

### 2.2 Clustering

Two clustering algorithms are available, configurable via `clustering.method`:

#### 2.2.1 Gaussian Mixture Models (GMM)

**Implementation:** scikit-learn `GaussianMixture` (Pedregosa et al., 2011)

**Algorithm:** Expectation-Maximization (EM) for fitting mixture of multivariate Gaussians

**Parameters:**
- `n_components`: Number of clusters (determined by treebank metadata)
- `covariance_type`: `full` (default) - each component has its own covariance matrix
- `init_params`: `kmeans` - K-Means initialization for faster convergence
- `max_iter`: Maximum EM iterations (default: `300`)
- `reg_covar`: Regularization added to covariance diagonal (default: `1e-4`) - prevents singular matrices
- `random_state`: Random seed for reproducibility
- `verbose`: Iteration logging level

**Output:**
- Soft cluster assignments: `predict_proba()` returns probability distributions
- Cluster IDs: Argmax of probability distribution per sentence
- Model quality metric: Bayesian Information Criterion (BIC)

**Advantages:**
- Provides probabilistic assignments (uncertainty quantification)
- Models cluster shape and variance
- Well-suited for overlapping distributions

**Limitations:**
- CPU-only implementation (no GPU acceleration available in scikit-learn)
- Computationally expensive for high-dimensional data
- May fail on single-component fits (addressed via special handling)

#### 2.2.2 K-Means

**Implementation:** scikit-learn `KMeans` (CPU) or cuML `KMeans` (GPU)

**Algorithm:** Iterative centroid-based partitioning

**Parameters:**
- `n_clusters`: Number of clusters (determined by treebank metadata)
- `init`: Initialization method (`k-means++` for better convergence)
- `n_init`: Number of initializations (scikit-learn: `auto`, cuML: `10`)
- `max_iter`: Maximum EM iterations (default: `300`)
- `random_state`: Random seed for reproducibility
- `device`: Compute device (cuML-specific)

**GPU Acceleration:**
When GPU is available and cuML is installed:
- Uses RAPIDS cuML for GPU-accelerated clustering
- Automatic fallback to scikit-learn if GPU unavailable
- Significant speedup (2-10× faster) for large datasets

**Soft Assignment Conversion:**
K-Means produces hard assignments, but soft probabilities are derived for compatibility:
```python
distances = model.transform(embeddings)  # Distance to each centroid
scores = -distances / temperature         # Negative distance scores
probs = softmax(scores)                   # Softmax normalization
```

**Output:**
- Soft cluster assignments (derived from distances)
- Cluster IDs: Assigned to nearest centroid
- Model quality metric: Inertia (within-cluster sum of squares)

**Advantages:**
- GPU acceleration available (cuML)
- Faster convergence than GMM
- Scalable to large datasets

**Limitations:**
- Assumes spherical clusters
- Derived probabilities are approximations

#### 2.2.3 Multi-Split Treebank Handling

**Treebank-Level Clustering:**
When a treebank has multiple splits (train/dev/test), all splits are combined before clustering to ensure consistent genre structure:

**Algorithm:**
1. **Group splits by treebank:** Collect all embeddings from train/dev/test splits
2. **Combine embeddings:** Use `np.vstack()` to merge into single array
3. **Track split membership:** Maintain mapping of each sentence to its original split
4. **Cluster combined data:** Perform clustering once on the merged embeddings
5. **Distribute results:** Filter cluster assignments back to individual splits for storage

**Rationale:**
This ensures that all splits of the same treebank share the same clustering structure. For example, if `cs_pdtc` has train/dev/test splits, they are clustered together to maintain consistent genre boundaries across all splits, rather than potentially getting different cluster structures for each split.

**Example:**
```
cs_pdtc:train (50,000 sentences) ┐
cs_pdtc:dev   (10,000 sentences) ├─→ Combined (70,000 sentences) → Cluster → Distribute back
cs_pdtc:test  (10,000 sentences) ┘
```

#### 2.2.4 Virtual Split Handling

When sentence-level genre metadata is available at sufficient quality, multi-genre treebanks are decomposed into single-genre virtual splits that span all available splits.
Quality gates are configuration-driven:
- `evaluation.metadata_validation.coverage_threshold` (default: `0.95`)
- `evaluation.metadata_validation.min_genre_sentences` (default: `100`)

**Detection:**
```python
can_create_virtual_splits = (
    n_genres >= 2 and
    len(sentence_genres) / len(all_sentences) >= coverage_threshold and
    n_eligible_genres_with_at_least_min_sentences >= 2
)
```

**Decomposition:**
For each genre `g` in a multi-genre treebank:
1. Extract sentences with `genre == g` from all splits
2. Group by original split for storage
3. Create virtual split key: `(treebank_code, split, genre)`
4. Store as single-genre cluster (trivial assignment)

If a sentence yields multiple conflicting genre labels from metadata extraction,
it is excluded from virtual split assignment (no arbitrary first-label choice).
Metadata extraction failures (split-level or sentence-level) are logged as warnings
and affected items are skipped.

**Example:**
```
de_pud:test (1000 sentences, genres: [news, wiki])
→ de_pud:test:news (500 sentences, genre: news)
→ de_pud:test:wiki (500 sentences, genre: wiki)

cs_pdtc:train/dev/test (70,000 sentences, genres: [academic, news, spoken, learner-essays])
→ cs_pdtc:train:academic (4,000 sentences, genre: academic)
→ cs_pdtc:train:news     (40,000 sentences, genre: news)
→ cs_pdtc:train:spoken   (6,000 sentences, genre: spoken)
→ ... (similar for dev and test)
```

Both virtual splits and the original multi-genre treebank are retained for different purposes:
- **Virtual splits:** Used as single-genre references during bootstrap labeling
- **Original treebank:** Used for clustering evaluation

### 2.3 Bootstrap Schedule Creation

**Scheduler:** Custom implementation based on Müller-Eberstein et al. (2021)

**Algorithm:**
1. **Initialize known genres:**
   ```python
   known_genres = {combo[0] for combo in genre_combinations
                   if len(combo) == 1}
   ```

2. **Iterative environment creation:**
   For each iteration until convergence or `max_iterations`:

   a. **Identify predictable combinations:**
      ```python
      predictable = {combo for combo in all_combinations
                     if (combo ∩ known_genres ≠ ∅) and
                        (combo ∉ known_combinations)}
      ```

   b. **Identify disjunct combinations:**
      ```python
      disjunct = {combo for combo in all_combinations
                  if combo ∩ known_genres == ∅}
      ```

   c. **Update known genres:**
      After labeling predictable combinations, newly resolved genres
      are added to `known_genres` for the next iteration.

3. **Termination:**
   The schedule terminates when:
   - All combinations are known, OR
   - No progress is made in an iteration, OR
   - `max_iterations` is reached

**Parameters:**
- `max_iterations`: Maximum bootstrap iterations (default: `10`)

**Output:**
```python
[
    {
        'known': ['news', 'wiki'],           # Genres available as references
        'predict': [('blog', 'news'), ...],  # Combinations to label
        'disjunct': [('legal', 'medical')]   # Unresolvable combinations
    },
    ...
]
```

### 2.4 Genre Labeling

#### 2.4.1 Reference Embedding Construction

For each known genre `g`, compute reference embedding:

```python
genre_embedding[g] = weighted_mean(
    values=[cluster.embedding
            for (treebank, split, *_) in single_genre_sources
            if genre(treebank, split) == g
            for cluster in clusters(treebank, split)],
    weights=[len(cluster.sent_ids)
             for (treebank, split, *_) in single_genre_sources
             if genre(treebank, split) == g
             for cluster in clusters(treebank, split)]
)
```

By default, this computes sentence-count weighted averages of cluster centroids from:
1. Single-genre treebanks with genre `g`
2. Virtual splits with genre `g`

#### 2.4.2 Cluster Labeling

For each cluster `c` in a multi-genre treebank:

1. **Compute similarities:**
   ```python
   similarity[g] = 1 - cosine_distance(cluster_emb[c], genre_emb[g])
   ```

2. **Assign best match:**
   ```python
   best_genre = argmax_g(similarity[g])
   confidence = similarity[best_genre]
   ```

3. **Apply uncertainty thresholds:**
   ```python
   margin = top1_similarity - top2_similarity
   if confidence >= min_confidence and margin >= min_margin:
       method = "bootstrap-labeled"
   else:
       method = "bootstrap-inferred"
   label_all_sentences(cluster[c], best_genre, confidence, method)
   ```

**Parameters:**
- `min_confidence`: Minimum top-1 cosine similarity threshold (default: `0.8`)
- `min_margin`: Minimum top1-top2 cosine similarity gap (default: `0.05`)

**Behavioral note:**
Thresholds do **not** suppress labeling. They control the uncertainty flag in `method`:
- High confidence: `confidence >= min_confidence` **and** `margin >= min_margin` → `bootstrap-labeled`
- Low confidence: otherwise → `bootstrap-inferred`

This design favors high recall/coverage during bootstrap. Downstream analysis can filter to `bootstrap-labeled` only when higher precision is required.

**Distance Metric:**
Cosine distance is used throughout:
```
cosine_distance(u, v) = 1 - (u · v) / (||u|| × ||v||)
```
Range: [0, 2], where 0 = identical, 1 = orthogonal, 2 = opposite

#### 2.4.3 Labeling Methods

Each sentence receives a method tag indicating its labeling source:
- `single-genre-treebank`: From original single-genre treebank
- `virtual-split`: From virtual split with sentence-level metadata
- `bootstrap-labeled`: From multi-genre cluster labeling meeting both uncertainty thresholds
- `bootstrap-inferred`: From multi-genre cluster labeling that fails either threshold

### 2.5 Configuration System

The pipeline uses YAML configuration files for reproducibility:

**Core Parameters:**
```yaml
embeddings:
  model: "intfloat/multilingual-e5-large"
  pooling: "mean"
  batch_size: 8
  layer: -1
  device: "auto"
  cache_dir: "/path/to/cache/"

clustering:
  method: "gmm"              # or "kmeans"
  level: "treebank"
  seed: 42
  device: "auto"             # GPU support for kmeans only
  max_iter: 300              # GMM/K-Means: maximum clustering iterations
  reg_covar: 1e-4            # GMM: covariance regularization (prevents singular matrices)

bootstrapping:
  min_confidence: 0.8
  min_margin: 0.05
  max_iterations: 10
  fail_on_incomplete: false

evaluation:
  treebank_sets:               # Optional named sets for reproducible comparisons
    literature_small:
      - "de_pud"
      - "cs_pdtc"
  metadata_validation:
    coverage_threshold: 0.95    # Also used as production virtual-split gate
    min_genre_sentences: 100    # Also used as production virtual-split gate

genre_extraction:
  mapping_path: "configs/genre_mappings.json"
  patterns_path: ["configs/metadata_patterns.json"]
  canonical_genres: null     # or list of genre labels
```

### 2.6 Output Format

Results are exported as Apache Parquet files maintaining UD structure:

**File Structure:**
```
output/
├── genres/
│   ├── {treebank_code}-{split}.parquet
│   └── all_genres.parquet
└── clusters/
    └── cluster_assignments.parquet
```

**Parquet Schema:**
```python
{
    'sent_id': str,        # UD sentence identifier
    'genre': str,          # Assigned genre label
    'confidence': float,   # Similarity score [0, 1]
    'method': str,         # Labeling method
    'treebank': str,       # Source treebank code
    'split': str           # UD split (train/dev/test)
}
```

### 2.7 Quality Metrics

The pipeline computes several quality indicators:

**Cluster Quality (per treebank):**
- Silhouette coefficient: Measure of cluster separation
- Calinski-Harabasz index: Ratio of between-cluster to within-cluster variance
- Davies-Bouldin index: Average similarity between clusters

**Genre Separation (global):**
- Pairwise centroid distances: Matrix of distances between all genre pairs
- Closest/furthest genre pairs: Identify confusable vs. distinct genres

**Bootstrap Quality:**
- Resolution rate: Percentage of sentences successfully labeled
- Confidence distribution: Histogram of assignment confidence scores
- Cross-lingual consistency: Genre assignments across language families

## 3. Evaluation Framework

### 3.1 Clustering Evaluation

The evaluation framework tests the actual clustering and labeling performance on multi-genre treebanks with known sentence-level genre metadata.

**Methodology:**
1. Split multi-genre treebanks into train/test folds
2. For training treebanks: **Combine all splits (train/dev/test) of the same treebank**, create virtual splits using sentence metadata from combined data, compute cluster centroids for each virtual split, and build reference genre embeddings from these centroids (mirroring production)
3. For test treebanks: **Combine all splits (train/dev/test) of the same treebank**, cluster once, and label clusters using the reference embeddings
4. Evaluate sentence-level accuracy against ground truth

**Mirroring Production in Evaluation:**
The evaluation faithfully mirrors the production implementation in two key ways:

1. **Reference Construction (Training Data):**
   - **Combines all splits** (train/dev/test) of the same training treebank
   - Creates virtual splits from the combined data using sentence-level metadata
   - Computes cluster centroids for each virtual split
   - Computes weighted averages of these cluster centroids per genre to create reference embeddings (`bootstrapping.reference_weighting`, default: `sentence_count`)
   - Example:
     - `cs_pdtc:train` + `cs_pdtc:dev` (70K sentences) → combined
     - Extract `cs_pdtc:news` virtual split (40K sentences) → cluster centroid
     - Reference for 'news' = weighted mean of all 'news' cluster centroids
   - This matches production's use of virtual split cluster embeddings as references

2. **Treebank-Level Clustering (Test Data):**
   - **Combines all splits** (train/dev/test) of the same test treebank before clustering
   - This ensures the evaluation tests the same clustering structure used in production
   - Example: If `de_pud` is in the test fold, all its splits are combined, clustered together, and evaluated jointly

**Bootstrap Configuration:**
- **`min_confidence` / `min_margin`**: Evaluation uses the same uncertainty thresholds as production. Cluster assignments are always labeled, but tracked as `bootstrap-labeled` vs `bootstrap-inferred` for analysis.
- **`reference_weighting`**: Shared reference aggregation strategy (`sentence_count` or `uniform`) used by both production labeling and evaluation.
- **`max_iterations`**: Upper bound for schedule iterations in the shared bootstrap runner.
- **`anchor_mode`**:
  - `strict`: only fold-train anchors (best for unknown-data generalization estimates)
  - `parity`: adds leakage-safe single-genre anchors for literature-style comparability

**Cross-Validation:**
- K-fold cross-validation (configurable K)
- Grouping options: by language, by treebank, or ungrouped
- **Important:** When `group_by="treebank"`, all splits of the same treebank stay together in the same fold to prevent data leakage
- In `parity` anchor mode, additional single-genre anchors are filtered to avoid test leakage (same test treebank always excluded; same test language excluded when `group_by="language"`)
- Stratification to maintain genre distribution
- Supports reusable named treebank sets (`evaluation.treebank_sets` + `--set`) for literature comparisons
- Supports progressive cumulative set evaluation (`--progressive`) to assess scaling up to full virtual-split coverage

**Metrics:**
- Overall accuracy: Percentage of correctly labeled sentences
- Micro-F1 (instance-labeled treebanks): Sentence-level micro-averaged F1
- Purity (PUR): Standard cluster purity over predicted label groups
- Agreement (AGR): Cross-treebank dominant-label consistency for the same true genre
- Overlap error (ΔBC): Inverse Bhattacharyya overlap between predicted and true treebank genre distributions
- Per-genre precision, recall, F1-score
- Confusion matrix: Genre-to-genre error patterns
- Per-fold variance: Stability across different train/test splits

**Distinction from Production:**
The evaluation clusters *mixed* sentences (testing actual problem), whereas production receives pre-separated virtual splits (when available). This ensures evaluation measures realistic performance.

## 4. Computational Requirements

**Memory:**
- Embedding model: ~2-3 GB GPU/RAM
- Clustering: ~1-2 GB per 10K sentences
- Caching: ~100-500 MB per treebank

**Compute Time (single treebank, 1000 sentences):**
- Embedding (CPU): ~2-5 minutes
- Embedding (GPU): ~30-60 seconds
- Clustering (GMM): ~10-30 seconds
- Clustering (K-Means GPU): ~2-5 seconds

**Scalability:**
- Embeddings are cached and reusable
- Clustering is treebank-independent (parallelizable)
- GPU acceleration provides 5-10× speedup for large datasets

## References

Müller-Eberstein, M., van der Goot, R., & Plank, B. (2021). Genre as Weak Supervision for Cross-lingual Dependency Parsing. *Proceedings of EMNLP 2021*.

Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

Wang, L., et al. (2024). Multilingual E5 Text Embeddings: A Technical Report. *arXiv preprint arXiv:2402.05672*.

Wolf, T., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. *Proceedings of EMNLP 2020: System Demonstrations*.
