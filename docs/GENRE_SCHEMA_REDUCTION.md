# Reduced Genre Schema Analysis

The current public release train uses `label_schema=ud`, the 18-label UD-v2
release metadata inventory. Do not silently change that inventory in a patch
release. Reduced inventories are exploratory until they become a separate
explicit `label_schema` train.

## Goal

Use two complementary views before proposing a reduced inventory:

- rational grouping from linguistic/text-type criteria
- data-driven grouping from unsupervised cluster co-occurrence, with evaluation
  confusion and embedding centroids used as diagnostic context

The first implementation target is UD `2.18`, because it is the current default
release and has generated local artifacts.

## Inputs

The reusable candidate projection config is:

```bash
configs/genre_schema_reduction.yaml
```

It lists the current source genres and several complete source-to-target
candidate mappings. A candidate must map every source genre exactly once, and all
target genres must be declared.

The included `convergent_functional_7` candidate is the first balanced proposal
to inspect after running UD 2.18 analysis. It keeps functional distinctions for
expository, instructional, literary, regulatory, and ambiguous web material, but
also incorporates the strict cluster-merge evidence for `blog/email/reviews/social`
and `news/spoken`.

The analysis command reads a generated release directory:

```bash
output/2.18-community-release/genres/
```

It uses `all_genres.parquet` and `clusters/cluster_assignments.parquet`.
Automatic data-driven component suggestions are based on `cluster_merge_score`,
which combines metadata-derived cluster co-occurrence with same-split
co-occurrence. If a full evaluation export is available, the command also
projects the confusion matrix onto each reduced candidate and reports pairwise
evaluation confusion.

Optionally pass `--cluster-state .../cluster_state.pkl` to add
metadata-derived embedding-centroid similarity. This pickle is large for full UD
releases, so it is not loaded by default. Centroid similarity is reported as a
diagnostic column but is not used to create automatic component candidates,
because sentence-embedding spaces can have high baseline cosine similarity
between many unrelated genre centroids.

## Workflow

Export full evaluation results when running an evaluation:

```bash
HF_DATASETS_CACHE="/mnt/scratch/egon/huggingface/datasets/" \
HF_HUB_CACHE="/mnt/scratch/egon/huggingface/hub/" \
UV_CACHE_DIR="/tmp/ud-genre-bootstrap-uv-cache" \
uv run ud-genre-bootstrap evaluate \
  --release-matrix configs/releases/full-ud-v1.1.1.yaml \
  --ud-version 2.18 \
  --n-folds 10 \
  --group-by language \
  --export-results output/2.18-community-release/genres/evaluation/full_results.json
```

Run the schema analysis:

```bash
UV_CACHE_DIR="/tmp/ud-genre-bootstrap-uv-cache" \
uv run ud-genre-bootstrap analyze-genre-schema \
  --release-dir output/2.18-community-release/genres \
  --candidate-config configs/genre_schema_reduction.yaml \
  --evaluation-results output/2.18-community-release/genres/evaluation/full_results.json \
  --ud-version 2.18 \
  --output output/schema-reduction/ud2.18
```

If no full evaluation export exists yet, omit `--evaluation-results`. The report
will still include support, provenance, cluster co-occurrence, and candidate
cluster-purity scores.

To include embedding-centroid evidence as well:

```bash
UV_CACHE_DIR="/tmp/ud-genre-bootstrap-uv-cache" \
uv run ud-genre-bootstrap analyze-genre-schema \
  --release-dir output/2.18-community-release/genres \
  --candidate-config configs/genre_schema_reduction.yaml \
  --cluster-state output/2.18-community-release/genres/clusters/cluster_state.pkl \
  --ud-version 2.18
```

## Outputs

The command writes:

- `report.md`: human-readable summary with support, candidate scores, and pair
  evidence
- `candidate_mappings.json`: rational candidate mappings plus conservative
  data-driven component candidates
- `genre_similarity.tsv`: ranked pairwise merge evidence
- `projection_scores.json`: machine-readable candidate scores and projected
  evaluation metrics when available
- `genre_similarity_heatmap.png` and `genre_similarity_dendrogram.png` when
  plotting dependencies are installed

## Interpretation

Use metadata-derived rows as the primary evidence. All-label cluster evidence is
secondary because cluster-derived labels already depend on the current
bootstrapping process.

Treat centroid similarity as supporting context only. If centroid values are
uniformly high, rely on `cluster_merge_score`, projected evaluation metrics, and
the rational mappings instead of centroid-driven components.

Do not choose a target number of reduced genres upfront. Inspect whether the
same merges appear in:

- rational candidate mappings
- high pairwise merge evidence
- data-driven component candidates
- projected evaluation improvements

Sparse labels such as `email`, `medical`, and `government` need special care:
they may look easy to merge because they have few anchors, not because the
boundary is conceptually weak.

If a reduced inventory is later promoted, create a new release train with a new
`label_schema`. Keep `full-ud-v1` as the historical UD-v2 inventory.
