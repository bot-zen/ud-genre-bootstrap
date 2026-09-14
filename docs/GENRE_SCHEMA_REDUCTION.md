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

The current serious seed candidate is `udmultigenre_informed_9`. It is a
reduced, full-coverage proposal informed by UD-MULTIGENRE, CMC/register
considerations, and local UD 2.17/2.18 evidence. The earlier
`convergent_functional_7` candidate is kept as a stress-test candidate because
it exposes the strongest purely cluster-driven reduction, including the debated
`news/spoken` merge.

## Decision Sources

The decision process intentionally separates four kinds of evidence:

- UD-MULTIGENRE: Danilova and Stymne (2023) manually analyze UD documentation
  and metadata patterns, reorganize UD data into instance-level genres, and
  argue that several UD labels are inconsistent or not coherent genre labels.
  Relevant source: https://aclanthology.org/2023.mrl-1.19/
- UD-MULTIGENRE repository: the released mapping/pattern files document the
  concrete source-pattern decisions used by the dataset.
  Relevant source: https://github.com/UppsalaNLP/UD-MULTIGENRE
- CMC/register work: medium and register should be treated as separate sources
  of linguistic variation, especially for CMC data. This argues against
  collapsing `spoken` with written public genres just because one cluster
  signal is borderline. Relevant sources:
  https://cmc2026.org/abstracts/#paper-128 and
  https://www.benjamins.com/catalog/rs.22009.sch
- Local release evidence: `analyze-genre-schema` reports over UD 2.18 and UD
  2.17 provide support counts, cluster-merge evidence, projected evaluation
  scores when available, and diagnostic centroid/evaluation evidence.

UD-MULTIGENRE is not adopted wholesale. It covers a curated subset, excludes
some data for coherence, and introduces finer labels such as `guide`,
`interviews`, `parliament`, `QA`, and `textbook`. The candidate here is a
full-coverage reduced schema for this release pipeline, so some distinctions are
kept as residual classes rather than removed from the dataset.

## Current Seed

`udmultigenre_informed_9` maps the current 18 UD-derived source labels to nine
target labels:

| Target | Source labels | Decision rationale |
| --- | --- | --- |
| `expository` | `academic`, `medical`, `nonfiction`, `wiki` | Informational prose supercategory. UD-MULTIGENRE treats `medical` as academic-like and shows that `nonfiction` contains multiple informational subtypes. `wiki` is reduced into this group for full coverage even though UD-MULTIGENRE keeps it distinct. |
| `grammar_examples` | `grammar-examples` | Residual/special-purpose class. UD-MULTIGENRE excludes grammar examples as not a coherent genre; local evidence does not support merging it with learner essays. |
| `interactional` | `blog`, `email`, `reviews`, `social` | Linguistically plausible CMC-adjacent group for addressive, opinionated, or socially situated writing. It is also the most stable local merge family. |
| `learner_essays` | `learner-essays` | Kept separate because it is meaningful in UD-MULTIGENRE and local evidence does not support merging it with grammar examples. |
| `literary` | `bible`, `fiction`, `poetry` | Broad narrative/literary supercategory. This is a reduced-schema approximation, not a claim that religious narrative, fiction, and verse are interchangeable. |
| `news` | `news` | Kept separate because its source/function is well established enough and the proposed `news/spoken` merge is conceptually weak. |
| `regulatory` | `government`, `legal` | Lossy full-coverage approximation. UD-MULTIGENRE treats governmental administrative text as legal-like but separates parliamentary material when metadata permits. |
| `spoken` | `spoken` | Kept separate as a medium/register distinction. UD-MULTIGENRE limits spoken to spontaneous speech and moves planned/parliamentary speech elsewhere when possible. |
| `web_residual` | `web` | Residual/channel-like class. UD-MULTIGENRE excludes web as not a coherent single genre; local evidence does not give a stable merge target. |

## Evidence Snapshot

The current UD 2.18 report with evaluation and centroid context is:

```bash
output/schema-reduction/ud2.18/report.md
```

The cross-version check without evaluation/centroid context is:

```bash
output/schema-reduction/ud2.17/report.md
```

Candidate-level scores currently show the expected tradeoff:

| Candidate | UD version | Target labels | Cluster purity | Gain | Eval micro-F1 | Eval macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `udmultigenre_informed_9` | 2.18 | 9 | 0.807 | 0.023 | 0.444 | 0.359 |
| `udmultigenre_informed_9` | 2.17 | 9 | 0.814 | 0.024 | n/a | n/a |
| `convergent_functional_7` | 2.18 | 7 | 0.864 | 0.080 | 0.453 | 0.386 |
| `convergent_functional_7` | 2.17 | 7 | 0.867 | 0.078 | n/a | n/a |

The 7-way candidate scores higher because it collapses more labels, including
`news/spoken`. It is useful as an upper-bound stress test for compression, but
the 9-way candidate is the better scientific seed because it preserves
distinctions that are independently motivated by UD-MULTIGENRE and
medium/register research.

The strongest stable cluster evidence supports:

- `academic + fiction`: `0.760` in UD 2.17 and `0.713` in UD 2.18. This is
  useful as a stress-test signal but is not adopted directly because it conflates
  expository and literary functions.
- `email + social`: `0.985` in UD 2.17 and `0.987` in UD 2.18.
- `email + reviews`: `0.899` in UD 2.17 and `0.923` in UD 2.18.
- `blog + email`: weaker in UD 2.17 (`0.434`) but stronger in UD 2.18 (`0.616`);
  it is included in `interactional` because the linguistic grouping is coherent.
- `news + spoken`: borderline in UD 2.17 (`0.492`) and UD 2.18 (`0.501`), but
  rejected for the serious seed because UD-MULTIGENRE and register/medium
  considerations argue for keeping spoken separate.

The rejected `convergent_functional_7` candidate scores well in projected UD
2.18 metrics, but that gain depends partly on collapsing `news` and `spoken`.
For a scientific argument to the UD or CMC community, the stronger position is
to prefer the slightly less compressed but better motivated
`udmultigenre_informed_9` candidate.

## Argument Outline

A later scientific argument can be framed as follows:

1. The original UD-v2 genre inventory is useful as release metadata but mixes
   genre, topic, medium, source channel, and special-purpose material.
2. UD-MULTIGENRE demonstrates that documentation- and metadata-based
   reclassification improves coherence, but its curated 17-label inventory is
   not directly a full-coverage reduced schema.
3. CMC/register work motivates keeping medium and register separate; therefore
   `spoken` should not be merged with `news` without stronger independent
   evidence.
4. Local UD 2.17/2.18 evidence supports an interactional written group
   (`blog/email/reviews/social`) and supports treating `web` and
   `grammar-examples` as residual/special labels.
5. The proposed 9-way seed keeps full coverage while making every merge either
   conceptually motivated, externally supported, locally supported, or
   explicitly marked as lossy.

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

## Materializing Projected Release Data

The serious seed can be materialized as its own release train without
overwriting the default `ud` train. The train is:

```bash
configs/releases/full-udmultigenre_informed_9-v1.0.0.yaml
```

It publishes to dedicated HF branches such as
`udmultigenre_informed_9/2.18` and immutable tags such as
`artifact/full-udmultigenre_informed_9-v1.0.0/ud2.18`. It must not move
`main`, `2.18`, or `2.17`.

Generate projected release data from existing `ud` release artifacts:

```bash
UV_CACHE_DIR="/tmp/ud-genre-bootstrap-uv-cache" \
uv run ud-genre-bootstrap project-genre-schema \
  --source-release-dir output/2.18-community-release/genres \
  --release-matrix configs/releases/full-udmultigenre_informed_9-v1.0.0.yaml \
  --ud-version 2.18 \
  --candidate-config configs/genre_schema_reduction.yaml \
  --candidate udmultigenre_informed_9

UV_CACHE_DIR="/tmp/ud-genre-bootstrap-uv-cache" \
uv run ud-genre-bootstrap project-genre-schema \
  --source-release-dir output/2.17-community-release/genres \
  --release-matrix configs/releases/full-udmultigenre_informed_9-v1.0.0.yaml \
  --ud-version 2.17 \
  --candidate-config configs/genre_schema_reduction.yaml \
  --candidate udmultigenre_informed_9
```

Each projected row keeps the projected target label in `genre` and preserves the
pre-projection UD label in `source_genre`.

Publish through the HF Git checkout:

```bash
UV_CACHE_DIR="/tmp/ud-genre-bootstrap-uv-cache" \
uv run ud-genre-bootstrap publish \
  --release-matrix configs/releases/full-udmultigenre_informed_9-v1.0.0.yaml \
  --ud-version 2.18 \
  --hf-repo-dir ../ud_genre-hf \
  --push

UV_CACHE_DIR="/tmp/ud-genre-bootstrap-uv-cache" \
uv run ud-genre-bootstrap publish \
  --release-matrix configs/releases/full-udmultigenre_informed_9-v1.0.0.yaml \
  --ud-version 2.17 \
  --hf-repo-dir ../ud_genre-hf \
  --push
```

Use `../ud_genre-hf` for HF publication. The sibling `../ud-genre` checkout is
the original `personads/ud-genre` repository and is not the HF dataset checkout.

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
