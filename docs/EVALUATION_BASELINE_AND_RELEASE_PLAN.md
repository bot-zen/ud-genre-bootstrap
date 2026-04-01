# Evaluation Baseline And Community Release Plan

This document locks the current evaluation interpretation and the current end-user baseline, and turns the next full-data release into an explicit plan rather than an implicit sequence of commands.

## 1. Why We Keep Two Evaluation Views

The framework serves two different evaluation targets:

- `paper_parity`: reconstruct the original fixed-partition GMM+L setting from Muller-Eberstein et al. as closely as possible
- `generalization`: estimate how well the released sentence-level genre layer behaves for end users on broader, language-held-out data

These targets should not be conflated.

### 1.1 Treebank-Level Comparability

The original paper is centered on a fixed UD v2.8 split and clustering-oriented metrics:

- Purity (`PUR`)
- Agreement (`AGR`)
- overlap error (`ΔBC`)
- micro-F1 over the subset of treebanks with instance-level labels

These metrics are useful for comparing against the original GMM+L benchmark, but they do not directly answer the question an end user cares about most:

- did the framework assign the right label to this sentence?

### 1.2 Sentence-Level Utility

For the released annotation layer, the primary metric family is sentence-level:

- overall micro-F1-equivalent accuracy
- macro-F1
- per-genre precision / recall / F1
- confusion matrices

These metrics reflect whether the released labels are practically usable for downstream filtering, stratified evaluation, and genre-aware data selection.

The treebank-level metrics remain important, but as secondary comparability diagnostics rather than as the primary optimization target.

## 2. Paper-Style Evaluation And Achieved Parity

### 2.1 Original Paper Protocol

The original GMM+L evaluation uses a fixed global UD v2.8 split derived from `102-915-204.pkl`, same-protocol anchor construction, and reports the clustering-oriented metrics above.

Published GMM+L values from the paper:

- micro-F1: `0.540`
- `PUR`: `1.000`
- `AGR`: `1.000`
- `ΔBC`: `0.040`

### 2.2 Reconstructed Protocol Parity In This Repository

After correcting the sentence split map and aligning the evaluation flow to the paper protocol, the strict reconstructed parity run is:

- log: `output/logs/apples-2.8-paper-parity-uniform.log`
- micro-F1: `0.2805`
- macro-F1: `0.1218`
- `PUR`: `0.4315`
- `AGR`: `0.6250`
- `ΔBC`: `0.2408`

### 2.3 Implementation Parity

The decisive result is not the lower parity score by itself, but the parity audit against an original-like reimplementation:

- report: `output/parity_audit/paper_parity_vs_original.md`
- machine-readable output: `output/parity_audit/paper_parity_vs_original.json`
- prediction agreement: `1.00` on all `8` scored treebanks

This means the remaining gap to the published paper numbers is not primarily an implementation gap in GMM clustering or iterative bootstrapped labeling.

The remaining difference is downstream of the algorithm, mainly due to:

- differences between paper treebank-level inventories and the sentence-level gold recoverable today from the reconstructed subset
- anchor sparsity on the reconstructed subset
- the fact that the end-user-relevant evaluator is stricter than the original treebank-level comparability framing

## 3. Locked End-User Baseline

The current locked baseline for future improvements is the broader language-grouped 10-fold generalization run on UD v2.17.

Command used:

```bash
HF_HUB_OFFLINE=0 uv run ud-genre-bootstrap evaluate \
  --config configs/sweeps/how_universal-generalization-e5_large-k5-anchor_combined.yaml \
  --set all_focused \
  --n-folds 10 \
  --group-by language \
  | tee output/logs/all_focused-generalization-e5_large-k10-anchor_combined-baseline.log
```

Result from `output/logs/all_focused-generalization-e5_large-k10-anchor_combined-baseline.log`:

- `Mean Fold Acc (Micro-F1)`: `0.3901 +/- 0.1458`
- `Overall Acc (Micro-F1)`: `0.3333`
- `Macro-F1`: `0.2636`
- macro fold mean: `0.3245 +/- 0.1629`
- `PUR`: `0.5568`
- `AGR`: `0.5922`
- `ΔBC`: `0.0589`
- missing anchor genres: `email, government`

Why this is the main baseline:

- it uses the broader `all_focused` evaluation set
- it groups folds by language, which is closer to the real deployment condition than arbitrary split refolding
- it measures sentence-level behavior directly
- it surfaces anchor sparsity rather than hiding it inside a single fixed benchmark score

Future improvements should therefore be interpreted along two axes:

1. Do they preserve or improve protocol parity?
2. Do they improve the locked end-user baseline above?

## 4. Full-Data Run Plan

Before any new improvement sweep becomes the new default, the community release path should be pinned down.

### 4.1 Target

Produce one promoted full-data run for the current UD release (`v2.17`) that is:

- reproducible
- versioned
- joinable with `commul/universal_dependencies`
- publishable as a stable Hugging Face dataset revision

### 4.2 Freeze Before Running

Before launching the full run, freeze and commit a dedicated release config, for example:

- `configs/2.17-community-release.yaml`

That config should pin:

- `ud_source`
- `ud_version`
- embedding model and pooling
- clustering method
- uncertainty thresholds
- mapping files
- output repos / revisions
- whether the release follows the locked baseline exactly or a later promoted production profile

Do not start the community release run from an ad hoc sweep config.

### 4.3 Release Implementation Status

The release-grade export path is now implemented.

Implemented release behavior:

- `all_genres.parquet` now exports the primary join key:
  - `treebank`
  - `split`
  - `sent_id`
- `all_genres.parquet` now also carries row-level provenance:
  - `ud_version`
  - `ud_source_revision`
  - `model`
  - `pooling`
  - `clustering_method`
  - `config_name`
  - `run_id`
- local exports now write release support artifacts automatically:
  - `README.md`
  - `run_metadata.json`
  - `config.snapshot.yaml`
  - `evaluation/baseline_summary.json` when configured
  - copied mapping files under `mappings/`
- `push_to_hub()` now uploads the full non-pickle release artifact set instead of parquet files only
- `run` / `label` now trigger `push_to_hub()` when `output.push_to_hub=true`

Community release no longer relies on `sent_id` alone.
The primary join key is now `(treebank, split, sent_id)` throughout the release export path.

### 4.4 Recommended Full-Run Sequence

1. Freeze the release config and output location.
2. Run preflight coverage checks:

```bash
uv run ud-genre-bootstrap coverage --config configs/2.17-community-release.yaml
```

3. Spot-check known tricky treebanks:

```bash
uv run ud-genre-bootstrap test-genres --config configs/2.17-community-release.yaml --treebank ru_taiga --split train --limit 0 --no-examples
uv run ud-genre-bootstrap test-genres --config configs/2.17-community-release.yaml --treebank be_hse --split train --limit 0 --no-examples
uv run ud-genre-bootstrap test-genres --config configs/2.17-community-release.yaml --treebank en_ewt --split train --limit 0 --no-examples
```

4. Run the full pipeline in resumable stages rather than as one opaque command:

```bash
uv run ud-genre-bootstrap embed   --config configs/2.17-community-release.yaml
uv run ud-genre-bootstrap cluster --config configs/2.17-community-release.yaml
uv run ud-genre-bootstrap label   --config configs/2.17-community-release.yaml
```

5. Validate output counts and method distribution:

- sentence count vs UD source
- `bootstrap-labeled` vs `bootstrap-inferred` counts
- missing / null genre rate
- joinability back to UD by `(treebank, split, sent_id)`

6. Only after that, push the promoted release to Hugging Face.

## 5. Community Hugging Face Release Plan

### 5.1 Repository Layout

Recommended repos:

- genre labels: `commul/ud-genres`
- embeddings: keep separate, model-specific repos such as `commul/ud-embeddings-multilingual-e5-large`

Recommended branch / revision policy:

- one stable revision per UD release, e.g. `2.17`
- optional moving alias such as `main` only for the currently promoted default
- experimental runs should not overwrite the promoted release revision

### 5.2 Required Release Artifacts

At minimum, the genre release should contain:

- `all_genres.parquet`
- `clusters/cluster_assignments.parquet` for auditability
- frozen config snapshot used for the run
- `run_metadata.json`
- `evaluation/baseline_summary.json`
- dataset card / README
- copies or references for the mapping files used by the run

### 5.3 Required Columns For `all_genres.parquet`

Minimum community-facing schema:

- `treebank`
- `split`
- `sent_id`
- `genre`
- `confidence`
- `method`

Recommended provenance columns:

- `ud_version`
- `ud_source_revision`
- `model`
- `pooling`
- `clustering_method`
- `config_name`
- `run_id`

### 5.4 Dataset Card Content

The dataset card should explicitly state:

- that labels are derived rather than authoritative
- the promoted config used for the release
- the meaning of `confidence` and `method`
- the current locked evaluation baseline
- the distinction between protocol parity and end-user evaluation
- known limitations of sentence-level recoverability from metadata

### 5.5 Promotion Rule

The community release should not be replaced automatically by a run that only improves paper-parity scores.

Promotion should require:

1. no regression in release-blocking validation checks
2. acceptable parity behavior
3. improvement, or at least no material regression, on the locked end-user baseline

## 6. Immediate Next Work

Before starting new improvement sweeps, the next concrete tasks are now operational:

1. run the preflight checks with `configs/2.17-community-release.yaml`
2. run the full v2.17 pipeline once in resumable stages under that frozen config
3. validate row counts, method counts, and release artifacts in `output/2.17-community-release/genres`
4. only after validation, promote and upload the release revision on Hugging Face
