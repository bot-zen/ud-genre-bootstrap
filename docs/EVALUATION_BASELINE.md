# Evaluation Baseline And Promotion Criteria

This document locks the current evaluation interpretation, the current end-user
baseline, and the promotion criteria for release candidates. The reusable
release mechanics live in `docs/RELEASE.md`.

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

The current locked baseline for future improvements is the expanded
language-grouped 10-fold generalization run on UD v2.18. It evaluates every
UD v2.18 treebank split with usable multi-genre sentence metadata under the
current metadata patterns, while using the broader single-genre and
virtual-split material as the anchor/reference pool.

This run was produced immediately before the `full-ud-v1.1.1` matrix bump,
using the same evaluation recipe and metadata patterns as the current train.

Command used:

```bash
HF_DATASETS_CACHE="/mnt/scratch/egon/huggingface/datasets/" \
HF_HUB_CACHE="/mnt/scratch/egon/huggingface/hub/" \
HF_DATASETS_OFFLINE=0 \
HF_HUB_OFFLINE=0 \
UV_CACHE_DIR=/tmp/ud-genre-bootstrap-uv-cache \
uv run ud-genre-bootstrap evaluate \
  --release-matrix configs/releases/full-ud-v1.1.0.yaml \
  --ud-version 2.18 \
  --n-folds 10 \
  --group-by language \
  2>&1 | tee output/logs/2.18-all_available-generalization-e5_large-k10-anchor_combined-baseline.log
```

Result from `output/logs/2.18-all_available-generalization-e5_large-k10-anchor_combined-baseline.log`:

- evaluated sentences: `629900`
- multi-genre treebank splits: `79`
- instance-labeled treebanks: `44`
- `Mean Fold Acc (Micro-F1)`: `0.3373 +/- 0.1300`
- `Overall Acc (Micro-F1)`: `0.3324`
- `Macro-F1`: `0.2699`
- macro fold mean: `0.2837 +/- 0.1246`
- `PUR`: `0.5190`
- `AGR`: `0.5062`
- `ΔBC`: `0.0760`
- missing anchor genres: `email`

Why this is the main baseline:

- it uses the discovered `all_available` evaluation set for the current UD
  release and metadata patterns
- it groups folds by language, which is closer to the real deployment condition than arbitrary split refolding
- it measures sentence-level behavior directly
- it surfaces anchor sparsity rather than hiding it inside a single fixed benchmark score

Future improvements should therefore be interpreted along two axes:

1. Do they preserve or improve protocol parity?
2. Do they improve the locked end-user baseline above?

## 4. Release Promotion Criteria

The operational release process is documented in `docs/RELEASE.md`. This file
only defines the evaluation gate for deciding whether a generated artifact
should become a promoted public release.

A release candidate should not replace the public artifact merely because it
improves paper-parity metrics. Promotion requires:

1. no regression in release-blocking validation checks
2. acceptable paper-parity behavior for comparability with the original protocol
3. improvement, or at least no material regression, on the locked end-user
   generalization baseline
4. complete release provenance through `run_metadata.json`,
   `release_manifest.json`, source tags, and HF artifact tags

The current promoted artifacts and their status are recorded in
`configs/releases/genre_artifacts.yaml`.

## 5. Documentation Boundary

Use these documents as follows:

- `docs/EVALUATION_BASELINE.md`: evaluation interpretation and promotion criteria
- `docs/RELEASE.md`: reusable release workflow, versioning, branching, tagging,
  and publishing

Do not add full procedural release files for each UD version. For future UD
backfills, add the config and registry entry, then follow `docs/RELEASE.md`.
