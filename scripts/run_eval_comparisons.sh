#!/usr/bin/env bash

set -euo pipefail
set -o pipefail

LOG_DIR="${LOG_DIR:-output/logs}"
N_FOLDS="${N_FOLDS:-5}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
SPLIT_MAP_PATH="${SPLIT_MAP_PATH:-configs/apples/paper-split-map-v2.8.parquet}"

mkdir -p "${LOG_DIR}"

LOG_FILES=()

run_logged() {
  local log_name="$1"
  shift

  local log_path="${LOG_DIR}/${log_name}.log"
  echo
  echo "==> ${log_name}"
  echo "Command: $*"
  env HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" "$@" |& tee "${log_path}"
  LOG_FILES+=("${log_path}")
}

# 2.17 cross-validation comparisons
run_logged \
  "how_universal-generalization-mbert-k${N_FOLDS}" \
  uv run ud-genre-bootstrap evaluate \
  --config configs/sweeps/how_universal-generalization-mbert-k5.yaml \
  --set how_universal \
  --n-folds "${N_FOLDS}" \
  --group-by language

run_logged \
  "how_universal-generalization-e5_large-k${N_FOLDS}" \
  uv run ud-genre-bootstrap evaluate \
  --config configs/sweeps/how_universal-generalization-e5_large-k5.yaml \
  --set how_universal \
  --n-folds "${N_FOLDS}" \
  --group-by language

run_logged \
  "how_universal-comparability-e5_large-k${N_FOLDS}" \
  uv run ud-genre-bootstrap evaluate \
  --config configs/sweeps/how_universal-comparability-e5_large-k5.yaml \
  --set how_universal \
  --n-folds "${N_FOLDS}" \
  --group-by none

# 2.8 fixed-partition apples-to-apples comparisons
if [[ -f "${SPLIT_MAP_PATH}" ]]; then
  run_logged \
    "apples-2.8-fixed-strict-sentence_count" \
    uv run ud-genre-bootstrap evaluate \
    --config configs/sweeps/2.8-apples-fixed-strict-sentence_count.yaml \
    --fixed-partition \
    --sentence-split-map "${SPLIT_MAP_PATH}" \
    --anchor-partition train \
    --anchor-partition dev \
    --test-partition test \
    --group-by none

  run_logged \
    "apples-2.8-fixed-strict-uniform" \
    uv run ud-genre-bootstrap evaluate \
    --config configs/sweeps/2.8-apples-fixed-strict-uniform.yaml \
    --fixed-partition \
    --sentence-split-map "${SPLIT_MAP_PATH}" \
    --anchor-partition train \
    --anchor-partition dev \
    --test-partition test \
    --group-by none

  run_logged \
    "apples-2.8-paper-parity-uniform" \
    uv run ud-genre-bootstrap evaluate \
    --config configs/sweeps/2.8-apples-fixed-parity-uniform.yaml \
    --sentence-split-map "${SPLIT_MAP_PATH}" \
    --test-partition test

  run_logged \
    "apples-2.8-fixed-strict-uniform-e5_large" \
    uv run ud-genre-bootstrap evaluate \
    --config configs/sweeps/2.8-apples-fixed-strict-uniform-e5_large.yaml \
    --fixed-partition \
    --sentence-split-map "${SPLIT_MAP_PATH}" \
    --anchor-partition train \
    --anchor-partition dev \
    --test-partition test \
    --group-by none
else
  echo
  echo "Skipping apples-to-apples fixed-partition runs: split map not found at ${SPLIT_MAP_PATH}"
fi

echo

echo "Done. Metric summary:"
for log_file in "${LOG_FILES[@]}"; do
  echo "===== ${log_file}"
  rg -n \
    "Mean Fold Acc|Overall Acc|Macro-F1|Purity \(PUR\)|Agreement \(AGR, treebank-level\)|Overlap Error \(ΔBC, treebank-level\)|Anchors by Genre|Missing Anchor Genres" \
    "${log_file}" || true
done
