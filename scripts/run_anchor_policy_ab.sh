#!/usr/bin/env bash

set -euo pipefail
set -o pipefail

LOG_DIR="${LOG_DIR:-output/logs}"
N_FOLDS="${N_FOLDS:-5}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

mkdir -p "${LOG_DIR}"

run_logged() {
  local name="$1"
  shift
  local log_path="${LOG_DIR}/${name}.log"

  echo
  echo "==> ${name}"
  echo "Command: $*"
  env HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" "$@" |& tee "${log_path}"
}

run_logged \
  "how_universal-generalization-e5_large-k${N_FOLDS}-anchor_train_virtual" \
  uv run ud-genre-bootstrap evaluate \
  --config configs/sweeps/how_universal-generalization-e5_large-k5-anchor_train_virtual.yaml \
  --set how_universal \
  --n-folds "${N_FOLDS}" \
  --group-by language

run_logged \
  "how_universal-generalization-e5_large-k${N_FOLDS}-anchor_combined" \
  uv run ud-genre-bootstrap evaluate \
  --config configs/sweeps/how_universal-generalization-e5_large-k5-anchor_combined.yaml \
  --set how_universal \
  --n-folds "${N_FOLDS}" \
  --group-by language

echo
for f in \
  "${LOG_DIR}/how_universal-generalization-e5_large-k${N_FOLDS}-anchor_train_virtual.log" \
  "${LOG_DIR}/how_universal-generalization-e5_large-k${N_FOLDS}-anchor_combined.log"
do
  echo "===== ${f}"
  rg -n \
    "Mean Fold Acc|Overall Acc|Macro-F1|Purity \(PUR\)|Agreement \(AGR, treebank-level\)|Overlap Error \(ΔBC, treebank-level\)|Anchor Policy|Anchors by Genre|Missing Anchor Genres" \
    "${f}" || true
done
