#!/usr/bin/env bash

set -euo pipefail
set -o pipefail

SPLIT_MAP_PATH="${SPLIT_MAP_PATH:-configs/apples/paper-split-map-v2.8.parquet}"
LOG_DIR="${LOG_DIR:-output/logs}"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${SPLIT_MAP_PATH}" ]]; then
  echo "Split-map not found: ${SPLIT_MAP_PATH}" >&2
  exit 1
fi

echo "Using split-map: ${SPLIT_MAP_PATH}"
echo "Logs: ${LOG_DIR}"

echo
echo "[1/2] Paper-parity apples run: mBERT + same-partition single-genre anchors + uniform"
HF_HUB_OFFLINE=0 uv run ud-genre-bootstrap evaluate \
  --config configs/sweeps/2.8-apples-fixed-parity-uniform.yaml \
  --sentence-split-map "${SPLIT_MAP_PATH}" \
  --test-partition test \
  |& tee "${LOG_DIR}/apples-2.8-paper-parity-uniform.log"

echo
echo "[2/2] Fixed-partition apples run: e5-large + strict + uniform"
HF_HUB_OFFLINE=0 uv run ud-genre-bootstrap evaluate \
  --config configs/sweeps/2.8-apples-fixed-strict-uniform-e5_large.yaml \
  --fixed-partition \
  --sentence-split-map "${SPLIT_MAP_PATH}" \
  --anchor-partition train \
  --anchor-partition dev \
  --test-partition test \
  --group-by none \
  |& tee "${LOG_DIR}/apples-2.8-fixed-strict-uniform-e5_large.log"

echo
echo "Done. Metric summary:"
for f in \
  "${LOG_DIR}/apples-2.8-paper-parity-uniform.log" \
  "${LOG_DIR}/apples-2.8-fixed-strict-uniform-e5_large.log"
do
  echo "===== ${f}"
  rg -n \
    "Mean Fold Acc|Overall Acc|Macro-F1|Purity \\(PUR\\)|Agreement \\(AGR, treebank-level\\)|Overlap Error \\(ΔBC, treebank-level\\)" \
    "${f}" || true
done
