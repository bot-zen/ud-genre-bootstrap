mkdir -p output/logs
set -o pipefail

HF_HUB_OFFLINE=0 uv run ud-genre-bootstrap evaluate \
  --config configs/sweeps/2.8-apples-fixed-strict-sentence_count.yaml \
  --fixed-partition \
  --sentence-split-map configs/apples/paper-split-map-v2.8.parquet \
  --anchor-partition train \
  --anchor-partition dev \
  --test-partition test \
  --group-by none \
  | tee output/logs/apples-2.8-fixed-strict-sentence_count.log

HF_HUB_OFFLINE=0 uv run ud-genre-bootstrap evaluate \
  --config configs/sweeps/2.8-apples-fixed-strict-uniform.yaml \
  --fixed-partition \
  --sentence-split-map configs/apples/paper-split-map-v2.8.parquet \
  --anchor-partition train \
  --anchor-partition dev \
  --test-partition test \
  --group-by none \
  | tee output/logs/apples-2.8-fixed-strict-uniform.log

#Quick metric extraction after runs:
#
#for f in output/logs/apples-2.8-fixed-strict-*.log; do
#  echo "===== $f"
#  rg -n "Overall Acc|Macro-F1|Purity|Agreement \\(AGR, treebank-level\\)|Overlap Error \\(ΔBC, treebank-level\\)" "$f"
#done
