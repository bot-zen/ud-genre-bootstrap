mkdir -p output/sweeps/2.17/logs

uv run ud-genre-bootstrap evaluate --config configs/sweeps/how_universal-comparability-mbert.yaml --set how_universal --n-folds 3 | tee output/sweeps/2.17/logs/comparability-mbert.log
uv run ud-genre-bootstrap evaluate --config configs/sweeps/how_universal-comparability-xlmr_large.yaml --set how_universal --n-folds 3 | tee output/sweeps/2.17/logs/comparability-xlmr_large.log
uv run ud-genre-bootstrap evaluate --config configs/sweeps/how_universal-comparability-e5_large.yaml --set how_universal --n-folds 3 | tee output/sweeps/2.17/logs/comparability-e5_large.log

uv run ud-genre-bootstrap evaluate --config configs/sweeps/how_universal-generalization-mbert.yaml --set how_universal --n-folds 3 | tee output/sweeps/2.17/logs/generalization-mbert.log
uv run ud-genre-bootstrap evaluate --config configs/sweeps/how_universal-generalization-xlmr_large.yaml --set how_universal --n-folds 3 | tee output/sweeps/2.17/logs/generalization-xlmr_large.log
uv run ud-genre-bootstrap evaluate --config configs/sweeps/how_universal-generalization-e5_large.yaml --set how_universal --n-folds 3 | tee output/sweeps/2.17/logs/generalization-e5_large.log
