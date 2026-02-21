from pathlib import Path
import copy
import yaml

base_path = Path("configs/2.17-local.yaml")
base = yaml.safe_load(base_path.read_text())

models = {
    "mbert": ("bert-base-multilingual-cased", 16),
    "xlmr_large": ("xlm-roberta-large", 8),
    "e5_large": ("intfloat/multilingual-e5-large", 16),
}

modes = {
    "comparability": {"group_by": None, "anchor_mode": "parity"},
    "generalization": {"group_by": "language", "anchor_mode": "strict"},
}

for mode, mode_cfg in modes.items():
    for tag, (model_name, batch_size) in models.items():
        cfg = copy.deepcopy(base)

        # Keep only the target set for this sweep
        how_universal = cfg["evaluation"]["treebank_sets"]["how_universal"]
        cfg["evaluation"]["treebank_sets"] = {"how_universal": how_universal}

        # Stabilize key knobs across both modes
        cfg["clustering"]["method"] = "gmm"
        cfg["clustering"]["fit_sample_size"] = None
        cfg["bootstrapping"]["reference_weighting"] = "sentence_count"
        cfg["bootstrapping"]["min_confidence"] = 0.8
        cfg["bootstrapping"]["min_margin"] = 0.05
        cfg["evaluation"]["metadata_validation"]["k"] = 3
        cfg["evaluation"]["metadata_validation"]["method"] = "kfold"

        # Mode separation
        cfg["evaluation"]["metadata_validation"]["group_by"] = mode_cfg["group_by"]
        cfg["evaluation"]["metadata_validation"]["anchor_mode"] = mode_cfg["anchor_mode"]

        # Model sweep
        cfg["embeddings"]["model"] = model_name
        cfg["embeddings"]["pooling"] = "mean"
        cfg["embeddings"]["batch_size"] = batch_size
        cfg["embeddings"]["layer"] = -1

        # Avoid known missing HF configs warning/noise
        excl = set(cfg.get("exclude_treebanks") or [])
        excl.update(["ar_nyuad", "ja_bccwj", "pt_cintil"])
        cfg["exclude_treebanks"] = sorted(excl)

        cfg["output"]["genres_path"] = f"output/sweeps/2.17/{mode}/{tag}/genres"

        out = Path(f"configs/sweeps/how_universal-{mode}-{tag}.yaml")
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))

print("Wrote 6 configs to configs/sweeps/")
