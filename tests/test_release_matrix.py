import pytest

from ud_genre_bootstrap.utils.release_identity import resolve_release_identity
from ud_genre_bootstrap.utils.release_matrix import load_release_matrix_config


MATRIX = "configs/releases/full-ud-v1.0.1.yaml"


def test_release_matrix_resolves_default_ud_version():
    cfg = load_release_matrix_config(MATRIX)
    identity = resolve_release_identity(cfg)

    assert cfg.ud_version == "2.18"
    assert cfg.output.genres_path == "output/2.18-community-release/genres"
    assert cfg.output.run_id == "full-ud-v1.0.1-ud2.18"
    assert identity["train_id"] == "full-ud-v1.0.1"
    assert identity["artifact_key"] == "full-ud-v1.0.1-ud2.18"
    assert identity["hf_tag"] == "artifact/full-ud-v1.0.1/ud2.18"
    assert identity["hf_revisions"] == ["2.18", "artifact/full-ud-v1.0.1/ud2.18"]
    assert identity["source_tag"] == "source/full-ud-v1.0.1"
    assert getattr(cfg, "_release_matrix_path").endswith("full-ud-v1.0.1.yaml")
    assert getattr(cfg, "_release_profile_path").endswith("full-ud.yaml")


def test_release_matrix_applies_per_version_overrides():
    cfg = load_release_matrix_config(MATRIX, ud_version="2.17")

    assert cfg.ud_version == "2.17"
    assert cfg.output.baseline_summary_path == (
        "configs/baselines/2.17-all_focused-generalization-e5_large-k10-anchor_combined.json"
    )
    assert "all_focused" in cfg.evaluation.treebank_sets
    assert "egy_ujaen" in cfg.evaluation.treebank_sets["all_focused"]
    assert cfg.output.embeddings_revision == "2.17"
    assert cfg.output.ud_source_revision == "2.17"


def test_release_matrix_rejects_unsupported_ud_version():
    with pytest.raises(ValueError, match="supported_ud_versions"):
        load_release_matrix_config(MATRIX, ud_version="2.6")
