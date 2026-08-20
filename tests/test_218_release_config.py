from ud_genre_bootstrap.utils.release_identity import resolve_release_identity
from ud_genre_bootstrap.utils.release_matrix import load_release_matrix_config


def test_218_release_matrix_resolves_default_train_identity():
    cfg = load_release_matrix_config("configs/releases/full-ud-v1.0.0.yaml", ud_version="2.18")
    identity = resolve_release_identity(cfg)

    assert cfg.ud_version == "2.18"
    assert identity["train_id"] == "full-ud-v1.0.0"
    assert identity["artifact_key"] == "full-ud-v1.0.0-ud2.18"
    assert identity["hf_branches"] == ["2.18"]
    assert identity["hf_tag"] == "artifact/full-ud-v1.0.0/ud2.18"
    assert identity["source_branch"] == "release/full-ud-v1"
    assert identity["source_tag"] == "source/full-ud-v1.0.0"
