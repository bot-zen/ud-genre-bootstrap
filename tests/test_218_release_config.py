from ud_genre_bootstrap.utils.config import load_config
from ud_genre_bootstrap.utils.release_identity import resolve_release_identity


def test_218_community_release_config_has_candidate_identity():
    cfg = load_config("configs/2.18-community-release.yaml")

    identity = resolve_release_identity(cfg)

    assert cfg.ud_version == "2.18"
    assert identity["artifact_id"] == "ud2.18-full-ud-v1"
    assert identity["hf_branches"] == ["2.18"]
    assert identity["hf_tag"] == "artifact/ud2.18-full-ud-v1"
    assert identity["source_branch"] == "release/v1"
    assert identity["source_tag"] == "source/ud2.18-full-ud-v1"
