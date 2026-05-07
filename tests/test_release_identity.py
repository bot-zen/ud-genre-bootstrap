import pytest

from ud_genre_bootstrap.utils.config import Config
from ud_genre_bootstrap.utils.release_identity import (
    parse_artifact_id,
    resolve_release_identity,
    validate_artifact_id,
)


def test_parse_artifact_id_valid():
    parsed = parse_artifact_id("ud2.17-full-ud-v1")

    assert parsed == {
        "ud_version": "2.17",
        "scope": "full",
        "label_schema": "ud",
        "artifact_version": "v1",
    }


@pytest.mark.parametrize(
    "artifact_id",
    [
        "2.17-full-ud-v1",
        "ud2.17-ud-v1",
        "ud2.17-full-v1",
        "ud2.17-full-ud",
    ],
)
def test_parse_artifact_id_rejects_missing_components(artifact_id):
    with pytest.raises(ValueError, match="artifact_id"):
        parse_artifact_id(artifact_id)


def test_validate_artifact_id_rejects_inconsistent_components():
    with pytest.raises(ValueError, match="inconsistent"):
        validate_artifact_id("ud2.17-full-ud-v1", label_schema="udmultigenre")


def test_config_parses_release_identity():
    cfg = Config.from_dict(
        {
            "ud_version": "2.17",
            "release": {
                "artifact_id": "ud2.17-full-ud-v1",
                "scope": "full",
                "label_schema": "ud",
                "artifact_version": "v1",
                "hf_repo": "commul/ud-genres",
                "hf_revisions": ["2.17", "ud2.17-full-ud-v1"],
                "git_branch": "release/ud-2.17",
                "git_tag": "artifact/ud2.17-full-ud-v1",
            },
        }
    )

    identity = resolve_release_identity(cfg)

    assert identity["artifact_id"] == "ud2.17-full-ud-v1"
    assert identity["hf_revisions"] == ["2.17", "ud2.17-full-ud-v1"]
    assert identity["git_tag"] == "artifact/ud2.17-full-ud-v1"
