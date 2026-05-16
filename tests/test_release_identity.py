import pytest

from ud_genre_bootstrap.utils.config import Config
from ud_genre_bootstrap.utils.release_identity import (
    active_release_registry_entries,
    load_release_registry,
    parse_artifact_id,
    parse_artifact_version,
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
        "artifact_version_normalized": "v1.0.0",
    }


@pytest.mark.parametrize(
    ("artifact_id", "normalized"),
    [
        ("ud2.17-full-ud-v1", "v1.0.0"),
        ("ud2.17-full-ud-v1.2", "v1.2.0"),
        ("ud2.17-full-ud-v1.2.1", "v1.2.1"),
    ],
)
def test_parse_artifact_id_accepts_semver_artifact_versions(artifact_id, normalized):
    parsed = parse_artifact_id(artifact_id)

    assert parsed["artifact_version_normalized"] == normalized


@pytest.mark.parametrize("version", ["v1", "v1.2", "v1.2.1"])
def test_parse_artifact_version_valid(version):
    assert parse_artifact_version(version)["normalized"].startswith("v1.")


@pytest.mark.parametrize(
    "artifact_id",
    [
        "2.17-full-ud-v1",
        "ud2.17-ud-v1",
        "ud2.17-full-v1",
        "ud2.17-full-ud",
        "ud2.17-full-ud-v0",
        "ud2.17-full-ud-v1.",
        "ud2.17-full-ud-v1.2.3.4",
    ],
)
def test_parse_artifact_id_rejects_missing_components(artifact_id):
    with pytest.raises(ValueError, match="artifact_id"):
        parse_artifact_id(artifact_id)


def test_validate_artifact_id_rejects_inconsistent_components():
    with pytest.raises(ValueError, match="inconsistent"):
        validate_artifact_id("ud2.17-full-ud-v1", label_schema="udmultigenre")


def test_validate_artifact_id_treats_version_shorthands_as_equivalent():
    parsed = validate_artifact_id(
        "ud2.17-full-ud-v1",
        artifact_version="v1.0.0",
    )

    assert parsed["artifact_version_normalized"] == "v1.0.0"


def test_config_parses_release_identity():
    cfg = Config.from_dict(
        {
            "ud_version": "2.17",
            "release": {
                "artifact_id": "ud2.17-full-ud-v1",
                "scope": "full",
                "label_schema": "ud",
                "artifact_version": "v1",
                "hf_repo": "commul/ud_genre",
                "hf_branches": ["2.17"],
                "hf_tag": "artifact/ud2.17-full-ud-v1",
                "source_branch": "release/v1",
                "source_tag": "source/ud2.17-full-ud-v1",
            },
        }
    )

    identity = resolve_release_identity(cfg)

    assert identity["artifact_id"] == "ud2.17-full-ud-v1"
    assert identity["hf_branches"] == ["2.17"]
    assert identity["hf_tag"] == "artifact/ud2.17-full-ud-v1"
    assert identity["source_tag"] == "source/ud2.17-full-ud-v1"
    assert identity["git_tag"] == "source/ud2.17-full-ud-v1"


def test_release_registry_validates_and_selects_active_entries(tmp_path):
    registry = tmp_path / "genre_artifacts.yaml"
    registry.write_text(
        """
artifacts:
  - artifact_id: "ud2.17-full-ud-v1.2.1"
    ud_version: "2.17"
    scope: "full"
    label_schema: "ud"
    artifact_version: "v1.2.1"
    status: "active"
    change_scope: "source_milestone"
    hf_repo: "commul/ud_genre"
    hf_branches: ["2.17"]
    hf_tag: "artifact/ud2.17-full-ud-v1.2.1"
    source_repo: "git@example.test/ud-genre-bootstrap.git"
    source_branch: "release/v1.2"
    source_tag: "source/ud2.17-full-ud-v1.2.1"
    source_config: "configs/2.17-community-release.yaml"
  - artifact_id: "ud2.16-full-ud-v1"
    ud_version: "2.16"
    scope: "full"
    label_schema: "ud"
    artifact_version: "v1"
    status: "deprecated"
    change_scope: "artifact_patch"
    hf_repo: "commul/ud_genre"
    hf_branches: ["2.16"]
    hf_tag: "artifact/ud2.16-full-ud-v1"
    source_repo: "git@example.test/ud-genre-bootstrap.git"
    source_branch: "release/v1"
    source_tag: "source/ud2.16-full-ud-v1"
    source_config: "configs/2.16-community-release.yaml"
""",
        encoding="utf-8",
    )

    entries = load_release_registry(registry)
    active_entries = active_release_registry_entries(registry)

    assert entries[0]["artifact_version_normalized"] == "v1.2.1"
    assert [entry["artifact_id"] for entry in active_entries] == [
        "ud2.17-full-ud-v1.2.1"
    ]
