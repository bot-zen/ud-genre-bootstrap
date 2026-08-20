import pytest

from ud_genre_bootstrap.utils.config import Config
from ud_genre_bootstrap.utils.release_identity import (
    active_release_registry_entries,
    build_artifact_key,
    build_source_branch,
    load_release_registry,
    parse_artifact_id,
    parse_artifact_key,
    parse_artifact_version,
    parse_train_id,
    resolve_release_identity,
    validate_artifact_key,
    validate_train_id,
)


def test_parse_train_id_valid():
    parsed = parse_train_id("full-ud-v1.0.1")

    assert parsed == {
        "scope": "full",
        "label_schema": "ud",
        "artifact_version": "v1.0.1",
        "artifact_version_normalized": "v1.0.1",
    }


@pytest.mark.parametrize("version", ["v1", "v1.2", "v1.2.1"])
def test_parse_artifact_version_valid(version):
    assert parse_artifact_version(version)["normalized"].startswith("v1.")


def test_parse_artifact_key_valid():
    parsed = parse_artifact_key("full-ud-v1.0.1-ud2.18")

    assert parsed["train_id"] == "full-ud-v1.0.1"
    assert parsed["ud_version"] == "2.18"
    assert parsed["scope"] == "full"
    assert parsed["label_schema"] == "ud"
    assert parsed["artifact_version_normalized"] == "v1.0.1"


@pytest.mark.parametrize(
    "value",
    [
        "ud2.18-full-ud-v1.0.1",
        "full-v1.0.0-ud2.18",
        "full-ud-ud2.18",
        "full-ud-v0.0.1-ud2.18",
        "full-ud-v1.0.1",
    ],
)
def test_parse_artifact_key_rejects_missing_components(value):
    with pytest.raises(ValueError, match="artifact_key"):
        parse_artifact_key(value)


def test_legacy_parse_artifact_id_remains_accepted_for_snapshots():
    parsed = parse_artifact_id("ud2.17-full-ud-v1")

    assert parsed["legacy"] is True
    assert parsed["train_id"] == "full-ud-v1"
    assert parsed["artifact_version_normalized"] == "v1.0.0"


def test_validate_artifact_key_rejects_inconsistent_components():
    with pytest.raises(ValueError, match="inconsistent"):
        validate_artifact_key("full-ud-v1.0.1-ud2.18", label_schema="udmultigenre")


def test_train_helpers_build_derived_names():
    assert build_artifact_key(train_id="full-ud-v1.0.1", ud_version="2.18") == (
        "full-ud-v1.0.1-ud2.18"
    )
    assert build_source_branch("full-ud-v1.0.1") == "release/full-ud-v1"
    assert validate_train_id("full-ud-v1", artifact_version="v1.0.0")[
        "artifact_version_normalized"
    ] == "v1.0.0"


def test_config_resolves_train_identity():
    cfg = Config.from_dict(
        {
            "ud_version": "2.18",
            "release": {
                "train_id": "full-ud-v1.0.1",
                "artifact_key": "full-ud-v1.0.1-ud2.18",
                "scope": "full",
                "label_schema": "ud",
                "artifact_version": "v1.0.1",
                "inventory_status": "partial",
                "hf_repo": "commul/ud_genre",
                "hf_branches": ["2.18"],
                "hf_tag": "artifact/full-ud-v1.0.1/ud2.18",
                "source_branch": "release/full-ud-v1",
                "source_tag": "source/full-ud-v1.0.1",
            },
        }
    )

    identity = resolve_release_identity(cfg)

    assert identity["train_id"] == "full-ud-v1.0.1"
    assert identity["artifact_key"] == "full-ud-v1.0.1-ud2.18"
    assert identity["artifact_id"] == "full-ud-v1.0.1-ud2.18"
    assert identity["inventory_status"] == "partial"
    assert identity["hf_branches"] == ["2.18"]
    assert identity["hf_tag"] == "artifact/full-ud-v1.0.1/ud2.18"
    assert identity["source_branch"] == "release/full-ud-v1"
    assert identity["source_tag"] == "source/full-ud-v1.0.1"


def test_release_registry_validates_and_selects_rebuild_trains(tmp_path):
    registry = tmp_path / "genre_artifacts.yaml"
    registry.write_text(
        """
trains:
  - train_id: "full-ud-v1.0.1"
    scope: "full"
    label_schema: "ud"
    artifact_version: "v1.0.1"
    status: "partial"
    hf_repo: "commul/ud_genre"
    supported_ud_versions: ["2.17", "2.18"]
    default_ud_version: "2.18"
    release_profile: "configs/release_profiles/full-ud.yaml"
    release_matrix: "configs/releases/full-ud-v1.0.1.yaml"
    source_repo: "git@example.test/ud-genre-bootstrap.git"
    source_branch: "release/full-ud-v1"
    source_tag: "source/full-ud-v1.0.1"
  - train_id: "full-ud-v1.0.0"
    scope: "full"
    label_schema: "ud"
    artifact_version: "v1.0.0"
    status: "deprecated"
    hf_repo: "commul/ud_genre"
    supported_ud_versions: ["2.17"]
    default_ud_version: "2.17"
    release_profile: "configs/release_profiles/full-ud.yaml"
    release_matrix: "configs/releases/full-ud-v1.0.0.yaml"
    source_repo: "git@example.test/ud-genre-bootstrap.git"
    source_branch: "release/full-ud-v1"
    source_tag: "source/full-ud-v1.0.0"
""",
        encoding="utf-8",
    )

    entries = load_release_registry(registry)
    active_entries = active_release_registry_entries(registry)

    assert entries[0]["artifact_version_normalized"] == "v1.0.1"
    assert [entry["train_id"] for entry in active_entries] == ["full-ud-v1.0.1"]
