import json

from ud_genre_bootstrap.utils.config import Config
from ud_genre_bootstrap.utils.release_artifacts import write_release_artifacts


def test_write_release_artifacts_records_identity_and_provenance(tmp_path):
    mapping_path = tmp_path / "genre_mappings.json"
    mapping_path.write_text('{"news": "news"}\n', encoding="utf-8")
    config_path = tmp_path / "release.yaml"
    config_path.write_text("ud_version: '2.17'\n", encoding="utf-8")
    all_genres_path = tmp_path / "all_genres.parquet"
    all_genres_path.write_text("stub", encoding="utf-8")

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
            "genre_extraction": {
                "mapping_path": str(mapping_path),
            },
            "output": {
                "genres_path": str(tmp_path),
                "genres_hf_repo": "commul/ud-genres",
                "genres_revision": "2.17",
            },
        }
    )
    setattr(cfg, "_config_path", str(config_path))

    artifacts = write_release_artifacts(
        cfg,
        tmp_path,
        {"total_sentences": 1, "labeled_sentences": 1, "genre_counts": {"news": 1}},
        all_genres_path=all_genres_path,
    )

    run_metadata = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "release_manifest.json").read_text(encoding="utf-8"))
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert artifacts["release_manifest"] == "release_manifest.json"
    assert run_metadata["artifact_id"] == "ud2.17-full-ud-v1"
    assert run_metadata["scope"] == "full"
    assert run_metadata["label_schema"] == "ud"
    assert run_metadata["hf_revisions"] == ["2.17", "ud2.17-full-ud-v1"]
    assert run_metadata["git_branch"] == "release/ud-2.17"
    assert run_metadata["git_tag"] == "artifact/ud2.17-full-ud-v1"
    assert run_metadata["config_hash"]
    assert run_metadata["mapping_file_hashes"]["mappings/genre_mappings.json"]
    assert run_metadata["algorithm_recipe"]["embeddings"]["model"] == cfg.embeddings.model
    assert run_metadata["algorithm_recipe"]["thresholds"]["min_confidence"] == 0.8

    assert manifest["artifact_id"] == run_metadata["artifact_id"]
    assert manifest["mapping_file_hashes"] == run_metadata["mapping_file_hashes"]
    assert "revision=\"2.17\"" in readme
    assert "Artifact ID: `ud2.17-full-ud-v1`" in readme
    assert "Label schema: `ud`" in readme
