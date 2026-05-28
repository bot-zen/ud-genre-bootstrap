import json
import subprocess

import pandas as pd
import pytest
import yaml

from ud_genre_bootstrap.utils.config import Config
from ud_genre_bootstrap.utils.release_artifacts import (
    list_release_publish_files,
    publish_release_directory_to_hf_git,
    write_release_artifacts,
)


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
                "hf_repo": "commul/ud_genre",
                "hf_branches": ["2.17"],
                "hf_tag": "artifact/ud2.17-full-ud-v1",
                "source_repo": "git@example.test/ud-genre-bootstrap.git",
                "source_branch": "release/v1",
                "source_tag": "source/ud2.17-full-ud-v1",
            },
            "genre_extraction": {
                "mapping_path": str(mapping_path),
            },
            "output": {
                "genres_path": str(tmp_path),
                "genres_hf_repo": "commul/ud_genre",
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
    assert run_metadata["hf_branches"] == ["2.17"]
    assert run_metadata["hf_tag"] == "artifact/ud2.17-full-ud-v1"
    assert run_metadata["source_branch"] == "release/v1"
    assert run_metadata["source_tag"] == "source/ud2.17-full-ud-v1"
    assert run_metadata["ud_source"] == "hf://commul/universal_dependencies"
    assert run_metadata["ud_source_revision"] == "2.17"
    assert run_metadata["config_hash"]
    assert run_metadata["mapping_file_hashes"]["mappings/genre_mappings.json"]
    assert run_metadata["source_files"]["config"]["path"] == str(config_path)
    assert run_metadata["source_files"]["mappings"][0]["path"] == str(mapping_path)
    assert run_metadata["algorithm_recipe"]["embeddings"]["model"] == cfg.embeddings.model
    assert run_metadata["algorithm_recipe"]["thresholds"]["min_confidence"] == 0.8

    assert manifest["artifact_id"] == run_metadata["artifact_id"]
    assert manifest["ud_source"] == run_metadata["ud_source"]
    assert manifest["ud_source_revision"] == run_metadata["ud_source_revision"]
    assert manifest["hf_payload"] == [
        "README.md",
        "all_genres.parquet",
        "release_manifest.json",
    ]
    assert manifest["mapping_file_hashes"] == run_metadata["mapping_file_hashes"]
    assert readme.startswith("---\n")
    card_metadata = yaml.safe_load(readme.split("---", 2)[1])
    assert card_metadata["pretty_name"] == "UD Genre Labels 2.17"
    assert card_metadata["task_categories"] == ["text-classification"]
    assert "universal-dependencies" in card_metadata["tags"]
    assert card_metadata["size_categories"] == ["n<1K"]
    assert "revision=\"2.17\"" in readme
    assert "revision=\"artifact/ud2.17-full-ud-v1\"" in readme
    assert "Artifact ID: `ud2.17-full-ud-v1`" in readme
    assert "Label schema: `ud`" in readme


def _git(repo_dir, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_git_repo(repo_dir):
    repo_dir.mkdir(parents=True, exist_ok=True)
    _git(repo_dir, "init")
    _git(repo_dir, "config", "user.email", "tests@example.test")
    _git(repo_dir, "config", "user.name", "Tests")
    _git(repo_dir, "config", "commit.gpgsign", "false")
    _git(repo_dir, "config", "tag.gpgsign", "false")


def _make_source_repo(tmp_path, source_tag="source/ud2.17-full-ud-v1"):
    source_repo = tmp_path / "source"
    _init_git_repo(source_repo)
    (source_repo / "source.txt").write_text("source\n", encoding="utf-8")
    _git(source_repo, "add", "source.txt")
    _git(source_repo, "commit", "-m", "source")
    _git(source_repo, "tag", source_tag)
    return source_repo


def _make_publish_config(tmp_path):
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
                "hf_default_branch": "main",
                "source_branch": "release/v1",
                "source_tag": "source/ud2.17-full-ud-v1",
            },
            "output": {
                "genres_path": str(tmp_path / "release"),
                "genres_hf_repo": "commul/ud_genre",
                "genres_revision": "2.17",
            },
        }
    )
    return cfg


def _write_release_data(release_dir):
    release_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{
            "treebank": "xx_demo",
            "split": "train",
            "sent_id": "1",
            "genre": "news",
            "method": "single-genre-treebank",
        }]
    ).to_parquet(release_dir / "all_genres.parquet")


def test_list_release_publish_files_uses_minimal_payload(tmp_path):
    for path in [
        "README.md",
        "all_genres.parquet",
        "release_manifest.json",
        "config.snapshot.yaml",
    ]:
        (tmp_path / path).write_text("stub", encoding="utf-8")

    assert [
        path.relative_to(tmp_path).as_posix()
        for path in list_release_publish_files(tmp_path)
    ] == ["README.md", "all_genres.parquet", "release_manifest.json"]


def test_publish_release_directory_to_hf_git_commits_minimal_payload(tmp_path):
    source_repo = _make_source_repo(tmp_path)
    hf_repo = tmp_path / "hf"
    _init_git_repo(hf_repo)

    cfg = _make_publish_config(tmp_path)
    release_dir = tmp_path / "release"
    _write_release_data(release_dir)

    result = publish_release_directory_to_hf_git(
        cfg,
        release_dir,
        hf_repo,
        include_main=True,
        source_repo_dir=source_repo,
    )

    assert result["files"] == ["README.md", "all_genres.parquet", "release_manifest.json"]
    assert (hf_repo / "README.md").exists()
    assert (hf_repo / "all_genres.parquet").exists()
    assert (hf_repo / "release_manifest.json").exists()
    assert not (hf_repo / "config.snapshot.yaml").exists()
    assert _git(hf_repo, "rev-parse", "--abbrev-ref", "HEAD") == "2.17"
    assert _git(hf_repo, "rev-parse", "artifact/ud2.17-full-ud-v1^{}") == result["hf_commit"]
    assert _git(hf_repo, "rev-parse", "main") == result["hf_commit"]


def test_publish_release_directory_rejects_dirty_source_repo(tmp_path):
    source_repo = _make_source_repo(tmp_path)
    (source_repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    hf_repo = tmp_path / "hf"
    _init_git_repo(hf_repo)

    cfg = _make_publish_config(tmp_path)
    release_dir = tmp_path / "release"
    _write_release_data(release_dir)

    with pytest.raises(ValueError, match="Source repository must be clean"):
        publish_release_directory_to_hf_git(
            cfg,
            release_dir,
            hf_repo,
            source_repo_dir=source_repo,
        )
