import json
import subprocess

import pandas as pd
import pytest
import yaml

from ud_genre_bootstrap.utils.config import Config
from ud_genre_bootstrap.utils.release_artifacts import (
    list_release_publish_files,
    prepare_release_directory,
    publish_release_directory_to_hf_git,
    write_release_artifacts,
)


def test_write_release_artifacts_records_identity_and_provenance(tmp_path):
    mapping_path = tmp_path / "genre_mappings.json"
    mapping_path.write_text('{"news": "news"}\n', encoding="utf-8")
    baseline_path = tmp_path / "baseline_summary.json"
    baseline_path.write_text(
        json.dumps({
            "name": "UD v2.17 all_focused generalization baseline",
            "description": "Locked release-train quality context.",
            "protocol": "generalization",
            "ud_version": "2.17",
            "treebank_set": "all_focused",
            "config": "configs/sweeps/baseline.yaml",
            "source_log": "output/logs/baseline.log",
            "metrics": {
                "overall_micro_f1": 0.3333,
                "macro_f1": 0.2636,
                "purity": 0.5568,
                "agreement_treebank": 0.5922,
                "overlap_error_treebank": 0.0589,
                "missing_anchor_genres": ["email", "government"],
            },
        }),
        encoding="utf-8",
    )
    config_path = tmp_path / "release.yaml"
    config_path.write_text("ud_version: '2.17'\n", encoding="utf-8")
    all_genres_path = tmp_path / "all_genres.parquet"
    all_genres_path.write_text("stub", encoding="utf-8")

    cfg = Config.from_dict(
        {
            "ud_version": "2.17",
            "release": {
                "train_id": "full-ud-v1.0.2",
                "artifact_key": "full-ud-v1.0.2-ud2.17",
                "scope": "full",
                "label_schema": "ud",
                "artifact_version": "v1.0.2",
                "inventory_status": "partial",
                "hf_repo": "commul/ud_genre",
                "hf_branches": ["2.17"],
                "hf_tag": "artifact/full-ud-v1.0.2/ud2.17",
                "source_repo": "git@github.com:bot-zen/ud-genre-bootstrap.git",
                "source_branch": "release/full-ud-v1",
                "source_tag": "source/full-ud-v1.0.2",
            },
            "genre_extraction": {
                "mapping_path": str(mapping_path),
            },
            "output": {
                "genres_path": str(tmp_path),
                "genres_hf_repo": "commul/ud_genre",
                "genres_revision": "2.17",
                "baseline_summary_path": str(baseline_path),
            },
        }
    )
    setattr(cfg, "_config_path", str(config_path))

    artifacts = write_release_artifacts(
        cfg,
        tmp_path,
        {
            "total_sentences": 4,
            "labeled_sentences": 4,
            "method_counts": {
                "cluster-derived": 2,
                "single-genre-treebank": 1,
                "virtual-split": 1,
            },
            "genre_counts": {"news": 3, "wiki": 1},
            "confidence_summary": {"mean": 0.85, "median": 0.9},
        },
        all_genres_path=all_genres_path,
    )

    run_metadata = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "release_manifest.json").read_text(encoding="utf-8"))
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert artifacts["release_manifest"] == "release_manifest.json"
    assert run_metadata["train_id"] == "full-ud-v1.0.2"
    assert run_metadata["artifact_key"] == "full-ud-v1.0.2-ud2.17"
    assert run_metadata["artifact_id"] == "full-ud-v1.0.2-ud2.17"
    assert run_metadata["inventory_status"] == "partial"
    assert run_metadata["scope"] == "full"
    assert run_metadata["label_schema"] == "ud"
    assert run_metadata["hf_branches"] == ["2.17"]
    assert run_metadata["hf_tag"] == "artifact/full-ud-v1.0.2/ud2.17"
    assert run_metadata["source_branch"] == "release/full-ud-v1"
    assert run_metadata["source_tag"] == "source/full-ud-v1.0.2"
    assert run_metadata["ud_source"] == "hf://universal-dependencies/universal_dependencies"
    assert run_metadata["ud_source_revision"] == "2.17"
    assert run_metadata["config_hash"]
    assert run_metadata["mapping_file_hashes"]["mappings/genre_mappings.json"]
    assert run_metadata["source_files"]["config"]["path"] == str(config_path)
    assert run_metadata["source_files"]["mappings"][0]["path"] == str(mapping_path)
    assert run_metadata["algorithm_recipe"]["embeddings"]["model"] == cfg.embeddings.model
    assert "thresholds" not in run_metadata["algorithm_recipe"]
    assert run_metadata["label_summary"]["total_sentences"] == 4
    assert run_metadata["label_summary"]["provenance_groups"]["clustering_derived"][
        "count"
    ] == 2
    assert run_metadata["evaluation_summary"]["protocol"] == "generalization"
    assert run_metadata["evaluation_summary"]["metrics"]["overall_micro_f1"] == 0.3333
    assert "news" in run_metadata["canonical_genres"]
    assert run_metadata["noncanonical_genre_counts"] == {}
    assert run_metadata["canonical_methods"] == [
        "single-genre-treebank",
        "virtual-split",
        "cluster-derived",
    ]
    assert run_metadata["noncanonical_method_counts"] == {}

    assert manifest["train_id"] == run_metadata["train_id"]
    assert manifest["artifact_key"] == run_metadata["artifact_key"]
    assert manifest["artifact_id"] == run_metadata["artifact_id"]
    assert manifest["ud_source"] == run_metadata["ud_source"]
    assert manifest["ud_source_revision"] == run_metadata["ud_source_revision"]
    assert manifest["hf_payload"] == [
        "README.md",
        "all_genres.parquet",
        "release_manifest.json",
    ]
    assert manifest["mapping_file_hashes"] == run_metadata["mapping_file_hashes"]
    assert manifest["label_summary"] == run_metadata["label_summary"]
    assert manifest["evaluation_summary"] == run_metadata["evaluation_summary"]
    assert manifest["canonical_genres"] == run_metadata["canonical_genres"]
    assert manifest["noncanonical_genre_counts"] == {}
    assert manifest["canonical_methods"] == run_metadata["canonical_methods"]
    assert manifest["noncanonical_method_counts"] == {}
    assert readme.startswith("---\n")
    card_metadata = yaml.safe_load(readme.split("---", 2)[1])
    assert card_metadata["pretty_name"] == "UD Genre Labels 2.17"
    assert card_metadata["license"] == "apache-2.0"
    assert card_metadata["annotations_creators"] == ["machine-generated"]
    assert card_metadata["language_creators"] == ["crowdsourced"]
    assert card_metadata["multilinguality"] == ["multilingual"]
    assert card_metadata["task_categories"] == ["text-classification"]
    assert "universal-dependencies" in card_metadata["tags"]
    assert "derived-annotations" in card_metadata["tags"]
    assert card_metadata["size_categories"] == ["n<1K"]
    assert card_metadata["configs"] == [
        {
            "config_name": "default",
            "data_files": [
                {
                    "split": "train",
                    "path": "all_genres.parquet",
                }
            ],
            "default": True,
        }
    ]
    assert "revision=\"2.17\"" in readme
    assert "revision=\"artifact/full-ud-v1.0.2/ud2.17\"" in readme
    assert "## Dataset Description" in readme
    assert "- Repository: https://github.com/bot-zen/ud-genre-bootstrap" in readme
    assert "- Point of Contact: appliedlinguisticsdevs@eurac.edu" in readme
    assert "https://universaldependencies.org/udw26/papers/41_Paper.pdf" in readme
    assert (
        "[universal-dependencies/universal_dependencies]"
        "(https://huggingface.co/datasets/universal-dependencies/universal_dependencies)"
        in readme
    )
    assert "The `train` split is the single exported split" in readme
    assert "## Joining With Universal Dependencies" in readme
    assert "Train ID: `full-ud-v1.0.2`" in readme
    assert "Artifact key: `full-ud-v1.0.2-ud2.17`" in readme
    assert "Label schema: `ud`" in readme
    assert "Canonical labels: `academic" in readme
    assert "Source repo: `https://github.com/bot-zen/ud-genre-bootstrap`" in readme
    assert "`run_id`: compact row-level provenance" in readme
    assert "## Label Coverage And Provenance" in readme
    assert "- Total UD sentences in artifact: `4`" in readme
    assert "- Clustering-derived labels: `2` (50.0% of labeled sentences)" in readme
    assert "| `cluster-derived` | Assigned by cluster-to-reference genre similarity." in readme
    assert "## Genre Distribution" in readme
    assert "| `news` | 3 | 75.0% |" in readme
    assert "## Evaluation Summary" in readme
    assert "- Protocol: `generalization`" in readme
    assert "| Overall Acc / Micro-F1 | 0.3333 |" in readme
    assert "Missing anchor genres in this baseline: `email, government`" in readme


def test_prepare_release_directory_rejects_legacy_method_values(tmp_path):
    labels_path = tmp_path / "all_genres.parquet"
    pd.DataFrame(
        [
            {
                "treebank": "tb",
                "split": "train",
                "sent_id": "s1",
                "genre": "news",
                "confidence": 0.5,
                "method": "bootstrap-inferred",
            }
        ]
    ).to_parquet(labels_path, index=False)

    cfg = Config.from_dict(
        {
            "ud_version": "2.18",
            "release": {
                "train_id": "full-ud-v1.1.0",
                "scope": "full",
                "label_schema": "ud",
                "artifact_version": "v1.1.0",
            },
            "output": {"genres_path": str(tmp_path)},
        }
    )

    with pytest.raises(ValueError, match="non-canonical method values"):
        prepare_release_directory(cfg, tmp_path)


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


def _make_source_repo(tmp_path, source_tag="source/full-ud-v1.0.2"):
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
                "train_id": "full-ud-v1.0.2",
                "artifact_key": "full-ud-v1.0.2-ud2.17",
                "scope": "full",
                "label_schema": "ud",
                "artifact_version": "v1.0.2",
                "hf_repo": "commul/ud_genre",
                "hf_branches": ["2.17"],
                "hf_tag": "artifact/full-ud-v1.0.2/ud2.17",
                "hf_default_branch": "main",
                "source_branch": "release/full-ud-v1",
                "source_tag": "source/full-ud-v1.0.2",
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


def test_prepare_release_directory_rejects_noncanonical_exported_labels(tmp_path):
    cfg = Config.from_dict(
        {
            "ud_version": "2.18",
            "release": {
                "train_id": "full-ud-v1.0.2",
                "artifact_key": "full-ud-v1.0.2-ud2.18",
                "scope": "full",
                "label_schema": "ud",
                "artifact_version": "v1.0.2",
                "hf_repo": "commul/ud_genre",
                "hf_branches": ["2.18"],
                "hf_tag": "artifact/full-ud-v1.0.2/ud2.18",
                "source_branch": "release/full-ud-v1",
                "source_tag": "source/full-ud-v1.0.2",
            },
            "genre_extraction": {
                "canonical_genres": ["news", "grammar-examples"],
            },
            "output": {
                "genres_path": str(tmp_path),
                "genres_hf_repo": "commul/ud_genre",
                "genres_revision": "2.18",
            },
        }
    )
    pd.DataFrame(
        [
            {
                "treebank": "nhi_mesotree",
                "split": "test",
                "sent_id": "1",
                "genre": "examples",
                "method": "single-genre-treebank",
            }
        ]
    ).to_parquet(tmp_path / "all_genres.parquet")

    with pytest.raises(ValueError, match="examples=1"):
        prepare_release_directory(cfg, tmp_path)


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
    assert _git(hf_repo, "rev-parse", "artifact/full-ud-v1.0.2/ud2.17^{}") == result["hf_commit"]
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
