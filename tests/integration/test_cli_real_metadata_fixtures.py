"""Real-fixture CLI integration tests for metadata extraction flows."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import yaml
from datasets import Dataset
from datasets import load_dataset as datasets_load_dataset
from typer.testing import CliRunner

import ud_genre_bootstrap.cli as cli_module
import ud_genre_bootstrap.utils.data_loader as data_loader_module


def _write_config(
    config_path: Path,
    ud_source: str,
    output_dir: Path,
    cache_dir: Path,
    mapping_path: Path,
    patterns_path: Path,
) -> Path:
    """Write a minimal config file for CLI integration tests."""
    config = {
        "ud_version": "2.17",
        "ud_source": ud_source,
        "include_treebanks": ["xx_demo"],
        "embeddings": {
            "model": "xlm-roberta-base",
            "pooling": "mean",
            "batch_size": 2,
            "device": "cpu",
            "cache_dir": str(cache_dir),
        },
        "clustering": {
            "method": "kmeans",
            "device": "cpu",
            "max_iter": 10,
        },
        "evaluation": {
            "metadata_validation": {
                "k": 1,
                "group_by": "treebank",
                "coverage_threshold": 0.95,
            },
        },
        "genre_extraction": {
            "mapping_path": str(mapping_path),
            "patterns_path": str(patterns_path),
        },
        "output": {
            "genres_path": str(output_dir),
            "push_to_hub": False,
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


@pytest.fixture(params=["local", "hf"])
def real_fixture_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
):
    """Build real metadata fixtures for local and HF-backed paths."""
    source_mode = request.param
    ud_root = tmp_path / "ud"
    treebank_dir = ud_root / "UD_Demo" / "r2.17"
    treebank_dir.mkdir(parents=True, exist_ok=True)

    conllu_path = treebank_dir / "xx_demo-ud-train.conllu"
    conllu_path.write_text(
        "# sent_id = n001\n"
        "# text = Local sentence for news.\n"
        "1\tLocal\tlocal\tADJ\t_\t_\t2\tamod\t_\t_\n"
        "2\tsentence\tsentence\tNOUN\t_\t_\t0\troot\t_\t_\n"
        "3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_\n"
        "\n"
        "# sent_id = w001\n"
        "# text = Local sentence for wiki.\n"
        "1\tLocal\tlocal\tADJ\t_\t_\t2\tamod\t_\t_\n"
        "2\tsentence\tsentence\tNOUN\t_\t_\t0\troot\t_\t_\n"
        "3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_\n"
        "\n",
        encoding="utf-8",
    )

    parquet_path = tmp_path / "xx_demo-train.parquet"
    Dataset.from_list(
        [
            {
                "sent_id": "n001",
                "text": "HF sentence for news.",
                "comments": ["__SENT_ID__", "__TEXT__"],
            },
            {
                "sent_id": "w001",
                "text": "HF sentence for wiki.",
                "comments": ["__SENT_ID__", "__TEXT__"],
            },
        ]
    ).to_parquet(parquet_path)

    mapping_path = tmp_path / "genre_mappings.json"
    mapping_path.write_text('{"n": "news", "w": "wiki"}', encoding="utf-8")

    patterns_path = tmp_path / "metadata_patterns.json"
    patterns_path.write_text(
        '{"xx_demo": [{"pattern": "# sent_id = ([nw])", "genre": "$1"}]}',
        encoding="utf-8",
    )

    metadata: Dict = {
        "xx_demo": {
            "lcode": "xx",
            "genre": ["news", "wiki"],
            "splits": {
                "train": {
                    "files": ["UD_Demo/r2.17/xx_demo-ud-train.conllu"],
                }
            },
        }
    }
    monkeypatch.setattr(data_loader_module.UDDataLoader, "_load_metadata", lambda self: metadata)

    if source_mode == "hf":
        def _fake_load_dataset(repo_id: str, treebank_code: str, split: str, revision: str):
            assert repo_id == "dummy/repo"
            assert treebank_code == "xx_demo"
            return datasets_load_dataset(
                "parquet",
                data_files={split: str(parquet_path)},
                split=split,
            )

        monkeypatch.setattr(data_loader_module, "load_dataset", _fake_load_dataset)
        ud_source = "hf://dummy/repo"
    else:
        ud_source = f"local://{ud_root}"

    output_dir = tmp_path / "output"
    cache_dir = tmp_path / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    config_path = _write_config(
        config_path=tmp_path / f"config-{source_mode}.yaml",
        ud_source=ud_source,
        output_dir=output_dir,
        cache_dir=cache_dir,
        mapping_path=mapping_path,
        patterns_path=patterns_path,
    )

    return {
        "mode": source_mode,
        "config_path": config_path,
    }


def test_coverage_with_real_metadata_fixtures(real_fixture_setup):
    """Coverage command should report full coverage on real fixture data."""
    runner = CliRunner()
    result = runner.invoke(
        cli_module.app,
        [
            "coverage",
            "--config",
            str(real_fixture_setup["config_path"]),
            "--treebank",
            "xx_demo",
            "--threshold",
            "1.0",
            "--no-splits",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Fully covered: 1 treebanks" in result.stdout


def test_test_genres_with_real_metadata_fixtures(real_fixture_setup):
    """Pattern extraction should produce complete genre coverage on fixture data."""
    runner = CliRunner()
    result = runner.invoke(
        cli_module.app,
        [
            "test-genres",
            "--config",
            str(real_fixture_setup["config_path"]),
            "--treebank",
            "xx_demo",
            "--split",
            "train",
            "--limit",
            "0",
            "--no-examples",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Coverage" in result.stdout
    assert "100.0%" in result.stdout
    assert "news" in result.stdout
    assert "wiki" in result.stdout


class _StubClusteringEvaluator:
    """Lightweight evaluator for CLI evaluate flow."""

    def __init__(
        self,
        n_folds: int,
        group_by: str,
        min_confidence: float,
        min_margin: float,
        max_iterations: int = 10,
        anchor_mode: str = "strict",
    ):
        self.n_folds = n_folds

    def k_fold_validate(
        self,
        multi_genre_treebanks,
        sentence_metadata,
        embeddings_by_tb,
        clusterer,
        single_genre_treebanks=None,
    ):
        return {
            "mean_accuracy": 0.5,
            "std_accuracy": 0.0,
            "overall_accuracy": 0.5,
            "num_folds": self.n_folds,
            "fold_accuracies": [0.5] * self.n_folds,
        }


def test_evaluate_with_real_metadata_scan(real_fixture_setup, monkeypatch: pytest.MonkeyPatch):
    """Evaluate command should scan real metadata and reach CV stage."""
    monkeypatch.setattr(
        "ud_genre_bootstrap.evaluation.validator.ClusteringEvaluator",
        _StubClusteringEvaluator,
    )

    def _fake_generate_embeddings(self, treebank_filter=None, overwrite=False):
        return {
            ("xx_demo", "train"): {
                "sent_id": ["n001", "w001"],
                "embedding": np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float32),
            }
        }

    monkeypatch.setattr(
        cli_module.GenreBootstrapper,
        "_generate_embeddings",
        _fake_generate_embeddings,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.app,
        [
            "evaluate",
            "--config",
            str(real_fixture_setup["config_path"]),
            "--treebank",
            "xx_demo",
            "--n-folds",
            "1",
            "--group-by",
            "treebank",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Found 1 multi-genre treebank splits for evaluation" in result.stdout
    assert "Cross-Validation Results" in result.stdout
