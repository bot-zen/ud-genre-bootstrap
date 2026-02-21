"""Real-fixture CLI integration tests for metadata extraction flows."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import yaml
from datasets import Dataset
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
    include_treebanks: list[str] | None = None,
    eval_k: int = 1,
    eval_group_by: str = "treebank",
) -> Path:
    """Write a minimal config file for CLI integration tests."""
    config = {
        "ud_version": "2.17",
        "ud_source": ud_source,
        "include_treebanks": include_treebanks or ["xx_demo"],
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
                "k": eval_k,
                "group_by": eval_group_by,
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

    hf_rows = [
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
    parquet_path = tmp_path / "xx_demo-train.parquet"
    Dataset.from_list(hf_rows).to_parquet(parquet_path)

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
            return Dataset.from_list(hf_rows)

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
        anchor_pool_policy: str = "auto",
        reference_weighting: str = "sentence_count",
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


@pytest.fixture(params=["local", "hf"])
def real_eval_golden_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
):
    """Build deterministic multi-split fixtures for golden evaluate-metrics tests."""
    source_mode = request.param
    ud_root = tmp_path / "ud"
    treebank_dir = ud_root / "UD_Demo" / "r2.17"
    treebank_dir.mkdir(parents=True, exist_ok=True)

    def _split_rows(split_name: str):
        return [
            {
                "sent_id": f"n_{split_name}_1",
                "text": f"{split_name} sentence 1 (news).",
                "genre": "news",
            },
            {
                "sent_id": f"n_{split_name}_2",
                "text": f"{split_name} sentence 2 (news).",
                "genre": "news",
            },
            {
                "sent_id": f"w_{split_name}_1",
                "text": f"{split_name} sentence 1 (wiki).",
                "genre": "wiki",
            },
            {
                "sent_id": f"w_{split_name}_2",
                "text": f"{split_name} sentence 2 (wiki).",
                "genre": "wiki",
            },
        ]

    hf_rows_by_split = {}
    for split_name in ("train", "dev"):
        conllu_path = treebank_dir / f"xx_demo-ud-{split_name}.conllu"
        lines = []
        for row in _split_rows(split_name):
            lines.extend(
                [
                    f"# sent_id = {row['sent_id']}",
                    f"# text = {row['text']}",
                    "1\tToken\ttoken\tNOUN\t_\t_\t0\troot\t_\t_",
                    "",
                ]
            )
        conllu_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        parquet_path = tmp_path / f"xx_demo-{split_name}.parquet"
        Dataset.from_list(
            [
                {
                    "sent_id": row["sent_id"],
                    "text": row["text"],
                    "comments": ["__SENT_ID__", "__TEXT__"],
                }
                for row in _split_rows(split_name)
            ]
        ).to_parquet(parquet_path)
        hf_rows_by_split[split_name] = [
            {
                "sent_id": row["sent_id"],
                "text": row["text"],
                "comments": ["__SENT_ID__", "__TEXT__"],
            }
            for row in _split_rows(split_name)
        ]

    mapping_path = tmp_path / "genre_mappings.json"
    mapping_path.write_text('{"n": "news", "w": "wiki"}', encoding="utf-8")

    patterns_path = tmp_path / "metadata_patterns.json"
    patterns_path.write_text(
        '{"xx_demo": [{"pattern": "(?:#\\\\s*)?sent_id\\\\s*=\\\\s*([nw])", "genre": "$1"}]}',
        encoding="utf-8",
    )

    metadata: Dict = {
        "xx_demo": {
            "lcode": "xx",
            "genre": ["news", "wiki"],
            "splits": {
                "train": {
                    "files": ["UD_Demo/r2.17/xx_demo-ud-train.conllu"],
                },
                "dev": {
                    "files": ["UD_Demo/r2.17/xx_demo-ud-dev.conllu"],
                },
            },
        }
    }
    monkeypatch.setattr(data_loader_module.UDDataLoader, "_load_metadata", lambda self: metadata)

    if source_mode == "hf":

        def _fake_load_dataset(repo_id: str, treebank_code: str, split: str, revision: str):
            assert repo_id == "dummy/repo"
            assert treebank_code == "xx_demo"
            return Dataset.from_list(hf_rows_by_split[split])

        monkeypatch.setattr(data_loader_module, "load_dataset", _fake_load_dataset)
        ud_source = "hf://dummy/repo"
    else:
        ud_source = f"local://{ud_root}"

    output_dir = tmp_path / "output"
    cache_dir = tmp_path / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    config_path = _write_config(
        config_path=tmp_path / f"config-golden-{source_mode}.yaml",
        ud_source=ud_source,
        output_dir=output_dir,
        cache_dir=cache_dir,
        mapping_path=mapping_path,
        patterns_path=patterns_path,
        include_treebanks=["xx_demo"],
        eval_k=2,
        eval_group_by="none",
    )

    return {
        "mode": source_mode,
        "config_path": config_path,
    }


def test_evaluate_reports_full_deterministic_metric_bundle(real_eval_golden_setup, monkeypatch: pytest.MonkeyPatch):
    """Evaluate should emit a stable full metric bundle on deterministic fixtures."""
    captured_results = []

    def _capture_results(results: Dict):
        captured_results.append(results)

    def _fake_generate_embeddings(self, treebank_filter=None, overwrite=False):
        genre_to_embedding = {
            "news": np.array([1.0, 0.0], dtype=np.float32),
            "wiki": np.array([0.0, 1.0], dtype=np.float32),
        }
        embeddings_by_tb = {}
        treebanks = treebank_filter or self.data_loader.get_treebank_codes()

        for tb_code in treebanks:
            for split_name in self.data_loader.get_available_splits(tb_code):
                sent_ids = []
                vectors = []
                sentence_iter = self.data_loader.iter_treebank_sentences(
                    tb_code,
                    split_name,
                    metadata_only=True,
                )
                for idx, sentence in enumerate(sentence_iter):
                    sent_id = sentence.get("sent_id", f"{tb_code}_{split_name}_{idx}")
                    genres = self.genre_mapper.extract_genres_from_metadata(sentence, tb_code)
                    if not genres:
                        continue
                    sent_ids.append(sent_id)
                    vectors.append(genre_to_embedding[genres[0]])

                embeddings_by_tb[(tb_code, split_name)] = {
                    "sent_id": sent_ids,
                    "embedding": np.stack(vectors).astype(np.float32),
                }
        return embeddings_by_tb

    monkeypatch.setattr(cli_module.GenreBootstrapper, "_generate_embeddings", _fake_generate_embeddings)
    monkeypatch.setattr(cli_module, "_display_evaluation_results", _capture_results)
    monkeypatch.setattr(cli_module, "_save_clustering_confusion_matrix", lambda *_args, **_kwargs: None)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.app,
        [
            "evaluate",
            "--config",
            str(real_eval_golden_setup["config_path"]),
            "--treebank",
            "xx_demo",
            "--n-folds",
            "2",
            "--group-by",
            "none",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert len(captured_results) == 1

    metrics = captured_results[0]
    expected_keys = {
        "mean_accuracy",
        "std_accuracy",
        "overall_accuracy",
        "confusion_matrix",
        "genre_labels",
        "classification_report",
        "fold_accuracies",
        "num_folds",
        "total_sentences",
        "num_sentences_per_fold",
        "micro_f1_instance",
        "macro_f1_instance",
        "purity",
        "agreement_treebank",
        "agreement_by_genre_treebank",
        "overlap_error_treebank",
        "overlap_error_weighted_treebank",
        "overlap_error_by_treebank_treebank",
        "instance_labeled_treebanks_treebank",
        "agreement_split",
        "agreement_by_genre_split",
        "overlap_error_split",
        "overlap_error_weighted_split",
        "overlap_error_by_treebank_split",
        "instance_labeled_treebanks_split",
        "mean_macro_f1_instance",
        "std_macro_f1_instance",
        "mean_purity",
        "std_purity",
        "mean_agreement_treebank",
        "std_agreement_treebank",
        "mean_overlap_error_treebank",
        "std_overlap_error_treebank",
        "mean_agreement_split",
        "std_agreement_split",
        "mean_overlap_error_split",
        "std_overlap_error_split",
        "anchor_policy",
        "anchor_counts_by_genre",
        "missing_anchor_genres",
        "fold_anchor_diagnostics",
    }
    assert set(metrics.keys()) == expected_keys

    assert metrics["num_folds"] == 2
    assert metrics["total_sentences"] == 8
    assert metrics["num_sentences_per_fold"] == [4, 4]
    assert metrics["fold_accuracies"] == pytest.approx([1.0, 1.0])
    assert metrics["mean_accuracy"] == pytest.approx(1.0)
    assert metrics["std_accuracy"] == pytest.approx(0.0)
    assert metrics["overall_accuracy"] == pytest.approx(1.0)
    assert metrics["micro_f1_instance"] == pytest.approx(1.0)
    assert metrics["macro_f1_instance"] == pytest.approx(1.0)
    assert metrics["purity"] == pytest.approx(1.0)
    assert metrics["agreement_treebank"] == pytest.approx(1.0)
    assert metrics["agreement_split"] == pytest.approx(1.0)
    assert metrics["overlap_error_treebank"] == pytest.approx(0.0)
    assert metrics["overlap_error_split"] == pytest.approx(0.0)
    assert metrics["overlap_error_weighted_treebank"] == pytest.approx(0.0)
    assert metrics["overlap_error_weighted_split"] == pytest.approx(0.0)

    assert metrics["instance_labeled_treebanks_treebank"] == 1
    assert metrics["instance_labeled_treebanks_split"] == 2
    assert metrics["anchor_policy"] == "train_virtual"
    assert metrics["missing_anchor_genres"] == []
    assert len(metrics["fold_anchor_diagnostics"]) == 2
    assert set(metrics["anchor_counts_by_genre"].keys()) == {"news", "wiki"}
    assert metrics["genre_labels"] == ["news", "wiki"]
    assert metrics["confusion_matrix"] == [[4, 0], [0, 4]]

    assert metrics["agreement_by_genre_treebank"] == {"news": 1.0, "wiki": 1.0}
    assert metrics["agreement_by_genre_split"] == {"news": 1.0, "wiki": 1.0}
    assert metrics["overlap_error_by_treebank_treebank"] == {"xx_demo": 0.0}
    assert metrics["overlap_error_by_treebank_split"] == {
        "xx_demo:dev": 0.0,
        "xx_demo:train": 0.0,
    }
