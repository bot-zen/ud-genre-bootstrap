"""Integration coverage for CLI command flows across HF and local sources."""

from __future__ import annotations

import sys
import types
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import ud_genre_bootstrap.cli as cli_module
from ud_genre_bootstrap.utils.config import Config


@pytest.fixture(params=["hf://commul/universal_dependencies", "local:///tmp/ud"])
def ud_source(request) -> str:
    """Provide both supported source schemes."""
    return request.param


@pytest.fixture
def cfg(tmp_path: Path, ud_source: str) -> Config:
    """Create a compact config suitable for CLI integration tests."""
    cfg = Config()
    cfg.ud_source = ud_source
    cfg.include_treebanks = ["xx_demo", "yy_demo"]
    cfg.output.genres_path = str(tmp_path / "genres")
    cfg.embeddings.cache_dir = str(tmp_path / "embeddings")
    cfg.evaluation.metadata_validation.k = 2
    cfg.evaluation.metadata_validation.group_by = "treebank"
    cfg.clustering.device = "cpu"
    Path(cfg.output.genres_path).mkdir(parents=True, exist_ok=True)
    Path(cfg.embeddings.cache_dir).mkdir(parents=True, exist_ok=True)
    return cfg


class StubGenreMapper:
    """Simple metadata genre extractor used by stubs."""

    canonical_genres = {"news", "wiki"}

    def __init__(self, *args, **kwargs):
        pass

    def extract_genres_from_metadata(self, sentence: Dict, treebank_code: str) -> List[str]:
        genre = sentence.get("genre")
        if genre:
            return [genre]
        return []


class StubDataLoader:
    """Data loader stub that can serve both metadata-only and full sentence flows."""

    last_ud_source: Optional[str] = None
    last_metadata_path: Optional[Path] = None

    def __init__(
        self,
        ud_source: str,
        ud_version: str = "2.17",
        metadata_path: Optional[Path] = None,
    ):
        self.ud_source = ud_source
        self.ud_version = ud_version
        self.metadata_path = metadata_path
        StubDataLoader.last_ud_source = ud_source
        StubDataLoader.last_metadata_path = metadata_path

    def get_treebank_codes(self) -> List[str]:
        return ["xx_demo", "yy_demo"]

    def get_available_splits(self, treebank_code: str) -> List[str]:
        return ["train"]

    def get_all_treebank_metadata(self) -> List[Dict]:
        return [
            {"id": "xx_demo", "genres": ["news", "wiki"], "language": "xx"},
            {"id": "yy_demo", "genres": ["news", "wiki"], "language": "yy"},
        ]

    def get_treebank_genres(self, treebank_code: str) -> List[str]:
        return ["news", "wiki"]

    def iter_treebank_sentences(
        self,
        treebank_code: str,
        split: str = "train",
        metadata_only: bool = False,
    ) -> Iterator[Dict]:
        rows = [
            {
                "sent_id": f"{treebank_code}-1",
                "text": "Sentence one",
                "genre": "news",
                "comments": [],
            },
            {
                "sent_id": f"{treebank_code}-2",
                "text": "Sentence two",
                "genre": "wiki",
                "comments": [],
            },
        ]
        for row in rows:
            if metadata_only:
                yield {
                    "sent_id": row["sent_id"],
                    "text": row["text"],
                    "genre": row["genre"],
                    "comments": row["comments"],
                }
            else:
                yield row

    def load_treebank(self, treebank_code: str, split: str = "train"):
        return list(self.iter_treebank_sentences(treebank_code, split, metadata_only=False))

    def iter_all_treebanks(
        self,
        split: Optional[str] = None,
        treebank_filter: Optional[List[str]] = None,
    ) -> Iterator[Tuple[str, str, List[Dict]]]:
        treebanks = treebank_filter or ["xx_demo", "yy_demo"]
        for tb in treebanks:
            yield tb, "train", list(self.iter_treebank_sentences(tb, "train", metadata_only=False))


class StubBootstrapper:
    """Bootstrapper stub that satisfies all command interactions."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.data_loader = StubDataLoader(cfg.ud_source, cfg.ud_version)
        self.genre_mapper = StubGenreMapper()
        self.clusterer = object()
        self.final_labels = {}
        self.embedding_generator = types.SimpleNamespace(embedding_cache={})
        self.treebank_clusters = {
            ("xx_demo", "train"): {
                "genres": ["news", "wiki"],
                "cluster_result": {
                    "clusters": {
                        0: {"sent_ids": ["xx_demo-1"], "confidence": 0.95},
                        1: {"sent_ids": ["xx_demo-2"], "confidence": 0.55},
                    },
                    "metrics": {},
                },
            }
        }

    def fit(self, treebank_filter: Optional[List[str]] = None) -> Dict:
        return {
            "labeled_sentences": 2,
            "treebanks_processed": len(treebank_filter or ["xx_demo", "yy_demo"]),
        }

    def _generate_embeddings(
        self,
        treebank_filter: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> Dict:
        treebanks = treebank_filter or ["xx_demo", "yy_demo"]
        output = {}
        for tb in treebanks:
            key = (tb, "train")
            output[key] = {
                "sent_id": [f"{tb}-1", f"{tb}-2"],
                "embedding": [[0.1, 0.0], [0.0, 0.1]],
            }
            self.embedding_generator.embedding_cache[f"{tb}_train"] = output[key]
        return output

    def _cluster_treebanks(self, embeddings_by_tb: Dict):
        for (tb, split), _ in embeddings_by_tb.items():
            self.treebank_clusters[(tb, split)] = {
                "genres": ["news", "wiki"],
                "cluster_result": {
                    "clusters": {
                        0: {"sent_ids": [f"{tb}-1"], "confidence": 0.95},
                        1: {"sent_ids": [f"{tb}-2"], "confidence": 0.55},
                    },
                    "metrics": {},
                },
            }

    def _compute_cluster_embeddings(self, embeddings_by_tb: Dict):
        return None

    def _create_schedule(self) -> List[Dict]:
        return [{"known": ["news"], "predict": ["wiki"], "disjunct": []}]

    def execute_bootstrap_labeling(self, schedule: Optional[List[Dict]] = None):
        self.final_labels = {
            "xx_demo-1": ("news", 0.95, "bootstrap-labeled"),
            "xx_demo-2": ("wiki", 0.55, "bootstrap-inferred"),
        }

    def _export_results(self) -> Dict:
        return {
            "labeled_sentences": 2,
            "method_counts": {"bootstrap-labeled": 1, "bootstrap-inferred": 1},
        }

    def load_cluster_state(self, path: Path) -> Dict:
        return self._generate_embeddings()


class StubClusteringEvaluator:
    """Small evaluator stub for evaluate command integration."""

    last_fixed_partition_call: Optional[Dict] = None

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
        protocol: str = "generalization",
    ):
        self.n_folds = n_folds
        self.group_by = group_by
        self.anchor_mode = anchor_mode
        self.anchor_pool_policy = anchor_pool_policy
        self.protocol = protocol

    def k_fold_validate(
        self,
        multi_genre_treebanks: List[Dict],
        sentence_metadata: Dict,
        embeddings_by_tb: Dict,
        clusterer,
        single_genre_treebanks: Optional[List[Dict]] = None,
    ) -> Dict:
        return {
            "mean_accuracy": 0.5,
            "std_accuracy": 0.0,
            "overall_accuracy": 0.5,
            "micro_f1_instance": 0.5,
            "macro_f1_instance": 0.5,
            "purity": 0.5,
            "agreement_treebank": 0.75,
            "overlap_error_treebank": 0.25,
            "agreement_split": 0.70,
            "overlap_error_split": 0.30,
            "instance_labeled_treebanks_treebank": len({tb["treebank"] for tb in multi_genre_treebanks}),
            "instance_labeled_treebanks_split": len(multi_genre_treebanks),
            "evaluation_mode": "cross_validation",
            "evaluation_protocol": self.protocol,
            "anchor_policy": self.anchor_pool_policy,
            "anchor_counts_by_genre": {"news": 1},
            "missing_anchor_genres": [],
            "num_folds": self.n_folds,
            "fold_accuracies": [0.5] * self.n_folds,
        }

    def fixed_partition_validate(
        self,
        test_treebanks: List[Dict],
        train_treebanks: List[Tuple[str, str]],
        sentence_metadata: Dict,
        embeddings_by_tb: Dict,
        clusterer,
        single_genre_treebanks: Optional[List[Dict]] = None,
        scoring_treebanks: Optional[List[Dict]] = None,
    ) -> Dict:
        StubClusteringEvaluator.last_fixed_partition_call = {
            "test_treebanks": test_treebanks,
            "train_treebanks": train_treebanks,
            "single_genre_treebanks": single_genre_treebanks or [],
            "scoring_treebanks": scoring_treebanks or [],
        }
        if self.protocol == "paper_parity":
            assert train_treebanks == []
            assert all("split_keys" in tb for tb in test_treebanks)
            assert all("split" not in tb for tb in test_treebanks)
            assert all(len(tb.get("genres", [])) >= 2 for tb in test_treebanks)
            assert all("split_keys" in tb for tb in (single_genre_treebanks or []))
            assert all(len(tb.get("genres", [])) == 1 for tb in (single_genre_treebanks or []))
            assert all("split_keys" in tb for tb in (scoring_treebanks or []))
            assert all(len(tb.get("genres", [])) >= 2 for tb in (scoring_treebanks or []))
        scored_treebanks = scoring_treebanks or test_treebanks
        return {
            "mean_accuracy": 0.5,
            "std_accuracy": 0.0,
            "overall_accuracy": 0.5,
            "micro_f1_instance": 0.5,
            "macro_f1_instance": 0.5,
            "purity": 0.5,
            "agreement_treebank": 0.75,
            "overlap_error_treebank": 0.25,
            "agreement_split": 0.70,
            "overlap_error_split": 0.30,
            "instance_labeled_treebanks_treebank": len(
                {tb["treebank"] for tb in scored_treebanks}
            ),
            "instance_labeled_treebanks_split": len(scored_treebanks),
            "evaluation_mode": "fixed_partition",
            "evaluation_protocol": self.protocol,
            "anchor_policy": self.anchor_pool_policy,
            "anchor_counts_by_genre": {"news": 1},
            "missing_anchor_genres": [],
            "num_folds": 1,
            "fold_accuracies": [0.5],
        }


def _patch_common_cli(monkeypatch, cfg: Config):
    """Patch common heavy dependencies for command flow tests."""
    monkeypatch.setattr(cli_module, "load_config_from_path", lambda _: cfg)
    monkeypatch.setattr(cli_module, "GenreBootstrapper", StubBootstrapper)
    monkeypatch.setattr(cli_module, "_save_cluster_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_module, "_display_cluster_stats", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_module, "_display_schedule_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_module, "_display_results", lambda *args, **kwargs: None)


def test_pipeline_commands_cover_hf_and_local_sources(monkeypatch, cfg: Config):
    """`run`, `embed`, `cluster`, `label`, and `info` should execute for both sources."""
    _patch_common_cli(monkeypatch, cfg)
    runner = CliRunner()

    for command in (["run"], ["embed"], ["cluster"], ["label"], ["info"]):
        result = runner.invoke(cli_module.app, command)
        assert result.exit_code == 0, f"{command} failed: {result.stdout}"


def test_evaluate_command_cover_hf_and_local_sources(monkeypatch, cfg: Config):
    """`evaluate` should complete for both HF and local source schemes."""
    _patch_common_cli(monkeypatch, cfg)
    monkeypatch.setattr(
        "ud_genre_bootstrap.evaluation.validator.ClusteringEvaluator",
        StubClusteringEvaluator,
    )
    monkeypatch.setattr("ud_genre_bootstrap.utils.genre_mapping.GenreMapper", StubGenreMapper)
    runner = CliRunner()

    result = runner.invoke(cli_module.app, ["evaluate", "--n-folds", "2", "--group-by", "treebank"])
    assert result.exit_code == 0, result.stdout


def test_evaluate_command_supports_sentence_split_map(monkeypatch, cfg: Config, tmp_path: Path):
    """`evaluate` should accept sentence split map filtering options."""
    _patch_common_cli(monkeypatch, cfg)
    monkeypatch.setattr(
        "ud_genre_bootstrap.evaluation.validator.ClusteringEvaluator",
        StubClusteringEvaluator,
    )
    monkeypatch.setattr("ud_genre_bootstrap.utils.genre_mapping.GenreMapper", StubGenreMapper)

    split_map_path = tmp_path / "split_map.csv"
    split_map_path.write_text(
        "partition,global_index,treebank,split,sent_id\n"
        "train,0,xx_demo,train,xx_demo-1\n"
        "train,1,xx_demo,train,xx_demo-2\n"
        "train,2,yy_demo,train,yy_demo-1\n"
        "train,3,yy_demo,train,yy_demo-2\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.app,
        [
            "evaluate",
            "--n-folds",
            "2",
            "--group-by",
            "treebank",
            "--sentence-split-map",
            str(split_map_path),
            "--split-partition",
            "train",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Sentence split map:" in result.stdout


def test_evaluate_command_supports_paper_parity_mode(monkeypatch, cfg: Config, tmp_path: Path):
    """`evaluate --protocol paper_parity` should auto-run strict same-partition evaluation."""
    _patch_common_cli(monkeypatch, cfg)
    monkeypatch.setattr(
        "ud_genre_bootstrap.evaluation.validator.ClusteringEvaluator",
        StubClusteringEvaluator,
    )
    monkeypatch.setattr("ud_genre_bootstrap.utils.genre_mapping.GenreMapper", StubGenreMapper)
    monkeypatch.setattr(
        cli_module,
        "resolve_paper_evaluation_treebank_genres",
        lambda _data_loader: {
            "xx_demo": ["email", "news", "wiki"],
            "yy_demo": ["news", "wiki"],
        },
    )

    monkeypatch.setattr(
        StubDataLoader,
        "get_all_treebank_metadata",
        lambda self: [
            {"id": "xx_demo", "genres": ["news", "wiki"], "language": "xx"},
            {"id": "yy_demo", "genres": ["news", "wiki"], "language": "yy"},
            {"id": "mono_demo", "genres": ["news"], "language": "zz"},
        ],
    )
    monkeypatch.setattr(
        StubDataLoader,
        "get_treebank_genres",
        lambda self, treebank_code: ["news"] if treebank_code == "mono_demo" else ["news", "wiki"],
    )

    original_iter = StubDataLoader.iter_treebank_sentences

    def _iter_treebank_sentences(self, treebank_code: str, split: str = "train", metadata_only: bool = False):
        if treebank_code == "mono_demo":
            rows = [
                {
                    "sent_id": "mono_demo-1",
                    "text": "Mono one",
                    "genre": "news",
                    "comments": [],
                },
                {
                    "sent_id": "mono_demo-2",
                    "text": "Mono two",
                    "genre": "news",
                    "comments": [],
                },
            ]
            for row in rows:
                if metadata_only:
                    yield {
                        "sent_id": row["sent_id"],
                        "text": row["text"],
                        "genre": row["genre"],
                        "comments": row["comments"],
                    }
                else:
                    yield row
            return

        yield from original_iter(self, treebank_code, split, metadata_only)

    monkeypatch.setattr(StubDataLoader, "iter_treebank_sentences", _iter_treebank_sentences)

    split_map_path = tmp_path / "paper_split_map.csv"
    split_map_path.write_text(
        "partition,global_index,treebank,split,sent_id\n"
        "test,0,xx_demo,train,xx_demo-1\n"
        "test,1,xx_demo,train,xx_demo-2\n"
        "test,2,yy_demo,train,yy_demo-1\n"
        "test,3,yy_demo,train,yy_demo-2\n"
        "test,4,mono_demo,train,mono_demo-1\n"
        "test,5,mono_demo,train,mono_demo-2\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.app,
        [
            "evaluate",
            "--protocol",
            "paper_parity",
            "--sentence-split-map",
            str(split_map_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Protocol: paper_parity" in result.stdout
    assert "Paper-parity anchor source:" in result.stdout
    assert "Evaluation Mode: fixed_partition" in result.stdout
    assert "Evaluation Protocol: paper_parity" in result.stdout
    assert "Protocol Deviations: none" in result.stdout

    call = StubClusteringEvaluator.last_fixed_partition_call
    assert call is not None
    assert call["train_treebanks"] == []
    assert {tb["treebank"] for tb in call["test_treebanks"]} == {"xx_demo", "yy_demo"}
    assert any(
        tb["treebank"] == "xx_demo" and tb["genres"] == ["news", "wiki"]
        for tb in call["test_treebanks"]
    )
    assert any(
        tb["treebank"] == "xx_demo" and tb["genres"] == ["email", "news", "wiki"]
        for tb in call["scoring_treebanks"]
    )
    assert {tb["treebank"] for tb in call["single_genre_treebanks"]} == {"mono_demo"}
    assert {tb["treebank"] for tb in call["scoring_treebanks"]} == {"xx_demo", "yy_demo"}


def test_evaluate_command_restricts_paper_parity_to_paper_scope(monkeypatch, cfg: Config, tmp_path: Path):
    """Paper-parity mode should exclude non-paper evaluation targets while keeping flow valid."""
    _patch_common_cli(monkeypatch, cfg)
    monkeypatch.setattr(
        "ud_genre_bootstrap.evaluation.validator.ClusteringEvaluator",
        StubClusteringEvaluator,
    )
    monkeypatch.setattr("ud_genre_bootstrap.utils.genre_mapping.GenreMapper", StubGenreMapper)
    monkeypatch.setattr(
        cli_module,
        "resolve_paper_evaluation_treebank_genres",
        lambda _data_loader: {"xx_demo": ["news", "wiki"]},
    )
    monkeypatch.setattr(
        StubDataLoader,
        "get_all_treebank_metadata",
        lambda self: [
            {"id": "xx_demo", "genres": ["news", "wiki"], "language": "xx"},
            {"id": "yy_demo", "genres": ["news", "wiki"], "language": "yy"},
            {"id": "mono_demo", "genres": ["news"], "language": "zz"},
        ],
    )
    monkeypatch.setattr(
        StubDataLoader,
        "get_treebank_genres",
        lambda self, treebank_code: ["news"] if treebank_code == "mono_demo" else ["news", "wiki"],
    )

    original_iter = StubDataLoader.iter_treebank_sentences

    def _iter_treebank_sentences(self, treebank_code: str, split: str = "train", metadata_only: bool = False):
        if treebank_code == "mono_demo":
            rows = [
                {
                    "sent_id": "mono_demo-1",
                    "text": "Mono one",
                    "genre": "news",
                    "comments": [],
                },
                {
                    "sent_id": "mono_demo-2",
                    "text": "Mono two",
                    "genre": "news",
                    "comments": [],
                },
            ]
            for row in rows:
                if metadata_only:
                    yield {
                        "sent_id": row["sent_id"],
                        "text": row["text"],
                        "genre": row["genre"],
                        "comments": row["comments"],
                    }
                else:
                    yield row
            return

        yield from original_iter(self, treebank_code, split, metadata_only)

    monkeypatch.setattr(StubDataLoader, "iter_treebank_sentences", _iter_treebank_sentences)

    split_map_path = tmp_path / "paper_scope_split_map.csv"
    split_map_path.write_text(
        "partition,global_index,treebank,split,sent_id\n"
        "test,0,xx_demo,train,xx_demo-1\n"
        "test,1,xx_demo,train,xx_demo-2\n"
        "test,2,yy_demo,train,yy_demo-1\n"
        "test,3,yy_demo,train,yy_demo-2\n"
        "test,4,mono_demo,train,mono_demo-1\n"
        "test,5,mono_demo,train,mono_demo-2\n",
        encoding="utf-8",
    )


    runner = CliRunner()
    result = runner.invoke(
        cli_module.app,
        [
            "evaluate",
            "--protocol",
            "paper_parity",
            "--sentence-split-map",
            str(split_map_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Paper sentence-evaluation scope:" in result.stdout
    assert "excluding 1 non-paper treebank(s): yy_demo" in result.stdout
    assert "Instance-labeled Treebanks (treebank-level): 1" in result.stdout

    call = StubClusteringEvaluator.last_fixed_partition_call
    assert call is not None
    assert {tb["treebank"] for tb in call["test_treebanks"]} == {"xx_demo", "yy_demo"}
    assert {tb["treebank"] for tb in call["scoring_treebanks"]} == {"xx_demo"}
    assert {tb["treebank"] for tb in call["single_genre_treebanks"]} == {"mono_demo"}


def test_evaluate_command_supports_fixed_partition_mode(
    monkeypatch, cfg: Config, tmp_path: Path
):
    """`evaluate --fixed-partition` should run a single holdout evaluation."""
    _patch_common_cli(monkeypatch, cfg)
    monkeypatch.setattr(
        "ud_genre_bootstrap.evaluation.validator.ClusteringEvaluator",
        StubClusteringEvaluator,
    )
    monkeypatch.setattr("ud_genre_bootstrap.utils.genre_mapping.GenreMapper", StubGenreMapper)

    split_map_path = tmp_path / "fixed_split_map.csv"
    split_map_path.write_text(
        "partition,global_index,treebank,split,sent_id\n"
        "train,0,xx_demo,train,xx_demo-1\n"
        "train,1,xx_demo,train,xx_demo-2\n"
        "train,2,yy_demo,train,yy_demo-1\n"
        "train,3,yy_demo,train,yy_demo-2\n"
        "test,4,xx_demo,train,xx_demo-1\n"
        "test,5,xx_demo,train,xx_demo-2\n"
        "test,6,yy_demo,train,yy_demo-1\n"
        "test,7,yy_demo,train,yy_demo-2\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.app,
        [
            "evaluate",
            "--fixed-partition",
            "--sentence-split-map",
            str(split_map_path),
            "--anchor-partition",
            "train",
            "--test-partition",
            "test",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Fixed-partition mode:" in result.stdout
    assert "Running fixed-partition holdout evaluation" in result.stdout


def test_evaluate_multi_set_comparison_shows_extended_metrics(monkeypatch, cfg: Config):
    """`evaluate` comparison table should include extended clustering metrics."""
    _patch_common_cli(monkeypatch, cfg)
    monkeypatch.setattr(
        "ud_genre_bootstrap.evaluation.validator.ClusteringEvaluator",
        StubClusteringEvaluator,
    )
    monkeypatch.setattr("ud_genre_bootstrap.utils.genre_mapping.GenreMapper", StubGenreMapper)
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "evaluate",
            "--n-folds",
            "2",
            "--group-by",
            "treebank",
            "--treebank-set",
            "set_a=xx_demo,yy_demo",
            "--treebank-set",
            "set_b=yy_demo,xx_demo",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Evaluation Set Comparison" in result.stdout
    assert "Macro-F1" in result.stdout
    assert "PUR" in result.stdout
    assert "AGR" in result.stdout
    assert "ΔBC" in result.stdout


def test_build_sentence_split_map_command(tmp_path: Path):
    """Split-map converter command should emit explicit sentence rows."""
    treebanks_root = tmp_path / "ud"
    treebank_dir = treebanks_root / "UD_Demo"
    treebank_dir.mkdir(parents=True, exist_ok=True)
    (treebank_dir / "xx_demo-ud-train.conllu").write_text(
        "# sent_id = n001\n"
        "# text = One.\n"
        "1\tOne\tone\tNUM\t_\t_\t0\troot\t_\t_\n"
        "\n"
        "# sent_id = n002\n"
        "# text = Two.\n"
        "1\tTwo\ttwo\tNUM\t_\t_\t0\troot\t_\t_\n"
        "\n",
        encoding="utf-8",
    )
    (treebanks_root / "metadata.json").write_text(
        json.dumps(
            {
                "xx_demo": {
                    "lcode": "xx",
                    "genre": ["news", "wiki"],
                    "splits": {
                        "train": {
                            "files": ["UD_Demo/xx_demo-ud-train.conllu"],
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    split_pickle_path = tmp_path / "split.pkl"
    import pickle

    with split_pickle_path.open("wb") as handle:
        pickle.dump({"train": [0], "dev": [1], "test": []}, handle)

    output_path = tmp_path / "split_map.csv"
    runner = CliRunner()
    result = runner.invoke(
        cli_module.app,
        [
            "build-sentence-split-map",
            "--ud-source",
            f"local://{treebanks_root}",
            "--split-pickle",
            str(split_pickle_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert output_path.exists()
    frame = pd.read_csv(output_path)
    assert list(frame["sent_id"]) == ["n001", "n002"]
    assert list(frame["partition"]) == ["train", "dev"]


def test_coverage_command_cover_hf_and_local_sources(monkeypatch, cfg: Config):
    """`coverage` should run against the shared analyzer path for both sources."""
    _patch_common_cli(monkeypatch, cfg)
    runner = CliRunner()

    result = runner.invoke(cli_module.app, ["coverage", "--threshold", "0.5", "--no-splits"])
    assert result.exit_code == 0, result.stdout


def test_test_genres_command_cover_hf_and_local_sources(monkeypatch, cfg: Config):
    """`test-genres` should use configured source when initializing data loader."""
    _patch_common_cli(monkeypatch, cfg)
    cfg.metadata_path = "/tmp/test-metadata.json"
    monkeypatch.setattr("ud_genre_bootstrap.utils.data_loader.UDDataLoader", StubDataLoader)
    monkeypatch.setattr("ud_genre_bootstrap.utils.genre_mapping.GenreMapper", StubGenreMapper)
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "test-genres",
            "--treebank",
            "xx_demo",
            "--split",
            "train",
            "--limit",
            "2",
            "--no-examples",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert StubDataLoader.last_ud_source == cfg.ud_source
    assert StubDataLoader.last_metadata_path == Path("/tmp/test-metadata.json")


def _install_fake_xgenre_modules(monkeypatch, tmp_path: Path):
    """Install lightweight stand-ins for transformers/torch/plotting dependencies."""
    class _DummyTensor(list):
        def to(self, device):
            return self

    class _DummyTokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

        def __call__(
            self,
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ):
            n = len(texts)
            return {
                "input_ids": _DummyTensor([[1, 2, 3]] * n),
                "attention_mask": _DummyTensor([[1, 1, 1]] * n),
            }

    class _DummyModel:
        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **inputs):
            n = len(inputs["input_ids"])
            return types.SimpleNamespace(logits=[[0.0] * 9 for _ in range(n)])

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoTokenizer = _DummyTokenizer
    transformers_mod.AutoModelForSequenceClassification = _DummyModel
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    class _DummyPred:
        def __init__(self, idx: int):
            self._idx = idx

        def item(self):
            return self._idx

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    torch_mod = types.ModuleType("torch")
    torch_mod.device = lambda value: value
    torch_mod.no_grad = lambda: _NoGrad()
    torch_mod.argmax = lambda logits, dim=-1: [_DummyPred(2) for _ in logits]  # "News"
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    seaborn_mod = types.ModuleType("seaborn")
    seaborn_mod.heatmap = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "seaborn", seaborn_mod)

    pyplot_mod = types.ModuleType("matplotlib.pyplot")
    pyplot_mod.figure = lambda *args, **kwargs: None
    pyplot_mod.xlabel = lambda *args, **kwargs: None
    pyplot_mod.ylabel = lambda *args, **kwargs: None
    pyplot_mod.title = lambda *args, **kwargs: None
    pyplot_mod.tight_layout = lambda *args, **kwargs: None
    pyplot_mod.close = lambda *args, **kwargs: None
    pyplot_mod.savefig = lambda path, **kwargs: Path(path).write_bytes(b"fake")

    matplotlib_mod = types.ModuleType("matplotlib")
    matplotlib_mod.pyplot = pyplot_mod
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib_mod)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot_mod)


def test_evaluate_xgenre_command_cover_hf_and_local_sources(
    monkeypatch,
    cfg: Config,
    tmp_path: Path,
):
    """`evaluate-xgenre` should execute end-to-end for both source schemes."""
    _patch_common_cli(monkeypatch, cfg)
    _install_fake_xgenre_modules(monkeypatch, tmp_path)
    monkeypatch.setattr("ud_genre_bootstrap.utils.data_loader.UDDataLoader", StubDataLoader)

    genres_dir = Path(cfg.output.genres_path)
    genres_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "treebank": "xx_demo",
                "split": "train",
                "sent_id": "xx_demo-1",
                "genre": "news",
                "confidence": 0.95,
                "method": "bootstrap-labeled",
            },
            {
                "treebank": "xx_demo",
                "split": "train",
                "sent_id": "xx_demo-2",
                "genre": "wiki",
                "confidence": 0.55,
                "method": "bootstrap-inferred",
            },
        ]
    ).to_parquet(genres_dir / "all_genres.parquet", index=False)

    runner = CliRunner()
    output_dir = tmp_path / "xgenre_eval"
    result = runner.invoke(cli_module.app, ["evaluate-xgenre", "--output", str(output_dir)])

    assert result.exit_code == 0, result.stdout
    assert (output_dir / "xgenre_predictions.parquet").exists()
    assert (output_dir / "xgenre_metrics.json").exists()
    assert (output_dir / "confusion_matrix.png").exists()


def test_visualize_clusters_command_flow(monkeypatch, tmp_path: Path):
    """`visualize-clusters` should complete with lightweight plotting/reduction stubs."""
    runner = CliRunner()

    clusters_dir = tmp_path / "clusters"
    emb_dir = tmp_path / "emb"
    clusters_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "treebank": "xx_demo",
                "split": "train",
                "sent_id": "xx_demo-1",
                "cluster_id": 0,
                "confidence": 0.95,
            },
            {
                "treebank": "xx_demo",
                "split": "train",
                "sent_id": "xx_demo-2",
                "cluster_id": 1,
                "confidence": 0.55,
            },
        ]
    ).to_parquet(clusters_dir / "cluster_assignments.parquet", index=False)

    pd.DataFrame(
        [
            {"treebank": "xx_demo", "split": "train", "sent_id": "xx_demo-1", "genre": "news"},
            {"treebank": "xx_demo", "split": "train", "sent_id": "xx_demo-2", "genre": "wiki"},
        ]
    ).to_parquet(clusters_dir / "all_genres.parquet", index=False)

    np.save(emb_dir / "xx_demo-train.npy", np.array([[0.1, 0.2], [0.2, 0.1]], dtype=np.float32))
    (emb_dir / "xx_demo-train_ids.txt").write_text("xx_demo-1\nxx_demo-2\n", encoding="utf-8")

    class _DummyUMAP:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, embeddings):
            return np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    umap_mod = types.ModuleType("umap")
    umap_mod.UMAP = _DummyUMAP
    monkeypatch.setitem(sys.modules, "umap", umap_mod)

    class _DummyFigure:
        def update_traces(self, **kwargs):
            return None

        def update_layout(self, **kwargs):
            return None

        def write_html(self, output_path):
            Path(output_path).write_text("<html><body>ok</body></html>", encoding="utf-8")

    plotly_mod = types.ModuleType("plotly")
    express_mod = types.ModuleType("plotly.express")
    express_mod.scatter = lambda *args, **kwargs: _DummyFigure()
    plotly_mod.express = express_mod
    monkeypatch.setitem(sys.modules, "plotly", plotly_mod)
    monkeypatch.setitem(sys.modules, "plotly.express", express_mod)

    output_file = tmp_path / "viz.html"
    result = runner.invoke(
        cli_module.app,
        [
            "visualize-clusters",
            "--clusters",
            str(clusters_dir),
            "--embeddings",
            str(emb_dir),
            "--output",
            str(output_file),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert output_file.exists()
