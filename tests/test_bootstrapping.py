"""Tests for bootstrapping functionality."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from ud_genre_bootstrap.bootstrapping.bootstrapper import GenreBootstrapper
from ud_genre_bootstrap.clustering.clustering_utils import ClusteringOperations
from ud_genre_bootstrap.utils.config import Config


class TestGenreExport:
    """Test _export_results() creating all_genres.parquet."""

    def test_export_creates_all_genres_parquet(self, tmp_path):
        """Test that _export_results creates all_genres.parquet with correct structure."""
        # Create a mock bootstrapper with final_labels
        config = Config()
        config.output.genres_path = str(tmp_path)

        bootstrapper = GenreBootstrapper(config)

        # Mock final_labels
        bootstrapper.final_labels = {
            "en_ewt-ud-train-00001": ("news", 0.85, "bootstrap-labeled"),
            "en_ewt-ud-train-00002": ("news", 0.78, "bootstrap-labeled"),
            "de_gsd-ud-test-00042": ("wiki", 0.43, "bootstrap-inferred"),
            "fr_gsd-ud-dev-00015": ("fiction", 0.91, "bootstrap-labeled"),
        }

        # Export results
        results = bootstrapper._export_results()

        # Check that file was created
        output_file = tmp_path / "all_genres.parquet"
        assert output_file.exists(), "all_genres.parquet should be created"

        # Load and verify contents
        df = pd.read_parquet(output_file)

        # Check columns
        assert set(df.columns) == {"sent_id", "genre", "confidence", "method"}

        # Check number of rows
        assert len(df) == 4

        # Check data types
        assert df["sent_id"].dtype == object
        assert df["genre"].dtype == object
        assert df["confidence"].dtype == float
        assert df["method"].dtype == object

        # Check specific values
        news_row = df[df["sent_id"] == "en_ewt-ud-train-00001"].iloc[0]
        assert news_row["genre"] == "news"
        assert news_row["confidence"] == 0.85
        assert news_row["method"] == "bootstrap-labeled"

        # Check low confidence row
        wiki_row = df[df["sent_id"] == "de_gsd-ud-test-00042"].iloc[0]
        assert wiki_row["genre"] == "wiki"
        assert wiki_row["confidence"] == 0.43
        assert wiki_row["method"] == "bootstrap-inferred"

        # Check results statistics
        assert results["total_sentences"] == 4
        assert results["labeled_sentences"] == 4
        assert results["genre_counts"]["news"] == 2
        assert results["genre_counts"]["wiki"] == 1
        assert results["genre_counts"]["fiction"] == 1
        assert results["method_counts"]["bootstrap-labeled"] == 3
        assert results["method_counts"]["bootstrap-inferred"] == 1

    def test_export_handles_empty_labels(self, tmp_path):
        """Test that export handles empty final_labels gracefully."""
        config = Config()
        config.output.genres_path = str(tmp_path)

        bootstrapper = GenreBootstrapper(config)
        bootstrapper.final_labels = {}

        results = bootstrapper._export_results()

        # Should not create file or create empty file
        output_file = tmp_path / "all_genres.parquet"
        assert not output_file.exists() or pd.read_parquet(output_file).empty

        assert results["total_sentences"] == 0
        assert results["labeled_sentences"] == 0

    def test_export_handles_none_confidence(self, tmp_path):
        """Test that export handles None confidence values."""
        config = Config()
        config.output.genres_path = str(tmp_path)

        bootstrapper = GenreBootstrapper(config)
        bootstrapper.final_labels = {
            "test-001": ("news", None, "metadata"),
        }

        results = bootstrapper._export_results()

        output_file = tmp_path / "all_genres.parquet"
        df = pd.read_parquet(output_file)

        # Confidence should be None (pd.NA in parquet)
        assert pd.isna(df.iloc[0]["confidence"])


class TestCrossLingualReport:
    """Test _generate_cross_lingual_report() functionality."""

    def test_cross_lingual_report_detects_multilingual_genres(self, caplog):
        """Test that cross-lingual report correctly identifies genres across languages."""
        config = Config()
        bootstrapper = GenreBootstrapper(config)

        # Mock genre_combination_clusters with multi-language data
        bootstrapper.genre_combination_clusters = {
            ("news",): {
                ("en_ewt", "train"): [
                    {
                        "cluster_id": 0,
                        "sent_ids": ["en_ewt-train-001", "en_ewt-train-002"],
                        "embedding": np.array([1.0, 2.0, 3.0]),
                        "confidence": 0.85,
                    }
                ],
                ("de_gsd", "test"): [
                    {
                        "cluster_id": 0,
                        "sent_ids": ["de_gsd-test-001"],
                        "embedding": np.array([1.1, 2.1, 3.1]),
                        "confidence": 0.82,
                    }
                ],
            },
            ("wiki",): {
                ("en_ewt", "train"): [
                    {
                        "cluster_id": 1,
                        "sent_ids": ["en_ewt-train-003"],
                        "embedding": np.array([4.0, 5.0, 6.0]),
                        "confidence": 0.78,
                    }
                ],
            },
        }

        # Mock final_labels
        bootstrapper.final_labels = {
            "en_ewt-train-001": ("news", 0.85, "bootstrap-labeled"),
            "en_ewt-train-002": ("news", 0.85, "bootstrap-labeled"),
            "de_gsd-test-001": ("news", 0.82, "bootstrap-labeled"),
            "en_ewt-train-003": ("wiki", 0.78, "bootstrap-labeled"),
        }

        # Generate report
        import logging
        caplog.set_level(logging.INFO)
        bootstrapper._generate_cross_lingual_report()

        # Check log output
        log_text = caplog.text

        # Should report news in multiple languages
        assert "Genre: NEWS" in log_text
        assert "Found in 2 language(s)" in log_text
        assert "en:" in log_text
        assert "de:" in log_text

        # Should report wiki in single language
        assert "Genre: WIKI" in log_text
        assert "Found in 1 language(s)" in log_text

        # Should show cross-lingual consistency check
        assert "CROSS-LINGUAL CONSISTENCY CHECK" in log_text
        assert "Found 1 genre(s) spanning multiple languages" in log_text
        assert "news: de, en" in log_text or "news: en, de" in log_text

    def test_cross_lingual_report_warns_on_no_multilingual_genres(self, caplog):
        """Test that report warns when no genres span multiple languages."""
        config = Config()
        bootstrapper = GenreBootstrapper(config)

        # Mock with only single-language genres
        bootstrapper.genre_combination_clusters = {
            ("news",): {
                ("en_ewt", "train"): [
                    {
                        "cluster_id": 0,
                        "sent_ids": ["en_ewt-train-001"],
                        "embedding": np.array([1.0, 2.0, 3.0]),
                        "confidence": 0.85,
                    }
                ],
            },
            ("wiki",): {
                ("de_gsd", "test"): [
                    {
                        "cluster_id": 0,
                        "sent_ids": ["de_gsd-test-001"],
                        "embedding": np.array([4.0, 5.0, 6.0]),
                        "confidence": 0.78,
                    }
                ],
            },
        }

        bootstrapper.final_labels = {
            "en_ewt-train-001": ("news", 0.85, "bootstrap-labeled"),
            "de_gsd-test-001": ("wiki", 0.78, "bootstrap-labeled"),
        }

        import logging
        caplog.set_level(logging.WARNING)
        bootstrapper._generate_cross_lingual_report()

        log_text = caplog.text

        # Should warn about no cross-lingual genres
        assert "No genres found spanning multiple languages" in log_text
        assert "clustering may be separating by language" in log_text

    def test_cross_lingual_report_calculates_statistics(self, caplog):
        """Test that report correctly calculates cluster and sentence counts."""
        config = Config()
        bootstrapper = GenreBootstrapper(config)

        bootstrapper.genre_combination_clusters = {
            ("news",): {
                ("en_ewt", "train"): [
                    {
                        "cluster_id": 0,
                        "sent_ids": ["en-001", "en-002", "en-003"],
                        "embedding": np.array([1.0, 2.0, 3.0]),
                        "confidence": 0.85,
                    },
                    {
                        "cluster_id": 1,
                        "sent_ids": ["en-004", "en-005"],
                        "embedding": np.array([1.1, 2.1, 3.1]),
                        "confidence": 0.80,
                    },
                ],
                ("de_gsd", "test"): [
                    {
                        "cluster_id": 0,
                        "sent_ids": ["de-001"],
                        "embedding": np.array([1.2, 2.2, 3.2]),
                        "confidence": 0.75,
                    },
                ],
            },
        }

        bootstrapper.final_labels = {
            "en-001": ("news", 0.85, "bootstrap-labeled"),
            "en-002": ("news", 0.85, "bootstrap-labeled"),
            "en-003": ("news", 0.85, "bootstrap-labeled"),
            "en-004": ("news", 0.80, "bootstrap-labeled"),
            "en-005": ("news", 0.80, "bootstrap-labeled"),
            "de-001": ("news", 0.75, "bootstrap-labeled"),
        }

        import logging
        caplog.set_level(logging.INFO)
        bootstrapper._generate_cross_lingual_report()

        log_text = caplog.text

        # Check total counts
        assert "Found in 2 language(s), 3 cluster(s), 6 sentence(s)" in log_text

        # Check per-language counts
        assert "en: 2 cluster(s), 5 sent(s)" in log_text
        assert "de: 1 cluster(s), 1 sent(s)" in log_text

        # Check average confidence (en should be ~0.825, de should be 0.75)
        assert "avg_conf=0.8" in log_text  # English avg
        assert "avg_conf=0.75" in log_text  # German


class TestBootstrapperIntegration:
    """Integration tests for bootstrapper with export."""

    def test_fit_produces_all_genres_file(self, tmp_path, monkeypatch):
        """Test that fit() pipeline produces all_genres.parquet."""
        # This would require a full mock setup or fixture data
        # Skipping for now as it requires extensive mocking
        pytest.skip("Requires full pipeline mocking")


class TestPipelineSegments:
    """Unit tests for shared pipeline segment execution."""

    def test_execute_bootstrap_labeling_runs_all_stages_in_order(self, monkeypatch):
        """execute_bootstrap_labeling() should run schedule -> single -> cluster -> report."""
        bootstrapper = GenreBootstrapper(Config())
        call_order = []
        mock_schedule = [{"known": ["news"], "predict": [], "disjunct": []}]

        def _mock_create_schedule():
            call_order.append("schedule")
            return mock_schedule

        def _mock_label_single():
            call_order.append("single")

        def _mock_label_clusters(schedule):
            call_order.append(("clusters", schedule))

        def _mock_report():
            call_order.append("report")

        monkeypatch.setattr(bootstrapper, "_create_schedule", _mock_create_schedule)
        monkeypatch.setattr(bootstrapper, "_label_single_genre_treebanks", _mock_label_single)
        monkeypatch.setattr(bootstrapper, "_label_clusters", _mock_label_clusters)
        monkeypatch.setattr(bootstrapper, "_generate_cross_lingual_report", _mock_report)

        out_schedule = bootstrapper.execute_bootstrap_labeling()

        assert out_schedule == mock_schedule
        assert call_order == [
            "schedule",
            "single",
            ("clusters", mock_schedule),
            "report",
        ]

    def test_execute_bootstrap_labeling_uses_provided_schedule(self, monkeypatch):
        """Provided schedule should be reused without recomputing."""
        bootstrapper = GenreBootstrapper(Config())
        call_order = []
        provided_schedule = [{"known": ["news"], "predict": [], "disjunct": []}]

        def _mock_create_schedule():
            raise AssertionError("_create_schedule should not be called")

        def _mock_label_single():
            call_order.append("single")

        def _mock_label_clusters(schedule):
            call_order.append(("clusters", schedule))

        def _mock_report():
            call_order.append("report")

        monkeypatch.setattr(bootstrapper, "_create_schedule", _mock_create_schedule)
        monkeypatch.setattr(bootstrapper, "_label_single_genre_treebanks", _mock_label_single)
        monkeypatch.setattr(bootstrapper, "_label_clusters", _mock_label_clusters)
        monkeypatch.setattr(bootstrapper, "_generate_cross_lingual_report", _mock_report)

        out_schedule = bootstrapper.execute_bootstrap_labeling(schedule=provided_schedule)

        assert out_schedule == provided_schedule
        assert call_order == [
            "single",
            ("clusters", provided_schedule),
            "report",
        ]

    def test_label_environment_preserves_metadata_derived_methods(self):
        """bootstrap labeling should not overwrite virtual/single-genre labels."""
        bootstrapper = GenreBootstrapper(Config())
        bootstrapper.genre_combination_clusters = {
            ("news", "wiki"): {
                ("xx_demo", "test"): [
                    {
                        "cluster_id": 0,
                        "sent_ids": ["sid_virtual", "sid_single", "sid_new"],
                        "embedding": np.array([1.0, 0.0]),
                        "confidence": 1.0,
                    }
                ]
            }
        }
        bootstrapper.final_labels = {
            "sid_virtual": ("news", 1.0, "virtual-split"),
            "sid_single": ("news", 1.0, "single-genre-treebank"),
        }

        environment = {"predict": [("news", "wiki")]}
        known_embeddings = {
            "news": np.array([1.0, 0.0]),
            "wiki": np.array([0.0, 1.0]),
        }

        bootstrapper._label_environment(environment, known_embeddings)

        assert bootstrapper.final_labels["sid_virtual"] == ("news", 1.0, "virtual-split")
        assert bootstrapper.final_labels["sid_single"] == ("news", 1.0, "single-genre-treebank")
        assert bootstrapper.final_labels["sid_new"][0] == "news"
        assert bootstrapper.final_labels["sid_new"][2] == "bootstrap-labeled"

    def test_label_environment_assigns_threshold_based_methods(self):
        """Clusters should always be labeled, with method decided by uncertainty thresholds."""
        config = Config()
        config.bootstrapping.min_confidence = 0.8
        config.bootstrapping.min_margin = 0.0
        bootstrapper = GenreBootstrapper(config)

        bootstrapper.genre_combination_clusters = {
            ("news", "wiki"): {
                ("xx_demo", "test"): [
                    {
                        "cluster_id": 0,
                        "sent_ids": ["sid_high"],
                        "embedding": np.array([1.0, 0.0]),
                        "confidence": 1.0,
                    },
                    {
                        "cluster_id": 1,
                        "sent_ids": ["sid_low"],
                        "embedding": np.array([0.6, 0.6]),
                        "confidence": 1.0,
                    },
                ]
            }
        }

        environment = {"predict": [("news", "wiki")]}
        known_embeddings = {
            "news": np.array([1.0, 0.0]),
            "wiki": np.array([0.0, 1.0]),
        }

        bootstrapper._label_environment(environment, known_embeddings)

        assert bootstrapper.final_labels["sid_high"][2] == "bootstrap-labeled"
        assert bootstrapper.final_labels["sid_low"][2] == "bootstrap-inferred"
        assert bootstrapper.final_labels["sid_high"][1] >= config.bootstrapping.min_confidence
        assert bootstrapper.final_labels["sid_low"][1] < config.bootstrapping.min_confidence

    def test_label_environment_marks_near_ties_as_inferred_by_margin(self):
        """Near-tie top-1/top-2 scores should be flagged as inferred even at high confidence."""
        config = Config()
        config.bootstrapping.min_confidence = 0.8
        config.bootstrapping.min_margin = 0.02
        bootstrapper = GenreBootstrapper(config)

        bootstrapper.genre_combination_clusters = {
            ("news", "wiki"): {
                ("xx_demo", "test"): [
                    {
                        "cluster_id": 0,
                        "sent_ids": ["sid_tie"],
                        "embedding": np.array([1.0, 0.0]),
                        "confidence": 1.0,
                    },
                ]
            }
        }

        environment = {"predict": [("news", "wiki")]}
        # Very similar references to force a small top1-top2 margin.
        known_embeddings = {
            "news": np.array([1.0, 0.0]),
            "wiki": np.array([0.99, 0.1]),
        }

        bootstrapper._label_environment(environment, known_embeddings)

        label = bootstrapper.final_labels["sid_tie"]
        assert label[0] == "news"
        assert label[1] >= config.bootstrapping.min_confidence
        assert label[2] == "bootstrap-inferred"

    def test_label_environment_uses_shared_cluster_labeling_logic(self, monkeypatch):
        """Bootstrap labeling should delegate cluster scoring to shared clustering ops."""
        bootstrapper = GenreBootstrapper(Config())
        bootstrapper.genre_combination_clusters = {
            ("news", "wiki"): {
                ("xx_demo", "test"): [
                    {
                        "cluster_id": 0,
                        "sent_ids": ["sid_1", "sid_2"],
                        "embedding": np.array([1.0, 0.0]),
                        "confidence": 1.0,
                    }
                ]
            }
        }

        captured = {}

        def _mock_assign_cluster_label(centroid, references):
            captured["centroid"] = centroid
            captured["references"] = references
            return ("wiki", 0.42, "bootstrap-inferred", [("wiki", 0.42), ("news", 0.41)])

        monkeypatch.setattr(
            bootstrapper.clustering_ops,
            "assign_cluster_label",
            _mock_assign_cluster_label,
        )

        environment = {"predict": [("news", "wiki")]}
        known_embeddings = {
            "news": np.array([1.0, 0.0]),
            "wiki": np.array([0.0, 1.0]),
        }

        bootstrapper._label_environment(environment, known_embeddings)

        assert np.array_equal(captured["centroid"], np.array([1.0, 0.0]))
        assert set(captured["references"].keys()) == {"news", "wiki"}
        assert bootstrapper.final_labels["sid_1"] == ("wiki", 0.42, "bootstrap-inferred")
        assert bootstrapper.final_labels["sid_2"] == ("wiki", 0.42, "bootstrap-inferred")


class TestBootstrapperConfigWiring:
    """Tests for clusterer configuration wiring in bootstrapper initialization."""

    def test_kmeans_clusterer_receives_max_iter_from_config(self):
        """K-Means clusterer should use configurable max_iter from clustering config."""
        config = Config()
        config.clustering.method = "kmeans"
        config.clustering.max_iter = 123
        config.bootstrapping.min_margin = 0.11

        bootstrapper = GenreBootstrapper(config)

        assert type(bootstrapper.clusterer).__name__ == "KMeansClusterer"
        assert bootstrapper.clusterer.max_iter == 123
        assert bootstrapper.clustering_ops.min_margin == 0.11

    def test_gmm_clusterer_receives_fit_sample_size_from_config(self):
        """GMM clusterer should use configurable fit_sample_size from clustering config."""
        config = Config()
        config.clustering.method = "gmm"
        config.clustering.fit_sample_size = 50000

        bootstrapper = GenreBootstrapper(config)

        assert type(bootstrapper.clusterer).__name__ == "GMMClusterer"
        assert bootstrapper.clusterer.fit_sample_size == 50000

    def test_cluster_treebanks_uses_configured_virtual_split_quality_gates(self, monkeypatch):
        """Production clustering should read virtual-split quality gates from config."""
        config = Config()
        config.evaluation.metadata_validation.coverage_threshold = 0.93
        config.evaluation.metadata_validation.min_genre_sentences = 42
        bootstrapper = GenreBootstrapper(config)

        embeddings_by_tb = {
            ("xx_demo", "train"): {
                "sent_id": ["sid_1", "sid_2"],
                "embedding": np.array([[1.0, 0.0], [0.0, 1.0]]),
            }
        }

        def _mock_load_treebank(_tb_code, _split):
            raise RuntimeError("mock load failure to force fallback path")

        monkeypatch.setattr(bootstrapper.data_loader, "load_treebank", _mock_load_treebank)
        monkeypatch.setattr(
            bootstrapper.data_loader,
            "get_treebank_genres",
            lambda _tb_code: ["news", "wiki"],
        )

        captured = {}

        def _mock_check_virtual_split_coverage(
            _combined_embeddings,
            _all_sent_ids,
            _sent_id_to_split,
            _sentence_metadata,
            _tb_code,
            coverage_threshold=0.8,
            min_genre_sentences=1,
        ):
            captured["coverage_threshold"] = coverage_threshold
            captured["min_genre_sentences"] = min_genre_sentences
            return False, set()

        monkeypatch.setattr(
            bootstrapper.clustering_ops,
            "check_virtual_split_coverage",
            _mock_check_virtual_split_coverage,
        )

        def _mock_cluster_treebank(embeddings, sent_ids, n_genres):
            return {
                "clusters": {
                    0: {
                        "sent_ids": sent_ids,
                        "size": len(sent_ids),
                        "confidence": 1.0,
                    }
                },
                "metrics": {},
            }

        monkeypatch.setattr(bootstrapper.clusterer, "cluster_treebank", _mock_cluster_treebank)

        bootstrapper._cluster_treebanks(embeddings_by_tb)

        assert captured["coverage_threshold"] == 0.93
        assert captured["min_genre_sentences"] == 42


class TestReferenceEmbeddingConstruction:
    """Tests for sentence-count weighted reference embedding construction."""

    def test_get_known_genre_embeddings_uses_sentence_count_weighting(self):
        """Single-genre references should weight each cluster centroid by cluster size."""
        bootstrapper = GenreBootstrapper(Config())
        bootstrapper.genre_combination_clusters = {
            ("news",): {
                ("tb_small", "train"): [
                    {
                        "cluster_id": 0,
                        "sent_ids": ["s1"],
                        "embedding": np.array([0.0, 2.0]),
                        "confidence": 1.0,
                    }
                ],
                ("tb_large", "train"): [
                    {
                        "cluster_id": 0,
                        "sent_ids": [f"l{i}" for i in range(9)],
                        "embedding": np.array([2.0, 0.0]),
                        "confidence": 1.0,
                    }
                ],
            }
        }

        known = bootstrapper._get_known_genre_embeddings(["news"])

        np.testing.assert_allclose(known["news"], np.array([1.8, 0.2]))

    def test_virtual_split_reference_embeddings_use_sentence_count_weighting(self):
        """Virtual split references should weight treebank centroids by split sentence counts."""
        ops = ClusteringOperations()
        virtual_splits_by_treebank = {
            "tb_small": {
                "news": {
                    "sent_ids": ["s1"],
                    "embeddings": np.array([[0.0, 2.0]]),
                    "split_distribution": {"train": 1},
                }
            },
            "tb_large": {
                "news": {
                    "sent_ids": ["l1", "l2", "l3"],
                    "embeddings": np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]),
                    "split_distribution": {"train": 3},
                }
            },
        }

        known = ops.build_reference_embeddings_from_virtual_splits(virtual_splits_by_treebank)

        np.testing.assert_allclose(known["news"], np.array([1.5, 0.5]))

    def test_get_known_genre_embeddings_can_use_uniform_weighting(self):
        """Uniform weighting should average cluster centroids equally across sources."""
        config = Config()
        config.bootstrapping.reference_weighting = "uniform"
        bootstrapper = GenreBootstrapper(config)
        bootstrapper.genre_combination_clusters = {
            ("news",): {
                ("tb_small", "train"): [
                    {
                        "cluster_id": 0,
                        "sent_ids": ["s1"],
                        "embedding": np.array([0.0, 2.0]),
                        "confidence": 1.0,
                    }
                ],
                ("tb_large", "train"): [
                    {
                        "cluster_id": 0,
                        "sent_ids": [f"l{i}" for i in range(9)],
                        "embedding": np.array([2.0, 0.0]),
                        "confidence": 1.0,
                    }
                ],
            }
        }

        known = bootstrapper._get_known_genre_embeddings(["news"])

        np.testing.assert_allclose(known["news"], np.array([1.0, 1.0]))

    def test_virtual_split_reference_embeddings_can_use_uniform_weighting(self):
        """Uniform weighting should average virtual split centroids equally."""
        ops = ClusteringOperations(reference_weighting="uniform")
        virtual_splits_by_treebank = {
            "tb_small": {
                "news": {
                    "sent_ids": ["s1"],
                    "embeddings": np.array([[0.0, 2.0]]),
                    "split_distribution": {"train": 1},
                }
            },
            "tb_large": {
                "news": {
                    "sent_ids": ["l1", "l2", "l3"],
                    "embeddings": np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]),
                    "split_distribution": {"train": 3},
                }
            },
        }

        known = ops.build_reference_embeddings_from_virtual_splits(virtual_splits_by_treebank)

        np.testing.assert_allclose(known["news"], np.array([1.0, 1.0]))


class TestSharedClusterLabeling:
    """Tests for shared cluster labeling logic used across pipelines."""

    def test_assign_cluster_label_returns_genre_confidence_method(self):
        """Shared assignment should return top genre, confidence, and method."""
        ops = ClusteringOperations(min_confidence=0.8, min_margin=0.05)
        centroid = np.array([1.0, 0.0])
        reference_embeddings = {
            "news": np.array([1.0, 0.0]),
            "wiki": np.array([0.0, 1.0]),
        }

        best_genre, confidence, method, sorted_sims = ops.assign_cluster_label(
            centroid, reference_embeddings
        )

        assert best_genre == "news"
        assert confidence == pytest.approx(1.0)
        assert method == "bootstrap-labeled"
        assert sorted_sims[0][0] == "news"
        assert sorted_sims[0][1] == pytest.approx(1.0)

    def test_assign_cluster_label_uses_margin_for_uncertainty(self):
        """High top-1 similarity with a tiny margin should be marked as inferred."""
        ops = ClusteringOperations(min_confidence=0.8, min_margin=0.02)
        centroid = np.array([1.0, 0.0])
        reference_embeddings = {
            "news": np.array([1.0, 0.0]),
            "wiki": np.array([0.99, 0.1]),
        }

        best_genre, confidence, method, _ = ops.assign_cluster_label(
            centroid, reference_embeddings
        )

        assert best_genre == "news"
        assert confidence >= 0.8
        assert method == "bootstrap-inferred"

    def test_label_clusters_delegates_to_assign_cluster_label(self, monkeypatch):
        """Batch cluster labeling should delegate per-cluster assignment to shared helper."""
        ops = ClusteringOperations(min_confidence=0.8)
        cluster_centroids = {
            0: np.array([1.0, 0.0]),
            1: np.array([0.0, 1.0]),
        }
        reference_embeddings = {
            "news": np.array([1.0, 0.0]),
            "wiki": np.array([0.0, 1.0]),
        }

        calls = []

        def _mock_assign_cluster_label(centroid, references):
            calls.append(tuple(centroid.tolist()))
            assert references is reference_embeddings
            return ("news", 0.5, "bootstrap-inferred", [("news", 0.5), ("wiki", 0.4)])

        monkeypatch.setattr(ops, "assign_cluster_label", _mock_assign_cluster_label)

        labels, high_conf_count, low_conf_count = ops.label_clusters(
            cluster_centroids, reference_embeddings
        )

        assert len(calls) == 2
        assert labels[0] == ("news", 0.5, "bootstrap-inferred")
        assert labels[1] == ("news", 0.5, "bootstrap-inferred")
        assert high_conf_count == 0
        assert low_conf_count == 2

    def test_label_cluster_descriptors_delegates_to_assign_cluster_label(self, monkeypatch):
        """Descriptor labeling should reuse assign_cluster_label and propagate sentence labels."""
        ops = ClusteringOperations(min_confidence=0.8)
        cluster_descriptors = [
            {
                "cluster_id": 0,
                "embedding": np.array([1.0, 0.0]),
                "sent_ids": ["s1", "s2"],
            },
            {
                "cluster_id": 1,
                "embedding": np.array([0.0, 1.0]),
                "sent_ids": ["s3"],
            },
        ]
        reference_embeddings = {
            "news": np.array([1.0, 0.0]),
            "wiki": np.array([0.0, 1.0]),
        }

        calls = []

        def _mock_assign_cluster_label(centroid, references):
            calls.append(tuple(centroid.tolist()))
            assert references is reference_embeddings
            return ("news", 0.5, "bootstrap-inferred", [("news", 0.5), ("wiki", 0.4)])

        monkeypatch.setattr(ops, "assign_cluster_label", _mock_assign_cluster_label)

        (
            cluster_labels,
            sentence_labels,
            cluster_similarities,
            high_conf_count,
            low_conf_count,
        ) = ops.label_cluster_descriptors(cluster_descriptors, reference_embeddings)

        assert len(calls) == 2
        assert cluster_labels[0] == ("news", 0.5, "bootstrap-inferred")
        assert cluster_labels[1] == ("news", 0.5, "bootstrap-inferred")
        assert sentence_labels["s1"] == ("news", 0.5, "bootstrap-inferred")
        assert sentence_labels["s2"] == ("news", 0.5, "bootstrap-inferred")
        assert sentence_labels["s3"] == ("news", 0.5, "bootstrap-inferred")
        assert cluster_similarities[0][0] == ("news", 0.5)
        assert high_conf_count == 0
        assert low_conf_count == 2

    def test_label_predictable_combinations_enforces_one_to_one_and_combo_restriction(
        self, monkeypatch
    ):
        """Matching should be one-to-one and only consider genres in the current combination."""
        ops = ClusteringOperations(min_confidence=0.0, min_margin=0.0)
        genre_combination_clusters = {
            ("news", "wiki"): {
                ("xx_demo", "test"): [
                    {
                        "cluster_id": 0,
                        "sent_ids": ["sid_c0"],
                        "embedding": np.array([0.0]),
                    },
                    {
                        "cluster_id": 1,
                        "sent_ids": ["sid_c1"],
                        "embedding": np.array([1.0]),
                    },
                ]
            }
        }
        final_labels = {}
        seen_reference_keys = []

        def _mock_assign_cluster_label(centroid, references):
            seen_reference_keys.append(set(references.keys()))
            if float(centroid[0]) == 0.0:
                return (
                    "news",
                    0.99,
                    "bootstrap-labeled",
                    [("news", 0.99), ("wiki", 0.98)],
                )
            return (
                "news",
                0.95,
                "bootstrap-labeled",
                [("news", 0.95), ("wiki", 0.20)],
            )

        monkeypatch.setattr(ops, "assign_cluster_label", _mock_assign_cluster_label)

        ops.label_predictable_combinations(
            predict_combinations=[("news", "wiki")],
            genre_combination_clusters=genre_combination_clusters,
            known_embeddings={
                "news": np.array([1.0, 0.0]),
                "wiki": np.array([0.0, 1.0]),
                "spoken": np.array([0.0, -1.0]),  # Must be ignored here.
            },
            final_labels=final_labels,
        )

        assert seen_reference_keys
        assert all(keys == {"news", "wiki"} for keys in seen_reference_keys)
        assert final_labels["sid_c0"][0] == "news"
        # One-to-one forces the second cluster to the remaining genre.
        assert final_labels["sid_c1"][0] == "wiki"

    def test_label_predictable_combinations_mutates_cluster_pool_iteratively(
        self, monkeypatch
    ):
        """Assigned clusters should be promoted and unresolved parts moved to reduced combinations."""
        ops = ClusteringOperations(min_confidence=0.0, min_margin=0.0)
        genre_combination_clusters = {
            ("news", "spoken", "wiki"): {
                ("xx_demo", "test"): [
                    {
                        "cluster_id": 0,
                        "sent_ids": ["sid_news"],
                        "embedding": np.array([0.0]),
                    },
                    {
                        "cluster_id": 1,
                        "sent_ids": ["sid_wiki"],
                        "embedding": np.array([1.0]),
                    },
                    {
                        "cluster_id": 2,
                        "sent_ids": ["sid_spoken"],
                        "embedding": np.array([2.0]),
                    },
                ]
            }
        }
        final_labels = {}

        def _mock_assign_cluster_label(centroid, references):
            if float(centroid[0]) == 0.0:
                return (
                    "news",
                    0.90,
                    "bootstrap-labeled",
                    [("news", 0.90), ("wiki", 0.10)],
                )
            if float(centroid[0]) == 1.0:
                return (
                    "wiki",
                    0.95,
                    "bootstrap-labeled",
                    [("wiki", 0.95), ("news", 0.20)],
                )
            return (
                "news",
                0.85,
                "bootstrap-labeled",
                [("news", 0.85), ("wiki", 0.80)],
            )

        monkeypatch.setattr(ops, "assign_cluster_label", _mock_assign_cluster_label)

        summary = ops.label_predictable_combinations(
            predict_combinations=[("news", "spoken", "wiki")],
            genre_combination_clusters=genre_combination_clusters,
            known_embeddings={
                "news": np.array([1.0, 0.0]),
                "wiki": np.array([0.0, 1.0]),
            },
            final_labels=final_labels,
        )

        assert summary["labels_assigned"] == 3
        assert final_labels["sid_news"][0] == "news"
        assert final_labels["sid_wiki"][0] == "wiki"
        assert final_labels["sid_spoken"][0] == "spoken"
        assert final_labels["sid_spoken"][2] == "bootstrap-inferred"
        assert ("news", "spoken", "wiki") not in genre_combination_clusters
        assert ("news",) in genre_combination_clusters
        assert ("wiki",) in genre_combination_clusters
        assert ("spoken",) in genre_combination_clusters

class TestVirtualSplitQualityGates:
    """Tests for virtual-split quality gate behavior."""

    def test_coverage_check_respects_min_genre_sentences(self):
        """Only genres meeting min_genre_sentences should count toward virtual-split creation."""
        ops = ClusteringOperations()
        all_sent_ids = [f"sid_{i}" for i in range(10)]
        sent_id_to_split = {sid: "train" for sid in all_sent_ids}
        sentence_metadata = {}

        # 8/10 sentences have metadata: 6 news + 2 wiki.
        for i, sid in enumerate(all_sent_ids[:8]):
            sentence_metadata[("xx_demo", "train", sid)] = "news" if i < 6 else "wiki"

        can_create, eligible_genres = ops.check_virtual_split_coverage(
            combined_embeddings=np.zeros((10, 2)),
            all_sent_ids=all_sent_ids,
            sent_id_to_split=sent_id_to_split,
            sentence_metadata=sentence_metadata,
            tb_code="xx_demo",
            coverage_threshold=0.8,
            min_genre_sentences=3,
        )

        assert not can_create
        assert eligible_genres == {"news"}

        can_create, eligible_genres = ops.check_virtual_split_coverage(
            combined_embeddings=np.zeros((10, 2)),
            all_sent_ids=all_sent_ids,
            sent_id_to_split=sent_id_to_split,
            sentence_metadata=sentence_metadata,
            tb_code="xx_demo",
            coverage_threshold=0.8,
            min_genre_sentences=2,
        )

        assert can_create
        assert eligible_genres == {"news", "wiki"}

    def test_cluster_treebanks_skips_ambiguous_metadata_for_virtual_splits(self, monkeypatch):
        """Sentences with multiple extracted genres should not be forced into a virtual split."""
        config = Config()
        config.evaluation.metadata_validation.coverage_threshold = 0.6
        config.evaluation.metadata_validation.min_genre_sentences = 1
        bootstrapper = GenreBootstrapper(config)

        embeddings_by_tb = {
            ("xx_demo", "train"): {
                "sent_id": ["sid_a", "sid_b", "sid_c"],
                "embedding": np.array([[1.0, 0.0], [0.0, 1.0], [0.2, 0.2]]),
            }
        }

        dataset = [
            {"sent_id": "sid_a", "mock_genres": ["news"]},
            {"sent_id": "sid_b", "mock_genres": ["wiki"]},
            {"sent_id": "sid_c", "mock_genres": ["news", "wiki"]},  # ambiguous
        ]

        monkeypatch.setattr(bootstrapper.data_loader, "load_treebank", lambda _tb, _sp: dataset)
        monkeypatch.setattr(bootstrapper.data_loader, "get_treebank_genres", lambda _tb: ["news", "wiki"])
        monkeypatch.setattr(
            bootstrapper.genre_mapper,
            "extract_genres_from_metadata",
            lambda sentence, _tb: sentence.get("mock_genres", []),
        )

        monkeypatch.setattr(
            bootstrapper.clusterer,
            "cluster_treebank",
            lambda embeddings, sent_ids, n_genres: {
                "clusters": {
                    0: {"sent_ids": ["sid_a", "sid_c"], "size": 2, "confidence": 1.0},
                    1: {"sent_ids": ["sid_b"], "size": 1, "confidence": 1.0},
                },
                "metrics": {},
            },
        )

        bootstrapper._cluster_treebanks(embeddings_by_tb)

        news_virtual = bootstrapper.treebank_clusters[("xx_demo", "train", "news")]
        wiki_virtual = bootstrapper.treebank_clusters[("xx_demo", "train", "wiki")]

        assert news_virtual["cluster_result"]["clusters"][0]["sent_ids"] == ["sid_a"]
        assert wiki_virtual["cluster_result"]["clusters"][0]["sent_ids"] == ["sid_b"]

        regular_clusters = bootstrapper.treebank_clusters[("xx_demo", "train")]["cluster_result"]["clusters"]
        assert any("sid_c" in cluster["sent_ids"] for cluster in regular_clusters.values())

    def test_cluster_treebanks_warns_on_split_metadata_extraction_errors(self, monkeypatch, caplog):
        """Split-level metadata extraction failures should be visible in logs."""
        config = Config()
        bootstrapper = GenreBootstrapper(config)

        embeddings_by_tb = {
            ("xx_demo", "train"): {
                "sent_id": ["sid_1", "sid_2"],
                "embedding": np.array([[1.0, 0.0], [0.0, 1.0]]),
            }
        }

        def _mock_load_treebank(_tb_code, _split):
            raise RuntimeError("metadata boom")

        monkeypatch.setattr(bootstrapper.data_loader, "load_treebank", _mock_load_treebank)
        monkeypatch.setattr(
            bootstrapper.data_loader,
            "get_treebank_genres",
            lambda _tb_code: ["news", "wiki"],
        )
        monkeypatch.setattr(
            bootstrapper.clusterer,
            "cluster_treebank",
            lambda embeddings, sent_ids, n_genres: {
                "clusters": {
                    0: {"sent_ids": sent_ids, "size": len(sent_ids), "confidence": 1.0}
                },
                "metrics": {},
            },
        )

        import logging
        caplog.set_level(logging.WARNING)
        bootstrapper._cluster_treebanks(embeddings_by_tb)

        assert "Metadata extraction failed for 1 split(s) in xx_demo" in caplog.text
        assert "metadata boom" in caplog.text

    def test_cluster_treebanks_warns_on_sentence_metadata_extraction_errors(self, monkeypatch, caplog):
        """Sentence-level metadata extraction failures should be visible in logs."""
        config = Config()
        bootstrapper = GenreBootstrapper(config)

        embeddings_by_tb = {
            ("xx_demo", "train"): {
                "sent_id": ["sid_ok", "sid_fail"],
                "embedding": np.array([[1.0, 0.0], [0.0, 1.0]]),
            }
        }

        dataset = [
            {"sent_id": "sid_ok"},
            {"sent_id": "sid_fail"},
        ]

        monkeypatch.setattr(bootstrapper.data_loader, "load_treebank", lambda _tb, _sp: dataset)
        monkeypatch.setattr(
            bootstrapper.data_loader,
            "get_treebank_genres",
            lambda _tb_code: ["news", "wiki"],
        )

        def _mock_extract_genres(sentence, _tb):
            if sentence["sent_id"] == "sid_fail":
                raise ValueError("bad metadata")
            return ["news"]

        monkeypatch.setattr(bootstrapper.genre_mapper, "extract_genres_from_metadata", _mock_extract_genres)
        monkeypatch.setattr(
            bootstrapper.clusterer,
            "cluster_treebank",
            lambda embeddings, sent_ids, n_genres: {
                "clusters": {
                    0: {"sent_ids": sent_ids, "size": len(sent_ids), "confidence": 1.0}
                },
                "metrics": {},
            },
        )

        import logging
        caplog.set_level(logging.WARNING)
        bootstrapper._cluster_treebanks(embeddings_by_tb)

        assert "Metadata extraction failed for 1 sentence(s) in xx_demo; skipped them" in caplog.text
