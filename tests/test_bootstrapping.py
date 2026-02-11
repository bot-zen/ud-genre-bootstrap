"""Tests for bootstrapping functionality."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from ud_genre_bootstrap.bootstrapping.bootstrapper import GenreBootstrapper
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
        """Clusters should always be labeled, with method decided by confidence threshold."""
        config = Config()
        config.bootstrapping.min_confidence = 0.8
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
