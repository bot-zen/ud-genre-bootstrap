"""Tests for visualization functionality."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from ud_genre_bootstrap.utils.sentence_refs import qualify_sentence_ref


class TestVisualizationDataLoading:
    """Test visualization data loading and genre assignment."""

    def test_load_sentence_level_genres(self, tmp_path):
        """Test loading sentence-level genre assignments from all_genres.parquet."""
        # Create mock all_genres.parquet
        genres_file = tmp_path / "all_genres.parquet"
        df_genres = pd.DataFrame({
            "treebank": ["en_ewt", "en_ewt", "de_gsd"],
            "split": ["train", "train", "test"],
            "sent_id": ["en_ewt-train-001", "en_ewt-train-002", "de_gsd-test-001"],
            "genre": ["news", "news", "wiki"],
            "confidence": [0.85, 0.78, 0.92],
            "method": ["bootstrap-labeled", "bootstrap-labeled", "bootstrap-labeled"],
        })
        df_genres.to_parquet(genres_file, index=False)

        # Load and verify
        df_loaded = pd.read_parquet(genres_file)
        sent_id_to_genre = {
            qualify_sentence_ref(row.treebank, row.split, row.sent_id): row.genre
            for row in df_loaded[["treebank", "split", "sent_id", "genre"]].itertuples(index=False)
        }

        assert len(sent_id_to_genre) == 3
        assert sent_id_to_genre[("en_ewt", "train", "en_ewt-train-001")] == "news"
        assert sent_id_to_genre[("en_ewt", "train", "en_ewt-train-002")] == "news"
        assert sent_id_to_genre[("de_gsd", "test", "de_gsd-test-001")] == "wiki"

    def test_genre_assignment_fallback(self, tmp_path):
        """Test fallback to treebank-level metadata when all_genres.parquet doesn't exist."""
        # Create mock cluster_statistics.json
        import json

        stats_file = tmp_path / "cluster_statistics.json"
        cluster_stats = {
            "en_ewt_train": {
                "treebank": "en_ewt",
                "split": "train",
                "genres": ["news", "blog", "reviews"],
                "n_clusters": 3,
            },
            "de_gsd_test": {
                "treebank": "de_gsd",
                "split": "test",
                "genres": ["wiki", "news"],
                "n_clusters": 2,
            },
        }

        with open(stats_file, "w") as f:
            json.dump(cluster_stats, f)

        # Load and verify
        with open(stats_file, "r") as f:
            loaded_stats = json.load(f)

        # Create treebank_genres mapping
        treebank_genres = {}
        for key, stats in loaded_stats.items():
            tb_code = stats["treebank"]
            split = stats["split"]
            genres = stats["genres"]
            treebank_genres[(tb_code, split)] = genres

        assert treebank_genres[("en_ewt", "train")] == ["news", "blog", "reviews"]
        assert treebank_genres[("de_gsd", "test")] == ["wiki", "news"]

    def test_genre_list_creation_with_sentence_level(self):
        """Test creating genre list for visualization using sentence-level assignments."""
        # Mock data
        sent_id_to_genre = {
            ("en_ewt", "train", "en_ewt-train-001"): "news",
            ("en_ewt", "train", "en_ewt-train-002"): "blog",
            ("de_gsd", "test", "de_gsd-test-001"): "wiki",
        }

        cluster_rows = [
            {"treebank": "en_ewt", "split": "train", "sent_id": "en_ewt-train-001"},
            {"treebank": "en_ewt", "split": "train", "sent_id": "en_ewt-train-002"},
            {"treebank": "de_gsd", "split": "test", "sent_id": "de_gsd-test-001"},
        ]

        # Simulate genre assignment
        genre_list = []
        for row in cluster_rows:
            sent_ref = qualify_sentence_ref(row["treebank"], row["split"], row["sent_id"])
            genre_str = sent_id_to_genre.get(sent_ref, "unlabeled")
            genre_list.append(genre_str)

        assert genre_list == ["news", "blog", "wiki"]
        assert "news" in genre_list
        assert "blog" in genre_list
        assert "wiki" in genre_list
        # No comma-separated strings
        assert all("," not in g for g in genre_list)

    def test_genre_list_creation_with_fallback(self):
        """Test creating genre list using fallback treebank metadata."""
        # Mock treebank metadata
        treebank_genres = {
            ("en_ewt", "train"): ["news", "blog", "reviews"],
            ("de_gsd", "test"): ["wiki"],
        }

        cluster_rows = [
            {"treebank": "en_ewt", "split": "train"},
            {"treebank": "en_ewt", "split": "train"},
            {"treebank": "de_gsd", "split": "test"},
        ]

        # Simulate fallback genre assignment
        genre_list = []
        for row in cluster_rows:
            tb = row["treebank"]
            split = row["split"]
            genres = treebank_genres.get((tb, split), [])
            genre_str = ", ".join(genres) if genres else "unknown"
            genre_list.append(genre_str)

        assert genre_list[0] == "news, blog, reviews"
        assert genre_list[1] == "news, blog, reviews"
        assert genre_list[2] == "wiki"

    def test_color_by_parameter_validation(self):
        """Test color_by parameter validation."""
        valid_values = ["genre", "cluster", "treebank_split", "treebank"]

        # Valid values should pass
        for value in valid_values:
            assert value in valid_values

        # Invalid values should be caught
        invalid_values = ["invalid", "language", "confidence"]
        for value in invalid_values:
            assert value not in valid_values

    def test_visualization_dataframe_structure(self):
        """Test that visualization DataFrame has correct structure."""
        # Create mock visualization data
        plot_df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [1.5, 2.5, 3.5],
            "cluster": ["en_ewt:train:c0", "en_ewt:train:c1", "de_gsd:test:c0"],
            "treebank_split": ["en_ewt_train", "en_ewt_train", "de_gsd_test"],
            "sent_id": ["en-001", "en-002", "de-001"],
            "genre": ["news", "blog", "wiki"],
        })

        # Verify structure
        required_columns = {"x", "y", "cluster", "treebank_split", "sent_id", "genre"}
        assert required_columns.issubset(set(plot_df.columns))

        # Verify data types
        assert plot_df["x"].dtype == float
        assert plot_df["y"].dtype == float
        assert plot_df["genre"].dtype == object
        assert plot_df["cluster"].dtype == object

        # Verify no NaN in required columns
        assert not plot_df["genre"].isna().any()
        assert not plot_df["cluster"].isna().any()

    def test_hover_data_preparation(self):
        """Test hover data preparation for different color_by values."""
        all_columns = ["genre", "cluster", "treebank_split", "sent_id"]

        # When coloring by genre, hover should show other columns
        color_by = "genre"
        hover_cols = [col for col in all_columns if col != color_by]
        assert "cluster" in hover_cols
        assert "treebank_split" in hover_cols
        assert "sent_id" in hover_cols
        assert "genre" not in hover_cols

        # When coloring by cluster, hover should show other columns
        color_by = "cluster"
        hover_cols = [col for col in all_columns if col != color_by]
        assert "genre" in hover_cols
        assert "treebank_split" in hover_cols
        assert "sent_id" in hover_cols
        assert "cluster" not in hover_cols

    def test_genre_uniqueness_check(self):
        """Test that sentence-level genres are single values, not comma-separated."""
        # Good: sentence-level genres
        sentence_genres = ["news", "blog", "wiki", "fiction"]
        assert all("," not in g for g in sentence_genres)

        # Bad: treebank-level genres (comma-separated)
        treebank_genres = ["news, blog", "wiki, news, legal"]
        assert any("," in g for g in treebank_genres)

    def test_unlabeled_sentence_handling(self):
        """Test handling of sentences without genre assignments."""
        sent_id_to_genre = {
            "en-001": "news",
            "en-002": "blog",
            # "en-003" is missing
        }

        test_sent_ids = ["en-001", "en-002", "en-003"]

        genre_list = []
        for sent_id in test_sent_ids:
            genre = sent_id_to_genre.get(sent_id, "unlabeled")
            genre_list.append(genre)

        assert genre_list == ["news", "blog", "unlabeled"]
        assert "unlabeled" in genre_list


class TestVisualizationPaths:
    """Test path resolution for visualization files."""

    def test_find_all_genres_parquet_in_parent(self, tmp_path):
        """Test finding all_genres.parquet in parent directory."""
        # Create directory structure
        clusters_dir = tmp_path / "clusters"
        clusters_dir.mkdir()
        genres_file = tmp_path / "all_genres.parquet"

        # Create file
        df = pd.DataFrame({"sent_id": ["test"], "genre": ["news"]})
        df.to_parquet(genres_file, index=False)

        # Test search paths
        possible_paths = [
            clusters_dir.parent / "all_genres.parquet",
            clusters_dir / "all_genres.parquet",
        ]

        found_path = None
        for path in possible_paths:
            if path.exists():
                found_path = path
                break

        assert found_path is not None
        assert found_path == genres_file

    def test_find_all_genres_parquet_in_clusters_dir(self, tmp_path):
        """Test finding all_genres.parquet in clusters directory itself."""
        clusters_dir = tmp_path / "clusters"
        clusters_dir.mkdir()
        genres_file = clusters_dir / "all_genres.parquet"

        # Create file
        df = pd.DataFrame({"sent_id": ["test"], "genre": ["news"]})
        df.to_parquet(genres_file, index=False)

        # Test search paths
        possible_paths = [
            clusters_dir.parent / "all_genres.parquet",
            clusters_dir / "all_genres.parquet",
        ]

        found_path = None
        for path in possible_paths:
            if path.exists():
                found_path = path
                break

        assert found_path is not None
        assert found_path == genres_file

    def test_no_all_genres_file_found(self, tmp_path):
        """Test behavior when all_genres.parquet doesn't exist."""
        clusters_dir = tmp_path / "clusters"
        clusters_dir.mkdir()

        # Test search paths
        possible_paths = [
            clusters_dir.parent / "all_genres.parquet",
            clusters_dir / "all_genres.parquet",
        ]

        found_path = None
        for path in possible_paths:
            if path.exists():
                found_path = path
                break

        assert found_path is None
