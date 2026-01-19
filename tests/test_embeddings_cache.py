"""Tests for embedding caching functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset

from ud_genre_bootstrap.embeddings.generator import EmbeddingGenerator


class TestEmbeddingCache:
    """Test embedding caching and loading."""

    def test_save_and_load_embeddings(self):
        """Test saving and loading embeddings from cache."""
        # Create temporary cache directory
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create test dataset
            test_sentences = ["Hello world", "Test sentence", "Another example"]
            test_ids = ["sent_1", "sent_2", "sent_3"]
            dataset = Dataset.from_dict({
                "text": test_sentences,
                "sent_id": test_ids
            })

            # Generate and save embeddings
            # Using a small model for testing
            generator = EmbeddingGenerator(
                model_name="prajjwal1/bert-tiny",
                pooling="mean",
                batch_size=2,
                device="cpu"
            )

            result = generator.embed_treebank(
                treebank_code="test_tb",
                split="train",
                dataset=dataset,
                output_path=cache_dir
            )

            # Verify files were created
            embedding_file = cache_dir / "test_tb-train.npy"
            ids_file = cache_dir / "test_tb-train_ids.txt"

            assert embedding_file.exists(), "Embedding file should be created"
            assert ids_file.exists(), "IDs file should be created"

            # Load from cache
            loaded = EmbeddingGenerator.load_embeddings(
                treebank_code="test_tb",
                split="train",
                cache_dir=cache_dir
            )

            assert loaded is not None, "Should load cached embeddings"
            assert "sent_id" in loaded
            assert "embedding" in loaded
            assert loaded["sent_id"] == test_ids
            np.testing.assert_array_equal(loaded["embedding"], result["embedding"])

    def test_load_nonexistent_cache(self):
        """Test loading from cache when files don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Try loading nonexistent cache
            loaded = EmbeddingGenerator.load_embeddings(
                treebank_code="nonexistent",
                split="train",
                cache_dir=cache_dir
            )

            assert loaded is None, "Should return None when cache doesn't exist"

    def test_cache_directory_creation(self):
        """Test that cache directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "nested" / "cache"

            # Create test dataset
            test_sentences = ["Test"]
            test_ids = ["sent_1"]
            dataset = Dataset.from_dict({
                "text": test_sentences,
                "sent_id": test_ids
            })

            # Generate and save embeddings
            generator = EmbeddingGenerator(
                model_name="prajjwal1/bert-tiny",
                pooling="mean",
                batch_size=2,
                device="cpu"
            )

            generator.embed_treebank(
                treebank_code="test_tb",
                split="train",
                dataset=dataset,
                output_path=cache_dir
            )

            # Verify directory was created
            assert cache_dir.exists(), "Cache directory should be created"
            assert (cache_dir / "test_tb-train.npy").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
