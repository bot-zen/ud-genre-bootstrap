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

    def test_lazy_loading(self):
        """Test that model is not loaded until actually needed."""
        # Create generator - model should not be loaded yet
        generator = EmbeddingGenerator(
            model_name="prajjwal1/bert-tiny",
            pooling="mean",
            batch_size=2,
            device="cpu"
        )

        # Model should not be loaded yet
        assert not generator._model_loaded, "Model should not be loaded on init"
        assert generator.model is None, "Model should be None until loaded"
        assert generator.tokenizer is None, "Tokenizer should be None until loaded"

        # Now use the model - should trigger lazy loading
        embeddings = generator.embed_sentences(["Test sentence"])

        # Model should now be loaded
        assert generator._model_loaded, "Model should be loaded after use"
        assert generator.model is not None, "Model should exist after use"
        assert generator.tokenizer is not None, "Tokenizer should exist after use"
        assert embeddings.shape[0] == 1, "Should have embeddings for 1 sentence"


class TestEmbeddingInputValidation:
    """Test validation of sentence text before embedding."""

    def test_embed_dataset_rejects_missing_text(self):
        dataset = Dataset.from_dict(
            {
                "sent_id": ["s1", "s2"],
                "text": ["valid sentence", None],
            }
        )
        generator = EmbeddingGenerator(model_name="dummy", device="cpu")

        with pytest.raises(ValueError, match="missing or invalid `text` values"):
            generator.embed_dataset(dataset)

    def test_embed_dataset_rejects_non_string_text(self):
        dataset = Dataset.from_dict(
            {
                "sent_id": ["s1"],
                "text": [["token1", "token2"]],
            }
        )
        generator = EmbeddingGenerator(model_name="dummy", device="cpu")

        with pytest.raises(ValueError, match="missing or invalid `text` values"):
            generator.embed_dataset(dataset)

    def test_embed_dataset_passes_plain_strings_to_embed_sentences(self, monkeypatch):
        dataset = Dataset.from_dict(
            {
                "sent_id": ["s1", "s2"],
                "text": ["Sentence one.", "Sentence two."],
            }
        )
        generator = EmbeddingGenerator(model_name="dummy", device="cpu")
        captured = {}

        def _fake_embed_sentences(sentences):
            captured["sentences"] = list(sentences)
            return np.zeros((len(sentences), 2), dtype=np.float32)

        monkeypatch.setattr(generator, "embed_sentences", _fake_embed_sentences)

        result = generator.embed_dataset(dataset)

        assert captured["sentences"] == ["Sentence one.", "Sentence two."]
        assert result["sent_id"] == ["s1", "s2"]
        assert result["embedding"].shape == (2, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
