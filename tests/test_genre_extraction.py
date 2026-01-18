"""Tests for genre extraction and mapping."""

import pytest
from pathlib import Path
from ud_genre_bootstrap.utils.genre_mapping import GenreMapper


class TestGenreMapper:
    """Test GenreMapper class."""

    def test_direct_genre_field(self):
        """Test extraction from direct genre field."""
        mapper = GenreMapper()
        sentence = {"genre": "news"}
        genres = mapper.extract_genres_from_metadata(sentence, "test_tb")
        assert "news" in genres

    def test_standard_comment_format(self):
        """Test extraction from standard UD comment."""
        mapper = GenreMapper()
        sentence = {"comments": ["# newdoc genre = blog"]}
        genres = mapper.extract_genres_from_metadata(sentence, "test_tb")
        assert "blog" in genres

    def test_pattern_with_capture_group(self):
        """Test pattern matching with capture group substitution."""
        # Create temp pattern file
        import tempfile
        import json

        patterns = {
            "test_tb": [
                {
                    "pattern": r"# sent_id = (n|w)",
                    "genre": "$1"
                }
            ]
        }

        mappings = {
            "n": "news",
            "w": "wiki"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(patterns, f)
            patterns_path = Path(f.name)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mappings, f)
            mappings_path = Path(f.name)

        try:
            mapper = GenreMapper(
                genre_mapping_path=mappings_path,
                metadata_patterns_path=patterns_path
            )

            # Test news pattern
            sentence = {"comments": ["# sent_id = n01001011"]}
            genres = mapper.extract_genres_from_metadata(sentence, "test_tb")
            assert "news" in genres, f"Expected 'news', got {genres}"

            # Test wiki pattern
            sentence = {"comments": ["# sent_id = w01001049"]}
            genres = mapper.extract_genres_from_metadata(sentence, "test_tb")
            assert "wiki" in genres, f"Expected 'wiki', got {genres}"

        finally:
            patterns_path.unlink()
            mappings_path.unlink()

    def test_pud_treebank_pattern(self):
        """Test realistic PUD treebank genre extraction."""
        import tempfile
        import json

        patterns = {
            "de_pud": [
                {
                    "pattern": r"# sent_id = ([nw])",
                    "genre": "$1"
                }
            ]
        }

        mappings = {
            "n": "news",
            "w": "wiki"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(patterns, f)
            patterns_path = Path(f.name)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mappings, f)
            mappings_path = Path(f.name)

        try:
            mapper = GenreMapper(
                genre_mapping_path=mappings_path,
                metadata_patterns_path=patterns_path
            )

            # Realistic PUD sentence
            sentence = {
                "sent_id": "n01001011",
                "comments": [
                    "# newdoc id = n01001",
                    "# sent_id = n01001011",
                    "# parallel_id = pud/n01001011"
                ]
            }
            genres = mapper.extract_genres_from_metadata(sentence, "de_pud")
            assert "news" in genres, f"Expected 'news' in {genres}"

        finally:
            patterns_path.unlink()
            mappings_path.unlink()

    def test_genre_normalization(self):
        """Test genre normalization with mappings."""
        import tempfile
        import json

        mappings = {
            "weblog": "blog",
            "newspaper": "news"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mappings, f)
            mappings_path = Path(f.name)

        try:
            mapper = GenreMapper(genre_mapping_path=mappings_path)

            assert mapper.normalize_genre("weblog") == "blog"
            assert mapper.normalize_genre("newspaper") == "news"
            assert mapper.normalize_genre("news") == "news"  # Already canonical

        finally:
            mappings_path.unlink()

    def test_treebank_specific_mapping(self):
        """Test treebank-specific genre mappings."""
        import tempfile
        import json

        mappings = {
            "web": "web",
            "en_ewt:web": "blog"  # en_ewt uses 'web' to mean 'blog'
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mappings, f)
            mappings_path = Path(f.name)

        try:
            mapper = GenreMapper(genre_mapping_path=mappings_path)

            # Generic mapping
            assert mapper.normalize_genre("web") == "web"
            # Treebank-specific override
            assert mapper.normalize_genre("web", "en_ewt") == "blog"

        finally:
            mappings_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
