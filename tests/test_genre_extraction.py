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

    def test_alternative_comment_format(self):
        """Test extraction from alternative comment format without newdoc."""
        mapper = GenreMapper()
        sentence = {"comments": ["# genre = news"]}
        genres = mapper.extract_genres_from_metadata(sentence, "test_tb")
        assert "news" in genres

    def test_extraction_deduplicates_preserving_order(self):
        """Duplicate genres should be removed without losing deterministic order."""
        mapper = GenreMapper()
        sentence = {
            "genre": "news",
            "comments": [
                "# genre = news",
                "# genre = blog",
                "# genre = news",
            ],
        }
        genres = mapper.extract_genres_from_metadata(sentence, "test_tb")
        assert genres == ["news", "blog"]

    def test_genre_with_hyphen(self):
        """Test extraction of genre names with hyphens."""
        mapper = GenreMapper()
        sentence = {"comments": ["# genre = grammar-examples"]}
        genres = mapper.extract_genres_from_metadata(sentence, "test_tb")
        assert "grammar-examples" in genres

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

            # Generic mapping (no treebank specified)
            assert mapper.normalize_genre("web") == "web"
            # Treebank-specific override (even for canonical genres)
            assert mapper.normalize_genre("web", "en_ewt") == "blog"

        finally:
            mappings_path.unlink()

    def test_treebank_specific_mapping_without_patterns(self):
        """Test treebank-specific mappings work without patterns (standard extraction)."""
        import tempfile
        import json

        # No patterns file - only mappings!
        mappings = {
            "weblog": "blog",  # Global mapping
            "de_gsd:web": "blog",  # de_gsd-specific: 'web' means 'blog'
            "fr_gsd:web": "web",   # fr_gsd-specific: 'web' stays 'web'
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mappings, f)
            mappings_path = Path(f.name)

        try:
            # No patterns needed!
            mapper = GenreMapper(
                genre_mapping_path=mappings_path,
                metadata_patterns_path=None
            )

            # Test 1: de_gsd overrides 'web' to 'blog'
            sentence = {"comments": ["# genre = web"]}
            genres = mapper.extract_genres_from_metadata(sentence, "de_gsd")
            assert "blog" in genres, f"Expected 'blog', got {genres}"

            # Test 2: fr_gsd keeps 'web' as 'web'
            genres = mapper.extract_genres_from_metadata(sentence, "fr_gsd")
            assert "web" in genres, f"Expected 'web', got {genres}"

            # Test 3: en_ewt has no override, keeps canonical 'web'
            genres = mapper.extract_genres_from_metadata(sentence, "en_ewt")
            assert "web" in genres, f"Expected 'web', got {genres}"

            # Test 4: Global mapping still works
            sentence = {"comments": ["# genre = weblog"]}
            genres = mapper.extract_genres_from_metadata(sentence, "en_ewt")
            assert "blog" in genres, f"Expected 'blog', got {genres}"

        finally:
            mappings_path.unlink()

    def test_patternless_genre_mapping_in_patterns_file(self):
        """Test genre_mapping without pattern in metadata_patterns file."""
        import tempfile
        import json

        # Define treebank-specific mappings in patterns file WITHOUT patterns
        patterns = {
            "de_lit": [
                {
                    "genre_mapping": {
                        "fragments": "nonfiction",
                        "poetry": "fiction"
                    }
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(patterns, f)
            patterns_path = Path(f.name)

        try:
            mapper = GenreMapper(
                genre_mapping_path=None,
                metadata_patterns_path=patterns_path
            )

            # Test mapping defined in patterns file
            sentence = {"comments": ["# genre = fragments"]}
            genres = mapper.extract_genres_from_metadata(sentence, "de_lit")
            assert "nonfiction" in genres, f"Expected 'nonfiction', got {genres}"

            sentence = {"comments": ["# genre = poetry"]}
            genres = mapper.extract_genres_from_metadata(sentence, "de_lit")
            assert "fiction" in genres, f"Expected 'fiction', got {genres}"

            # Test unmapped genre passes through
            sentence = {"comments": ["# genre = news"]}
            genres = mapper.extract_genres_from_metadata(sentence, "de_lit")
            assert "news" in genres, f"Expected 'news', got {genres}"

            # Test other treebank not affected
            sentence = {"comments": ["# genre = fragments"]}
            genres = mapper.extract_genres_from_metadata(sentence, "fr_gsd")
            assert "fragments" in genres, f"Expected 'fragments', got {genres}"

        finally:
            patterns_path.unlink()

    def test_default_genre_for_entire_treebank(self):
        """Test setting default genre for treebank with no genre metadata."""
        import tempfile
        import json

        # Define default genres for treebanks
        patterns = {
            "xx_news": [{"genre": "news"}],
            "yy_fiction": [{"genre": "fiction"}]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(patterns, f)
            patterns_path = Path(f.name)

        try:
            mapper = GenreMapper(
                genre_mapping_path=None,
                metadata_patterns_path=patterns_path
            )

            # Test 1: Sentence with NO genre metadata - should use default
            sentence = {"comments": ["# sent_id = test-001"]}
            genres = mapper.extract_genres_from_metadata(sentence, "xx_news")
            assert "news" in genres, f"Expected 'news', got {genres}"

            # Test 2: Different treebank with different default
            genres = mapper.extract_genres_from_metadata(sentence, "yy_fiction")
            assert "fiction" in genres, f"Expected 'fiction', got {genres}"

            # Test 3: Sentence WITH genre metadata - should use metadata, not default
            sentence_with_genre = {"comments": ["# genre = blog"]}
            genres = mapper.extract_genres_from_metadata(sentence_with_genre, "xx_news")
            assert "blog" in genres and "news" not in genres, \
                f"Expected only 'blog', got {genres}"

            # Test 4: Treebank without default - should return empty
            genres = mapper.extract_genres_from_metadata(sentence, "zz_test")
            assert len(genres) == 0, f"Expected empty, got {genres}"

        finally:
            patterns_path.unlink()

    def test_multiple_capture_groups_combined(self):
        """Test combining multiple capture groups for split matches."""
        import tempfile
        import json

        # Czech CAC pattern: extract first and last letter from doc ID
        patterns = {
            "cs_cac": [
                {
                    "pattern": r"# newdoc id = ([a-z])\d+([a-z])",
                    "genre": "$1$2"
                }
            ]
        }

        mappings = {
            "aw": "news",
            "as": "news",
            "bw": "nonfiction"
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

            # Test extracting "aw" from "a01w"
            sentence = {"comments": ["# newdoc id = a01w"]}
            genres = mapper.extract_genres_from_metadata(sentence, "cs_cac")
            assert "news" in genres, f"Expected 'news', got {genres}"

            # Test extracting "bw" from "b12w"
            sentence = {"comments": ["# newdoc id = b12w"]}
            genres = mapper.extract_genres_from_metadata(sentence, "cs_cac")
            assert "nonfiction" in genres, f"Expected 'nonfiction', got {genres}"

        finally:
            patterns_path.unlink()
            mappings_path.unlink()

    def test_partial_inline_genre_mapping_with_global_fallback(self):
        """Test that global mappings work when inline mapping is partial."""
        import tempfile
        import json

        # Pattern with partial inline mapping
        patterns = {
            "test_tb": [
                {
                    "pattern": r"# source = (.+)",
                    "genre_mapping": {
                        "news": "news",
                        "magazine": "news"
                        # "blog" and "weblog" are NOT in inline mapping
                    }
                }
            ]
        }

        # Global mapping should handle values not in inline mapping
        mappings = {
            "blog": "blog",
            "weblog": "blog"
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

            # Test inline mapping
            sentence = {"comments": ["# source = news"]}
            genres = mapper.extract_genres_from_metadata(sentence, "test_tb")
            assert "news" in genres, f"Expected 'news', got {genres}"

            # Test global mapping (not in inline mapping)
            sentence = {"comments": ["# source = blog"]}
            genres = mapper.extract_genres_from_metadata(sentence, "test_tb")
            assert "blog" in genres, f"Expected 'blog', got {genres}"

            # Test global normalization (not in inline mapping)
            sentence = {"comments": ["# source = weblog"]}
            genres = mapper.extract_genres_from_metadata(sentence, "test_tb")
            assert "blog" in genres, f"Expected 'blog', got {genres}"

        finally:
            patterns_path.unlink()
            mappings_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
