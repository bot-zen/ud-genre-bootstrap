"""Genre mapping and extraction utilities."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import json


class GenreMapper:
    """Handle genre extraction and mapping to canonical UD genres."""

    # Canonical UD genre labels
    CANONICAL_GENRES = {
        "academic",
        "blog",
        "email",
        "fiction",
        "government",
        "grammar-examples",
        "legal",
        "medical",
        "news",
        "nonfiction",
        "reviews",
        "social",
        "spoken",
        "web",
        "wiki",
    }

    def __init__(
        self,
        genre_mapping_path: Optional[Path] = None,
        metadata_patterns_path: Optional[Path] = None,
    ):
        """Initialize genre mapper.

        Args:
            genre_mapping_path: Path to JSON with non-standard -> UD genre mappings
            metadata_patterns_path: Path to JSON with sentence-level genre patterns
        """
        self.genre_mappings = self._load_genre_mappings(genre_mapping_path)
        self.metadata_patterns = self._load_metadata_patterns(metadata_patterns_path)

    def _load_genre_mappings(self, path: Optional[Path]) -> Dict[str, str]:
        """Load genre mappings from JSON file.

        Format: {"treebank_genre": "ud_canonical_genre", ...}
        """
        if path is None or not path.exists():
            return {}

        with open(path) as f:
            return json.load(f)

    def _load_metadata_patterns(self, path: Optional[Path]) -> Dict:
        """Load metadata extraction patterns from JSON file.

        Format similar to ud28/meta.json with patterns for sentence headers
        """
        if path is None or not path.exists():
            return {}

        with open(path) as f:
            return json.load(f)

    def normalize_genre(self, genre: str, treebank_code: Optional[str] = None) -> str:
        """Normalize a genre label to canonical UD genre.

        Args:
            genre: Raw genre label
            treebank_code: Optional treebank code for treebank-specific mappings

        Returns:
            Canonical UD genre label
        """
        # Check if already canonical
        if genre in self.CANONICAL_GENRES:
            return genre

        # Try direct mapping
        if genre in self.genre_mappings:
            return self.genre_mappings[genre]

        # Try treebank-specific mapping
        if treebank_code:
            tb_key = f"{treebank_code}:{genre}"
            if tb_key in self.genre_mappings:
                return self.genre_mappings[tb_key]

        # TODO: Implement fuzzy matching or raise warning
        return genre  # Return as-is if no mapping found

    def extract_genres_from_metadata(
        self, sentence: Dict, treebank_code: str
    ) -> List[str]:
        """Extract genres from sentence-level metadata fields.

        Args:
            sentence: Sentence dictionary with metadata
            treebank_code: Treebank code for pattern lookup

        Returns:
            List of extracted genre labels
        """
        genres = []

        # Method 1: Direct genre field in sentence metadata
        if "genre" in sentence:
            genres.append(sentence["genre"])

        # Method 2: Check CoNLL-U comments (# newdoc genre = ...)
        if "comments" in sentence:
            for comment in sentence["comments"]:
                match = re.search(r"#\s*newdoc\s+genre\s*=\s*(\w+)", comment)
                if match:
                    genres.append(match.group(1))

        # Method 3: Use treebank-specific patterns
        if treebank_code in self.metadata_patterns:
            patterns = self.metadata_patterns[treebank_code]
            # Apply patterns to extract genres from sentence comments/metadata
            if "comments" in sentence and patterns:
                for comment in sentence["comments"]:
                    for pattern_dict in patterns:
                        if isinstance(pattern_dict, dict):
                            pattern = pattern_dict.get("pattern", "")
                            genre_template = pattern_dict.get("genre", "")
                            genre_mapping = pattern_dict.get("genre_mapping", None)

                            if pattern:
                                match = re.search(pattern, comment)
                                if match:
                                    if genre_mapping:
                                        # Use genre_mapping dict to map captured value
                                        captured = match.group(1) if match.groups() else None
                                        if captured and captured in genre_mapping:
                                            genres.append(genre_mapping[captured])
                                    elif genre_template:
                                        # Substitute capture groups in genre template
                                        genre_value = genre_template
                                        # Replace $1, $2, etc. with captured groups
                                        for i, group in enumerate(match.groups(), 1):
                                            if group:
                                                genre_value = genre_value.replace(f"${i}", group)
                                        genres.append(genre_value)
                        elif isinstance(pattern_dict, str):
                            # Simple string matching
                            if pattern_dict in comment:
                                # Extract genre from comment
                                match = re.search(r"genre[:\s=]+(\w+)", comment, re.IGNORECASE)
                                if match:
                                    genres.append(match.group(1))

        # Normalize all genres
        normalized = [
            self.normalize_genre(g, treebank_code) for g in genres if g
        ]
        return list(set(normalized))  # Remove duplicates

    def get_treebank_genres(
        self, treebank_metadata: Dict, treebank_code: str
    ) -> Set[str]:
        """Get genres for a treebank from its metadata.

        Args:
            treebank_metadata: Treebank metadata dictionary
            treebank_code: Treebank code

        Returns:
            Set of canonical genre labels
        """
        genres = set()

        # Extract from metadata 'Genre' field
        if "Genre" in treebank_metadata:
            genre_str = treebank_metadata["Genre"]
            # Genres may be space-separated
            raw_genres = genre_str.split()
            genres.update(
                self.normalize_genre(g, treebank_code) for g in raw_genres
            )

        return genres

    def validate_genre(self, genre: str) -> bool:
        """Check if a genre is a canonical UD genre.

        Args:
            genre: Genre label to validate

        Returns:
            True if canonical, False otherwise
        """
        return genre in self.CANONICAL_GENRES
