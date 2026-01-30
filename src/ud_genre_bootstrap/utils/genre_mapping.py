"""Genre mapping and extraction utilities."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import json


class GenreMapper:
    """Handle genre extraction and mapping to canonical UD genres."""

    # Default canonical UD genre labels (can be overridden via config)
    DEFAULT_UD_GENRES = {
        "academic",
        "bible",
        "blog",
        "email",
        "fiction",
        "government",
        "grammar-examples",
        "learner-essays",
        "legal",
        "medical",
        "news",
        "nonfiction",
        "poetry",
        "reviews",
        "social",
        "spoken",
        "web",
        "wiki",
    }

    def __init__(
        self,
        genre_mapping_path: Optional[Path] = None,
        metadata_patterns_path: Optional[Union[Path, List[Path]]] = None,
        canonical_genres: Optional[List[str]] = None,
        data_loader=None,
    ):
        """Initialize genre mapper.

        Args:
            genre_mapping_path: Path to JSON with non-standard -> UD genre mappings
            metadata_patterns_path: Path or list of paths to JSON files with sentence-level genre patterns
            canonical_genres: Optional list of canonical genre labels (overrides default set)
            data_loader: Optional UDDataLoader for accessing treebank metadata
        """
        self.genre_mappings = self._load_genre_mappings(genre_mapping_path)
        self.metadata_patterns = self._load_metadata_patterns(metadata_patterns_path)
        self.data_loader = data_loader

        # Use provided canonical genres or fall back to default
        if canonical_genres is not None:
            self.canonical_genres = set(canonical_genres)
        else:
            self.canonical_genres = self.DEFAULT_UD_GENRES

    def _load_genre_mappings(self, path: Optional[Path]) -> Dict[str, str]:
        """Load genre mappings from JSON file.

        Format: {"treebank_genre": "ud_canonical_genre", ...}
        """
        if path is None or not path.exists():
            return {}

        with open(path) as f:
            return json.load(f)

    def _load_metadata_patterns(self, paths: Optional[Union[Path, List[Path]]]) -> Dict:
        """Load metadata extraction patterns from JSON file(s).

        Args:
            paths: Single path or list of paths to pattern files

        Returns:
            Merged dictionary of patterns. Later files override earlier ones for
            the same treebank code.

        Format similar to ud28/meta.json with patterns for sentence headers
        """
        if paths is None:
            return {}

        # Normalize to list
        if isinstance(paths, (str, Path)):
            paths = [Path(paths)]
        else:
            paths = [Path(p) for p in paths]

        # Load and merge patterns from all files
        merged_patterns = {}
        for path in paths:
            if not path.exists():
                continue

            with open(path) as f:
                patterns = json.load(f)

            # Merge patterns
            for treebank_code, treebank_patterns in patterns.items():
                # Skip comments
                if treebank_code.startswith("_"):
                    continue

                if treebank_code in merged_patterns:
                    # Extend existing patterns
                    merged_patterns[treebank_code].extend(treebank_patterns)
                else:
                    # Add new treebank patterns
                    merged_patterns[treebank_code] = treebank_patterns

        return merged_patterns

    def normalize_genre(self, genre: str, treebank_code: Optional[str] = None) -> str:
        """Normalize a genre label to canonical UD genre.

        Args:
            genre: Raw genre label
            treebank_code: Optional treebank code for treebank-specific mappings

        Returns:
            Canonical UD genre label
        """
        # Priority 1: Treebank-specific mapping (allows overriding canonical genres)
        if treebank_code:
            tb_key = f"{treebank_code}:{genre}"
            if tb_key in self.genre_mappings:
                return self.genre_mappings[tb_key]

        # Priority 2: Global mapping (even if genre is canonical, mapping takes precedence)
        if genre in self.genre_mappings:
            return self.genre_mappings[genre]

        # Priority 3: Already canonical (no mapping needed)
        # This is now only reached if there's no explicit mapping
        if genre in self.canonical_genres:
            return genre

        # No mapping found, return as-is
        return genre

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

        default_genre = None

        # Check if treebank has a single genre in metadata
        if self.data_loader:
            treebank_genres = self.data_loader.get_treebank_genres(treebank_code)
            if len(treebank_genres) == 1:
                # Single genre means all sentences have this genre
                default_genre = treebank_genres[0]

        # Collect treebank-specific settings (without patterns)
        treebank_genre_mappings = {}
        if treebank_code in self.metadata_patterns:
            for pattern_dict in self.metadata_patterns[treebank_code]:
                if isinstance(pattern_dict, dict) and "pattern" not in pattern_dict:
                    # Pattern-less genre_mapping for this treebank
                    if "genre_mapping" in pattern_dict:
                        treebank_genre_mappings.update(pattern_dict["genre_mapping"])

                    # Default genre when no genre metadata exists
                    # Pattern-based default takes precedence over metadata
                    if "genre" in pattern_dict:
                        default_genre = pattern_dict["genre"]

        # Method 1: Direct genre field in sentence metadata
        if "genre" in sentence:
            genres.append(sentence["genre"])

        # Method 2: Check CoNLL-U comments for standard genre metadata
        if "comments" in sentence:
            for comment in sentence["comments"]:
                # Standard UD format: # newdoc genre = ...
                match = re.search(r"#\s*newdoc\s+genre\s*=\s*(\S+)", comment)
                if match:
                    genres.append(match.group(1))
                    continue

                # Alternative format: # genre = ... (without newdoc)
                match = re.search(r"#\s+genre\s*=\s*(\S+)", comment)
                if match:
                    genres.append(match.group(1))

        # Apply treebank-specific genre_mappings to genres extracted so far
        if treebank_genre_mappings:
            # Map genres using treebank-specific mappings
            mapped_genres = []
            for genre in genres:
                if genre in treebank_genre_mappings:
                    mapped_genres.append(treebank_genre_mappings[genre])
                else:
                    mapped_genres.append(genre)
            genres = mapped_genres

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
                                    # First, construct genre value from template (if any)
                                    if genre_template:
                                        # Substitute capture groups in genre template
                                        genre_value = genre_template
                                        # Replace $1, $2, etc. with captured groups
                                        for i, group in enumerate(match.groups(), 1):
                                            if group:
                                                genre_value = genre_value.replace(f"${i}", group)
                                    else:
                                        # No template, use first capture group
                                        genre_value = match.group(1) if match.groups() else None

                                    # Then, apply genre_mapping if it exists
                                    if genre_value:
                                        if genre_mapping and genre_value in genre_mapping:
                                            # Use inline mapping
                                            genres.append(genre_mapping[genre_value])
                                        else:
                                            # Not in inline mapping, add raw value
                                            # It will be normalized by global genre_mappings later
                                            genres.append(genre_value)
                        elif isinstance(pattern_dict, str):
                            # Simple string matching
                            if pattern_dict in comment:
                                # Extract genre from comment
                                match = re.search(r"genre[:\s=]+(\w+)", comment, re.IGNORECASE)
                                if match:
                                    genres.append(match.group(1))

        # Method 4: Use default genre if no genre was extracted
        if not genres and default_genre:
            genres.append(default_genre)

        # Normalize all genres
        normalized = [
            self.normalize_genre(g, treebank_code) for g in genres if g
        ]
        return list(set(normalized))  # Remove duplicates

    def validate_genre(self, genre: str) -> bool:
        """Check if a genre is a canonical UD genre.

        Args:
            genre: Genre label to validate

        Returns:
            True if canonical, False otherwise
        """
        return genre in self.canonical_genres
