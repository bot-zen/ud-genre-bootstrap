"""Data loading utilities for Universal Dependencies."""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


class UDDataLoader:
    """Load Universal Dependencies data from HuggingFace or local files."""

    def __init__(
        self,
        ud_source: str,
        ud_version: str = "2.17",
        metadata_path: Optional[Path] = None,
    ):
        """Initialize UD data loader.

        Args:
            ud_source: Either "hf://commul/universal_dependencies" or local path
            ud_version: UD version (used as HF revision)
            metadata_path: Optional path to metadata.json file
        """
        self.ud_source = ud_source
        self.ud_version = ud_version
        self.metadata_path = metadata_path
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load UD metadata from HuggingFace or local file.

        Returns:
            Dictionary containing treebank metadata
        """
        if self.metadata_path:
            # Load from local file
            logger.info(f"Loading metadata from: {self.metadata_path}")
            with open(self.metadata_path) as f:
                return json.load(f)

        # Try to find metadata in standard locations
        possible_paths = [
            # Local UD data location
            Path("/home/egon/Documents/work/phd/src/data/external/huggingface/universal_dependencies/metadata.json"),
            Path("/home/egon/phd/src/data/external/huggingface/universal_dependencies/metadata.json"),
            # Relative to current directory
            Path("../huggingface/universal_dependencies/metadata.json"),
        ]

        for path in possible_paths:
            if path.exists():
                logger.info(f"Loading metadata from: {path}")
                with open(path) as f:
                    return json.load(f)

        # If no local metadata found, return empty dict
        logger.warning("No metadata.json found, treebank enumeration will not work")
        return {}

    def get_treebank_codes(self) -> List[str]:
        """Get list of all treebank codes (e.g., 'en_ewt', 'fr_gsd').

        Returns:
            List of treebank codes
        """
        return list(self.metadata.keys())

    def get_language_treebanks(self) -> Dict[str, List[str]]:
        """Get mapping of languages to their treebanks.

        Returns:
            Dictionary: {language: [treebank_codes]}
        """
        language_map = {}
        for tb_code, tb_meta in self.metadata.items():
            # Extract language code (e.g., 'en' from 'en_ewt')
            lang_code = tb_code.split('_')[0]

            if lang_code not in language_map:
                language_map[lang_code] = []

            language_map[lang_code].append(tb_code)

        return language_map

    def load_treebank(
        self, treebank_code: str, split: str = "train"
    ) -> Dataset:
        """Load a specific treebank split.

        Args:
            treebank_code: Treebank code (e.g., 'en_ewt')
            split: Split name ('train', 'dev', or 'test')

        Returns:
            HuggingFace Dataset containing the treebank data
        """
        if self.ud_source.startswith("hf://"):
            repo_id = self.ud_source.replace("hf://", "")
            return load_dataset(
                repo_id,
                treebank_code,
                split=split,
                revision=self.ud_version,
            )
        else:
            # TODO: Load from local files using ud-hf-parquet-tools
            raise NotImplementedError("Local file loading not yet implemented")

    def iter_all_treebanks(
        self, split: Optional[str] = None
    ) -> Iterator[Tuple[str, str, Dataset]]:
        """Iterate over all treebanks.

        Args:
            split: If specified, only yield this split; otherwise yield all splits

        Yields:
            Tuple of (treebank_code, split_name, dataset)
        """
        treebank_codes = self.get_treebank_codes()

        for tb_code in treebank_codes:
            splits_to_load = [split] if split else ["train", "dev", "test"]

            for split_name in splits_to_load:
                try:
                    dataset = self.load_treebank(tb_code, split_name)
                    yield tb_code, split_name, dataset
                except Exception as e:
                    # Skip if split doesn't exist
                    continue

    def get_treebank_genres(self, treebank_code: str) -> List[str]:
        """Get genres for a specific treebank from metadata.

        Args:
            treebank_code: Treebank code

        Returns:
            List of genre labels for this treebank
        """
        if treebank_code not in self.metadata:
            logger.warning(f"Treebank {treebank_code} not found in metadata")
            return []

        tb_meta = self.metadata[treebank_code]

        # Extract genres from metadata
        if 'genre' in tb_meta:
            return tb_meta['genre']
        elif 'genres' in tb_meta:
            return tb_meta['genres']

        return []

    def get_all_treebank_metadata(self) -> List[Dict]:
        """Get metadata for all treebanks.

        Returns:
            List of treebank metadata dicts with 'id' and 'genres' fields
        """
        treebank_data = []
        for tb_code, tb_meta in self.metadata.items():
            genres = self.get_treebank_genres(tb_code)
            language = tb_meta.get('lcode', tb_code.split('_')[0])

            treebank_data.append({
                'id': tb_code,
                'genres': genres,
                'language': language,
            })

        return treebank_data

    def extract_sentence_text(self, sentence: Dict) -> str:
        """Extract text from a UD sentence.

        Args:
            sentence: Sentence dictionary from dataset

        Returns:
            Sentence text as string
        """
        # Handle different text representations
        if "text" in sentence:
            return sentence["text"]

        # Fallback: join tokens
        if "tokens" in sentence:
            return " ".join(sentence["tokens"])

        # Another fallback: join form field
        if "form" in sentence:
            return " ".join(sentence["form"])

        raise ValueError("Cannot extract text from sentence")
