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

    def get_available_splits(self, treebank_code: str) -> List[str]:
        """Get list of available splits for a treebank.

        Args:
            treebank_code: Treebank code

        Returns:
            List of available split names (e.g., ['train', 'dev', 'test'])
        """
        if treebank_code not in self.metadata:
            return []

        tb_meta = self.metadata[treebank_code]
        if 'splits' not in tb_meta:
            return []

        return list(tb_meta['splits'].keys())

    def _resolve_local_split_files(
        self,
        treebank_code: str,
        split: str,
        local_path: str = None,
    ) -> List[Path]:
        """Resolve local CoNLL-U files for a treebank split.

        Args:
            treebank_code: Treebank code
            split: Split name
            local_path: Base path to UD repos (overrides self.ud_source)

        Returns:
            List of existing CoNLL-U file paths
        """
        if treebank_code not in self.metadata:
            raise ValueError(f"Treebank {treebank_code} not found in metadata")

        tb_meta = self.metadata[treebank_code]
        if 'splits' not in tb_meta or split not in tb_meta['splits']:
            available_splits = self.get_available_splits(treebank_code)
            if available_splits:
                raise ValueError(
                    f"Split '{split}' not found for {treebank_code}. "
                    f"Available splits: {', '.join(available_splits)}"
                )
            else:
                raise ValueError(f"No splits found in metadata for {treebank_code}")

        split_info = tb_meta['splits'][split]
        file_paths = split_info['files']

        base_path_str = local_path if local_path else self.ud_source
        if base_path_str.startswith("local://"):
            base_path_str = base_path_str.replace("local://", "")
        base_path = Path(base_path_str)
        if not base_path.is_absolute():
            base_path = base_path.resolve()

        resolved_paths = []
        for rel_path in file_paths:
            file_path = base_path / rel_path

            # Metadata paths may include version dirs (e.g., r2.17) not present locally.
            if not file_path.exists():
                parts = Path(rel_path).parts
                if len(parts) >= 3:
                    alt_path = base_path / parts[0] / parts[2]
                    if alt_path.exists():
                        file_path = alt_path

            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            resolved_paths.append(file_path)

        return resolved_paths

    def _load_local_treebank(self, treebank_code: str, split: str, local_path: str = None) -> Dataset:
        """Load treebank from local CoNLL-U files.

        Args:
            treebank_code: Treebank code
            split: Split name
            local_path: Base path to UD repos (overrides self.ud_source)

        Returns:
            Dataset with parsed sentences
        """
        # Parse CoNLL-U files
        sentences = []
        for file_path in self._resolve_local_split_files(treebank_code, split, local_path):
            logger.debug(f"Loading {file_path}")
            sentences.extend(self._parse_conllu_file(file_path))

        # Convert to HuggingFace Dataset
        from datasets import Dataset as HFDataset
        return HFDataset.from_list(sentences)

    def _parse_conllu_file(self, file_path: Path) -> List[Dict]:
        """Parse a CoNLL-U file into sentence dictionaries.

        Args:
            file_path: Path to .conllu file

        Returns:
            List of sentence dictionaries
        """
        sentences = []
        current_sentence = {
            'sent_id': None,
            'text': None,
            'comments': [],
            'tokens': [],
            'lemmas': [],
            'upos': [],
            'xpos': [],
            'feats': [],
            'head': [],
            'deprel': [],
            'deps': [],
            'misc': [],
        }

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')

                # Empty line = end of sentence
                if not line:
                    if current_sentence['sent_id']:
                        sentences.append(current_sentence)
                        current_sentence = {
                            'sent_id': None,
                            'text': None,
                            'comments': [],
                            'tokens': [],
                            'lemmas': [],
                            'upos': [],
                            'xpos': [],
                            'feats': [],
                            'head': [],
                            'deprel': [],
                            'deps': [],
                            'misc': [],
                        }
                    continue

                # Comment line
                if line.startswith('#'):
                    current_sentence['comments'].append(line)

                    # Extract sent_id
                    if line.startswith('# sent_id'):
                        parts = line.split('=', 1)
                        if len(parts) == 2:
                            current_sentence['sent_id'] = parts[1].strip()
                    # Extract text
                    elif line.startswith('# text'):
                        parts = line.split('=', 1)
                        if len(parts) == 2:
                            current_sentence['text'] = parts[1].strip()
                    continue

                # Token line
                parts = line.split('\t')
                if len(parts) == 10:
                    # Skip multiword tokens (e.g., 1-2)
                    if '-' in parts[0] or '.' in parts[0]:
                        continue

                    current_sentence['tokens'].append(parts[1])  # FORM
                    current_sentence['lemmas'].append(parts[2])  # LEMMA
                    current_sentence['upos'].append(parts[3])    # UPOS
                    current_sentence['xpos'].append(parts[4])    # XPOS
                    current_sentence['feats'].append(parts[5])   # FEATS
                    current_sentence['head'].append(parts[6])    # HEAD
                    current_sentence['deprel'].append(parts[7])  # DEPREL
                    current_sentence['deps'].append(parts[8])    # DEPS
                    current_sentence['misc'].append(parts[9])    # MISC

        # Don't forget last sentence
        if current_sentence['sent_id']:
            sentences.append(current_sentence)

        return sentences

    def _iter_conllu_sentence_metadata(self, file_path: Path) -> Iterator[Dict]:
        """Iterate sentence-level metadata from a CoNLL-U file.

        This parser keeps only metadata comments and sentence IDs, avoiding
        token-level parsing for tasks that only need genre extraction.

        Args:
            file_path: Path to .conllu file

        Yields:
            Sentence dictionaries with at least `sent_id`, `text`, and `comments`
        """
        current_sent_id = None
        current_text = None
        current_comments = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')

                # Empty line = end of sentence
                if not line:
                    if current_sent_id is not None or current_comments:
                        yield {
                            'sent_id': current_sent_id,
                            'text': current_text,
                            'comments': current_comments,
                        }
                    current_sent_id = None
                    current_text = None
                    current_comments = []
                    continue

                # Keep only comment lines for metadata extraction
                if line.startswith('#'):
                    current_comments.append(line)
                    if line.startswith('# sent_id'):
                        parts = line.split('=', 1)
                        if len(parts) == 2:
                            current_sent_id = parts[1].strip()
                    elif line.startswith('# text'):
                        parts = line.split('=', 1)
                        if len(parts) == 2:
                            current_text = parts[1].strip()

        # Don't forget last sentence when file doesn't end with blank line
        if current_sent_id is not None or current_comments:
            yield {
                'sent_id': current_sent_id,
                'text': current_text,
                'comments': current_comments,
            }

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
            # Load from local CoNLL-U files
            # Remove local:// prefix if present
            local_path = self.ud_source
            if local_path.startswith("local://"):
                local_path = local_path.replace("local://", "")
            return self._load_local_treebank(treebank_code, split, local_path)

    def iter_treebank_sentences(
        self,
        treebank_code: str,
        split: str = "train",
        metadata_only: bool = False,
    ) -> Iterator[Dict]:
        """Iterate over sentences in a treebank split.

        Args:
            treebank_code: Treebank code (e.g., 'en_ewt')
            split: Split name ('train', 'dev', or 'test')
            metadata_only: If True, parse only metadata fields needed for genre extraction

        Yields:
            Sentence dictionaries
        """
        if not metadata_only:
            dataset = self.load_treebank(treebank_code, split)
            for sentence in dataset:
                yield sentence
            return

        if self.ud_source.startswith("hf://"):
            dataset = self.load_treebank(treebank_code, split)
            for sentence in dataset:
                metadata_sentence = {}
                if 'sent_id' in sentence:
                    metadata_sentence['sent_id'] = sentence['sent_id']
                if 'text' in sentence:
                    metadata_sentence['text'] = sentence['text']
                if 'comments' in sentence:
                    metadata_sentence['comments'] = sentence['comments']
                if 'genre' in sentence:
                    metadata_sentence['genre'] = sentence['genre']
                yield metadata_sentence if metadata_sentence else sentence
            return

        local_path = self.ud_source
        if local_path.startswith("local://"):
            local_path = local_path.replace("local://", "")

        for file_path in self._resolve_local_split_files(treebank_code, split, local_path):
            logger.debug(f"Loading metadata from {file_path}")
            yield from self._iter_conllu_sentence_metadata(file_path)

    def iter_all_treebanks(
        self, split: Optional[str] = None, treebank_filter: Optional[List[str]] = None
    ) -> Iterator[Tuple[str, str, Dataset]]:
        """Iterate over all treebanks.

        Args:
            split: If specified, only yield this split; otherwise yield all splits
            treebank_filter: If specified, only yield these treebank codes

        Yields:
            Tuple of (treebank_code, split_name, dataset)
        """
        treebank_codes = self.get_treebank_codes()

        # Filter treebanks if specified
        if treebank_filter:
            treebank_codes = [tb for tb in treebank_codes if tb in treebank_filter]

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
