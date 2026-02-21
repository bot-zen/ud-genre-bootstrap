"""Data loading utilities for Universal Dependencies."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

try:
    from ud_hf_parquet_tools import (
        materialize_comment_markers_batch as _materialize_comment_markers_batch,
    )
except Exception:
    _materialize_comment_markers_batch = None


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
            ud_source: Either "hf://<dataset-repo>" or "local://<path-to-ud-root>"
            ud_version: UD version (used as HF revision)
            metadata_path: Optional path to metadata.json file
        """
        self.ud_source = ud_source
        self.ud_version = ud_version
        self.metadata_path = metadata_path
        self.source_type, source_target = self._parse_ud_source(ud_source)
        self.hf_repo_id = source_target if self.source_type == "hf" else None
        self.local_root = (
            Path(source_target).resolve()
            if self.source_type == "local"
            else None
        )
        self.metadata = self._load_metadata()

    @staticmethod
    def _parse_ud_source(ud_source: str) -> Tuple[str, str]:
        """Parse and validate UD source URI."""
        if ud_source.startswith("hf://"):
            repo_id = ud_source.replace("hf://", "", 1).strip()
            if not repo_id:
                raise ValueError(
                    "Invalid ud_source: missing HuggingFace repo after 'hf://'"
                )
            return "hf", repo_id

        if ud_source.startswith("local://"):
            local_path = ud_source.replace("local://", "", 1).strip()
            if not local_path:
                raise ValueError(
                    "Invalid ud_source: missing local path after 'local://'"
                )
            return "local", local_path

        raise ValueError(
            "Invalid ud_source: expected 'hf://<repo-id>' or 'local://<path>'"
        )

    def _is_hf_source(self) -> bool:
        """Return whether source is HuggingFace, with fallback for test stubs."""
        source_type = getattr(self, "source_type", None)
        if source_type is not None:
            return source_type == "hf"
        return str(getattr(self, "ud_source", "")).startswith("hf://")

    def _get_local_root(self) -> Path:
        """Resolve local root path, with fallback for test stubs."""
        local_root = getattr(self, "local_root", None)
        if local_root is not None:
            return Path(local_root)

        ud_source = str(getattr(self, "ud_source", ""))
        if ud_source.startswith("local://"):
            return Path(ud_source.replace("local://", "", 1)).resolve()

        raise ValueError(
            "Local source requested but ud_source is not a valid 'local://' URI"
        )

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

        if self._is_hf_source():
            try:
                from huggingface_hub import hf_hub_download

                metadata_file = hf_hub_download(
                    repo_id=self.hf_repo_id,
                    filename="metadata.json",
                    repo_type="dataset",
                    revision=self.ud_version,
                )
                logger.info(f"Loading metadata from HuggingFace: {metadata_file}")
                with open(metadata_file) as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning(
                    "Could not load metadata.json from HF source %s@%s: %s",
                    self.hf_repo_id,
                    self.ud_version,
                    exc,
                )
                return {}

        metadata_file = self._get_local_root() / "metadata.json"
        if metadata_file.exists():
            logger.info(f"Loading metadata from local source: {metadata_file}")
            with open(metadata_file) as f:
                return json.load(f)

        logger.warning(
            "No metadata.json found at local source root: %s",
            metadata_file,
        )
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
    ) -> List[Path]:
        """Resolve local CoNLL-U files for a treebank split.

        Args:
            treebank_code: Treebank code
            split: Split name

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

        base_path = self._get_local_root()

        resolved_paths = []
        for rel_path in file_paths:
            file_path = base_path / rel_path

            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            resolved_paths.append(file_path)

        return resolved_paths

    def _load_local_treebank(self, treebank_code: str, split: str) -> Dataset:
        """Load treebank from local CoNLL-U files.

        Args:
            treebank_code: Treebank code
            split: Split name

        Returns:
            Dataset with parsed sentences
        """
        # Parse CoNLL-U files
        sentences = []
        for file_path in self._resolve_local_split_files(treebank_code, split):
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
        if self._is_hf_source():
            return load_dataset(
                self.hf_repo_id,
                treebank_code,
                split=split,
                revision=self.ud_version,
            )

        # Load from local CoNLL-U files
        return self._load_local_treebank(treebank_code, split)

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

        if self._is_hf_source():
            dataset = self.load_treebank(treebank_code, split)
            yield from self._iter_hf_metadata_sentences(dataset)
            return

        for file_path in self._resolve_local_split_files(treebank_code, split):
            logger.debug(f"Loading metadata from {file_path}")
            yield from self._iter_conllu_sentence_metadata(file_path)

    def _iter_hf_metadata_sentences(self, dataset: Dataset) -> Iterator[Dict[str, Any]]:
        """Iterate metadata rows from HF datasets with optional DuckDB materialization."""
        metadata_fields = ("sent_id", "text", "comments", "genre")
        select_columns = [field for field in metadata_fields if field in dataset.column_names]
        if select_columns:
            dataset = dataset.select_columns(select_columns)

        if _materialize_comment_markers_batch:
            try:
                for batch in dataset.iter(batch_size=2048):
                    materialized_batch = _materialize_comment_markers_batch(batch)
                    if not materialized_batch:
                        continue

                    first_column = next(iter(materialized_batch.values()))
                    row_count = len(first_column)

                    for idx in range(row_count):
                        metadata_sentence = {}
                        for field in metadata_fields:
                            if field not in materialized_batch:
                                continue
                            value = materialized_batch[field][idx]
                            if value is not None:
                                metadata_sentence[field] = value
                        yield metadata_sentence
                return
            except Exception as exc:
                logger.warning(
                    "Failed DuckDB comment materialization for HF metadata path; "
                    "falling back to row iteration: %s",
                    exc,
                )

        for sentence in dataset:
            metadata_sentence = {}
            if "sent_id" in sentence:
                metadata_sentence["sent_id"] = sentence["sent_id"]
            if "text" in sentence:
                metadata_sentence["text"] = sentence["text"]
            if "comments" in sentence:
                metadata_sentence["comments"] = sentence["comments"]
            if "genre" in sentence:
                metadata_sentence["genre"] = sentence["genre"]
            yield metadata_sentence if metadata_sentence else sentence

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
