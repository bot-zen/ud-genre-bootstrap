"""Data loading utilities for Universal Dependencies."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from datasets import Dataset, load_dataset

from ud_genre_bootstrap.utils.conllu import (
    iter_conllu_sentence_metadata,
    iter_sentence_dicts_with_inherited_comment_metadata,
)

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
        allow_partial_source: bool = False,
    ):
        """Initialize UD data loader.

        Args:
            ud_source: Either "hf://<dataset-repo>" or "local://<path-to-ud-root>"
            ud_version: UD version (used as HF revision)
            metadata_path: Optional path to metadata.json file
            allow_partial_source: If True, skip metadata-declared splits that fail
                to load. Defaults to strict release-safe behavior.

        """
        self.ud_source = ud_source
        self.ud_version = ud_version
        self.metadata_path = metadata_path
        self.allow_partial_source = allow_partial_source
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
        missing_paths = []
        for rel_path in file_paths:
            file_path = base_path / rel_path

            if not file_path.exists():
                missing_paths.append(file_path)
                continue

            resolved_paths.append(file_path)

        if missing_paths:
            formatted_paths = ", ".join(str(path) for path in missing_paths)
            message = (
                f"Missing {len(missing_paths)} file(s) for expected UD split "
                f"{treebank_code}:{split}: {formatted_paths}"
            )
            if getattr(self, "allow_partial_source", False):
                logger.warning("%s; skipping because allow_partial_source=True", message)
            else:
                raise FileNotFoundError(
                    f"{message}. Set allow_partial_ud_source: true only for "
                    "explicit partial-cache diagnostics."
                )

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
        for metadata in iter_conllu_sentence_metadata(file_path, include_tokens=True):
            if not metadata.sent_id:
                continue

            current_sentence = metadata.to_sentence_dict(
                include_inherited_comments=False,
                include_comment_metadata=False,
            )
            current_sentence.update({
                'tokens': [],
                'lemmas': [],
                'upos': [],
                'xpos': [],
                'feats': [],
                'head': [],
                'deprel': [],
                'deps': [],
                'misc': [],
            })

            for parts in metadata.token_rows:
                if len(parts) != 10:
                    continue

                # Skip multiword tokens (e.g., 1-2) and empty nodes.
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
        for metadata in iter_conllu_sentence_metadata(file_path):
            yield metadata.to_sentence_dict(
                include_inherited_comments=True,
                include_comment_metadata=True,
            )

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
            yield from self._iter_hf_metadata_sentences(dataset, split=split)
            return

        for file_path in self._resolve_local_split_files(treebank_code, split):
            logger.debug(f"Loading metadata from {file_path}")
            yield from self._iter_conllu_sentence_metadata(file_path)

    def _iter_hf_metadata_sentences(
        self,
        dataset: Dataset,
        split: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Iterate metadata rows from HF datasets with optional DuckDB materialization."""
        yield from iter_sentence_dicts_with_inherited_comment_metadata(
            self._iter_materialized_hf_metadata_rows(dataset),
            split=split,
        )

    def _iter_materialized_hf_metadata_rows(
        self,
        dataset: Dataset,
    ) -> Iterator[Dict[str, Any]]:
        """Iterate row-local metadata from HF datasets."""
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

        if not treebank_codes:
            raise RuntimeError(
                "No UD treebank metadata is available. For hf:// sources this "
                "usually means metadata.json is not available in the local cache "
                "or could not be downloaded."
            )

        # Filter treebanks if specified
        if treebank_filter:
            treebank_codes = [tb for tb in treebank_codes if tb in treebank_filter]

        if not treebank_codes:
            raise ValueError(
                "No requested treebanks were found in UD metadata: "
                f"{', '.join(treebank_filter or [])}"
            )

        loaded_split_count = 0
        for tb_code in treebank_codes:
            available_splits = self.get_available_splits(tb_code)
            if split:
                if available_splits and split not in available_splits:
                    continue
                splits_to_load = [split]
            elif available_splits:
                splits_to_load = available_splits
            else:
                message = f"No metadata-declared splits found for treebank {tb_code}"
                if getattr(self, "allow_partial_source", False):
                    logger.warning("%s; skipping because allow_partial_source=True", message)
                    continue
                raise RuntimeError(message)

            for split_name in splits_to_load:
                try:
                    dataset = self.load_treebank(tb_code, split_name)
                except Exception as exc:
                    message = f"Failed to load expected UD split {tb_code}:{split_name}"
                    if getattr(self, "allow_partial_source", False):
                        logger.warning(
                            "%s; skipping because allow_partial_source=True: %s",
                            message,
                            exc,
                        )
                        continue
                    raise RuntimeError(
                        f"{message}. If this is an hf:// source in offline mode, "
                        "ensure the complete UD revision is cached first. Set "
                        "allow_partial_ud_source: true only for explicit "
                        "partial-cache diagnostics."
                    ) from exc

                loaded_split_count += 1
                yield tb_code, split_name, dataset

        if loaded_split_count == 0:
            raise RuntimeError(
                "No UD treebank splits were loaded. This usually indicates an "
                "incomplete local cache, unavailable source data, or a split filter "
                "that does not match the selected UD metadata."
            )

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
