"""Utilities for converting paper global-index splits to explicit sentence maps."""

from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from ud_genre_bootstrap.utils.data_loader import UDDataLoader


@dataclass
class SplitMapBuildStats:
    """Summary of index-split conversion."""

    output_path: Path
    available_partitions: Tuple[str, ...]
    selected_partitions: Tuple[str, ...]
    scanned_treebanks: int
    scanned_files: int
    scanned_sentences: int
    target_indices: int
    matched_target_indices: int
    rows_written: int
    rows_missing_sent_id: int
    unsupported_files: int
    load_errors: int
    load_error_treebanks: Tuple[str, ...]
    unmatched_target_indices: int
    max_target_index: int


def _normalize_partition_selection(partitions: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not partitions:
        return tuple()
    normalized: List[str] = []
    for partition in partitions:
        cleaned = str(partition).strip()
        if not cleaned:
            continue
        if cleaned not in normalized:
            normalized.append(cleaned)
    return tuple(normalized)


def _get_split_file_paths(data_loader: UDDataLoader, treebank_code: str, split_name: str) -> List[str]:
    """Read split file paths from metadata and return them sorted."""
    tb_meta = getattr(data_loader, "metadata", {}).get(treebank_code, {})
    split_meta = tb_meta.get("splits", {}).get(split_name, {})
    file_paths = split_meta.get("files", [])
    normalized_paths: List[str] = []
    for raw_path in file_paths:
        path_str = str(raw_path).strip()
        if path_str:
            normalized_paths.append(path_str)
    return sorted(normalized_paths)


def _treebank_sort_key(data_loader: UDDataLoader, treebank_code: str) -> Tuple[str, str]:
    """Sort treebanks by first metadata file path (paper-compatible traversal)."""
    split_names = data_loader.get_available_splits(treebank_code)
    first_file_paths: List[str] = []
    for split_name in split_names:
        split_files = _get_split_file_paths(data_loader, treebank_code, split_name)
        if split_files:
            first_file_paths.append(split_files[0])

    if first_file_paths:
        return (min(first_file_paths), treebank_code)
    return (f"~{treebank_code}", treebank_code)


def _split_sort_key(data_loader: UDDataLoader, treebank_code: str, split_name: str) -> Tuple[str, str]:
    """Sort splits by first metadata file path (paper-compatible traversal)."""
    split_files = _get_split_file_paths(data_loader, treebank_code, split_name)
    if split_files:
        return (split_files[0], split_name)
    return (f"~{split_name}", split_name)


def _extract_sent_id(sentence: Dict) -> Optional[str]:
    """Extract sent_id from sentence fields (with comment fallback)."""
    sent_id = sentence.get("sent_id")
    if sent_id is not None:
        sent_id_str = str(sent_id).strip()
        if sent_id_str:
            return sent_id_str

    comments = sentence.get("comments") or []
    if isinstance(comments, str):
        comments = [comments]
    for raw_comment in comments:
        comment = str(raw_comment).strip()
        if comment.startswith("#"):
            comment = comment[1:].strip()
        if not comment.lower().startswith("sent_id"):
            continue
        parts = comment.split("=", 1)
        if len(parts) != 2:
            continue
        extracted = parts[1].strip()
        if extracted:
            return extracted
    return None


class _SplitMapWriter:
    def write(
        self,
        partition: str,
        global_index: int,
        treebank: str,
        split: str,
        sent_id: str,
    ) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _ParquetSplitMapWriter(_SplitMapWriter):
    def __init__(self, output_path: Path, batch_size: int = 100_000):
        self.output_path = output_path
        self.batch_size = batch_size
        self.schema = pa.schema(
            [
                pa.field("partition", pa.string()),
                pa.field("global_index", pa.int64()),
                pa.field("treebank", pa.string()),
                pa.field("split", pa.string()),
                pa.field("sent_id", pa.string()),
            ]
        )
        self.writer = pq.ParquetWriter(output_path, self.schema)
        self._buffer = {
            "partition": [],
            "global_index": [],
            "treebank": [],
            "split": [],
            "sent_id": [],
        }

    def _flush(self) -> None:
        if not self._buffer["partition"]:
            return
        table = pa.table(self._buffer, schema=self.schema)
        self.writer.write_table(table)
        for key in self._buffer:
            self._buffer[key].clear()

    def write(
        self,
        partition: str,
        global_index: int,
        treebank: str,
        split: str,
        sent_id: str,
    ) -> None:
        self._buffer["partition"].append(partition)
        self._buffer["global_index"].append(global_index)
        self._buffer["treebank"].append(treebank)
        self._buffer["split"].append(split)
        self._buffer["sent_id"].append(sent_id)
        if len(self._buffer["partition"]) >= self.batch_size:
            self._flush()

    def close(self) -> None:
        self._flush()
        self.writer.close()


class _CsvSplitMapWriter(_SplitMapWriter):
    def __init__(self, output_path: Path, delimiter: str = ","):
        self.handle = output_path.open("w", encoding="utf-8", newline="")
        self.writer = csv.writer(self.handle, delimiter=delimiter)
        self.writer.writerow(["partition", "global_index", "treebank", "split", "sent_id"])

    def write(
        self,
        partition: str,
        global_index: int,
        treebank: str,
        split: str,
        sent_id: str,
    ) -> None:
        self.writer.writerow([partition, global_index, treebank, split, sent_id])

    def close(self) -> None:
        self.handle.close()


def _build_writer(output_path: Path) -> _SplitMapWriter:
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        return _ParquetSplitMapWriter(output_path)
    if suffix == ".csv":
        return _CsvSplitMapWriter(output_path, delimiter=",")
    if suffix == ".tsv":
        return _CsvSplitMapWriter(output_path, delimiter="\t")
    raise ValueError(
        f"Unsupported output format '{output_path.suffix}'. "
        "Use .parquet, .csv, or .tsv."
    )


def _load_partition_indices(split_pickle_path: Path) -> Dict[str, List[int]]:
    with split_pickle_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Split pickle must contain a dict: {partition: [global_indices]}.")

    normalized: Dict[str, List[int]] = {}
    for partition_name, raw_indices in payload.items():
        if not isinstance(raw_indices, list):
            raise ValueError(f"Partition '{partition_name}' must contain a list of indices.")
        indices: List[int] = []
        for raw_idx in raw_indices:
            if isinstance(raw_idx, bool):
                raise ValueError(
                    f"Invalid index value in partition '{partition_name}': {raw_idx!r}"
                )
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid index value in partition '{partition_name}': {raw_idx!r}"
                ) from exc
            if idx < 0:
                raise ValueError(
                    f"Negative global index in partition '{partition_name}': {idx}"
                )
            indices.append(idx)
        normalized[str(partition_name)] = indices
    return normalized


def build_sentence_split_map_from_index_split(
    split_pickle_path: Path,
    output_path: Path,
    partitions: Optional[Sequence[str]] = None,
    *,
    ud_source: Optional[str] = None,
    ud_version: str = "2.17",
    metadata_path: Optional[Path] = None,
    treebanks_root: Optional[Path] = None,
) -> SplitMapBuildStats:
    """Convert paper global-index split file to explicit sentence split mapping.

    Output columns:
      partition, global_index, treebank, split, sent_id
    """
    if treebanks_root is not None and ud_source is not None:
        raise ValueError(
            "Provide either treebanks_root or ud_source, not both."
        )
    if treebanks_root is not None:
        ud_source = f"local://{treebanks_root}"
    if not ud_source:
        raise ValueError(
            "ud_source is required (e.g., 'hf://universal-dependencies/universal_dependencies' "
            "or 'local:///path/to/ud')."
        )

    data_loader = UDDataLoader(
        ud_source=ud_source,
        ud_version=ud_version,
        metadata_path=metadata_path,
    )

    partition_indices = _load_partition_indices(split_pickle_path)
    available_partitions = tuple(sorted(partition_indices.keys()))
    selected_partitions = _normalize_partition_selection(partitions) or available_partitions

    unknown_partitions = [p for p in selected_partitions if p not in partition_indices]
    if unknown_partitions:
        raise ValueError(
            f"Unknown partition(s): {', '.join(unknown_partitions)}. "
            f"Available: {', '.join(available_partitions)}"
        )

    index_to_partition: Dict[int, str] = {}
    for partition_name in selected_partitions:
        for idx in partition_indices[partition_name]:
            existing = index_to_partition.get(idx)
            if existing is not None:
                raise ValueError(
                    f"Global index {idx} appears in multiple partitions: "
                    f"'{existing}' and '{partition_name}'."
                )
            index_to_partition[idx] = partition_name

    writer = _build_writer(output_path)
    scanned_treebanks = 0
    scanned_files = 0
    scanned_sentences = 0
    unsupported_files = 0
    load_errors = 0
    load_error_treebanks: set[str] = set()
    matched_target_indices = 0
    rows_written = 0
    rows_missing_sent_id = 0

    treebank_codes = sorted(
        data_loader.get_treebank_codes(),
        key=lambda treebank_code: _treebank_sort_key(data_loader, treebank_code),
    )
    if not treebank_codes:
        raise ValueError(
            "No treebanks found in metadata. Ensure metadata.json is available for "
            "the selected source or pass metadata_path."
        )

    try:
        for treebank_code in treebank_codes:
            scanned_treebanks += 1
            split_names = sorted(
                data_loader.get_available_splits(treebank_code),
                key=lambda split_name: _split_sort_key(data_loader, treebank_code, split_name),
            )
            for split_name in split_names:
                scanned_files += 1
                try:
                    sentence_iter = data_loader.iter_treebank_sentences(
                        treebank_code,
                        split_name,
                        metadata_only=True,
                    )
                    for sentence in sentence_iter:
                        sent_id = _extract_sent_id(sentence)
                        partition = index_to_partition.get(scanned_sentences)
                        if partition is not None:
                            matched_target_indices += 1
                            if sent_id:
                                writer.write(
                                    partition=partition,
                                    global_index=scanned_sentences,
                                    treebank=treebank_code,
                                    split=split_name,
                                    sent_id=sent_id,
                                )
                                rows_written += 1
                            else:
                                rows_missing_sent_id += 1
                        scanned_sentences += 1
                except Exception:
                    load_errors += 1
                    load_error_treebanks.add(treebank_code)
                    continue
    finally:
        writer.close()

    target_indices = len(index_to_partition)
    unmatched_target_indices = max(0, target_indices - matched_target_indices)
    max_target_index = max(index_to_partition.keys()) if index_to_partition else -1

    return SplitMapBuildStats(
        output_path=output_path,
        available_partitions=available_partitions,
        selected_partitions=tuple(selected_partitions),
        scanned_treebanks=scanned_treebanks,
        scanned_files=scanned_files,
        scanned_sentences=scanned_sentences,
        target_indices=target_indices,
        matched_target_indices=matched_target_indices,
        rows_written=rows_written,
        rows_missing_sent_id=rows_missing_sent_id,
        unsupported_files=unsupported_files,
        load_errors=load_errors,
        load_error_treebanks=tuple(sorted(load_error_treebanks)),
        unmatched_target_indices=unmatched_target_indices,
        max_target_index=max_target_index,
    )
