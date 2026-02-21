"""Utilities for loading and applying sentence split maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

SplitKey = Tuple[str, str]


@dataclass
class SentenceSplitMap:
    """In-memory lookup for sentence-level evaluation split filtering."""

    source_path: Path
    split_to_sent_ids: Dict[SplitKey, Set[str]]
    total_rows: int
    selected_rows: int
    dropped_rows: int
    selected_partitions: Tuple[str, ...]

    @property
    def split_keys(self) -> Set[SplitKey]:
        """Return split keys represented in the map."""
        return set(self.split_to_sent_ids.keys())

    def includes_split(self, treebank: str, split: str) -> bool:
        """Return whether a treebank split is present in the map."""
        return (treebank, split) in self.split_to_sent_ids

    def includes_sentence(self, treebank: str, split: str, sent_id: str) -> bool:
        """Return whether a sentence belongs to the selected split map."""
        sent_ids = self.split_to_sent_ids.get((treebank, split))
        if sent_ids is None:
            return False
        return sent_id in sent_ids


@dataclass
class EmbeddingFilterStats:
    """Summary of embedding filtering against a sentence split map."""

    kept_splits: int = 0
    dropped_splits: int = 0
    kept_sentences: int = 0
    dropped_sentences: int = 0


def _read_split_map_frame(split_map_path: Path) -> pd.DataFrame:
    suffix = split_map_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(split_map_path)
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(split_map_path, sep=sep)
    raise ValueError(
        f"Unsupported split-map format '{split_map_path.suffix}'. "
        "Use .parquet, .csv, or .tsv."
    )


def _normalize_partition_selection(partitions: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not partitions:
        return tuple()
    normalized = []
    for part in partitions:
        cleaned = str(part).strip()
        if not cleaned:
            continue
        if cleaned not in normalized:
            normalized.append(cleaned)
    return tuple(normalized)


def load_sentence_split_map(
    split_map_path: Path,
    partitions: Optional[Sequence[str]] = None,
) -> SentenceSplitMap:
    """Load a sentence split map from parquet/csv/tsv.

    Expected columns:
      - treebank
      - split
      - sent_id
    Optional:
      - partition (used when filtering by `partitions`)
    """
    frame = _read_split_map_frame(split_map_path)
    required_columns = {"treebank", "split", "sent_id"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Split map is missing required column(s): {missing}. "
            "Required: treebank, split, sent_id."
        )

    total_rows = int(len(frame))
    selected_partitions = _normalize_partition_selection(partitions)

    if selected_partitions:
        if "partition" not in frame.columns:
            raise ValueError(
                "Split map filtering by partition requested, but column 'partition' is missing."
            )
        partition_values = frame["partition"].astype(str)
        available_partitions = set(partition_values.unique())
        unknown = [part for part in selected_partitions if part not in available_partitions]
        if unknown:
            raise ValueError(
                f"Unknown split-map partition(s): {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(available_partitions))}"
            )
        frame = frame[partition_values.isin(selected_partitions)]

    selected_rows = int(len(frame))

    before_drop = len(frame)
    frame = frame.dropna(subset=["treebank", "split", "sent_id"])
    dropped_rows = int(before_drop - len(frame))

    split_to_sent_ids: Dict[SplitKey, Set[str]] = {}
    for treebank, split, sent_id in zip(
        frame["treebank"].astype(str),
        frame["split"].astype(str),
        frame["sent_id"].astype(str),
    ):
        treebank_key = treebank.strip()
        split_key = split.strip()
        sent_id_key = sent_id.strip()
        if not treebank_key or not split_key or not sent_id_key:
            dropped_rows += 1
            continue
        split_to_sent_ids.setdefault((treebank_key, split_key), set()).add(sent_id_key)

    return SentenceSplitMap(
        source_path=split_map_path,
        split_to_sent_ids=split_to_sent_ids,
        total_rows=total_rows,
        selected_rows=selected_rows,
        dropped_rows=dropped_rows,
        selected_partitions=selected_partitions,
    )


def filter_embeddings_by_sentence_split_map(
    embeddings_by_tb: Mapping[SplitKey, Mapping[str, object]],
    split_map: SentenceSplitMap,
) -> Tuple[Dict[SplitKey, Dict[str, object]], EmbeddingFilterStats]:
    """Filter embeddings so they match a sentence split map exactly."""
    filtered: Dict[SplitKey, Dict[str, object]] = {}
    stats = EmbeddingFilterStats()

    for split_key, emb_data in embeddings_by_tb.items():
        allowed_sent_ids = split_map.split_to_sent_ids.get(split_key)
        if allowed_sent_ids is None:
            stats.dropped_splits += 1
            sent_ids = emb_data.get("sent_id", [])
            stats.dropped_sentences += len(sent_ids) if isinstance(sent_ids, list) else 0
            continue

        sent_ids = emb_data.get("sent_id", [])
        embeddings = emb_data.get("embedding")
        if sent_ids is None or embeddings is None:
            stats.dropped_splits += 1
            continue

        keep_indices: List[int] = [
            idx for idx, sent_id in enumerate(sent_ids) if sent_id in allowed_sent_ids
        ]
        if not keep_indices:
            stats.dropped_splits += 1
            stats.dropped_sentences += len(sent_ids)
            continue

        if isinstance(embeddings, np.ndarray):
            filtered_embeddings = embeddings[keep_indices]
        else:
            filtered_embeddings = [embeddings[idx] for idx in keep_indices]

        filtered_sent_ids = [sent_ids[idx] for idx in keep_indices]
        dropped_here = len(sent_ids) - len(filtered_sent_ids)

        filtered[split_key] = {
            **emb_data,
            "sent_id": filtered_sent_ids,
            "embedding": filtered_embeddings,
        }
        stats.kept_splits += 1
        stats.kept_sentences += len(filtered_sent_ids)
        stats.dropped_sentences += dropped_here

    return filtered, stats
