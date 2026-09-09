"""Shared CoNLL-U sentence metadata iteration utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

COMMENT_KV_RE = re.compile(r"^(?P<key>[^=]+?)\s*=\s*(?P<value>.*)$")
CONLLU_SPLIT_RE = re.compile(r"-ud-(?P<split>train|dev|test)\.conllu$")


def normalize_comment_key(key: str) -> str:
    """Normalize a CoNLL-U comment key for comparison and aggregation."""
    return " ".join(key.lower().split())


def parse_comment_key_value(
    comment: str,
    *,
    strip_conllu_comment_marker: bool = True,
) -> Optional[Tuple[str, str]]:
    """Parse comment key/value metadata into normalized key/value pairs."""
    comment = comment.strip()
    if strip_conllu_comment_marker and comment.startswith("#"):
        comment = comment[1:].strip()

    match = COMMENT_KV_RE.match(comment)
    if not match:
        return None
    return (
        normalize_comment_key(match.group("key")),
        match.group("value").strip(),
    )


def split_from_conllu_path(file_path: Path) -> Optional[str]:
    """Infer the UD split name from a CoNLL-U file name."""
    match = CONLLU_SPLIT_RE.search(file_path.name)
    if not match:
        return None
    return match.group("split")


def render_inherited_comments(
    comment_values: Dict[str, List[str]],
    inherited_newdoc_metadata: Dict[str, List[str]],
    inherited_newpar_metadata: Dict[str, List[str]],
) -> List[str]:
    """Render inherited document/paragraph metadata as synthetic comments."""
    existing = {
        (key, value)
        for key, values in comment_values.items()
        for value in values
    }
    inherited = []
    for metadata in (
        inherited_newdoc_metadata,
        inherited_newpar_metadata,
    ):
        for key, values in metadata.items():
            for value in values:
                if (key, value) not in existing:
                    inherited.append(f"# {key} = {value}")
    return inherited


def _normal_comment_text(
    comment: str,
    *,
    strip_conllu_comment_marker: bool,
) -> str:
    comment = comment.strip()
    if strip_conllu_comment_marker and comment.startswith("#"):
        comment = comment[1:].strip()
    return " ".join(comment.split())


def _has_scope_boundary(
    comments: List[str],
    scoped_values: Dict[str, List[str]],
    prefix: str,
    *,
    strip_conllu_comment_marker: bool,
) -> bool:
    return (
        prefix in scoped_values
        or f"{prefix} id" in scoped_values
        or any(
            _normal_comment_text(
                comment,
                strip_conllu_comment_marker=strip_conllu_comment_marker,
            )
            == prefix
            for comment in comments
        )
    )


@dataclass
class CommentMetadataInheritanceTracker:
    """Carry document and paragraph comment metadata across ordered sentences."""

    inherited_newdoc_metadata: Dict[str, List[str]] = field(default_factory=dict)
    inherited_newpar_metadata: Dict[str, List[str]] = field(default_factory=dict)
    strip_conllu_comment_marker: bool = True

    def apply(
        self,
        comments: List[str],
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        """Update state from one sentence's comments and return inherited metadata."""
        comment_values = _comment_values(
            comments,
            strip_conllu_comment_marker=self.strip_conllu_comment_marker,
        )
        sentence_newdoc_metadata = _scoped_comment_values(
            comments,
            "newdoc",
            strip_conllu_comment_marker=self.strip_conllu_comment_marker,
        )
        sentence_newpar_metadata = _scoped_comment_values(
            comments,
            "newpar",
            strip_conllu_comment_marker=self.strip_conllu_comment_marker,
        )

        if sentence_newdoc_metadata:
            if _has_scope_boundary(
                comments,
                sentence_newdoc_metadata,
                "newdoc",
                strip_conllu_comment_marker=self.strip_conllu_comment_marker,
            ):
                self.inherited_newdoc_metadata = {}
                self.inherited_newpar_metadata = {}
            for key, values in sentence_newdoc_metadata.items():
                self.inherited_newdoc_metadata[key] = list(values)

        if sentence_newpar_metadata:
            if _has_scope_boundary(
                comments,
                sentence_newpar_metadata,
                "newpar",
                strip_conllu_comment_marker=self.strip_conllu_comment_marker,
            ):
                self.inherited_newpar_metadata = {}
            for key, values in sentence_newpar_metadata.items():
                self.inherited_newpar_metadata[key] = list(values)

        return (
            comment_values,
            _copy_metadata(self.inherited_newdoc_metadata),
            _copy_metadata(self.inherited_newpar_metadata),
        )


@dataclass
class ConlluSentenceMetadata:
    """Sentence metadata parsed from one CoNLL-U sentence block."""

    file_path: Path
    split: Optional[str]
    index: int
    sent_id: Optional[str]
    text: Optional[str]
    comments: List[str]
    comment_values: Dict[str, List[str]]
    inherited_newdoc_metadata: Dict[str, List[str]]
    inherited_newpar_metadata: Dict[str, List[str]]
    token_rows: List[List[str]] = field(default_factory=list)

    def inherited_comments(self) -> List[str]:
        """Render inherited document/paragraph metadata as synthetic comments."""
        return render_inherited_comments(
            self.comment_values,
            self.inherited_newdoc_metadata,
            self.inherited_newpar_metadata,
        )

    def to_sentence_dict(
        self,
        *,
        include_inherited_comments: bool = True,
        include_comment_metadata: bool = True,
    ) -> Dict:
        """Convert the parsed metadata into the sentence dict used downstream."""
        comments = list(self.comments)
        if include_inherited_comments:
            comments.extend(self.inherited_comments())

        sentence = {
            "sent_id": self.sent_id,
            "text": self.text,
            "comments": comments,
        }
        if include_comment_metadata:
            sentence.update({
                "file": str(self.file_path),
                "split": self.split,
                "comment_values": self.comment_values,
                "inherited_newdoc_metadata": self.inherited_newdoc_metadata,
                "inherited_newpar_metadata": self.inherited_newpar_metadata,
            })
        return sentence


def _append_value(mapping: Dict[str, List[str]], key: str, value: str) -> None:
    mapping.setdefault(key, []).append(value)


def _copy_metadata(mapping: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {key: list(values) for key, values in mapping.items()}


def _comment_values(
    comments: List[str],
    *,
    strip_conllu_comment_marker: bool = True,
) -> Dict[str, List[str]]:
    values: Dict[str, List[str]] = {}
    for comment in comments:
        parsed = parse_comment_key_value(
            comment,
            strip_conllu_comment_marker=strip_conllu_comment_marker,
        )
        if not parsed:
            continue
        key, value = parsed
        _append_value(values, key, value)
    return values


def _scoped_comment_values(
    comments: List[str],
    prefix: str,
    *,
    strip_conllu_comment_marker: bool = True,
) -> Dict[str, List[str]]:
    values: Dict[str, List[str]] = {}
    for comment in comments:
        parsed = parse_comment_key_value(
            comment,
            strip_conllu_comment_marker=strip_conllu_comment_marker,
        )
        if not parsed:
            continue
        key, value = parsed
        if key == prefix or key.startswith(f"{prefix} "):
            _append_value(values, key, value)
    return values


def _extract_comment_value(
    comment_values: Dict[str, List[str]],
    key: str,
) -> Optional[str]:
    values = comment_values.get(key) or []
    if not values:
        return None
    return values[-1]


def iter_conllu_sentence_metadata(
    file_path: Path,
    *,
    include_tokens: bool = False,
) -> Iterator[ConlluSentenceMetadata]:
    """Iterate CoNLL-U sentences with parsed and inherited metadata."""
    split = split_from_conllu_path(file_path)
    current_comments: List[str] = []
    current_token_rows: List[List[str]] = []
    inheritance_tracker = CommentMetadataInheritanceTracker()
    sentence_index = 0

    def emit_current_sentence() -> Optional[ConlluSentenceMetadata]:
        nonlocal sentence_index

        if not current_comments and not current_token_rows:
            return None

        comment_values, inherited_newdoc_metadata, inherited_newpar_metadata = (
            inheritance_tracker.apply(current_comments)
        )

        sentence = ConlluSentenceMetadata(
            file_path=file_path,
            split=split,
            index=sentence_index,
            sent_id=_extract_comment_value(comment_values, "sent_id"),
            text=_extract_comment_value(comment_values, "text"),
            comments=list(current_comments),
            comment_values=comment_values,
            inherited_newdoc_metadata=_copy_metadata(inherited_newdoc_metadata),
            inherited_newpar_metadata=_copy_metadata(inherited_newpar_metadata),
            token_rows=list(current_token_rows) if include_tokens else [],
        )
        sentence_index += 1
        return sentence

    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                sentence = emit_current_sentence()
                if sentence:
                    yield sentence
                current_comments = []
                current_token_rows = []
                continue

            if line.startswith("#"):
                current_comments.append(line)
                continue

            if include_tokens:
                current_token_rows.append(line.split("\t"))

    sentence = emit_current_sentence()
    if sentence:
        yield sentence


def iter_sentence_dicts_with_inherited_comment_metadata(
    sentences: Iterable[Dict[str, Any]],
    *,
    split: Optional[str] = None,
    file_path: Optional[Path] = None,
    include_comment_metadata: bool = True,
    strip_conllu_comment_marker: bool = False,
) -> Iterator[Dict[str, Any]]:
    """Yield ordered sentence dictionaries with inherited comment metadata added.

    HF parquet rows store comment bodies without the leading CoNLL-U line marker,
    so the default keeps literal leading ``#`` characters as comment content.
    """
    inheritance_tracker = CommentMetadataInheritanceTracker(
        strip_conllu_comment_marker=strip_conllu_comment_marker,
    )

    for sentence in sentences:
        metadata_sentence = dict(sentence)
        comments = list(metadata_sentence.get("comments") or [])
        string_comments = [comment for comment in comments if isinstance(comment, str)]
        comment_values, inherited_newdoc_metadata, inherited_newpar_metadata = (
            inheritance_tracker.apply(string_comments)
        )
        metadata_sentence["comments"] = comments + render_inherited_comments(
            comment_values,
            inherited_newdoc_metadata,
            inherited_newpar_metadata,
        )

        if include_comment_metadata:
            if file_path is not None:
                metadata_sentence["file"] = str(file_path)
            if split is not None:
                metadata_sentence["split"] = split
            metadata_sentence["comment_values"] = comment_values
            metadata_sentence["inherited_newdoc_metadata"] = inherited_newdoc_metadata
            metadata_sentence["inherited_newpar_metadata"] = inherited_newpar_metadata

        yield metadata_sentence
