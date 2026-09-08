"""Audit UD treebank READMEs for genre extraction opportunities."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from ud_genre_bootstrap.utils.genre_mapping import GenreMapper

README_NAMES = ("README.md", "README.txt")
CONLLU_NAME_RE = re.compile(r"(?P<treebank>.+)-ud-(?P<split>train|dev|test)\.conllu$")
README_GENRE_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:genre|genres)\s*:\s*(?P<value>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
DIRECT_GENRE_RE = re.compile(r"^#\s*(?:newdoc\s+)?genre\s*=\s*(?P<genre>\S+)")
COMMENT_KEY_RE = re.compile(r"^#\s*(?P<key>[^=]+?)\s*=")
README_HINTS = {
    "explicit_genre_comment": re.compile(r"#\s*(?:newdoc\s+)?genre\s*=", re.IGNORECASE),
    "sent_id": re.compile(r"\bsent[_ -]?id\b", re.IGNORECASE),
    "document_id": re.compile(r"\b(?:newdoc|doc(?:ument)?[_ -]?id|doc[_ -]?name)\b", re.IGNORECASE),
    "file_source": re.compile(r"\b(?:file(?:name)?|source|origin|corpus part)\b", re.IGNORECASE),
}
PRIORITY_ORDER = {
    "high": 0,
    "medium": 1,
    "integrated": 2,
    "covered": 3,
    "low": 4,
    "unknown": 5,
}
AUDIT_SORT_MODES = {
    "priority",
    "priority-sentences",
    "sentences",
    "uncovered-sentences",
}


@dataclass
class TreebankReadmeDescriptor:
    """Static files belonging to one UD treebank directory."""

    treebank: str
    dirname: str
    directory: Path
    readme_path: Optional[Path]
    conllu_files: List[Path]


@dataclass
class SentenceMetadataScan:
    """Sentence-level metadata summary for one treebank."""

    total_sentences: int = 0
    sentences_with_extracted_genre: int = 0
    extracted_genre_counts: Counter = field(default_factory=Counter)
    raw_direct_genre_counts: Counter = field(default_factory=Counter)
    normalized_direct_genre_counts: Counter = field(default_factory=Counter)
    alias_counts: Counter = field(default_factory=Counter)
    unmapped_raw_genres: Counter = field(default_factory=Counter)
    noncanonical_extracted_genres: Counter = field(default_factory=Counter)
    comment_key_counts: Counter = field(default_factory=Counter)
    sent_id_token_counts: Counter = field(default_factory=Counter)
    scanned_files: List[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Return the share of scanned sentences with an extracted genre."""
        if self.total_sentences == 0:
            return 0.0
        return self.sentences_with_extracted_genre / self.total_sentences


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_optional_genre(
    mapper: GenreMapper,
    genre: str,
    treebank_code: Optional[str] = None,
) -> Optional[str]:
    normalized = mapper.normalize_genre(genre, treebank_code)
    if normalized is None:
        return None
    return str(normalized)


def _metadata_path_candidates(ud_root: Path, ud_version: Optional[str]) -> List[Path]:
    candidates = [ud_root / "metadata.json", ud_root.parent / "metadata.json"]
    if ud_version:
        candidates.extend([
            ud_root / f"metadata-{ud_version}.json",
            ud_root.parent / f"metadata-{ud_version}.json",
        ])
    return candidates


def resolve_default_metadata_path(
    ud_root: Path,
    ud_version: Optional[str] = None,
) -> Optional[Path]:
    """Find a likely metadata JSON file next to a local UD treebank checkout."""
    for candidate in _metadata_path_candidates(ud_root, ud_version):
        if candidate.exists():
            return candidate
    return None


def load_treebank_metadata(
    metadata_path: Optional[Path],
    ud_root: Path,
    ud_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Load metadata from an explicit path or a conventional local location."""
    resolved_path = metadata_path or resolve_default_metadata_path(ud_root, ud_version)
    if not resolved_path:
        return {}
    return _read_json(resolved_path)


def _treebank_code_from_conllu(path: Path) -> Optional[str]:
    match = CONLLU_NAME_RE.match(path.name)
    if not match:
        return None
    return match.group("treebank")


def _find_readme_path(treebank_dir: Path) -> Optional[Path]:
    for readme_name in README_NAMES:
        candidate = treebank_dir / readme_name
        if candidate.exists():
            return candidate
    return None


def _find_conllu_files(treebank_dir: Path) -> List[Path]:
    direct_files = sorted(treebank_dir.glob("*-ud-*.conllu"))
    if direct_files:
        return direct_files
    return sorted(treebank_dir.rglob("*-ud-*.conllu"))


def discover_treebank_readmes(
    ud_root: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[TreebankReadmeDescriptor]:
    """Discover treebank directories, README files, and CoNLL-U files."""
    metadata = metadata or {}
    dirname_to_code = {
        str(entry.get("dirname")): treebank_code
        for treebank_code, entry in metadata.items()
        if isinstance(entry, dict) and entry.get("dirname")
    }

    descriptors: List[TreebankReadmeDescriptor] = []
    for treebank_dir in sorted(path for path in ud_root.glob("UD_*") if path.is_dir()):
        conllu_files = _find_conllu_files(treebank_dir)
        treebank_code = dirname_to_code.get(treebank_dir.name)
        if not treebank_code:
            for conllu_file in conllu_files:
                treebank_code = _treebank_code_from_conllu(conllu_file)
                if treebank_code:
                    break
        if not treebank_code:
            continue

        descriptors.append(
            TreebankReadmeDescriptor(
                treebank=treebank_code,
                dirname=treebank_dir.name,
                directory=treebank_dir,
                readme_path=_find_readme_path(treebank_dir),
                conllu_files=conllu_files,
            )
        )

    return descriptors


def _candidate_readme_terms(mapper: GenreMapper) -> List[str]:
    terms = set(mapper.canonical_genres)
    for source, target in mapper.genre_mappings.items():
        if source.startswith("_") or ":" in source:
            continue
        if len(source) < 3:
            continue
        if target is None:
            continue
        terms.add(str(source))
    return sorted(terms, key=lambda value: (-len(value), value))


def _term_pattern(term: str) -> re.Pattern:
    return re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )


def extract_readme_genres(
    readme_text: str,
    mapper: GenreMapper,
    treebank_code: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Extract normalized genre labels from README ``Genre:`` lines."""
    found_genres = []
    genre_lines = []
    terms = _candidate_readme_terms(mapper)

    for match in README_GENRE_LINE_RE.finditer(readme_text):
        value = match.group("value").strip()
        if not value:
            continue
        genre_lines.append(value)
        for term in terms:
            if not _term_pattern(term).search(value):
                continue
            normalized = _normalize_optional_genre(mapper, term, treebank_code)
            if normalized and normalized in mapper.canonical_genres:
                found_genres.append(normalized)

    return sorted(set(found_genres)), genre_lines


def _normalize_metadata_genres(
    metadata_entry: Dict[str, Any],
    mapper: GenreMapper,
    treebank_code: str,
) -> List[str]:
    metadata_genres = metadata_entry.get("genre", []) if metadata_entry else []
    normalized = []
    for genre in metadata_genres or []:
        normalized_genre = _normalize_optional_genre(mapper, str(genre), treebank_code)
        if normalized_genre and normalized_genre in mapper.canonical_genres:
            normalized.append(normalized_genre)
    return sorted(set(normalized))


def _treebank_patternless_genre_mapping(
    mapper: GenreMapper,
    treebank_code: str,
) -> Dict[str, Any]:
    mapping: Dict[str, Any] = {}
    for pattern_entry in mapper.metadata_patterns.get(treebank_code, []):
        if not isinstance(pattern_entry, dict):
            continue
        if pattern_entry.get("pattern"):
            continue
        mapping.update(pattern_entry.get("genre_mapping") or {})
    return mapping


def _normalize_direct_genre(
    mapper: GenreMapper,
    raw_genre: str,
    treebank_code: str,
) -> Optional[str]:
    patternless_mapping = _treebank_patternless_genre_mapping(mapper, treebank_code)
    if raw_genre in patternless_mapping:
        mapped = patternless_mapping[raw_genre]
        if mapped is None:
            return None
        return _normalize_optional_genre(mapper, str(mapped), treebank_code)
    return _normalize_optional_genre(mapper, raw_genre, treebank_code)


def _iter_conllu_sentence_metadata(file_path: Path) -> Iterator[Dict[str, Any]]:
    current_sent_id = None
    current_text = None
    current_comments = []

    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                if current_sent_id is not None or current_comments:
                    yield {
                        "sent_id": current_sent_id,
                        "text": current_text,
                        "comments": current_comments,
                    }
                current_sent_id = None
                current_text = None
                current_comments = []
                continue

            if not line.startswith("#"):
                continue

            current_comments.append(line)
            if line.startswith("# sent_id"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    current_sent_id = parts[1].strip()
            elif line.startswith("# text"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    current_text = parts[1].strip()

    if current_sent_id is not None or current_comments:
        yield {
            "sent_id": current_sent_id,
            "text": current_text,
            "comments": current_comments,
        }


def _extract_direct_genre_comments(comments: Iterable[str]) -> List[str]:
    genres = []
    for comment in comments:
        match = DIRECT_GENRE_RE.search(comment.strip())
        if match:
            genres.append(match.group("genre"))
    return genres


def _comment_key(comment: str) -> Optional[str]:
    match = COMMENT_KEY_RE.match(comment.strip())
    if not match:
        return None
    return " ".join(match.group("key").lower().split())


def _sent_id_tokens(sent_id: Optional[str]) -> List[str]:
    if not sent_id:
        return []
    return [
        token.lower()
        for token in re.split(r"[^A-Za-z]+", sent_id)
        if token and token.lower() not in {"ud", "train", "dev", "test"}
    ][:4]


def scan_sentence_metadata(
    descriptor: TreebankReadmeDescriptor,
    mapper: GenreMapper,
    *,
    max_sentences: int = 0,
) -> SentenceMetadataScan:
    """Scan local CoNLL-U sentence comments for current genre extraction signals."""
    scan = SentenceMetadataScan(
        scanned_files=[
            str(path.relative_to(descriptor.directory))
            for path in descriptor.conllu_files
        ]
    )
    canonical_genres = set(mapper.canonical_genres)

    for conllu_file in descriptor.conllu_files:
        for sentence in _iter_conllu_sentence_metadata(conllu_file):
            if max_sentences and scan.total_sentences >= max_sentences:
                return scan

            scan.total_sentences += 1
            comments = sentence.get("comments", []) or []
            for comment in comments:
                key = _comment_key(comment)
                if key:
                    scan.comment_key_counts[key] += 1

            for token in _sent_id_tokens(sentence.get("sent_id")):
                scan.sent_id_token_counts[token] += 1

            for raw_genre in _extract_direct_genre_comments(comments):
                scan.raw_direct_genre_counts[raw_genre] += 1
                normalized = _normalize_direct_genre(mapper, raw_genre, descriptor.treebank)
                if normalized and normalized in canonical_genres:
                    scan.normalized_direct_genre_counts[normalized] += 1
                    if normalized != raw_genre:
                        scan.alias_counts[f"{raw_genre} -> {normalized}"] += 1
                elif normalized:
                    scan.unmapped_raw_genres[raw_genre] += 1

            extracted_genres = [
                str(genre)
                for genre in mapper.extract_genres_from_metadata(sentence, descriptor.treebank)
                if genre
            ]
            if extracted_genres:
                scan.sentences_with_extracted_genre += 1
                primary_genre = extracted_genres[0]
                scan.extracted_genre_counts[primary_genre] += 1
                if primary_genre not in canonical_genres:
                    scan.noncanonical_extracted_genres[primary_genre] += 1

    return scan


def _readme_hints(readme_text: str, readme_genres: List[str]) -> List[str]:
    hints = [
        hint_name
        for hint_name, pattern in README_HINTS.items()
        if pattern.search(readme_text)
    ]
    if readme_genres:
        hints.append("genre_line")
    return hints


def _classify_treebank(
    *,
    declared_genres: List[str],
    has_configured_patterns: bool,
    readme_hints: List[str],
    scan: SentenceMetadataScan,
    threshold: float,
) -> Tuple[str, str, List[str]]:
    is_multi_genre = len(declared_genres) >= 2
    reasons: List[str] = []

    if scan.unmapped_raw_genres:
        reasons.append("direct sentence genre comments include labels outside the schema")
        return "high", "add_mapping", reasons

    if scan.noncanonical_extracted_genres:
        reasons.append("current extraction emits labels outside the schema")
        return "high", "fix_pattern_or_mapping", reasons

    if scan.coverage >= threshold and scan.alias_counts:
        reasons.append("direct sentence genre aliases are already mapped to canonical labels")
        return "integrated", "covered_by_alias_mapping", reasons

    if not is_multi_genre:
        reasons.append("README or metadata declares a single genre")
        return "covered", "single_genre_treebank_metadata", reasons

    if scan.coverage >= threshold:
        if has_configured_patterns:
            reasons.append("configured patterns cover the multi-genre treebank")
            return "covered", "covered_by_configured_patterns", reasons
        if scan.raw_direct_genre_counts:
            reasons.append("built-in direct genre comment extraction covers the treebank")
            return "covered", "covered_by_direct_metadata", reasons
        reasons.append("current extraction reaches the coverage threshold")
        return "covered", "covered", reasons

    if has_configured_patterns:
        reasons.append("configured patterns exist but do not reach the coverage threshold")
        return "high", "review_existing_pattern", reasons

    if "explicit_genre_comment" in readme_hints and scan.raw_direct_genre_counts:
        reasons.append("README and sentence comments indicate direct genre metadata")
        return "high", "inspect_direct_genre_coverage", reasons

    id_hint_names = {"sent_id", "document_id", "file_source"}
    if id_hint_names.intersection(readme_hints):
        reasons.append("README mentions ID/source conventions for a multi-genre treebank")
        return "medium", "inspect_readme_hint_for_pattern", reasons

    reasons.append("multi-genre treebank lacks obvious sentence-level extraction hints")
    return "low", "manual_review", reasons


def _counter_to_dict(counter: Counter, limit: Optional[int] = None) -> Dict[str, int]:
    items = counter.most_common(limit)
    return {str(key): int(value) for key, value in items}


def _audit_descriptor(
    descriptor: TreebankReadmeDescriptor,
    mapper: GenreMapper,
    metadata_entry: Dict[str, Any],
    *,
    threshold: float,
    max_sentences_per_treebank: int,
) -> Dict[str, Any]:
    readme_text = ""
    if descriptor.readme_path:
        readme_text = descriptor.readme_path.read_text(encoding="utf-8", errors="replace")

    readme_genres, readme_genre_lines = extract_readme_genres(
        readme_text,
        mapper,
        descriptor.treebank,
    )
    metadata_genres = _normalize_metadata_genres(metadata_entry, mapper, descriptor.treebank)
    declared_genres = readme_genres or metadata_genres
    scan = scan_sentence_metadata(
        descriptor,
        mapper,
        max_sentences=max_sentences_per_treebank,
    )
    hints = _readme_hints(readme_text, readme_genres)
    has_configured_patterns = descriptor.treebank in mapper.metadata_patterns
    priority, action, reasons = _classify_treebank(
        declared_genres=declared_genres,
        has_configured_patterns=has_configured_patterns,
        readme_hints=hints,
        scan=scan,
        threshold=threshold,
    )

    return {
        "treebank": descriptor.treebank,
        "dirname": descriptor.dirname,
        "readme_path": str(descriptor.readme_path) if descriptor.readme_path else None,
        "readme_genres": readme_genres,
        "readme_genre_lines": readme_genre_lines,
        "metadata_genres": metadata_genres,
        "declared_genres": declared_genres,
        "configured_patterns": has_configured_patterns,
        "priority": priority,
        "action": action,
        "reasons": reasons,
        "readme_hints": hints,
        "sentence_scan": {
            "total_sentences": scan.total_sentences,
            "sentences_with_extracted_genre": scan.sentences_with_extracted_genre,
            "coverage": scan.coverage,
            "extracted_genre_counts": _counter_to_dict(scan.extracted_genre_counts),
            "raw_direct_genre_counts": _counter_to_dict(scan.raw_direct_genre_counts),
            "normalized_direct_genre_counts": _counter_to_dict(
                scan.normalized_direct_genre_counts
            ),
            "alias_counts": _counter_to_dict(scan.alias_counts),
            "unmapped_raw_genres": _counter_to_dict(scan.unmapped_raw_genres),
            "noncanonical_extracted_genres": _counter_to_dict(
                scan.noncanonical_extracted_genres
            ),
            "comment_key_counts": _counter_to_dict(scan.comment_key_counts, limit=20),
            "sent_id_token_counts": _counter_to_dict(scan.sent_id_token_counts, limit=20),
            "scanned_files": scan.scanned_files,
        },
    }


def _uncovered_sentence_count(item: Dict[str, Any]) -> int:
    scan = item["sentence_scan"]
    return int(scan["total_sentences"]) - int(scan["sentences_with_extracted_genre"])


def _sort_audit_items(
    items: List[Dict[str, Any]],
    *,
    sort_by: str = "priority",
) -> List[Dict[str, Any]]:
    if sort_by not in AUDIT_SORT_MODES:
        valid_modes = ", ".join(sorted(AUDIT_SORT_MODES))
        raise ValueError(f"Unknown audit sort mode '{sort_by}'. Use one of: {valid_modes}.")

    if sort_by == "priority-sentences":
        return sorted(
            items,
            key=lambda item: (
                PRIORITY_ORDER.get(item["priority"], 99),
                item["action"],
                -int(item["sentence_scan"]["total_sentences"]),
                float(item["sentence_scan"]["coverage"]),
                item["treebank"],
            ),
        )

    if sort_by == "sentences":
        return sorted(
            items,
            key=lambda item: (
                -int(item["sentence_scan"]["total_sentences"]),
                PRIORITY_ORDER.get(item["priority"], 99),
                item["action"],
                float(item["sentence_scan"]["coverage"]),
                item["treebank"],
            ),
        )

    if sort_by == "uncovered-sentences":
        return sorted(
            items,
            key=lambda item: (
                -_uncovered_sentence_count(item),
                PRIORITY_ORDER.get(item["priority"], 99),
                item["action"],
                float(item["sentence_scan"]["coverage"]),
                item["treebank"],
            ),
        )

    return sorted(
        items,
        key=lambda item: (
            PRIORITY_ORDER.get(item["priority"], 99),
            item["action"],
            float(item["sentence_scan"]["coverage"]),
            item["treebank"],
        ),
    )


def audit_genre_readmes(
    *,
    ud_root: Path,
    genre_mapper: GenreMapper,
    ud_version: Optional[str] = None,
    metadata_path: Optional[Path] = None,
    treebank_filter: Optional[Iterable[str]] = None,
    exclude_treebanks: Optional[Iterable[str]] = None,
    threshold: float = 0.95,
    max_sentences_per_treebank: int = 0,
    sort_by: str = "priority",
) -> Dict[str, Any]:
    """Audit local UD treebank READMEs and sentence metadata for extraction gaps."""
    ud_root = Path(ud_root).resolve()
    metadata = load_treebank_metadata(metadata_path, ud_root, ud_version)
    selected_treebanks = set(treebank_filter or [])
    excluded_treebanks = set(exclude_treebanks or [])
    descriptors = discover_treebank_readmes(ud_root, metadata)
    if selected_treebanks:
        descriptors = [
            descriptor
            for descriptor in descriptors
            if descriptor.treebank in selected_treebanks
        ]
    if excluded_treebanks:
        descriptors = [
            descriptor
            for descriptor in descriptors
            if descriptor.treebank not in excluded_treebanks
        ]

    treebank_items = []
    readme_genre_counts: Counter = Counter()
    metadata_genre_counts: Counter = Counter()
    for descriptor in descriptors:
        item = _audit_descriptor(
            descriptor,
            genre_mapper,
            metadata.get(descriptor.treebank, {}),
            threshold=threshold,
            max_sentences_per_treebank=max_sentences_per_treebank,
        )
        treebank_items.append(item)
        readme_genre_counts.update(item["readme_genres"])
        metadata_genre_counts.update(item["metadata_genres"])

    treebank_items = _sort_audit_items(treebank_items, sort_by=sort_by)
    action_counts = Counter(item["action"] for item in treebank_items)
    priority_counts = Counter(item["priority"] for item in treebank_items)
    candidate_items = [
        item
        for item in treebank_items
        if item["priority"] in {"high", "medium"}
    ]

    return {
        "ud_version": ud_version,
        "ud_root": str(ud_root),
        "metadata_path": str(metadata_path) if metadata_path else (
            str(resolve_default_metadata_path(ud_root, ud_version) or "")
        ),
        "threshold": threshold,
        "max_sentences_per_treebank": max_sentences_per_treebank,
        "sort_by": sort_by,
        "canonical_genres": sorted(genre_mapper.canonical_genres),
        "readme_genre_counts": _counter_to_dict(readme_genre_counts),
        "metadata_genre_counts": _counter_to_dict(metadata_genre_counts),
        "summary": {
            "treebanks": len(treebank_items),
            "readme_files": sum(1 for item in treebank_items if item["readme_path"]),
            "readme_genre_mentions": int(sum(readme_genre_counts.values())),
            "metadata_genre_mentions": int(sum(metadata_genre_counts.values())),
            "multi_genre_treebanks": sum(
                1 for item in treebank_items if len(item["declared_genres"]) >= 2
            ),
            "configured_pattern_treebanks": sum(
                1 for item in treebank_items if item["configured_patterns"]
            ),
            "candidate_treebanks": len(candidate_items),
            "priority_counts": _counter_to_dict(priority_counts),
            "action_counts": _counter_to_dict(action_counts),
        },
        "candidates": candidate_items,
        "treebanks": treebank_items,
    }


def write_audit_json(report: Dict[str, Any], output_path: Path) -> None:
    """Write an audit report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


def render_audit_markdown(report: Dict[str, Any], max_candidates: int = 50) -> str:
    """Render a compact Markdown audit report."""
    summary = report["summary"]
    lines = [
        "# Genre README Audit",
        "",
        f"- UD version: `{report.get('ud_version') or 'unknown'}`",
        f"- UD root: `{report['ud_root']}`",
        f"- Treebanks audited: `{summary['treebanks']}`",
        f"- README files: `{summary['readme_files']}`",
        f"- README genre mentions: `{summary['readme_genre_mentions']}`",
        f"- Candidate treebanks: `{summary['candidate_treebanks']}`",
        f"- Sort order: `{report.get('sort_by') or 'priority'}`",
        "",
        "## README Genre Scale",
        "",
        "| Genre | Treebanks |",
        "| --- | ---: |",
    ]
    for genre, count in report["readme_genre_counts"].items():
        lines.append(f"| `{genre}` | {count} |")

    lines.extend([
        "",
        "## Priority Summary",
        "",
        "| Priority | Treebanks |",
        "| --- | ---: |",
    ])
    for priority, count in report["summary"]["priority_counts"].items():
        lines.append(f"| `{priority}` | {count} |")

    lines.extend([
        "",
        "## Suggested Steps",
        "",
        "1. Fix `high` priority `add_mapping` and `fix_pattern_or_mapping` items first.",
        "2. Review `high` priority `review_existing_pattern` items before regeneration.",
        "3. Add only unambiguous `medium` priority README-derived patterns.",
        "4. Re-run `coverage` and this audit after mapping or pattern changes.",
        "5. Defer unclear `low` priority items until after the first regenerated artifact.",
        "",
        "## Candidates",
        "",
        (
            "| Priority | Action | Treebank | Sentences | Uncovered | Coverage | "
            "Declared genres | Reason |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ])
    for item in report["candidates"][:max_candidates]:
        declared = ", ".join(item["declared_genres"]) or "n/a"
        reason = "; ".join(item["reasons"]) or "n/a"
        total_sentences = int(item["sentence_scan"]["total_sentences"])
        uncovered_sentences = _uncovered_sentence_count(item)
        coverage = float(item["sentence_scan"]["coverage"]) * 100
        lines.append(
            "| "
            f"`{item['priority']}` | `{item['action']}` | `{item['treebank']}` | "
            f"{total_sentences} | {uncovered_sentences} | {coverage:.1f}% | "
            f"{declared} | {reason} |"
        )

    return "\n".join(lines) + "\n"


def write_audit_markdown(
    report: Dict[str, Any],
    output_path: Path,
    *,
    max_candidates: int = 50,
) -> None:
    """Write an audit report as Markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_audit_markdown(report, max_candidates=max_candidates),
        encoding="utf-8",
    )
