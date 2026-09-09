"""Release artifact helpers for genre-label exports."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from ud_genre_bootstrap.utils.genre_mapping import GenreMapper
from ud_genre_bootstrap.utils.release_identity import resolve_release_identity

logger = logging.getLogger(__name__)

HF_PUBLISH_FILES = ("README.md", "all_genres.parquet", "release_manifest.json")
PROJECT_REPOSITORY_URL = "https://github.com/bot-zen/ud-genre-bootstrap"
UD_WORKSHOP_PAPER_URL = "https://universaldependencies.org/udw26/papers/41_Paper.pdf"
POINT_OF_CONTACT = "appliedlinguisticsdevs@eurac.edu"


def resolve_config_name(config) -> str:
    """Resolve a stable config label for exported artifacts."""
    if getattr(config.output, "config_name", None):
        return str(config.output.config_name)

    config_path = getattr(config, "_config_path", None)
    if config_path:
        return Path(config_path).stem

    return "default"


def resolve_run_id(config) -> str:
    """Resolve a stable run identifier for exported artifacts."""
    if getattr(config.output, "run_id", None):
        return str(config.output.run_id)

    return f"{config.ud_version}-{resolve_config_name(config)}"


def resolve_ud_source_revision(config) -> str:
    """Resolve the source revision string recorded in community exports."""
    if getattr(config.output, "ud_source_revision", None):
        return str(config.output.ud_source_revision)

    return str(
        config.output.genres_revision
        or config.output.embeddings_revision
        or config.ud_version
    )


def build_release_row_metadata(config) -> Dict[str, Any]:
    """Build static provenance columns for row-level export."""
    return {
        "ud_version": str(config.ud_version),
        "model": str(config.embeddings.model),
        "pooling": str(config.embeddings.pooling),
        "clustering_method": str(config.clustering.method),
        "config_name": resolve_config_name(config),
        "run_id": resolve_run_id(config),
    }


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest for a local file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_if_present(source: Optional[str], destination_dir: Path) -> Optional[str]:
    if not source:
        return None

    source_path = Path(source)
    if not source_path.exists():
        return None

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source_path.name
    shutil.copy2(source_path, destination)
    return str(destination.relative_to(destination_dir.parent))


def _collect_mapping_files(config) -> List[Dict[str, str]]:
    mapping_dir = Path(config.output.genres_path) / "mappings"
    copied: List[Dict[str, str]] = []

    mapping_path = getattr(config.genre_extraction, "mapping_path", None)
    copied_path = _copy_if_present(mapping_path, mapping_dir)
    if copied_path:
        source_path = Path(mapping_path)
        copied.append({
            "source": str(mapping_path),
            "copied_to": copied_path,
            "sha256": file_sha256(source_path),
        })

    patterns_path = getattr(config.genre_extraction, "patterns_path", None)
    pattern_paths = patterns_path if isinstance(patterns_path, list) else [patterns_path]
    for pattern_path in pattern_paths:
        copied_path = _copy_if_present(pattern_path, mapping_dir)
        if copied_path:
            source_path = Path(pattern_path)
            copied.append({
                "source": str(pattern_path),
                "copied_to": copied_path,
                "sha256": file_sha256(source_path),
            })

    return copied


def _write_config_snapshot(config, output_path: Path) -> str:
    snapshot_path = output_path / "config.snapshot.yaml"
    config_path = getattr(config, "_config_path", None)

    if config_path and Path(config_path).exists():
        shutil.copy2(config_path, snapshot_path)
    else:
        with open(snapshot_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(config), f, sort_keys=False, allow_unicode=False)

    return snapshot_path.name


def _load_baseline_summary(config, output_path: Path) -> Optional[Dict[str, Any]]:
    baseline_path = getattr(config.output, "baseline_summary_path", None)
    if not baseline_path:
        return None

    baseline_source = Path(baseline_path)
    if not baseline_source.exists():
        return None

    eval_dir = output_path / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    destination = eval_dir / "baseline_summary.json"
    shutil.copy2(baseline_source, destination)

    with open(destination, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    baseline["copied_to"] = str(destination.relative_to(output_path))
    baseline["source"] = str(baseline_source)
    return baseline


def build_algorithm_recipe(config) -> Dict[str, Any]:
    """Build the algorithm recipe recorded separately from public artifact identity."""
    return {
        "embeddings": {
            "model": str(config.embeddings.model),
            "pooling": str(config.embeddings.pooling),
            "batch_size": int(config.embeddings.batch_size),
            "layer": int(config.embeddings.layer),
        },
        "clustering": {
            "method": str(config.clustering.method),
            "level": str(config.clustering.level),
            "seed": int(config.clustering.seed),
            "max_iter": int(config.clustering.max_iter),
            "fit_sample_size": config.clustering.fit_sample_size,
            "reg_covar": float(config.clustering.reg_covar),
        },
        "thresholds": {
            "min_confidence": float(config.bootstrapping.min_confidence),
            "min_margin": float(config.bootstrapping.min_margin),
        },
        "bootstrapping": {
            "reference_weighting": str(config.bootstrapping.reference_weighting),
            "max_iterations": int(config.bootstrapping.max_iterations),
            "fail_on_incomplete": bool(config.bootstrapping.fail_on_incomplete),
            "unresolved_handling": str(config.bootstrapping.unresolved_handling),
        },
        "seed": int(config.clustering.seed),
    }


def _git_output(args: List[str], cwd: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    output = result.stdout.strip()
    return output or None


def _git_required(args: List[str], cwd: Path) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_run(args: List[str], cwd: Path) -> None:
    subprocess.run(
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _git_status_porcelain(cwd: Path) -> str:
    return _git_required(["git", "status", "--porcelain"], cwd)


def _git_tag_target(cwd: Path, tag: str) -> Optional[str]:
    return _git_output(["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{}}"], cwd)


def _git_current_commit(cwd: Path) -> str:
    return _git_required(["git", "rev-parse", "HEAD"], cwd)


def _git_tracked_files(cwd: Path) -> List[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=cwd,
        check=True,
        capture_output=True,
    )
    return [
        Path(raw.decode("utf-8"))
        for raw in result.stdout.split(b"\0")
        if raw
    ]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _build_git_metadata(release_identity: Dict[str, Any]) -> Dict[str, Any]:
    repo_root = _repo_root()
    detected_branch = _git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    detected_tags_raw = _git_output(["git", "tag", "--points-at", "HEAD"], repo_root)
    detected_tags = detected_tags_raw.splitlines() if detected_tags_raw else []
    detected_repo = _git_output(["git", "config", "--get", "remote.origin.url"], repo_root)
    detected_commit = _git_output(["git", "rev-parse", "HEAD"], repo_root)

    source_repo = release_identity.get("source_repo") or detected_repo
    source_branch = release_identity.get("source_branch") or detected_branch
    source_tag = release_identity.get("source_tag") or None
    source_commit = release_identity.get("source_commit") or detected_commit
    return {
        "repo": source_repo,
        "commit": source_commit,
        "branch": source_branch,
        "tag": source_tag,
        "configured_repo": release_identity.get("source_repo") or None,
        "configured_branch": release_identity.get("source_branch") or None,
        "configured_tag": release_identity.get("source_tag") or None,
        "configured_commit": release_identity.get("source_commit") or None,
        "detected_branch": detected_branch,
        "detected_tags": detected_tags,
        "detected_repo": detected_repo,
        "detected_commit": detected_commit,
    }


def list_release_upload_files(output_path: Path) -> List[Path]:
    """List release files eligible for HF upload."""
    return sorted(
        path
        for path in output_path.rglob("*")
        if path.is_file() and path.suffix != ".pkl" and not path.name.startswith(".")
    )


def list_release_publish_files(output_path: Path) -> List[Path]:
    """List the minimal Git-backed HF artifact payload."""
    return [
        output_path / relative_path
        for relative_path in HF_PUBLISH_FILES
        if (output_path / relative_path).exists()
    ]


def _canonical_genres_for_config(config) -> Set[str]:
    configured_genres = getattr(config.genre_extraction, "canonical_genres", None)
    if configured_genres is not None:
        return {str(genre) for genre in configured_genres}
    return set(GenreMapper.DEFAULT_UD_GENRES)


def _format_genre_counts(genre_counts: Dict[str, int]) -> str:
    return ", ".join(
        f"{genre}={count}"
        for genre, count in sorted(
            genre_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _round_float(value: Any, digits: int = 4) -> Optional[float]:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _percentage(count: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((count / denominator) * 100.0, 2)


def _format_int(value: Any) -> str:
    return f"{_safe_int(value):,}"


def _format_percent(value: Any) -> str:
    rounded = _round_float(value, digits=2)
    if rounded is None:
        return "n/a"
    return f"{rounded:.1f}%"


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "none"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _sorted_counts(counts: Optional[Dict[str, Any]]) -> List[tuple[str, int]]:
    return sorted(
        (
            (str(label), _safe_int(count))
            for label, count in (counts or {}).items()
            if label is not None
        ),
        key=lambda item: (-item[1], item[0]),
    )


def _method_description(method: str) -> str:
    descriptions = {
        "single-genre-treebank": "Directly inherited from a single-genre UD treebank.",
        "virtual-split": "Directly inherited from sentence/document metadata in a mixed treebank.",
        "bootstrap-labeled": "Cluster-derived label meeting confidence and margin thresholds.",
        "bootstrap-inferred": "Cluster-derived label below one or both uncertainty thresholds.",
    }
    return descriptions.get(method, "Other exported label provenance.")


def build_label_summary(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact label-count and provenance summary for release artifacts."""
    total_sentences = _safe_int(stats.get("total_sentences"))
    labeled_sentences = _safe_int(stats.get("labeled_sentences"))
    unlabeled_sentences = max(total_sentences - labeled_sentences, 0)
    method_counts = dict(_sorted_counts(stats.get("method_counts")))
    genre_counts = dict(_sorted_counts(stats.get("genre_counts")))

    metadata_methods = ("single-genre-treebank", "virtual-split")
    clustering_methods = ("bootstrap-labeled", "bootstrap-inferred")
    metadata_count = sum(method_counts.get(method, 0) for method in metadata_methods)
    clustering_count = sum(method_counts.get(method, 0) for method in clustering_methods)
    known_group_count = metadata_count + clustering_count
    other_count = max(labeled_sentences - known_group_count, 0)

    return {
        "total_sentences": total_sentences,
        "labeled_sentences": labeled_sentences,
        "unlabeled_sentences": unlabeled_sentences,
        "label_coverage_percent": _percentage(labeled_sentences, total_sentences),
        "exported_genres": len(genre_counts),
        "method_counts": method_counts,
        "method_percentages": {
            method: _percentage(count, labeled_sentences)
            for method, count in method_counts.items()
        },
        "method_descriptions": {
            method: _method_description(method)
            for method in method_counts
        },
        "provenance_groups": {
            "metadata_derived": {
                "count": metadata_count,
                "share_of_labeled_percent": _percentage(metadata_count, labeled_sentences),
                "methods": list(metadata_methods),
            },
            "clustering_derived": {
                "count": clustering_count,
                "share_of_labeled_percent": _percentage(clustering_count, labeled_sentences),
                "methods": list(clustering_methods),
            },
            "high_confidence_clustering": {
                "count": method_counts.get("bootstrap-labeled", 0),
                "share_of_labeled_percent": _percentage(
                    method_counts.get("bootstrap-labeled", 0),
                    labeled_sentences,
                ),
            },
            "lower_confidence_clustering": {
                "count": method_counts.get("bootstrap-inferred", 0),
                "share_of_labeled_percent": _percentage(
                    method_counts.get("bootstrap-inferred", 0),
                    labeled_sentences,
                ),
            },
            "other": {
                "count": other_count,
                "share_of_labeled_percent": _percentage(other_count, labeled_sentences),
            },
        },
        "genre_counts": genre_counts,
        "genre_percentages": {
            genre: _percentage(count, labeled_sentences)
            for genre, count in genre_counts.items()
        },
        "confidence_summary": stats.get("confidence_summary", {}),
    }


def _infer_evaluation_protocol(baseline_summary: Dict[str, Any]) -> Optional[str]:
    protocol = baseline_summary.get("protocol")
    if protocol:
        return str(protocol)

    text = " ".join(
        str(baseline_summary.get(field) or "")
        for field in ("name", "description", "command", "config")
    ).lower()
    if "paper_parity" in text or "paper-parity" in text:
        return "paper_parity"
    if "generalization" in text:
        return "generalization"
    return None


def _infer_evaluation_ud_version(baseline_summary: Dict[str, Any]) -> Optional[str]:
    explicit = baseline_summary.get("ud_version")
    if explicit:
        return str(explicit)

    text = " ".join(
        str(baseline_summary.get(field) or "")
        for field in ("name", "description", "command", "config", "source_log")
    )
    match = re.search(r"(?:UD\s*v?|ud[_ -]?)(2\.\d+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def build_evaluation_summary(
    baseline_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build compact evaluation metadata for release cards and manifests."""
    if not baseline_summary:
        return {
            "available": False,
            "note": "No locked evaluation baseline is configured for this artifact.",
        }

    return {
        "available": True,
        "name": baseline_summary.get("name"),
        "description": baseline_summary.get("description"),
        "protocol": _infer_evaluation_protocol(baseline_summary),
        "ud_version": _infer_evaluation_ud_version(baseline_summary),
        "treebank_set": baseline_summary.get("treebank_set"),
        "config": baseline_summary.get("config"),
        "source_log": baseline_summary.get("source_log"),
        "source": baseline_summary.get("source"),
        "copied_to": baseline_summary.get("copied_to"),
        "command": baseline_summary.get("command"),
        "metrics": baseline_summary.get("metrics", {}),
    }


def _format_label_summary_lines(label_summary: Dict[str, Any]) -> List[str]:
    provenance = label_summary.get("provenance_groups", {})
    metadata_group = provenance.get("metadata_derived", {})
    clustering_group = provenance.get("clustering_derived", {})

    lines = [
        "## Label Coverage And Provenance",
        f"- Total UD sentences in artifact: `{_format_int(label_summary['total_sentences'])}`",
        f"- Labeled sentences: `{_format_int(label_summary['labeled_sentences'])}` "
        f"({_format_percent(label_summary['label_coverage_percent'])})",
        f"- Unlabeled sentences: `{_format_int(label_summary['unlabeled_sentences'])}`",
        f"- Genres exported: `{_format_int(label_summary['exported_genres'])}`",
        "- Metadata-derived labels: "
        f"`{_format_int(metadata_group.get('count'))}` "
        f"({_format_percent(metadata_group.get('share_of_labeled_percent'))} of labeled sentences)",
        "- Clustering-derived labels: "
        f"`{_format_int(clustering_group.get('count'))}` "
        f"({_format_percent(clustering_group.get('share_of_labeled_percent'))} of labeled sentences)",
        "",
        "| Method | Meaning | Sentences | Share of labeled |",
        "| --- | --- | ---: | ---: |",
    ]

    method_counts = label_summary.get("method_counts", {})
    method_percentages = label_summary.get("method_percentages", {})
    method_descriptions = label_summary.get("method_descriptions", {})
    if method_counts:
        for method, count in method_counts.items():
            lines.append(
                f"| `{method}` | {method_descriptions.get(method, '')} | "
                f"{_format_int(count)} | {_format_percent(method_percentages.get(method))} |"
            )
    else:
        lines.append("| n/a | No method counts were recorded. | 0 | n/a |")

    confidence_summary = label_summary.get("confidence_summary") or {}
    if confidence_summary:
        lines.extend([
            "",
            "Confidence scores are top-1 cluster-label similarity scores where available.",
            f"- Mean confidence: `{_format_metric(confidence_summary.get('mean'))}`",
            f"- Median confidence: `{_format_metric(confidence_summary.get('median'))}`",
        ])

    lines.append("")
    return lines


def _format_genre_distribution_lines(label_summary: Dict[str, Any]) -> List[str]:
    lines = [
        "## Genre Distribution",
        "| Genre | Sentences | Share of labeled |",
        "| --- | ---: | ---: |",
    ]
    genre_counts = label_summary.get("genre_counts", {})
    genre_percentages = label_summary.get("genre_percentages", {})
    if genre_counts:
        for genre, count in genre_counts.items():
            lines.append(
                f"| `{genre}` | {_format_int(count)} | "
                f"{_format_percent(genre_percentages.get(genre))} |"
            )
    else:
        lines.append("| n/a | 0 | n/a |")
    lines.append("")
    return lines


def _format_evaluation_summary_lines(
    config,
    evaluation_summary: Dict[str, Any],
) -> List[str]:
    lines = ["## Evaluation Summary"]
    if not evaluation_summary.get("available"):
        lines.extend([
            "No locked evaluation baseline is configured for this artifact.",
            "",
        ])
        return lines

    baseline_ud_version = evaluation_summary.get("ud_version")
    if baseline_ud_version and str(baseline_ud_version) != str(config.ud_version):
        scope_note = (
            f"Train-level quality context measured on UD {baseline_ud_version}; "
            f"this artifact targets UD {config.ud_version}."
        )
    elif baseline_ud_version:
        scope_note = f"Locked evaluation baseline measured on UD {baseline_ud_version}."
    else:
        scope_note = "Locked train-level quality baseline for this release train."

    lines.extend([
        f"- Baseline: `{evaluation_summary.get('name') or 'locked baseline'}`",
        f"- Scope: {scope_note}",
        f"- Protocol: `{evaluation_summary.get('protocol') or 'n/a'}`",
        f"- Treebank set: `{evaluation_summary.get('treebank_set') or 'n/a'}`",
        f"- Config: `{evaluation_summary.get('config') or 'n/a'}`",
        f"- Source log: `{evaluation_summary.get('source_log') or 'n/a'}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ])

    metrics = evaluation_summary.get("metrics", {})
    metric_rows = [
        ("Overall Acc / Micro-F1", metrics.get("overall_micro_f1")),
        ("Macro-F1", metrics.get("macro_f1")),
        ("Mean Fold Micro-F1", metrics.get("mean_fold_micro_f1")),
        ("Std Fold Micro-F1", metrics.get("std_fold_micro_f1")),
        ("Mean Fold Macro-F1", metrics.get("mean_fold_macro_f1")),
        ("Std Fold Macro-F1", metrics.get("std_fold_macro_f1")),
        ("Purity (PUR)", metrics.get("purity")),
        ("Agreement (AGR)", metrics.get("agreement_treebank")),
        ("Overlap Error (Delta BC)", metrics.get("overlap_error_treebank")),
    ]
    for label, value in metric_rows:
        if value is not None:
            lines.append(f"| {label} | {_format_metric(value)} |")

    missing_anchor_genres = metrics.get("missing_anchor_genres")
    if missing_anchor_genres:
        lines.extend([
            "",
            "- Missing anchor genres in this baseline: "
            f"`{', '.join(str(genre) for genre in missing_anchor_genres)}`",
        ])

    lines.append("")
    return lines


def validate_release_genre_inventory(config, stats: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure a release export only contains configured canonical labels."""
    canonical_genres = _canonical_genres_for_config(config)
    genre_counts = {
        str(genre): int(count)
        for genre, count in (stats.get("genre_counts") or {}).items()
        if genre is not None
    }
    noncanonical_counts = {
        genre: count
        for genre, count in genre_counts.items()
        if genre not in canonical_genres
    }
    stats["canonical_genres"] = sorted(canonical_genres)
    stats["noncanonical_genre_counts"] = noncanonical_counts

    if noncanonical_counts:
        release_identity = resolve_release_identity(config)
        raise ValueError(
            "Release export contains non-canonical genre labels for "
            f"label_schema={release_identity['label_schema']!r}: "
            f"{_format_genre_counts(noncanonical_counts)}. "
            "Update the mapping/pattern config or regenerate the labels before "
            "publishing."
        )

    return stats


def summarize_exported_labels_file(output_file: Path) -> Dict[str, Any]:
    """Summarize an exported ``all_genres.parquet`` file."""
    import pandas as pd

    if not output_file.exists():
        return {
            "total_sentences": 0,
            "labeled_sentences": 0,
            "method_counts": {},
            "genre_counts": {},
            "confidence_summary": {},
            "canonical_genres": [],
            "noncanonical_genre_counts": {},
        }

    df = pd.read_parquet(output_file)
    method_counts = {
        str(method): int(count)
        for method, count in df.get("method", pd.Series(dtype=object)).value_counts().items()
    }
    genre_counts = {
        str(genre): int(count)
        for genre, count in df.get("genre", pd.Series(dtype=object)).dropna().value_counts().items()
    }
    confidence_summary: Dict[str, Any] = {}
    if "confidence" in df.columns:
        confidence = pd.to_numeric(df["confidence"], errors="coerce").dropna()
        if not confidence.empty:
            confidence_summary = {
                "count": int(confidence.count()),
                "mean": round(float(confidence.mean()), 4),
                "median": round(float(confidence.median()), 4),
                "min": round(float(confidence.min()), 4),
                "p25": round(float(confidence.quantile(0.25)), 4),
                "p75": round(float(confidence.quantile(0.75)), 4),
                "max": round(float(confidence.max()), 4),
            }
    return {
        "total_sentences": int(len(df)),
        "labeled_sentences": int(df.get("genre", pd.Series(dtype=object)).notna().sum()),
        "method_counts": method_counts,
        "genre_counts": genre_counts,
        "confidence_summary": confidence_summary,
        "canonical_genres": [],
        "noncanonical_genre_counts": {},
    }


def _size_category(row_count: int) -> str:
    if row_count < 1_000:
        return "n<1K"
    if row_count < 10_000:
        return "1K<n<10K"
    if row_count < 100_000:
        return "10K<n<100K"
    if row_count < 1_000_000:
        return "100K<n<1M"
    if row_count < 10_000_000:
        return "1M<n<10M"
    if row_count < 100_000_000:
        return "10M<n<100M"
    return "n>100M"


def _build_dataset_card_yaml(config, stats: Dict[str, Any]) -> str:
    """Build Hugging Face dataset-card YAML metadata."""
    metadata = {
        "pretty_name": f"UD Genre Labels {config.ud_version}",
        "license": "apache-2.0",
        "annotations_creators": ["machine-generated"],
        "language_creators": ["crowdsourced"],
        "multilinguality": ["multilingual"],
        "task_categories": ["text-classification"],
        "tags": [
            "universal-dependencies",
            "genre-classification",
            "sentence-classification",
            "multilingual",
            "linguistics",
            "derived-annotations",
            "text",
            "tabular",
            "datasets",
        ],
        "size_categories": [_size_category(int(stats.get("total_sentences", 0)))],
    }
    return "---\n" + yaml.safe_dump(metadata, sort_keys=False) + "---\n"


def _hf_dataset_url(ud_source: str) -> str:
    """Return a clickable URL for an HF dataset source when possible."""
    if ud_source.startswith("hf://"):
        repo_id = ud_source.replace("hf://", "", 1).strip("/")
        if repo_id:
            return f"https://huggingface.co/datasets/{repo_id}"
    return ud_source


def _display_source_repo(repo: Optional[str]) -> str:
    """Return a user-facing repository URL for GitHub SSH/HTTPS inputs."""
    if not repo:
        return PROJECT_REPOSITORY_URL

    repo = repo.strip()
    if repo.startswith("git@github.com:"):
        path = repo.replace("git@github.com:", "", 1)
        if path.endswith(".git"):
            path = path[:-4]
        return f"https://github.com/{path}"

    if repo.startswith("https://github.com/") and repo.endswith(".git"):
        return repo[:-4]

    return repo


def _build_dataset_card(
    config,
    stats: Dict[str, Any],
    baseline_summary: Optional[Dict[str, Any]],
    mapping_files: List[Dict[str, str]],
    release_identity: Dict[str, Any],
    git_metadata: Dict[str, Any],
    config_hash: str,
) -> str:
    label_summary = build_label_summary(stats)
    evaluation_summary = build_evaluation_summary(baseline_summary)
    mapping_lines = [
        f"- `{entry['source']}` (sha256: `{entry['sha256']}`)"
        for entry in mapping_files
    ] or ["- none copied"]

    public_revision = (
        release_identity.get("hf_branches", [None])[0]
        if release_identity.get("hf_branches")
        else config.ud_version
    )
    immutable_revision = release_identity.get("hf_tag") or release_identity["artifact_key"]
    hf_repo = release_identity.get("hf_repo") or config.output.genres_hf_repo
    ud_source_url = _hf_dataset_url(str(config.ud_source))
    source_repo_url = _display_source_repo(
        git_metadata.get("repo") or release_identity.get("source_repo")
    )

    body = "\n".join([
        f"# UD Genre Labels {release_identity['artifact_key']}",
        "",
        "Derived sentence-level genre annotations for the "
        f"[universal-dependencies/universal_dependencies]({ud_source_url}) Universal Dependencies dataset.",
        "These labels are produced by the bootstrapping pipeline and are not "
        "authoritative gold annotations.",
        "",
        "## Dataset Description",
        f"- Homepage: {source_repo_url}",
        f"- Repository: {source_repo_url}",
        f"- Source dataset: [{config.ud_source}]({ud_source_url})",
        f"- Paper: {UD_WORKSHOP_PAPER_URL}",
        f"- Point of Contact: {POINT_OF_CONTACT}",
        "",
        "## Dataset Summary",
        "This dataset provides a sentence-level genre layer aligned to the "
        "[universal-dependencies/universal_dependencies]"
        f"({ud_source_url}) Parquet release.",
        "Each row contains one derived genre label for one UD sentence and can be "
        "joined back to the UD source data by `(treebank, split, sent_id)`.",
        "",
        "The export is a derived annotation layer, not a replacement for the UD "
        "treebanks and not a hand-validated gold genre dataset.",
        "",
        *_format_label_summary_lines(label_summary),
        *_format_genre_distribution_lines(label_summary),
        "## Loading",
        "```python",
        "from datasets import load_dataset",
        "",
        "genres = load_dataset(",
        f"    \"{hf_repo}\",",
        f"    revision=\"{public_revision}\",",
        "    split=\"train\",",
        ")",
        "```",
        "",
        "The `train` split is the single exported split containing all sentence-level "
        "genre labels for this artifact.",
        "",
        "For immutable provenance, load the artifact tag:",
        "",
        "```python",
        "genres = load_dataset(",
        f"    \"{hf_repo}\",",
        f"    revision=\"{immutable_revision}\",",
        "    split=\"train\",",
        ")",
        "```",
        "",
        "## Joining With Universal Dependencies",
        "```python",
        "from datasets import load_dataset",
        "",
        "genres = load_dataset(",
        f"    \"{hf_repo}\",",
        f"    revision=\"{public_revision}\",",
        "    split=\"train\",",
        ")",
        "",
        "ud = load_dataset(",
        "    \"universal-dependencies/universal_dependencies\",",
        "    \"en_ewt\",",
        f"    revision=\"{resolve_ud_source_revision(config)}\",",
        "    split=\"train\",",
        ")",
        "",
        "genre_by_key = {",
        "    (row[\"treebank\"], row[\"split\"], row[\"sent_id\"]): row[\"genre\"]",
        "    for row in genres",
        "    if row[\"treebank\"] == \"en_ewt\" and row[\"split\"] == \"train\"",
        "}",
        "",
        "first = ud[0]",
        "genre = genre_by_key.get((\"en_ewt\", \"train\", first[\"sent_id\"]))",
        "```",
        "",
        "## Release Identity",
        f"- Train ID: `{release_identity['train_id']}`",
        f"- Artifact key: `{release_identity['artifact_key']}`",
        f"- Inventory status: `{release_identity.get('inventory_status') or 'n/a'}`",
        f"- HF branches: `{', '.join(release_identity.get('hf_branches', []))}`",
        f"- HF tag: `{release_identity.get('hf_tag') or 'n/a'}`",
        f"- HF default branch: `{release_identity.get('hf_default_branch') or 'main'}`",
        f"- HF repo: `{hf_repo}`",
        f"- UD version: `{release_identity['ud_version']}`",
        f"- Scope: `{release_identity['scope']}`",
        f"- Label schema: `{release_identity['label_schema']}`",
        f"- Artifact version: `{release_identity['artifact_version']}`",
        f"- Source repo: `{source_repo_url}`",
        f"- Source commit: `{git_metadata.get('commit') or 'unknown'}`",
        f"- Source branch: `{git_metadata.get('branch') or 'unknown'}`",
        f"- Source tag: `{git_metadata.get('tag') or 'none configured'}`",
        f"- Config SHA-256: `{config_hash}`",
        "- Canonical labels: "
        f"`{', '.join(stats.get('canonical_genres', [])) or 'not recorded'}`",
        "",
        "## Release Configuration",
        f"- Config: `{resolve_config_name(config)}`",
        f"- Run ID: `{resolve_run_id(config)}`",
        f"- UD source: `{config.ud_source}`",
        f"- UD source revision: `{resolve_ud_source_revision(config)}`",
        f"- Embeddings: `{config.embeddings.model}` / `{config.embeddings.pooling}`",
        f"- Clustering: `{config.clustering.method}`",
        f"- Reference weighting: `{config.bootstrapping.reference_weighting}`",
        "",
        "## Output Columns",
        "- `treebank`, `split`, `sent_id`: primary join key back to UD",
        "- `genre`: derived sentence label",
        "- `confidence`: top-1 similarity score for the assigned cluster label",
        "- `method`: `single-genre-treebank`, `virtual-split`, `bootstrap-labeled`, "
        "or `bootstrap-inferred`",
        "- `ud_version`, `model`, `pooling`, `clustering_method`, `config_name`, "
        "`run_id`: compact row-level provenance",
        "",
        "## Evaluation Framing",
        "- `paper_parity` is used only for comparison with the original GMM+L paper protocol.",
        "- End-user quality is tracked with sentence-level generalization metrics, "
        "which are stricter and more directly relevant for downstream annotation use.",
        "- Known limitation: some paper-era treebank genre inventories are not fully "
        "recoverable from current sentence-level metadata subsets.",
        "",
        *_format_evaluation_summary_lines(config, evaluation_summary),
        "## Release Summary",
        f"- Total sentences: `{_format_int(label_summary.get('total_sentences'))}`",
        f"- Labeled sentences: `{_format_int(label_summary.get('labeled_sentences'))}`",
        f"- Genres exported: `{_format_int(label_summary.get('exported_genres'))}`",
        "- Methods exported: "
        f"`{', '.join(label_summary.get('method_counts', {}).keys()) or 'none'}`",
        "",
        "## Source Mapping Files",
        *mapping_lines,
        "",
        "## Citation",
        f"Please cite the UD Workshop paper associated with this dataset: {UD_WORKSHOP_PAPER_URL}",
        "",
        "## Contact",
        f"Point of Contact: {POINT_OF_CONTACT}",
        "",
    ]) + "\n"
    return _build_dataset_card_yaml(config, stats) + body


def write_release_artifacts(
    config,
    output_path: Path,
    stats: Dict[str, Any],
    *,
    all_genres_path: Optional[Path] = None,
) -> Dict[str, str]:
    """Write release metadata files alongside exported labels."""
    output_path.mkdir(parents=True, exist_ok=True)
    stats = validate_release_genre_inventory(config, stats)

    snapshot_name = _write_config_snapshot(config, output_path)
    config_hash = file_sha256(output_path / snapshot_name)
    baseline_summary = _load_baseline_summary(config, output_path)
    mapping_files = _collect_mapping_files(config)
    mapping_file_hashes = {
        entry["copied_to"]: entry["sha256"]
        for entry in mapping_files
    }
    release_identity = resolve_release_identity(config)
    git_metadata = _build_git_metadata(release_identity)
    algorithm_recipe = build_algorithm_recipe(config)
    label_summary = build_label_summary(stats)
    evaluation_summary = build_evaluation_summary(baseline_summary)
    config_source_path = getattr(config, "_config_path", None)
    profile_source_path = getattr(config, "_release_profile_path", None)
    matrix_source_path = getattr(config, "_release_matrix_path", None)
    profile_hash = (
        file_sha256(Path(profile_source_path))
        if profile_source_path and Path(profile_source_path).exists()
        else None
    )
    matrix_hash = (
        file_sha256(Path(matrix_source_path))
        if matrix_source_path and Path(matrix_source_path).exists()
        else None
    )
    source_files = {
        "config": {
            "path": str(config_source_path) if config_source_path else None,
            "sha256": (
                file_sha256(Path(config_source_path))
                if config_source_path and Path(config_source_path).exists()
                else config_hash
            ),
        },
        "release_profile": (
            {"path": str(profile_source_path), "sha256": profile_hash}
            if profile_source_path and profile_hash
            else None
        ),
        "release_matrix": (
            {"path": str(matrix_source_path), "sha256": matrix_hash}
            if matrix_source_path and matrix_hash
            else None
        ),
        "baseline_summary": (
            {
                "path": baseline_summary.get("source"),
                "sha256": file_sha256(Path(baseline_summary["source"])),
            }
            if baseline_summary and baseline_summary.get("source")
            and Path(baseline_summary["source"]).exists()
            else None
        ),
        "mappings": [
            {
                "path": entry["source"],
                "sha256": entry["sha256"],
            }
            for entry in mapping_files
        ],
    }

    readme_path = output_path / "README.md"
    readme_path.write_text(
        _build_dataset_card(
            config,
            stats,
            baseline_summary,
            mapping_files,
            release_identity,
            git_metadata,
            config_hash,
        ),
        encoding="utf-8",
    )

    cluster_assignments = output_path / "clusters" / "cluster_assignments.parquet"
    cluster_statistics = output_path / "clusters" / "cluster_statistics.json"
    artifacts = {
        "all_genres": (
            str(all_genres_path.relative_to(output_path))
            if all_genres_path and all_genres_path.exists()
            else None
        ),
        "cluster_assignments": (
            str(cluster_assignments.relative_to(output_path))
            if cluster_assignments.exists()
            else None
        ),
        "cluster_statistics": (
            str(cluster_statistics.relative_to(output_path))
            if cluster_statistics.exists()
            else None
        ),
        "config_snapshot": snapshot_name,
        "baseline_summary": baseline_summary.get("copied_to") if baseline_summary else None,
        "dataset_card": readme_path.name,
        "release_manifest": "release_manifest.json",
    }
    hf_payload = [
        relative_path
        for relative_path in HF_PUBLISH_FILES
        if relative_path == "release_manifest.json" or (output_path / relative_path).exists()
    ]

    run_metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_id": release_identity["train_id"],
        "artifact_key": release_identity["artifact_key"],
        "artifact_id": release_identity["artifact_id"],
        "inventory_status": release_identity["inventory_status"],
        "scope": release_identity["scope"],
        "label_schema": release_identity["label_schema"],
        "artifact_version": release_identity["artifact_version"],
        "artifact_version_normalized": release_identity["artifact_version_normalized"],
        "hf_repo": release_identity["hf_repo"],
        "hf_branches": release_identity["hf_branches"],
        "hf_tag": release_identity["hf_tag"],
        "hf_default_branch": release_identity["hf_default_branch"],
        "hf_revisions": release_identity["hf_revisions"],
        "source_repo": git_metadata.get("repo"),
        "source_commit": git_metadata.get("commit"),
        "source_branch": git_metadata.get("branch"),
        "source_tag": git_metadata.get("tag"),
        "git_commit": git_metadata.get("commit"),
        "git_branch": git_metadata.get("branch"),
        "git_tag": git_metadata.get("tag"),
        "profile_hash": profile_hash,
        "matrix_hash": matrix_hash,
        "config_hash": config_hash,
        "mapping_file_hashes": mapping_file_hashes,
        "source_files": source_files,
        "algorithm_recipe": algorithm_recipe,
        "release_identity": release_identity,
        "git": git_metadata,
        "run_id": resolve_run_id(config),
        "config_name": resolve_config_name(config),
        "ud_version": str(config.ud_version),
        "ud_source": str(config.ud_source),
        "ud_source_revision": resolve_ud_source_revision(config),
        "embeddings": {
            "model": str(config.embeddings.model),
            "pooling": str(config.embeddings.pooling),
            "batch_size": int(config.embeddings.batch_size),
            "layer": int(config.embeddings.layer),
        },
        "clustering": {
            "method": str(config.clustering.method),
            "level": str(config.clustering.level),
            "seed": int(config.clustering.seed),
            "fit_sample_size": config.clustering.fit_sample_size,
        },
        "bootstrapping": {
            "min_confidence": float(config.bootstrapping.min_confidence),
            "min_margin": float(config.bootstrapping.min_margin),
            "reference_weighting": str(config.bootstrapping.reference_weighting),
            "max_iterations": int(config.bootstrapping.max_iterations),
        },
        "stats": stats,
        "label_summary": label_summary,
        "evaluation_summary": evaluation_summary,
        "canonical_genres": stats.get("canonical_genres", []),
        "noncanonical_genre_counts": stats.get("noncanonical_genre_counts", {}),
        "artifacts": artifacts,
        "hf_payload": hf_payload,
        "mapping_files": mapping_files,
    }

    run_metadata_path = output_path / "run_metadata.json"
    with open(run_metadata_path, "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    release_manifest = {
        "train_id": release_identity["train_id"],
        "artifact_key": release_identity["artifact_key"],
        "artifact_id": release_identity["artifact_id"],
        "inventory_status": release_identity["inventory_status"],
        "ud_version": release_identity["ud_version"],
        "scope": release_identity["scope"],
        "label_schema": release_identity["label_schema"],
        "artifact_version": release_identity["artifact_version"],
        "artifact_version_normalized": release_identity["artifact_version_normalized"],
        "hf_repo": release_identity["hf_repo"],
        "hf_branches": release_identity["hf_branches"],
        "hf_tag": release_identity["hf_tag"],
        "hf_default_branch": release_identity["hf_default_branch"],
        "hf_revisions": release_identity["hf_revisions"],
        "source_repo": git_metadata.get("repo"),
        "source_commit": git_metadata.get("commit"),
        "source_branch": git_metadata.get("branch"),
        "source_tag": git_metadata.get("tag"),
        "git_commit": git_metadata.get("commit"),
        "git_branch": git_metadata.get("branch"),
        "git_tag": git_metadata.get("tag"),
        "config_name": resolve_config_name(config),
        "run_id": resolve_run_id(config),
        "ud_source": str(config.ud_source),
        "ud_source_revision": resolve_ud_source_revision(config),
        "profile_hash": profile_hash,
        "matrix_hash": matrix_hash,
        "config_hash": config_hash,
        "mapping_file_hashes": mapping_file_hashes,
        "source_files": source_files,
        "algorithm_recipe": algorithm_recipe,
        "label_summary": label_summary,
        "evaluation_summary": evaluation_summary,
        "canonical_genres": stats.get("canonical_genres", []),
        "noncanonical_genre_counts": stats.get("noncanonical_genre_counts", {}),
        "artifacts": artifacts,
        "hf_payload": hf_payload,
    }
    manifest_path = output_path / "release_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(release_manifest, f, indent=2)

    return {
        "config_snapshot": snapshot_name,
        "baseline_summary": baseline_summary.get("copied_to") if baseline_summary else "",
        "dataset_card": readme_path.name,
        "run_metadata": run_metadata_path.name,
        "release_manifest": manifest_path.name,
    }


def prepare_release_directory(config, output_path: Path) -> Dict[str, str]:
    """Regenerate release metadata for an existing exported labels directory."""
    output_path.mkdir(parents=True, exist_ok=True)
    labels_path = output_path / "all_genres.parquet"
    stats = summarize_exported_labels_file(labels_path)
    return write_release_artifacts(
        config,
        output_path,
        stats,
        all_genres_path=labels_path if labels_path.exists() else None,
    )


def _validate_source_release_tag(source_repo_dir: Path, release_identity: Dict[str, Any]) -> None:
    source_status = _git_status_porcelain(source_repo_dir)
    if source_status:
        raise ValueError(
            "Source repository must be clean before publishing. "
            "Commit or remove pending changes first."
        )

    source_tag = release_identity.get("source_tag")
    if not source_tag:
        raise ValueError("release.source_tag is required for Git-backed publishing")

    source_commit = _git_current_commit(source_repo_dir)
    configured_commit = release_identity.get("source_commit")
    if configured_commit and str(configured_commit) != source_commit:
        raise ValueError(
            f"Configured source_commit {configured_commit} does not match "
            f"current source commit {source_commit}"
        )

    tag_target = _git_tag_target(source_repo_dir, str(source_tag))
    if not tag_target:
        raise ValueError(f"Source tag not found: {source_tag}")
    if tag_target != source_commit:
        raise ValueError(
            f"Source tag {source_tag} points at {tag_target}, "
            f"not current source commit {source_commit}"
        )


def _validate_hf_checkout(hf_repo_dir: Path) -> None:
    if not hf_repo_dir.exists() or not (hf_repo_dir / ".git").exists():
        raise ValueError(f"HF repo directory is not a Git checkout: {hf_repo_dir}")

    hf_status = _git_status_porcelain(hf_repo_dir)
    if hf_status:
        raise ValueError(
            "HF repository checkout must be clean before publishing. "
            "Commit or remove pending changes first."
        )


def _clear_tracked_payload(hf_repo_dir: Path) -> None:
    for relative_path in _git_tracked_files(hf_repo_dir):
        if relative_path.as_posix() == ".gitattributes":
            continue

        target = hf_repo_dir / relative_path
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists() or target.is_symlink():
            target.unlink()


def _copy_publish_payload(output_path: Path, hf_repo_dir: Path) -> List[str]:
    missing = [
        relative_path
        for relative_path in HF_PUBLISH_FILES
        if not (output_path / relative_path).exists()
    ]
    if missing:
        raise ValueError(
            f"Release directory is missing HF payload files: {', '.join(missing)}"
        )

    copied: List[str] = []
    for relative_path in HF_PUBLISH_FILES:
        source = output_path / relative_path
        destination = hf_repo_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(relative_path)
    return copied


def _has_staged_changes(repo_dir: Path) -> bool:
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=repo_dir,
        capture_output=True,
    )
    return result.returncode == 1


def publish_release_directory_to_hf_git(
    config,
    output_path: Path,
    hf_repo_dir: Path,
    *,
    include_main: bool = False,
    push: bool = False,
    dry_run: bool = False,
    source_repo_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Publish a release directory through a local HF dataset Git checkout."""
    output_path = Path(output_path)
    hf_repo_dir = Path(hf_repo_dir)
    source_repo_dir = Path(source_repo_dir) if source_repo_dir else _repo_root()

    release_identity = resolve_release_identity(config)
    hf_branches = [
        str(branch).strip()
        for branch in release_identity.get("hf_branches", [])
        if str(branch).strip()
    ]
    if not hf_branches:
        raise ValueError("At least one HF branch is required for Git-backed publishing")

    hf_tag = str(release_identity["hf_tag"])
    hf_default_branch = str(release_identity.get("hf_default_branch") or "main")
    target_branches = list(hf_branches)
    if include_main and hf_default_branch not in target_branches:
        target_branches.append(hf_default_branch)

    if not dry_run:
        _validate_source_release_tag(source_repo_dir, release_identity)
        _validate_hf_checkout(hf_repo_dir)

    prepare_release_directory(config, output_path)
    publish_files = [
        str(path.relative_to(output_path))
        for path in list_release_publish_files(output_path)
    ]

    plan = {
        "repo_id": release_identity.get("hf_repo"),
        "hf_repo_dir": str(hf_repo_dir),
        "train_id": release_identity["train_id"],
        "artifact_key": release_identity["artifact_key"],
        "artifact_id": release_identity["artifact_id"],
        "source_tag": release_identity.get("source_tag"),
        "hf_branches": hf_branches,
        "hf_tag": hf_tag,
        "hf_default_branch": hf_default_branch,
        "target_branches": target_branches,
        "files": publish_files,
        "dry_run": dry_run,
        "pushed": False,
    }
    if dry_run:
        return plan

    primary_branch = hf_branches[0]
    _git_run(["git", "checkout", "-B", primary_branch], hf_repo_dir)
    _clear_tracked_payload(hf_repo_dir)
    copied_files = _copy_publish_payload(output_path, hf_repo_dir)
    _git_run(["git", "add", "-A"], hf_repo_dir)

    if _has_staged_changes(hf_repo_dir):
        _git_run(
            ["git", "commit", "-m", f"Publish {release_identity['artifact_key']}"],
            hf_repo_dir,
        )

    hf_commit = _git_current_commit(hf_repo_dir)
    existing_tag_target = _git_tag_target(hf_repo_dir, hf_tag)
    if existing_tag_target and existing_tag_target != hf_commit:
        raise ValueError(
            f"HF tag {hf_tag} already exists at {existing_tag_target}; "
            f"refusing to move immutable artifact tag to {hf_commit}"
        )
    if not existing_tag_target:
        _git_run(["git", "tag", hf_tag, hf_commit], hf_repo_dir)

    for branch in target_branches:
        if branch != primary_branch:
            _git_run(["git", "branch", "-f", branch, hf_commit], hf_repo_dir)

    if push:
        for branch in target_branches:
            _git_run(["git", "push", "origin", branch], hf_repo_dir)
        _git_run(["git", "push", "origin", hf_tag], hf_repo_dir)

    return {
        **plan,
        "files": copied_files,
        "hf_commit": hf_commit,
        "dry_run": False,
        "pushed": push,
    }


def upload_release_directory_to_hub(
    config,
    output_path: Path,
    repo_id: str,
    revisions: List[str],
) -> Dict[str, Any]:
    """Upload an existing release directory to one or more HF dataset revisions."""
    from huggingface_hub import HfApi, create_repo

    normalized_revisions = [
        str(revision).strip()
        for revision in revisions
        if str(revision).strip()
    ]
    if not normalized_revisions:
        raise ValueError("At least one Hugging Face revision is required")

    token = config.output.hf_token
    if not token:
        raise ValueError(
            "No Hugging Face token configured. "
            "Set `output.hf_token` in config or via environment-backed config expansion."
        )

    create_repo(
        repo_id,
        token=token,
        repo_type="dataset",
        exist_ok=True,
        private=False,
    )
    logger.info("Created/verified repo: %s", repo_id)

    prepare_release_directory(config, output_path)
    upload_files = list_release_upload_files(output_path)

    api = HfApi()
    for target_revision in normalized_revisions:
        if hasattr(api, "create_branch") and target_revision != "main":
            try:
                api.create_branch(
                    repo_id=repo_id,
                    branch=target_revision,
                    repo_type="dataset",
                    token=token,
                    exist_ok=True,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to create/verify revision %s on %s: %s",
                    target_revision,
                    repo_id,
                    exc,
                )

        for artifact_path in upload_files:
            relative_path = artifact_path.relative_to(output_path)
            api.upload_file(
                path_or_fileobj=str(artifact_path),
                path_in_repo=str(relative_path),
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                revision=target_revision,
            )
            logger.info(
                "Uploaded %s to %s (revision: %s)",
                relative_path,
                repo_id,
                target_revision,
            )

    logger.info(
        "Release upload complete: %s (revisions: %s)",
        repo_id,
        ", ".join(normalized_revisions),
    )
    return {
        "repo_id": repo_id,
        "revisions": normalized_revisions,
        "files": [str(path.relative_to(output_path)) for path in upload_files],
    }
