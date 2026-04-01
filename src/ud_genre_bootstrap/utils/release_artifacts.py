"""Release artifact helpers for genre-label exports."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


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
    """Static provenance columns for row-level export."""
    return {
        "ud_version": str(config.ud_version),
        "ud_source_revision": resolve_ud_source_revision(config),
        "model": str(config.embeddings.model),
        "pooling": str(config.embeddings.pooling),
        "clustering_method": str(config.clustering.method),
        "config_name": resolve_config_name(config),
        "run_id": resolve_run_id(config),
    }


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
        copied.append({"source": str(mapping_path), "copied_to": copied_path})

    patterns_path = getattr(config.genre_extraction, "patterns_path", None)
    pattern_paths = patterns_path if isinstance(patterns_path, list) else [patterns_path]
    for pattern_path in pattern_paths:
        copied_path = _copy_if_present(pattern_path, mapping_dir)
        if copied_path:
            copied.append({"source": str(pattern_path), "copied_to": copied_path})

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


def _build_dataset_card(
    config,
    stats: Dict[str, Any],
    baseline_summary: Optional[Dict[str, Any]],
    mapping_files: List[Dict[str, str]],
) -> str:
    baseline_lines: List[str] = []
    if baseline_summary:
        metrics = baseline_summary.get("metrics", {})
        baseline_lines = [
            "## Locked Baseline",
            f"- Overall Acc (Micro-F1): `{metrics.get('overall_micro_f1', 'n/a')}`",
            f"- Macro-F1: `{metrics.get('macro_f1', 'n/a')}`",
            f"- Purity (PUR): `{metrics.get('purity', 'n/a')}`",
            f"- Agreement (AGR): `{metrics.get('agreement_treebank', 'n/a')}`",
            f"- Overlap Error (ΔBC): `{metrics.get('overlap_error_treebank', 'n/a')}`",
            f"- Source: `{baseline_summary.get('source', 'n/a')}`",
            "",
        ]

    mapping_lines = [
        f"- `{entry['copied_to']}` (source: `{entry['source']}`)"
        for entry in mapping_files
    ] or ["- none copied"]

    return "\n".join([
        f"# UD Genre Labels {config.ud_version}",
        "",
        "Derived sentence-level genre annotations for Universal Dependencies treebanks.",
        "These labels are produced by the bootstrapping pipeline and are not authoritative gold annotations.",
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
        "- `method`: `single-genre-treebank`, `virtual-split`, `bootstrap-labeled`, or `bootstrap-inferred`",
        "",
        "## Evaluation Framing",
        "- `paper_parity` is used only for comparison with the original GMM+L paper protocol.",
        "- End-user quality is tracked with sentence-level generalization metrics, which are stricter and more directly relevant for downstream annotation use.",
        "- Known limitation: some paper-era treebank genre inventories are not fully recoverable from current sentence-level metadata subsets.",
        "",
        *baseline_lines,
        "## Release Summary",
        f"- Total sentences: `{stats.get('total_sentences', 0)}`",
        f"- Labeled sentences: `{stats.get('labeled_sentences', 0)}`",
        f"- Genres exported: `{len(stats.get('genre_counts', {}))}`",
        f"- Methods exported: `{', '.join(sorted(stats.get('method_counts', {}).keys())) or 'none'}`",
        "",
        "## Mapping Files",
        *mapping_lines,
        "",
    ]) + "\n"


def write_release_artifacts(
    config,
    output_path: Path,
    stats: Dict[str, Any],
    *,
    all_genres_path: Optional[Path] = None,
) -> Dict[str, str]:
    """Write release metadata files alongside exported labels."""
    output_path.mkdir(parents=True, exist_ok=True)

    snapshot_name = _write_config_snapshot(config, output_path)
    baseline_summary = _load_baseline_summary(config, output_path)
    mapping_files = _collect_mapping_files(config)

    readme_path = output_path / "README.md"
    readme_path.write_text(
        _build_dataset_card(config, stats, baseline_summary, mapping_files),
        encoding="utf-8",
    )

    cluster_assignments = output_path / "clusters" / "cluster_assignments.parquet"
    cluster_statistics = output_path / "clusters" / "cluster_statistics.json"
    run_metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
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
        "artifacts": {
            "all_genres": str(all_genres_path.relative_to(output_path)) if all_genres_path and all_genres_path.exists() else None,
            "cluster_assignments": str(cluster_assignments.relative_to(output_path)) if cluster_assignments.exists() else None,
            "cluster_statistics": str(cluster_statistics.relative_to(output_path)) if cluster_statistics.exists() else None,
            "config_snapshot": snapshot_name,
            "baseline_summary": baseline_summary.get("copied_to") if baseline_summary else None,
            "dataset_card": readme_path.name,
        },
        "mapping_files": mapping_files,
    }

    run_metadata_path = output_path / "run_metadata.json"
    with open(run_metadata_path, "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    return {
        "config_snapshot": snapshot_name,
        "baseline_summary": baseline_summary.get("copied_to") if baseline_summary else "",
        "dataset_card": readme_path.name,
        "run_metadata": run_metadata_path.name,
    }
