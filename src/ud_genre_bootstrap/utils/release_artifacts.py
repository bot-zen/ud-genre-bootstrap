"""Release artifact helpers for genre-label exports."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ud_genre_bootstrap.utils.release_identity import resolve_release_identity


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
        "ud_source_revision": resolve_ud_source_revision(config),
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _build_git_metadata(release_identity: Dict[str, Any]) -> Dict[str, Any]:
    repo_root = _repo_root()
    detected_branch = _git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    detected_tags_raw = _git_output(["git", "tag", "--points-at", "HEAD"], repo_root)
    detected_tags = detected_tags_raw.splitlines() if detected_tags_raw else []

    git_branch = release_identity.get("git_branch") or detected_branch
    git_tag = release_identity.get("git_tag") or None
    return {
        "commit": _git_output(["git", "rev-parse", "HEAD"], repo_root),
        "branch": git_branch,
        "tag": git_tag,
        "configured_branch": release_identity.get("git_branch") or None,
        "configured_tag": release_identity.get("git_tag") or None,
        "detected_branch": detected_branch,
        "detected_tags": detected_tags,
    }


def list_release_upload_files(output_path: Path) -> List[Path]:
    """List release files eligible for HF upload."""
    return sorted(
        path
        for path in output_path.rglob("*")
        if path.is_file() and path.suffix != ".pkl" and not path.name.startswith(".")
    )


def _build_dataset_card(
    config,
    stats: Dict[str, Any],
    baseline_summary: Optional[Dict[str, Any]],
    mapping_files: List[Dict[str, str]],
    release_identity: Dict[str, Any],
    git_metadata: Dict[str, Any],
    config_hash: str,
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
        f"- `{entry['copied_to']}` (source: `{entry['source']}`, sha256: `{entry['sha256']}`)"
        for entry in mapping_files
    ] or ["- none copied"]

    public_revision = (
        release_identity.get("hf_revisions", [None])[0]
        if release_identity.get("hf_revisions")
        else config.ud_version
    )

    return "\n".join([
        f"# UD Genre Labels {release_identity['artifact_id']}",
        "",
        "Derived sentence-level genre annotations for Universal Dependencies treebanks.",
        "These labels are produced by the bootstrapping pipeline and are not "
        "authoritative gold annotations.",
        "",
        "## Loading",
        "```python",
        "from datasets import load_dataset",
        "",
        "genres = load_dataset(",
        f"    \"{release_identity.get('hf_repo') or config.output.genres_hf_repo}\",",
        f"    revision=\"{public_revision}\",",
        "    split=\"train\",",
        ")",
        "```",
        "",
        "## Release Identity",
        f"- Artifact ID: `{release_identity['artifact_id']}`",
        f"- Public revisions: `{', '.join(release_identity.get('hf_revisions', []))}`",
        f"- HF repo: `{release_identity.get('hf_repo') or 'n/a'}`",
        f"- UD version: `{release_identity['ud_version']}`",
        f"- Scope: `{release_identity['scope']}`",
        f"- Label schema: `{release_identity['label_schema']}`",
        f"- Artifact version: `{release_identity['artifact_version']}`",
        f"- Git commit: `{git_metadata.get('commit') or 'unknown'}`",
        f"- Git branch: `{git_metadata.get('branch') or 'unknown'}`",
        f"- Git tag: `{git_metadata.get('tag') or 'none configured'}`",
        f"- Config SHA-256: `{config_hash}`",
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
        "",
        "## Evaluation Framing",
        "- `paper_parity` is used only for comparison with the original GMM+L paper protocol.",
        "- End-user quality is tracked with sentence-level generalization metrics, "
        "which are stricter and more directly relevant for downstream annotation use.",
        "- Known limitation: some paper-era treebank genre inventories are not fully "
        "recoverable from current sentence-level metadata subsets.",
        "",
        *baseline_lines,
        "## Release Summary",
        f"- Total sentences: `{stats.get('total_sentences', 0)}`",
        f"- Labeled sentences: `{stats.get('labeled_sentences', 0)}`",
        f"- Genres exported: `{len(stats.get('genre_counts', {}))}`",
        "- Methods exported: "
        f"`{', '.join(sorted(stats.get('method_counts', {}).keys())) or 'none'}`",
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

    run_metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_id": release_identity["artifact_id"],
        "scope": release_identity["scope"],
        "label_schema": release_identity["label_schema"],
        "artifact_version": release_identity["artifact_version"],
        "hf_repo": release_identity["hf_repo"],
        "hf_revisions": release_identity["hf_revisions"],
        "git_commit": git_metadata.get("commit"),
        "git_branch": git_metadata.get("branch"),
        "git_tag": git_metadata.get("tag"),
        "config_hash": config_hash,
        "mapping_file_hashes": mapping_file_hashes,
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
        "artifacts": artifacts,
        "mapping_files": mapping_files,
    }

    run_metadata_path = output_path / "run_metadata.json"
    with open(run_metadata_path, "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    release_manifest = {
        "artifact_id": release_identity["artifact_id"],
        "ud_version": release_identity["ud_version"],
        "scope": release_identity["scope"],
        "label_schema": release_identity["label_schema"],
        "artifact_version": release_identity["artifact_version"],
        "hf_repo": release_identity["hf_repo"],
        "hf_revisions": release_identity["hf_revisions"],
        "git_commit": git_metadata.get("commit"),
        "git_branch": git_metadata.get("branch"),
        "git_tag": git_metadata.get("tag"),
        "config_name": resolve_config_name(config),
        "run_id": resolve_run_id(config),
        "config_hash": config_hash,
        "mapping_file_hashes": mapping_file_hashes,
        "algorithm_recipe": algorithm_recipe,
        "artifacts": artifacts,
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
