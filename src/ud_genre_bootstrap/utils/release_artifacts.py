"""Release artifact helpers for genre-label exports."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ud_genre_bootstrap.utils.release_identity import resolve_release_identity

logger = logging.getLogger(__name__)

HF_PUBLISH_FILES = ("README.md", "all_genres.parquet", "release_manifest.json")


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


def summarize_exported_labels_file(output_file: Path) -> Dict[str, Any]:
    """Summarize an exported ``all_genres.parquet`` file."""
    import pandas as pd

    if not output_file.exists():
        return {
            "total_sentences": 0,
            "labeled_sentences": 0,
            "method_counts": {},
            "genre_counts": {},
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
    return {
        "total_sentences": int(len(df)),
        "labeled_sentences": int(df.get("genre", pd.Series(dtype=object)).notna().sum()),
        "method_counts": method_counts,
        "genre_counts": genre_counts,
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
        "task_categories": ["text-classification"],
        "tags": [
            "universal-dependencies",
            "genre-classification",
            "sentence-classification",
            "multilingual",
            "text",
            "tabular",
            "datasets",
        ],
        "size_categories": [_size_category(int(stats.get("total_sentences", 0)))],
    }
    return "---\n" + yaml.safe_dump(metadata, sort_keys=False) + "---\n"


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
        f"- `{entry['source']}` (sha256: `{entry['sha256']}`)"
        for entry in mapping_files
    ] or ["- none copied"]

    public_revision = (
        release_identity.get("hf_branches", [None])[0]
        if release_identity.get("hf_branches")
        else config.ud_version
    )
    immutable_revision = release_identity.get("hf_tag") or release_identity["artifact_id"]

    body = "\n".join([
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
        "For immutable provenance, load the artifact tag:",
        "",
        "```python",
        "genres = load_dataset(",
        f"    \"{release_identity.get('hf_repo') or config.output.genres_hf_repo}\",",
        f"    revision=\"{immutable_revision}\",",
        "    split=\"train\",",
        ")",
        "```",
        "",
        "## Release Identity",
        f"- Artifact ID: `{release_identity['artifact_id']}`",
        f"- HF branches: `{', '.join(release_identity.get('hf_branches', []))}`",
        f"- HF tag: `{release_identity.get('hf_tag') or 'n/a'}`",
        f"- HF default branch: `{release_identity.get('hf_default_branch') or 'main'}`",
        f"- HF repo: `{release_identity.get('hf_repo') or 'n/a'}`",
        f"- UD version: `{release_identity['ud_version']}`",
        f"- Scope: `{release_identity['scope']}`",
        f"- Label schema: `{release_identity['label_schema']}`",
        f"- Artifact version: `{release_identity['artifact_version']}`",
        f"- Source repo: `{git_metadata.get('repo') or 'unknown'}`",
        f"- Source commit: `{git_metadata.get('commit') or 'unknown'}`",
        f"- Source branch: `{git_metadata.get('branch') or 'unknown'}`",
        f"- Source tag: `{git_metadata.get('tag') or 'none configured'}`",
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
        "## Source Mapping Files",
        *mapping_lines,
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
    config_source_path = getattr(config, "_config_path", None)
    source_files = {
        "config": {
            "path": str(config_source_path) if config_source_path else None,
            "sha256": (
                file_sha256(Path(config_source_path))
                if config_source_path and Path(config_source_path).exists()
                else config_hash
            ),
        },
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
        "artifact_id": release_identity["artifact_id"],
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
        "artifacts": artifacts,
        "hf_payload": hf_payload,
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
        "config_hash": config_hash,
        "mapping_file_hashes": mapping_file_hashes,
        "source_files": source_files,
        "algorithm_recipe": algorithm_recipe,
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
            ["git", "commit", "-m", f"Publish {release_identity['artifact_id']}"],
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
