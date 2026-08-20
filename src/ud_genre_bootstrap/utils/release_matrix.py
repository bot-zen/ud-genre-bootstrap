"""Resolve train-based release matrices into runtime configs."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ud_genre_bootstrap.utils.config import (
    Config,
    _expand_config_variables,
    _expand_env_vars,
)
from ud_genre_bootstrap.utils.release_identity import (
    build_artifact_key,
    build_source_branch,
    parse_train_id,
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_relative(path: str, base_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def load_release_matrix_config(
    matrix_path: Path | str,
    *,
    ud_version: Optional[str] = None,
) -> Config:
    """Load a train matrix and resolve one UD-version runtime config."""
    matrix_path = Path(matrix_path)
    matrix = _load_yaml(matrix_path)
    matrix_dir = matrix_path.parent

    train = matrix.get("train", {})
    train_id = str(train.get("train_id") or matrix.get("train_id") or "")
    if not train_id:
        raise ValueError("release matrix must define train.train_id")
    train_parts = parse_train_id(train_id)

    versions = matrix.get("versions", {})
    if not isinstance(versions, dict) or not versions:
        raise ValueError("release matrix must define a non-empty versions mapping")

    supported_versions = [
        str(version)
        for version in train.get("supported_ud_versions", matrix.get("supported_ud_versions", versions.keys()))
    ]
    selected_ud_version = str(
        ud_version
        or train.get("default_ud_version")
        or matrix.get("default_ud_version")
        or ""
    )
    if not selected_ud_version:
        raise ValueError("pass --ud-version or set train.default_ud_version in the release matrix")
    if selected_ud_version not in supported_versions:
        raise ValueError(
            f"UD version {selected_ud_version!r} is not listed in supported_ud_versions"
        )

    version_override = versions.get(selected_ud_version)
    if version_override is None:
        raise ValueError(f"release matrix has no version entry for {selected_ud_version!r}")

    profile_ref = (
        matrix.get("release_profile")
        or train.get("release_profile")
        or matrix.get("profile")
    )
    if not profile_ref:
        raise ValueError("release matrix must define release_profile")
    profile_path = _resolve_relative(str(profile_ref), matrix_dir)
    profile = _load_yaml(profile_path)

    config_dict = _deep_merge(profile, matrix.get("defaults", {}))
    config_dict = _deep_merge(config_dict, version_override or {})
    config_dict["ud_version"] = selected_ud_version

    artifact_key = build_artifact_key(
        train_id=train_id,
        ud_version=selected_ud_version,
    )
    release = config_dict.setdefault("release", {})
    release.setdefault("train_id", train_id)
    release.setdefault("artifact_key", artifact_key)
    release.setdefault("scope", train.get("scope") or train_parts["scope"])
    release.setdefault("label_schema", train.get("label_schema") or train_parts["label_schema"])
    release.setdefault(
        "artifact_version",
        train.get("artifact_version") or train_parts["artifact_version"],
    )
    release.setdefault("inventory_status", train.get("status"))
    hf_tag = f"artifact/{train_id}/ud{selected_ud_version}"
    release.setdefault("hf_repo", train.get("hf_repo"))
    release.setdefault("hf_branches", [selected_ud_version])
    release.setdefault("hf_tag", hf_tag)
    release.setdefault("hf_revisions", [selected_ud_version, hf_tag])
    release.setdefault("hf_default_branch", train.get("hf_default_branch", "main"))
    release.setdefault("source_repo", train.get("source_repo"))
    release.setdefault("source_branch", train.get("source_branch") or build_source_branch(train_id))
    release.setdefault("source_tag", train.get("source_tag") or f"source/{train_id}")

    output = config_dict.setdefault("output", {})
    output.setdefault("genres_path", f"output/{selected_ud_version}-community-release/genres")
    output.setdefault("embeddings_revision", selected_ud_version)
    output.setdefault("genres_revision", selected_ud_version)
    output.setdefault("config_name", f"{selected_ud_version}-community-release")
    output.setdefault("run_id", artifact_key)
    output.setdefault("ud_source_revision", selected_ud_version)

    config_dict = _expand_env_vars(config_dict)
    config_dict = _expand_config_variables(config_dict)
    cfg = Config.from_dict(config_dict)
    setattr(cfg, "_release_matrix_path", str(matrix_path))
    setattr(cfg, "_release_profile_path", str(profile_path))
    setattr(cfg, "_config_path", str(matrix_path))
    return cfg
