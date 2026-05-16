"""Release identity helpers for promoted genre artifacts."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ARTIFACT_VERSION_PATTERN = re.compile(
    r"^v(?P<major>[1-9][0-9]*)"
    r"(?:\.(?P<minor>0|[1-9][0-9]*))?"
    r"(?:\.(?P<patch>0|[1-9][0-9]*))?$"
)
ARTIFACT_ID_PATTERN = re.compile(
    r"^ud(?P<ud_version>\d+(?:\.\d+)+)-"
    r"(?P<scope>[a-z][a-z0-9_]*)-"
    r"(?P<label_schema>[a-z][a-z0-9_]*)-"
    r"(?P<artifact_version>v[1-9][0-9]*(?:\.(?:0|[1-9][0-9]*)){0,2})$"
)

VALID_REGISTRY_STATUSES = {"active", "deprecated", "superseded"}
VALID_REGISTRY_CHANGE_SCOPES = {"source_milestone", "artifact_patch"}


def parse_artifact_version(version: str) -> Dict[str, Any]:
    """Parse artifact version strings such as ``v1``, ``v1.2``, or ``v1.2.1``."""
    if not isinstance(version, str) or not version.strip():
        raise ValueError("artifact_version must be a non-empty string")

    match = ARTIFACT_VERSION_PATTERN.fullmatch(version.strip())
    if not match:
        raise ValueError(
            "artifact_version must match 'vMAJOR', 'vMAJOR.MINOR', "
            "or 'vMAJOR.MINOR.PATCH', for example 'v1.2.1'"
        )

    major = int(match.group("major"))
    minor = int(match.group("minor") or 0)
    patch = int(match.group("patch") or 0)
    return {
        "artifact_version": version.strip(),
        "major": major,
        "minor": minor,
        "patch": patch,
        "normalized": f"v{major}.{minor}.{patch}",
    }


def parse_artifact_id(artifact_id: str) -> Dict[str, str]:
    """Parse a canonical genre artifact id.

    Expected shape: ``ud<UD version>-<scope>-<label schema>-<artifact version>``.
    Example: ``ud2.17-full-ud-v1``.
    """
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise ValueError("artifact_id must be a non-empty string")

    match = ARTIFACT_ID_PATTERN.fullmatch(artifact_id.strip())
    if not match:
        raise ValueError(
            "artifact_id must match "
            "'ud<UD version>-<scope>-<label_schema>-<artifact_version>', "
            "for example 'ud2.17-full-ud-v1'"
        )

    parsed = match.groupdict()
    parsed["artifact_version_normalized"] = parse_artifact_version(
        parsed["artifact_version"]
    )["normalized"]
    return parsed


def build_artifact_id(
    *,
    ud_version: str,
    scope: str,
    label_schema: str,
    artifact_version: str,
) -> str:
    """Build a canonical artifact id from explicit identity components."""
    return f"ud{ud_version}-{scope}-{label_schema}-{artifact_version}"


def validate_artifact_id(
    artifact_id: str,
    *,
    ud_version: Optional[str] = None,
    scope: Optional[str] = None,
    label_schema: Optional[str] = None,
    artifact_version: Optional[str] = None,
) -> Dict[str, str]:
    """Validate artifact id syntax and optional component consistency."""
    parsed = parse_artifact_id(artifact_id)
    expected = {
        "ud_version": ud_version,
        "scope": scope,
        "label_schema": label_schema,
    }
    mismatches = [
        f"{field}={parsed[field]!r} does not match {value!r}"
        for field, value in expected.items()
        if value is not None and parsed[field] != str(value)
    ]
    if artifact_version is not None:
        expected_version = parse_artifact_version(str(artifact_version))["normalized"]
        if parsed["artifact_version_normalized"] != expected_version:
            mismatches.append(
                "artifact_version="
                f"{parsed['artifact_version']!r} does not match {artifact_version!r}"
            )
    if mismatches:
        raise ValueError(f"artifact_id components are inconsistent: {', '.join(mismatches)}")
    return parsed


def _dedupe(values: Iterable[Any]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for value in values:
        if value is None:
            continue
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def resolve_release_identity(config) -> Dict[str, Any]:
    """Resolve release identity from config with legacy output fallbacks."""
    release = getattr(config, "release", None)

    scope = str(getattr(release, "scope", None) or "full")
    label_schema = str(getattr(release, "label_schema", None) or "ud")
    artifact_version = str(getattr(release, "artifact_version", None) or "v1")
    artifact_id = getattr(release, "artifact_id", None) or build_artifact_id(
        ud_version=str(config.ud_version),
        scope=scope,
        label_schema=label_schema,
        artifact_version=artifact_version,
    )

    parsed = validate_artifact_id(
        artifact_id,
        ud_version=str(config.ud_version),
        scope=scope,
        label_schema=label_schema,
        artifact_version=artifact_version,
    )

    output = getattr(config, "output", None)
    hf_repo = (
        getattr(release, "hf_repo", None)
        or getattr(output, "genres_hf_repo", None)
        or ""
    )
    configured_branches = getattr(release, "hf_branches", None) or []
    configured_revisions = getattr(release, "hf_revisions", None) or []
    fallback_revision = (
        getattr(output, "genres_revision", None)
        or getattr(config, "ud_version", None)
    )
    hf_branches = _dedupe(configured_branches)
    if not hf_branches:
        legacy_branch_candidates = [
            revision
            for revision in _dedupe(configured_revisions)
            if not str(revision).startswith("artifact/")
            and str(revision) != artifact_id
        ]
        hf_branches = _dedupe(legacy_branch_candidates or [fallback_revision])
    hf_revisions = _dedupe(configured_revisions or [fallback_revision])
    hf_tag = getattr(release, "hf_tag", None) or f"artifact/{artifact_id}"
    hf_default_branch = getattr(release, "hf_default_branch", None) or "main"
    source_branch = (
        getattr(release, "source_branch", None)
        or getattr(release, "git_branch", None)
        or ""
    )
    source_tag = (
        getattr(release, "source_tag", None)
        or getattr(release, "git_tag", None)
        or f"source/{artifact_id}"
    )
    source_repo = getattr(release, "source_repo", None) or ""
    source_commit = getattr(release, "source_commit", None) or ""

    return {
        "artifact_id": artifact_id,
        "ud_version": parsed["ud_version"],
        "scope": parsed["scope"],
        "label_schema": parsed["label_schema"],
        "artifact_version": parsed["artifact_version"],
        "artifact_version_normalized": parsed["artifact_version_normalized"],
        "hf_repo": str(hf_repo) if hf_repo else "",
        "hf_branches": hf_branches,
        "hf_tag": str(hf_tag),
        "hf_default_branch": str(hf_default_branch),
        "hf_revisions": hf_revisions,
        "source_repo": str(source_repo) if source_repo else "",
        "source_branch": str(source_branch) if source_branch else "",
        "source_tag": str(source_tag),
        "source_commit": str(source_commit) if source_commit else "",
        "git_branch": str(source_branch) if source_branch else "",
        "git_tag": str(source_tag),
    }


def resolve_release_hf_repo(config) -> str:
    """Resolve the Hugging Face dataset repo for genre artifacts."""
    return str(resolve_release_identity(config).get("hf_repo") or "")


def resolve_release_hf_revisions(
    config,
    overrides: Optional[Iterable[str] | str] = None,
) -> List[str]:
    """Resolve upload target revisions, honoring CLI overrides when provided."""
    if overrides is not None:
        if isinstance(overrides, str):
            return _dedupe([overrides])
        return _dedupe(overrides)

    revisions = resolve_release_identity(config).get("hf_revisions", [])
    return _dedupe(revisions)


def resolve_release_hf_branches(
    config,
    overrides: Optional[Iterable[str] | str] = None,
) -> List[str]:
    """Resolve moving HF dataset branches for Git-backed publishing."""
    if overrides is not None:
        if isinstance(overrides, str):
            return _dedupe([overrides])
        return _dedupe(overrides)

    branches = resolve_release_identity(config).get("hf_branches", [])
    return _dedupe(branches)


def validate_release_registry_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Validate one promoted artifact registry entry."""
    artifact_id = str(entry.get("artifact_id", ""))
    parsed = validate_artifact_id(
        artifact_id,
        ud_version=entry.get("ud_version"),
        scope=entry.get("scope"),
        label_schema=entry.get("label_schema"),
        artifact_version=entry.get("artifact_version"),
    )

    required_fields = [
        "status",
        "change_scope",
        "hf_repo",
        "hf_branches",
        "hf_tag",
        "source_config",
        "source_repo",
        "source_branch",
        "source_tag",
    ]
    missing = [
        field
        for field in required_fields
        if entry.get(field) in (None, "", [])
    ]
    if missing:
        raise ValueError(
            f"release registry entry {artifact_id!r} is missing: {', '.join(missing)}"
        )

    status = str(entry["status"])
    if status not in VALID_REGISTRY_STATUSES:
        raise ValueError(
            f"release registry entry {artifact_id!r} has invalid status {status!r}"
        )

    change_scope = str(entry["change_scope"])
    if change_scope not in VALID_REGISTRY_CHANGE_SCOPES:
        raise ValueError(
            f"release registry entry {artifact_id!r} has invalid change_scope "
            f"{change_scope!r}"
        )

    expected_hf_tag = f"artifact/{artifact_id}"
    if str(entry["hf_tag"]) != expected_hf_tag:
        raise ValueError(
            f"release registry entry {artifact_id!r} hf_tag must be "
            f"{expected_hf_tag!r}"
        )

    expected_source_tag = f"source/{artifact_id}"
    if str(entry["source_tag"]) != expected_source_tag:
        raise ValueError(
            f"release registry entry {artifact_id!r} source_tag must be "
            f"{expected_source_tag!r}"
        )

    return {
        **entry,
        "artifact_version_normalized": parsed["artifact_version_normalized"],
    }


def load_release_registry(path: Path | str) -> List[Dict[str, Any]]:
    """Load and validate the promoted genre artifact registry."""
    registry_path = Path(path)
    with open(registry_path, "r", encoding="utf-8") as handle:
        registry = yaml.safe_load(handle) or {}

    artifacts = registry.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("release registry must contain an 'artifacts' list")

    return [validate_release_registry_entry(entry) for entry in artifacts]


def active_release_registry_entries(path: Path | str) -> List[Dict[str, Any]]:
    """Return active promoted artifacts selected for source-wide rebuilds."""
    return [
        entry
        for entry in load_release_registry(path)
        if entry.get("status") == "active"
    ]
