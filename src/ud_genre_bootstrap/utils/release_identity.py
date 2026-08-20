"""Release identity helpers for promoted genre release trains."""

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
TRAIN_ID_PATTERN = re.compile(
    r"^(?P<scope>[a-z][a-z0-9_]*)-"
    r"(?P<label_schema>[a-z][a-z0-9_]*)-"
    r"(?P<artifact_version>v[1-9][0-9]*(?:\.(?:0|[1-9][0-9]*)){0,2})$"
)
ARTIFACT_KEY_PATTERN = re.compile(
    r"^(?P<train_id>"
    r"(?P<scope>[a-z][a-z0-9_]*)-"
    r"(?P<label_schema>[a-z][a-z0-9_]*)-"
    r"(?P<artifact_version>v[1-9][0-9]*(?:\.(?:0|[1-9][0-9]*)){0,2})"
    r")-ud(?P<ud_version>\d+(?:\.\d+)+)$"
)
LEGACY_ARTIFACT_ID_PATTERN = re.compile(
    r"^ud(?P<ud_version>\d+(?:\.\d+)+)-"
    r"(?P<scope>[a-z][a-z0-9_]*)-"
    r"(?P<label_schema>[a-z][a-z0-9_]*)-"
    r"(?P<artifact_version>v[1-9][0-9]*(?:\.(?:0|[1-9][0-9]*)){0,2})$"
)

VALID_REGISTRY_STATUSES = {"partial", "default_hotfix", "complete", "deprecated"}


def parse_artifact_version(version: str) -> Dict[str, Any]:
    """Parse artifact version strings such as ``v1.0.0``, ``v1.2``, or ``v1.2.1``."""
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


def parse_train_id(train_id: str) -> Dict[str, Any]:
    """Parse a synchronized release train id such as ``full-ud-v1.0.1``."""
    if not isinstance(train_id, str) or not train_id.strip():
        raise ValueError("train_id must be a non-empty string")

    match = TRAIN_ID_PATTERN.fullmatch(train_id.strip())
    if not match:
        raise ValueError(
            "train_id must match '<scope>-<label_schema>-<artifact_version>', "
            "for example 'full-ud-v1.0.1'"
        )

    parsed = match.groupdict()
    parsed["artifact_version_normalized"] = parse_artifact_version(
        parsed["artifact_version"]
    )["normalized"]
    return parsed


def build_train_id(*, scope: str, label_schema: str, artifact_version: str) -> str:
    """Build a synchronized release train id."""
    return f"{scope}-{label_schema}-{artifact_version}"


def validate_train_id(
    train_id: str,
    *,
    scope: Optional[str] = None,
    label_schema: Optional[str] = None,
    artifact_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate train id syntax and optional component consistency."""
    parsed = parse_train_id(train_id)
    mismatches = []
    if scope is not None and parsed["scope"] != str(scope):
        mismatches.append(f"scope={parsed['scope']!r} does not match {scope!r}")
    if label_schema is not None and parsed["label_schema"] != str(label_schema):
        mismatches.append(
            f"label_schema={parsed['label_schema']!r} does not match {label_schema!r}"
        )
    if artifact_version is not None:
        expected_version = parse_artifact_version(str(artifact_version))["normalized"]
        if parsed["artifact_version_normalized"] != expected_version:
            mismatches.append(
                "artifact_version="
                f"{parsed['artifact_version']!r} does not match {artifact_version!r}"
            )
    if mismatches:
        raise ValueError(f"train_id components are inconsistent: {', '.join(mismatches)}")
    return parsed


def parse_artifact_key(artifact_key: str) -> Dict[str, Any]:
    """Parse a per-UD artifact key in a synchronized train.

    Expected shape: ``<train_id>-ud<UD version>``.
    Example: ``full-ud-v1.0.1-ud2.18``.
    """
    if not isinstance(artifact_key, str) or not artifact_key.strip():
        raise ValueError("artifact_key must be a non-empty string")

    match = ARTIFACT_KEY_PATTERN.fullmatch(artifact_key.strip())
    if not match:
        raise ValueError(
            "artifact_key must match '<train_id>-ud<UD version>', "
            "for example 'full-ud-v1.0.1-ud2.18'"
        )

    parsed = match.groupdict()
    parsed["artifact_version_normalized"] = parse_artifact_version(
        parsed["artifact_version"]
    )["normalized"]
    return parsed


def build_artifact_key(*, train_id: str, ud_version: str) -> str:
    """Build a per-UD artifact key from a release train id."""
    validate_train_id(train_id)
    return f"{train_id}-ud{ud_version}"


def validate_artifact_key(
    artifact_key: str,
    *,
    ud_version: Optional[str] = None,
    scope: Optional[str] = None,
    label_schema: Optional[str] = None,
    artifact_version: Optional[str] = None,
    train_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate artifact key syntax and optional component consistency."""
    parsed = parse_artifact_key(artifact_key)
    expected = {
        "ud_version": ud_version,
        "scope": scope,
        "label_schema": label_schema,
        "train_id": train_id,
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
        raise ValueError(
            f"artifact_key components are inconsistent: {', '.join(mismatches)}"
        )
    return parsed


def parse_artifact_id(artifact_id: str) -> Dict[str, Any]:
    """Parse a public artifact identifier.

    New release-train artifacts use ``<train_id>-ud<UD version>``. Legacy
    ``ud<UD version>-<scope>-<label_schema>-<artifact_version>`` ids are still
    accepted for compatibility with older config snapshots.
    """
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise ValueError("artifact_id must be a non-empty string")

    try:
        parsed = parse_artifact_key(artifact_id)
        parsed["legacy"] = False
        return parsed
    except ValueError:
        pass

    match = LEGACY_ARTIFACT_ID_PATTERN.fullmatch(artifact_id.strip())
    if not match:
        raise ValueError(
            "artifact_id must match '<train_id>-ud<UD version>' "
            "or legacy 'ud<UD version>-<scope>-<label_schema>-<artifact_version>'"
        )

    parsed = match.groupdict()
    parsed["train_id"] = build_train_id(
        scope=parsed["scope"],
        label_schema=parsed["label_schema"],
        artifact_version=parsed["artifact_version"],
    )
    parsed["artifact_version_normalized"] = parse_artifact_version(
        parsed["artifact_version"]
    )["normalized"]
    parsed["legacy"] = True
    return parsed


def build_artifact_id(
    *,
    ud_version: str,
    scope: str,
    label_schema: str,
    artifact_version: str,
) -> str:
    """Build a compatibility artifact id from explicit identity components."""
    train_id = build_train_id(
        scope=scope,
        label_schema=label_schema,
        artifact_version=artifact_version,
    )
    return build_artifact_key(train_id=train_id, ud_version=ud_version)


def validate_artifact_id(
    artifact_id: str,
    *,
    ud_version: Optional[str] = None,
    scope: Optional[str] = None,
    label_schema: Optional[str] = None,
    artifact_version: Optional[str] = None,
    train_id: Optional[str] = None,
) -> Dict[str, str]:
    """Validate artifact id syntax and optional component consistency."""
    parsed = parse_artifact_id(artifact_id)
    expected = {
        "ud_version": ud_version,
        "scope": scope,
        "label_schema": label_schema,
        "train_id": train_id,
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


def build_source_branch(train_id: str) -> str:
    """Build the shared source release branch for a train major version."""
    parsed = parse_train_id(train_id)
    major = parse_artifact_version(parsed["artifact_version"])["major"]
    return f"release/{parsed['scope']}-{parsed['label_schema']}-v{major}"


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
    train_id = getattr(release, "train_id", None) or build_train_id(
        scope=scope,
        label_schema=label_schema,
        artifact_version=artifact_version,
    )
    train_parsed = validate_train_id(
        train_id,
        scope=scope,
        label_schema=label_schema,
        artifact_version=artifact_version,
    )
    artifact_key = (
        getattr(release, "artifact_key", None)
        or getattr(release, "artifact_id", None)
        or build_artifact_key(train_id=train_id, ud_version=str(config.ud_version))
    )

    parsed = validate_artifact_id(
        artifact_key,
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
            and str(revision) != artifact_key
        ]
        hf_branches = _dedupe(legacy_branch_candidates or [fallback_revision])
    hf_revisions = _dedupe(configured_revisions or [fallback_revision])
    hf_tag = (
        getattr(release, "hf_tag", None)
        or f"artifact/{train_id}/ud{config.ud_version}"
    )
    hf_default_branch = getattr(release, "hf_default_branch", None) or "main"
    source_branch = (
        getattr(release, "source_branch", None)
        or getattr(release, "git_branch", None)
        or build_source_branch(train_id)
    )
    source_tag = (
        getattr(release, "source_tag", None)
        or getattr(release, "git_tag", None)
        or f"source/{train_id}"
    )
    source_repo = getattr(release, "source_repo", None) or ""
    source_commit = getattr(release, "source_commit", None) or ""
    inventory_status = getattr(release, "inventory_status", None) or ""

    return {
        "train_id": train_id,
        "artifact_key": artifact_key,
        "artifact_id": artifact_key,
        "inventory_status": str(inventory_status) if inventory_status else "",
        "ud_version": parsed["ud_version"],
        "scope": train_parsed["scope"],
        "label_schema": train_parsed["label_schema"],
        "artifact_version": train_parsed["artifact_version"],
        "artifact_version_normalized": train_parsed["artifact_version_normalized"],
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
    """Validate one promoted release train registry entry."""
    train_id = str(entry.get("train_id", ""))
    parsed = validate_train_id(
        train_id,
        scope=entry.get("scope"),
        label_schema=entry.get("label_schema"),
        artifact_version=entry.get("artifact_version"),
    )

    required_fields = [
        "status",
        "hf_repo",
        "supported_ud_versions",
        "default_ud_version",
        "release_matrix",
        "release_profile",
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
            f"release registry entry {train_id!r} is missing: {', '.join(missing)}"
        )

    status = str(entry["status"])
    if status not in VALID_REGISTRY_STATUSES:
        raise ValueError(
            f"release registry entry {train_id!r} has invalid status {status!r}"
        )

    if entry["default_ud_version"] not in entry["supported_ud_versions"]:
        raise ValueError(
            f"release registry entry {train_id!r} default_ud_version must be "
            "one of supported_ud_versions"
        )

    expected_source_tag = f"source/{train_id}"
    if str(entry["source_tag"]) != expected_source_tag:
        raise ValueError(
            f"release registry entry {train_id!r} source_tag must be "
            f"{expected_source_tag!r}"
        )

    return {
        **entry,
        "artifact_version_normalized": parsed["artifact_version_normalized"],
    }


def load_release_registry(path: Path | str) -> List[Dict[str, Any]]:
    """Load and validate the promoted genre release-train registry."""
    registry_path = Path(path)
    with open(registry_path, "r", encoding="utf-8") as handle:
        registry = yaml.safe_load(handle) or {}

    artifacts = registry.get("trains", registry.get("artifacts", []))
    if not isinstance(artifacts, list):
        raise ValueError("release registry must contain a 'trains' list")

    return [validate_release_registry_entry(entry) for entry in artifacts]


def active_release_registry_entries(path: Path | str) -> List[Dict[str, Any]]:
    """Return release trains selected for source-wide rebuilds."""
    return [
        entry
        for entry in load_release_registry(path)
        if entry.get("status") in {"partial", "default_hotfix", "complete"}
    ]
