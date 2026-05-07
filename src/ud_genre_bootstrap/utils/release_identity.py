"""Release identity helpers for promoted genre artifacts."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Dict, List, Optional

ARTIFACT_ID_PATTERN = re.compile(
    r"^ud(?P<ud_version>\d+(?:\.\d+)+)-"
    r"(?P<scope>[a-z][a-z0-9_]*)-"
    r"(?P<label_schema>[a-z][a-z0-9_]*)-"
    r"(?P<artifact_version>v[1-9][0-9]*)$"
)


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

    return match.groupdict()


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
        "artifact_version": artifact_version,
    }
    mismatches = [
        f"{field}={parsed[field]!r} does not match {value!r}"
        for field, value in expected.items()
        if value is not None and parsed[field] != str(value)
    ]
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
    configured_revisions = getattr(release, "hf_revisions", None) or []
    fallback_revision = (
        getattr(output, "genres_revision", None)
        or getattr(config, "ud_version", None)
    )
    hf_revisions = _dedupe(configured_revisions or [fallback_revision])

    return {
        "artifact_id": artifact_id,
        "ud_version": parsed["ud_version"],
        "scope": parsed["scope"],
        "label_schema": parsed["label_schema"],
        "artifact_version": parsed["artifact_version"],
        "hf_repo": str(hf_repo) if hf_repo else "",
        "hf_revisions": hf_revisions,
        "git_branch": getattr(release, "git_branch", None) or "",
        "git_tag": getattr(release, "git_tag", None) or "",
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
