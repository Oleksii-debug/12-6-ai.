"""Deterministic installed-environment inventory for CI and release evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "12-6.environment-inventory.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_NAME_NORMALIZER = re.compile(r"[-_.]+")


class EnvironmentInventoryError(ValueError):
    """Raised when environment evidence is malformed or ambiguous."""


def canonical_distribution_name(name: str) -> str:
    normalized = _NAME_NORMALIZER.sub("-", name.strip()).lower()
    if not normalized:
        raise EnvironmentInventoryError("distribution name must not be empty")
    return normalized


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _selected_license_metadata(metadata: importlib.metadata.PackageMetadata) -> dict[str, Any]:
    classifiers = sorted(
        value.strip()
        for value in (metadata.get_all("Classifier") or [])
        if value.strip().startswith("License ::")
    )
    expression = (metadata.get("License-Expression") or "").strip() or None
    license_value = (metadata.get("License") or "").strip() or None
    return {
        "license_expression": expression,
        "license": license_value,
        "license_classifiers": classifiers,
    }


def _selected_provenance(distribution: importlib.metadata.Distribution) -> dict[str, Any]:
    raw = distribution.read_text("direct_url.json")
    if not raw:
        return {"editable": False, "vcs": None, "vcs_commit_id": None}
    try:
        direct_url = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EnvironmentInventoryError("invalid installed direct_url.json") from exc
    directory = direct_url.get("dir_info") or {}
    vcs_info = direct_url.get("vcs_info") or {}
    vcs = vcs_info.get("vcs")
    commit_id = vcs_info.get("commit_id")
    if commit_id is not None and not isinstance(commit_id, str):
        raise EnvironmentInventoryError("direct_url vcs commit_id must be a string")
    return {
        "editable": bool(directory.get("editable", False)),
        "vcs": vcs if isinstance(vcs, str) and vcs else None,
        "vcs_commit_id": commit_id or None,
    }


def installed_distribution_record(distribution: importlib.metadata.Distribution) -> dict[str, Any]:
    metadata = distribution.metadata
    raw_name = (metadata.get("Name") or "").strip()
    version = (distribution.version or "").strip()
    if not raw_name or not version:
        raise EnvironmentInventoryError("installed distribution is missing name/version metadata")
    record = distribution.read_text("RECORD")
    return {
        "name": canonical_distribution_name(raw_name),
        "version": version,
        **_selected_license_metadata(metadata),
        "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest() if record else None,
        "provenance": _selected_provenance(distribution),
    }


def declared_requirements(pyproject_path: str | Path) -> dict[str, list[str]]:
    path = Path(pyproject_path)
    project = tomllib.loads(path.read_text(encoding="utf-8")).get("project", {})
    runtime = project.get("dependencies", []) or []
    optional = project.get("optional-dependencies", {}) or {}
    if not isinstance(runtime, list) or not isinstance(optional, dict):
        raise EnvironmentInventoryError("pyproject dependency declarations are malformed")
    result: dict[str, list[str]] = {"runtime": sorted(str(item) for item in runtime)}
    for group, requirements in sorted(optional.items()):
        if not isinstance(requirements, list):
            raise EnvironmentInventoryError(f"optional dependency group {group!r} must be a list")
        result[f"optional:{group}"] = sorted(str(item) for item in requirements)
    return result


def _validate_packages(packages: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for package in packages:
        item = dict(package)
        name = canonical_distribution_name(str(item.get("name", "")))
        version = str(item.get("version", "")).strip()
        if not version:
            raise EnvironmentInventoryError(f"distribution {name!r} has an empty version")
        if name in seen:
            raise EnvironmentInventoryError(
                f"ambiguous installed distribution {name!r}: {seen[name]!r} and {version!r}"
            )
        seen[name] = version
        item["name"] = name
        item["version"] = version
        record_hash = item.get("record_sha256")
        if record_hash is not None and _SHA256.fullmatch(str(record_hash)) is None:
            raise EnvironmentInventoryError(f"distribution {name!r} has invalid RECORD SHA-256")
        normalized.append(item)
    return sorted(normalized, key=lambda item: (item["name"], item["version"]))


def build_environment_inventory(
    *,
    repository: str,
    source_sha: str,
    python_version: str,
    python_implementation: str,
    requirements: Mapping[str, Iterable[str]],
    packages: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build and hash a deterministic environment inventory."""

    if not repository or "/" not in repository:
        raise EnvironmentInventoryError("repository must be an owner/name identity")
    if _GIT_OBJECT.fullmatch(source_sha) is None:
        raise EnvironmentInventoryError("source_sha must be a full lowercase Git object id")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", python_version):
        raise EnvironmentInventoryError("python_version must be exact X.Y.Z")
    implementation = python_implementation.strip().lower()
    if not implementation:
        raise EnvironmentInventoryError("python implementation must not be empty")

    package_list = _validate_packages(packages)
    requirement_map = {
        str(group): sorted(str(requirement) for requirement in values)
        for group, values in sorted(requirements.items())
    }
    unresolved_license_count = sum(
        1
        for package in package_list
        if not package.get("license_expression")
        and not package.get("license")
        and not package.get("license_classifiers")
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "repository": repository,
        "source_sha": source_sha,
        "python": {"implementation": implementation, "version": python_version},
        "declared_requirements": requirement_map,
        "packages": package_list,
        "summary": {
            "package_count": len(package_list),
            "unresolved_license_metadata_count": unresolved_license_count,
        },
    }
    payload["inventory_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def validate_environment_inventory(inventory: Mapping[str, Any]) -> None:
    """Verify schema, identities, counts and the self-hash before downstream binding."""

    payload = dict(inventory)
    claimed_hash = payload.pop("inventory_sha256", None)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise EnvironmentInventoryError("unsupported environment inventory schema")
    if _SHA256.fullmatch(str(claimed_hash or "")) is None:
        raise EnvironmentInventoryError("inventory_sha256 must be a lowercase SHA-256")
    expected_hash = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if claimed_hash != expected_hash:
        raise EnvironmentInventoryError("environment inventory hash mismatch")

    packages = _validate_packages(payload.get("packages", []))
    summary = payload.get("summary")
    if not isinstance(summary, dict) or summary.get("package_count") != len(packages):
        raise EnvironmentInventoryError("environment package count mismatch")
    source_sha = str(payload.get("source_sha", ""))
    if _GIT_OBJECT.fullmatch(source_sha) is None:
        raise EnvironmentInventoryError("environment source_sha is not a full Git object id")


def capture_current_environment(
    *, repository: str, source_sha: str, pyproject_path: str | Path = "pyproject.toml"
) -> dict[str, Any]:
    package_records = [
        installed_distribution_record(distribution)
        for distribution in importlib.metadata.distributions()
    ]
    return build_environment_inventory(
        repository=repository,
        source_sha=source_sha,
        python_version=platform.python_version(),
        python_implementation=sys.implementation.name,
        requirements=declared_requirements(pyproject_path),
        packages=package_records,
    )
