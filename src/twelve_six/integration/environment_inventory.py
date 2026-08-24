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
    if not isinstance(direct_url, dict):
        raise EnvironmentInventoryError("installed direct_url.json must contain an object")
    directory = direct_url.get("dir_info") or {}
    vcs_info = direct_url.get("vcs_info") or {}
    if not isinstance(directory, dict) or not isinstance(vcs_info, dict):
        raise EnvironmentInventoryError("installed direct_url provenance fields must be objects")
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
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    project = document.get("project", {})
    build_system = document.get("build-system", {})
    if not isinstance(project, dict) or not isinstance(build_system, dict):
        raise EnvironmentInventoryError("pyproject project/build-system declarations are malformed")
    runtime = project.get("dependencies", []) or []
    optional = project.get("optional-dependencies", {}) or {}
    build_requires = build_system.get("requires", []) or []
    if (
        not isinstance(runtime, list)
        or not isinstance(optional, dict)
        or not isinstance(build_requires, list)
    ):
        raise EnvironmentInventoryError("pyproject dependency declarations are malformed")
    result: dict[str, list[str]] = {
        "build-system": sorted(str(item) for item in build_requires),
        "runtime": sorted(str(item) for item in runtime),
    }
    for group, requirements in sorted(optional.items()):
        if not isinstance(requirements, list):
            raise EnvironmentInventoryError(f"optional dependency group {group!r} must be a list")
        result[f"optional:{group}"] = sorted(str(item) for item in requirements)
    return result


def _installation_evidence(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    record_hash = value.get("record_sha256")
    if record_hash is not None and _SHA256.fullmatch(str(record_hash)) is None:
        raise EnvironmentInventoryError(f"distribution {name!r} has invalid RECORD SHA-256")
    provenance = value.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise EnvironmentInventoryError(f"distribution {name!r} provenance must be an object")
    return {
        "record_sha256": record_hash,
        "provenance": dict(provenance),
    }


def _normalize_package(package: Mapping[str, Any]) -> dict[str, Any]:
    item = dict(package)
    name = canonical_distribution_name(str(item.get("name", "")))
    version = str(item.get("version", "")).strip()
    if not version:
        raise EnvironmentInventoryError(f"distribution {name!r} has an empty version")
    item["name"] = name
    item["version"] = version

    existing_installations = item.pop("installations", None)
    if existing_installations is None:
        installation = _installation_evidence(
            {
                "record_sha256": item.pop("record_sha256", None),
                "provenance": item.pop("provenance", {}),
            },
            name,
        )
        installations = [installation]
    else:
        if "record_sha256" in item or "provenance" in item:
            raise EnvironmentInventoryError(
                f"distribution {name!r} mixes normalized and raw installation evidence"
            )
        if not isinstance(existing_installations, list) or not existing_installations:
            raise EnvironmentInventoryError(f"distribution {name!r} installations must be non-empty")
        installations = []
        for evidence in existing_installations:
            if not isinstance(evidence, Mapping):
                raise EnvironmentInventoryError(
                    f"distribution {name!r} installation evidence must be an object"
                )
            installations.append(_installation_evidence(evidence, name))

    unique_installations = {
        canonical_json_bytes(evidence): evidence for evidence in installations
    }
    item["installations"] = sorted(
        unique_installations.values(), key=lambda evidence: canonical_json_bytes(evidence)
    )
    return item


def _package_core(package: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in package.items() if key != "installations"}


def _validate_packages(packages: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for package in packages:
        item = _normalize_package(package)
        name = item["name"]
        previous = by_name.get(name)
        if previous is not None:
            if canonical_json_bytes(_package_core(previous)) != canonical_json_bytes(
                _package_core(item)
            ):
                raise EnvironmentInventoryError(
                    f"ambiguous installed distribution {name!r}: conflicting package metadata"
                )
            combined = [*previous["installations"], *item["installations"]]
            unique = {canonical_json_bytes(evidence): evidence for evidence in combined}
            previous["installations"] = sorted(
                unique.values(), key=lambda evidence: canonical_json_bytes(evidence)
            )
            continue
        by_name[name] = item
    return sorted(by_name.values(), key=lambda item: (item["name"], item["version"]))


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
