"""Lock-bound dependency SBOM and security-evidence contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from twelve_six.integration.dependency_lock import (
    PROJECT_DISTRIBUTION,
    SUPPORTED_PROFILES,
    canonical_distribution_name,
    canonical_json_bytes,
    sha256_file,
    validate_lock_index,
    validate_profile_manifest,
)

SBOM_SCHEMA_VERSION = "12-6.dependency-sbom.v1"
EVIDENCE_SCHEMA_VERSION = "12-6.dependency-security-evidence.v1"
DEFAULT_REPOSITORY = "Oleksii-debug/12-6-ai."
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_LOCK_LINE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s]+) "
    r"--hash=sha256:(?P<hash>[0-9a-f]{64})$"
)


class DependencySecurityError(ValueError):
    """Raised when dependency-security evidence is stale, incomplete, or tampered."""


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_source_sha(source_sha: str) -> str:
    if _GIT_SHA.fullmatch(source_sha) is None:
        raise DependencySecurityError("source SHA must be a full lowercase Git object id")
    return source_sha


def component_key(name: str, version: str) -> str:
    return f"{canonical_distribution_name(name)}=={version}"


def _parse_lock(path: Path, group: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _LOCK_LINE.fullmatch(line)
        if match is None:
            raise DependencySecurityError(f"invalid exact lock line in {path}: {line!r}")
        name = canonical_distribution_name(match.group("name"))
        version = match.group("version")
        key = component_key(name, version)
        if name in seen:
            raise DependencySecurityError(f"duplicate locked distribution in {group}: {name}")
        seen.add(name)
        records.append(
            {
                "name": name,
                "version": version,
                "group": group,
                "artifact_sha256": match.group("hash"),
                "key": key,
            }
        )
    return records


def _profile_components(*, root: Path, profile_id: str, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    versions: dict[str, str] = {}
    locks = manifest["locks"]
    for group in sorted(locks):
        lock_record = locks[group]
        records = _parse_lock(root / lock_record["path"], group)
        if len(records) != lock_record["package_count"]:
            raise DependencySecurityError(f"{profile_id}/{group} package count mismatch")
        for record in records:
            name = record["name"]
            version = record["version"]
            prior_version = versions.get(name)
            if prior_version is not None and prior_version != version:
                raise DependencySecurityError(
                    f"conflicting locked versions for {name}: {prior_version} vs {version}"
                )
            versions[name] = version
            key = record["key"]
            current = merged.setdefault(
                key,
                {
                    "name": name,
                    "version": version,
                    "purl": f"pkg:pypi/{name}@{version}",
                    "groups": [],
                    "artifact_sha256": [],
                },
            )
            current["groups"].append(group)
            current["artifact_sha256"].append(record["artifact_sha256"])
    result: list[dict[str, Any]] = []
    for key in sorted(merged):
        item = merged[key]
        item["groups"] = sorted(set(item["groups"]))
        item["artifact_sha256"] = sorted(set(item["artifact_sha256"]))
        result.append(item)
    return result


def build_lock_sbom(
    *,
    root: str | Path,
    source_sha: str,
    repository_full_name: str = DEFAULT_REPOSITORY,
    index_path: str | Path = "requirements/locks/index.json",
) -> dict[str, Any]:
    """Build a deterministic, source-bound SBOM from the validated committed locks."""
    source_sha = _require_source_sha(source_sha)
    root_path = Path(root)
    index = validate_lock_index(root=root_path, index_path=index_path)
    profiles: dict[str, Any] = {}
    for profile_id in sorted(SUPPORTED_PROFILES):
        index_record = index["profiles"][profile_id]
        manifest = validate_profile_manifest(
            root=root_path,
            manifest_path=index_record["path"],
            enforce_current_platform=False,
        )
        components = _profile_components(root=root_path, profile_id=profile_id, manifest=manifest)
        profiles[profile_id] = {
            "profile_manifest_sha256": manifest["manifest_sha256"],
            "component_count": len(components),
            "components": components,
        }
    payload: dict[str, Any] = {
        "schema_version": SBOM_SCHEMA_VERSION,
        "project": PROJECT_DISTRIBUTION,
        "repository_full_name": repository_full_name,
        "source_sha": source_sha,
        "lock_index": {
            "semantic_sha256": index["index_sha256"],
            "file_sha256": sha256_file(root_path / index_path),
        },
        "profiles": profiles,
    }
    payload["sbom_sha256"] = _sha256_json(payload)
    return payload


def unique_components(sbom: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the cross-profile component union with explicit profile membership."""
    merged: dict[str, dict[str, Any]] = {}
    for profile_id, profile in sorted(sbom["profiles"].items()):
        for component in profile["components"]:
            key = component_key(component["name"], component["version"])
            current = merged.setdefault(
                key,
                {
                    "key": key,
                    "name": component["name"],
                    "version": component["version"],
                    "purl": component["purl"],
                    "profiles": [],
                },
            )
            current["profiles"].append(profile_id)
    for current in merged.values():
        current["profiles"] = sorted(set(current["profiles"]))
    return [merged[key] for key in sorted(merged)]


def _validate_scan_component_record(record: Mapping[str, Any]) -> bool:
    license_record = record.get("license")
    advisory_record = record.get("advisories")
    if not isinstance(license_record, Mapping) or not isinstance(advisory_record, Mapping):
        raise DependencySecurityError("component scan record is incomplete")
    license_status = license_record.get("status")
    if license_status not in {"DECLARED", "UNRESOLVED"}:
        raise DependencySecurityError("invalid license evidence status")
    if not _SHA256.fullmatch(str(license_record.get("metadata_sha256", ""))):
        raise DependencySecurityError("license metadata digest is invalid")
    if advisory_record.get("status") != "QUERIED":
        raise DependencySecurityError("advisory evidence was not queried")
    if not _SHA256.fullmatch(str(advisory_record.get("response_sha256", ""))):
        raise DependencySecurityError("advisory response digest is invalid")
    vulnerabilities = advisory_record.get("vulnerabilities")
    if not isinstance(vulnerabilities, list):
        raise DependencySecurityError("advisory vulnerabilities must be a list")
    seen: set[str] = set()
    for vulnerability in vulnerabilities:
        if not isinstance(vulnerability, Mapping):
            raise DependencySecurityError("vulnerability entry must be an object")
        vulnerability_id = vulnerability.get("id")
        if not isinstance(vulnerability_id, str) or not vulnerability_id.strip():
            raise DependencySecurityError("vulnerability id is missing")
        if vulnerability_id in seen:
            raise DependencySecurityError("duplicate vulnerability id")
        seen.add(vulnerability_id)
    return license_status == "UNRESOLVED" or bool(vulnerabilities)


def build_security_evidence(
    *,
    sbom: Mapping[str, Any],
    generated_at: str,
    license_records: Mapping[str, Mapping[str, Any]],
    advisory_records: Mapping[str, Mapping[str, Any]],
    scan_sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind online metadata/advisory observations to one exact source+lock SBOM."""
    components = unique_components(sbom)
    expected_keys = {component["key"] for component in components}
    if set(license_records) != expected_keys or set(advisory_records) != expected_keys:
        raise DependencySecurityError("scan component set does not match lock-derived SBOM")
    if set(scan_sources) != {"osv", "pypi"}:
        raise DependencySecurityError("scan source set must be exactly osv+pypi")
    for source in scan_sources.values():
        if source.get("status") != "SUCCESS":
            raise DependencySecurityError("dependency evidence source is incomplete")

    observations: list[dict[str, Any]] = []
    review_required = False
    for component in components:
        key = component["key"]
        record = {
            **component,
            "license": dict(license_records[key]),
            "advisories": dict(advisory_records[key]),
        }
        review_required = _validate_scan_component_record(record) or review_required
        observations.append(record)

    payload: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "repository_full_name": sbom["repository_full_name"],
        "source_sha": sbom["source_sha"],
        "generated_at": generated_at,
        "lock_index": dict(sbom["lock_index"]),
        "sbom_sha256": sbom["sbom_sha256"],
        "scan_sources": {key: dict(value) for key, value in sorted(scan_sources.items())},
        "status": (
            "EVIDENCE_COMPLETE_REVIEW_REQUIRED"
            if review_required
            else "EVIDENCE_COMPLETE_NO_REVIEW_FINDINGS"
        ),
        "components": observations,
        "truth_boundary": {
            "audit_verdict": False,
            "license_approval": False,
            "vulnerability_risk_acceptance": False,
            "promotion_authority": False,
        },
    }
    payload["evidence_sha256"] = _sha256_json(payload)
    return payload


def validate_security_evidence(
    *,
    root: str | Path,
    evidence: Mapping[str, Any],
    expected_source_sha: str,
    max_age_hours: float | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Offline validation against current lock bytes, source SHA, freshness and self-hash."""
    expected_source_sha = _require_source_sha(expected_source_sha)
    document = dict(evidence)
    claimed = document.pop("evidence_sha256", None)
    if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise DependencySecurityError("unsupported dependency security evidence schema")
    if _SHA256.fullmatch(str(claimed or "")) is None or _sha256_json(document) != claimed:
        raise DependencySecurityError("dependency security evidence self-hash mismatch")
    if evidence.get("source_sha") != expected_source_sha:
        raise DependencySecurityError("dependency security evidence source SHA mismatch")

    sbom = build_lock_sbom(root=root, source_sha=expected_source_sha)
    if evidence.get("repository_full_name") != sbom["repository_full_name"]:
        raise DependencySecurityError("dependency security repository identity mismatch")
    if evidence.get("lock_index") != sbom["lock_index"]:
        raise DependencySecurityError("dependency security lock identity mismatch")
    if evidence.get("sbom_sha256") != sbom["sbom_sha256"]:
        raise DependencySecurityError("dependency security SBOM identity mismatch")

    generated_raw = evidence.get("generated_at")
    if not isinstance(generated_raw, str):
        raise DependencySecurityError("dependency security generated_at is missing")
    try:
        generated = datetime.fromisoformat(generated_raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DependencySecurityError("dependency security generated_at is invalid") from exc
    if generated.tzinfo is None:
        raise DependencySecurityError("dependency security generated_at must be timezone-aware")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise DependencySecurityError("validation current time must be timezone-aware")
    generated = generated.astimezone(timezone.utc)
    current = current.astimezone(timezone.utc)
    if generated > current + timedelta(minutes=5):
        raise DependencySecurityError("dependency security evidence timestamp is in the future")
    if max_age_hours is not None:
        if max_age_hours <= 0:
            raise DependencySecurityError("max_age_hours must be positive")
        if current - generated > timedelta(hours=max_age_hours):
            raise DependencySecurityError("dependency security evidence is stale")

    sources = evidence.get("scan_sources")
    if not isinstance(sources, Mapping) or set(sources) != {"osv", "pypi"}:
        raise DependencySecurityError("dependency security scan sources are malformed")
    for source in sources.values():
        if not isinstance(source, Mapping) or source.get("status") != "SUCCESS":
            raise DependencySecurityError("dependency security scan source is incomplete")

    expected = {item["key"]: item for item in unique_components(sbom)}
    records = evidence.get("components")
    if not isinstance(records, list):
        raise DependencySecurityError("dependency security components must be a list")
    actual: dict[str, Mapping[str, Any]] = {}
    review_required = False
    for record in records:
        if not isinstance(record, Mapping):
            raise DependencySecurityError("dependency security component must be an object")
        key = record.get("key")
        if not isinstance(key, str) or key in actual:
            raise DependencySecurityError("duplicate or invalid dependency security component key")
        actual[key] = record
        expected_record = expected.get(key)
        if expected_record is None:
            raise DependencySecurityError("dependency security component is not in current locks")
        for field in ("name", "version", "purl", "profiles"):
            if record.get(field) != expected_record[field]:
                raise DependencySecurityError(f"dependency security component {field} mismatch")
        review_required = _validate_scan_component_record(record) or review_required
    if set(actual) != set(expected):
        raise DependencySecurityError("dependency security component set mismatch")
    expected_status = (
        "EVIDENCE_COMPLETE_REVIEW_REQUIRED"
        if review_required
        else "EVIDENCE_COMPLETE_NO_REVIEW_FINDINGS"
    )
    if evidence.get("status") != expected_status:
        raise DependencySecurityError("dependency security review status mismatch")
    if evidence.get("truth_boundary") != {
        "audit_verdict": False,
        "license_approval": False,
        "vulnerability_risk_acceptance": False,
        "promotion_authority": False,
    }:
        raise DependencySecurityError("dependency security truth boundary mismatch")
    return dict(evidence)


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
