"""Deterministic SBOM and bound dependency adjudication evidence seams."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .dependency_lock import (
    canonical_distribution_name,
    canonical_json_bytes,
    sha256_file,
    validate_lock_index,
    validate_profile_manifest,
)

EVIDENCE_SCHEMA_VERSION = "12-6.supply-chain-evidence.v1"
ADJUDICATION_SCHEMA_VERSION = "12-6.dependency-adjudication.v1"
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_LOCK_LINE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;@/\\]+)"
    r"(?P<hashes>(?: --hash=sha256:[0-9a-f]{64})+)$"
)
_HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})")
_STATUSES = frozenset({"PASS", "FAIL", "UNKNOWN"})
_KINDS = frozenset({"vulnerability", "license"})


class SupplyChainEvidenceError(ValueError):
    """Raised when supply-chain evidence is incomplete, stale, or tampered."""


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupplyChainEvidenceError(f"cannot read JSON evidence {path}") from exc
    if not isinstance(value, dict):
        raise SupplyChainEvidenceError(f"JSON evidence {path} must contain an object")
    return value


def _validate_source_sha(source_sha: str) -> None:
    if source_sha != "UNBOUND_LOCAL" and _GIT_SHA.fullmatch(source_sha) is None:
        raise SupplyChainEvidenceError("source SHA must be full lowercase Git SHA or UNBOUND_LOCAL")


def _read_locked_components(root: Path, profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    components: dict[str, dict[str, Any]] = {}
    locks = profile.get("locks")
    if not isinstance(locks, dict):
        raise SupplyChainEvidenceError("profile locks are malformed")

    for group, record in sorted(locks.items()):
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise SupplyChainEvidenceError(f"lock record {group!r} is malformed")
        lock_path = root / record["path"]
        for line_number, raw in enumerate(lock_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            match = _LOCK_LINE.fullmatch(line)
            if match is None:
                raise SupplyChainEvidenceError(
                    f"unparseable exact lock line {record['path']}:{line_number}"
                )
            name = canonical_distribution_name(match.group("name"))
            version = match.group("version")
            hashes = tuple(sorted(set(_HASH.findall(match.group("hashes")))))
            if not hashes:
                raise SupplyChainEvidenceError(f"locked component {name} has no SHA-256")
            current = components.get(name)
            if current is None:
                components[name] = {
                    "name": name,
                    "version": version,
                    "hashes": hashes,
                    "groups": {str(group)},
                }
                continue
            if current["version"] != version:
                raise SupplyChainEvidenceError(
                    f"locked component version conflict for {name}: "
                    f"{current['version']} vs {version}"
                )
            current["hashes"] = tuple(sorted({*current["hashes"], *hashes}))
            current["groups"].add(str(group))

    normalized: list[dict[str, Any]] = []
    for name, record in sorted(components.items()):
        normalized.append(
            {
                "name": name,
                "version": record["version"],
                "hashes": list(record["hashes"]),
                "groups": sorted(record["groups"]),
            }
        )
    return normalized


def _default_adjudication(kind: str) -> dict[str, Any]:
    return {
        "status": "UNKNOWN",
        "reason": f"NO_{kind.upper()}_ADJUDICATION_BOUND",
        "tool": None,
        "evidence_ref": None,
    }


def load_bound_adjudication(
    path: str | Path,
    *,
    kind: str,
    source_sha: str,
    profile_id: str,
    lock_index_sha256: str,
) -> dict[str, Any]:
    if kind not in _KINDS:
        raise SupplyChainEvidenceError(f"unsupported adjudication kind: {kind}")
    document = _load_json(Path(path))
    expected = {
        "schema_version": ADJUDICATION_SCHEMA_VERSION,
        "kind": kind,
        "source_sha": source_sha,
        "profile_id": profile_id,
        "lock_index_sha256": lock_index_sha256,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise SupplyChainEvidenceError(f"{kind} adjudication binding mismatch for {key}")
    status = document.get("status")
    if status not in _STATUSES:
        raise SupplyChainEvidenceError(f"{kind} adjudication has invalid status")
    tool = document.get("tool")
    evidence_ref = document.get("evidence_ref")
    if status in {"PASS", "FAIL"}:
        if not isinstance(tool, dict) or not tool.get("name") or not tool.get("version"):
            raise SupplyChainEvidenceError(f"resolved {kind} adjudication requires tool identity")
        if not isinstance(evidence_ref, str) or not evidence_ref.strip():
            raise SupplyChainEvidenceError(f"resolved {kind} adjudication requires evidence_ref")
    return {
        "status": status,
        "reason": document.get("reason"),
        "tool": tool,
        "evidence_ref": evidence_ref,
        "adjudication_sha256": sha256_file(path),
    }


def build_supply_chain_documents(
    *,
    root: str | Path,
    profile_id: str,
    source_sha: str,
    vulnerability_adjudication: str | Path | None = None,
    license_adjudication: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root_path = Path(root).resolve()
    _validate_source_sha(source_sha)
    index_path = root_path / "requirements/locks/index.json"
    index = validate_lock_index(root=root_path, index_path="requirements/locks/index.json")
    profile_record = index["profiles"].get(profile_id)
    if not isinstance(profile_record, dict) or not isinstance(profile_record.get("path"), str):
        raise SupplyChainEvidenceError(f"profile {profile_id!r} is not bound by lock index")
    profile = validate_profile_manifest(
        root=root_path,
        manifest_path=profile_record["path"],
        enforce_current_platform=False,
    )
    components = _read_locked_components(root_path, profile)

    sbom_components: list[dict[str, Any]] = []
    for component in components:
        name = component["name"]
        version = component["version"]
        sbom_components.append(
            {
                "type": "library",
                "bom-ref": f"pkg:pypi/{quote(name)}@{quote(version)}",
                "name": name,
                "version": version,
                "purl": f"pkg:pypi/{quote(name)}@{quote(version)}",
                "hashes": [
                    {"alg": "SHA-256", "content": digest} for digest in component["hashes"]
                ],
                "properties": [
                    {"name": "12-6:lock-group", "value": group}
                    for group in component["groups"]
                ],
            }
        )

    sbom: dict[str, Any] = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": "twelve-six-ai",
                "version": "0.2.0.dev0",
            },
            "properties": [
                {"name": "12-6:source-sha", "value": source_sha},
                {"name": "12-6:lock-profile", "value": profile_id},
                {"name": "12-6:lock-index-sha256", "value": index["index_sha256"]},
            ],
        },
        "components": sbom_components,
    }

    vulnerability = _default_adjudication("vulnerability")
    if vulnerability_adjudication is not None:
        vulnerability = load_bound_adjudication(
            vulnerability_adjudication,
            kind="vulnerability",
            source_sha=source_sha,
            profile_id=profile_id,
            lock_index_sha256=index["index_sha256"],
        )
    license_state = _default_adjudication("license")
    if license_adjudication is not None:
        license_state = load_bound_adjudication(
            license_adjudication,
            kind="license",
            source_sha=source_sha,
            profile_id=profile_id,
            lock_index_sha256=index["index_sha256"],
        )

    evidence: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "source_sha": source_sha,
        "profile_id": profile_id,
        "lock_index": {
            "path": "requirements/locks/index.json",
            "file_sha256": sha256_file(index_path),
            "index_sha256": index["index_sha256"],
        },
        "lock_profile": {
            "path": profile_record["path"],
            "file_sha256": profile_record["sha256"],
            "manifest_sha256": profile["manifest_sha256"],
        },
        "component_count": len(components),
        "components_sha256": _sha256_json(components),
        "sbom": {
            "format": "CycloneDX",
            "spec_version": "1.6",
            "sha256": _sha256_json(sbom),
        },
        "vulnerability": vulnerability,
        "license": license_state,
    }
    evidence["evidence_sha256"] = _sha256_json(evidence)
    return sbom, evidence


def write_supply_chain_documents(
    *,
    sbom: Mapping[str, Any],
    evidence: Mapping[str, Any],
    sbom_path: str | Path,
    evidence_path: str | Path,
) -> None:
    Path(sbom_path).write_bytes(canonical_json_bytes(dict(sbom)))
    Path(evidence_path).write_bytes(canonical_json_bytes(dict(evidence)))


def validate_supply_chain_evidence(
    *,
    root: str | Path,
    sbom_path: str | Path,
    evidence_path: str | Path,
    expected_source_sha: str,
    require_resolved: bool,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    sbom = _load_json(Path(sbom_path))
    evidence = _load_json(Path(evidence_path))
    claimed = evidence.get("evidence_sha256")
    payload = dict(evidence)
    payload.pop("evidence_sha256", None)
    if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise SupplyChainEvidenceError("unsupported supply-chain evidence schema")
    if claimed != _sha256_json(payload):
        raise SupplyChainEvidenceError("supply-chain evidence self-hash mismatch")
    if evidence.get("source_sha") != expected_source_sha:
        raise SupplyChainEvidenceError("supply-chain evidence source SHA mismatch")
    if evidence.get("sbom", {}).get("sha256") != _sha256_json(sbom):
        raise SupplyChainEvidenceError("SBOM hash mismatch")
    index = validate_lock_index(root=root_path, index_path="requirements/locks/index.json")
    lock_record = evidence.get("lock_index")
    if not isinstance(lock_record, dict):
        raise SupplyChainEvidenceError("lock index evidence is malformed")
    if lock_record.get("index_sha256") != index["index_sha256"]:
        raise SupplyChainEvidenceError("lock semantic identity mismatch")
    if lock_record.get("file_sha256") != sha256_file(root_path / "requirements/locks/index.json"):
        raise SupplyChainEvidenceError("lock file identity mismatch")
    for kind in sorted(_KINDS):
        record = evidence.get(kind)
        if not isinstance(record, dict) or record.get("status") not in _STATUSES:
            raise SupplyChainEvidenceError(f"{kind} evidence status is malformed")
        if require_resolved and record["status"] != "PASS":
            raise SupplyChainEvidenceError(
                f"release preflight requires {kind}=PASS; got {record['status']}"
            )
    return evidence
