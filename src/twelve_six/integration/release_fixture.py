"""Deterministic, fail-closed release fixture primitives; NOT production release signing."""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import re
import stat
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SCHEMA = "12-6.release-fixture.v1"
_NOTICE_SCHEMA = "12-6.third-party-notices.v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_PATH = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*\Z")
_ALLOWED = {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC", "PSF-2.0"}


class FixtureIntegrityError(ValueError):
    """A fixture release/update/rollback input violated a trust boundary."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _path(name: str) -> str:
    if (not isinstance(name, str) or not _SAFE_PATH.fullmatch(name)
            or name.startswith("/") or "//" in name
            or any(piece in {"", ".", ".."} for piece in name.split("/"))
            or name.endswith("/")):
        raise FixtureIntegrityError("unsafe fixture path")
    return name


def build_notice_inventory(sbom: Mapping[str, Any],
                           license_observations: Mapping[str, Mapping[str, Any]]
                           ) -> dict[str, Any]:
    """Report exactly lock-bound packages; no unknown license is approved."""
    from .dependency_security import unique_components

    components = unique_components(sbom)
    if not isinstance(sbom.get("sbom_sha256"), str) or not _SHA.fullmatch(
        sbom["sbom_sha256"]
    ):
        raise FixtureIntegrityError("SBOM identity missing")
    if set(license_observations) != {item["key"] for item in components}:
        raise FixtureIntegrityError("notice component coverage mismatch")
    notices = []
    for item in components:
        observation = license_observations[item["key"]]
        if not isinstance(observation, Mapping):
            raise FixtureIntegrityError("invalid notice observation")
        expression = observation.get("license_expression")
        provenance = observation.get("metadata_sha256")
        if not isinstance(provenance, str) or not _SHA.fullmatch(provenance):
            raise FixtureIntegrityError("missing publisher license metadata digest")
        approved = expression in _ALLOWED and observation.get("status") == "DECLARED"
        notices.append({
            "component": item["key"], "purl": item["purl"], "profiles": item["profiles"],
            "declared_license": expression if isinstance(expression, str) else None,
            "metadata_sha256": provenance,
            "approval": "CANDIDATE_PERMISSIVE_REVIEW_REQUIRED" if approved else "UNRESOLVED",
            "full_license_text_verified": False,
        })
    payload = {"schema_version": _NOTICE_SCHEMA, "sbom_sha256": sbom["sbom_sha256"],
               "notices": notices, "release_approved": False}
    payload["notices_sha256"] = _digest(_canonical(payload))
    return payload


def verify_notice_inventory(sbom: Mapping[str, Any], notices: Mapping[str, Any]) -> None:
    payload = dict(notices)
    claimed = payload.pop("notices_sha256", None)
    if (not isinstance(claimed, str) or not _SHA.fullmatch(claimed)
            or _digest(_canonical(payload)) != claimed):
        raise FixtureIntegrityError("notice inventory hash mismatch")
    if notices.get("schema_version") != _NOTICE_SCHEMA:
        raise FixtureIntegrityError("unknown notice inventory schema")
    observed = {}
    rows = notices.get("notices")
    if not isinstance(rows, list):
        raise FixtureIntegrityError("notice inventory malformed")
    for row in rows:
        if not isinstance(row, dict) or row.get("component") in observed:
            raise FixtureIntegrityError("duplicate or malformed notice")
        observed[row["component"]] = {
            "status": "DECLARED" if row.get("approval") == "CANDIDATE_PERMISSIVE_REVIEW_REQUIRED" else "UNRESOLVED",
            "license_expression": row.get("declared_license"),
            "metadata_sha256": row.get("metadata_sha256"),
        }
        if row.get("full_license_text_verified") is not False:
            raise FixtureIntegrityError("unsubstantiated license text approval")
    if build_notice_inventory(sbom, observed) != notices:
        raise FixtureIntegrityError("notice inventory drift")


def pack_fixture(root: str | Path, relative_paths: list[str]) -> bytes:
    """Create reproducible ZIP, rejecting symlinks, unsafe paths and duplicates."""
    path_root = Path(root).resolve(strict=True)
    names = [_path(item) for item in relative_paths]
    if not names or len(set(names)) != len(names):
        raise FixtureIntegrityError("empty or duplicate fixture entries")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(names):
            target = path_root
            for part in name.split("/"):
                target /= part
                if target.is_symlink():
                    raise FixtureIntegrityError("symlink fixture entry refused")
            if not target.is_file() or not target.resolve(strict=True).is_relative_to(path_root):
                raise FixtureIntegrityError("fixture entry missing or unsafe")
            if not stat.S_ISREG(target.stat().st_mode):
                raise FixtureIntegrityError("fixture entry not regular")
            metadata = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            metadata.create_system = 3
            metadata.external_attr = (stat.S_IFREG | 0o644) << 16
            metadata.compress_type = zipfile.ZIP_STORED
            archive.writestr(metadata, target.read_bytes())
    return output.getvalue()


def inspect_fixture(archive_bytes: bytes, expected_hash: str) -> dict[str, str]:
    if len(archive_bytes) > 64 * 1024 * 1024:
        raise FixtureIntegrityError("oversized fixture archive")
    if not isinstance(expected_hash, str) or not _SHA.fullmatch(expected_hash):
        raise FixtureIntegrityError("invalid expected artifact hash")
    if not hmac.compare_digest(_digest(archive_bytes), expected_hash):
        raise FixtureIntegrityError("artifact hash mismatch")
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            names = archive.namelist()
            if not names or len(set(names)) != len(names):
                raise FixtureIntegrityError("empty or duplicate ZIP members")
            result = {}
            for entry in archive.infolist():
                name = _path(entry.filename)
                mode = entry.external_attr >> 16
                if entry.is_dir() or mode & 0o170000 != stat.S_IFREG:
                    raise FixtureIntegrityError("nonregular ZIP member")
                if entry.flag_bits & 1:
                    raise FixtureIntegrityError("encrypted ZIP member")
                if entry.file_size > 16 * 1024 * 1024:
                    raise FixtureIntegrityError("oversized fixture member")
                result[name] = _digest(archive.read(entry))
            return result
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise FixtureIntegrityError("unreadable fixture archive") from exc


def sign_fixture(*, key: bytes, archive_sha256: str, release_id: str,
                 generation: int, previous_sha256: str | None) -> dict[str, Any]:
    """HMAC fixture-only attestation. Real signing and release keys: Plan 10."""
    if not isinstance(key, bytes) or len(key) < 32:
        raise FixtureIntegrityError("fixture key must be >= 32 bytes")
    if not isinstance(archive_sha256, str) or not _SHA.fullmatch(archive_sha256):
        raise FixtureIntegrityError("invalid artifact digest")
    if not isinstance(release_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,62}", release_id):
        raise FixtureIntegrityError("invalid release id")
    if type(generation) is not int or generation < 1:
        raise FixtureIntegrityError("invalid release generation")
    if previous_sha256 is not None and (
        not isinstance(previous_sha256, str) or not _SHA.fullmatch(previous_sha256)
    ):
        raise FixtureIntegrityError("invalid predecessor digest")
    payload = {
        "schema_version": _SCHEMA, "release_id": release_id, "generation": generation,
        "archive_sha256": archive_sha256, "previous_sha256": previous_sha256,
        "signature_type": "TEST_FIXTURE_HMAC_SHA256_NOT_PRODUCTION",
    }
    payload["signature"] = hmac.new(key, _canonical(payload), hashlib.sha256).hexdigest()
    return payload


def verify_fixture(*, key: bytes, receipt: Mapping[str, Any],
                   archive_bytes: bytes) -> None:
    if not isinstance(receipt, Mapping):
        raise FixtureIntegrityError("missing receipt")
    reference = sign_fixture(
        key=key, archive_sha256=receipt.get("archive_sha256"),
        release_id=receipt.get("release_id"), generation=receipt.get("generation"),
        previous_sha256=receipt.get("previous_sha256"),
    )
    if set(receipt) != set(reference) or not isinstance(receipt.get("signature"), str):
        raise FixtureIntegrityError("receipt schema mismatch")
    if not hmac.compare_digest(receipt["signature"], reference["signature"]):
        raise FixtureIntegrityError("fixture signature mismatch")
    inspect_fixture(archive_bytes, receipt["archive_sha256"])


def admit_update(*, current: Mapping[str, Any], proposed: Mapping[str, Any],
                 key: bytes, current_bytes: bytes, proposed_bytes: bytes) -> None:
    verify_fixture(key=key, receipt=current, archive_bytes=current_bytes)
    verify_fixture(key=key, receipt=proposed, archive_bytes=proposed_bytes)
    if (current.get("schema_version") != _SCHEMA
            or proposed["release_id"] != current.get("release_id")
            or type(current.get("generation")) is not int
            or proposed["generation"] != current["generation"] + 1
            or proposed["previous_sha256"] != current.get("archive_sha256")):
        raise FixtureIntegrityError("update predecessor/generation mismatch")


def admit_rollback(*, current: Mapping[str, Any], previous: Mapping[str, Any],
                   key: bytes, current_bytes: bytes, previous_bytes: bytes) -> None:
    verify_fixture(key=key, receipt=current, archive_bytes=current_bytes)
    verify_fixture(key=key, receipt=previous, archive_bytes=previous_bytes)
    if (current.get("schema_version") != _SCHEMA
            or current.get("release_id") != previous["release_id"]
            or type(current.get("generation")) is not int
            or current["generation"] != previous["generation"] + 1
            or current.get("previous_sha256") != previous["archive_sha256"]):
        raise FixtureIntegrityError("rollback predecessor/generation mismatch")
