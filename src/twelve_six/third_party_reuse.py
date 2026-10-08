"""Fail-closed Plan-1 third-party code reuse and Base ancestry checks.

No downloader, model loader, scheduler, registry, or evaluator is created here.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any

HEX = re.compile(r"^[0-9a-f]{64}$")
AUTHORITIES = {
    "model": "twelve_six.model",
    "checkpoint": "twelve_six.checkpoint",
    "scheduler": "twelve_six.training",
    "evaluation": "twelve_six.capability_map",
}
ASSET_KEYS = {
    "name", "role", "status", "upstream_url", "version", "source_sha256",
    "license_spdx", "license_evidence_sha256", "security_posture",
    "data_rights", "model_weights", "replacement_boundary",
}
LICENSES = {"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "ISC"}


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _bad_constant(value: str) -> Any:
    raise ValueError("nonfinite JSON")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite JSON")
    return result


def _strict(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or len(raw) > 1048576:
        raise ValueError("bounded UTF-8 bytes required")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                           parse_constant=_bad_constant, parse_float=_finite_float)
    except ValueError as exc:
        if str(exc) == "duplicate JSON key":
            raise
        raise ValueError("invalid strict JSON") from exc
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("invalid strict JSON") from exc
    if type(value) is not dict:
        raise ValueError("object root required")
    return value


def _hash(value: object) -> bool:
    return type(value) is str and HEX.fullmatch(value) is not None


def validate_reuse_catalog(raw: bytes) -> dict[str, Any]:
    payload = _strict(raw)
    if set(payload) != {"schema_version", "canonical_authorities", "assets"}:
        raise ValueError("reuse schema drift")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("reuse schema version drift")
    if payload["canonical_authorities"] != AUTHORITIES:
        raise ValueError("second canonical authority forbidden")
    assets = payload["assets"]
    if type(assets) is not list or not assets:
        raise ValueError("assets must be a nonempty list")
    observed: set[str] = set()
    for asset in assets:
        if type(asset) is not dict or set(asset) != ASSET_KEYS:
            raise ValueError("incomplete asset evidence")
        name = asset["name"]
        if type(name) is not str or not re.fullmatch(r"[a-z][a-z0-9_-]{1,63}", name):
            raise ValueError("invalid asset identifier")
        if name in observed:
            raise ValueError("duplicate asset")
        observed.add(name)
        if type(asset["role"]) is not str or asset["role"] not in {
            "code_adapter", "dataset", "model_weights"
        }:
            raise ValueError("invalid role")
        boundary = asset["replacement_boundary"]
        if type(boundary) is not str or not boundary.startswith("adapter:"):
            raise ValueError("replaceable adapter boundary required")
        if any(token in boundary.lower().split(":")[1].split(".")
               for token in ("model", "checkpoint", "scheduler", "evaluation", "registry")):
            raise ValueError("adapter cannot replace canonical authority")
        status = asset["status"]
        if type(status) is not str or status not in {
            "CANDIDATE_UNQUALIFIED", "REVIEWED_CODE_ONLY"
        }:
            raise ValueError("invalid admission status")
        if status == "CANDIDATE_UNQUALIFIED":
            if any(asset[field] is not None for field in (
                "version", "source_sha256", "license_evidence_sha256"
            )) or asset["security_posture"] != "UNKNOWN":
                raise ValueError("candidate cannot claim partial admission")
            continue
        if asset["role"] != "code_adapter" or asset["model_weights"] != "NONE":
            raise ValueError("no foreign weights in canonical Base")
        if asset["data_rights"] != "NOT_APPLICABLE_CODE":
            raise ValueError("code permission is not dataset permission")
        if asset["security_posture"] != "REVIEWED":
            raise ValueError("unknown security posture")
        if type(asset["license_spdx"]) is not str or asset["license_spdx"] not in LICENSES:
            raise ValueError("unknown/incompatible license")
        if type(asset["upstream_url"]) is not str or not asset["upstream_url"].startswith(
            "https://"
        ):
            raise ValueError("missing HTTPS source")
        if type(asset["version"]) is not str or not asset["version"].strip():
            raise ValueError("exact version required")
        if not _hash(asset["source_sha256"]) or not _hash(
            asset["license_evidence_sha256"]
        ):
            raise ValueError("source/license hash required")
    return payload


def require_reviewed_code(
    raw: bytes,
    name: str,
    exact_source_sha256: str,
    *,
    independently_pinned_catalog_sha256: str,
) -> dict[str, Any]:
    """Inspect a reviewed-code claim only against an independent approved catalog pin.

    The pin must be obtained from a trusted review, never recalculated from the
    supplied untrusted raw value. Callers must hash the actual source artifact
    and independently verify their reviewer's security/license assessment.
    """
    if not _hash(independently_pinned_catalog_sha256) or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), independently_pinned_catalog_sha256
    ):
        raise ValueError("independently approved catalog digest required")
    payload = validate_reuse_catalog(raw)
    for asset in payload["assets"]:
        if asset["name"] == name:
            if asset["status"] != "REVIEWED_CODE_ONLY":
                raise ValueError("unqualified code asset")
            if not _hash(exact_source_sha256) or asset["source_sha256"] != exact_source_sha256:
                raise ValueError("source hash drift")
            return dict(asset)
    raise ValueError("unknown external asset")



def _hash_regular_archive(path: str | Path) -> str:
    """Digest an actual regular review artifact without trusting path/symlink aliases."""
    source = Path(path)
    before = source.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise ValueError("review artifact must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        handle = os.open(source, flags)
    except OSError as exc:
        raise ValueError("review artifact cannot be opened safely") from exc
    try:
        opened = os.fstat(handle)
        if not stat.S_ISREG(opened.st_mode) or (
            before.st_dev, before.st_ino, before.st_size
        ) != (opened.st_dev, opened.st_ino, opened.st_size):
            raise ValueError("review artifact changed while opening")
        digest = hashlib.sha256()
        while chunk := os.read(handle, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(handle)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            after.st_dev, after.st_ino, after.st_size
        ):
            raise ValueError("review artifact changed while reading")
        return digest.hexdigest()
    finally:
        os.close(handle)


def verify_reviewed_code_archive(
    catalog: bytes,
    name: str,
    *,
    source_archive: str | Path,
    license_file: str | Path,
    independently_pinned_catalog_sha256: str,
) -> dict[str, Any]:
    """Check concrete source and license bytes against a separately approved record.

    This never installs or executes the reviewed asset. It does not perform the
    human/security review: the out-of-band catalog SHA-256 must attest that review.
    """
    actual_source = _hash_regular_archive(source_archive)
    approved = require_reviewed_code(
        catalog,
        name,
        actual_source,
        independently_pinned_catalog_sha256=independently_pinned_catalog_sha256,
    )
    license_hash = _hash_regular_archive(license_file)
    if not hmac.compare_digest(license_hash, approved["license_evidence_sha256"]):
        raise ValueError("reviewed code license bytes drifted")
    return approved


def validate_base_lineage(payload: dict[str, Any], trusted_genesis: dict[str, Any]) -> str:
    """Require closed first-party ancestry rooted in separately verified scratch init.

    This does not authenticate the caller's trusted_genesis or checkpoint bytes.
    Those must be authenticated by the canonical checkpoint authority.
    """
    expected = {
        "schema_version", "genesis_id", "model_spec_sha256",
        "init_spec_sha256", "head_id", "checkpoints"
    }
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("Base lineage schema drift")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported Base ancestry version")
    if type(trusted_genesis) is not dict or set(trusted_genesis) != {
        "genesis_id", "model_spec_sha256", "init_spec_sha256", "origin"
    } or trusted_genesis["origin"] != "local_random_init":
        raise ValueError("independently trusted scratch genesis required")
    for key in ("genesis_id", "model_spec_sha256", "init_spec_sha256"):
        if not _hash(payload[key]) or payload[key] != trusted_genesis[key]:
            raise ValueError("foreign lineage/model identity drift")
    checkpoints = payload["checkpoints"]
    if type(checkpoints) is not list or not checkpoints or len(checkpoints) > 2048:
        raise ValueError("complete checkpoint graph required")
    parents: dict[str, list[str]] = {}
    for item in checkpoints:
        if type(item) is not dict or set(item) != {"id", "parents", "origin"}:
            raise ValueError("invalid checkpoint")
        node, refs = item["id"], item["parents"]
        if not _hash(node) or node in parents:
            raise ValueError("invalid or duplicate checkpoint identity")
        if item["origin"] != "canonical_base":
            raise ValueError("foreign/pretrained/posttrained checkpoint")
        if type(refs) is not list or len(refs) > 1 or any(not _hash(x) for x in refs):
            raise ValueError("ambiguous parents")
        parents[node] = refs
    root = payload["genesis_id"]
    if root not in parents or parents[root] != [] or payload["head_id"] not in parents:
        raise ValueError("missing scratch genesis/head")
    for node, refs in parents.items():
        if node != root and len(refs) != 1:
            raise ValueError("unexpected second Base root")
        if any(p not in parents for p in refs):
            raise ValueError("foreign checkpoint parent")
    completed: set[str] = {root}
    for node in parents:
        cursor = node
        seen: set[str] = set()
        while cursor not in completed:
            if cursor in seen:
                raise ValueError("ancestry cycle")
            seen.add(cursor)
            cursor = parents[cursor][0]
        completed.update(seen)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def prepare_trusted_base_checkpoint(
    directory: str | Path,
    *,
    lineage_bytes: bytes,
    trusted_genesis_bytes: bytes,
    expected_genesis_sha256: str,
) -> Any:
    """Admit canonical checkpoint bytes only after independently pinned Base ancestry.

    This is an opt-in gate in front of the incumbent checkpoint snapshot/loader,
    not a second checkpoint authority. The expected genesis hash MUST come from
    an independently authenticated trust root, not from lineage_bytes.
    The returned VerifiedCheckpoint is consumed by checkpoint.load_verified_checkpoint.
    No claim about provenance is made for callers bypassing this gated API.
    """
    if not _hash(expected_genesis_sha256):
        raise ValueError("independently pinned genesis SHA-256 required")
    if type(trusted_genesis_bytes) is not bytes:
        raise ValueError("trusted genesis must be bytes")
    digest = hashlib.sha256(trusted_genesis_bytes).hexdigest()
    if not hmac.compare_digest(digest, expected_genesis_sha256):
        raise ValueError("trusted genesis digest mismatch")

    genesis = _strict(trusted_genesis_bytes)
    lineage = _strict(lineage_bytes)
    validate_base_lineage(lineage, genesis)

    # Canonical checkpoint authority authenticates the actual immutable payload
    # and its manifest BEFORE any model, optimizer or RNG mutation.
    from .checkpoint import prepare_checkpoint_load

    verified = prepare_checkpoint_load(directory)
    manifest = verified.manifest
    identity = manifest["identity"]
    if manifest["checkpoint_id"] != lineage["head_id"]:
        raise ValueError("checkpoint head is not the admitted Base lineage")
    if identity["model_spec_hash"] != lineage["model_spec_sha256"]:
        raise ValueError("Base ModelSpec does not match checkpoint")
    training_config = identity.get("training_config")
    if (
        type(training_config) is not dict
        or training_config.get("init_spec_sha256") != lineage["init_spec_sha256"]
    ):
        raise ValueError("Base InitSpec is not bound to checkpoint")
    return verified


def verify_installed_wheel_record(
    package: str,
    *,
    expected_version: str,
    independently_pinned_record_sha256: str,
) -> dict[str, str | int]:
    """Verify actual installed wheel bytes against an independently trusted RECORD pin.

    The RECORD itself is self-declared and NOT evidence unless its SHA-256 was
    independently authenticated outside this installation. This is a bounded
    installed-byte integrity check, not a source-license/security review or
    an authorization to use a dataset, model, or unreviewed wheel.
    """
    import base64
    import csv
    import io
    from importlib import metadata
    from pathlib import PurePosixPath

    if type(package) is not str or not re.fullmatch(r"[A-Za-z0-9_.-]{2,100}", package):
        raise ValueError("invalid distribution name")
    if type(expected_version) is not str or not expected_version.strip():
        raise ValueError("exact installed version required")
    if not _hash(independently_pinned_record_sha256):
        raise ValueError("independent installed RECORD pin required")

    def normalize(value: str) -> str:
        return re.sub(r"[-_.]+", "-", value).lower()
    dist = metadata.distribution(package)
    actual_name = dist.metadata.get("Name")
    if (
        type(actual_name) is not str
        or normalize(actual_name) != normalize(package)
        or dist.version != expected_version
    ):
        raise ValueError("installed distribution identity/version drift")
    record = dist.read_text("RECORD")
    if type(record) is not str:
        raise ValueError("installed wheel RECORD missing")
    raw_record = record.encode("utf-8")
    if not raw_record or len(raw_record) > 4 * 1024 * 1024:
        raise ValueError("installed RECORD out of bounds")
    observed_pin = hashlib.sha256(raw_record).hexdigest()
    if not hmac.compare_digest(observed_pin, independently_pinned_record_sha256):
        raise ValueError("installed RECORD pin drift")

    try:
        rows = list(csv.reader(io.StringIO(record), strict=True))
    except csv.Error as exc:
        raise ValueError("invalid wheel RECORD CSV") from exc
    if not rows or len(rows) > 50000:
        raise ValueError("invalid installed file inventory size")

    observed_paths: set[str] = set()
    self_records = 0
    verified_files = 0
    for row in rows:
        if len(row) != 3:
            raise ValueError("invalid wheel RECORD row")
        name, digest_field, size_field = row
        relative = PurePosixPath(name)
        if (
            not name
            or name.startswith(("/", "\\"))
            or "\\" in name
            or ":" in name
            or relative.is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))
            or name in observed_paths
        ):
            raise ValueError("unsafe or duplicate installed file path")
        observed_paths.add(name)
        if not digest_field:
            if (
                not name.endswith(".dist-info/RECORD")
                or size_field
                or self_records
            ):
                raise ValueError("unhashed installed file")
            self_records += 1
            continue
        if not digest_field.startswith("sha256=") or not size_field.isdecimal():
            raise ValueError("unsupported installed file hash/size")
        try:
            encoded = digest_field.removeprefix("sha256=")
            decoded = base64.b64decode(
                encoded + "=" * ((-len(encoded)) % 4),
                altchars=b"-_", validate=True,
            )
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError("invalid wheel RECORD digest") from exc
        if len(decoded) != 32:
            raise ValueError("invalid installed digest length")
        location = dist.locate_file(name)
        if str(location) == name:
            raise ValueError("unresolved installed file location")
        path = Path(location)
        try:
            size = path.lstat().st_size
            actual_hash = _hash_regular_archive(path)
        except (OSError, ValueError) as exc:
            raise ValueError("installed file unavailable or unsafe") from exc
        if size != int(size_field) or not hmac.compare_digest(
            bytes.fromhex(actual_hash), decoded
        ):
            raise ValueError("installed file contents drift")
        verified_files += 1
    if self_records != 1 or not verified_files:
        raise ValueError("incomplete installed RECORD")
    return {
        "distribution": normalize(actual_name),
        "version": dist.version,
        "record_sha256": observed_pin,
        "verified_files": verified_files,
    }
