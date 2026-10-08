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
from urllib.parse import urlsplit

HEX = re.compile(r"^[0-9a-f]{64}$")
# Bound hashing of untrusted external review artifacts before and during reads.
MAX_REVIEW_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
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
# Strictly enumerated permissive SPDX *terms*, not an automatic legal approval.
# Real NumPy and PyTorch metadata use composite expressions with distinct terms.
# A human/source-byte/license-notice review and independent catalog pin remain mandatory.
LICENSES = {
    "0BSD", "Apache-2.0", "Apache-2.0 WITH LLVM-exception", "BSD-2-Clause",
    "BSD-3-Clause", "BSL-1.0", "CC0-1.0", "ISC", "MIT", "Zlib",
}


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
    # Python's C JSON encoder/decoder may accept nesting far deeper than the
    # interpreter recursion limit. Put a deterministic platform-independent
    # structural bound on all catalog, genesis, and Base lineage claims.
    pending: list[tuple[Any, int]] = [(value, 0)]
    observed_nodes = 0
    while pending:
        item, depth = pending.pop()
        observed_nodes += 1
        if depth > 64 or observed_nodes > 32768:
            raise ValueError("strict JSON nesting/node budget exceeded")
        if type(item) is dict:
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
    return value


def _hash(value: object) -> bool:
    return type(value) is str and HEX.fullmatch(value) is not None


def _reviewable_spdx_conjunction(value: object) -> bool:
    """Only recognize bounded all-permissive SPDX AND expressions.

    Parsing an expression does not certify its truth, supplied notice coverage,
    compatibility of bundled binary components, or a code-review decision.
    OR/unknown licenses, unapproved exceptions and alternate syntax fail closed.
    """
    if type(value) is not str or not value or len(value) > 4096:
        return False
    terms = value.split(" AND ")
    return (
        1 <= len(terms) <= 32
        and len(set(terms)) == len(terms)
        and all(term in LICENSES for term in terms)
    )


def _valid_https_source(value: object) -> bool:
    """Reject ambiguous or unauthenticatable external code-source URLs."""
    if type(value) is not str or not value or len(value) > 2048:
        return False
    if any(char.isspace() or ord(char) < 33 or ord(char) == 127 or char == "\\" for char in value):
        return False
    try:
        parsed = urlsplit(value)
        return (
            parsed.scheme == "https"
            and bool(parsed.hostname)
            # URL parsers permit non-DNS hosts like "." and "example..org".
            # An independently reviewable HTTPS code origin must have a
            # bounded, well-formed ASCII DNS hostname.
            and len(parsed.hostname) <= 253
            and "." in parsed.hostname
            # A dotted numeric host can be interpreted as a local/private
            # IP address (including abbreviated IPv4 forms) rather than a
            # reviewable DNS origin. Never accept it as upstream code source.
            and re.search(r"[a-z]", parsed.hostname.rsplit(".", 1)[-1]) is not None
            and all(
                re.fullmatch(
                    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?",
                    label,
                )
                for label in parsed.hostname.split(".")
            )
            # Unicode confusables and encoded authority delimiters cannot be
            # independently reviewed as an unambiguous source origin.
            and parsed.netloc.isascii()
            and "%" not in parsed.netloc
            and parsed.username is None
            and parsed.password is None
            and parsed.fragment == ""
            # urlsplit() treats an explicit but empty port as None.
            # Reject that ambiguous origin instead of silently accepting it.
            and not parsed.netloc.endswith(":")
            and (parsed.port is None or parsed.port > 0)
        )
    except ValueError:
        return False


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
        # A catalog must not disguise model/registry/checkpoint authority inside
        # a nested adapter path (such as "adapter:vendor.model_registry").
        if asset["replacement_boundary"] != f"adapter:{name}":
            raise ValueError("adapter boundary must equal the asset identifier")
        status = asset["status"]
        if type(status) is not str or status not in {
            "CANDIDATE_UNQUALIFIED", "REVIEWED_CODE_ONLY"
        }:
            raise ValueError("invalid admission status")
        if status == "CANDIDATE_UNQUALIFIED":
            if (
                any(asset[field] is not None for field in (
                    "version", "source_sha256", "license_evidence_sha256",
                    "upstream_url", "license_spdx",
                ))
                or asset["security_posture"] != "UNKNOWN"
                or asset["data_rights"] != "NOT_ASSESSED"
                or asset["model_weights"] != "NONE"
            ):
                raise ValueError("candidate cannot claim partial admission")
            continue
        if asset["role"] != "code_adapter" or asset["model_weights"] != "NONE":
            raise ValueError("no foreign weights in canonical Base")
        if asset["data_rights"] != "NOT_APPLICABLE_CODE":
            raise ValueError("code permission is not dataset permission")
        if asset["security_posture"] != "REVIEWED":
            raise ValueError("unknown security posture")
        if not _reviewable_spdx_conjunction(asset["license_spdx"]):
            raise ValueError("unknown/incompatible license")
        if not _valid_https_source(asset["upstream_url"]):
            raise ValueError("missing HTTPS source")
        if (
            type(asset["version"]) is not str
            or re.fullmatch(r"(?:[0-9]|v[0-9])[A-Za-z0-9._+!-]{0,127}", asset["version"])
            is None
        ):
            raise ValueError("exact version required; ranges/aliases forbidden")
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
    # Reject oversized attacker inputs before any digest work, not just at JSON parsing.
    if type(raw) is not bytes or len(raw) > 1048576:
        raise ValueError("bounded reviewed catalog bytes required")
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
    # Normalize the lexical path without resolving redirects, so inherited
    # symlinks/junctions cannot redirect a reviewed artifact outside its tree.
    source = Path(os.path.abspath(os.fspath(path)))
    def ancestor_identities() -> tuple[tuple[int, int, int], ...]:
        identities = []
        for ancestor in source.parents:
            info = ancestor.lstat()
            if (
                not stat.S_ISDIR(info.st_mode)
                or ancestor.is_symlink()
                or getattr(ancestor, "is_junction", lambda: False)()
            ):
                raise ValueError("review artifact parent directory is a symlink/junction")
            identities.append((info.st_dev, info.st_ino, info.st_mode))
        return tuple(identities)

    parents_before = ancestor_identities()

    def fingerprint(info: os.stat_result) -> tuple[int, int, int, int, int]:
        # Catch in-place, same-size replacement during archive verification.
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    before = source.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise ValueError("review artifact must be a regular non-symlink file")
    # A regular file can be swapped for a FIFO after lstat: avoid a blocking open.
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        handle = os.open(source, flags)
    except OSError as exc:
        raise ValueError("review artifact cannot be opened safely") from exc
    try:
        opened = os.fstat(handle)
        if not stat.S_ISREG(opened.st_mode) or fingerprint(before) != fingerprint(opened):
            raise ValueError("review artifact changed while opening")
        if opened.st_size > MAX_REVIEW_ARTIFACT_BYTES:
            raise ValueError("review artifact size limit exceeded")
        digest = hashlib.sha256()
        read_bytes = 0
        while chunk := os.read(handle, 1024 * 1024):
            read_bytes += len(chunk)
            if read_bytes > MAX_REVIEW_ARTIFACT_BYTES:
                raise ValueError("review artifact size limit exceeded")
            digest.update(chunk)
        after = os.fstat(handle)
        if fingerprint(opened) != fingerprint(after):
            raise ValueError("review artifact changed while reading")
        # A pathname can be swapped while the original descriptor remains stable.
        # Never attest bytes for a path now naming a different artifact.
        try:
            named = source.lstat()
        except OSError as exc:
            raise ValueError("review artifact path changed while reading") from exc
        if not stat.S_ISREG(named.st_mode) or fingerprint(opened) != fingerprint(named):
            raise ValueError("review artifact path changed while reading")
        # Revalidate each ancestor after hashing, not just the opened leaf:
        # a directory can be renamed and replaced with a symlink mid-read.
        if ancestor_identities() != parents_before:
            raise ValueError("review artifact parent directory changed while reading")
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
    # Authenticate the code review and admission status BEFORE reading any
    # potentially large, attacker-selected source or license file. A bad pin or
    # unqualified candidate must never trigger archive I/O.
    if type(catalog) is not bytes or len(catalog) > 1048576:
        raise ValueError("bounded reviewed catalog bytes required")
    if not _hash(independently_pinned_catalog_sha256) or not hmac.compare_digest(
        hashlib.sha256(catalog).hexdigest(), independently_pinned_catalog_sha256
    ):
        raise ValueError("independently approved catalog digest required")
    vetted_catalog = validate_reuse_catalog(catalog)
    named = next(
        (asset for asset in vetted_catalog["assets"] if asset["name"] == name),
        None,
    )
    if named is None:
        raise ValueError("unknown external asset")
    if named["status"] != "REVIEWED_CODE_ONLY":
        raise ValueError("unqualified code asset")

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
    if not _hash(payload["head_id"]):
        raise ValueError("invalid checkpoint head")
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
    expected_lineage_sha256: str,
) -> Any:
    """Admit canonical checkpoint bytes only after independently pinned Base ancestry.

    This is an opt-in gate in front of the incumbent checkpoint snapshot/loader,
    not a second checkpoint authority. Both independently pinned genesis and
    complete lineage hashes must come from authenticated trust roots,
    authenticated trust roots, never recalculated from caller-supplied bytes.
    The returned VerifiedCheckpoint is consumed by checkpoint.load_verified_checkpoint.
    No claim about provenance is made for callers bypassing this gated API.
    """
    # Bound both externally supplied Base claims before SHA-256 processing.
    if type(lineage_bytes) is not bytes or len(lineage_bytes) > 1048576:
        raise ValueError("bounded Base lineage bytes required")
    if type(trusted_genesis_bytes) is not bytes or len(trusted_genesis_bytes) > 1048576:
        raise ValueError("bounded trusted genesis bytes required")
    if not _hash(expected_lineage_sha256):
        raise ValueError("independently pinned lineage SHA-256 required")
    if type(lineage_bytes) is not bytes or not hmac.compare_digest(
        hashlib.sha256(lineage_bytes).hexdigest(), expected_lineage_sha256
    ):
        raise ValueError("independently pinned Base lineage digest mismatch")
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
    import sysconfig
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
    license_notices: list[tuple[str, str]] = []
    unverified_bytecode_files = 0
    for row in rows:
        if len(row) != 3:
            raise ValueError("invalid wheel RECORD row")
        name, digest_field, size_field = row
        relative = PurePosixPath(name)
        segments = name.split("/")
        ordinary_entry = all(part not in ("", ".", "..") for part in segments)
        # PEP 427 RECORD legitimately contains generated console scripts outside
        # site-packages (e.g. NumPy f2py and PyTorch torchrun). Only the exact
        # current environment's bin/Scripts directory is eligible; arbitrary
        # "../" traversal and subdirectories remain forbidden.
        scripts_entry = False
        if (
            len(segments) == 5
            and segments[:3] == ["..", "..", ".."]
            and segments[3] in ("bin", "Scripts")
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}", segments[4])
        ):
            scripts_dir = sysconfig.get_path("scripts")
            if scripts_dir:
                location = Path(os.path.abspath(os.fspath(dist.locate_file(name))))
                scripts_entry = location.parent == Path(os.path.abspath(scripts_dir))
        if (
            not name
            or name.startswith(("/", "\\"))
            or "\\" in name
            or ":" in name
            or relative.is_absolute()
            or not (ordinary_entry or scripts_entry)
            or name in observed_paths
        ):
            raise ValueError("unsafe or duplicate installed file path")
        observed_paths.add(name)
        if not digest_field:
            # Python installers create interpreter-specific .pyc caches that
            # do not belong to the publisher's signed wheel. Observe these
            # but NEVER count them as verified code in an admission receipt.
            # Source/native binaries and arbitrary unhashed files stay denied.
            parts = name.split("/")
            if (
                not size_field
                and len(parts) >= 3
                and parts[-2] == "__pycache__"
                and re.fullmatch(
                    r"[a-zA-Z_][a-zA-Z0-9_.-]*\.cpython-\d+"
                    r"(?:\.opt-[0-2])?\.pyc",
                    parts[-1],
                )
            ):
                unverified_bytecode_files += 1
                continue
            if (
                not name.endswith(".dist-info/RECORD")
                or size_field
                or self_records
            ):
                raise ValueError("unhashed installed file")
            # The sole unhashed RECORD must belong to this exact distribution.
            # A separately pinned foreign RECORD is not proof of this wheel.
            stem = name.removesuffix(".dist-info/RECORD")
            if "-" not in stem:
                raise ValueError("installed RECORD identity/version drift")
            record_name, record_version = stem.rsplit("-", 1)
            if (
                normalize(record_name) != normalize(actual_name)
                or record_version != dist.version
            ):
                raise ValueError("installed RECORD identity/version drift")
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
        # All observed notice bytes were verified against a trusted wheel
        # RECORD above. This inventory is a reviewer lead, NOT legal approval.
        basename = name.split("/")[-1].lower()
        if re.fullmatch(
            r"(?:licen[cs]e|licenses|notices?|copying|copyright)"
            r"(?:[._-][a-z0-9._+-]{1,100})?",
            basename,
        ):
            license_notices.append((name, actual_hash))
            if len(license_notices) > 256:
                raise ValueError("too many wheel license notices")
    if self_records != 1 or not verified_files:
        raise ValueError("incomplete installed RECORD")
    notice_identity = hashlib.sha256(json.dumps(
        sorted(license_notices), separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")).hexdigest()
    return {
        "unverified_bytecode_files": unverified_bytecode_files,
        "license_notice_count": len(license_notices),
        "license_notice_inventory_sha256": notice_identity,
        "distribution": normalize(actual_name),
        "version": dist.version,
        "record_sha256": observed_pin,
        "verified_files": verified_files,
    }


def verify_reviewed_installed_backend(
    catalog: bytes,
    name: str,
    *,
    source_archive: str | Path,
    license_file: str | Path,
    distribution: str,
    independently_pinned_catalog_sha256: str,
    independently_pinned_record_sha256: str,
    independently_pinned_notice_inventory_sha256: str,
) -> dict[str, Any]:
    """Admit code only after BOTH upstream review and installed wheel checks.

    The approved upstream catalog pin, actual upstream source/license bytes,
    and installed wheel RECORD pin must be authenticated independently.
    Merely hashing local metadata or copying a catalog does not supply the
    independent review. No package is imported, executed or installed here.
    This function has no authority to admit datasets or model weights.
    """
    # Validate the review BEFORE dereferencing any untrusted archive or wheel.
    if type(distribution) is not str or not re.fullmatch(
        r"[A-Za-z0-9_.-]{2,100}", distribution
    ):
        raise ValueError("invalid external distribution")
    if not _hash(independently_pinned_record_sha256):
        raise ValueError("independent installed RECORD pin required")
    if not _hash(independently_pinned_notice_inventory_sha256):
        raise ValueError("independent full-license-notice inventory pin required")
    # Reviewed source and installed wheel must denote the SAME published code.
    # Other optional asset-to-wheel aliases require their own versioned,
    # reviewed mapping; never guess a distribution from untrusted input.
    approved_distribution_names = {
        "pytorch": "torch",
        "numpy": "numpy",
        "safetensors": "safetensors",
    }
    if type(name) is not str or distribution != approved_distribution_names.get(name):
        raise ValueError("unreviewed upstream-to-wheel distribution binding")
    if type(catalog) is not bytes or len(catalog) > 1048576:
        raise ValueError("bounded reviewed catalog bytes required")
    if not _hash(independently_pinned_catalog_sha256) or not hmac.compare_digest(
        hashlib.sha256(catalog).hexdigest(), independently_pinned_catalog_sha256
    ):
        raise ValueError("independently approved catalog digest required")
    approved_catalog = validate_reuse_catalog(catalog)
    named = next((a for a in approved_catalog["assets"] if a["name"] == name), None)
    if named is None:
        raise ValueError("unknown external asset")
    if named["status"] != "REVIEWED_CODE_ONLY":
        raise ValueError("unqualified code asset")
    # A project may be published under a different distribution name (e.g.
    # 'pytorch' upstream vs 'torch' on PyPI); require an explicit caller identity.
    # Both upstream and wheel checks must pass before returning any claim.
    reviewed = verify_reviewed_code_archive(
        catalog,
        name,
        source_archive=source_archive,
        license_file=license_file,
        independently_pinned_catalog_sha256=independently_pinned_catalog_sha256,
    )
    installed = verify_installed_wheel_record(
        distribution,
        expected_version=reviewed["version"],
        independently_pinned_record_sha256=independently_pinned_record_sha256,
    )
    if installed["unverified_bytecode_files"]:
        raise ValueError("installed wheel has unattested executable Python bytecode")
    if installed["license_notice_count"] < 1:
        raise ValueError("wheel has no reviewable LICENSE/NOTICE file")
    if not hmac.compare_digest(
        installed["license_notice_inventory_sha256"],
        independently_pinned_notice_inventory_sha256,
    ):
        raise ValueError("wheel license inventory digest drift")
    # The return value records checks only. It does not create a second package
    # registry, nor confer permission to load foreign model/data artifacts.
    return {
        "name": name,
        "status": "REVIEWED_CODE_ONLY",
        "version": reviewed["version"],
        "source_sha256": reviewed["source_sha256"],
        "upstream_url": reviewed["upstream_url"],
        "license_spdx": reviewed["license_spdx"],
        "license_evidence_sha256": reviewed["license_evidence_sha256"],
        "security_posture": reviewed["security_posture"],
        "replacement_boundary": reviewed["replacement_boundary"],
        "distribution": installed["distribution"],
        "record_sha256": installed["record_sha256"],
        "license_notice_count": installed["license_notice_count"],
        "license_notice_inventory_sha256": installed["license_notice_inventory_sha256"],
        "verified_files": installed["verified_files"],
        "unverified_bytecode_files": installed["unverified_bytecode_files"],
        "data_rights": "NOT_APPLICABLE_CODE",
        "model_weights": "NONE",
    }


def load_trusted_base_checkpoint(
    directory: str | Path,
    *,
    lineage_bytes: bytes,
    trusted_genesis_bytes: bytes,
    expected_genesis_sha256: str,
    expected_lineage_sha256: str,
    model: Any,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    strict_model: bool = True,
    restore_rng: bool = True,
) -> Any:
    """Canonical Base restore facade: attest before the first target mutation.

    No second checkpoint loader is introduced. The ordinary checkpoint loader
    remains a generic unqualified checkpoint API, not Base provenance evidence.
    """
    verified = prepare_trusted_base_checkpoint(
        directory,
        lineage_bytes=lineage_bytes,
        trusted_genesis_bytes=trusted_genesis_bytes,
        expected_genesis_sha256=expected_genesis_sha256,
        expected_lineage_sha256=expected_lineage_sha256,
    )
    from .checkpoint import load_verified_checkpoint

    return load_verified_checkpoint(
        verified,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        strict_model=strict_model,
        restore_rng=restore_rng,
    )
