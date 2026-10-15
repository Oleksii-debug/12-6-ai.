"""Independent Plan-2 S18 bundle integrity preflight, NOT release authority.

This checks bytes and cross-artifact identities of a proposed Plan-2 -> Plan-9
handoff. It cannot validate legal rights, upstream S15-S17 acceptance, or
authorize training. An external trusted SHA-256 pin is mandatory to avoid
self-signed bundle promotion. This intentionally has no 'approve' mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

SCHEMA = "12-6.plan2.s18.integrity-preflight.v1"
MANIFEST = "release-manifest.json"
REQUIRED = frozenset({
    "dataset-manifest.json", "tokenizer-manifest.json", "shards-manifest.json",
    "exposure-manifest.json", "rights-attribution.json", "plan9-binding.json",
})
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SHA1 = re.compile(r"[0-9a-f]{40}\Z")
MAX_MEMBERS = 10000
MAX_JSON_BYTES = 2 * 1024 * 1024


class IntegrityDenied(ValueError):
    """Missing, changed, untrusted, or inconsistent handoff evidence."""


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise IntegrityDenied(reason)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _regular_file(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise IntegrityDenied("missing or unreadable member") from exc
    _require(stat.S_ISREG(mode), "non-regular or symlink member")


def _read_regular(path: Path, *, limit: int | None = None) -> bytes:
    _regular_file(path)
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as stream:
            st = os.fstat(stream.fileno())
            _require(stat.S_ISREG(st.st_mode), "non-regular opened member")
            if limit is not None:
                _require(st.st_size <= limit, "oversized JSON member")
                raw = stream.read(limit + 1)
                _require(len(raw) <= limit, "oversized JSON member")
                return raw
            return stream.read()
    except OSError as exc:
        raise IntegrityDenied("unreadable or replaced member") from exc


def _json(path: Path) -> dict[str, Any]:
    raw = _read_regular(path, limit=MAX_JSON_BYTES)
    try:
        def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for key, value in pairs:
                _require(key not in out, "duplicate JSON key")
                out[key] = value
            return out
        doc = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                         parse_constant=lambda _: (_ for _ in ()).throw(
                             IntegrityDenied("nonfinite JSON value")))
    except IntegrityDenied:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise IntegrityDenied("invalid JSON member") from exc
    _require(type(doc) is dict, "noncanonical JSON member")
    try:
        canonical = _canonical(doc)
    except (ValueError, RecursionError) as exc:
        raise IntegrityDenied("invalid JSON member") from exc
    _require(raw == canonical, "noncanonical JSON member")
    return doc


def _sha256_file(path: Path) -> str:
    _regular_file(path)
    digest = hashlib.sha256()
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            _require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode),
                     "non-regular opened member")
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise IntegrityDenied("unreadable or replaced member") from exc
    return digest.hexdigest()


def _name(value: Any) -> str:
    _require(type(value) is str and bool(value) and len(value) <= 240,
             "invalid member path")
    _require(value not in {".", ".."} and not value.startswith("/")
             and "\\" not in value and "\x00" not in value,
             "unsafe member path")
    parts = value.split("/")
    _require(all(part not in {"", ".", ".."} for part in parts),
             "unsafe member path")
    return value


def _check_member(root: Path, name: str) -> Path:
    path = root
    for component in name.split("/")[:-1]:
        path = path / component
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise IntegrityDenied("missing member parent") from exc
        _require(stat.S_ISDIR(mode), "symlink or non-directory member parent")
    return path / name.split("/")[-1]


def _bind(doc: dict[str, Any], expected: dict[str, str]) -> None:
    for key, digest in expected.items():
        _require(doc.get(key) == digest, "cross-artifact identity mismatch: " + key)


def verify(root: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    """Verify pinned physical bytes; always return NON-AUTHORIZING evidence."""
    _require(type(expected_manifest_sha256) is str
             and SHA256.fullmatch(expected_manifest_sha256) is not None,
             "trusted external manifest SHA-256 pin required")
    root = Path(root)
    try:
        _require(stat.S_ISDIR(root.lstat().st_mode), "unsafe release root")
    except OSError as exc:
        raise IntegrityDenied("missing release root") from exc
    manifest_path = root / MANIFEST
    _require(_sha256_file(manifest_path) == expected_manifest_sha256,
             "trusted manifest SHA-256 mismatch")
    manifest = _json(manifest_path)
    _require(set(manifest) == {"schema_version", "producer_commit_sha1",
                               "producer_tree_sha1", "members"},
             "invalid release manifest fields")
    _require(manifest["schema_version"] == SCHEMA, "unknown release manifest schema")
    _require(type(manifest["producer_commit_sha1"]) is str
             and SHA1.fullmatch(manifest["producer_commit_sha1"]) is not None,
             "invalid producer commit")
    _require(type(manifest["producer_tree_sha1"]) is str
             and SHA1.fullmatch(manifest["producer_tree_sha1"]) is not None,
             "invalid producer tree")
    members = manifest["members"]
    _require(type(members) is dict and REQUIRED.issubset(members)
             and len(members) <= MAX_MEMBERS, "incomplete or oversized member inventory")
    for key, value in members.items():
        _name(key)
        _require(key != MANIFEST and type(value) is str
                 and SHA256.fullmatch(value) is not None,
                 "invalid member digest")
    def _check_inventory() -> None:
        observed: set[str] = set()
        visited_directories = 0
        permitted_directories = {"/".join(name.split("/")[:i])
                                 for name in members
                                 for i in range(1, len(name.split("/")))}
        for directory, dirs, files in os.walk(root, topdown=True, followlinks=False):
            visited_directories += 1
            _require(visited_directories <= MAX_MEMBERS + 1, "oversized directory inventory")
            parent = Path(directory)
            for name in dirs:
                path = parent / name
                _require(stat.S_ISDIR(path.lstat().st_mode),
                         "symlink or non-directory member parent")
                _require(path.relative_to(root).as_posix() in permitted_directories,
                         "unlisted directory")
            for name in files:
                path = parent / name
                _regular_file(path)
                observed.add(path.relative_to(root).as_posix())
            _require(len(observed) <= MAX_MEMBERS + 1, "oversized actual inventory")
        _require(observed == set(members) | {MANIFEST}, "unlisted or missing member")

    _check_inventory()
    for name, expected in members.items():
        _require(_sha256_file(_check_member(root, name)) == expected,
                 "member SHA-256 mismatch: " + name)
    dataset = members["dataset-manifest.json"]
    tokenizer = members["tokenizer-manifest.json"]
    shards = members["shards-manifest.json"]
    exposure = members["exposure-manifest.json"]
    rights = members["rights-attribution.json"]
    _bind(_json(root / "shards-manifest.json"),
          {"dataset_sha256": dataset, "tokenizer_sha256": tokenizer})
    _bind(_json(root / "exposure-manifest.json"), {"shards_sha256": shards})
    _bind(_json(root / "rights-attribution.json"), {"dataset_sha256": dataset})
    binding = _json(root / "plan9-binding.json")
    _bind(binding, {"dataset_sha256": dataset, "tokenizer_sha256": tokenizer,
                    "shards_sha256": shards, "exposure_sha256": exposure,
                    "rights_sha256": rights})
    _require(binding.get("campaign_authorized") is False,
             "Plan9 campaign authority must remain independent")
    # Recheck after parsing to detect concurrent mutation during the preflight.
    for name, expected in members.items():
        _require(_sha256_file(_check_member(root, name)) == expected,
                 "member changed during verification: " + name)
    _require(_sha256_file(manifest_path) == expected_manifest_sha256,
             "manifest changed during verification")
    # The initial inventory predates hashing and JSON parsing. Detect late
    # unlisted files or directories without claiming an atomic filesystem view.
    _check_inventory()
    return {
        "schema_version": SCHEMA,
        "decision": "BYTE_INTEGRITY_ONLY_NOT_RELEASE",
        "trusted_manifest_sha256": expected_manifest_sha256,
        "producer_commit_sha1": manifest["producer_commit_sha1"],
        "member_count": len(members),
        "member_sha256": dict(sorted(members.items())),
        "production_release_authorized": False,
        "plan9_campaign_authorized": False,
        "terminal_done": False,
        "remaining_independent_gates": [
            "S15_exact_accepted_main_and_candidate_qualification",
            "S16_independent_rights_privacy_UA_EN_holdouts_decontamination",
            "S17_physical_clean_rebuild_and_exposure_qualification",
            "S18_external_source_provenance_and_license_review",
            "S18_Plan9_real_consumer_acceptance_and_accepted_main_readback",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan2 S18 integrity-only preflight")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trusted-manifest-sha256", required=True)
    args = parser.parse_args()
    try:
        report = verify(args.root, args.trusted_manifest_sha256)
    except IntegrityDenied as exc:
        print(json.dumps({"decision": "DENIED", "reason": str(exc),
                          "production_release_authorized": False}, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
