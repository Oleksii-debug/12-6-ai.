"""Backward-compatible trusted-family authority including Franko1901.

The current 27-family authority remains immutable. This successor adds exactly one
already-physical, already-deduplicated, post-DATA-232/G05/G06 Ukrainian family:
ua.verba.public-domain.franko1901.

For any request that does not include Franko1901, projection and root are delegated
to the current authority unchanged. This module grants no capacity, tokenizer, or
training authority; it authenticates family identity/stratum semantics only.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from twelve_six.data import trusted_family_authority_current as parent

PARENT_CURRENT_MODULE_BLOB_SHA1 = "faa23fb0802d99465835748cf8db0f173ef6abeb"
MAX_AUTHORITY_FILE_BYTES = 1_048_576

FRANKO_FAMILY = "ua.verba.public-domain.franko1901"
FRANKO_AUTHORITY: dict[str, str] = {
    "path": "configs/data/d03_franko1901_terminal_execution_v1.json",
    "blob_sha1": "35902a3e6724d3f45ffb2578b34b9dddd51c76ab",
    "stratum": "uk",
    "family_field": "source.source_family",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}".encode("ascii") + bytes((0,))
    return hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_authority_bytes(relative_path: str) -> bytes:
    path = _repo_root() / relative_path
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            f"trusted Franko authority is not a regular file: {relative_path}"
        )
    size = path.stat().st_size
    if size <= 0 or size > MAX_AUTHORITY_FILE_BYTES:
        raise ValueError(f"trusted Franko authority size is invalid: {relative_path}")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError(
            f"trusted Franko authority changed while reading: {relative_path}"
        )
    return raw


def _verify_git_blob(relative_path: str, expected_blob_sha1: str) -> bytes:
    raw = _read_authority_bytes(relative_path)
    actual = _git_blob_sha1(raw)
    if actual != expected_blob_sha1:
        raise ValueError(f"trusted authority Git blob drift: {relative_path}")
    return raw


def _declared_family(raw: bytes, field_path: str, relative_path: str) -> str:
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"trusted authority JSON invalid: {relative_path}") from exc
    if not isinstance(document, Mapping):
        raise ValueError(f"trusted authority root invalid: {relative_path}")
    value: Any = document
    for field in field_path.split("."):
        if not isinstance(value, Mapping) or field not in value:
            raise ValueError(
                f"trusted authority family field missing: {relative_path}"
            )
        value = value[field]
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"trusted authority family field invalid: {relative_path}"
        )
    return value


def _source_family_identity(family: str, meta: Mapping[str, str]) -> str:
    return _sha256(
        {
            "authority_git_blob_sha1": meta["blob_sha1"],
            "source_family": family,
            "canonical_stratum": meta["stratum"],
        }
    )


def _franko_row() -> dict[str, Any]:
    return {
        "family": FRANKO_FAMILY,
        "source_family_identity_sha256": _source_family_identity(
            FRANKO_FAMILY, FRANKO_AUTHORITY
        ),
        "language": "uk",
        "modalities": ["text"],
        "stratum": "uk",
    }


TRUSTED_FAMILY_SEMANTICS: dict[str, dict[str, Any]] = copy.deepcopy(
    parent.TRUSTED_FAMILY_SEMANTICS
)
TRUSTED_FAMILY_SEMANTICS[FRANKO_FAMILY] = _franko_row()

if len(parent.TRUSTED_FAMILY_SEMANTICS) != 27:
    raise RuntimeError("parent current trusted family universe drift")
if len(TRUSTED_FAMILY_SEMANTICS) != 28:
    raise RuntimeError("Franko trusted family universe must contain exactly 28 families")

FRANKO_AUTHORITY_CHAIN: dict[str, Any] = {
    "schema": "12-6.d03-trusted-family-authority-chain.v3",
    "parent": {
        "module_git_blob_sha1": PARENT_CURRENT_MODULE_BLOB_SHA1,
        "authority_chain": parent.CURRENT_AUTHORITY_CHAIN,
        "family_count": len(parent.TRUSTED_FAMILY_SEMANTICS),
    },
    "added_source_authority": {
        "family": FRANKO_FAMILY,
        "path": FRANKO_AUTHORITY["path"],
        "blob_sha1": FRANKO_AUTHORITY["blob_sha1"],
        "stratum": FRANKO_AUTHORITY["stratum"],
        "family_field": FRANKO_AUTHORITY["family_field"],
    },
    "family_count_total": len(TRUSTED_FAMILY_SEMANTICS),
    "truth_boundary": {
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
    },
}


def _verify_franko_authority() -> None:
    parent_path = "src/twelve_six/data/trusted_family_authority_current.py"
    _verify_git_blob(parent_path, PARENT_CURRENT_MODULE_BLOB_SHA1)
    raw = _verify_git_blob(
        FRANKO_AUTHORITY["path"],
        FRANKO_AUTHORITY["blob_sha1"],
    )
    declared = _declared_family(
        raw,
        FRANKO_AUTHORITY["family_field"],
        FRANKO_AUTHORITY["path"],
    )
    if declared != FRANKO_FAMILY:
        raise ValueError("trusted Franko authority family declaration drift")


def trusted_family_projection(families: Iterable[str]) -> list[dict[str, Any]]:
    requested = set(families)
    if FRANKO_FAMILY not in requested:
        return parent.trusted_family_projection(requested)

    unknown = requested - set(TRUSTED_FAMILY_SEMANTICS)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(
            f"survivor family absent from trusted Franko authority: {names}"
        )

    _verify_franko_authority()
    parent_families = requested - {FRANKO_FAMILY}
    parent_rows = (
        parent.trusted_family_projection(parent_families)
        if parent_families
        else []
    )
    rows = parent_rows + [copy.deepcopy(TRUSTED_FAMILY_SEMANTICS[FRANKO_FAMILY])]
    return sorted(rows, key=lambda row: row["family"])


def trusted_family_authority_root_sha256(families: Iterable[str]) -> str:
    requested = set(families)
    if FRANKO_FAMILY not in requested:
        return parent.trusted_family_authority_root_sha256(requested)

    projection = trusted_family_projection(requested)
    return _sha256(
        {
            "authority_chain": FRANKO_AUTHORITY_CHAIN,
            "survivor_family_projection": projection,
        }
    )
