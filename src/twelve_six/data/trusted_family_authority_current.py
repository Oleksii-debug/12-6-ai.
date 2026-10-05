"""Current source-family authority with backward-compatible D03 expansion.

The historical DATA526/V8 authority remains immutable. This module preserves its
projection/root byte-for-byte when a survivor set contains only V1 families. When
the current D03 graph contains one of the six newly qualified code families, it
extends that authority with repository-pinned source-authority Git blobs.

This grants no capacity or training credit by itself. It only authenticates family
identity/stratum semantics so the existing NEXT100-106 balance gate can evaluate an
already-physical, already-deduplicated, post-Q/P inventory.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from twelve_six.data.trusted_family_authority_v1 import (
    TRUSTED_AUTHORITY_CHAIN as V1_TRUSTED_AUTHORITY_CHAIN,
)
from twelve_six.data.trusted_family_authority_v1 import (
    TRUSTED_FAMILY_SEMANTICS as V1_TRUSTED_FAMILY_SEMANTICS,
)
from twelve_six.data.trusted_family_authority_v1 import (
    trusted_family_authority_root_sha256 as v1_trusted_family_authority_root_sha256,
)
from twelve_six.data.trusted_family_authority_v1 import (
    trusted_family_projection as v1_trusted_family_projection,
)

PARENT_V1_MODULE_BLOB_SHA1 = "9382332d2d0d5c09aa5582e6f949b1630c459cbb"
MAX_AUTHORITY_FILE_BYTES = 1_048_576

SOURCE_AUTHORITY_FILES: dict[str, dict[str, str]] = {
    "github:pydantic/pydantic": {
        "path": "configs/data/next100_048_pydantic_code_rights_v1.json",
        "blob_sha1": "504ef934145ed0711743f781dc9f47b07ad7accd",
        "stratum": "code",
        "family_field": "source_family",
    },
    "code.scipy.project": {
        "path": "configs/data/scipy_v118_source_authority_v1.json",
        "blob_sha1": "8bd1b020e324d33fdbcdda8619fbae7e73224d7e",
        "stratum": "code",
        "family_field": "source_family",
    },
    "github:pandas-dev/pandas": {
        "path": "configs/data/next100_050_pandas_source_authority_v2.json",
        "blob_sha1": "a97ccc1fcf097970abc84218e0fbd8088fa32887",
        "stratum": "code",
        "family_field": "bounded_source.source_family",
    },
    "github:fastapi/typer": {
        "path": "configs/data/next100_052_typer_source_authority_v2.json",
        "blob_sha1": "fc9168f2752d46fae76092999f449f0897fe0ca7",
        "stratum": "code",
        "family_field": "bounded_source.source_family",
    },
    "github:Textualize/rich": {
        "path": "configs/data/next100_051_rich_code_rights_v1.json",
        "blob_sha1": "4b4160814ddb97cb47bf45b4af2ed1b9ce8fef9e",
        "stratum": "code",
        "family_field": "source_family",
    },
    "github:fastapi/fastapi": {
        "path": "configs/data/next100_044_fastapi_code_rights_policy_v1.json",
        "blob_sha1": "8ee76ccc2ca3ff40d7e3d6463670d99e49051b44",
        "stratum": "code",
        "family_field": "upstream.canonical_family_id",
    },
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
            f"trusted source authority is not a regular file: {relative_path}"
        )
    size = path.stat().st_size
    if size <= 0 or size > MAX_AUTHORITY_FILE_BYTES:
        raise ValueError(f"trusted source authority size is invalid: {relative_path}")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError(
            f"trusted source authority changed while reading: {relative_path}"
        )
    return raw


def _verify_git_blob(relative_path: str, expected_blob_sha1: str) -> bytes:
    raw = _read_authority_bytes(relative_path)
    actual = _git_blob_sha1(raw)
    if actual != expected_blob_sha1:
        raise ValueError(f"trusted source authority Git blob drift: {relative_path}")
    return raw


def _declared_family(raw: bytes, field_path: str, relative_path: str) -> str:
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"trusted source authority JSON invalid: {relative_path}"
        ) from exc
    if not isinstance(document, Mapping):
        raise ValueError(f"trusted source authority root invalid: {relative_path}")
    value: Any = document
    for field in field_path.split("."):
        if not isinstance(value, Mapping) or field not in value:
            raise ValueError(
                f"trusted source authority family field missing: {relative_path}"
            )
        value = value[field]
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"trusted source authority family field invalid: {relative_path}"
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


def _row(family: str, meta: Mapping[str, str]) -> dict[str, Any]:
    stratum = meta["stratum"]
    if stratum == "code":
        language = "und"
        modalities = ["code"]
    else:
        language = stratum
        modalities = ["text"]
    return {
        "family": family,
        "source_family_identity_sha256": _source_family_identity(family, meta),
        "language": language,
        "modalities": modalities,
        "stratum": stratum,
    }


ADDED_FAMILY_SEMANTICS: dict[str, dict[str, Any]] = {
    family: _row(family, meta) for family, meta in SOURCE_AUTHORITY_FILES.items()
}

TRUSTED_FAMILY_SEMANTICS: dict[str, dict[str, Any]] = copy.deepcopy(
    V1_TRUSTED_FAMILY_SEMANTICS
)
TRUSTED_FAMILY_SEMANTICS.update(copy.deepcopy(ADDED_FAMILY_SEMANTICS))

if len(V1_TRUSTED_FAMILY_SEMANTICS) != 21:
    raise RuntimeError("parent trusted family universe drift")
if len(TRUSTED_FAMILY_SEMANTICS) != 27:
    raise RuntimeError("current trusted family universe must contain exactly 27 families")

CURRENT_AUTHORITY_CHAIN: dict[str, Any] = {
    "schema": "12-6.d03-trusted-family-authority-chain.v2",
    "parent": {
        "module_git_blob_sha1": PARENT_V1_MODULE_BLOB_SHA1,
        "authority_chain": V1_TRUSTED_AUTHORITY_CHAIN,
        "family_count": len(V1_TRUSTED_FAMILY_SEMANTICS),
    },
    "added_source_authorities": {
        family: {
            "path": meta["path"],
            "blob_sha1": meta["blob_sha1"],
            "stratum": meta["stratum"],
            "family_field": meta["family_field"],
        }
        for family, meta in sorted(SOURCE_AUTHORITY_FILES.items())
    },
    "family_count_total": len(TRUSTED_FAMILY_SEMANTICS),
}


def _verify_extended_authority_files(families: set[str]) -> None:
    added = families & set(SOURCE_AUTHORITY_FILES)
    if not added:
        return
    parent_path = "src/twelve_six/data/trusted_family_authority_v1.py"
    _verify_git_blob(parent_path, PARENT_V1_MODULE_BLOB_SHA1)
    for family in sorted(added):
        meta = SOURCE_AUTHORITY_FILES[family]
        raw = _verify_git_blob(meta["path"], meta["blob_sha1"])
        declared = _declared_family(raw, meta["family_field"], meta["path"])
        if declared != family:
            raise ValueError(
                f"trusted source authority family declaration drift: {meta['path']}"
            )


def trusted_family_projection(families: Iterable[str]) -> list[dict[str, Any]]:
    requested = set(families)
    unknown = requested - set(TRUSTED_FAMILY_SEMANTICS)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(
            f"survivor family absent from trusted current authority: {names}"
        )

    if not (requested & set(SOURCE_AUTHORITY_FILES)):
        return v1_trusted_family_projection(requested)

    _verify_extended_authority_files(requested)
    return [
        copy.deepcopy(TRUSTED_FAMILY_SEMANTICS[family])
        for family in sorted(requested)
    ]


def trusted_family_authority_root_sha256(families: Iterable[str]) -> str:
    requested = set(families)
    if not (requested & set(SOURCE_AUTHORITY_FILES)):
        return v1_trusted_family_authority_root_sha256(requested)

    projection = trusted_family_projection(requested)
    return _sha256(
        {
            "authority_chain": CURRENT_AUTHORITY_CHAIN,
            "survivor_family_projection": projection,
        }
    )
