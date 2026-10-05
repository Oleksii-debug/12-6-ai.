"""Additive trusted-family semantics for newly converged bounded code sources.

V1 remains authoritative and byte-for-byte compatible for its original 21-family
universe.  V2 adds semantic family/stratum identities for six independently
qualified bounded code-source authorities.  This module grants no corpus byte
credit, balance success, tokenizer fitting, model training, or evaluation use.

The extension identities bind only family semantics to exact repository authority
Git blobs.  Physical survivor membership and byte capacity must still come from a
separate authenticated post-G05/G06 inventory before balance/family-cap logic may
consume a family.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from twelve_six.data import trusted_family_authority_v1 as v1

SCHEMA = "12-6.d03-trusted-family-authority-extension.v2"
EXTENSION_AUTHORITY_SPECS: dict[str, dict[str, str]] = {
    "github:pydantic/pydantic": {
        "authority_path": "configs/data/next100_048_pydantic_code_rights_v1.json",
        "authority_git_blob_sha1": "504ef934145ed0711743f781dc9f47b07ad7accd",
    },
    "code.scipy.project": {
        "authority_path": "configs/data/scipy_v118_source_authority_v1.json",
        "authority_git_blob_sha1": "8bd1b020e324d33fdbcdda8619fbae7e73224d7e",
    },
    "github:pandas-dev/pandas": {
        "authority_path": "configs/data/next100_050_pandas_source_authority_v2.json",
        "authority_git_blob_sha1": "a97ccc1fcf097970abc84218e0fbd8088fa32887",
    },
    "github:fastapi/typer": {
        "authority_path": "configs/data/next100_052_typer_source_authority_v2.json",
        "authority_git_blob_sha1": "fc9168f2752d46fae76092999f449f0897fe0ca7",
    },
    "github:Textualize/rich": {
        "authority_path": "configs/data/next100_051_rich_code_rights_v1.json",
        "authority_git_blob_sha1": "4b4160814ddb97cb47bf45b4af2ed1b9ce8fef9e",
    },
    "github:fastapi/fastapi": {
        "authority_path": "configs/data/next100_044_fastapi_code_rights_policy_v1.json",
        "authority_git_blob_sha1": "8ee76ccc2ca3ff40d7e3d6463670d99e49051b44",
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
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()  # noqa: S324


def _extension_family_identity(
    family: str,
    authority: Mapping[str, str],
) -> str:
    return _sha256(
        {
            "schema": SCHEMA,
            "source_family": family,
            "canonical_stratum": "code",
            "authority_path": authority["authority_path"],
            "authority_git_blob_sha1": authority["authority_git_blob_sha1"],
        }
    )


def _extension_row(family: str) -> dict[str, Any]:
    authority = EXTENSION_AUTHORITY_SPECS[family]
    return {
        "family": family,
        "source_family_identity_sha256": _extension_family_identity(
            family,
            authority,
        ),
        "language": "und",
        "modalities": ["code"],
        "stratum": "code",
    }


TRUSTED_FAMILY_SEMANTICS: dict[str, dict[str, Any]] = {
    family: dict(row)
    for family, row in v1.TRUSTED_FAMILY_SEMANTICS.items()
}
TRUSTED_FAMILY_SEMANTICS.update(
    {
        family: _extension_row(family)
        for family in EXTENSION_AUTHORITY_SPECS
    }
)

if len(TRUSTED_FAMILY_SEMANTICS) != len(v1.TRUSTED_FAMILY_SEMANTICS) + 6:
    raise RuntimeError("trusted family V2 universe must add exactly six families")


def verify_extension_authority_blobs(
    repo_root: Path | None = None,
) -> dict[str, str]:
    """Verify the exact authority bytes backing all six V2 family semantics."""

    root = (
        Path(repo_root).resolve(strict=True)
        if repo_root is not None
        else Path(__file__).resolve(strict=True).parents[3]
    )
    observed: dict[str, str] = {}
    for family, authority in sorted(EXTENSION_AUTHORITY_SPECS.items()):
        path = root / authority["authority_path"]
        raw = path.read_bytes()
        blob = _git_blob_sha1(raw)
        expected = authority["authority_git_blob_sha1"]
        if blob != expected:
            raise ValueError(f"trusted family authority blob drift: {family}")
        observed[family] = blob
    return observed


def trusted_family_projection(
    families: Iterable[str],
) -> list[dict[str, Any]]:
    requested = set(families)
    unknown = requested - set(TRUSTED_FAMILY_SEMANTICS)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"survivor family absent from trusted V2 authority: {names}")
    if requested <= set(v1.TRUSTED_FAMILY_SEMANTICS):
        return v1.trusted_family_projection(requested)
    return [
        dict(TRUSTED_FAMILY_SEMANTICS[family])
        for family in sorted(requested)
    ]


def trusted_family_authority_root_sha256(
    families: Iterable[str],
) -> str:
    requested = set(families)
    unknown = requested - set(TRUSTED_FAMILY_SEMANTICS)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"survivor family absent from trusted V2 authority: {names}")
    base_families = requested & set(v1.TRUSTED_FAMILY_SEMANTICS)
    extension_families = requested & set(EXTENSION_AUTHORITY_SPECS)
    if not extension_families:
        return v1.trusted_family_authority_root_sha256(base_families)

    extension_authorities = [
        {
            "family": family,
            **EXTENSION_AUTHORITY_SPECS[family],
        }
        for family in sorted(extension_families)
    ]
    return _sha256(
        {
            "schema": SCHEMA,
            "base_family_authority_root_sha256": (
                v1.trusted_family_authority_root_sha256(base_families)
            ),
            "extension_authorities": extension_authorities,
            "survivor_family_projection": trusted_family_projection(requested),
        }
    )
