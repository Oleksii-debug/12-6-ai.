"""Immutable source-family authority for the current DATA526/V8 learned-20M graph.

The candidate family map/provenance documents are projections only.  Trust comes from
repository-pinned authority identities that predate the candidate invocation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

V6_BLOB_SHA1 = "13789effe506a815e92e4f0e22ada773d366f316"
V6_REGISTRY_IDENTITY_SHA256 = "c7b988081a270cd53c37f22721499568ec44f67daa998c87573b43c463642eae"
V5_BLOB_SHA1 = "2dcc57cfba8ab6d600bc431a8713f7b8e305dcbf"
V4_BLOB_SHA1 = "60924a7cb76dc76bbff26a340184f54a2c374c83"
V4_REGISTRY_IDENTITY_SHA256 = "9fc400a3144b46c481e45d043b0a3365eb2129c83bbacde6f9e7af8a41fadc58"
DEDUP_PARENT_BLOB_SHA1 = "c1e05f09490e25f6fed765dfb70d900717528f4d"
PR818_EVIDENCE_BLOB_SHA1 = "b2b9b361dbf0e6c883dca4d7f006812c69cdd4e0"
PR818_REPORT_IDENTITY_SHA256 = "80f3a8f20dfb82825e6c89ac1f76f2f41233296b8a426a1cf466b0fcc0892985"
V8_BLOB_SHA1 = "51d72c0448c4197807780e38b3cbbb94be36ba96"

TRUSTED_AUTHORITY_CHAIN: dict[str, Any] = {
    "schema": "12-6.d03-trusted-family-authority-chain.v1",
    "v6": {
        "blob_sha1": V6_BLOB_SHA1,
        "registry_identity_sha256": V6_REGISTRY_IDENTITY_SHA256,
        "prior_v5_blob_sha1": V5_BLOB_SHA1,
        "pr818_evidence_blob_sha1": PR818_EVIDENCE_BLOB_SHA1,
        "pr818_report_identity_sha256": PR818_REPORT_IDENTITY_SHA256,
    },
    "v4": {
        "blob_sha1": V4_BLOB_SHA1,
        "registry_identity_sha256": V4_REGISTRY_IDENTITY_SHA256,
        "dedup_parent_blob_sha1": DEDUP_PARENT_BLOB_SHA1,
    },
    "v8": {
        "blob_sha1": V8_BLOB_SHA1,
        "family_counts": {"uk": 4, "en": 5, "code": 12},
        "family_count_total": 21,
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


def _derived_dedup_parent_family_identity(family: str, stratum: str) -> str:
    return _sha256(
        {
            "authority_git_blob_sha1": DEDUP_PARENT_BLOB_SHA1,
            "source_family": family,
            "canonical_stratum": stratum,
        }
    )


def _row(
    family: str,
    *,
    stratum: str,
    source_family_identity_sha256: str,
) -> dict[str, Any]:
    if stratum == "code":
        language = "und"
        modalities = ["code"]
    else:
        language = stratum
        modalities = ["text"]
    return {
        "family": family,
        "source_family_identity_sha256": source_family_identity_sha256,
        "language": language,
        "modalities": modalities,
        "stratum": stratum,
    }


_BASE = {
    "ua.rada.open-data.laws-texts": "uk",
    "ua.literature.lesia-ukrainka.na-krylah-pisen.1892-lviv": "uk",
    "en.standardebooks.manual": "en",
    "github:encode/httpx": "code",
    "github:psf/requests": "code",
    "github:django/django": "code",
    "github:Kludex/starlette": "code",
}

TRUSTED_FAMILY_SEMANTICS: dict[str, dict[str, Any]] = {
    family: _row(
        family,
        stratum=stratum,
        source_family_identity_sha256=_derived_dedup_parent_family_identity(
            family, stratum
        ),
    )
    for family, stratum in _BASE.items()
}

TRUSTED_FAMILY_SEMANTICS.update(
    {
        "ua.kmu.portal.secretariat-news": _row(
            "ua.kmu.portal.secretariat-news",
            stratum="uk",
            source_family_identity_sha256="1f068e6cc5ce3fc4a51d8477acee31fab5a0178e15f49225b57de94c5178f7d9",
        ),
        "ua.verba.public-domain.nomis1864": _row(
            "ua.verba.public-domain.nomis1864",
            stratum="uk",
            source_family_identity_sha256="85f596e79b0ec6479d2ef815e2a6a9bdbfaa55993c797309c1ea4d93b1d9b0e7",
        ),
        "en.mdn.webdocs.prose": _row(
            "en.mdn.webdocs.prose",
            stratum="en",
            source_family_identity_sha256="0f5dbd5313f8196811e2a99f77eb8698c6bc69f69648d76a7e240ee9757ecc47",
        ),
        "en.usgov.nist.technical-series": _row(
            "en.usgov.nist.technical-series",
            stratum="en",
            source_family_identity_sha256="3ffba0fcd08ab42e940b2db12ffafb6f7234ad0bae6f7fe523071497485b9d1c",
        ),
        "github:numpy/numpy": _row(
            "github:numpy/numpy",
            stratum="code",
            source_family_identity_sha256="e9d2ce633915d6b6844b35e4abb0188974ef4791b208362c4f106ec0ad79ca70",
        ),
        "python.cpython.documentation": _row(
            "python.cpython.documentation",
            stratum="en",
            source_family_identity_sha256="46a00dc70db690ae2b3c4495a75283e7e752bdccb1047d4318c2ebadfa392f0d",
        ),
        "en.project-gutenberg.public-domain-books": _row(
            "en.project-gutenberg.public-domain-books",
            stratum="en",
            source_family_identity_sha256="1b1bad11b688826ee4f73701c08e3b5af76ba16e8d8a806e008d5b84bee0b97b",
        ),
        "github:python-attrs/attrs": _row(
            "github:python-attrs/attrs",
            stratum="code",
            source_family_identity_sha256="151e593c3b67ae4c7686323983e6c45306a870b732573ee4820c0c017b65a7d4",
        ),
        "github:pallets/flask": _row(
            "github:pallets/flask",
            stratum="code",
            source_family_identity_sha256="0adcaf8045583380a295fea5cad42457bb686958fa5b1d986700eb9bd927d08b",
        ),
        "github:pallets/click": _row(
            "github:pallets/click",
            stratum="code",
            source_family_identity_sha256="1e35efcc2c2036e3d886b011cc95553a94a41df88f0f8c36e737378fae119313",
        ),
        "github:pallets/jinja": _row(
            "github:pallets/jinja",
            stratum="code",
            source_family_identity_sha256="35a988c002ecfe09311d91eb4a887c34977a8a4e933a883ef43914d2e4898f1b",
        ),
        "github:pallets/werkzeug": _row(
            "github:pallets/werkzeug",
            stratum="code",
            source_family_identity_sha256="ea4219d2e62645bae613d5b34d42b7579a5cc9553a85631643c7b45cec080fda",
        ),
        "github:agronholm/anyio": _row(
            "github:agronholm/anyio",
            stratum="code",
            source_family_identity_sha256="62f81f9602b1f795536036e70a1c8ca5498b4f8969239ac92f5de77a846f477d",
        ),
        "github:pytest-dev/pytest": _row(
            "github:pytest-dev/pytest",
            stratum="code",
            source_family_identity_sha256="9a7437e4486af8e25295e19058bf27551bf04e4a7a2a22ba7cf31c2a1cfe37ab",
        ),
    }
)

if len(TRUSTED_FAMILY_SEMANTICS) != 21:
    raise RuntimeError("trusted DATA526/V8 family universe must contain exactly 21 families")


def trusted_family_projection(families: Iterable[str]) -> list[dict[str, Any]]:
    requested = set(families)
    unknown = requested - set(TRUSTED_FAMILY_SEMANTICS)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"survivor family absent from trusted DATA526/V8 authority: {names}")
    return [
        dict(TRUSTED_FAMILY_SEMANTICS[family])
        for family in sorted(requested)
    ]


def trusted_family_authority_root_sha256(families: Iterable[str]) -> str:
    projection = trusted_family_projection(families)
    return _sha256(
        {
            "authority_chain": TRUSTED_AUTHORITY_CHAIN,
            "survivor_family_projection": projection,
        }
    )
