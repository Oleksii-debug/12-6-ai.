"""Project exact PR #474 attrs source bytes into the incumbent D03 dedup contract.

This module owns no duplicate-matching science and grants no corpus capacity.  It only
turns the four exact attrs 26.1.0 implementation objects admitted by PR #474 into the
source-row/payload shape consumed by the incumbent NEXT100-065 V3 dedup lineage.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

SOURCE_FAMILY: Final = "github:python-attrs/attrs"
SOURCE_ID: Final = "code.python-attrs.attrs.core-v1"
RELEASE_TAG: Final = "26.1.0"
UPSTREAM_COMMIT: Final = "7bfc49e9b22d5ba25b6e429524c3d49fee27cb36"
UPSTREAM_TREE: Final = "31beb3550ee7198eba22b862471ad6ea7bfb16d2"
UPSTREAM_PRODUCT_PR: Final = 474
UPSTREAM_PRODUCT_HEAD: Final = "cda0232d5574ef91eae0d7e0b7fa5efddcbe218b"
TERMINAL_AUTHORITY_SHA256: Final = (
    "151e593c3b67ae4c7686323983e6c45306a870b732573ee4820c0c017b65a7d4"
)
UPSTREAM_WORKFLOW_RUN_ID: Final = 33_006_080_831
UPSTREAM_ARTIFACT_ID: Final = 9_621_650_719
UPSTREAM_ARTIFACT_DIGEST: Final = (
    "a8176b50a2254fcb50a6f80ca82b63459ba8e9cfddba904b16e5ac79f9c55ff2"
)
LICENSE_GIT_BLOB_SHA1: Final = "2bd6453d255e19b973f19b128596a8b6dd65b2c3"
ADMITTED_OBJECT_COUNT: Final = 4
ADMITTED_SOURCE_BYTES: Final = 170_435
INCUMBENT_INVENTORY_SCHEMA: Final = "12-6.next100-065-cross-source-dedup.v3"
RECEIPT_SCHEMA: Final = "12-6.d03-attrs-dedup-intake-receipt.v1"
SWARM_ISSUE: Final = 1835


@dataclass(frozen=True)
class FileAuthority:
    path: str
    git_blob_sha1: str
    size_bytes: int


FILE_AUTHORITIES: Final = (
    FileAuthority(
        path="src/attr/_make.py",
        git_blob_sha1="4b32d6a71b0d91f3c4eb9ae615771aa46cae00eb",
        size_bytes=106_129,
    ),
    FileAuthority(
        path="src/attr/_funcs.py",
        git_blob_sha1="1adb50021373d9c09fcb9db0641bbc03248d54a3",
        size_bytes=16_479,
    ),
    FileAuthority(
        path="src/attr/validators.py",
        git_blob_sha1="0b1a294432d294c4f154be2d9439d825c3ec0781",
        size_bytes=21_553,
    ),
    FileAuthority(
        path="src/attr/_next_gen.py",
        git_blob_sha1="4ccd0da2446dc126ce936b054581a527e247cabc",
        size_bytes=26_274,
    ),
)


class AttrsDedupIntakeError(RuntimeError):
    """Fail-closed attrs dedup-intake contract error."""


@dataclass(frozen=True)
class AttrsDedupProjection:
    receipt: dict[str, Any]
    sources: tuple[dict[str, Any], ...]
    payloads: dict[str, bytes]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AttrsDedupIntakeError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    prefix = b"blob " + str(len(payload)).encode("ascii") + b"\0"
    return hashlib.sha1(prefix + payload).hexdigest()  # noqa: S324 - Git object identity


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _authority_ref() -> str:
    return (
        f"PR{UPSTREAM_PRODUCT_PR}:{UPSTREAM_PRODUCT_HEAD}:"
        f"{TERMINAL_AUTHORITY_SHA256}"
    )


def _source_id(path: str) -> str:
    return f"attrs-{RELEASE_TAG}:{path}"


def _matcher_row(authority: FileAuthority, payload: bytes) -> dict[str, Any]:
    payload_sha256 = _sha256(payload)
    source_id = _source_id(authority.path)
    return {
        "source_id": source_id,
        "source_family": SOURCE_FAMILY,
        "stable_origin_id": f"{SOURCE_FAMILY}@{UPSTREAM_COMMIT}:{authority.path}",
        "stable_object_id": f"sha256:{payload_sha256}",
        "modality": "code",
        "evidence_status": "DEDICATED_TERMINAL",
        "authority_ref": _authority_ref(),
        "declared_capacity_bytes": len(payload),
        "expected_raw_bytes": len(payload),
        "expected_raw_sha256": payload_sha256,
        "acquisition_url": (
            "https://raw.githubusercontent.com/python-attrs/attrs/"
            f"{UPSTREAM_COMMIT}/{authority.path}"
        ),
        "origin_key": f"{SOURCE_FAMILY}:{authority.path}",
    }


def _validate_payload(
    authority: FileAuthority,
    payload: bytes,
    *,
    index: int,
) -> None:
    _require(type(payload) is bytes, f"payload {index} must be exact bytes")
    _require(len(payload) == authority.size_bytes, f"payload {index} byte size drift")
    _require(
        _git_blob_sha1(payload) == authority.git_blob_sha1,
        f"payload {index} Git blob identity drift",
    )
    try:
        payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AttrsDedupIntakeError(f"payload {index} is not strict UTF-8") from exc


def _projection_receipt(
    *,
    sources: Sequence[Mapping[str, Any]],
    payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    source_lines = b"".join(_canonical(dict(source)) + b"\n" for source in sources)
    payload_ledger = [
        {
            "source_id": source["source_id"],
            "path": authority.path,
            "bytes": len(payloads[source["source_id"]]),
            "sha256": _sha256(payloads[source["source_id"]]),
            "git_blob_sha1": authority.git_blob_sha1,
        }
        for authority, source in zip(FILE_AUTHORITIES, sources, strict=True)
    ]
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "swarm_issue": SWARM_ISSUE,
        "local_free_only": True,
        "upstream_authority": {
            "product_pr": UPSTREAM_PRODUCT_PR,
            "product_head": UPSTREAM_PRODUCT_HEAD,
            "terminal_authority_sha256": TERMINAL_AUTHORITY_SHA256,
            "workflow_run_id": UPSTREAM_WORKFLOW_RUN_ID,
            "artifact_id": UPSTREAM_ARTIFACT_ID,
            "artifact_digest_sha256": UPSTREAM_ARTIFACT_DIGEST,
            "source_id": SOURCE_ID,
            "source_family": SOURCE_FAMILY,
            "release_tag": RELEASE_TAG,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_tree": UPSTREAM_TREE,
            "license_git_blob_sha1": LICENSE_GIT_BLOB_SHA1,
        },
        "projection": {
            "incumbent_inventory_schema": INCUMBENT_INVENTORY_SCHEMA,
            "object_count": len(sources),
            "payload_bytes": sum(len(payload) for payload in payloads.values()),
            "inventory_rows_sha256": _sha256(source_lines),
            "payload_ledger_sha256": _sha256(_canonical(payload_ledger)),
            "payload_ledger": payload_ledger,
            "matching_science_owned_here": False,
            "base_inventory_composition_owned_here": False,
        },
        "downstream_gates": {
            "incumbent_global_cross_source_dedup": "NOT_RUN",
            "reserved_evaluation_decontamination": "NOT_RUN",
            "post_composition_quality_privacy": "NOT_RUN",
            "balance_family_caps": "NOT_RUN",
            "cluster_safe_split": "NOT_RUN",
            "deterministic_packing": "NOT_RUN",
            "two_clean_builds": "NOT_RUN",
            "postpack_unique_loss_accounting": "NOT_RUN",
        },
        "truth_boundary": {
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates_executed_on_real_targets": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights": False,
            "external_llm_api_data_or_intelligence": False,
        },
    }
    receipt["receipt_identity_sha256"] = _sha256(_canonical(receipt))
    return receipt


def _project_payloads_for_authority(
    payloads_by_path: Mapping[str, bytes],
    authorities: Sequence[FileAuthority],
) -> AttrsDedupProjection:
    """Project payloads under an explicit authority vector; public API fixes that vector."""
    _require(type(payloads_by_path) is dict, "payload mapping must be exact dict")
    expected_paths = [authority.path for authority in authorities]
    _require(len(set(expected_paths)) == len(expected_paths), "authority paths are not unique")
    _require(
        set(payloads_by_path) == set(expected_paths),
        "payload path set differs from exact authority",
    )

    sources: list[dict[str, Any]] = []
    projected_payloads: dict[str, bytes] = {}
    for index, authority in enumerate(authorities):
        payload = payloads_by_path[authority.path]
        _validate_payload(authority, payload, index=index)
        row = _matcher_row(authority, payload)
        source_id = row["source_id"]
        _require(source_id not in projected_payloads, f"duplicate source id: {source_id}")
        sources.append(row)
        projected_payloads[source_id] = payload

    receipt = _projection_receipt(sources=sources, payloads=projected_payloads)
    return AttrsDedupProjection(
        receipt=receipt,
        sources=tuple(sources),
        payloads=projected_payloads,
    )


def project_attrs_payloads(payloads_by_path: Mapping[str, bytes]) -> AttrsDedupProjection:
    """Project the exact four PR #474 attrs payloads into incumbent V3 source rows."""
    projection = _project_payloads_for_authority(payloads_by_path, FILE_AUTHORITIES)
    _require(
        len(projection.sources) == ADMITTED_OBJECT_COUNT,
        "projected object count differs from terminal authority",
    )
    _require(
        sum(len(payload) for payload in projection.payloads.values()) == ADMITTED_SOURCE_BYTES,
        "projected byte total differs from terminal authority",
    )
    return projection


def project_attrs_root(source_root: Path) -> AttrsDedupProjection:
    """Read each exact attrs file once from a pinned checkout and project it."""
    _require(type(source_root) is Path, "source_root must be exact Path")
    try:
        resolved_root = source_root.resolve(strict=True)
    except OSError as exc:
        raise AttrsDedupIntakeError(f"cannot resolve source root: {source_root}") from exc
    _require(resolved_root.is_dir(), "source root is not a directory")

    payloads: dict[str, bytes] = {}
    for authority in FILE_AUTHORITIES:
        candidate = source_root / authority.path
        _require(not candidate.is_symlink(), f"source file is symlink: {authority.path}")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise AttrsDedupIntakeError(
                f"cannot resolve source file: {authority.path}"
            ) from exc
        _require(
            resolved.is_relative_to(resolved_root),
            f"source path escapes source root: {authority.path}",
        )
        _require(resolved.is_file(), f"source path is not a file: {authority.path}")
        try:
            with resolved.open("rb") as handle:
                payloads[authority.path] = handle.read()
        except OSError as exc:
            raise AttrsDedupIntakeError(
                f"cannot read source file: {authority.path}"
            ) from exc
    return project_attrs_payloads(payloads)


def validate_projection(projection: AttrsDedupProjection) -> None:
    """Revalidate the durable projection receipt and payload coverage fail closed."""
    _require(type(projection) is AttrsDedupProjection, "projection type drift")
    _require(len(projection.sources) == ADMITTED_OBJECT_COUNT, "projection row count drift")
    _require(
        set(projection.payloads) == {row.get("source_id") for row in projection.sources},
        "projection payload coverage drift",
    )
    rebuilt = project_attrs_payloads(
        {
            authority.path: projection.payloads[_source_id(authority.path)]
            for authority in FILE_AUTHORITIES
        }
    )
    _require(
        _canonical(rebuilt.sources) == _canonical(projection.sources),
        "projection source rows drift",
    )
    _require(
        _canonical(rebuilt.receipt) == _canonical(projection.receipt),
        "projection receipt drift",
    )
