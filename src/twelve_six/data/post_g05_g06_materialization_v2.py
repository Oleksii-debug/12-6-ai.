"""Provenance-guarded successor to post-G05/G06 materialization V1.

V1 remains historical evidence code. V2 composes the exact V1 byte transform with
an independently authenticated external-LLM provenance quarantine. It intentionally
does not claim whole-corpus external-LLM cleanliness; it proves only that the known
Nomis1864 contaminated payload/source identity is absent from the input graph.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from twelve_six.data.external_llm_provenance_quarantine_v1 import (
    EXPECTED_AUTHORITY_IDENTITY_SHA256,
    ExternalLLMProvenanceQuarantineError,
    reject_quarantined_records,
)
from twelve_six.data.post_g05_g06_materialization_v1 import materialize_post_g05_g06

SCHEMA = "12-6.d03-post-g05-g06-materialization.v2"


class PostG05G06MaterializationV2Error(ValueError):
    """Raised when V2 provenance or implementation binding fails closed."""


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise PostG05G06MaterializationV2Error(message)


def _cjson(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(raw)}\0".encode("ascii") + raw,
        usedforsecurity=False,
    ).hexdigest()


def _implementation_blob(expected_git_blob_sha1: str) -> str:
    _need(
        isinstance(expected_git_blob_sha1, str)
        and len(expected_git_blob_sha1) == 40
        and all(ch in "0123456789abcdef" for ch in expected_git_blob_sha1),
        "expected V2 materializer implementation Git blob must be lowercase 40-hex",
    )
    try:
        raw = Path(__file__).resolve(strict=True).read_bytes()
    except OSError as exc:
        raise PostG05G06MaterializationV2Error(
            "cannot resolve running V2 materializer implementation"
        ) from exc
    actual = _git_blob_sha1(raw)
    _need(actual == expected_git_blob_sha1, "V2 materializer implementation Git blob drift")
    return actual


def _upgrade_evidence(
    evidence_v1: Mapping[str, Any],
    *,
    quarantine_identity_sha256: str,
    implementation_blob_sha1: str,
) -> dict[str, Any]:
    evidence = copy.deepcopy(dict(evidence_v1))
    base_identity = evidence.pop("materialization_identity_sha256", None)
    _need(
        isinstance(base_identity, str) and len(base_identity) == 64,
        "V1 materialization identity missing",
    )
    truth = evidence.get("truth_boundary")
    _need(isinstance(truth, dict), "V1 truth boundary missing")
    _need(
        truth.pop("external_llm_or_api_used_for_data_or_intelligence", None) is False,
        "V1 external-LLM truth field drift",
    )
    input_authority = evidence.get("input")
    _need(isinstance(input_authority, dict), "V1 input authority missing")
    input_authority["external_llm_provenance_quarantine_identity_sha256"] = (
        quarantine_identity_sha256
    )
    evidence["schema_version"] = SCHEMA
    evidence["materializer_v2_implementation_git_blob_sha1"] = implementation_blob_sha1
    evidence["base_v1_materialization_identity_sha256"] = base_identity
    evidence["provenance_guard"] = {
        "known_external_llm_contamination_absent": True,
        "quarantine_identity_sha256": quarantine_identity_sha256,
        "whole_corpus_external_llm_cleanliness_claimed": False,
        "successor_authority_rebuild_required_for_invalidated_v5_v6_v8": True,
    }
    evidence["remaining_materialization_blockers"] = sorted(
        set(evidence.get("remaining_materialization_blockers", []))
        | {"SUCCESSOR_CORPUS_AUTHORITY_REBUILD_REQUIRED"}
    )
    core = dict(evidence)
    evidence["materialization_identity_sha256"] = _sha256(_cjson(core))
    return evidence


def materialize_post_g05_g06_v2(
    *,
    records: Sequence[Mapping[str, Any]],
    provenance_quarantine_authority: Mapping[str, Any],
    expected_provenance_quarantine_identity_sha256: str = (
        EXPECTED_AUTHORITY_IDENTITY_SHA256
    ),
    expected_materializer_v2_implementation_git_blob_sha1: str,
    **v1_kwargs: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Run the exact V1 transform only after the known provenance quarantine passes."""
    implementation_blob = _implementation_blob(
        expected_materializer_v2_implementation_git_blob_sha1
    )
    try:
        quarantine_identity = reject_quarantined_records(
            records,
            provenance_quarantine_authority,
            expected_identity_sha256=expected_provenance_quarantine_identity_sha256,
        )
    except ExternalLLMProvenanceQuarantineError as exc:
        raise PostG05G06MaterializationV2Error(str(exc)) from exc

    output, inventory, evidence_v1 = materialize_post_g05_g06(
        records=records,
        **v1_kwargs,
    )
    evidence_v2 = _upgrade_evidence(
        evidence_v1,
        quarantine_identity_sha256=quarantine_identity,
        implementation_blob_sha1=implementation_blob,
    )
    return output, inventory, evidence_v2
