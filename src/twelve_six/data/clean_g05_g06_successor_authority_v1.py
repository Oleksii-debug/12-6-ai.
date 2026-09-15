"""Fail-closed clean G05/G06 successor authority preparation.

The physical Nomis-free source/global-dedup/DATA526 successor is integrated and
can be bound as an upstream source anchor. Downstream retained-inventory,
reserved-decontamination, G05/G06 execution, and two independent replay
authorities do not yet have an integrated physical trust root here.

Accordingly this module can emit only PREPARED_NOT_EXECUTED evidence. It has no
code path that turns caller-supplied hashes or self-sealed candidate documents
into a terminal corpus/G05/G06 authority.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from twelve_six.data.final_g05_g06_coverage_v1 import (
    QUALITY_GRANULARITY_IDENTITY_SHA256,
    QUALITY_POLICY_IDENTITY_SHA256,
)

SCHEMA = "12-6.d03-clean-g05-g06-successor-authority.v2"
SOURCE_ANCHOR_SCHEMA = "12-6.d03-nomis-free-integrated-source-anchor.v1"
AUTHORITY_STATUS = "PREPARED_NOT_EXECUTED"
PROVENANCE_SCOPE = "WHOLE_CORPUS_EXTERNAL_LLM_CLEANLINESS_NOT_CLAIMED"

INTEGRATED_SOURCE_MERGE_SHA = "5e8ced70926a2e7f14aad984cf552f2dc646aa59"
SOURCE_RELEASE_HEAD_SHA = "07754c5a1d61669061e608323ea35ddc093bb946"
PHYSICAL_EXECUTION_HEAD_SHA = "3d7dd363f6b1701694c00c76c6353077f80d1492"
PHYSICAL_RUN_ID = 34911721640
PHYSICAL_JOB_ID = 104200523132
PHYSICAL_ARTIFACT_ID = 10374891614
PHYSICAL_ARTIFACT_ZIP_SHA256 = (
    "663eec03d04252b6de574bf97ea0975703e0ba25779ca27340cb3acea941943a"
)
PHYSICAL_SOURCE_REPORT_SHA256 = (
    "db72eb1d4f86cd025741efd0c612c1f2e124ce24dfa133548c375d781331c93c"
)
PHYSICAL_SURVIVOR_AUTHORITY_SHA256 = (
    "e1c94f5eed4afa78a63d577fe33a67e28305c1e084c27cd1061061ad113f8ce5"
)
PHYSICAL_DATA526_EVIDENCE_IDENTITY_SHA256 = (
    "45ac421c9e4a5930330c1516af79e1b67d0535fc90b768ecfe9df54dd0a86c5c"
)

NOMIS_RECORD_ID = "ua.verba.nomis1864.bounded24"
NOMIS_FAMILY = "ua.verba.public-domain.nomis1864"
NOMIS_PAYLOAD_SHA256 = (
    "1eb91dbd631898c6a2efe274b700a5be0deaca243c0a9d5d30994ddadcf43598"
)
PR462_AUTHORITY_SHA256 = (
    "85f596e79b0ec6479d2ef815e2a6a9bdbfaa55993c797309c1ea4d93b1d9b0e7"
)

_REQUIRED_DOWNSTREAM_ROOTS = (
    "retained_inventory_identity_sha256",
    "records_jsonl_sha256",
    "decontamination_authority_sha256",
    "g05_g06_coverage_identity_sha256",
    "privacy_policy_identity_sha256",
    "privacy_implementation_git_blob_sha",
    "qualification_authority_set_identity_sha256",
    "replay_execution_authority_a_sha256",
    "replay_execution_authority_b_sha256",
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")

_ZERO_FALSE_FIELDS = (
    "whole_corpus_external_llm_cleanliness_claimed",
    "current_retained_corpus_launch_authoritative",
    "tokenizer_fit_authorized",
    "training_executed",
    "learned_weights_created",
    "final_test_outcomes_read",
    "paid_compute_used",
    "foreign_pretrained_weights",
)
_ZERO_INT_FIELDS = (
    "authorized_optimized_target_exposure",
    "optimizer_updates_executed_on_real_targets",
)

_CLEAN_CANDIDATE_BOOL_FIELDS = (
    "nomis1864_admitted",
    "pr462_authority_admitted",
    "nomis1864_quarantine_enforced",
    "whole_corpus_external_llm_cleanliness_claimed",
    "current_retained_corpus_launch_authoritative",
    "tokenizer_fit_authorized",
    "training_executed",
    "learned_weights_created",
    "final_test_outcomes_read",
    "paid_compute_used",
    "foreign_pretrained_weights",
)
_REPLAY_COUNT_FIELDS = ("covered_record_count", "covered_payload_bytes")


class CleanG05G06SuccessorError(ValueError):
    """Raised when clean-successor authority cannot be proved fail-closed."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise CleanG05G06SuccessorError(message)


def _cjson(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _self_hash(document: Mapping[str, Any], field: str) -> str:
    body = dict(document)
    body.pop(field, None)
    return _sha(_cjson(body))


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise CleanG05G06SuccessorError(f"{field} must be lowercase SHA-256")
    return value


def _git_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        raise CleanG05G06SuccessorError(
            f"{field} must be a lowercase 40-hex Git object id"
        )
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CleanG05G06SuccessorError(f"{field} must be a positive integer")
    return value


def _exact_bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise CleanG05G06SuccessorError(f"{field} must be a boolean")
    return value


def _verify_zero_boundary(document: Mapping[str, Any], *, prefix: str) -> None:
    for field in _ZERO_FALSE_FIELDS:
        _require(document.get(field) is False, f"{prefix}.{field} must be false")
    for field in _ZERO_INT_FIELDS:
        value = document.get(field)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value == 0,
            f"{prefix}.{field} must be zero",
        )


def integrated_nomis_free_source_anchor() -> dict[str, Any]:
    """Return the fixed, non-caller-selected upstream physical source anchor."""
    anchor: dict[str, Any] = {
        "schema": SOURCE_ANCHOR_SCHEMA,
        "integrated_source_merge_sha": INTEGRATED_SOURCE_MERGE_SHA,
        "source_release_head_sha": SOURCE_RELEASE_HEAD_SHA,
        "physical_execution_head_sha": PHYSICAL_EXECUTION_HEAD_SHA,
        "physical_run_id": PHYSICAL_RUN_ID,
        "physical_job_id": PHYSICAL_JOB_ID,
        "physical_artifact_id": PHYSICAL_ARTIFACT_ID,
        "physical_artifact_zip_sha256": PHYSICAL_ARTIFACT_ZIP_SHA256,
        "physical_source_report_sha256": PHYSICAL_SOURCE_REPORT_SHA256,
        "physical_survivor_authority_sha256": (
            PHYSICAL_SURVIVOR_AUTHORITY_SHA256
        ),
        "physical_data526_evidence_identity_sha256": (
            PHYSICAL_DATA526_EVIDENCE_IDENTITY_SHA256
        ),
        "nomis_record_id": NOMIS_RECORD_ID,
        "nomis_family": NOMIS_FAMILY,
        "nomis_payload_sha256": NOMIS_PAYLOAD_SHA256,
        "pr462_authority_sha256": PR462_AUTHORITY_SHA256,
        "nomis1864_admitted": False,
        "pr462_authority_admitted": False,
        "nomis1864_quarantine_enforced": True,
        "current_retained_corpus_launch_authoritative": False,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
    }
    anchor["source_anchor_identity_sha256"] = _self_hash(
        anchor, "source_anchor_identity_sha256"
    )
    return anchor


def _verify_integrated_source_anchor(document: Mapping[str, Any]) -> None:
    expected = integrated_nomis_free_source_anchor()
    _require(
        set(document) == set(expected),
        "integrated source anchor schema drift",
    )
    for key, value in expected.items():
        if key == "source_anchor_identity_sha256":
            continue
        if type(value) is bool:
            _require(
                type(document.get(key)) is bool and document.get(key) is value,
                f"integrated source anchor drift: {key}",
            )
        else:
            _require(
                document.get(key) == value,
                f"integrated source anchor drift: {key}",
            )
    _require(
        document.get("source_anchor_identity_sha256")
        == _self_hash(document, "source_anchor_identity_sha256"),
        "integrated source anchor self-hash mismatch",
    )


def build_prepared_clean_g05_g06_successor_authority() -> dict[str, Any]:
    """Emit only the current prepared, zero-credit downstream authority state."""
    source_anchor = integrated_nomis_free_source_anchor()
    downstream_roots = {key: None for key in _REQUIRED_DOWNSTREAM_ROOTS}
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": AUTHORITY_STATUS,
        "provenance_scope": PROVENANCE_SCOPE,
        "terminal_authority_available": False,
        "source_anchor": source_anchor,
        "quality_policy_identity_sha256": QUALITY_POLICY_IDENTITY_SHA256,
        "quality_granularity_identity_sha256": (
            QUALITY_GRANULARITY_IDENTITY_SHA256
        ),
        "required_downstream_roots": downstream_roots,
        "whole_corpus_external_llm_cleanliness_claimed": False,
        "current_retained_corpus_launch_authoritative": False,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
    }
    result["successor_authority_identity_sha256"] = _self_hash(
        result, "successor_authority_identity_sha256"
    )
    return result


def verify_clean_g05_g06_successor_authority(document: Mapping[str, Any]) -> None:
    """Verify the only authority state currently reachable: prepared/not executed."""
    expected = build_prepared_clean_g05_g06_successor_authority()
    _require(set(document) == set(expected), "successor authority root schema drift")
    _require(document.get("schema") == SCHEMA, "successor authority schema drift")
    _require(
        document.get("status") == AUTHORITY_STATUS,
        "successor authority must remain PREPARED_NOT_EXECUTED",
    )
    _require(
        document.get("terminal_authority_available") is False,
        "terminal authority cannot be claimed",
    )
    _require(
        document.get("provenance_scope") == PROVENANCE_SCOPE,
        "successor provenance scope drift",
    )

    source_anchor = document.get("source_anchor")
    _require(isinstance(source_anchor, Mapping), "source anchor missing")
    _verify_integrated_source_anchor(source_anchor)

    _require(
        document.get("quality_policy_identity_sha256")
        == QUALITY_POLICY_IDENTITY_SHA256,
        "quality policy identity drift",
    )
    _require(
        document.get("quality_granularity_identity_sha256")
        == QUALITY_GRANULARITY_IDENTITY_SHA256,
        "quality granularity identity drift",
    )

    roots = document.get("required_downstream_roots")
    _require(isinstance(roots, Mapping), "required downstream roots missing")
    _require(
        set(roots) == set(_REQUIRED_DOWNSTREAM_ROOTS),
        "required downstream root schema drift",
    )
    _require(
        all(roots[key] is None for key in _REQUIRED_DOWNSTREAM_ROOTS),
        "untrusted downstream root cannot be promoted",
    )
    _verify_zero_boundary(document, prefix="successor")
    _require(
        document.get("successor_authority_identity_sha256")
        == _self_hash(document, "successor_authority_identity_sha256"),
        "successor authority self-hash mismatch",
    )


def _verify_nonterminal_candidate_scalar_types(
    clean_input_binding: Mapping[str, Any],
    replay_receipts: Sequence[Mapping[str, Any]],
) -> None:
    """Close type-alias defects even though terminal construction is unavailable."""
    for field in _CLEAN_CANDIDATE_BOOL_FIELDS:
        _exact_bool(clean_input_binding.get(field), f"clean_input.{field}")

    for index, receipt in enumerate(replay_receipts):
        for field in _REPLAY_COUNT_FIELDS:
            _positive_int(receipt.get(field), f"replay[{index}].{field}")


def build_clean_g05_g06_successor_authority(
    *,
    clean_input_binding: Mapping[str, Any],
    g05_g06_coverage: Mapping[str, Any],
    replay_receipts: Sequence[Mapping[str, Any]],
) -> NoReturn:
    """Refuse terminal authority until independently authenticated roots exist.

    Candidate documents are deliberately not allowed to provide their own
    ``expected_*`` values. The current integrated source anchor proves only the
    Nomis-free source/global-dedup/DATA526 successor. A later same-lineage
    change may make terminal construction reachable only after retained
    inventory, reserved decontamination, G05/G06 coverage, privacy, and two
    physical replay execution authorities are independently rooted.
    """
    _require(
        isinstance(clean_input_binding, Mapping),
        "clean input candidate must be a mapping",
    )
    _require(
        isinstance(g05_g06_coverage, Mapping),
        "G05/G06 coverage candidate must be a mapping",
    )
    _require(
        isinstance(replay_receipts, Sequence)
        and not isinstance(replay_receipts, (str, bytes))
        and len(replay_receipts) == 2
        and all(isinstance(item, Mapping) for item in replay_receipts),
        "exactly two replay receipt mappings are required",
    )
    _verify_nonterminal_candidate_scalar_types(clean_input_binding, replay_receipts)
    raise CleanG05G06SuccessorError(
        "terminal G05/G06 authority is unavailable: independently authenticated "
        "retained/decontamination/G05-G06/replay execution roots are not integrated"
    )


def require_terminal_root_shape(value: Any, field: str, *, git: bool = False) -> str:
    """Validate future trusted-carrier scalar shapes without granting trust."""
    if git:
        return _git_sha(value, field)
    return _sha256(value, field)
