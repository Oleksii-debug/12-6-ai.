"""Fail-closed composition of terminal learned-20M launch-input authorities.

This module does not tokenize, pack, count source bytes as loss positions, authorize
training, or inspect final-test payloads. It binds independently expected identities
from the canonical D03/D04/D10 authorities, the authenticated deterministic-double-
pack proof, and the canonical V2 two-clean verifier.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from twelve_six.data.deterministic_double_pack import DOUBLE_PACK_PROOF_SCHEMA
from twelve_six.data.unique_loss_ledger_v2 import (
    LEDGER_SCHEMA,
    POSITION_POLICY,
    REQUIRED_STAGE_BINDINGS,
)
from twelve_six.packing.two_clean_build import TwoCleanBuildError, verify_proof

REPOSITORY = "Oleksii-debug/12-6-ai."
LAUNCH_INPUT_SCHEMA = "12-6.learned20m-launch-input-authority.v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

_AUTHORITY_KEYS = frozenset(
    {
        "schema_version",
        "binding_status",
        "data_spine",
        "carrier",
        "claim_boundary",
        "authority_identity_sha256",
    }
)
_DATA_SPINE_KEYS = frozenset(
    {
        "terminal_corpus_authority_identity_sha256",
        "stage_bindings",
        "deterministic_double_pack_proof_identity_sha256",
        "terminal_record_inventory_digest_sha256",
        "terminal_payload_inventory_digest_sha256",
        "terminal_split_application_identity_sha256",
        "terminal_split_spec_identity_sha256",
        "terminal_split_train_record_membership_sha256",
        "canonical_build_sha256",
        "two_clean_proof_identity_sha256",
        "two_clean_input_packet_identity_sha256",
        "two_clean_runtime_identity_sha256",
        "materialization_identity_sha256",
        "unique_loss_ledger_identity_sha256",
        "tokenizer_identity_sha256",
        "packing_identity_sha256",
        "one_pass_unique_nonignored_causal_loss_positions",
        "requested_unique_loss_positions",
    }
)
_DOUBLE_PACK_PROOF_KEYS = frozenset(
    {
        "schema_version",
        "terminal_corpus_authority_identity_sha256",
        "terminal_record_inventory_digest_sha256",
        "terminal_payload_inventory_digest_sha256",
        "terminal_split_application_identity_sha256",
        "terminal_split_spec_identity_sha256",
        "terminal_split_train_record_membership_sha256",
        "stage_bindings",
        "tokenizer_identity_sha256",
        "materialization_identity_sha256",
        "packing_identity_sha256",
        "ledger_identity_sha256",
        "canonical_build_sha256",
        "build_a_canonical_sha256",
        "build_b_canonical_sha256",
        "one_pass_unique_nonignored_causal_loss_positions",
        "retained_train_records_matched_to_terminal_inventory",
        "retained_train_record_membership_verified",
        "retained_document_isolation_verified",
        "heldout_reservation_verified",
        "independent_builds_byte_identical",
        "training_authorized_by_this_proof",
        "proof_identity_sha256",
    }
)
_CARRIER_OUTPUT_KEYS = frozenset(
    {
        "repository",
        "git_sha",
        "modelspec_sha256",
        "initialization_identity_sha256",
        "canonical_base",
        "foreign_pretrained_weights_used",
        "terminal",
        "workflow_run_id",
        "workflow_status",
        "workflow_conclusion",
        "workflow_head_sha",
        "evidence_sha256",
    }
)
_CLAIM_BOUNDARY = {
    "contains_source_text": False,
    "final_test_payload_consumed": False,
    "authorizes_training": False,
    "authorizes_compute": False,
    "authorized_optimized_target_exposure": 0,
    "replay_padding_or_replacement_can_increase_unique_capacity": False,
}


class LaunchInputAuthorityError(ValueError):
    """Raised when the terminal launch-input chain cannot be trusted."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_obj(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise LaunchInputAuthorityError(f"{field} must be exact lowercase SHA-256")
    return value


def _require_git_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise LaunchInputAuthorityError(f"{field} must be exact lowercase Git SHA")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LaunchInputAuthorityError(f"{field} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LaunchInputAuthorityError(f"{field} must be a non-negative integer")
    return value


def _normalize_stage_bindings(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(REQUIRED_STAGE_BINDINGS):
        raise LaunchInputAuthorityError(
            f"{field} must contain exactly the canonical five stage bindings"
        )
    return {
        name: _require_sha256(value[name], f"{field}.{name}")
        for name in REQUIRED_STAGE_BINDINGS
    }


def _verify_self_hash(
    value: Mapping[str, Any],
    *,
    identity_field: str,
    expected_identity_sha256: str,
    label: str,
) -> dict[str, Any]:
    expected = _require_sha256(
        expected_identity_sha256,
        f"expected_{label}_identity_sha256",
    )
    payload = dict(value)
    observed = _require_sha256(payload.get(identity_field), identity_field)
    body = dict(payload)
    body.pop(identity_field, None)
    if _sha256_obj(body) != observed:
        raise LaunchInputAuthorityError(f"{label} self-identity mismatch")
    if observed != expected:
        raise LaunchInputAuthorityError(
            f"{label} does not match independently expected identity"
        )
    return payload


def _verify_two_clean_proof(
    proof: Mapping[str, Any],
    *,
    expected_identity_sha256: str,
    expected_input_packet_identity_sha256: str,
    expected_corpus_identity_sha256: str,
    expected_stage_bindings: Mapping[str, str],
    expected_tokenizer_identity_sha256: str,
    expected_packing_identity_sha256: str,
    expected_runtime_identity_sha256: str,
) -> dict[str, Any]:
    """Delegate the V2 closed-world freshness contract to its canonical verifier."""
    try:
        return verify_proof(
            proof,
            expected_proof_identity_sha256=expected_identity_sha256,
            expected_input_packet_identity_sha256=(
                expected_input_packet_identity_sha256
            ),
            expected_terminal_corpus_identity_sha256=expected_corpus_identity_sha256,
            expected_stage_bindings=expected_stage_bindings,
            expected_tokenizer_identity_sha256=expected_tokenizer_identity_sha256,
            expected_packing_identity_sha256=expected_packing_identity_sha256,
            expected_runtime_identity_sha256=expected_runtime_identity_sha256,
        )
    except TwoCleanBuildError as exc:
        raise LaunchInputAuthorityError(
            f"canonical two-clean proof rejected: {exc}"
        ) from exc


def _verify_ledger(
    ledger: Mapping[str, Any],
    *,
    expected_identity_sha256: str,
    expected_materialization_identity_sha256: str,
    expected_stage_bindings: Mapping[str, str],
    expected_tokenizer_identity_sha256: str,
    expected_packing_identity_sha256: str,
) -> tuple[dict[str, Any], int]:
    value = _verify_self_hash(
        ledger,
        identity_field="ledger_identity_sha256",
        expected_identity_sha256=expected_identity_sha256,
        label="unique_loss_ledger",
    )
    if value.get("schema_version") != LEDGER_SCHEMA:
        raise LaunchInputAuthorityError("unexpected unique-loss ledger schema")
    if value.get("position_policy") != POSITION_POLICY:
        raise LaunchInputAuthorityError("unique-loss position policy drift")

    materialization = _require_sha256(
        value.get("materialization_identity_sha256"),
        "ledger.materialization_identity_sha256",
    )
    expected_materialization = _require_sha256(
        expected_materialization_identity_sha256,
        "expected_materialization_identity_sha256",
    )
    if materialization != expected_materialization:
        raise LaunchInputAuthorityError("ledger materialization substitution")

    expected_bindings = _normalize_stage_bindings(
        expected_stage_bindings, "expected_stage_bindings"
    )
    observed_bindings = _normalize_stage_bindings(
        value.get("stage_bindings"), "ledger.stage_bindings"
    )
    if observed_bindings != expected_bindings:
        raise LaunchInputAuthorityError("ledger stage authority substitution")

    tokenizer = value.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise LaunchInputAuthorityError("ledger tokenizer authority missing")
    tokenizer_identity = _require_sha256(
        tokenizer.get("identity_sha256"), "ledger.tokenizer.identity_sha256"
    )
    expected_tokenizer = _require_sha256(
        expected_tokenizer_identity_sha256,
        "expected_tokenizer_identity_sha256",
    )
    if tokenizer_identity != expected_tokenizer:
        raise LaunchInputAuthorityError("tokenizer authority substitution")
    if tokenizer.get("source_bytes_are_loss_positions") is not False:
        raise LaunchInputAuthorityError("source bytes cannot be loss positions")

    packing_identity = _require_sha256(
        value.get("packing_identity_sha256"), "packing_identity_sha256"
    )
    expected_packing = _require_sha256(
        expected_packing_identity_sha256,
        "expected_packing_identity_sha256",
    )
    if packing_identity != expected_packing:
        raise LaunchInputAuthorityError("packing authority substitution")

    if value.get("complete_one_pass") is not True:
        raise LaunchInputAuthorityError(
            "unique-loss ledger is not a complete one-pass ledger"
        )
    if _require_nonnegative_int(
        value.get("eligible_targets_not_packed"), "eligible_targets_not_packed"
    ) != 0:
        raise LaunchInputAuthorityError("eligible causal targets remain unpacked")
    if _require_nonnegative_int(
        value.get("padding_loss_positions"), "padding_loss_positions"
    ) != 0:
        raise LaunchInputAuthorityError("padding cannot create loss positions")
    if _require_nonnegative_int(
        value.get("cross_document_loss_positions"), "cross_document_loss_positions"
    ) != 0:
        raise LaunchInputAuthorityError(
            "cross-document positions are not unique authority"
        )
    if value.get("source_bytes_relabelled_as_loss_positions") is not False:
        raise LaunchInputAuthorityError(
            "source bytes cannot be relabelled as loss positions"
        )

    positions = _require_positive_int(
        value.get("one_pass_unique_nonignored_causal_loss_positions"),
        "one_pass_unique_nonignored_causal_loss_positions",
    )
    segments = value.get("segments")
    if not isinstance(segments, list) or not segments:
        raise LaunchInputAuthorityError("ledger segments must be a non-empty list")

    counted = 0
    segment_ids: set[str] = set()
    logical_ranges: set[tuple[str, int, int]] = set()
    ranges_by_document: dict[str, list[tuple[int, int]]] = {}
    for index, raw_segment in enumerate(segments):
        if not isinstance(raw_segment, Mapping):
            raise LaunchInputAuthorityError(f"segments[{index}] must be an object")
        segment_id = _require_sha256(
            raw_segment.get("segment_identity_sha256"),
            f"segments[{index}].segment_identity_sha256",
        )
        if segment_id in segment_ids:
            raise LaunchInputAuthorityError(
                "duplicate segment identity in unique-loss ledger"
            )
        segment_ids.add(segment_id)

        document_id = raw_segment.get("document_id")
        start = raw_segment.get("target_start")
        end = raw_segment.get("target_end")
        if not isinstance(document_id, str) or not document_id:
            raise LaunchInputAuthorityError(
                f"segments[{index}].document_id must be non-empty"
            )
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 1
            or end <= start
        ):
            raise LaunchInputAuthorityError(
                f"segments[{index}] target range is invalid"
            )

        logical_range = (document_id, start, end)
        if logical_range in logical_ranges:
            raise LaunchInputAuthorityError(
                "duplicate logical range in unique-loss ledger"
            )
        logical_ranges.add(logical_range)
        ranges_by_document.setdefault(document_id, []).append((start, end))

        count = _require_positive_int(
            raw_segment.get("loss_position_count"),
            f"segments[{index}].loss_position_count",
        )
        if count != end - start:
            raise LaunchInputAuthorityError(
                "segment loss count does not match target range"
            )
        counted += count

    for document_ranges in ranges_by_document.values():
        document_ranges.sort()
        previous_end: int | None = None
        for start, end in document_ranges:
            if previous_end is not None and start < previous_end:
                raise LaunchInputAuthorityError(
                    "overlapping logical ranges in unique-loss ledger"
                )
            previous_end = end

    if counted != positions:
        raise LaunchInputAuthorityError(
            "segment counts do not match unique-loss capacity"
        )
    return value, positions


def _verify_deterministic_double_pack_proof(
    proof: Any,
    *,
    expected_identity_sha256: Any,
    expected_terminal_corpus_identity_sha256: str,
    expected_stage_bindings: Mapping[str, str],
    expected_tokenizer_identity_sha256: str,
    expected_packing_identity_sha256: str,
    expected_materialization_identity_sha256: str,
    expected_ledger_identity_sha256: str,
    expected_unique_positions: int,
) -> dict[str, Any]:
    """Verify the canonical D04 proof and cross-bind it to the live D10 chain.

    D10 does not possess the producer's raw builds/split application, so it cannot
    rerun ``verify_deterministic_double_pack``. Instead it verifies the producer's
    closed-world proof self-hash against an independently supplied proof identity,
    then binds every behavior-bearing root to the already-authenticated launch
    inputs. A self-consistent replacement proof therefore cannot authorize itself.
    """
    if not isinstance(proof, Mapping):
        raise LaunchInputAuthorityError(
            "deterministic double-pack proof is required for launch binding"
        )
    value = dict(proof)
    if set(value) != set(_DOUBLE_PACK_PROOF_KEYS):
        raise LaunchInputAuthorityError(
            "deterministic double-pack proof has unexpected or missing fields"
        )
    if value.get("schema_version") != DOUBLE_PACK_PROOF_SCHEMA:
        raise LaunchInputAuthorityError(
            "unexpected deterministic double-pack proof schema"
        )

    expected_identity = _require_sha256(
        expected_identity_sha256,
        "expected_deterministic_double_pack_proof_identity_sha256",
    )
    observed_identity = _require_sha256(
        value.get("proof_identity_sha256"),
        "deterministic_double_pack_proof.proof_identity_sha256",
    )
    body = dict(value)
    body.pop("proof_identity_sha256", None)
    if _sha256_obj(body) != observed_identity:
        raise LaunchInputAuthorityError(
            "deterministic double-pack proof self-identity mismatch"
        )
    if observed_identity != expected_identity:
        raise LaunchInputAuthorityError(
            "deterministic double-pack proof does not match independently expected identity"
        )

    corpus_identity = _require_sha256(
        value.get("terminal_corpus_authority_identity_sha256"),
        "deterministic_double_pack_proof.terminal_corpus_authority_identity_sha256",
    )
    expected_corpus = _require_sha256(
        expected_terminal_corpus_identity_sha256,
        "expected_terminal_corpus_authority_identity_sha256",
    )
    if corpus_identity != expected_corpus:
        raise LaunchInputAuthorityError(
            "deterministic double-pack terminal corpus substitution"
        )

    observed_bindings = _normalize_stage_bindings(
        value.get("stage_bindings"),
        "deterministic_double_pack_proof.stage_bindings",
    )
    expected_bindings = _normalize_stage_bindings(
        expected_stage_bindings, "expected_stage_bindings"
    )
    if observed_bindings != expected_bindings:
        raise LaunchInputAuthorityError(
            "deterministic double-pack stage authority substitution"
        )

    crossbinds = (
        (
            "tokenizer_identity_sha256",
            expected_tokenizer_identity_sha256,
            "tokenizer",
        ),
        (
            "packing_identity_sha256",
            expected_packing_identity_sha256,
            "packing",
        ),
        (
            "materialization_identity_sha256",
            expected_materialization_identity_sha256,
            "materialization",
        ),
        (
            "ledger_identity_sha256",
            expected_ledger_identity_sha256,
            "unique-loss ledger",
        ),
    )
    for field, expected_value, label in crossbinds:
        observed = _require_sha256(
            value.get(field), f"deterministic_double_pack_proof.{field}"
        )
        expected = _require_sha256(
            expected_value, f"expected_deterministic_double_pack_{field}"
        )
        if observed != expected:
            raise LaunchInputAuthorityError(
                f"deterministic double-pack {label} substitution"
            )

    for field in (
        "terminal_record_inventory_digest_sha256",
        "terminal_payload_inventory_digest_sha256",
        "terminal_split_application_identity_sha256",
        "terminal_split_spec_identity_sha256",
        "terminal_split_train_record_membership_sha256",
        "canonical_build_sha256",
        "build_a_canonical_sha256",
        "build_b_canonical_sha256",
    ):
        _require_sha256(value.get(field), f"deterministic_double_pack_proof.{field}")

    canonical_build = value["canonical_build_sha256"]
    if (
        value["build_a_canonical_sha256"] != canonical_build
        or value["build_b_canonical_sha256"] != canonical_build
    ):
        raise LaunchInputAuthorityError(
            "deterministic double-pack canonical build hashes differ"
        )

    proof_positions = _require_positive_int(
        value.get("one_pass_unique_nonignored_causal_loss_positions"),
        (
            "deterministic_double_pack_proof."
            "one_pass_unique_nonignored_causal_loss_positions"
        ),
    )
    if proof_positions != expected_unique_positions:
        raise LaunchInputAuthorityError(
            "deterministic double-pack unique-loss capacity substitution"
        )
    _require_positive_int(
        value.get("retained_train_records_matched_to_terminal_inventory"),
        (
            "deterministic_double_pack_proof."
            "retained_train_records_matched_to_terminal_inventory"
        ),
    )
    for field in (
        "retained_train_record_membership_verified",
        "retained_document_isolation_verified",
        "heldout_reservation_verified",
        "independent_builds_byte_identical",
    ):
        if value.get(field) is not True:
            raise LaunchInputAuthorityError(
                f"deterministic double-pack {field} must be exact true boolean"
            )
    if value.get("training_authorized_by_this_proof") is not False:
        raise LaunchInputAuthorityError(
            "deterministic double-pack proof cannot authorize training"
        )
    return value


def _verify_carrier(
    carrier: Mapping[str, Any],
    *,
    expected_git_sha: str,
    expected_modelspec_sha256: str,
    expected_initialization_sha256: str,
    expected_workflow_run_id: int,
    expected_evidence_sha256: str,
) -> dict[str, Any]:
    value = dict(carrier)
    if value.get("repository") != REPOSITORY:
        raise LaunchInputAuthorityError("carrier repository mismatch")

    expected_git = _require_git_sha(expected_git_sha, "expected_carrier_git_sha")
    git_sha = _require_git_sha(value.get("git_sha"), "carrier.git_sha")
    if git_sha != expected_git:
        raise LaunchInputAuthorityError("carrier Git SHA substitution")

    model = _require_sha256(
        value.get("modelspec_sha256"), "carrier.modelspec_sha256"
    )
    expected_model = _require_sha256(
        expected_modelspec_sha256, "expected_modelspec_sha256"
    )
    if model != expected_model:
        raise LaunchInputAuthorityError("ModelSpec authority substitution")

    initialization = _require_sha256(
        value.get("initialization_identity_sha256"),
        "carrier.initialization_identity_sha256",
    )
    expected_initialization = _require_sha256(
        expected_initialization_sha256,
        "expected_initialization_identity_sha256",
    )
    if initialization != expected_initialization:
        raise LaunchInputAuthorityError("initialization authority substitution")

    if value.get("canonical_base") != "random_init":
        raise LaunchInputAuthorityError(
            "carrier is not canonical random initialization"
        )
    if value.get("foreign_pretrained_weights_used") is not False:
        raise LaunchInputAuthorityError("foreign pretrained weights are forbidden")
    if value.get("terminal") is not True:
        raise LaunchInputAuthorityError("carrier authority is nonterminal")

    run_id = _require_positive_int(
        value.get("workflow_run_id"), "carrier.workflow_run_id"
    )
    expected_run_id = _require_positive_int(
        expected_workflow_run_id, "expected_carrier_workflow_run_id"
    )
    if run_id != expected_run_id:
        raise LaunchInputAuthorityError("carrier workflow run substitution")
    if value.get("workflow_status") != "completed":
        raise LaunchInputAuthorityError("carrier workflow is not terminal completed")
    if value.get("workflow_conclusion") != "success":
        raise LaunchInputAuthorityError(
            "carrier exact-head CI is not terminal success"
        )
    workflow_head = _require_git_sha(
        value.get("workflow_head_sha"), "carrier.workflow_head_sha"
    )
    if workflow_head != git_sha:
        raise LaunchInputAuthorityError(
            "carrier workflow head does not match exact carrier Git SHA"
        )

    evidence = _require_sha256(
        value.get("evidence_sha256"), "carrier.evidence_sha256"
    )
    expected_evidence = _require_sha256(
        expected_evidence_sha256,
        "expected_carrier_evidence_sha256",
    )
    if evidence != expected_evidence:
        raise LaunchInputAuthorityError("carrier evidence substitution")
    return value


def build_launch_input_authority(
    two_clean_proof: Mapping[str, Any],
    unique_loss_ledger: Mapping[str, Any],
    carrier_authority: Mapping[str, Any],
    *,
    expected_two_clean_proof_identity_sha256: str,
    expected_two_clean_input_packet_identity_sha256: str,
    expected_unique_loss_ledger_identity_sha256: str,
    expected_terminal_corpus_authority_identity_sha256: str,
    expected_stage_bindings: Mapping[str, str],
    expected_tokenizer_identity_sha256: str,
    expected_packing_identity_sha256: str,
    expected_runtime_identity_sha256: str,
    expected_carrier_git_sha: str,
    expected_modelspec_sha256: str,
    expected_initialization_identity_sha256: str,
    expected_carrier_workflow_run_id: int,
    expected_carrier_evidence_sha256: str,
    requested_unique_loss_positions: int,
    deterministic_double_pack_proof: Mapping[str, Any] | None = None,
    expected_deterministic_double_pack_proof_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind terminal upstream identities without authorizing an optimizer step."""
    freshness_proof = _verify_two_clean_proof(
        two_clean_proof,
        expected_identity_sha256=expected_two_clean_proof_identity_sha256,
        expected_input_packet_identity_sha256=(
            expected_two_clean_input_packet_identity_sha256
        ),
        expected_corpus_identity_sha256=(
            expected_terminal_corpus_authority_identity_sha256
        ),
        expected_stage_bindings=expected_stage_bindings,
        expected_tokenizer_identity_sha256=expected_tokenizer_identity_sha256,
        expected_packing_identity_sha256=expected_packing_identity_sha256,
        expected_runtime_identity_sha256=expected_runtime_identity_sha256,
    )
    ledger, one_pass_capacity = _verify_ledger(
        unique_loss_ledger,
        expected_identity_sha256=expected_unique_loss_ledger_identity_sha256,
        expected_materialization_identity_sha256=(
            freshness_proof["materialization_identity_sha256"]
        ),
        expected_stage_bindings=expected_stage_bindings,
        expected_tokenizer_identity_sha256=expected_tokenizer_identity_sha256,
        expected_packing_identity_sha256=expected_packing_identity_sha256,
    )
    membership_proof = _verify_deterministic_double_pack_proof(
        deterministic_double_pack_proof,
        expected_identity_sha256=(
            expected_deterministic_double_pack_proof_identity_sha256
        ),
        expected_terminal_corpus_identity_sha256=(
            expected_terminal_corpus_authority_identity_sha256
        ),
        expected_stage_bindings=expected_stage_bindings,
        expected_tokenizer_identity_sha256=expected_tokenizer_identity_sha256,
        expected_packing_identity_sha256=expected_packing_identity_sha256,
        expected_materialization_identity_sha256=(
            freshness_proof["materialization_identity_sha256"]
        ),
        expected_ledger_identity_sha256=ledger["ledger_identity_sha256"],
        expected_unique_positions=one_pass_capacity,
    )

    requested = _require_positive_int(
        requested_unique_loss_positions, "requested_unique_loss_positions"
    )
    if requested > one_pass_capacity:
        raise LaunchInputAuthorityError(
            "requested unique optimized-target exposure exceeds one-pass unique capacity"
        )

    carrier = _verify_carrier(
        carrier_authority,
        expected_git_sha=expected_carrier_git_sha,
        expected_modelspec_sha256=expected_modelspec_sha256,
        expected_initialization_sha256=expected_initialization_identity_sha256,
        expected_workflow_run_id=expected_carrier_workflow_run_id,
        expected_evidence_sha256=expected_carrier_evidence_sha256,
    )
    stage_bindings = _normalize_stage_bindings(
        expected_stage_bindings, "expected_stage_bindings"
    )

    authority: dict[str, Any] = {
        "schema_version": LAUNCH_INPUT_SCHEMA,
        "binding_status": "READY_FOR_READINESS_BINDING",
        "data_spine": {
            "terminal_corpus_authority_identity_sha256": (
                expected_terminal_corpus_authority_identity_sha256
            ),
            "stage_bindings": stage_bindings,
            "deterministic_double_pack_proof_identity_sha256": membership_proof[
                "proof_identity_sha256"
            ],
            "terminal_record_inventory_digest_sha256": membership_proof[
                "terminal_record_inventory_digest_sha256"
            ],
            "terminal_payload_inventory_digest_sha256": membership_proof[
                "terminal_payload_inventory_digest_sha256"
            ],
            "terminal_split_application_identity_sha256": membership_proof[
                "terminal_split_application_identity_sha256"
            ],
            "terminal_split_spec_identity_sha256": membership_proof[
                "terminal_split_spec_identity_sha256"
            ],
            "terminal_split_train_record_membership_sha256": membership_proof[
                "terminal_split_train_record_membership_sha256"
            ],
            "canonical_build_sha256": membership_proof["canonical_build_sha256"],
            "two_clean_proof_identity_sha256": freshness_proof[
                "proof_identity_sha256"
            ],
            "two_clean_input_packet_identity_sha256": freshness_proof[
                "input_packet_identity_sha256"
            ],
            "two_clean_runtime_identity_sha256": freshness_proof[
                "runtime_identity_sha256"
            ],
            "materialization_identity_sha256": freshness_proof[
                "materialization_identity_sha256"
            ],
            "unique_loss_ledger_identity_sha256": ledger[
                "ledger_identity_sha256"
            ],
            "tokenizer_identity_sha256": expected_tokenizer_identity_sha256,
            "packing_identity_sha256": expected_packing_identity_sha256,
            "one_pass_unique_nonignored_causal_loss_positions": one_pass_capacity,
            "requested_unique_loss_positions": requested,
        },
        "carrier": {
            "repository": carrier["repository"],
            "git_sha": carrier["git_sha"],
            "modelspec_sha256": carrier["modelspec_sha256"],
            "initialization_identity_sha256": carrier[
                "initialization_identity_sha256"
            ],
            "canonical_base": "random_init",
            "foreign_pretrained_weights_used": False,
            "terminal": True,
            "workflow_run_id": carrier["workflow_run_id"],
            "workflow_status": "completed",
            "workflow_conclusion": "success",
            "workflow_head_sha": carrier["workflow_head_sha"],
            "evidence_sha256": carrier["evidence_sha256"],
        },
        "claim_boundary": dict(_CLAIM_BOUNDARY),
    }
    authority["authority_identity_sha256"] = _sha256_obj(authority)
    return authority


def verify_launch_input_authority(
    authority: Mapping[str, Any],
    *,
    expected_authority_identity_sha256: str,
) -> None:
    """Verify an authority against an independently supplied expected identity."""
    expected_identity = _require_sha256(
        expected_authority_identity_sha256,
        "expected_authority_identity_sha256",
    )
    value = dict(authority)
    if set(value) != set(_AUTHORITY_KEYS):
        raise LaunchInputAuthorityError(
            "launch-input authority has unexpected or missing fields"
        )
    if value.get("schema_version") != LAUNCH_INPUT_SCHEMA:
        raise LaunchInputAuthorityError("unexpected launch-input authority schema")
    if value.get("binding_status") != "READY_FOR_READINESS_BINDING":
        raise LaunchInputAuthorityError("launch-input binding status drift")

    data_spine = value.get("data_spine")
    if not isinstance(data_spine, Mapping) or set(data_spine) != set(
        _DATA_SPINE_KEYS
    ):
        raise LaunchInputAuthorityError(
            "launch-input data spine has unexpected or missing fields"
        )
    for field in (
        "terminal_corpus_authority_identity_sha256",
        "deterministic_double_pack_proof_identity_sha256",
        "terminal_record_inventory_digest_sha256",
        "terminal_payload_inventory_digest_sha256",
        "terminal_split_application_identity_sha256",
        "terminal_split_spec_identity_sha256",
        "terminal_split_train_record_membership_sha256",
        "canonical_build_sha256",
        "two_clean_proof_identity_sha256",
        "two_clean_input_packet_identity_sha256",
        "two_clean_runtime_identity_sha256",
        "materialization_identity_sha256",
        "unique_loss_ledger_identity_sha256",
        "tokenizer_identity_sha256",
        "packing_identity_sha256",
    ):
        _require_sha256(data_spine.get(field), f"data_spine.{field}")

    _normalize_stage_bindings(
        data_spine.get("stage_bindings"), "data_spine.stage_bindings"
    )
    capacity = _require_positive_int(
        data_spine.get("one_pass_unique_nonignored_causal_loss_positions"),
        "data_spine.one_pass_unique_nonignored_causal_loss_positions",
    )
    requested = _require_positive_int(
        data_spine.get("requested_unique_loss_positions"),
        "data_spine.requested_unique_loss_positions",
    )
    if requested > capacity:
        raise LaunchInputAuthorityError(
            "launch-input requested unique exposure exceeds one-pass unique capacity"
        )

    carrier = value.get("carrier")
    if not isinstance(carrier, Mapping) or set(carrier) != set(
        _CARRIER_OUTPUT_KEYS
    ):
        raise LaunchInputAuthorityError(
            "launch-input carrier has unexpected or missing fields"
        )
    if carrier.get("repository") != REPOSITORY:
        raise LaunchInputAuthorityError("launch-input carrier repository mismatch")
    git_sha = _require_git_sha(carrier.get("git_sha"), "carrier.git_sha")
    _require_sha256(carrier.get("modelspec_sha256"), "carrier.modelspec_sha256")
    _require_sha256(
        carrier.get("initialization_identity_sha256"),
        "carrier.initialization_identity_sha256",
    )
    if carrier.get("canonical_base") != "random_init":
        raise LaunchInputAuthorityError(
            "launch-input carrier is not random initialization"
        )
    if carrier.get("foreign_pretrained_weights_used") is not False:
        raise LaunchInputAuthorityError(
            "launch-input carrier foreign pretrained drift"
        )
    if carrier.get("terminal") is not True:
        raise LaunchInputAuthorityError("launch-input carrier terminality drift")
    _require_positive_int(carrier.get("workflow_run_id"), "carrier.workflow_run_id")
    if carrier.get("workflow_status") != "completed":
        raise LaunchInputAuthorityError(
            "launch-input carrier workflow is not completed"
        )
    if carrier.get("workflow_conclusion") != "success":
        raise LaunchInputAuthorityError(
            "launch-input carrier workflow is not success"
        )
    workflow_head = _require_git_sha(
        carrier.get("workflow_head_sha"), "carrier.workflow_head_sha"
    )
    if workflow_head != git_sha:
        raise LaunchInputAuthorityError(
            "launch-input carrier workflow head does not match carrier Git SHA"
        )
    _require_sha256(carrier.get("evidence_sha256"), "carrier.evidence_sha256")

    claim_boundary = value.get("claim_boundary")
    if not isinstance(claim_boundary, Mapping) or set(claim_boundary) != set(
        _CLAIM_BOUNDARY
    ):
        raise LaunchInputAuthorityError(
            "launch-input claim boundary has unexpected or missing fields"
        )
    for field in (
        "contains_source_text",
        "final_test_payload_consumed",
        "authorizes_training",
        "authorizes_compute",
        "replay_padding_or_replacement_can_increase_unique_capacity",
    ):
        if claim_boundary.get(field) is not False:
            raise LaunchInputAuthorityError(
                f"launch-input claim boundary {field} must be exact false boolean"
            )
    exposure = claim_boundary.get("authorized_optimized_target_exposure")
    if isinstance(exposure, bool) or not isinstance(exposure, int) or exposure != 0:
        raise LaunchInputAuthorityError(
            "launch-input claim boundary authorized_optimized_target_exposure "
            "must be exact integer zero"
        )

    observed = _require_sha256(
        value.get("authority_identity_sha256"), "authority_identity_sha256"
    )
    body = dict(value)
    body.pop("authority_identity_sha256", None)
    if _sha256_obj(body) != observed:
        raise LaunchInputAuthorityError(
            "launch-input authority self-identity mismatch"
        )
    if observed != expected_identity:
        raise LaunchInputAuthorityError(
            "launch-input authority does not match independently expected identity"
        )
