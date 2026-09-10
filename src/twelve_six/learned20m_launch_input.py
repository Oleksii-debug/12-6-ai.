"""Fail-closed composition of terminal learned-20M launch-input authorities.

This module does not tokenize, pack, count source bytes as loss positions, authorize
training, or inspect final-test payloads. It only binds independently expected
terminal identities produced by the canonical D03/D04/D10 authorities.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

REPOSITORY = "Oleksii-debug/12-6-ai."
LAUNCH_INPUT_SCHEMA = "12-6.learned20m-launch-input-authority.v1"
TWO_CLEAN_SCHEMA = "12-6.postpack-two-clean-proof.v1"
LEDGER_SCHEMA = "12-6.unique-loss-position-ledger.v2"
POSITION_POLICY = "logical-causal-token-target-postpack-v2"
_REQUIRED_STAGE_BINDINGS = (
    "normalization",
    "evaluation_reservations",
    "dedup",
    "split",
    "packing",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


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
    if not isinstance(value, Mapping) or set(value) != set(_REQUIRED_STAGE_BINDINGS):
        raise LaunchInputAuthorityError(
            f"{field} must contain exactly the canonical five stage bindings"
        )
    return {
        name: _require_sha256(value[name], f"{field}.{name}")
        for name in _REQUIRED_STAGE_BINDINGS
    }


def _verify_self_hash(
    value: Mapping[str, Any],
    *,
    identity_field: str,
    expected_identity_sha256: str,
    label: str,
) -> dict[str, Any]:
    expected = _require_sha256(expected_identity_sha256, f"expected_{label}_identity_sha256")
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
    expected_corpus_identity_sha256: str,
    expected_stage_bindings: Mapping[str, str],
) -> dict[str, Any]:
    value = _verify_self_hash(
        proof,
        identity_field="proof_identity_sha256",
        expected_identity_sha256=expected_identity_sha256,
        label="two_clean_proof",
    )
    if value.get("schema_version") != TWO_CLEAN_SCHEMA:
        raise LaunchInputAuthorityError("unexpected two-clean proof schema")
    if value.get("fresh_process_count") != 2 or value.get("byte_identical") is not True:
        raise LaunchInputAuthorityError("two-clean proof is not terminal byte-identical")
    build_a = _require_sha256(value.get("build_a_sha256"), "build_a_sha256")
    build_b = _require_sha256(value.get("build_b_sha256"), "build_b_sha256")
    if build_a != build_b:
        raise LaunchInputAuthorityError("two-clean build output hashes differ")
    expected_corpus = _require_sha256(
        expected_corpus_identity_sha256,
        "expected_terminal_corpus_authority_identity_sha256",
    )
    if value.get("terminal_corpus_authority_identity_sha256") != expected_corpus:
        raise LaunchInputAuthorityError("two-clean corpus authority substitution")
    expected_bindings = _normalize_stage_bindings(
        expected_stage_bindings, "expected_stage_bindings"
    )
    observed_bindings = _normalize_stage_bindings(
        value.get("stage_bindings"), "proof.stage_bindings"
    )
    if observed_bindings != expected_bindings:
        raise LaunchInputAuthorityError("two-clean stage authority substitution")
    _require_sha256(
        value.get("materialization_identity_sha256"),
        "materialization_identity_sha256",
    )
    if value.get("claim_boundary") != {
        "contains_source_text": False,
        "authorizes_training": False,
        "authorizes_paid_compute": False,
        "creates_positive_unique_loss_authority": False,
    }:
        raise LaunchInputAuthorityError("two-clean proof claim boundary drift")
    return value


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
        raise LaunchInputAuthorityError("unique-loss ledger is not a complete one-pass ledger")
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
        raise LaunchInputAuthorityError("cross-document positions are not unique authority")
    if value.get("source_bytes_relabelled_as_loss_positions") is not False:
        raise LaunchInputAuthorityError("source bytes cannot be relabelled as loss positions")

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
    for index, raw_segment in enumerate(segments):
        if not isinstance(raw_segment, Mapping):
            raise LaunchInputAuthorityError(f"segments[{index}] must be an object")
        segment_id = _require_sha256(
            raw_segment.get("segment_identity_sha256"),
            f"segments[{index}].segment_identity_sha256",
        )
        if segment_id in segment_ids:
            raise LaunchInputAuthorityError("duplicate segment identity in unique-loss ledger")
        segment_ids.add(segment_id)
        document_id = raw_segment.get("document_id")
        start = raw_segment.get("target_start")
        end = raw_segment.get("target_end")
        if not isinstance(document_id, str) or not document_id:
            raise LaunchInputAuthorityError(f"segments[{index}].document_id must be non-empty")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 1
            or end <= start
        ):
            raise LaunchInputAuthorityError(f"segments[{index}] target range is invalid")
        logical_range = (document_id, start, end)
        if logical_range in logical_ranges:
            raise LaunchInputAuthorityError("duplicate logical range in unique-loss ledger")
        logical_ranges.add(logical_range)
        count = _require_positive_int(
            raw_segment.get("loss_position_count"),
            f"segments[{index}].loss_position_count",
        )
        if count != end - start:
            raise LaunchInputAuthorityError("segment loss count does not match target range")
        counted += count
    if counted != positions:
        raise LaunchInputAuthorityError("segment counts do not match unique-loss capacity")
    return value, positions


def _verify_carrier(
    carrier: Mapping[str, Any],
    *,
    expected_git_sha: str,
    expected_modelspec_sha256: str,
    expected_initialization_sha256: str,
) -> dict[str, Any]:
    value = dict(carrier)
    if value.get("repository") != REPOSITORY:
        raise LaunchInputAuthorityError("carrier repository mismatch")
    git_sha = _require_git_sha(value.get("git_sha"), "carrier.git_sha")
    if git_sha != _require_git_sha(expected_git_sha, "expected_carrier_git_sha"):
        raise LaunchInputAuthorityError("carrier Git SHA substitution")
    model = _require_sha256(value.get("modelspec_sha256"), "carrier.modelspec_sha256")
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
        raise LaunchInputAuthorityError("carrier is not canonical random initialization")
    if value.get("foreign_pretrained_weights_used") is not False:
        raise LaunchInputAuthorityError("foreign pretrained weights are forbidden")
    if value.get("terminal") is not True:
        raise LaunchInputAuthorityError("carrier authority is nonterminal")
    if value.get("workflow_conclusion") != "success":
        raise LaunchInputAuthorityError("carrier exact-head CI is not terminal success")
    _require_positive_int(value.get("workflow_run_id"), "carrier.workflow_run_id")
    _require_sha256(value.get("evidence_sha256"), "carrier.evidence_sha256")
    return value


def build_launch_input_authority(
    two_clean_proof: Mapping[str, Any],
    unique_loss_ledger: Mapping[str, Any],
    carrier_authority: Mapping[str, Any],
    *,
    expected_two_clean_proof_identity_sha256: str,
    expected_unique_loss_ledger_identity_sha256: str,
    expected_terminal_corpus_authority_identity_sha256: str,
    expected_stage_bindings: Mapping[str, str],
    expected_tokenizer_identity_sha256: str,
    expected_packing_identity_sha256: str,
    expected_carrier_git_sha: str,
    expected_modelspec_sha256: str,
    expected_initialization_identity_sha256: str,
    requested_unique_loss_positions: int,
) -> dict[str, Any]:
    """Bind terminal upstream identities without authorizing an optimizer step."""
    proof = _verify_two_clean_proof(
        two_clean_proof,
        expected_identity_sha256=expected_two_clean_proof_identity_sha256,
        expected_corpus_identity_sha256=(
            expected_terminal_corpus_authority_identity_sha256
        ),
        expected_stage_bindings=expected_stage_bindings,
    )
    ledger, one_pass_capacity = _verify_ledger(
        unique_loss_ledger,
        expected_identity_sha256=expected_unique_loss_ledger_identity_sha256,
        expected_materialization_identity_sha256=proof["materialization_identity_sha256"],
        expected_stage_bindings=expected_stage_bindings,
        expected_tokenizer_identity_sha256=expected_tokenizer_identity_sha256,
        expected_packing_identity_sha256=expected_packing_identity_sha256,
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
            "two_clean_proof_identity_sha256": proof["proof_identity_sha256"],
            "materialization_identity_sha256": proof[
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
            "workflow_run_id": carrier["workflow_run_id"],
            "workflow_conclusion": "success",
            "evidence_sha256": carrier["evidence_sha256"],
        },
        "claim_boundary": {
            "contains_source_text": False,
            "final_test_payload_consumed": False,
            "authorizes_training": False,
            "authorizes_compute": False,
            "authorized_optimized_target_exposure": 0,
            "replay_padding_or_replacement_can_increase_unique_capacity": False,
        },
    }
    authority["authority_identity_sha256"] = _sha256_obj(authority)
    return authority


def verify_launch_input_authority(authority: Mapping[str, Any]) -> None:
    """Verify deterministic identity and immutable non-authorization boundary."""
    value = dict(authority)
    if value.get("schema_version") != LAUNCH_INPUT_SCHEMA:
        raise LaunchInputAuthorityError("unexpected launch-input authority schema")
    observed = _require_sha256(
        value.get("authority_identity_sha256"), "authority_identity_sha256"
    )
    body = dict(value)
    body.pop("authority_identity_sha256", None)
    if _sha256_obj(body) != observed:
        raise LaunchInputAuthorityError("launch-input authority self-identity mismatch")
    if value.get("binding_status") != "READY_FOR_READINESS_BINDING":
        raise LaunchInputAuthorityError("launch-input binding status drift")
    if value.get("claim_boundary") != {
        "contains_source_text": False,
        "final_test_payload_consumed": False,
        "authorizes_training": False,
        "authorizes_compute": False,
        "authorized_optimized_target_exposure": 0,
        "replay_padding_or_replacement_can_increase_unique_capacity": False,
    }:
        raise LaunchInputAuthorityError("launch-input claim boundary drift")
