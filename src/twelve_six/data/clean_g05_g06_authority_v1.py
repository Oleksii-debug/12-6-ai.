"""Clean-successor G05/G06 replay receipts and terminal composition authority.

This module does not implement quality or privacy policy. It binds two independently
executed LOCAL_FREE receipts from the incumbent G05/G06 mechanics to one externally
expected clean-survivor authority and one terminal native composition preflight.

The resulting authority is deliberately zero-credit: it is suitable as an input to
post-G05/G06 physical materialization, but it does not authorize tokenizer fitting,
optimized-target exposure, optimizer updates, training, learned weights, or final-test
access. Provenance is scoped: the known quarantined lineage must have been rejected,
while whole-corpus external-LLM cleanliness remains an explicit non-claim.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

SCHEMA = "12-6.d03-clean-g05-g06-authority.v1"
REPLAY_SCHEMA = "12-6.d03-clean-g05-g06-replay-receipt.v1"
PREFLIGHT_SCHEMA = "12-6.d03-current-g05-g06-composition-preflight.v1"
EXPECTED_QUARANTINE_IDENTITY_SHA256 = (
    "e9f29dd9f710fac057550e5cd671b7f412720e1ceb36568565a79909f11cf5b6"
)
EXPECTED_G05_IMPLEMENTATION_GIT_BLOB_SHA1 = (
    "c8963d2d697f3cd124a5b511d2883a93f8caf34d"
)
EXPECTED_G06_PRIVACY_IMPLEMENTATION_GIT_BLOB_SHA1 = (
    "bcc5938395724f6728ab212f98b39f2334b0f37d"
)

# These identities belong to the invalidated 272-record Nomis-contaminated survivor
# graph. A clean-successor receipt must never silently reuse them.
_INVALIDATED_CLEAN_INPUT_IDENTITIES = frozenset(
    {
        "284d122a9afd4e5d4676d78a202d3001e5cf87438776022d3004d61fdf10e579",
        "3f60cfe55435daf53908c492be358f36d7ebbc2ebee532c921a69ba92b2f6b25",
        "90e306ce74a82016c835a5e106088f7bdb586e13010301b5002ce5d55e11e27c",
        "36e30427a8f6bc089c911690af82b1c59bfec1c883f3107597ca7fb07838b8df",
    }
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_ALLOWED_MATERIALIZATION_BLOCKERS = frozenset(
    {
        "POST_G05_G06_EXCLUSION_MATERIALIZATION_REQUIRED",
        "G06_REDACTION_MATERIALIZATION_REQUIRED",
    }
)
_PREFLIGHT_KEYS = frozenset(
    {
        "schema",
        "status",
        "input_rows_sha256",
        "survivor_evidence_identity_sha256",
        "survivor_jsonl_sha256",
        "survivor_record_inventory_sha256",
        "survivor_payload_inventory_sha256",
        "g05_execution_identity_sha256",
        "g06_envelope_identity_sha256",
        "g06_execution_identity_sha256",
        "g06_terminal_qualification_identity_sha256",
        "g06_exact_byte_execution_terminal",
        "input_record_count",
        "input_utf8_bytes",
        "decision_matrix",
        "drop_record_id_sha256",
        "g05_partial_record_id_sha256",
        "g06_redaction_record_id_sha256",
        "unchanged_allow_record_id_sha256",
        "drop_record_count",
        "g05_partial_record_count",
        "g06_redaction_record_count",
        "unchanged_allow_record_count",
        "unchanged_allow_input_utf8_bytes",
        "blockers",
        "truth_boundary",
        "composition_preflight_identity_sha256",
    }
)
_REPLAY_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "execution_profile",
        "execution",
        "clean_input",
        "implementations",
        "deterministic_execution",
        "provenance_scope",
        "truth_boundary",
        "replay_identity_sha256",
    }
)
_EXECUTION_KEYS = frozenset(
    {
        "execution_head_git_sha",
        "run_id",
        "job_id",
        "artifact_id",
        "artifact_zip_sha256",
    }
)
_CLEAN_INPUT_KEYS = frozenset(
    {
        "upstream_clean_successor_authority_identity_sha256",
        "provenance_quarantine_identity_sha256",
        "input_rows_sha256",
        "survivor_evidence_identity_sha256",
        "survivor_jsonl_sha256",
        "survivor_record_inventory_sha256",
        "survivor_payload_inventory_sha256",
        "record_count",
        "payload_bytes",
        "source_object_count",
    }
)
_IMPLEMENTATION_KEYS = frozenset(
    {
        "g05_quality_execution_git_blob_sha1",
        "g06_privacy_implementation_git_blob_sha1",
        "g05_quality_threshold_policy_sha256",
        "g05_quality_granularity_policy_sha256",
        "g06_privacy_policy_sha256",
    }
)
_DETERMINISTIC_KEYS = frozenset(
    {
        "g05_execution_identity_sha256",
        "g05_execution_rows_sha256",
        "g06_envelope_identity_sha256",
        "g06_execution_identity_sha256",
        "g06_execution_rows_sha256",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "known_quarantined_lineage_absent",
        "absence_claim_bound_to_canonical_quarantine",
        "upstream_clean_successor_authority_required",
        "whole_corpus_external_llm_cleanliness_claimed",
    }
)
_TRUTH = {
    "current_corpus_eligible": False,
    "training_authorized_bytes": 0,
    "authorized_unique_loss_positions": 0,
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "optimizer_updates_executed_on_real_targets": 0,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights": False,
}
_PREFLIGHT_TRUTH = {
    **_TRUTH,
    "current_corpus_external_llm_free_claimed_by_this_preflight": False,
}


class CleanG05G06AuthorityError(ValueError):
    """Raised when a clean-successor replay or composition authority fails closed."""


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise CleanG05G06AuthorityError(message)


def _cjson(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha64(value: Any, field: str) -> str:
    _need(
        isinstance(value, str) and _HEX64.fullmatch(value) is not None,
        f"{field} must be lowercase SHA-256",
    )
    return value


def _sha40(value: Any, field: str) -> str:
    _need(
        isinstance(value, str) and _HEX40.fullmatch(value) is not None,
        f"{field} must be lowercase Git SHA-1",
    )
    return value


def _exact_int(value: Any, field: str, *, minimum: int = 0) -> int:
    _need(type(value) is int and value >= minimum, f"{field} must be an integer >= {minimum}")
    return value


def _obj(value: Any, field: str, keys: frozenset[str] | None = None) -> Mapping[str, Any]:
    _need(isinstance(value, Mapping), f"{field} must be an object")
    if keys is not None:
        _need(set(value) == keys, f"{field} schema is not closed")
    return value


def _self_hash(
    value: Mapping[str, Any],
    *,
    identity_field: str,
    expected_identity: str,
    field: str,
) -> str:
    expected = _sha64(expected_identity, f"expected {field} identity")
    claimed = _sha64(value.get(identity_field), f"{field}.{identity_field}")
    _need(claimed == expected, f"{field} identity is not independently expected")
    core = dict(value)
    del core[identity_field]
    _need(claimed == _sha256(_cjson(core)), f"{field} self-hash mismatch")
    return claimed


def _sorted_hash_list(value: Any, field: str) -> list[str]:
    _need(type(value) is list, f"{field} must be a list")
    result = [_sha64(item, f"{field}[{index}]") for index, item in enumerate(value)]
    _need(result == sorted(result), f"{field} must be sorted")
    _need(len(result) == len(set(result)), f"{field} contains duplicates")
    return result


def _clean_input(
    *,
    upstream_clean_successor_authority_identity_sha256: str,
    provenance_quarantine_identity_sha256: str,
    input_rows_sha256: str,
    survivor_evidence_identity_sha256: str,
    survivor_jsonl_sha256: str,
    survivor_record_inventory_sha256: str,
    survivor_payload_inventory_sha256: str,
    record_count: int,
    payload_bytes: int,
    source_object_count: int,
) -> dict[str, Any]:
    upstream = _sha64(
        upstream_clean_successor_authority_identity_sha256,
        "upstream clean successor authority identity",
    )
    quarantine = _sha64(
        provenance_quarantine_identity_sha256,
        "provenance quarantine identity",
    )
    _need(
        quarantine == EXPECTED_QUARANTINE_IDENTITY_SHA256,
        "provenance quarantine identity is not the canonical authority",
    )
    clean_identities = {
        "survivor_evidence_identity_sha256": _sha64(
            survivor_evidence_identity_sha256, "survivor evidence identity"
        ),
        "survivor_jsonl_sha256": _sha64(survivor_jsonl_sha256, "survivor JSONL"),
        "survivor_record_inventory_sha256": _sha64(
            survivor_record_inventory_sha256, "survivor record inventory"
        ),
        "survivor_payload_inventory_sha256": _sha64(
            survivor_payload_inventory_sha256, "survivor payload inventory"
        ),
    }
    for field, identity in clean_identities.items():
        _need(
            identity not in _INVALIDATED_CLEAN_INPUT_IDENTITIES,
            f"{field} reuses invalidated contaminated survivor identity",
        )
    return {
        "upstream_clean_successor_authority_identity_sha256": upstream,
        "provenance_quarantine_identity_sha256": quarantine,
        "input_rows_sha256": _sha64(input_rows_sha256, "input rows"),
        **clean_identities,
        "record_count": _exact_int(record_count, "record_count", minimum=1),
        "payload_bytes": _exact_int(payload_bytes, "payload_bytes", minimum=1),
        "source_object_count": _exact_int(
            source_object_count, "source_object_count", minimum=1
        ),
    }


def build_replay_receipt(
    *,
    execution_head_git_sha: str,
    run_id: int,
    job_id: int,
    artifact_id: int,
    artifact_zip_sha256: str,
    upstream_clean_successor_authority_identity_sha256: str,
    provenance_quarantine_identity_sha256: str,
    input_rows_sha256: str,
    survivor_evidence_identity_sha256: str,
    survivor_jsonl_sha256: str,
    survivor_record_inventory_sha256: str,
    survivor_payload_inventory_sha256: str,
    record_count: int,
    payload_bytes: int,
    source_object_count: int,
    g05_execution_identity_sha256: str,
    g05_execution_rows_sha256: str,
    g06_envelope_identity_sha256: str,
    g06_execution_identity_sha256: str,
    g06_execution_rows_sha256: str,
    g05_quality_threshold_policy_sha256: str,
    g05_quality_granularity_policy_sha256: str,
    g06_privacy_policy_sha256: str,
    g05_quality_execution_git_blob_sha1: str = (
        EXPECTED_G05_IMPLEMENTATION_GIT_BLOB_SHA1
    ),
    g06_privacy_implementation_git_blob_sha1: str = (
        EXPECTED_G06_PRIVACY_IMPLEMENTATION_GIT_BLOB_SHA1
    ),
    known_quarantined_lineage_absent: bool,
) -> dict[str, Any]:
    """Build a text-free receipt for one physical LOCAL_FREE G05/G06 replay."""
    _need(
        known_quarantined_lineage_absent is True,
        "known quarantined lineage absence must be physically checked before receipt",
    )
    g05_blob = _sha40(
        g05_quality_execution_git_blob_sha1,
        "G05 quality execution implementation Git blob",
    )
    _need(
        g05_blob == EXPECTED_G05_IMPLEMENTATION_GIT_BLOB_SHA1,
        "G05 implementation substitution",
    )
    g06_blob = _sha40(
        g06_privacy_implementation_git_blob_sha1,
        "G06 privacy implementation Git blob",
    )
    _need(
        g06_blob == EXPECTED_G06_PRIVACY_IMPLEMENTATION_GIT_BLOB_SHA1,
        "G06 privacy implementation substitution",
    )
    clean_input = _clean_input(
        upstream_clean_successor_authority_identity_sha256=(
            upstream_clean_successor_authority_identity_sha256
        ),
        provenance_quarantine_identity_sha256=provenance_quarantine_identity_sha256,
        input_rows_sha256=input_rows_sha256,
        survivor_evidence_identity_sha256=survivor_evidence_identity_sha256,
        survivor_jsonl_sha256=survivor_jsonl_sha256,
        survivor_record_inventory_sha256=survivor_record_inventory_sha256,
        survivor_payload_inventory_sha256=survivor_payload_inventory_sha256,
        record_count=record_count,
        payload_bytes=payload_bytes,
        source_object_count=source_object_count,
    )
    core = {
        "schema_version": REPLAY_SCHEMA,
        "status": "PHYSICAL_LOCAL_FREE_REPLAY_COMPLETE",
        "execution_profile": "LOCAL_FREE",
        "execution": {
            "execution_head_git_sha": _sha40(
                execution_head_git_sha, "execution head Git SHA"
            ),
            "run_id": _exact_int(run_id, "run_id", minimum=1),
            "job_id": _exact_int(job_id, "job_id", minimum=1),
            "artifact_id": _exact_int(artifact_id, "artifact_id", minimum=1),
            "artifact_zip_sha256": _sha64(
                artifact_zip_sha256, "artifact ZIP SHA-256"
            ),
        },
        "clean_input": clean_input,
        "implementations": {
            "g05_quality_execution_git_blob_sha1": g05_blob,
            "g06_privacy_implementation_git_blob_sha1": g06_blob,
            "g05_quality_threshold_policy_sha256": _sha64(
                g05_quality_threshold_policy_sha256,
                "G05 quality threshold policy",
            ),
            "g05_quality_granularity_policy_sha256": _sha64(
                g05_quality_granularity_policy_sha256,
                "G05 quality granularity policy",
            ),
            "g06_privacy_policy_sha256": _sha64(
                g06_privacy_policy_sha256, "G06 privacy policy"
            ),
        },
        "deterministic_execution": {
            "g05_execution_identity_sha256": _sha64(
                g05_execution_identity_sha256, "G05 execution identity"
            ),
            "g05_execution_rows_sha256": _sha64(
                g05_execution_rows_sha256, "G05 execution rows"
            ),
            "g06_envelope_identity_sha256": _sha64(
                g06_envelope_identity_sha256, "G06 envelope identity"
            ),
            "g06_execution_identity_sha256": _sha64(
                g06_execution_identity_sha256, "G06 execution identity"
            ),
            "g06_execution_rows_sha256": _sha64(
                g06_execution_rows_sha256, "G06 execution rows"
            ),
        },
        "provenance_scope": {
            "known_quarantined_lineage_absent": True,
            "absence_claim_bound_to_canonical_quarantine": True,
            "upstream_clean_successor_authority_required": True,
            "whole_corpus_external_llm_cleanliness_claimed": False,
        },
        "truth_boundary": dict(_TRUTH),
    }
    return {**core, "replay_identity_sha256": _sha256(_cjson(core))}


def validate_replay_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_identity_sha256: str,
) -> str:
    root = _obj(receipt, "replay receipt", _REPLAY_KEYS)
    _need(root.get("schema_version") == REPLAY_SCHEMA, "replay schema drift")
    _need(
        root.get("status") == "PHYSICAL_LOCAL_FREE_REPLAY_COMPLETE",
        "replay is not physically complete",
    )
    _need(root.get("execution_profile") == "LOCAL_FREE", "replay is not LOCAL_FREE")
    identity = _self_hash(
        root,
        identity_field="replay_identity_sha256",
        expected_identity=expected_identity_sha256,
        field="replay receipt",
    )
    execution = _obj(root.get("execution"), "replay.execution", _EXECUTION_KEYS)
    _sha40(execution.get("execution_head_git_sha"), "replay execution head")
    for key in ("run_id", "job_id", "artifact_id"):
        _exact_int(execution.get(key), f"replay.execution.{key}", minimum=1)
    _sha64(execution.get("artifact_zip_sha256"), "replay artifact ZIP")

    clean = _obj(root.get("clean_input"), "replay.clean_input", _CLEAN_INPUT_KEYS)
    reconstructed = _clean_input(
        upstream_clean_successor_authority_identity_sha256=clean.get(
            "upstream_clean_successor_authority_identity_sha256"
        ),
        provenance_quarantine_identity_sha256=clean.get(
            "provenance_quarantine_identity_sha256"
        ),
        input_rows_sha256=clean.get("input_rows_sha256"),
        survivor_evidence_identity_sha256=clean.get(
            "survivor_evidence_identity_sha256"
        ),
        survivor_jsonl_sha256=clean.get("survivor_jsonl_sha256"),
        survivor_record_inventory_sha256=clean.get(
            "survivor_record_inventory_sha256"
        ),
        survivor_payload_inventory_sha256=clean.get(
            "survivor_payload_inventory_sha256"
        ),
        record_count=clean.get("record_count"),
        payload_bytes=clean.get("payload_bytes"),
        source_object_count=clean.get("source_object_count"),
    )
    _need(dict(clean) == reconstructed, "replay clean-input binding drift")

    implementations = _obj(
        root.get("implementations"), "replay.implementations", _IMPLEMENTATION_KEYS
    )
    _need(
        implementations.get("g05_quality_execution_git_blob_sha1")
        == EXPECTED_G05_IMPLEMENTATION_GIT_BLOB_SHA1,
        "replay G05 implementation drift",
    )
    _need(
        implementations.get("g06_privacy_implementation_git_blob_sha1")
        == EXPECTED_G06_PRIVACY_IMPLEMENTATION_GIT_BLOB_SHA1,
        "replay G06 privacy implementation drift",
    )
    for key in (
        "g05_quality_threshold_policy_sha256",
        "g05_quality_granularity_policy_sha256",
        "g06_privacy_policy_sha256",
    ):
        _sha64(implementations.get(key), f"replay.implementations.{key}")

    deterministic = _obj(
        root.get("deterministic_execution"),
        "replay.deterministic_execution",
        _DETERMINISTIC_KEYS,
    )
    for key, value in deterministic.items():
        _sha64(value, f"replay.deterministic_execution.{key}")

    provenance = _obj(
        root.get("provenance_scope"), "replay.provenance_scope", _PROVENANCE_KEYS
    )
    _need(
        provenance
        == {
            "known_quarantined_lineage_absent": True,
            "absence_claim_bound_to_canonical_quarantine": True,
            "upstream_clean_successor_authority_required": True,
            "whole_corpus_external_llm_cleanliness_claimed": False,
        },
        "replay provenance scope widened",
    )
    _need(root.get("truth_boundary") == _TRUTH, "replay truth boundary widened")
    return identity


def _validate_preflight(
    preflight: Mapping[str, Any],
    *,
    expected_identity_sha256: str,
    expected_terminal_qualification_identity_sha256: str,
    replay: Mapping[str, Any],
) -> tuple[str, list[str]]:
    root = _obj(preflight, "composition preflight", _PREFLIGHT_KEYS)
    _need(root.get("schema") == PREFLIGHT_SCHEMA, "composition preflight schema drift")
    identity = _self_hash(
        root,
        identity_field="composition_preflight_identity_sha256",
        expected_identity=expected_identity_sha256,
        field="composition preflight",
    )
    _need(
        root.get("g06_exact_byte_execution_terminal") is True,
        "G06 exact-byte execution is not terminal",
    )
    terminal_identity = _sha64(
        root.get("g06_terminal_qualification_identity_sha256"),
        "preflight terminal G06 qualification",
    )
    _need(
        terminal_identity
        == _sha64(
            expected_terminal_qualification_identity_sha256,
            "expected terminal G06 qualification",
        ),
        "terminal G06 qualification identity is not independently expected",
    )
    clean = replay["clean_input"]
    deterministic = replay["deterministic_execution"]
    expected = {
        "input_rows_sha256": clean["input_rows_sha256"],
        "survivor_evidence_identity_sha256": clean[
            "survivor_evidence_identity_sha256"
        ],
        "survivor_jsonl_sha256": clean["survivor_jsonl_sha256"],
        "survivor_record_inventory_sha256": clean[
            "survivor_record_inventory_sha256"
        ],
        "survivor_payload_inventory_sha256": clean[
            "survivor_payload_inventory_sha256"
        ],
        "g05_execution_identity_sha256": deterministic[
            "g05_execution_identity_sha256"
        ],
        "g06_envelope_identity_sha256": deterministic[
            "g06_envelope_identity_sha256"
        ],
        "g06_execution_identity_sha256": deterministic[
            "g06_execution_identity_sha256"
        ],
        "input_record_count": clean["record_count"],
        "input_utf8_bytes": clean["payload_bytes"],
    }
    for key, value in expected.items():
        _need(root.get(key) == value, f"composition preflight lineage drift: {key}")

    _need(root.get("truth_boundary") == _PREFLIGHT_TRUTH, "preflight truth scope widened")
    _need(
        _exact_int(root.get("g05_partial_record_count"), "g05_partial_record_count")
        == 0,
        "G05 partial decisions are not terminally materializable",
    )
    _need(
        _sorted_hash_list(
            root.get("g05_partial_record_id_sha256"), "g05_partial_record_id_sha256"
        )
        == [],
        "G05 partial decision hashes present",
    )
    drop = _sorted_hash_list(
        root.get("drop_record_id_sha256"), "drop_record_id_sha256"
    )
    redact = _sorted_hash_list(
        root.get("g06_redaction_record_id_sha256"),
        "g06_redaction_record_id_sha256",
    )
    unchanged = _sorted_hash_list(
        root.get("unchanged_allow_record_id_sha256"),
        "unchanged_allow_record_id_sha256",
    )
    _need(
        len(drop) == _exact_int(root.get("drop_record_count"), "drop_record_count"),
        "drop decision count/hash mismatch",
    )
    _need(
        len(redact)
        == _exact_int(root.get("g06_redaction_record_count"), "g06_redaction_record_count"),
        "G06 redaction count/hash mismatch",
    )
    _need(
        len(unchanged)
        == _exact_int(
            root.get("unchanged_allow_record_count"), "unchanged_allow_record_count"
        ),
        "unchanged decision count/hash mismatch",
    )
    _need(not (set(drop) & set(redact)), "drop/redaction decision overlap")
    _need(not (set(drop) & set(unchanged)), "drop/unchanged decision overlap")
    _need(not (set(redact) & set(unchanged)), "redaction/unchanged decision overlap")

    blockers = root.get("blockers")
    _need(type(blockers) is list and blockers == sorted(set(blockers)), "preflight blockers drift")
    blocker_set = set(blockers)
    _need(
        blocker_set <= _ALLOWED_MATERIALIZATION_BLOCKERS,
        "preflight contains a non-materialization blocker",
    )
    expected_blockers: set[str] = set()
    if drop:
        expected_blockers.add("POST_G05_G06_EXCLUSION_MATERIALIZATION_REQUIRED")
    if redact:
        expected_blockers.add("G06_REDACTION_MATERIALIZATION_REQUIRED")
    _need(blocker_set == expected_blockers, "preflight blocker/decision mismatch")
    expected_status = (
        "BLOCKED_CURRENT_G05_G06_COMPOSITION"
        if blockers
        else "READY_FOR_FINAL_COVERAGE_BINDING"
    )
    _need(root.get("status") == expected_status, "preflight status/blocker mismatch")

    matrix = root.get("decision_matrix")
    _need(type(matrix) is list and bool(matrix), "decision matrix missing")
    matrix_root = _sha256(_cjson(matrix))
    _exact_int(
        root.get("unchanged_allow_input_utf8_bytes"),
        "unchanged_allow_input_utf8_bytes",
    )
    return identity, blockers + [matrix_root]


def build_clean_g05_g06_authority(
    *,
    replay_a: Mapping[str, Any],
    expected_replay_a_identity_sha256: str,
    replay_b: Mapping[str, Any],
    expected_replay_b_identity_sha256: str,
    composition_preflight: Mapping[str, Any],
    expected_composition_preflight_identity_sha256: str,
    expected_terminal_g06_qualification_identity_sha256: str,
    expected_upstream_clean_successor_authority_identity_sha256: str,
) -> dict[str, Any]:
    """Bind two independent physical replays to one terminal scoped authority."""
    replay_a_id = validate_replay_receipt(
        replay_a, expected_identity_sha256=expected_replay_a_identity_sha256
    )
    replay_b_id = validate_replay_receipt(
        replay_b, expected_identity_sha256=expected_replay_b_identity_sha256
    )
    _need(replay_a_id != replay_b_id, "two replay receipts must have distinct identities")

    execution_a = replay_a["execution"]
    execution_b = replay_b["execution"]
    _need(
        (execution_a["run_id"], execution_a["job_id"])
        != (execution_b["run_id"], execution_b["job_id"]),
        "two replay receipts do not prove distinct execution coordinates",
    )
    _need(
        execution_a["artifact_id"] != execution_b["artifact_id"],
        "two replay receipts must bind distinct artifacts",
    )

    for field in ("clean_input", "implementations", "deterministic_execution"):
        _need(replay_a[field] == replay_b[field], f"two-replay mismatch: {field}")

    clean = replay_a["clean_input"]
    upstream = _sha64(
        expected_upstream_clean_successor_authority_identity_sha256,
        "expected upstream clean successor authority",
    )
    _need(
        clean["upstream_clean_successor_authority_identity_sha256"] == upstream,
        "clean input is not bound to independently expected upstream authority",
    )

    preflight_id, blocker_and_matrix = _validate_preflight(
        composition_preflight,
        expected_identity_sha256=expected_composition_preflight_identity_sha256,
        expected_terminal_qualification_identity_sha256=(
            expected_terminal_g06_qualification_identity_sha256
        ),
        replay=replay_a,
    )
    matrix_root = blocker_and_matrix[-1]
    blockers = blocker_and_matrix[:-1]
    projection = {
        "clean_input": clean,
        "implementations": replay_a["implementations"],
        "deterministic_execution": replay_a["deterministic_execution"],
    }
    projection_root = _sha256(_cjson(projection))
    status = (
        "QUALIFIED_FOR_POST_G05_G06_MATERIALIZATION"
        if blockers
        else "QUALIFIED_FOR_FINAL_COVERAGE_BINDING"
    )
    core = {
        "schema_version": SCHEMA,
        "status": status,
        "execution_profile": "LOCAL_FREE_TWO_REPLAY",
        "clean_input": clean,
        "implementations": replay_a["implementations"],
        "two_replay_binding": {
            "replay_a_identity_sha256": replay_a_id,
            "replay_b_identity_sha256": replay_b_id,
            "deterministic_projection_sha256": projection_root,
            "replay_count": 2,
            "distinct_execution_coordinates": True,
            "distinct_artifacts": True,
        },
        "composition_preflight": {
            "composition_preflight_identity_sha256": preflight_id,
            "terminal_g06_qualification_identity_sha256": _sha64(
                expected_terminal_g06_qualification_identity_sha256,
                "expected terminal G06 qualification",
            ),
            "decision_matrix_sha256": matrix_root,
            "materialization_blockers": blockers,
        },
        "provenance_scope": {
            "known_quarantined_lineage_absent": True,
            "quarantine_identity_sha256": EXPECTED_QUARANTINE_IDENTITY_SHA256,
            "upstream_clean_successor_authority_identity_sha256": upstream,
            "legacy_g05_g06_stage_truth_is_not_corpus_provenance_authority": True,
            "whole_corpus_external_llm_cleanliness_claimed": False,
        },
        "truth_boundary": dict(_TRUTH),
    }
    return {**core, "authority_identity_sha256": _sha256(_cjson(core))}


def verify_clean_g05_g06_authority(
    authority: Mapping[str, Any],
    *,
    expected_authority_identity_sha256: str,
    expected_upstream_clean_successor_authority_identity_sha256: str,
    expected_replay_a_identity_sha256: str,
    expected_replay_b_identity_sha256: str,
    expected_composition_preflight_identity_sha256: str,
    expected_terminal_g06_qualification_identity_sha256: str,
) -> str:
    """Verify a final authority against independently retained identities."""
    expected_keys = {
        "schema_version",
        "status",
        "execution_profile",
        "clean_input",
        "implementations",
        "two_replay_binding",
        "composition_preflight",
        "provenance_scope",
        "truth_boundary",
        "authority_identity_sha256",
    }
    root = _obj(authority, "clean G05/G06 authority", frozenset(expected_keys))
    _need(root.get("schema_version") == SCHEMA, "clean authority schema drift")
    _need(
        root.get("status")
        in {
            "QUALIFIED_FOR_POST_G05_G06_MATERIALIZATION",
            "QUALIFIED_FOR_FINAL_COVERAGE_BINDING",
        },
        "clean authority status drift",
    )
    _need(
        root.get("execution_profile") == "LOCAL_FREE_TWO_REPLAY",
        "clean authority execution profile drift",
    )
    identity = _self_hash(
        root,
        identity_field="authority_identity_sha256",
        expected_identity=expected_authority_identity_sha256,
        field="clean G05/G06 authority",
    )
    clean = _obj(root.get("clean_input"), "clean authority input", _CLEAN_INPUT_KEYS)
    _need(
        clean.get("upstream_clean_successor_authority_identity_sha256")
        == _sha64(
            expected_upstream_clean_successor_authority_identity_sha256,
            "expected upstream clean successor authority",
        ),
        "clean authority upstream lineage drift",
    )
    two = _obj(root.get("two_replay_binding"), "two-replay binding")
    _need(
        two.get("replay_a_identity_sha256")
        == _sha64(expected_replay_a_identity_sha256, "expected replay A identity"),
        "replay A authority drift",
    )
    _need(
        two.get("replay_b_identity_sha256")
        == _sha64(expected_replay_b_identity_sha256, "expected replay B identity"),
        "replay B authority drift",
    )
    _need(two.get("replay_count") == 2, "two-replay count drift")
    _need(two.get("distinct_execution_coordinates") is True, "replay independence drift")
    _need(two.get("distinct_artifacts") is True, "replay artifact independence drift")

    preflight = _obj(root.get("composition_preflight"), "composition preflight binding")
    _need(
        preflight.get("composition_preflight_identity_sha256")
        == _sha64(
            expected_composition_preflight_identity_sha256,
            "expected composition preflight identity",
        ),
        "composition preflight authority drift",
    )
    _need(
        preflight.get("terminal_g06_qualification_identity_sha256")
        == _sha64(
            expected_terminal_g06_qualification_identity_sha256,
            "expected terminal G06 qualification identity",
        ),
        "terminal G06 qualification authority drift",
    )
    provenance = _obj(root.get("provenance_scope"), "clean authority provenance scope")
    _need(
        provenance.get("whole_corpus_external_llm_cleanliness_claimed") is False,
        "whole-corpus external-LLM cleanliness claim widened",
    )
    _need(
        provenance.get("legacy_g05_g06_stage_truth_is_not_corpus_provenance_authority")
        is True,
        "legacy stage truth trust direction drift",
    )
    _need(root.get("truth_boundary") == _TRUTH, "clean authority truth boundary widened")
    return identity
