"""Hash-only EVAL-647 overlap gate against the real post-G05/G06 corpus.

The caller supplies ephemeral retained training rows and the two already-sealed EVAL-647
payloads.  This module deliberately does not fetch, persist, or reconstruct raw text.  It
reuses the incumbent DATA-232 matcher and emits only hash-bound evidence.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from twelve_six.data._data232_decontamination_matching import (
    ALGORITHM,
    CODE_SKELETON,
    DEFAULT_THRESHOLDS,
    NORMALIZATION,
    DecontaminationError,
    _blocked_pairs,
    _fingerprint,
    _pair,
    _thresholds,
    stable_identity,
)

SCHEMA = "12-6.eval647-current-corpus-overlap.v1"
MATERIALIZATION_SCHEMA = "12-6.d03-post-g05-g06-materialization.v1"
MATERIALIZATION_STATUS = "MATERIALIZED_ZERO_CREDIT"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DecontaminationError(message)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{label} must be lowercase SHA-256",
    )
    return str(value)


def _reserved_rows(
    manifest: Mapping[str, Any], payloads: Sequence[bytes | str]
) -> tuple[list[dict[str, str]], str]:
    objects = manifest.get("objects")
    _require(isinstance(objects, list) and len(objects) == 2, "EVAL-647 must contain exactly two sealed objects")
    _require(len(payloads) == len(objects), "reserved payload count mismatch")
    materialization = manifest.get("materialization_evidence")
    _require(isinstance(materialization, Mapping), "materialization evidence binding missing")
    evidence_identity = _require_sha256(materialization.get("identity_sha256"), "EVAL-647 evidence identity")

    rows: list[dict[str, str]] = []
    for index, (obj, raw_payload) in enumerate(zip(objects, payloads, strict=True)):
        _require(isinstance(obj, Mapping), f"objects[{index}] must be an object")
        _require(obj.get("training_allowed") is False, "reserved object training boundary widened")
        _require(obj.get("tokenizer_fit_allowed") is False, "reserved object tokenizer boundary widened")
        _require(obj.get("permanent_future_training_exclusion") is True, "reserved object future exclusion missing")
        payload = raw_payload.encode("utf-8") if isinstance(raw_payload, str) else bytes(raw_payload)
        payload.decode("utf-8", errors="strict")
        _require(len(payload) == obj.get("expected_raw_bytes"), f"objects[{index}] byte count drift")
        _require(_sha256(payload) == obj.get("raw_sha256"), f"objects[{index}] raw SHA-256 drift")
        source_family = obj.get("source_family")
        repository = obj.get("repository")
        revision = obj.get("revision")
        path = obj.get("path")
        for label, value in (("source_family", source_family), ("repository", repository), ("revision", revision), ("path", path)):
            _require(isinstance(value, str) and bool(value), f"objects[{index}].{label} missing")
        rows.append(
            {
                "record_id": f"eval647:{index}:{obj['raw_sha256']}",
                "source_id": f"{repository}@{revision}:{path}",
                "source_family": str(source_family),
                "modality": "code",
                "text": payload.decode("utf-8"),
            }
        )
    return rows, evidence_identity


def build_report(
    training_records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    reserved_payloads: Sequence[bytes | str],
    materialization_evidence: Mapping[str, Any],
    *,
    actual_training_jsonl_sha256: str,
    expected_materialization_identity_sha256: str,
    expected_training_jsonl_sha256: str,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare current retained rows with EVAL-647 while keeping durable output text-free."""
    actual_training_jsonl_sha256 = _require_sha256(actual_training_jsonl_sha256, "actual training JSONL SHA-256")
    expected_training_jsonl_sha256 = _require_sha256(expected_training_jsonl_sha256, "expected training JSONL SHA-256")
    expected_materialization_identity_sha256 = _require_sha256(
        expected_materialization_identity_sha256, "expected materialization identity"
    )
    _require(
        actual_training_jsonl_sha256 == expected_training_jsonl_sha256,
        "training JSONL does not match independent current-corpus authority",
    )
    _require(materialization_evidence.get("schema_version") == MATERIALIZATION_SCHEMA, "materialization schema drift")
    _require(materialization_evidence.get("status") == MATERIALIZATION_STATUS, "materialization is not terminal zero-credit output")
    _require(materialization_evidence.get("execution_profile") == "LOCAL_FREE", "materialization execution profile drift")
    _require(
        materialization_evidence.get("materialization_identity_sha256") == expected_materialization_identity_sha256,
        "materialization identity is not independently expected",
    )
    result = materialization_evidence.get("result")
    _require(isinstance(result, Mapping), "materialization result missing")
    _require(result.get("record_payload_jsonl_sha256") == expected_training_jsonl_sha256, "materialization payload identity drift")
    _require(result.get("record_count") == len(training_records), "training row count does not match materialization evidence")
    truth = materialization_evidence.get("truth_boundary")
    _require(isinstance(truth, Mapping), "materialization truth boundary missing")
    _require(truth.get("final_test_outcomes_read") is False, "final-test outcomes were read")
    _require(truth.get("training_executed") is False, "input materialization already claims training")
    _require(truth.get("paid_compute_used") is False, "paid compute boundary widened")

    reserved, source_evidence_identity = _reserved_rows(manifest, reserved_payloads)
    t = _thresholds(thresholds)
    train_fp = [_fingerprint(row, t) for row in training_records]
    reserved_fp = [_fingerprint(row, t) for row in reserved]
    matches: list[dict[str, Any]] = []
    for left, right in sorted(_blocked_pairs(train_fp, reserved_fp)):
        matches.extend(_pair(train_fp[left], reserved_fp[right], "eval", t))

    public_matches: list[dict[str, Any]] = []
    for item in sorted(matches, key=lambda row: (row["train_record_id"], row["eval_record_id"], row["match_type"])):
        public = {key: value for key, value in item.items() if key not in {"train_record_id", "eval_record_id"}}
        public["train_record_id_sha256"] = _sha256(item["train_record_id"].encode("utf-8"))
        public["eval_record_id_sha256"] = _sha256(item["eval_record_id"].encode("utf-8"))
        public_matches.append(public)

    clean = not public_matches
    remaining = [
        "PROJECT_HISTORY_TOKENIZER_EXPOSURE_ZERO_PROVEN",
        "PROJECT_HISTORY_TRAINING_EXPOSURE_ZERO_PROVEN",
        "FUTURE_TRAINING_EXCLUSION_CONSUMED_BY_CORPUS_PIPELINE",
        "PURPOSE_SPECIFIC_EVALUATION_AUTHORITY_TERMINAL",
    ]
    if not clean:
        remaining.insert(2, "GLOBAL_CORPUS_EXACT_AND_NEAR_OVERLAP_ZERO_PROVEN")
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "PASS_CURRENT_CORPUS_OVERLAP_ONLY" if clean else "FAIL_CURRENT_CORPUS_OVERLAP",
        "matcher": {
            "algorithm": ALGORITHM,
            "normalization": NORMALIZATION,
            "code_skeleton": CODE_SKELETON,
            "thresholds": t,
        },
        "current_corpus": {
            "materialization_identity_sha256": expected_materialization_identity_sha256,
            "record_payload_jsonl_sha256": expected_training_jsonl_sha256,
            "record_count": len(training_records),
        },
        "eval647": {
            "source_materialization_evidence_identity_sha256": source_evidence_identity,
            "reserved_record_count": len(reserved),
        },
        "current_corpus_overlap_zero": clean,
        "match_evidence_count": len(public_matches),
        "match_evidence": public_matches,
        "completed_gate": "GLOBAL_CORPUS_EXACT_AND_NEAR_OVERLAP_ZERO_PROVEN" if clean else None,
        "remaining_successor_gates": remaining,
        "hash_only_evidence": True,
        "raw_payload_persisted_in_report": False,
        "selection_validation_records_authorized": 0,
        "truth_boundary": {
            "final_test_payload_accessed": False,
            "final_test_outcomes_read": False,
            "tokenizer_fit_authorized": False,
            "optimizer_updates_executed_on_real_targets": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "paid_compute_used": False,
        },
    }
    report["report_sha256"] = stable_identity("eval647-current-corpus-overlap-v1", report)
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    _require(report.get("schema_version") == SCHEMA, "report schema drift")
    body = dict(report)
    claimed = body.pop("report_sha256", None)
    _require(claimed == stable_identity("eval647-current-corpus-overlap-v1", body), "report self-hash mismatch")
    _require(report.get("hash_only_evidence") is True, "report is not hash-only")
    _require(report.get("raw_payload_persisted_in_report") is False, "raw payload persistence widened")
    _require(report.get("selection_validation_records_authorized") == 0, "overlap evidence cannot authorize evaluation records")
