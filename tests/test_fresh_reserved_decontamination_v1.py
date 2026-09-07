from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from twelve_six.data._data232_decontamination_matching import authority_composite_identity
from twelve_six.data.fresh_reserved_decontamination_v1 import (
    FreshReservedDecontaminationError,
    execute_fresh_reserved_decontamination,
    verify_fresh_reserved_decontamination,
)
from twelve_six.data.postdedup_inventory_v1 import (
    OUTPUT_SCHEMA,
    SELECTION_POLICY,
    SURVIVOR_SELECTION_RULE,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _inventory(payload: bytes = b"alpha beta gamma") -> dict[str, object]:
    core: dict[str, object] = {
        "schema_version": OUTPUT_SCHEMA,
        "selection_policy": SELECTION_POLICY,
        "upstream_survivor_selection_rule": SURVIVOR_SELECTION_RULE,
        "input_survivor_authority_sha256": _sha("survivor"),
        "retained_source_count": 1,
        "retained_sources": [
            {
                "source_id": "source-1",
                "source_family": "family-1",
                "modality": "en",
                "comparison_policy": "UTF8_EXACT_V1",
                "comparison_payload_bytes": len(payload),
                "comparison_payload_sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
        "raw_text_emitted": False,
        "reserved_evaluation_decontamination_complete": False,
        "authorized_training_exposure": 0,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "final_test_payload_read": False,
        "paid_compute_used": False,
    }
    core["inventory_identity_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    return core


def _authorities() -> tuple[dict[str, object], str, str]:
    value: dict[str, object] = {
        "authorities": [
            {
                "authority_id": "eval-selection",
                "identity_sha256": _sha("selection-source"),
                "role": "selection_validation",
                "source_sha": "1" * 40,
            },
            {
                "authority_id": "eval-final",
                "identity_sha256": _sha("final-source"),
                "role": "final_test",
                "source_sha": "2" * 40,
            },
        ]
    }
    selection = authority_composite_identity(value, {"selection_validation"})
    final = authority_composite_identity(value, {"final_test"})
    assert selection is not None and final is not None
    return value, selection, final


def _run(eval_text: str = "totally unrelated evaluation sentence") -> dict[str, object]:
    payload = b"alpha beta gamma"
    inventory = _inventory(payload)
    authorities, selection, final = _authorities()
    return execute_fresh_reserved_decontamination(
        inventory,
        {"source-1": payload},
        [
            {
                "record_id": "eval-1",
                "source_id": "eval-source",
                "source_family": "eval-family",
                "modality": "en",
                "text": eval_text,
            }
        ],
        authorities,
        expected_inventory_identity_sha256=str(inventory["inventory_identity_sha256"]),
        expected_survivor_authority_sha256=_sha("survivor"),
        selection_validation_identity=selection,
        final_test_identity=final,
        postdedup_handoff_git_sha="a" * 40,
        data232_matcher_git_sha="b" * 40,
    )


def _verify(report: dict[str, object]) -> None:
    _, selection, final = _authorities()
    verify_fresh_reserved_decontamination(
        report,
        expected_inventory_identity_sha256=str(
            report["upstream"]["postdedup_inventory_identity_sha256"]  # type: ignore[index]
        ),
        expected_survivor_authority_sha256=_sha("survivor"),
        selection_validation_identity=selection,
        final_test_identity=final,
        postdedup_handoff_git_sha="a" * 40,
        data232_matcher_git_sha="b" * 40,
    )


def test_clean_scan_is_zero_credit_and_verifiable() -> None:
    report = _run()
    assert report["status"] == "PASS_CLEAN"
    assert report["authorized_training_exposure"] == 0
    assert report["reserved_evaluation_decontamination_complete"] is True
    _verify(report)


def test_overlap_is_excluded_without_granting_training_authority() -> None:
    report = _run("alpha beta gamma")
    assert report["status"] == "PASS_WITH_EXCLUSIONS"
    assert report["counts"]["excluded_training_records"] == 1  # type: ignore[index]
    assert report["authorized_training_exposure"] == 0
    _verify(report)


def test_external_survivor_authority_mismatch_fails_closed() -> None:
    payload = b"alpha beta gamma"
    inventory = _inventory(payload)
    authorities, selection, final = _authorities()
    with pytest.raises(FreshReservedDecontaminationError, match="survivor authority"):
        execute_fresh_reserved_decontamination(
            inventory,
            {"source-1": payload},
            [
                {
                    "record_id": "e",
                    "source_id": "e",
                    "source_family": "e",
                    "modality": "en",
                    "text": "x",
                }
            ],
            authorities,
            expected_inventory_identity_sha256=str(
                inventory["inventory_identity_sha256"]
            ),
            expected_survivor_authority_sha256=_sha("other-survivor"),
            selection_validation_identity=selection,
            final_test_identity=final,
            postdedup_handoff_git_sha="a" * 40,
            data232_matcher_git_sha="b" * 40,
        )


def test_outcome_bearing_evaluation_authority_fails_closed() -> None:
    payload = b"alpha beta gamma"
    inventory = _inventory(payload)
    authorities, selection, final = _authorities()
    authorities["benchmark_score"] = 0.9
    with pytest.raises(FreshReservedDecontaminationError, match="outcome-bearing"):
        execute_fresh_reserved_decontamination(
            inventory,
            {"source-1": payload},
            [
                {
                    "record_id": "e",
                    "source_id": "e",
                    "source_family": "e",
                    "modality": "en",
                    "text": "x",
                }
            ],
            authorities,
            expected_inventory_identity_sha256=str(
                inventory["inventory_identity_sha256"]
            ),
            expected_survivor_authority_sha256=_sha("survivor"),
            selection_validation_identity=selection,
            final_test_identity=final,
            postdedup_handoff_git_sha="a" * 40,
            data232_matcher_git_sha="b" * 40,
        )


def test_self_consistent_training_promotion_is_rejected() -> None:
    report = deepcopy(_run())
    report["authorized_training_exposure"] = 1
    core = dict(report)
    core.pop("report_identity_sha256")
    report["report_identity_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    with pytest.raises(FreshReservedDecontaminationError, match="training exposure"):
        _verify(report)


def test_self_consistent_upstream_substitution_is_rejected() -> None:
    report = deepcopy(_run())
    report["upstream"]["data232_matcher_git_sha"] = "c" * 40  # type: ignore[index]
    core = dict(report)
    core.pop("report_identity_sha256")
    report["report_identity_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    with pytest.raises(FreshReservedDecontaminationError, match="data232_matcher_git_sha"):
        _verify(report)


def test_report_tamper_without_rehash_is_rejected() -> None:
    report = deepcopy(_run())
    report["paid_compute_used"] = True
    with pytest.raises(FreshReservedDecontaminationError, match="self-hash"):
        _verify(report)


def test_invalid_git_binding_is_rejected_before_execution() -> None:
    payload = b"alpha beta gamma"
    inventory = _inventory(payload)
    authorities, selection, final = _authorities()
    with pytest.raises(FreshReservedDecontaminationError, match="Git SHA"):
        execute_fresh_reserved_decontamination(
            inventory,
            {"source-1": payload},
            [
                {
                    "record_id": "e",
                    "source_id": "e",
                    "source_family": "e",
                    "modality": "en",
                    "text": "x",
                }
            ],
            authorities,
            expected_inventory_identity_sha256=str(
                inventory["inventory_identity_sha256"]
            ),
            expected_survivor_authority_sha256=_sha("survivor"),
            selection_validation_identity=selection,
            final_test_identity=final,
            postdedup_handoff_git_sha="not-a-sha",
            data232_matcher_git_sha="b" * 40,
        )
