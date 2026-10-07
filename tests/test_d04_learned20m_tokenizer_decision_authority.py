from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

from twelve_six.data.balanced_split_application_v1 import (
    APPLICATION_SCHEMA,
    CANONICAL_SPLIT_GIT_BLOB_SHA1,
    CANONICAL_SPLIT_SPEC_IDENTITY_SHA256,
    SELECTION_SCHEMA,
)
from twelve_six.tokenization.byte import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
)
from twelve_six.tokenization.decision_authority import (
    DECISION,
    STATUS,
    TokenizerDecisionError,
    bind_byte_baseline_decision,
    verify_byte_baseline_decision,
)

_ZERO_BOUNDARY = {
    "training_eligible": False,
    "evaluation_eligible": False,
    "tokenizer_fit_authorized": False,
    "model_training_authorized": False,
    "paid_compute_authorized": False,
    "final_test_outcomes_read": False,
    "authorized_optimized_target_exposure": 0,
}
_UPSTREAMS = {
    "retained_inventory_identity_sha256": "1" * 64,
    "decontamination_authority_sha256": "2" * 64,
    "dedup_authority_sha256": "3" * 64,
    "balance_policy_identity_sha256": "4" * 64,
    "balance_result_identity_sha256": "5" * 64,
}


def _self_hash(value: dict[str, Any], identity_field: str) -> str:
    core = dict(value)
    core.pop(identity_field, None)
    payload = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _selection(seed: str = "a") -> dict[str, Any]:
    rows = [
        {
            "record_id": f"r{i}-{seed}",
            "source_id": f"source-{i}",
            "family": "family-a" if i < 2 else "family-b",
            "stratum": ("uk", "ua", "en", "code")[i],
            "modality": "text",
            "payload_sha256": str(i + 6) * 64,
            "payload_bytes": i + 1,
            "near_duplicate_cluster_id": f"cluster-{i}",
            "purpose": "pretraining",
            "training_eligible": False,
            "evaluation_eligible": False,
            "evaluation_reserved": False,
        }
        for i in range(4)
    ]
    core: dict[str, Any] = {
        "schema": SELECTION_SCHEMA,
        "terminal": True,
        "status": "PASS",
        **_UPSTREAMS,
        "records": rows,
        "totals": {
            "record_count": 4,
            "source_bytes": 10,
            "family_source_bytes": {"family-a": 3, "family-b": 7},
            "stratum_source_bytes": {"code": 4, "en": 3, "ua": 2, "uk": 1},
        },
        "claim_boundary": _ZERO_BOUNDARY,
    }
    core["balanced_selection_identity_sha256"] = _self_hash(
        core, "balanced_selection_identity_sha256"
    )
    return core


def _application(selection: dict[str, Any]) -> dict[str, Any]:
    totals = selection["totals"]
    core: dict[str, Any] = {
        "schema": APPLICATION_SCHEMA,
        "status": "PASS_ZERO_CREDIT",
        "balanced_selection_identity_sha256": selection[
            "balanced_selection_identity_sha256"
        ],
        **{field: selection[field] for field in _UPSTREAMS},
        "canonical_split_git_blob_sha1": CANONICAL_SPLIT_GIT_BLOB_SHA1,
        "split_spec_identity_sha256": CANONICAL_SPLIT_SPEC_IDENTITY_SHA256,
        "selected_record_count": totals["record_count"],
        "selected_source_bytes": totals["source_bytes"],
        "selected_family_source_bytes": totals["family_source_bytes"],
        "selected_stratum_source_bytes": totals["stratum_source_bytes"],
        "split_family": {"schema": "test-only-split-family"},
        "claim_boundary": _ZERO_BOUNDARY,
    }
    core["application_identity_sha256"] = _self_hash(
        core, "application_identity_sha256"
    )
    return core


def _kwargs(selection: dict[str, Any], application: dict[str, Any]) -> dict[str, str]:
    return {
        "expected_selection_identity_sha256": selection[
            "balanced_selection_identity_sha256"
        ],
        "expected_application_identity_sha256": application["application_identity_sha256"],
        "expected_retained_inventory_identity_sha256": _UPSTREAMS[
            "retained_inventory_identity_sha256"
        ],
        "expected_decontamination_authority_sha256": _UPSTREAMS[
            "decontamination_authority_sha256"
        ],
        "expected_dedup_authority_sha256": _UPSTREAMS["dedup_authority_sha256"],
        "expected_balance_policy_identity_sha256": _UPSTREAMS[
            "balance_policy_identity_sha256"
        ],
        "expected_balance_result_identity_sha256": _UPSTREAMS[
            "balance_result_identity_sha256"
        ],
    }


def _bind() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    selection = _selection()
    application = _application(selection)
    report = bind_byte_baseline_decision(selection, application, **_kwargs(selection, application))
    return selection, application, report


def _verify(
    report: dict[str, Any], selection: dict[str, Any], application: dict[str, Any]
) -> None:
    verify_byte_baseline_decision(
        report,
        selection,
        application,
        **_kwargs(selection, application),
    )


def test_terminal_decision_binds_canonical_lineage_and_zero_authority() -> None:
    selection, application, report = _bind()

    assert report["status"] == STATUS
    assert report["decision"] == DECISION
    assert report["split_spec_identity_sha256"] == CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    assert report["tokenizer_version"] == BYTE_TOKENIZER_VERSION
    assert report["tokenizer_config_sha256"] == BYTE_TOKENIZER_HASH
    assert report["tokenizer_vocab_sha256"] == BYTE_VOCAB_HASH
    assert report["vocab_size"] == 256
    assert report["tokenizer_fit_executed"] is False
    assert report["training_authorized_by_this_report"] is False
    assert report["compute_authorized_by_this_report"] is False
    assert report["authorized_optimized_target_exposure"] == 0
    _verify(report, selection, application)


def test_decision_is_deterministic_for_same_exact_upstreams() -> None:
    selection, application, first = _bind()
    second = bind_byte_baseline_decision(
        selection,
        application,
        **_kwargs(selection, application),
    )
    assert first == second


def test_mixed_selection_and_split_lineages_fail_closed() -> None:
    selection_a = _selection("a")
    selection_b = _selection("b")
    application_b = _application(selection_b)
    kwargs = _kwargs(selection_a, application_b)

    with pytest.raises(TokenizerDecisionError, match="different balanced selection"):
        bind_byte_baseline_decision(selection_a, application_b, **kwargs)


def test_selection_rehash_cannot_replace_external_expected_identity() -> None:
    selection, application, _ = _bind()
    expected = _kwargs(selection, application)
    selection["records"][0]["source_id"] = "forged"
    selection["balanced_selection_identity_sha256"] = _self_hash(
        selection, "balanced_selection_identity_sha256"
    )

    with pytest.raises(TokenizerDecisionError, match="expected identity"):
        bind_byte_baseline_decision(selection, application, **expected)


def test_split_rehash_cannot_replace_external_expected_identity() -> None:
    selection, application, _ = _bind()
    expected = _kwargs(selection, application)
    application["split_family"] = {"schema": "forged"}
    application["application_identity_sha256"] = _self_hash(
        application, "application_identity_sha256"
    )

    with pytest.raises(TokenizerDecisionError, match="identity mismatch"):
        bind_byte_baseline_decision(selection, application, **expected)


def test_missing_split_spec_identity_fails_closed() -> None:
    selection, application, _ = _bind()
    del application["split_spec_identity_sha256"]

    with pytest.raises(TokenizerDecisionError, match="closed-world"):
        bind_byte_baseline_decision(
            selection,
            application,
            **_kwargs(selection, application),
        )


@pytest.mark.parametrize("bad_value", ["a" * 64, "A" * 64, "0" * 63, True, None])
def test_split_spec_identity_is_canonical_not_caller_selected(bad_value: object) -> None:
    selection, application, _ = _bind()
    application["split_spec_identity_sha256"] = bad_value
    application["application_identity_sha256"] = _self_hash(
        application, "application_identity_sha256"
    )
    kwargs = _kwargs(selection, application)

    with pytest.raises(TokenizerDecisionError):
        bind_byte_baseline_decision(selection, application, **kwargs)


def test_self_resealed_split_spec_substitution_fails_against_canonical_authority() -> None:
    selection, application, _ = _bind()
    application["split_spec_identity_sha256"] = "a" * 64
    application["application_identity_sha256"] = _self_hash(
        application, "application_identity_sha256"
    )
    kwargs = _kwargs(selection, application)

    with pytest.raises(TokenizerDecisionError, match="canonical split spec authority"):
        bind_byte_baseline_decision(selection, application, **kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("decision", "FIT_BPE"),
        ("split_spec_identity_sha256", "0" * 64),
        ("tokenizer_version", "forged"),
        ("tokenizer_config_sha256", "0" * 64),
        ("tokenizer_vocab_sha256", "1" * 64),
        ("vocab_size", 257),
        ("tokenizer_fit_executed", True),
        ("training_authorized_by_this_report", True),
        ("compute_authorized_by_this_report", True),
        ("authorized_optimized_target_exposure", 1),
        ("authorized_optimized_target_exposure", False),
    ],
)
def test_report_tamper_fails_closed(field: str, value: object) -> None:
    selection, application, report = _bind()
    tampered = copy.deepcopy(report)
    tampered[field] = value

    with pytest.raises(TokenizerDecisionError):
        _verify(tampered, selection, application)


def test_report_self_hash_tamper_fails_closed() -> None:
    selection, application, report = _bind()
    report["decision_identity_sha256"] = "f" * 64

    with pytest.raises(TokenizerDecisionError, match="identity mismatch"):
        _verify(report, selection, application)


def test_report_reseal_cannot_replace_canonical_split_spec_identity() -> None:
    selection, application, report = _bind()
    report["split_spec_identity_sha256"] = "a" * 64
    report["decision_identity_sha256"] = _self_hash(report, "decision_identity_sha256")

    with pytest.raises(TokenizerDecisionError, match="split-spec authority identity drift"):
        _verify(report, selection, application)


def test_unknown_report_field_fails_closed_even_with_rehashed_identity() -> None:
    selection, application, report = _bind()
    report["training_authorized"] = True
    report["decision_identity_sha256"] = _self_hash(report, "decision_identity_sha256")

    with pytest.raises(TokenizerDecisionError, match="closed-world"):
        _verify(report, selection, application)


def test_split_accounting_drift_fails_even_when_application_rehashed() -> None:
    selection, application, _ = _bind()
    application["selected_source_bytes"] = 999
    application["application_identity_sha256"] = _self_hash(
        application, "application_identity_sha256"
    )
    kwargs = _kwargs(selection, application)

    with pytest.raises(TokenizerDecisionError, match="selected_source_bytes drift"):
        bind_byte_baseline_decision(selection, application, **kwargs)


def test_split_upstream_lineage_drift_fails_even_when_rehashed() -> None:
    selection, application, _ = _bind()
    application["dedup_authority_sha256"] = "a" * 64
    application["application_identity_sha256"] = _self_hash(
        application, "application_identity_sha256"
    )
    kwargs = _kwargs(selection, application)

    with pytest.raises(TokenizerDecisionError, match="dedup_authority_sha256 lineage mismatch"):
        bind_byte_baseline_decision(selection, application, **kwargs)


def test_split_truth_boundary_widening_fails_even_when_rehashed() -> None:
    selection, application, _ = _bind()
    application["claim_boundary"] = {**_ZERO_BOUNDARY, "tokenizer_fit_authorized": True}
    application["application_identity_sha256"] = _self_hash(
        application, "application_identity_sha256"
    )
    kwargs = _kwargs(selection, application)

    with pytest.raises(TokenizerDecisionError, match="truth boundary widened"):
        bind_byte_baseline_decision(selection, application, **kwargs)


@pytest.mark.parametrize("bad_hash", ["A" * 64, "0" * 63, "g" * 64, True, None])
def test_expected_selection_hash_must_fail_closed(bad_hash: object) -> None:
    selection, application, _ = _bind()
    kwargs: dict[str, object] = _kwargs(selection, application)
    kwargs["expected_selection_identity_sha256"] = bad_hash

    with pytest.raises(TokenizerDecisionError):
        bind_byte_baseline_decision(selection, application, **kwargs)  # type: ignore[arg-type]
