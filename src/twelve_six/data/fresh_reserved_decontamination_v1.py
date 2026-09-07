"""Fresh reserved-evaluation decontamination over a survivor-bound corpus graph."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from twelve_six.data._data232_decontamination_matching import (
    DecontaminationError,
    authority_composite_identity,
)
from twelve_six.data.decontamination_authority_v2 import (
    build_report as build_data232_report,
)
from twelve_six.data.decontamination_authority_v2 import (
    verify_report as verify_data232_report,
)
from twelve_six.data.postdedup_decontam_handoff_v1 import (
    PostDedupDecontamHandoffError,
    prepare_ephemeral_data232_rows,
)

SCHEMA = "12-6.fresh-reserved-decontamination.v1"
EXECUTION_PROFILE = "LOCAL_FREE"


class FreshReservedDecontaminationError(RuntimeError):
    """Fail-closed fresh decontamination integration error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FreshReservedDecontaminationError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_obj(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{label} must be lowercase SHA-256",
    )
    return value


def _require_git_sha(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{40}", value) is not None,
        f"{label} must be lowercase Git SHA",
    )
    return value


def _authority_identity(
    authorities: Mapping[str, Any],
    *,
    role: str,
    expected: str,
) -> str:
    expected_sha = _require_sha256(expected, f"expected_{role}_identity")
    observed = authority_composite_identity(authorities, {role})
    _require(observed == expected_sha, f"{role} authority identity mismatch")
    return expected_sha


def _validate_handoff(
    handoff: Mapping[str, Any],
    *,
    expected_inventory_identity_sha256: str,
    expected_survivor_authority_sha256: str,
) -> None:
    _require(isinstance(handoff, Mapping), "post-dedup handoff evidence missing")
    _require(
        handoff.get("postdedup_inventory_identity_sha256")
        == expected_inventory_identity_sha256,
        "post-dedup handoff inventory identity mismatch",
    )
    _require(
        handoff.get("input_survivor_authority_sha256")
        == expected_survivor_authority_sha256,
        "post-dedup handoff survivor authority mismatch",
    )
    _require(
        handoff.get("raw_text_persisted_in_evidence") is False,
        "post-dedup handoff persisted raw text",
    )
    _require(
        handoff.get("final_test_payload_accessed") is False,
        "post-dedup handoff accessed final-test payload",
    )
    _require(
        handoff.get("final_test_outcomes_accessed") is False,
        "post-dedup handoff accessed final-test outcomes",
    )
    _require(
        handoff.get("authorized_training_exposure") == 0,
        "post-dedup handoff already grants training exposure",
    )
    _require_sha256(handoff.get("handoff_identity_sha256"), "handoff_identity_sha256")


def execute_fresh_reserved_decontamination(
    inventory: Mapping[str, Any],
    comparison_payloads: Mapping[str, bytes],
    evaluation_records: Sequence[Mapping[str, Any]],
    authorities: Mapping[str, Any],
    *,
    expected_inventory_identity_sha256: str,
    expected_survivor_authority_sha256: str,
    selection_validation_identity: str,
    final_test_identity: str,
    postdedup_handoff_git_sha: str,
    data232_matcher_git_sha: str,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the incumbent matcher on the exact survivor-bound ephemeral corpus."""
    inventory_sha = _require_sha256(
        expected_inventory_identity_sha256,
        "expected_inventory_identity_sha256",
    )
    survivor_sha = _require_sha256(
        expected_survivor_authority_sha256,
        "expected_survivor_authority_sha256",
    )
    handoff_git = _require_git_sha(postdedup_handoff_git_sha, "postdedup_handoff_git_sha")
    matcher_git = _require_git_sha(data232_matcher_git_sha, "data232_matcher_git_sha")
    _require(
        inventory.get("input_survivor_authority_sha256") == survivor_sha,
        "inventory survivor authority does not match external terminal authority",
    )

    selection_sha = _authority_identity(
        authorities,
        role="selection_validation",
        expected=selection_validation_identity,
    )
    final_sha = _authority_identity(
        authorities,
        role="final_test",
        expected=final_test_identity,
    )

    try:
        training_records, handoff = prepare_ephemeral_data232_rows(
            inventory,
            comparison_payloads,
            expected_inventory_identity_sha256=inventory_sha,
        )
        _validate_handoff(
            handoff,
            expected_inventory_identity_sha256=inventory_sha,
            expected_survivor_authority_sha256=survivor_sha,
        )
        data232_report = build_data232_report(
            training_records,
            evaluation_records,
            training_corpus_identity=inventory_sha,
            selection_validation_identity=selection_sha,
            final_test_identity=final_sha,
            authorities=authorities,
            thresholds=thresholds,
        )
        verify_data232_report(data232_report)
    except (PostDedupDecontamHandoffError, DecontaminationError) as exc:
        raise FreshReservedDecontaminationError(str(exc)) from exc

    _require(
        data232_report.get("status") in {"PASS_CLEAN", "PASS_WITH_EXCLUSIONS"},
        "DATA-232 returned non-terminal decontamination status",
    )
    _require(
        data232_report.get("training_corpus_identity") == inventory_sha,
        "DATA-232 training identity drift",
    )
    _require(
        data232_report.get("selection_validation_identity") == selection_sha,
        "DATA-232 selection-validation identity drift",
    )
    _require(
        data232_report.get("final_test_identity") == final_sha,
        "DATA-232 final-test reservation identity drift",
    )
    _require(
        data232_report.get("final_test_outcomes_read") is False,
        "DATA-232 read final-test outcomes",
    )
    _require(
        data232_report.get("training_executed") is False,
        "DATA-232 report claims training execution",
    )

    core: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": data232_report["status"],
        "execution_profile": EXECUTION_PROFILE,
        "upstream": {
            "postdedup_handoff_git_sha": handoff_git,
            "data232_matcher_git_sha": matcher_git,
            "postdedup_inventory_identity_sha256": inventory_sha,
            "survivor_authority_sha256": survivor_sha,
            "postdedup_handoff_identity_sha256": handoff["handoff_identity_sha256"],
            "selection_validation_identity_sha256": selection_sha,
            "final_test_reservation_identity_sha256": final_sha,
            "data232_report_sha256": data232_report["report_sha256"],
        },
        "counts": dict(data232_report["counts"]),
        "data232_report": data232_report,
        "raw_text_emitted": False,
        "final_test_payload_accessed": False,
        "final_test_outcomes_accessed": False,
        "reserved_evaluation_decontamination_complete": True,
        "authorized_training_exposure": 0,
        "post_composition_quality_privacy_complete": False,
        "balance_family_caps_complete": False,
        "cluster_safe_split_complete": False,
        "deterministic_packing_complete": False,
        "postpack_unique_loss_ledger_complete": False,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "optimizer_updates": 0,
        "paid_compute_used": False,
    }
    core["report_identity_sha256"] = _sha256_obj(core)
    return core


def verify_fresh_reserved_decontamination(
    report: Mapping[str, Any],
    *,
    expected_inventory_identity_sha256: str,
    expected_survivor_authority_sha256: str,
    selection_validation_identity: str,
    final_test_identity: str,
    postdedup_handoff_git_sha: str,
    data232_matcher_git_sha: str,
) -> None:
    """Verify durable output without trusting a self-consistent substitute report."""
    _require(isinstance(report, Mapping), "fresh decontamination report must be an object")
    _require(report.get("schema_version") == SCHEMA, "fresh decontamination schema drift")
    observed = _require_sha256(report.get("report_identity_sha256"), "report_identity_sha256")
    core = dict(report)
    core.pop("report_identity_sha256", None)
    _require(_sha256_obj(core) == observed, "fresh decontamination self-hash mismatch")

    upstream = report.get("upstream")
    _require(isinstance(upstream, Mapping), "fresh decontamination upstream binding missing")
    expected_inventory = _require_sha256(
        expected_inventory_identity_sha256,
        "expected_inventory_identity_sha256",
    )
    expected_survivor = _require_sha256(
        expected_survivor_authority_sha256,
        "expected_survivor_authority_sha256",
    )
    expected_selection = _require_sha256(
        selection_validation_identity,
        "selection_validation_identity",
    )
    expected_final = _require_sha256(final_test_identity, "final_test_identity")
    expected_handoff_git = _require_git_sha(
        postdedup_handoff_git_sha,
        "postdedup_handoff_git_sha",
    )
    expected_matcher_git = _require_git_sha(
        data232_matcher_git_sha,
        "data232_matcher_git_sha",
    )

    expected_bindings = {
        "postdedup_handoff_git_sha": expected_handoff_git,
        "data232_matcher_git_sha": expected_matcher_git,
        "postdedup_inventory_identity_sha256": expected_inventory,
        "survivor_authority_sha256": expected_survivor,
        "selection_validation_identity_sha256": expected_selection,
        "final_test_reservation_identity_sha256": expected_final,
    }
    for key, value in expected_bindings.items():
        _require(upstream.get(key) == value, f"fresh decontamination upstream drift: {key}")

    nested = report.get("data232_report")
    _require(isinstance(nested, Mapping), "DATA-232 report missing")
    try:
        verify_data232_report(nested)
    except DecontaminationError as exc:
        raise FreshReservedDecontaminationError(str(exc)) from exc
    _require(nested.get("status") == report.get("status"), "nested DATA-232 status drift")
    _require(
        nested.get("training_corpus_identity") == expected_inventory,
        "training identity drift",
    )
    _require(
        nested.get("selection_validation_identity") == expected_selection,
        "selection-validation identity drift",
    )
    _require(nested.get("final_test_identity") == expected_final, "final-test identity drift")
    _require(
        upstream.get("data232_report_sha256") == nested.get("report_sha256"),
        "nested DATA-232 report identity drift",
    )
    _require(report.get("counts") == nested.get("counts"), "fresh/nested count drift")

    _require(report.get("execution_profile") == EXECUTION_PROFILE, "execution profile drift")
    _require(report.get("raw_text_emitted") is False, "durable report leaked raw text")
    _require(
        report.get("final_test_payload_accessed") is False,
        "final-test payload accessed",
    )
    _require(report.get("final_test_outcomes_accessed") is False, "final-test outcomes accessed")
    _require(
        report.get("reserved_evaluation_decontamination_complete") is True,
        "decontam not complete",
    )
    _require(report.get("authorized_training_exposure") == 0, "training exposure fabricated")
    _require(report.get("tokenizer_fit_authorized") is False, "tokenizer authority fabricated")
    _require(report.get("model_training_executed") is False, "model training fabricated")
    _require(report.get("optimizer_updates") == 0, "optimizer updates fabricated")
    _require(report.get("paid_compute_used") is False, "paid compute fabricated")
