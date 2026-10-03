from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_TOOL = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "materialize_d03_final_g05_g06_coverage_v1.py"
)
_SPEC = importlib.util.spec_from_file_location("d03_g05_g06_materializer", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
m = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(m)

INPUT_ROOT = "1" * 64
SURVIVOR_EVIDENCE = "2" * 64
SURVIVOR_JSONL = "3" * 64
RECORD_ROOT = "4" * 64
PAYLOAD_ROOT = "5" * 64
PRIVACY_POLICY = "6" * 64
PRIVACY_IMPL = "7" * 40
_UNSET = object()


def _cjson(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _truth():
    return {
        "current_corpus_eligible": False,
        "training_authorized_bytes": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }


def _g05(statuses=("RETAIN_ALL", "REJECT_DOCUMENT")):
    rows = []
    for record_id, digit, nbytes, status in (
        ("r1", "a", 11, statuses[0]),
        ("r2", "b", 13, statuses[1]),
    ):
        retained = nbytes if status == "RETAIN_ALL" else 0
        rows.append(
            {
                "record_id": record_id,
                "mode": "en",
                "payload_sha256": digit * 64,
                "utf8_bytes": nbytes,
                "status": status,
                "retained_utf8_bytes": retained,
                "rejected_utf8_bytes": nbytes - retained,
            }
        )
    counts = {
        "records": 2,
        "retain_all": sum(row["status"] == "RETAIN_ALL" for row in rows),
        "retain_partial": sum(row["status"] == "RETAIN_PARTIAL" for row in rows),
        "reject_document": sum(row["status"] == "REJECT_DOCUMENT" for row in rows),
    }
    core = {
        "schema_version": m._G05_SCHEMA,
        "authority_class": m._G05_CLASS,
        "input_manifest_sha256": SURVIVOR_EVIDENCE,
        "input_rows_sha256": INPUT_ROOT,
        "execution_rows_sha256": _sha(_cjson(rows)),
        "counts": counts,
        "bytes": {
            "input_utf8_bytes": 24,
            "retained_utf8_bytes": sum(row["retained_utf8_bytes"] for row in rows),
            "rejected_utf8_bytes": sum(row["rejected_utf8_bytes"] for row in rows),
        },
        "records": rows,
        "truth_boundary": _truth(),
    }
    return {**core, "execution_identity_sha256": _sha(_cjson(core))}


def _g06(actions=("ALLOW", "QUARANTINE")):
    rows = []
    for record_id, digit, nbytes, action in (
        ("r1", "a", 11, actions[0]),
        ("r2", "b", 13, actions[1]),
    ):
        rows.append(
            {
                "record_id": record_id,
                "mode": "en",
                "payload_sha256": digit * 64,
                "utf8_bytes": nbytes,
                "action": action,
            }
        )
    counts = {key: 0 for key in ("ALLOW", "REDACT", "QUARANTINE", "EXCLUDE")}
    for row in rows:
        counts[row["action"]] += 1
    authority_core = {
        "schema_version": m._G06_SCHEMA,
        "authority_class": m._G06_CLASS,
        "expected_input_rows_sha256": INPUT_ROOT,
        "observed_input_rows_sha256": INPUT_ROOT,
        "privacy_binding": {
            "policy_sha256": PRIVACY_POLICY,
            "implementation_git_blob_sha1": PRIVACY_IMPL,
        },
        "execution_rows_sha256": _sha(_cjson(rows)),
        "counts": {
            "records": 2,
            "allow": counts["ALLOW"],
            "redact": counts["REDACT"],
            "quarantine": counts["QUARANTINE"],
            "exclude": counts["EXCLUDE"],
        },
        "total_input_utf8_bytes": 24,
        "records": rows,
        "truth_boundary": _truth(),
    }
    authority = {
        **authority_core,
        "execution_identity_sha256": _sha(_cjson(authority_core)),
    }
    envelope_core = {
        "schema_version": m._G06_ENVELOPE_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "dependency": {
            "survivor_materialization_evidence_identity_sha256": SURVIVOR_EVIDENCE,
            "survivor_record_payload_jsonl_sha256": SURVIVOR_JSONL,
            "survivor_record_inventory_digest_sha256": RECORD_ROOT,
            "survivor_payload_inventory_digest_sha256": PAYLOAD_ROOT,
            "survivor_record_count": 2,
            "survivor_total_payload_bytes": 24,
            "survivor_source_object_count": 2,
        },
        "g06_input_rows_sha256": INPUT_ROOT,
        "privacy_execution_authority": authority,
        "truth_boundary": _truth(),
    }
    return {**envelope_core, "evidence_identity_sha256": _sha(_cjson(envelope_core))}


def _reseal_g06(g06):
    authority = g06["privacy_execution_authority"]
    authority["execution_rows_sha256"] = _sha(_cjson(authority["records"]))
    authority_core = dict(authority)
    authority_core.pop("execution_identity_sha256", None)
    authority["execution_identity_sha256"] = _sha(_cjson(authority_core))
    envelope_core = dict(g06)
    envelope_core.pop("evidence_identity_sha256", None)
    g06["evidence_identity_sha256"] = _sha(_cjson(envelope_core))


def _terminal_qualification(g06):
    g06_identity = g06["privacy_execution_authority"]["execution_identity_sha256"]
    doc = {
        "schema": m._G06_TERMINAL_QUALIFICATION_SCHEMA,
        "status": m._G06_TERMINAL_QUALIFICATION_STATUS,
        "target_pr_number": 1,
        "target_head_git_sha": "8" * 40,
        "real_replay_head_git_sha": "9" * 40,
        "real_replay_run_id": 1,
        "real_replay_job_id": 2,
        "final_head_ci_run_id": 3,
        "final_head_ci_job_id": 4,
        "g06_envelope_identity_sha256": g06["evidence_identity_sha256"],
        "g06_execution_identity_sha256": g06_identity,
        "input_rows_sha256": INPUT_ROOT,
        "repeated_execution_evidence_sha256": g06["evidence_identity_sha256"],
        "artifact_id": 5,
        "artifact_zip_sha256": "a" * 64,
        "replay_record_count": 2,
        "replay_utf8_bytes": 24,
        "replay_count": 2,
        "independent_audit_issue_number": 6,
        "independent_audit_status": "PASS_FOR_INTEGRATION_RELEASED",
        "local_free_only": True,
        "head_change_invalidates": True,
        "truth_boundary": _truth(),
    }
    doc["qualification_identity_sha256"] = _sha(_cjson(doc))
    return doc


def _reseal_qualification(qualification):
    core = dict(qualification)
    core.pop("qualification_identity_sha256", None)
    qualification["qualification_identity_sha256"] = _sha(_cjson(core))


def _build(
    *,
    g05=None,
    g06=None,
    terminal=False,
    terminal_qualification=_UNSET,
    expected_terminal_qualification_identity=_UNSET,
):
    g05 = _g05() if g05 is None else g05
    g06 = _g06() if g06 is None else g06
    g06_identity = g06["privacy_execution_authority"]["execution_identity_sha256"]
    if terminal_qualification is _UNSET:
        terminal_qualification = _terminal_qualification(g06) if terminal else None
    if expected_terminal_qualification_identity is _UNSET:
        expected_terminal_qualification_identity = (
            terminal_qualification["qualification_identity_sha256"]
            if terminal_qualification is not None
            else None
        )
    return m.build_native_current_execution_preflight(
        g05_authority=g05,
        g06_execution_envelope=g06,
        expected_g05_execution_identity_sha256=g05["execution_identity_sha256"],
        expected_g06_envelope_identity_sha256=g06["evidence_identity_sha256"],
        expected_g06_execution_identity_sha256=g06_identity,
        g06_terminal_qualification=terminal_qualification,
        expected_g06_terminal_qualification_identity_sha256=(
            expected_terminal_qualification_identity
        ),
        expected_input_rows_sha256=INPUT_ROOT,
        expected_survivor_evidence_identity_sha256=SURVIVOR_EVIDENCE,
        expected_survivor_jsonl_sha256=SURVIVOR_JSONL,
        expected_survivor_record_inventory_sha256=RECORD_ROOT,
        expected_survivor_payload_inventory_sha256=PAYLOAD_ROOT,
        expected_survivor_record_count=2,
        expected_survivor_payload_bytes=24,
        expected_survivor_source_object_count=2,
        expected_privacy_policy_sha256=PRIVACY_POLICY,
        expected_privacy_implementation_git_blob_sha1=PRIVACY_IMPL,
    )


def test_native_mixed_outcome_is_blocked_without_fabricating_pass():
    result = _build()
    assert result["status"] == "BLOCKED_CURRENT_G05_G06_COMPOSITION"
    assert result["drop_record_count"] == 1
    assert result["unchanged_allow_record_count"] == 1
    assert result["blockers"] == [
        "G06_EXACT_BYTE_EXECUTION_AUTHORITY_NOT_TERMINAL",
        "POST_G05_G06_EXCLUSION_MATERIALIZATION_REQUIRED",
    ]
    assert result["truth_boundary"]["authorized_optimized_target_exposure"] == 0
    assert result["truth_boundary"]["training_executed"] is False


def test_terminal_clean_execution_is_only_ready_for_final_binding():
    result = _build(
        g05=_g05(("RETAIN_ALL", "RETAIN_ALL")),
        g06=_g06(("ALLOW", "ALLOW")),
        terminal=True,
    )
    assert result["status"] == "READY_FOR_FINAL_COVERAGE_BINDING"
    assert result["blockers"] == []
    assert result["truth_boundary"]["training_authorized_bytes"] == 0


def test_clean_execution_without_terminal_qualification_stays_blocked():
    result = _build(
        g05=_g05(("RETAIN_ALL", "RETAIN_ALL")),
        g06=_g06(("ALLOW", "ALLOW")),
    )
    assert result["status"] == "BLOCKED_CURRENT_G05_G06_COMPOSITION"
    assert result["blockers"] == ["G06_EXACT_BYTE_EXECUTION_AUTHORITY_NOT_TERMINAL"]


def test_redaction_requires_materialization_instead_of_false_allow():
    result = _build(
        g05=_g05(("RETAIN_ALL", "RETAIN_ALL")),
        g06=_g06(("ALLOW", "REDACT")),
        terminal=True,
    )
    assert result["g06_redaction_record_count"] == 1
    assert result["drop_record_count"] == 0
    assert result["blockers"] == ["G06_REDACTION_MATERIALIZATION_REQUIRED"]


def test_bool_alias_in_native_count_fails_closed():
    g06 = _g06()
    g06["privacy_execution_authority"]["counts"]["records"] = True
    _reseal_g06(g06)
    with pytest.raises(m.NativeCompositionError, match="exact integer"):
        _build(g06=g06)


def test_cross_execution_payload_substitution_fails_closed():
    g06 = _g06()
    g06["privacy_execution_authority"]["records"][0]["payload_sha256"] = "f" * 64
    _reseal_g06(g06)
    with pytest.raises(m.NativeCompositionError, match="payload binding drift"):
        _build(g06=g06)


def test_terminal_qualification_document_and_expected_identity_are_atomic():
    g06 = _g06(("ALLOW", "ALLOW"))
    qualification = _terminal_qualification(g06)
    with pytest.raises(m.NativeCompositionError, match="provided together"):
        _build(
            g05=_g05(("RETAIN_ALL", "RETAIN_ALL")),
            g06=g06,
            terminal_qualification=qualification,
            expected_terminal_qualification_identity=None,
        )
    with pytest.raises(m.NativeCompositionError, match="provided together"):
        _build(
            g05=_g05(("RETAIN_ALL", "RETAIN_ALL")),
            g06=g06,
            terminal_qualification=None,
            expected_terminal_qualification_identity="a" * 64,
        )


def test_terminal_qualification_requires_independently_expected_identity():
    g06 = _g06(("ALLOW", "ALLOW"))
    qualification = _terminal_qualification(g06)
    with pytest.raises(m.NativeCompositionError, match="not independently expected"):
        _build(
            g05=_g05(("RETAIN_ALL", "RETAIN_ALL")),
            g06=g06,
            terminal_qualification=qualification,
            expected_terminal_qualification_identity="b" * 64,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    (
        ("g06_envelope_identity_sha256", "c" * 64, "envelope lineage drift"),
        ("g06_execution_identity_sha256", "d" * 64, "execution lineage drift"),
        ("input_rows_sha256", "e" * 64, "input lineage drift"),
    ),
)
def test_coherently_resealed_terminal_qualification_lineage_substitution_fails_closed(
    field, replacement, match
):
    g06 = _g06(("ALLOW", "ALLOW"))
    qualification = _terminal_qualification(g06)
    qualification[field] = replacement
    _reseal_qualification(qualification)
    with pytest.raises(m.NativeCompositionError, match=match):
        _build(
            g05=_g05(("RETAIN_ALL", "RETAIN_ALL")),
            g06=g06,
            terminal_qualification=qualification,
            expected_terminal_qualification_identity=qualification[
                "qualification_identity_sha256"
            ],
        )


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    (
        (
            "independent_audit_status",
            "CHANGES_REQUIRED",
            "independent audit is not released PASS",
        ),
        ("replay_count", 1, "requires two independent replays"),
    ),
)
def test_coherently_resealed_terminal_qualification_status_drift_fails_closed(
    field, replacement, match
):
    g06 = _g06(("ALLOW", "ALLOW"))
    qualification = _terminal_qualification(g06)
    qualification[field] = replacement
    _reseal_qualification(qualification)
    with pytest.raises(m.NativeCompositionError, match=match):
        _build(
            g05=_g05(("RETAIN_ALL", "RETAIN_ALL")),
            g06=g06,
            terminal_qualification=qualification,
            expected_terminal_qualification_identity=qualification[
                "qualification_identity_sha256"
            ],
        )


def test_terminal_qualification_widened_truth_boundary_fails_closed():
    g06 = _g06(("ALLOW", "ALLOW"))
    qualification = _terminal_qualification(g06)
    qualification["truth_boundary"]["authorized_optimized_target_exposure"] = 1
    _reseal_qualification(qualification)
    with pytest.raises(m.NativeCompositionError, match="truth boundary drift"):
        _build(
            g05=_g05(("RETAIN_ALL", "RETAIN_ALL")),
            g06=g06,
            terminal_qualification=qualification,
            expected_terminal_qualification_identity=qualification[
                "qualification_identity_sha256"
            ],
        )


def test_terminal_qualification_unknown_field_fails_closed():
    g06 = _g06(("ALLOW", "ALLOW"))
    qualification = _terminal_qualification(g06)
    qualification["unexpected"] = True
    with pytest.raises(m.NativeCompositionError, match="schema is not closed"):
        _build(
            g05=_g05(("RETAIN_ALL", "RETAIN_ALL")),
            g06=g06,
            terminal_qualification=qualification,
            expected_terminal_qualification_identity=qualification[
                "qualification_identity_sha256"
            ],
        )
