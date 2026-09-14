from __future__ import annotations

import copy
import hashlib
import json

import pytest

from twelve_six.data import clean_g05_g06_authority_v1 as m


def _h(ch: str) -> str:
    return ch * 64


def _cjson(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _reseal(value: dict, identity_field: str) -> None:
    core = dict(value)
    core.pop(identity_field, None)
    value[identity_field] = hashlib.sha256(_cjson(core)).hexdigest()


def _replay(*, run_id=10, job_id=20, artifact_id=30, g05_identity=None):
    return m.build_replay_receipt(
        execution_head_git_sha="1" * 40,
        run_id=run_id,
        job_id=job_id,
        artifact_id=artifact_id,
        artifact_zip_sha256=_h("2"),
        upstream_clean_successor_authority_identity_sha256=_h("3"),
        provenance_quarantine_identity_sha256=m.EXPECTED_QUARANTINE_IDENTITY_SHA256,
        input_rows_sha256=_h("4"),
        survivor_evidence_identity_sha256=_h("5"),
        survivor_jsonl_sha256=_h("6"),
        survivor_record_inventory_sha256=_h("7"),
        survivor_payload_inventory_sha256=_h("8"),
        record_count=3,
        payload_bytes=33,
        source_object_count=3,
        g05_execution_identity_sha256=g05_identity or _h("a"),
        g05_execution_rows_sha256=_h("b"),
        g06_envelope_identity_sha256=_h("c"),
        g06_execution_identity_sha256=_h("d"),
        g06_execution_rows_sha256=_h("e"),
        g05_quality_threshold_policy_sha256=_h("f"),
        g05_quality_granularity_policy_sha256="0" * 64,
        g06_privacy_policy_sha256="1" * 64,
        known_quarantined_lineage_absent=True,
    )


def _preflight(replay, *, drop=0, redact=1):
    blockers = []
    if drop:
        blockers.append("POST_G05_G06_EXCLUSION_MATERIALIZATION_REQUIRED")
    if redact:
        blockers.append("G06_REDACTION_MATERIALIZATION_REQUIRED")
    drop_hashes = [_h("2")] if drop else []
    redact_hashes = [_h("3")] if redact else []
    unchanged_count = 3 - drop - redact
    unchanged_hashes = [_h(chr(ord("4") + index)) for index in range(unchanged_count)]
    core = {
        "schema": m.PREFLIGHT_SCHEMA,
        "status": (
            "BLOCKED_CURRENT_G05_G06_COMPOSITION"
            if blockers
            else "READY_FOR_FINAL_COVERAGE_BINDING"
        ),
        "input_rows_sha256": replay["clean_input"]["input_rows_sha256"],
        "survivor_evidence_identity_sha256": replay["clean_input"][
            "survivor_evidence_identity_sha256"
        ],
        "survivor_jsonl_sha256": replay["clean_input"]["survivor_jsonl_sha256"],
        "survivor_record_inventory_sha256": replay["clean_input"][
            "survivor_record_inventory_sha256"
        ],
        "survivor_payload_inventory_sha256": replay["clean_input"][
            "survivor_payload_inventory_sha256"
        ],
        "g05_execution_identity_sha256": replay["deterministic_execution"][
            "g05_execution_identity_sha256"
        ],
        "g06_envelope_identity_sha256": replay["deterministic_execution"][
            "g06_envelope_identity_sha256"
        ],
        "g06_execution_identity_sha256": replay["deterministic_execution"][
            "g06_execution_identity_sha256"
        ],
        "g06_terminal_qualification_identity_sha256": _h("9"),
        "g06_exact_byte_execution_terminal": True,
        "input_record_count": 3,
        "input_utf8_bytes": 33,
        "decision_matrix": [
            {
                "g05_status": "RETAIN_ALL",
                "g06_action": "ALLOW",
                "record_count": unchanged_count,
                "input_utf8_bytes": 11 * unchanged_count,
            }
        ],
        "drop_record_id_sha256": drop_hashes,
        "g05_partial_record_id_sha256": [],
        "g06_redaction_record_id_sha256": redact_hashes,
        "unchanged_allow_record_id_sha256": sorted(unchanged_hashes),
        "drop_record_count": drop,
        "g05_partial_record_count": 0,
        "g06_redaction_record_count": redact,
        "unchanged_allow_record_count": unchanged_count,
        "unchanged_allow_input_utf8_bytes": 11 * unchanged_count,
        "blockers": sorted(blockers),
        "truth_boundary": dict(m._PREFLIGHT_TRUTH),
    }
    result = dict(core)
    result["composition_preflight_identity_sha256"] = hashlib.sha256(
        _cjson(core)
    ).hexdigest()
    return result


def _build(a=None, b=None, preflight=None):
    a = _replay() if a is None else a
    b = _replay(run_id=11, job_id=21, artifact_id=31) if b is None else b
    preflight = _preflight(a) if preflight is None else preflight
    return m.build_clean_g05_g06_authority(
        replay_a=a,
        expected_replay_a_identity_sha256=a["replay_identity_sha256"],
        replay_b=b,
        expected_replay_b_identity_sha256=b["replay_identity_sha256"],
        composition_preflight=preflight,
        expected_composition_preflight_identity_sha256=preflight[
            "composition_preflight_identity_sha256"
        ],
        expected_terminal_g06_qualification_identity_sha256=_h("9"),
        expected_upstream_clean_successor_authority_identity_sha256=_h("3"),
    )


def test_two_replay_materialization_authority_is_zero_credit_and_scoped():
    authority = _build()
    assert authority["status"] == "QUALIFIED_FOR_POST_G05_G06_MATERIALIZATION"
    assert authority["two_replay_binding"]["replay_count"] == 2
    assert authority["truth_boundary"]["authorized_optimized_target_exposure"] == 0
    assert authority["truth_boundary"]["training_executed"] is False
    assert (
        authority["provenance_scope"]["whole_corpus_external_llm_cleanliness_claimed"]
        is False
    )
    assert m.verify_clean_g05_g06_authority(
        authority,
        expected_authority_identity_sha256=authority["authority_identity_sha256"],
        expected_upstream_clean_successor_authority_identity_sha256=_h("3"),
        expected_replay_a_identity_sha256=authority["two_replay_binding"][
            "replay_a_identity_sha256"
        ],
        expected_replay_b_identity_sha256=authority["two_replay_binding"][
            "replay_b_identity_sha256"
        ],
        expected_composition_preflight_identity_sha256=authority[
            "composition_preflight"
        ]["composition_preflight_identity_sha256"],
        expected_terminal_g06_qualification_identity_sha256=_h("9"),
    ) == authority["authority_identity_sha256"]


def test_no_materialization_decisions_routes_to_final_coverage_binding():
    a = _replay()
    authority = _build(a=a, preflight=_preflight(a, redact=0))
    assert authority["status"] == "QUALIFIED_FOR_FINAL_COVERAGE_BINDING"
    assert authority["composition_preflight"]["materialization_blockers"] == []


def test_replay_deterministic_projection_mismatch_fails_closed():
    a = _replay()
    b = _replay(run_id=11, job_id=21, artifact_id=31, g05_identity=_h("0"))
    with pytest.raises(m.CleanG05G06AuthorityError, match="two-replay mismatch"):
        _build(a=a, b=b, preflight=_preflight(a))


def test_duplicate_replay_execution_coordinates_fail_closed():
    a = _replay()
    b = _replay(run_id=10, job_id=20, artifact_id=31)
    with pytest.raises(
        m.CleanG05G06AuthorityError, match="distinct execution coordinates"
    ):
        _build(a=a, b=b, preflight=_preflight(a))


def test_duplicate_replay_artifact_fails_closed():
    a = _replay()
    b = _replay(run_id=11, job_id=21, artifact_id=30)
    with pytest.raises(m.CleanG05G06AuthorityError, match="distinct artifacts"):
        _build(a=a, b=b, preflight=_preflight(a))


def test_stale_contaminated_survivor_identity_is_rejected_at_receipt_construction():
    with pytest.raises(
        m.CleanG05G06AuthorityError,
        match="reuses invalidated contaminated survivor identity",
    ):
        m.build_replay_receipt(
            execution_head_git_sha="1" * 40,
            run_id=10,
            job_id=20,
            artifact_id=30,
            artifact_zip_sha256=_h("2"),
            upstream_clean_successor_authority_identity_sha256=_h("3"),
            provenance_quarantine_identity_sha256=m.EXPECTED_QUARANTINE_IDENTITY_SHA256,
            input_rows_sha256=_h("4"),
            survivor_evidence_identity_sha256=(
                "284d122a9afd4e5d4676d78a202d3001e5cf87438776022d3004d61fdf10e579"
            ),
            survivor_jsonl_sha256=_h("6"),
            survivor_record_inventory_sha256=_h("7"),
            survivor_payload_inventory_sha256=_h("8"),
            record_count=3,
            payload_bytes=33,
            source_object_count=3,
            g05_execution_identity_sha256=_h("a"),
            g05_execution_rows_sha256=_h("b"),
            g06_envelope_identity_sha256=_h("c"),
            g06_execution_identity_sha256=_h("d"),
            g06_execution_rows_sha256=_h("e"),
            g05_quality_threshold_policy_sha256=_h("f"),
            g05_quality_granularity_policy_sha256="0" * 64,
            g06_privacy_policy_sha256="1" * 64,
            known_quarantined_lineage_absent=True,
        )


def test_quarantine_authority_substitution_fails_closed():
    with pytest.raises(m.CleanG05G06AuthorityError, match="canonical authority"):
        kwargs = dict(
            execution_head_git_sha="1" * 40,
            run_id=10,
            job_id=20,
            artifact_id=30,
            artifact_zip_sha256=_h("2"),
            upstream_clean_successor_authority_identity_sha256=_h("3"),
            provenance_quarantine_identity_sha256=_h("0"),
            input_rows_sha256=_h("4"),
            survivor_evidence_identity_sha256=_h("5"),
            survivor_jsonl_sha256=_h("6"),
            survivor_record_inventory_sha256=_h("7"),
            survivor_payload_inventory_sha256=_h("8"),
            record_count=3,
            payload_bytes=33,
            source_object_count=3,
            g05_execution_identity_sha256=_h("a"),
            g05_execution_rows_sha256=_h("b"),
            g06_envelope_identity_sha256=_h("c"),
            g06_execution_identity_sha256=_h("d"),
            g06_execution_rows_sha256=_h("e"),
            g05_quality_threshold_policy_sha256=_h("f"),
            g05_quality_granularity_policy_sha256="0" * 64,
            g06_privacy_policy_sha256="1" * 64,
            known_quarantined_lineage_absent=True,
        )
        m.build_replay_receipt(**kwargs)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "g05_quality_execution_git_blob_sha1",
            "0" * 40,
            "G05 implementation substitution",
        ),
        (
            "g06_privacy_implementation_git_blob_sha1",
            "0" * 40,
            "G06 privacy implementation substitution",
        ),
    ],
)
def test_implementation_substitution_fails_closed(field, value, match):
    kwargs = {
        "execution_head_git_sha": "1" * 40,
        "run_id": 10,
        "job_id": 20,
        "artifact_id": 30,
        "artifact_zip_sha256": _h("2"),
        "upstream_clean_successor_authority_identity_sha256": _h("3"),
        "provenance_quarantine_identity_sha256": m.EXPECTED_QUARANTINE_IDENTITY_SHA256,
        "input_rows_sha256": _h("4"),
        "survivor_evidence_identity_sha256": _h("5"),
        "survivor_jsonl_sha256": _h("6"),
        "survivor_record_inventory_sha256": _h("7"),
        "survivor_payload_inventory_sha256": _h("8"),
        "record_count": 3,
        "payload_bytes": 33,
        "source_object_count": 3,
        "g05_execution_identity_sha256": _h("a"),
        "g05_execution_rows_sha256": _h("b"),
        "g06_envelope_identity_sha256": _h("c"),
        "g06_execution_identity_sha256": _h("d"),
        "g06_execution_rows_sha256": _h("e"),
        "g05_quality_threshold_policy_sha256": _h("f"),
        "g05_quality_granularity_policy_sha256": "0" * 64,
        "g06_privacy_policy_sha256": "1" * 64,
        "known_quarantined_lineage_absent": True,
        field: value,
    }
    with pytest.raises(m.CleanG05G06AuthorityError, match=match):
        m.build_replay_receipt(**kwargs)


def test_coherently_resealed_preflight_wrong_clean_root_fails_against_replay():
    a = _replay()
    preflight = _preflight(a)
    preflight["input_rows_sha256"] = _h("0")
    _reseal(preflight, "composition_preflight_identity_sha256")
    with pytest.raises(m.CleanG05G06AuthorityError, match="input_rows_sha256"):
        _build(a=a, preflight=preflight)


def test_preflight_provenance_scope_widening_fails_even_when_resealed():
    a = _replay()
    preflight = _preflight(a)
    preflight["truth_boundary"][
        "current_corpus_external_llm_free_claimed_by_this_preflight"
    ] = True
    _reseal(preflight, "composition_preflight_identity_sha256")
    with pytest.raises(m.CleanG05G06AuthorityError, match="truth scope widened"):
        _build(a=a, preflight=preflight)


def test_nonterminal_g06_qualification_fails_closed():
    a = _replay()
    preflight = _preflight(a)
    preflight["g06_exact_byte_execution_terminal"] = False
    preflight["g06_terminal_qualification_identity_sha256"] = None
    _reseal(preflight, "composition_preflight_identity_sha256")
    with pytest.raises(m.CleanG05G06AuthorityError, match="not terminal"):
        _build(a=a, preflight=preflight)


def test_g05_partial_decision_fails_closed_before_materialization():
    a = _replay()
    preflight = _preflight(a)
    preflight["g05_partial_record_count"] = 1
    preflight["g05_partial_record_id_sha256"] = [_h("f")]
    preflight["blockers"] = sorted(
        preflight["blockers"] + ["G05_PARTIAL_MATERIALIZATION_REQUIRED"]
    )
    _reseal(preflight, "composition_preflight_identity_sha256")
    with pytest.raises(m.CleanG05G06AuthorityError, match="not terminally materializable"):
        _build(a=a, preflight=preflight)


def test_extra_unknown_preflight_field_fails_closed():
    a = _replay()
    preflight = _preflight(a)
    preflight["caller_expected_root"] = _h("0")
    _reseal(preflight, "composition_preflight_identity_sha256")
    with pytest.raises(m.CleanG05G06AuthorityError, match="schema is not closed"):
        _build(a=a, preflight=preflight)


def test_final_authority_reseal_cannot_replace_independent_expected_root():
    authority = _build()
    original = authority["authority_identity_sha256"]
    mutated = copy.deepcopy(authority)
    mutated["provenance_scope"]["whole_corpus_external_llm_cleanliness_claimed"] = True
    _reseal(mutated, "authority_identity_sha256")
    with pytest.raises(m.CleanG05G06AuthorityError, match="not independently expected"):
        m.verify_clean_g05_g06_authority(
            mutated,
            expected_authority_identity_sha256=original,
            expected_upstream_clean_successor_authority_identity_sha256=_h("3"),
            expected_replay_a_identity_sha256=authority["two_replay_binding"][
                "replay_a_identity_sha256"
            ],
            expected_replay_b_identity_sha256=authority["two_replay_binding"][
                "replay_b_identity_sha256"
            ],
            expected_composition_preflight_identity_sha256=authority[
                "composition_preflight"
            ]["composition_preflight_identity_sha256"],
            expected_terminal_g06_qualification_identity_sha256=_h("9"),
        )
