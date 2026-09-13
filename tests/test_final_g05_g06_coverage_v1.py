from __future__ import annotations

import copy
import hashlib
import json

import pytest

from twelve_six.data import final_g05_g06_coverage_v1 as m


def _hash(document, field, *, newline=False):
    body = copy.deepcopy(document)
    body.pop(field, None)
    text = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if newline:
        text += "\n"
    return hashlib.sha256(text.encode()).hexdigest()


def _rid(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _record(record_id, digit, payload_bytes):
    return {
        "record_id": record_id,
        "source_id": f"src:{record_id}",
        "family": "family-a" if record_id != "r3" else "family-b",
        "modality": "en",
        "payload_sha256": digit * 64,
        "payload_bytes": payload_bytes,
        "comparison_policy_id": "cmp-v1",
        "comparison_sha256": hashlib.sha256(f"cmp:{record_id}".encode()).hexdigest(),
        "comparison_bytes": payload_bytes,
        "training_eligible": False,
        "evaluation_eligible": False,
    }


def _inventory():
    records = [_record("r1", "1", 11), _record("r2", "2", 13), _record("r3", "3", 17)]
    doc = {
        "schema": m.INVENTORY_SCHEMA,
        "record_count": 3,
        "retained_payload_bytes": 41,
        "records": records,
        "truth_boundary": {
            "training_eligible": False,
            "evaluation_eligible": False,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit": False,
            "optimizer_updates": 0,
            "model_training": False,
            "final_test_outcomes_accessed": False,
            "paid_compute_used": False,
        },
    }
    doc["inventory_identity_sha256"] = _hash(
        doc,
        "inventory_identity_sha256",
        newline=True,
    )
    return doc


RECORDS_SHA = "4" * 64
PRIVACY_POLICY = "5" * 64
PRIVACY_IMPL = "6" * 40


def _decontam(inv, excluded=("r2",)):
    excluded_hashes = [_rid(value) for value in excluded]
    excluded_bytes = sum(
        row["payload_bytes"] for row in inv["records"] if row["record_id"] in excluded
    )
    doc = {
        "schema": m.DECONTAM_SCHEMA,
        "retained_inventory_identity_sha256": inv["inventory_identity_sha256"],
        "records_jsonl_sha256": RECORDS_SHA,
        "verdict": "PASS_WITH_EXCLUSIONS" if excluded else "PASS_CLEAN",
        "input_record_count": 3,
        "input_payload_bytes": 41,
        "excluded_record_id_sha256": excluded_hashes,
        "survivor_record_count": 3 - len(excluded),
        "survivor_payload_bytes": 41 - excluded_bytes,
        "final_test_outcomes_read": False,
        "model_selection_performed": False,
        "training_authorized_by_this_report": False,
        "authorized_optimized_target_exposure": 0,
    }
    doc["decontamination_authority_sha256"] = _hash(
        doc,
        "decontamination_authority_sha256",
    )
    return doc


def _qualification(inv, record_ids, *, quality="ACCEPT", privacy="ALLOW", impl=PRIVACY_IMPL):
    by_id = {row["record_id"]: row for row in inv["records"]}
    rows = []
    for record_id in sorted(record_ids, key=_rid):
        source = by_id[record_id]
        rows.append(
            {
                "record_id_sha256": _rid(record_id),
                "payload_sha256": source["payload_sha256"],
                "payload_bytes": source["payload_bytes"],
                "quality_decision": quality,
                "privacy_action": privacy,
            }
        )
    doc = {
        "schema": m.QUALIFICATION_SCHEMA,
        "status": "PASS",
        "coverage_scope": m.QUALIFICATION_SCOPE,
        "quality_policy_identity_sha256": m.QUALITY_POLICY_SHA256,
        "quality_granularity_identity_sha256": m.QUALITY_GRANULARITY_SHA256,
        "privacy_policy_identity_sha256": PRIVACY_POLICY,
        "privacy_implementation_git_blob_sha": impl,
        "records": rows,
        "qualified_record_count": len(rows),
        "qualified_payload_bytes": sum(row["payload_bytes"] for row in rows),
        "final_test_outcomes_read": False,
        "training_authorized_by_this_authority": False,
        "authorized_optimized_target_exposure": 0,
    }
    doc["qualification_authority_identity_sha256"] = _hash(
        doc,
        "qualification_authority_identity_sha256",
    )
    return doc


def _build(inv=None, decontam=None, qualifications=None, expected_qualification_ids=None):
    inv = _inventory() if inv is None else inv
    decontam = _decontam(inv) if decontam is None else decontam
    if qualifications is None:
        qualifications = [_qualification(inv, ("r1", "r2", "r3"))]
    if expected_qualification_ids is None:
        expected_qualification_ids = [
            item["qualification_authority_identity_sha256"] for item in qualifications
        ]
    return m.build_final_g05_g06_coverage(
        retained_inventory=inv,
        expected_retained_inventory_identity_sha256=inv["inventory_identity_sha256"],
        decontamination_binding=decontam,
        expected_decontamination_authority_sha256=(
            decontam["decontamination_authority_sha256"]
        ),
        expected_records_jsonl_sha256=RECORDS_SHA,
        qualification_authorities=qualifications,
        expected_qualification_authority_identities_sha256=expected_qualification_ids,
        expected_privacy_policy_identity_sha256=PRIVACY_POLICY,
        expected_privacy_implementation_git_blob_sha=PRIVACY_IMPL,
    )


def test_projects_exact_final_survivors_and_matches_pr945_shape():
    result = _build()
    assert result["covered_record_count"] == 2
    assert result["covered_payload_bytes"] == 28
    assert {row["record_id_sha256"] for row in result["covered_records"]} == {
        _rid("r1"),
        _rid("r3"),
    }
    assert all(
        set(row) == {"record_id_sha256", "payload_sha256", "payload_bytes"}
        for row in result["covered_records"]
    )
    assert result["training_authorized_by_this_coverage"] is False
    assert result["authorized_optimized_target_exposure"] == 0


def test_partitioned_independent_authorities_are_supported():
    inv = _inventory()
    qualifications = [
        _qualification(inv, ("r1",)),
        _qualification(inv, ("r2", "r3")),
    ]
    assert _build(inv=inv, qualifications=qualifications)["covered_record_count"] == 2


def test_missing_final_survivor_fails_closed():
    inv = _inventory()
    qualifications = [_qualification(inv, ("r1", "r2"))]
    with pytest.raises(m.CoverageError, match="missing G05/G06"):
        _build(inv=inv, qualifications=qualifications)


def test_decontam_deleted_record_never_reenters_output():
    result = _build()
    assert _rid("r2") not in {row["record_id_sha256"] for row in result["covered_records"]}


def test_qualification_outside_inventory_fails_closed():
    inv = _inventory()
    qualification = _qualification(inv, ("r1", "r3"))
    qualification["records"].append(
        {
            "record_id_sha256": "a" * 64,
            "payload_sha256": "b" * 64,
            "payload_bytes": 3,
            "quality_decision": "ACCEPT",
            "privacy_action": "ALLOW",
        }
    )
    qualification["qualified_record_count"] += 1
    qualification["qualified_payload_bytes"] += 3
    qualification["qualification_authority_identity_sha256"] = _hash(
        qualification,
        "qualification_authority_identity_sha256",
    )
    with pytest.raises(m.CoverageError, match="outside retained inventory"):
        _build(inv=inv, qualifications=[qualification])


def test_payload_substitution_fails_closed():
    inv = _inventory()
    qualification = _qualification(inv, ("r1", "r2", "r3"))
    qualification["records"][0]["payload_sha256"] = "a" * 64
    qualification["qualification_authority_identity_sha256"] = _hash(
        qualification,
        "qualification_authority_identity_sha256",
    )
    with pytest.raises(m.CoverageError, match="payload SHA-256 mismatch"):
        _build(inv=inv, qualifications=[qualification])


@pytest.mark.parametrize(
    ("quality", "privacy", "message"),
    [("REJECT", "ALLOW", "non-accepted"), ("ACCEPT", "REDACT", "non-ALLOW")],
)
def test_nonterminal_g05_or_g06_decision_fails(quality, privacy, message):
    inv = _inventory()
    qualification = _qualification(
        inv,
        ("r1", "r2", "r3"),
        quality=quality,
        privacy=privacy,
    )
    with pytest.raises(m.CoverageError, match=message):
        _build(inv=inv, qualifications=[qualification])


def test_stale_privacy_implementation_fails_even_with_same_policy_hash():
    inv = _inventory()
    qualification = _qualification(inv, ("r1", "r2", "r3"), impl="7" * 40)
    with pytest.raises(m.CoverageError, match="implementation identity mismatch"):
        _build(inv=inv, qualifications=[qualification])


def test_self_consistent_qualification_substitution_fails_external_binding():
    inv = _inventory()
    qualification = _qualification(inv, ("r1", "r2", "r3"))
    original = qualification["qualification_authority_identity_sha256"]
    qualification["records"][0]["payload_bytes"] += 1
    qualification["qualified_payload_bytes"] += 1
    qualification["qualification_authority_identity_sha256"] = _hash(
        qualification,
        "qualification_authority_identity_sha256",
    )
    with pytest.raises(m.CoverageError, match="not independently expected"):
        _build(
            inv=inv,
            qualifications=[qualification],
            expected_qualification_ids=[original],
        )


def test_unknown_decontam_exclusion_fails_closed():
    inv = _inventory()
    decontam = _decontam(inv)
    decontam["excluded_record_id_sha256"] = ["a" * 64]
    decontam["decontamination_authority_sha256"] = _hash(
        decontam,
        "decontamination_authority_sha256",
    )
    with pytest.raises(m.CoverageError, match="unknown record"):
        _build(inv=inv, decontam=decontam)


def test_duplicate_cross_authority_coverage_fails_closed():
    inv = _inventory()
    qualifications = [
        _qualification(inv, ("r1", "r2")),
        _qualification(inv, ("r1", "r3")),
    ]
    with pytest.raises(m.CoverageError, match="multiple qualification"):
        _build(inv=inv, qualifications=qualifications)


def test_inventory_bool_byte_trap_fails_closed():
    inv = _inventory()
    inv["records"][0]["payload_bytes"] = True
    inv["retained_payload_bytes"] = 31
    inv["inventory_identity_sha256"] = _hash(
        inv,
        "inventory_identity_sha256",
        newline=True,
    )
    with pytest.raises(m.CoverageError, match="non-negative integer"):
        _build(inv=inv)


def test_durable_artifact_tamper_fails_self_hash():
    result = _build()
    expected = result["g05_g06_coverage_identity_sha256"]
    result["covered_payload_bytes"] += 1
    with pytest.raises(m.CoverageError):
        m.verify_final_g05_g06_coverage(
            result,
            expected_identity_sha256=expected,
            expected_retained_inventory_identity_sha256=(
                result["retained_inventory_identity_sha256"]
            ),
            expected_decontamination_authority_sha256=(
                result["decontamination_authority_sha256"]
            ),
            expected_records_jsonl_sha256=RECORDS_SHA,
            expected_privacy_policy_identity_sha256=PRIVACY_POLICY,
            expected_privacy_implementation_git_blob_sha=PRIVACY_IMPL,
        )


def test_verifier_rejects_privacy_implementation_substitution():
    result = _build()
    with pytest.raises(m.CoverageError, match="implementation identity mismatch"):
        m.verify_final_g05_g06_coverage(
            result,
            expected_identity_sha256=result["g05_g06_coverage_identity_sha256"],
            expected_retained_inventory_identity_sha256=(
                result["retained_inventory_identity_sha256"]
            ),
            expected_decontamination_authority_sha256=(
                result["decontamination_authority_sha256"]
            ),
            expected_records_jsonl_sha256=RECORDS_SHA,
            expected_privacy_policy_identity_sha256=PRIVACY_POLICY,
            expected_privacy_implementation_git_blob_sha="7" * 40,
        )


def _verify_durable(result):
    m.verify_final_g05_g06_coverage(
        result,
        expected_identity_sha256=result["g05_g06_coverage_identity_sha256"],
        expected_retained_inventory_identity_sha256=(
            result["retained_inventory_identity_sha256"]
        ),
        expected_decontamination_authority_sha256=(
            result["decontamination_authority_sha256"]
        ),
        expected_records_jsonl_sha256=RECORDS_SHA,
        expected_privacy_policy_identity_sha256=PRIVACY_POLICY,
        expected_privacy_implementation_git_blob_sha=PRIVACY_IMPL,
    )


def _reseal_durable(result):
    result["g05_g06_coverage_identity_sha256"] = _hash(
        result,
        "g05_g06_coverage_identity_sha256",
    )


def test_durable_verifier_accepts_builder_output_after_aud1041_hardening():
    _verify_durable(_build())


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("projection", "projection rule drift"),
        ("empty_roots", "roots missing"),
        ("duplicate_roots", "duplicate qualification authority root"),
        ("set_identity", "set identity drift"),
        ("record_accounting", "record projection accounting drift"),
        ("byte_accounting", "byte projection accounting drift"),
        ("bool_excluded_count", "non-negative integer"),
        ("bool_zero_exposure", "non-negative integer"),
        ("unknown_root", "root schema drift"),
    ],
)
def test_self_consistent_durable_semantic_rehash_fails_closed(case, message):
    result = _build()
    if case == "projection":
        result["projection_rule"] = "FORGED_PROJECTION_RULE"
    elif case == "empty_roots":
        result["qualification_authority_identities_sha256"] = []
        result["qualification_authority_set_identity_sha256"] = hashlib.sha256(
            b"[]"
        ).hexdigest()
    elif case == "duplicate_roots":
        root = result["qualification_authority_identities_sha256"][0]
        roots = [root, root]
        result["qualification_authority_identities_sha256"] = roots
        payload = json.dumps(roots, sort_keys=True, separators=(",", ":")).encode()
        result["qualification_authority_set_identity_sha256"] = hashlib.sha256(
            payload
        ).hexdigest()
    elif case == "set_identity":
        result["qualification_authority_set_identity_sha256"] = "f" * 64
    elif case == "record_accounting":
        result["predecontam_record_count"] += 1
    elif case == "byte_accounting":
        result["predecontam_payload_bytes"] += 1
    elif case == "bool_excluded_count":
        result["excluded_record_count"] = True
    elif case == "bool_zero_exposure":
        result["authorized_optimized_target_exposure"] = False
    elif case == "unknown_root":
        result["unsealed_semantic_field"] = "forged"
    else:  # pragma: no cover - parameterization is closed above.
        raise AssertionError(case)
    _reseal_durable(result)
    with pytest.raises(m.CoverageError, match=message):
        _verify_durable(result)
