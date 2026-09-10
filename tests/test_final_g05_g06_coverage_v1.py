from __future__ import annotations

import copy
import hashlib
import json

import pytest

from twelve_six.data.final_g05_g06_coverage_v1 import (
    DECONTAM_SCHEMA,
    FINAL_SCOPE,
    INVENTORY_ZERO_TRUTH,
    PREDECONTAM_SCOPE,
    PRIVACY_AUTHORITY_SCHEMA,
    QUALITY_AUTHORITY_SCHEMA,
    QUALITY_GRANULARITY_POLICY_SHA256,
    QUALITY_THRESHOLD_POLICY_SHA256,
    ZERO_TRUTH,
    CoverageError,
    authority_identity,
    bind_final_g05_g06_coverage,
    verify_coverage_report,
)

GIT = "a" * 40


def H(ch: str) -> str:
    return ch * 64


PRIVACY_POLICY = H("9")


def _rid(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _seal(doc: dict, field: str) -> dict:
    doc[field] = authority_identity(doc, field)
    return doc


def _seal_decontam(doc: dict) -> dict:
    payload = dict(doc)
    payload.pop("decontamination_authority_sha256", None)
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    doc["decontamination_authority_sha256"] = hashlib.sha256(raw).hexdigest()
    return doc


def _inventory() -> dict:
    rows = [
        {
            "record_id": "r1",
            "source_id": "s1",
            "family": "f1",
            "modality": "text",
            "payload_sha256": H("1"),
            "payload_bytes": 10,
            "training_eligible": False,
            "evaluation_eligible": False,
        },
        {
            "record_id": "r2",
            "source_id": "s1",
            "family": "f1",
            "modality": "text",
            "payload_sha256": H("2"),
            "payload_bytes": 20,
            "training_eligible": False,
            "evaluation_eligible": False,
        },
        {
            "record_id": "r3",
            "source_id": "s3",
            "family": "f2",
            "modality": "code",
            "payload_sha256": H("3"),
            "payload_bytes": 30,
            "training_eligible": False,
            "evaluation_eligible": False,
        },
    ]
    return _seal(
        {
            "schema": "twelve-six.expanded-postdedup-inventory.v1",
            "record_count": 3,
            "retained_payload_bytes": 60,
            "records": rows,
            "truth_boundary": dict(INVENTORY_ZERO_TRUTH),
        },
        "inventory_identity_sha256",
    )


def _decontam(inv: dict, *, exclusions: list[str] | None = None) -> dict:
    excluded = exclusions or [_rid("r2")]
    survivor_bytes = sum(
        row["payload_bytes"] for row in inv["records"] if _rid(row["record_id"]) not in excluded
    )
    return _seal_decontam(
        {
            "schema": DECONTAM_SCHEMA,
            "retained_inventory_identity_sha256": inv["inventory_identity_sha256"],
            "verdict": "PASS_WITH_EXCLUSIONS" if excluded else "PASS_CLEAN",
            "input_record_count": len(inv["records"]),
            "input_payload_bytes": sum(row["payload_bytes"] for row in inv["records"]),
            "excluded_record_id_sha256": excluded,
            "survivor_record_count": len(inv["records"]) - len(excluded),
            "survivor_payload_bytes": survivor_bytes,
            "final_test_outcomes_read": False,
            "model_selection_performed": False,
            "training_authorized_by_this_report": False,
            "authorized_optimized_target_exposure": 0,
        }
    )


def _evidence_rows(inv: dict, record_ids: list[str], decision: str) -> list[dict]:
    source = {row["record_id"]: row for row in inv["records"]}
    return [
        {
            "record_id_sha256": _rid(record_id),
            "payload_sha256": source[record_id]["payload_sha256"],
            "payload_bytes": source[record_id]["payload_bytes"],
            "decision": decision,
        }
        for record_id in record_ids
    ]


def _quality(inv: dict, decontam: dict, *, scope: str = FINAL_SCOPE) -> dict:
    ids = ["r1", "r3"] if scope == FINAL_SCOPE else ["r1", "r2", "r3"]
    body = {
        "schema": QUALITY_AUTHORITY_SCHEMA,
        "coverage_scope": scope,
        "retained_inventory_identity_sha256": inv["inventory_identity_sha256"],
        "quality_threshold_policy_sha256": QUALITY_THRESHOLD_POLICY_SHA256,
        "quality_granularity_policy_sha256": QUALITY_GRANULARITY_POLICY_SHA256,
        "records": _evidence_rows(inv, ids, "RETAIN"),
        "truth_boundary": dict(ZERO_TRUTH),
    }
    if scope == FINAL_SCOPE:
        body["decontamination_authority_sha256"] = decontam[
            "decontamination_authority_sha256"
        ]
    return _seal(body, "quality_authority_identity_sha256")


def _privacy(inv: dict, decontam: dict, *, scope: str = FINAL_SCOPE) -> dict:
    ids = ["r1", "r3"] if scope == FINAL_SCOPE else ["r1", "r2", "r3"]
    body = {
        "schema": PRIVACY_AUTHORITY_SCHEMA,
        "coverage_scope": scope,
        "retained_inventory_identity_sha256": inv["inventory_identity_sha256"],
        "privacy_policy_sha256": PRIVACY_POLICY,
        "records": _evidence_rows(inv, ids, "ALLOW"),
        "truth_boundary": dict(ZERO_TRUTH),
    }
    if scope == FINAL_SCOPE:
        body["decontamination_authority_sha256"] = decontam[
            "decontamination_authority_sha256"
        ]
    return _seal(body, "privacy_authority_identity_sha256")


def _bind(inv: dict, dec: dict, quality: dict, privacy: dict) -> dict:
    return bind_final_g05_g06_coverage(
        inventory=inv,
        decontamination_binding=dec,
        quality_authority=quality,
        privacy_authority=privacy,
        expected_inventory_identity_sha256=inv["inventory_identity_sha256"],
        expected_decontamination_authority_sha256=dec[
            "decontamination_authority_sha256"
        ],
        expected_quality_authority_identity_sha256=quality[
            "quality_authority_identity_sha256"
        ],
        expected_privacy_authority_identity_sha256=privacy[
            "privacy_authority_identity_sha256"
        ],
        expected_privacy_policy_sha256=PRIVACY_POLICY,
        source_git_sha=GIT,
    )


def _fixture(scope: str = FINAL_SCOPE) -> tuple[dict, dict, dict, dict]:
    inv = _inventory()
    dec = _decontam(inv)
    return inv, dec, _quality(inv, dec, scope=scope), _privacy(inv, dec, scope=scope)


def test_final_scope_happy_path_is_deterministic_text_free_and_zero_credit() -> None:
    inv, dec, quality, privacy = _fixture()
    first = _bind(inv, dec, quality, privacy)
    second = _bind(inv, dec, quality, privacy)
    assert first == second
    assert first["survivor_record_count"] == 2
    assert first["survivor_payload_bytes"] == 40
    assert first["quality_subset_preservation_proved"] is False
    assert first["privacy_subset_preservation_proved"] is False
    assert first["truth_boundary"] == ZERO_TRUTH
    encoded = json.dumps(first)
    assert '"r1"' not in encoded and '"r3"' not in encoded


def test_predecontam_scope_proves_unchanged_subset() -> None:
    inv, dec, quality, privacy = _fixture(PREDECONTAM_SCOPE)
    report = _bind(inv, dec, quality, privacy)
    assert report["quality_subset_preservation_proved"] is True
    assert report["privacy_subset_preservation_proved"] is True
    assert report["survivor_payload_bytes"] == 40


@pytest.mark.parametrize("which", ["quality", "privacy"])
def test_missing_evidence_fails_closed(which: str) -> None:
    inv, dec, quality, privacy = _fixture()
    target = quality if which == "quality" else privacy
    target["records"].pop()
    field = f"{which}_authority_identity_sha256"
    _seal(target, field)
    with pytest.raises(CoverageError, match="exactly cover"):
        _bind(inv, dec, quality, privacy)


def test_extra_privacy_evidence_fails_closed() -> None:
    inv, dec, quality, privacy = _fixture()
    privacy["records"].append(_evidence_rows(inv, ["r2"], "ALLOW")[0])
    _seal(privacy, "privacy_authority_identity_sha256")
    with pytest.raises(CoverageError, match="exactly cover"):
        _bind(inv, dec, quality, privacy)


@pytest.mark.parametrize("field,value,match", [
    ("payload_sha256", H("8"), "payload SHA drift"),
    ("payload_bytes", 11, "payload byte drift"),
])
def test_quality_payload_drift_fails(field: str, value: object, match: str) -> None:
    inv, dec, quality, privacy = _fixture()
    quality["records"][0][field] = value
    _seal(quality, "quality_authority_identity_sha256")
    with pytest.raises(CoverageError, match=match):
        _bind(inv, dec, quality, privacy)


def test_duplicate_evidence_fails_even_with_resealed_authority() -> None:
    inv, dec, quality, privacy = _fixture()
    quality["records"].append(copy.deepcopy(quality["records"][0]))
    _seal(quality, "quality_authority_identity_sha256")
    with pytest.raises(CoverageError, match="duplicate record evidence"):
        _bind(inv, dec, quality, privacy)


def test_quality_policy_substitution_fails() -> None:
    inv, dec, quality, privacy = _fixture()
    quality["quality_threshold_policy_sha256"] = H("8")
    _seal(quality, "quality_authority_identity_sha256")
    with pytest.raises(CoverageError, match="quality threshold policy identity mismatch"):
        _bind(inv, dec, quality, privacy)


def test_granularity_policy_substitution_fails() -> None:
    inv, dec, quality, privacy = _fixture()
    quality["quality_granularity_policy_sha256"] = H("8")
    _seal(quality, "quality_authority_identity_sha256")
    with pytest.raises(CoverageError, match="quality granularity policy identity mismatch"):
        _bind(inv, dec, quality, privacy)


def test_privacy_policy_substitution_fails() -> None:
    inv, dec, quality, privacy = _fixture()
    privacy["privacy_policy_sha256"] = H("8")
    _seal(privacy, "privacy_authority_identity_sha256")
    with pytest.raises(CoverageError, match="privacy policy identity mismatch"):
        _bind(inv, dec, quality, privacy)


def test_self_hash_is_not_external_authority() -> None:
    inv, dec, quality, privacy = _fixture()
    attacker = copy.deepcopy(quality)
    attacker["records"][0]["payload_bytes"] = 999
    _seal(attacker, "quality_authority_identity_sha256")
    with pytest.raises(CoverageError, match="quality external identity mismatch"):
        bind_final_g05_g06_coverage(
            inventory=inv,
            decontamination_binding=dec,
            quality_authority=attacker,
            privacy_authority=privacy,
            expected_inventory_identity_sha256=inv["inventory_identity_sha256"],
            expected_decontamination_authority_sha256=dec[
                "decontamination_authority_sha256"
            ],
            expected_quality_authority_identity_sha256=quality[
                "quality_authority_identity_sha256"
            ],
            expected_privacy_authority_identity_sha256=privacy[
                "privacy_authority_identity_sha256"
            ],
            expected_privacy_policy_sha256=PRIVACY_POLICY,
            source_git_sha=GIT,
        )


def test_quality_reject_cannot_be_promoted() -> None:
    inv, dec, quality, privacy = _fixture()
    quality["records"][0]["decision"] = "REJECT"
    _seal(quality, "quality_authority_identity_sha256")
    with pytest.raises(CoverageError, match="non-RETAIN"):
        _bind(inv, dec, quality, privacy)


@pytest.mark.parametrize("action", ["REDACT", "QUARANTINE", "EXCLUDE"])
def test_privacy_non_allow_cannot_be_promoted(action: str) -> None:
    inv, dec, quality, privacy = _fixture()
    privacy["records"][0]["decision"] = action
    _seal(privacy, "privacy_authority_identity_sha256")
    with pytest.raises(CoverageError, match="non-ALLOW"):
        _bind(inv, dec, quality, privacy)


def test_nonterminal_decontamination_fails() -> None:
    inv, dec, quality, privacy = _fixture()
    dec["verdict"] = "PENDING"
    _seal_decontam(dec)
    with pytest.raises(CoverageError, match="not terminal PASS"):
        _bind(inv, dec, quality, privacy)


def test_unknown_decontamination_exclusion_fails() -> None:
    inv = _inventory()
    dec = _decontam(inv, exclusions=[_rid("unknown")])
    quality = _quality(inv, dec)
    privacy = _privacy(inv, dec)
    with pytest.raises(CoverageError, match="unknown record"):
        _bind(inv, dec, quality, privacy)


def test_truth_boundary_weakening_fails() -> None:
    inv, dec, quality, privacy = _fixture()
    quality["truth_boundary"]["training_eligible"] = True
    _seal(quality, "quality_authority_identity_sha256")
    with pytest.raises(CoverageError, match="training_eligible"):
        _bind(inv, dec, quality, privacy)


def test_bool_is_not_payload_byte_integer() -> None:
    inv, dec, quality, privacy = _fixture()
    quality["records"][0]["payload_bytes"] = True
    _seal(quality, "quality_authority_identity_sha256")
    with pytest.raises(CoverageError, match="integer"):
        _bind(inv, dec, quality, privacy)


def test_final_scope_must_bind_exact_decontamination_authority() -> None:
    inv, dec, quality, privacy = _fixture()
    quality["decontamination_authority_sha256"] = H("7")
    _seal(quality, "quality_authority_identity_sha256")
    with pytest.raises(CoverageError, match="decontamination binding mismatch"):
        _bind(inv, dec, quality, privacy)


def test_coverage_report_verifier_rejects_tamper() -> None:
    inv, dec, quality, privacy = _fixture()
    report = _bind(inv, dec, quality, privacy)
    verify_coverage_report(
        report,
        expected_coverage_identity_sha256=report["coverage_identity_sha256"],
        expected_inventory_identity_sha256=inv["inventory_identity_sha256"],
        expected_decontamination_authority_sha256=dec[
            "decontamination_authority_sha256"
        ],
        expected_quality_authority_identity_sha256=quality[
            "quality_authority_identity_sha256"
        ],
        expected_privacy_authority_identity_sha256=privacy[
            "privacy_authority_identity_sha256"
        ],
    )
    report["survivor_payload_bytes"] += 1
    with pytest.raises(CoverageError, match="self-hash mismatch"):
        verify_coverage_report(
            report,
            expected_coverage_identity_sha256=report["coverage_identity_sha256"],
            expected_inventory_identity_sha256=inv["inventory_identity_sha256"],
            expected_decontamination_authority_sha256=dec[
                "decontamination_authority_sha256"
            ],
            expected_quality_authority_identity_sha256=quality[
                "quality_authority_identity_sha256"
            ],
            expected_privacy_authority_identity_sha256=privacy[
                "privacy_authority_identity_sha256"
            ],
        )
