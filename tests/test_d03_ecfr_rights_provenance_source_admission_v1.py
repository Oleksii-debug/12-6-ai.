from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE = (
    Path(__file__).parents[1]
    / "tools"
    / "verify_d03_ecfr_rights_provenance_source_admission_v1.py"
)
spec = importlib.util.spec_from_file_location("ecfr_source_admission", MODULE)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def load() -> dict:
    return mod.load_policy()


def unit(**overrides: object) -> dict:
    row = {
        "request_identity_sha256": mod.EXPECTED_REQUEST_IDENTITY,
        "source_id": mod.EXPECTED_SOURCE_ID,
        "family_id": mod.EXPECTED_FAMILY_ID,
        "embedded_in_ecfr_xml": True,
        "federal_regulatory_text": True,
        "external_incorporated_by_reference": False,
        "contractor_or_private_authorship": False,
        "transferred_copyright_notice_or_status": False,
        "image_or_media_payload": False,
        "ambiguous_provenance": False,
    }
    row.update(overrides)
    return row


def test_policy_binds_exact_incumbent_and_remains_zero_credit() -> None:
    policy = load()
    authority = policy["project_authority"]
    assert authority["request_path"] == str(mod.EXPECTED_REQUEST_PATH)
    assert authority["request_git_blob_sha1"] == mod.EXPECTED_REQUEST_BLOB_SHA1
    assert authority["materializer_path"] == str(mod.EXPECTED_MATERIALIZER_PATH)
    assert authority["materializer_git_blob_sha1"] == mod.EXPECTED_MATERIALIZER_BLOB_SHA1
    assert policy["candidate_binding"]["request_identity_sha256"] == mod.EXPECTED_REQUEST_IDENTITY
    assert policy["admission_policy"]["payload_source_admission_executed"] is False
    assert policy["truth_boundary"]["training_authorized_bytes"] == 0
    assert policy["truth_boundary"]["training_eligible"] is False


def test_embedded_attributed_federal_regulatory_text_is_conditional_only() -> None:
    decision = mod.assess_unit(unit(), load())
    assert decision == {
        "decision": "CONDITIONAL_SOURCE_ADMISSION",
        "reason": "embedded_attributed_federal_regulatory_text",
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "training_eligible": False,
        "legal_conclusion_claimed": False,
    }


@pytest.mark.parametrize(
    ("flag", "reason"),
    [
        ("external_incorporated_by_reference", "external_incorporated_by_reference"),
        ("contractor_or_private_authorship", "contractor_or_private_authorship"),
        (
            "transferred_copyright_notice_or_status",
            "transferred_copyright_notice_or_status",
        ),
        ("image_or_media_payload", "image_or_media_payload"),
        ("ambiguous_provenance", "ambiguous_provenance"),
    ],
)
def test_explicit_nonadmitted_content_classes_fail_closed(flag: str, reason: str) -> None:
    decision = mod.assess_unit(unit(**{flag: True}), load())
    assert decision == {"decision": "NOT_SOURCE_ADMITTED", "reason": reason}


def test_external_reference_without_embedded_payload_is_not_admitted() -> None:
    decision = mod.assess_unit(unit(embedded_in_ecfr_xml=False), load())
    assert decision == {
        "decision": "NOT_SOURCE_ADMITTED",
        "reason": "not_embedded_in_ecfr_xml",
    }


def test_unattributed_embedded_text_is_held_for_provenance_review() -> None:
    decision = mod.assess_unit(unit(federal_regulatory_text=False), load())
    assert decision == {
        "decision": "HOLD_PROVENANCE_REVIEW",
        "reason": "not_attributed_federal_regulatory_text",
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("request_identity_sha256", "0" * 64),
        ("source_id", "en.us.ecfr.other"),
        ("family_id", "us.federal-regulations.other"),
    ],
)
def test_unit_authority_substitution_fails_closed(field: str, replacement: str) -> None:
    with pytest.raises(mod.AdmissionError):
        mod.assess_unit(unit(**{field: replacement}), load())


@pytest.mark.parametrize(
    "field",
    [
        "embedded_in_ecfr_xml",
        "federal_regulatory_text",
        "external_incorporated_by_reference",
        "contractor_or_private_authorship",
        "transferred_copyright_notice_or_status",
        "image_or_media_payload",
        "ambiguous_provenance",
    ],
)
def test_unit_boolean_aliases_fail_closed(field: str) -> None:
    with pytest.raises(mod.AdmissionError, match="must be boolean"):
        mod.assess_unit(unit(**{field: 0}), load())


def test_unit_unknown_field_fails_closed() -> None:
    candidate = unit()
    candidate["extra"] = False
    with pytest.raises(mod.AdmissionError, match="unit keys must be exact"):
        mod.assess_unit(candidate, load())


def test_unit_missing_field_fails_closed() -> None:
    candidate = unit()
    candidate.pop("ambiguous_provenance")
    with pytest.raises(mod.AdmissionError, match="unit keys must be exact"):
        mod.assess_unit(candidate, load())


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("project_authority", "request_path", "configs/data/forged.json"),
        ("project_authority", "request_git_blob_sha1", "0" * 40),
        ("project_authority", "materializer_path", "tools/forged.py"),
        ("project_authority", "materializer_git_blob_sha1", "0" * 40),
        ("candidate_binding", "request_identity_sha256", "0" * 64),
        ("candidate_binding", "url", "https://www.ecfr.gov/current/title-1.xml"),
        ("admission_policy", "hosting_or_public_availability_sufficient_alone", True),
        ("admission_policy", "external_ibr_content_source_admitted", True),
        ("admission_policy", "legal_conclusion_claimed", True),
        ("admission_policy", "payload_source_admission_executed", True),
        ("truth_boundary", "training_eligible", True),
        ("truth_boundary", "paid_compute_used", True),
        ("truth_boundary", "foreign_pretrained_weights", True),
        ("truth_boundary", "external_llm_api_data_or_intelligence", True),
    ],
)
def test_policy_authority_or_truth_substitution_fails_closed(
    tmp_path: Path, section: str, key: str, value: object
) -> None:
    policy = load()
    policy[section][key] = value
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(mod.AdmissionError):
        mod.load_policy(path)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("admission_policy", "admitted_payload_records"),
        ("admission_policy", "admitted_payload_bytes"),
        ("truth_boundary", "canonical_capacity_credited"),
        ("truth_boundary", "family_credit_added"),
        ("truth_boundary", "training_authorized_bytes"),
        ("truth_boundary", "authorized_unique_loss_positions"),
        ("truth_boundary", "optimizer_updates"),
    ],
)
def test_numeric_zero_bool_alias_fails_closed(tmp_path: Path, section: str, key: str) -> None:
    policy = load()
    policy[section][key] = False
    path = tmp_path / "bool-alias.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(mod.AdmissionError, match="strict integer"):
        mod.load_policy(path)


@pytest.mark.parametrize(
    "section",
    [
        "project_authority",
        "candidate_binding",
        "admission_policy",
        "truth_boundary",
    ],
)
def test_unknown_authority_key_fails_closed(tmp_path: Path, section: str) -> None:
    policy = load()
    policy[section]["unknown"] = "forged"
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(mod.AdmissionError, match="keys must be exact"):
        mod.load_policy(path)


@pytest.mark.parametrize(
    "section",
    [
        "project_authority",
        "candidate_binding",
        "admission_policy",
        "truth_boundary",
    ],
)
def test_missing_authority_key_fails_closed(tmp_path: Path, section: str) -> None:
    policy = load()
    policy[section].pop(next(iter(policy[section])))
    path = tmp_path / "missing.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(mod.AdmissionError, match="keys must be exact"):
        mod.load_policy(path)


def test_primary_public_evidence_substitution_fails_closed(tmp_path: Path) -> None:
    policy = load()
    policy["primary_public_evidence"][0]["url"] = "https://example.invalid/"
    path = tmp_path / "evidence-substitution.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(mod.AdmissionError, match="primary_public_evidence"):
        mod.load_policy(path)


def test_primary_public_evidence_unknown_field_fails_closed(tmp_path: Path) -> None:
    policy = load()
    policy["primary_public_evidence"][0]["unchecked"] = "drift"
    path = tmp_path / "evidence-extra.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(mod.AdmissionError, match="keys must be exact"):
        mod.load_policy(path)


def test_request_blob_substitution_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    forged = tmp_path / "request.json"
    forged.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(mod, "REQUEST_PATH", forged)
    with pytest.raises(mod.AdmissionError, match="path substitution"):
        mod.load_policy()


def test_materializer_blob_substitution_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forged = tmp_path / "materializer.py"
    forged.write_text("# forged\n", encoding="utf-8")
    monkeypatch.setattr(mod, "MATERIALIZER_PATH", forged)
    with pytest.raises(mod.AdmissionError, match="materializer path substitution"):
        mod.load_policy()


def test_root_unknown_field_fails_closed(tmp_path: Path) -> None:
    policy = load()
    policy["unknown"] = {}
    path = tmp_path / "root-extra.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(mod.AdmissionError, match="policy root keys must be exact"):
        mod.load_policy(path)
