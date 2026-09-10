from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[1] / "tools" / "verify_d03_common_pile_caselaw_source_admission.py"
spec = importlib.util.spec_from_file_location("caselaw_source_admission", MODULE)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def load() -> dict:
    return mod.load_policy()


def record(
    *,
    source: str = "Caselaw Access Project",
    license_value: str = "Public Domain",
    url: str = "https://static.case.law/",
    metadata_extra: dict | None = None,
) -> dict:
    metadata = {"author": "PER CURIAM", "license": license_value, "url": url}
    metadata.update(metadata_extra or {})
    return {
        "id": "fixture/case.html",
        "source": source,
        "added": "2024-08-24T03:29:51.129235",
        "created": "2024-08-24T03:29:51.129683",
        "metadata": metadata,
        "text": "Fixture text is intentionally not inspected by the source-level authority.",
    }


def test_policy_binds_exact_candidate_and_remains_zero_credit() -> None:
    policy = load()
    binding = policy["candidate_binding"]
    assert binding["product_pr"] == 904
    assert binding["materializer_product_head_sha"] == mod.PRODUCT_HEAD
    assert binding["dataset"] == mod.DATASET
    assert [item["file"] for item in binding["objects"]] == [
        "cap_00044.jsonl.gz",
        "cap_00043.jsonl.gz",
    ]
    assert policy["admission_policy"]["payload_source_admission_executed"] is False
    assert policy["truth_boundary"]["training_authorized_bytes"] == 0
    assert policy["truth_boundary"]["training_eligible"] is False


def test_exact_cap_record_is_only_conditionally_source_admitted() -> None:
    decision = mod.assess_record(record(), load())
    assert decision["decision"] == "CONDITIONAL_SOURCE_ADMISSION"
    assert decision["training_authorized_bytes"] == 0
    assert decision["canonical_capacity_credited"] == 0
    assert decision["legal_conclusion_claimed"] is False


@pytest.mark.parametrize("source", ["Court Listener", "CourtListener", "Other Source"])
def test_non_cap_source_labels_are_not_admitted(source: str) -> None:
    decision = mod.assess_record(record(source=source), load())
    assert decision["decision"] == "NOT_SOURCE_ADMITTED"


def test_license_drift_is_not_admitted() -> None:
    decision = mod.assess_record(record(license_value="CC-BY"), load())
    assert decision == {
        "decision": "NOT_SOURCE_ADMITTED",
        "reason": "license_not_exact_public_domain",
    }


@pytest.mark.parametrize(
    "url",
    [
        "http://static.case.law/",
        "https://static.case.law.evil.example/",
        "https://case.law.evil.example/",
        "https://user@case.law/",
        "https://case.law:443/",
    ],
)
def test_cap_url_authority_spoofing_fails_closed(url: str) -> None:
    decision = mod.assess_record(record(url=url), load())
    assert decision["decision"] == "NOT_SOURCE_ADMITTED"
    assert decision["reason"] == "cap_url_authority_mismatch"


@pytest.mark.parametrize("key", mod.FORBIDDEN_EDITORIAL_KEYS)
def test_editorial_or_annotation_metadata_is_not_admitted(key: str) -> None:
    decision = mod.assess_record(record(metadata_extra={key: "fixture"}), load())
    assert decision == {
        "decision": "NOT_SOURCE_ADMITTED",
        "reason": "editorial_metadata_present",
    }


def test_candidate_identity_substitution_fails_closed() -> None:
    policy = load()
    binding = json.loads(json.dumps(policy["candidate_binding"]))
    binding["objects"][0]["bytes"] += 1
    with pytest.raises(mod.AdmissionError, match="candidate identity substitution"):
        mod.validate_candidate_identity(binding, policy)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("admission_policy", "common_pile_license_metadata_sufficient_alone", True),
        ("admission_policy", "accepted_source_exact", "CourtListener"),
        ("admission_policy", "payload_source_admission_executed", True),
        ("admission_policy", "admitted_payload_bytes", 1),
        ("truth_boundary", "training_authorized_bytes", 1),
        ("truth_boundary", "training_eligible", True),
        ("truth_boundary", "legal_conclusion_claimed", True),
    ],
)
def test_policy_truth_or_admission_mutation_fails_closed(
    tmp_path: Path, section: str, key: str, value: object
) -> None:
    policy = load()
    policy[section][key] = value
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(mod.AdmissionError):
        mod.load_policy(path)


def test_record_field_drift_fails_closed() -> None:
    row = record()
    row["extra"] = "drift"
    with pytest.raises(mod.AdmissionError, match="record field drift"):
        mod.assess_record(row, load())
