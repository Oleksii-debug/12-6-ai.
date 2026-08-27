from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "validate_culturax_ua_en_audit.py"
AUDIT_PATH = ROOT / "configs" / "audit" / "culturax_ua_en_rights_provenance_v1.json"
REGISTRY_PATH = ROOT / "configs" / "research" / "open_source_reuse_registry_v2.json"

SPEC = importlib.util.spec_from_file_location("culturax_audit_validator", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def _load() -> tuple[dict, dict]:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return audit, registry


def _assert_rejected(audit: dict, registry: dict) -> None:
    with pytest.raises(validator.AuditValidationError):
        validator.validate_payload(audit, registry)


def test_canonical_culturax_audit_is_fail_closed_and_registry_bound() -> None:
    audit, registry = _load()
    registry_raw = REGISTRY_PATH.read_bytes()
    validator.validate_payload(
        audit,
        registry,
        registry_blob_sha=validator._git_blob_sha(registry_raw),
    )


def test_rejects_training_authority_overclaim() -> None:
    audit, registry = _load()
    audit["project_training_credit"]["canonical_training_authorized"] = True
    _assert_rejected(audit, registry)


@pytest.mark.parametrize(
    "field",
    [
        "admitted_source_bytes",
        "admitted_tokenizer_tokens",
        "authorized_unique_causal_loss_positions",
    ],
)
def test_rejects_nonzero_project_training_credit(field: str) -> None:
    audit, registry = _load()
    audit["project_training_credit"][field] = 1
    _assert_rejected(audit, registry)


def test_rejects_package_license_as_blanket_source_authority() -> None:
    audit, registry = _load()
    audit["rights_and_privacy_conclusions"][
        "package_or_dataset_card_license_is_blanket_source_training_authority"
    ] = True
    _assert_rejected(audit, registry)


def test_rejects_oscar_packaging_license_as_crawled_text_ownership() -> None:
    audit, registry = _load()
    audit["rights_and_privacy_conclusions"][
        "oscar_cc0_metadata_packaging_is_crawled_text_ownership"
    ] = True
    _assert_rejected(audit, registry)


def test_rejects_loss_of_record_source_provenance() -> None:
    audit, registry = _load()
    audit["upstream_observation"]["record_provenance_fields"] = ["timestamp", "url"]
    _assert_rejected(audit, registry)


def test_rejects_privacy_caveat_erasure() -> None:
    audit, registry = _load()
    audit["rights_and_privacy_conclusions"]["personal_or_sensitive_information_may_remain"] = False
    _assert_rejected(audit, registry)


def test_rejects_evaluation_firewall_weakening() -> None:
    audit, registry = _load()
    audit["evaluation_and_lineage_firewall"]["benchmark_or_final_test_payload_accessed"] = True
    _assert_rejected(audit, registry)


def test_rejects_foreign_base_lineage_contamination() -> None:
    audit, registry = _load()
    audit["evaluation_and_lineage_firewall"]["foreign_pretrained_weights_imported"] = True
    _assert_rejected(audit, registry)


def test_rejects_live_registry_component_drift() -> None:
    audit, registry = _load()
    drifted = copy.deepcopy(registry)
    component = next(item for item in drifted["components"] if item["id"] == "CULTURAX")
    component["canonical_training_authorized"] = True
    _assert_rejected(audit, drifted)


def test_rejects_checked_out_registry_blob_drift() -> None:
    audit, registry = _load()
    with pytest.raises(validator.AuditValidationError):
        validator.validate_payload(audit, registry, registry_blob_sha="0" * 40)


def test_rejects_missing_successor_decontamination_gate() -> None:
    audit, registry = _load()
    requirements = audit["successor_acquisition_contract"]["required_before_any_training_credit"]
    audit["successor_acquisition_contract"]["required_before_any_training_credit"] = [
        item for item in requirements if "reserved-evaluation decontamination" not in item
    ]
    _assert_rejected(audit, registry)


def test_rejects_paid_compute_claim() -> None:
    audit, registry = _load()
    audit["compute_truth"]["paid_compute_used"] = True
    _assert_rejected(audit, registry)
