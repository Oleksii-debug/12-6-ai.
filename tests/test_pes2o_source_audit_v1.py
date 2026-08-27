from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "data" / "pes2o_source_audit_v1.json"
VALIDATOR = ROOT / "tools" / "validate_pes2o_source_audit_v1.py"


def _load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pes2o_source_audit_validator", VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load validator from {VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_VALIDATOR_MODULE = _load_validator()
AuditValidationError = _VALIDATOR_MODULE.AuditValidationError
validate_manifest = _VALIDATOR_MODULE.validate_manifest


def _manifest() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _reject(mutator) -> None:
    data = copy.deepcopy(_manifest())
    mutator(data)
    with pytest.raises(AuditValidationError):
        validate_manifest(data)


def test_manifest_is_fail_closed() -> None:
    validate_manifest(_manifest())


def test_dataset_package_license_cannot_authorize_training() -> None:
    _reject(lambda data: data["rights_gate"].__setitem__("dataset_package_license_is_training_authority", True))


def test_open_access_label_cannot_authorize_training() -> None:
    _reject(lambda data: data["rights_gate"].__setitem__("open_access_label_is_training_authority", True))


def test_training_authority_cannot_be_enabled() -> None:
    _reject(lambda data: data["rights_gate"].__setitem__("training_authorized", True))


def test_source_channel_cannot_claim_approved_rights() -> None:
    _reject(lambda data: data["source_channels"][0].__setitem__("rights_status", "APPROVED"))


def test_source_channel_cannot_invent_license_evidence() -> None:
    _reject(
        lambda data: data["source_channels"][0].__setitem__(
            "source_specific_license_evidence", ["UNVERIFIED_LICENSE"]
        )
    )


def test_authorized_bytes_cannot_be_nonzero() -> None:
    _reject(lambda data: data["source_channels"][1].__setitem__("authorized_bytes", 1))


def test_observed_schema_must_not_fabricate_license_field() -> None:
    def mutate(data: dict) -> None:
        data["upstream_identity"]["declared_derivation"]["observed_document_fields"].append("license")
        data["upstream_identity"]["declared_derivation"]["per_document_license_field_in_observed_schema"] = True

    _reject(mutate)


def test_upstream_validation_split_is_not_project_clearance() -> None:
    _reject(
        lambda data: data["contamination_gate"].__setitem__(
            "upstream_train_valid_split_is_project_evaluation_clearance", True
        )
    )


def test_decontamination_cannot_be_claimed_without_execution() -> None:
    _reject(lambda data: data["contamination_gate"].__setitem__("decontamination_executed", True))


def test_final_test_payload_cannot_be_accessed() -> None:
    _reject(
        lambda data: data["contamination_gate"].__setitem__(
            "benchmark_or_final_test_payload_accessed", True
        )
    )


def test_paid_compute_cannot_be_authorized() -> None:
    _reject(lambda data: data["hard_boundaries"].__setitem__("paid_compute_authorized", True))
