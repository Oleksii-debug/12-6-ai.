from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools/validate_next100_065d_registry_v4_guard.py"
CONFIG = ROOT / "configs/data/next100_065d_registry_v4_guard_v1.json"

spec = importlib.util.spec_from_file_location("next100_065d_registry_v4_guard", VALIDATOR)
assert spec is not None and spec.loader is not None
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def _data() -> dict:
    return validator.load_config(CONFIG)


def test_registry_v4_guard_static_contract_passes() -> None:
    validator.validate_static(_data())


def test_registry_v4_guard_rejects_superseded_v3_path() -> None:
    data = _data()
    data["canonical_registry"]["path"] = "configs/data/next100_063_terminal_source_registry_v3.json"
    with pytest.raises(validator.RegistryV4GuardError, match="path must be V4"):
        validator.validate_static(data)


def test_registry_v4_guard_rejects_registry_identity_drift() -> None:
    data = _data()
    data["canonical_registry"]["registry_identity_sha256"] = "0" * 64
    with pytest.raises(validator.RegistryV4GuardError, match="V4 registry identity drift"):
        validator.validate_static(data)


def test_registry_v4_guard_rejects_full_cpython_envelope_as_capacity() -> None:
    data = _data()
    data["required_embedded_authorities"]["cpython_accepted_only"][
        "numeric_training_capacity_bytes"
    ] = 17_901
    with pytest.raises(validator.RegistryV4GuardError, match="CPython eligible capacity drift"):
        validator.validate_static(data)


def test_registry_v4_guard_rejects_v6_capacity_not_equal_to_v4() -> None:
    data = _data()
    data["v6_reconciliation"]["expected_pre_global_dedup_capacity_bytes"]["total"] += 1
    with pytest.raises(validator.RegistryV4GuardError):
        validator.validate_static(data)


def test_registry_v4_guard_rejects_training_authorization() -> None:
    data = copy.deepcopy(_data())
    data["claim_boundary"]["training_authorized"] = True
    with pytest.raises(validator.RegistryV4GuardError, match="training_authorized"):
        validator.validate_static(data)
