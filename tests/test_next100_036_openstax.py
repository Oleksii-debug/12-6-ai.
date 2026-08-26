from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_next100_036_openstax.py"
AUTHORITY = ROOT / "evidence" / "next100_036" / "openstax_physics_source_authority.json"

spec = importlib.util.spec_from_file_location("next100_036_validator", VALIDATOR)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_terminal_authority_is_self_consistent() -> None:
    doc = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    assert module.validate_authority(doc) == []


def test_rights_reject_adds_no_capacity() -> None:
    doc = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    assert doc["terminal_verdict"] == "REJECT"
    assert doc["rights"]["model_training"]["status"] == "REJECT_NO_OPENSTAX_PERMISSION"
    assert doc["normalization"]["normalized_bytes"] == 0
    assert doc["dedup"]["admitted_capacity_delta_bytes"] == 0
    assert doc["family"]["independent_family_credit_if_rejected"] == 0


def test_no_nc_or_nd_selected() -> None:
    doc = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    assert doc["license"]["license_id"] == "CC-BY-4.0"
    assert doc["license"]["noncommercial_restriction"] is False
    assert doc["license"]["no_derivatives_restriction"] is False
