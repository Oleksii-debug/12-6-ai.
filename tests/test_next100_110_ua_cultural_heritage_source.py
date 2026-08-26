from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

CONFIG = Path("configs/data/next100_110_ua_cultural_heritage_source_audit_v1.json")
VALIDATOR = Path("tools/validate_next100_110_ua_cultural_heritage_source.py")


def _module():
    spec = importlib.util.spec_from_file_location("next100_110_validator", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config() -> dict[str, object]:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _reseal(module, config: dict[str, object]) -> dict[str, object]:
    config["authority_identity_sha256"] = module._identity(config)
    return config


def test_baseline_source_audit_passes_offline() -> None:
    module = _module()
    module.validate(_config())


def test_rejects_unsealed_mutation() -> None:
    module = _module()
    config = copy.deepcopy(_config())
    config["source"]["row_count_published"] += 1
    with pytest.raises(module.AuditError, match="authority identity mismatch"):
        module.validate(config)


def test_rejects_blanket_training_credit_even_if_resealed() -> None:
    module = _module()
    config = copy.deepcopy(_config())
    config["qualification_decision"]["training_capacity_credit_bytes"] = 2_910_000_000
    _reseal(module, config)
    with pytest.raises(module.AuditError, match="nonterminal candidate received byte credit"):
        module.validate(config)


def test_rejects_suppressed_dataset_card_conflict_even_if_resealed() -> None:
    module = _module()
    config = copy.deepcopy(_config())
    config["observed_live_card_conflict"]["present"] = False
    _reseal(module, config)
    with pytest.raises(module.AuditError, match="known rights/card conflict suppressed"):
        module.validate(config)


def test_rejects_family_cap_escape_even_if_resealed() -> None:
    module = _module()
    config = copy.deepcopy(_config())
    config["bounded_successor_contract"]["max_normalized_bytes"] = 9_000_000
    _reseal(module, config)
    with pytest.raises(module.AuditError, match="successor exceeds current one-family cap"):
        module.validate(config)


def test_rejects_rights_shortcut_removal_even_if_resealed() -> None:
    module = _module()
    config = copy.deepcopy(_config())
    config["bounded_successor_contract"]["prohibited_shortcuts"].remove(
        "treat_dataset_card_license_prose_as_per_record_rights_proof"
    )
    _reseal(module, config)
    with pytest.raises(module.AuditError, match="blanket rights shortcut no longer prohibited"):
        module.validate(config)
