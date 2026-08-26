from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/build_eval291_en_selection_validation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("eval291_builder", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval291_is_deterministic_and_fail_closed() -> None:
    module = _load_module()
    cfg = module.load_config()
    first = module.build(cfg)
    second = module.build(cfg)
    assert module.canonical_bytes(first) == module.canonical_bytes(second)
    assert first["authority_identity_sha256"] == (
        "23dc4bb52ff887a299d1cdad32cff352f2909e6a2cebcf4b2388a60337bf4460"
    )
    assert first["status"] == "BLOCKED_NO_TERMINAL_EN_EVALUATION_RESERVATION"
    assert first["documents"] == 0
    assert first["records"] == []
    assert first["eligible_source_objects"] == 0


def test_eval291_never_promotes_training_or_final_test_material() -> None:
    module = _load_module()
    authority = module.build(module.load_config())
    assert authority["truth_boundary"]["rights_inferred_from_training_permission"] is False
    assert authority["truth_boundary"]["training_bytes_reclassified"] is False
    assert authority["truth_boundary"]["final_test_outcomes_inspected"] is False
    assert authority["truth_boundary"]["final_test_bytes_copied"] is False
    assert authority["purpose_separation"]["selection_records_train_eligible"] is False
    assert authority["purpose_separation"]["selection_records_final_test_eligible"] is False
