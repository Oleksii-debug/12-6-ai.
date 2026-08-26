from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "train195_prerequisite_gate.py"
CONFIG_PATH = ROOT / "configs" / "experiments" / "train195_10m_lr_beta_transfer.json"

spec = importlib.util.spec_from_file_location("train195_prerequisite_gate", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_current_preregistration_is_valid_but_blocked():
    report = module.make_report(load_config())
    assert report["validation"] == "PASS"
    assert report["runnable"] is False
    assert report["scientific_status"] == module.BLOCKED
    assert set(report["blockers"]) == {"freeze_contract", "train125_lr_transfer", "train194_clipping"}


def test_missing_train125_cannot_materialize_absolute_lr_grid():
    cfg = load_config()
    cfg["staged_design"]["absolute_lr_candidates"] = [2.4e-4, 3.0e-4, 3.75e-4]
    report = module.make_report(cfg)
    assert report["validation"] == "FAIL"
    assert any("absolute LR candidates must remain null" in error for error in report["errors"])


def test_one_seed_promotion_is_rejected():
    cfg = load_config()
    cfg["decision_contract"]["minimum_paired_repeats"] = 1
    cfg["decision_contract"]["one_seed_can_promote"] = True
    report = module.make_report(cfg)
    assert report["validation"] == "FAIL"
    assert any("forbid promotion below three paired repeats" in error for error in report["errors"])


def test_ready_state_requires_exact_prerequisites_and_freeze():
    cfg = load_config()
    train125 = cfg["required_prerequisites"]["train125_lr_transfer"]
    train125.update(
        authority_present=True,
        source_sha="1" * 40,
        evidence_sha256="2" * 64,
        predicted_lr=3e-4,
        prediction_method="fixture-only-test",
        freeze_weight_decay=0.0,
        freeze_schedule_family="constant",
        freeze_batch_geometry={"microbatch": 1, "sequence_length": 256},
    )
    train194 = cfg["required_prerequisites"]["train194_clipping"]
    train194.update(
        authority_present=True,
        source_sha="3" * 40,
        evidence_sha256="4" * 64,
        accepted_clip_norm=4.0,
        nonfinite_fail_closed_before_clipping=True,
    )
    cfg["freeze_contract"].update(
        gradient_clip_norm=4.0,
        weight_decay=0.0,
        schedule_family="constant",
        batch_geometry={"microbatch": 1, "sequence_length": 256},
    )
    cfg["staged_design"]["absolute_lr_candidates"] = [2.4e-4, 3e-4, 3.75e-4]
    cfg["status"] = module.READY
    report = module.make_report(cfg)
    assert report["validation"] == "PASS"
    assert report["runnable"] is True
    assert report["blockers"] == []


def test_nonfinite_before_clipping_proof_is_mandatory():
    cfg = load_config()
    cfg["required_prerequisites"]["train194_clipping"].update(
        authority_present=True,
        source_sha="3" * 40,
        evidence_sha256="4" * 64,
        accepted_clip_norm=4.0,
        nonfinite_fail_closed_before_clipping=False,
    )
    report = module.make_report(cfg)
    assert report["validation"] == "FAIL"
