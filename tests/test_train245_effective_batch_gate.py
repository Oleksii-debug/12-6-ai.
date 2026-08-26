from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "train245_effective_batch_gate.py"
CONFIG_PATH = ROOT / "configs" / "experiments" / "train245_10m_effective_batch_v2.json"

spec = importlib.util.spec_from_file_location("train245_effective_batch_gate", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def materialize_train244(cfg):
    t = cfg["required_prerequisites"]["train244_optimizer"]
    t.update(
        authority_present=True,
        source_sha="a" * 40,
        evidence_sha256="b" * 64,
        decision="SELECT_OPTIMIZER_RECIPE",
        learning_rate=2.5e-4,
        beta1=0.9,
        beta2=0.99,
        weight_decay=0.0,
        epsilon=1e-8,
        gradient_clip_norm=4.0,
        schedule_family="constant",
        model_spec_sha256="c" * 64,
        data_identity="d" * 64,
        tokenizer_identity="s0-byte-v1",
        microbatch_size=1,
        sequence_length=256,
        precision="fp32",
        ordered_train_trace_identity="e" * 64,
        optimized_token_budget=131938,
    )
    for field in module.FREEZE_FIELDS:
        cfg["freeze_contract"][field] = t[field]
    cfg["status"] = module.READY
    cfg["result"]["decision"] = None
    return cfg


def test_current_config_is_valid_and_fail_closed_on_missing_train244():
    report = module.make_report(load_config())
    assert report["validation"] == "PASS"
    assert report["runnable"] is False
    assert report["scientific_status"] == module.BLOCKED
    assert report["decision"] == "INSUFFICIENT_EVIDENCE"
    assert report["blockers"] == ["train244_optimizer"]


def test_train46_executed_accumulation_authority_is_exactly_bound():
    cfg = load_config()
    cfg["consumed_accumulation_authority"]["workflow_run"] += 1
    report = module.make_report(cfg)
    assert report["validation"] == "FAIL"
    assert any("workflow_run" in error for error in report["errors"])


def test_batch_grid_cannot_expand_or_mix_microbatch_sweep():
    cfg = load_config()
    cfg["preregistered_batch_grid"]["gradient_accumulation_steps"] = [1, 2, 4, 8]
    cfg["preregistered_batch_grid"]["candidate_count"] = 4
    cfg["preregistered_batch_grid"]["microbatch_shape_fixed"] = False
    report = module.make_report(cfg)
    assert report["validation"] == "FAIL"
    assert any("[1,2,4]" in error for error in report["errors"])
    assert any("microbatch_shape_fixed" in error for error in report["errors"])


def test_blocked_state_rejects_fabricated_candidate_results():
    cfg = load_config()
    cfg["candidate_results"]["accumulation_2"] = {"held_out_bpb": 1.23}
    cfg["result"]["selected_gradient_accumulation_steps"] = 2
    report = module.make_report(cfg)
    assert report["validation"] == "FAIL"
    assert any("cannot contain numerical candidate results" in error for error in report["errors"])
    assert any("cannot select a batch" in error for error in report["errors"])


def test_ready_state_requires_train244_exact_freeze_and_keeps_grid_fixed():
    cfg = materialize_train244(deepcopy(load_config()))
    report = module.make_report(cfg)
    assert report["validation"] == "PASS"
    assert report["runnable"] is True
    assert report["scientific_status"] == module.READY
    assert report["blockers"] == []
    assert cfg["preregistered_batch_grid"]["gradient_accumulation_steps"] == [1, 2, 4]


def test_ready_state_rejects_optimizer_drift_from_train244():
    cfg = materialize_train244(deepcopy(load_config()))
    cfg["freeze_contract"]["beta2"] = 0.95
    report = module.make_report(cfg)
    assert report["validation"] == "FAIL"
    assert any("freeze_contract.beta2" in error for error in report["errors"])
