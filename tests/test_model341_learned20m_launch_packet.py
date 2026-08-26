from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/control/model341_learned20m_launch_packet_v1.json"
VALIDATOR_PATH = ROOT / "tools/validate_model341_learned20m_launch_packet.py"

spec = importlib.util.spec_from_file_location("model341_launch_validator", VALIDATOR_PATH)
assert spec is not None and spec.loader is not None
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def _load() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _terminal(scope: str, char: str = "1", decision: str = "pass") -> str:
    return f"github:{scope}@{char * 40}:{decision}"


def _populate_authorization_request(data: dict) -> None:
    for index, key in enumerate(sorted(data["required_authorities"])):
        char = format((index % 15) + 1, "x")
        data["required_authorities"][key] = _terminal(key, char)

    recipe = data["training_recipe"]
    recipe.update(
        {
            "seed_set": [3401, 3402],
            "optimizer": "AdamW(beta1=0.9,beta2=evidence-bound,eps=1e-8,wd=0.1)",
            "scheduler": "evidence-bound schedule with frozen warmup",
            "precision": "profiled accelerator precision with FP32 control evidence",
            "gradient_clip_norm": 1.0,
            "unique_loss_positions": 100_000_000,
            "total_training_exposure": 120_000_000,
            "replay_policy": "bounded replay; unique and repeated exposure counted separately",
            "stop_rule": "stop on NaN/Inf, recovery mismatch, preregistered loss/eval failure, or budget ceiling",
            "checkpoint_cadence": "evidence-bound cadence plus chronological final checkpoint",
            "resume_policy": "fresh-process exact-identity resume before material continuation",
        }
    )

    firewall = data["evaluation_firewall"]
    firewall.update(
        {
            "selection_validation_identity": "a" * 64,
            "final_test_identity": "b" * 64,
            "selection_schedule": "preregistered fixed selection checkpoints; final test remains sealed",
        }
    )

    envelope = data["compute_envelope"]
    envelope.update(
        {
            "estimated_training_flops": 1.0e16,
            "resource_shape": "profiled exact accelerator type/count and software stack",
            "profiled_loss_positions_per_second": 1000.0,
            "wall_clock_upper_bound_seconds": 200000.0,
            "maximum_budget_eur": 500.0,
        }
    )

    data["decision"] = validator.assess_packet(data)
    data["status"] = "READY_FOR_AUTHORIZATION_REQUEST"


def test_current_packet_is_valid_and_blocked() -> None:
    data = _load()
    assert validator.validate_packet(data) == []
    assert data["decision"]["current_blockers"] == validator.EXPECTED_BLOCKED
    assert data["decision"]["ready_for_authorization_request"] is False
    assert data["decision"]["ready_for_long_training"] is False


def test_model341_authority_drift_fails_closed() -> None:
    data = _load()
    data["authority"]["parameter_count"] = 20_000_000
    errors = validator.validate_packet(data)
    assert any("authority.parameter_count" in error for error in errors)


def test_bool_schema_alias_is_rejected() -> None:
    data = _load()
    data["schema_version"] = True
    errors = validator.validate_packet(data)
    assert any("schema_version" in error for error in errors)


def test_authorization_request_requires_every_terminal_authority() -> None:
    data = _load()
    _populate_authorization_request(data)
    assert validator.validate_packet(data) == []
    assert data["decision"]["ready_for_authorization_request"] is True
    assert data["decision"]["ready_for_short_horizon"] is False

    data["required_authorities"]["research_corpus_v1"] = None
    data["decision"] = validator.assess_packet(data)
    data["status"] = "BLOCKED_PENDING_TERMINAL_AUTHORITIES"
    assert data["decision"]["ready_for_authorization_request"] is False
    assert "terminal_authorities_missing" in data["decision"]["current_blockers"]


def test_queued_reference_is_never_terminal_authority() -> None:
    data = _load()
    data["required_authorities"]["research_corpus_v1"] = (
        "github:research-corpus@" + "1" * 40 + ":queued"
    )
    errors = validator.validate_packet(data)
    assert any("research_corpus_v1" in error for error in errors)


def test_launch_packet_cannot_self_attest() -> None:
    data = _load()
    data["required_authorities"]["independent_launch_audit"] = (
        "github:model341-learned20m-launch-packet-v1@" + "2" * 40 + ":pass"
    )
    errors = validator.validate_packet(data)
    assert any("independent_launch_audit" in error for error in errors)


def test_unique_loss_positions_cannot_be_relabelled_from_smaller_exposure() -> None:
    data = _load()
    _populate_authorization_request(data)
    data["training_recipe"]["total_training_exposure"] = 50_000_000
    data["decision"] = validator.assess_packet(data)
    data["status"] = "BLOCKED_PENDING_TERMINAL_AUTHORITIES"
    errors = validator.validate_packet(data)
    assert any("training exposure cannot be smaller" in error for error in errors)
    assert data["decision"]["ready_for_authorization_request"] is False


def test_final_test_firewall_cannot_be_opened_for_training() -> None:
    data = _load()
    _populate_authorization_request(data)
    data["evaluation_firewall"]["final_test_read_before_terminal_training"] = True
    data["decision"] = validator.assess_packet(data)
    data["status"] = "BLOCKED_PENDING_TERMINAL_AUTHORITIES"
    errors = validator.validate_packet(data)
    assert any("final test must stay unread" in error for error in errors)
    assert data["decision"]["ready_for_authorization_request"] is False


def test_boolean_compute_permission_is_not_authorization() -> None:
    data = _load()
    _populate_authorization_request(data)
    data["compute_envelope"]["compute_authorization"] = True
    errors = validator.validate_packet(data)
    assert any("compute_authorization" in error for error in errors)
    assert validator.assess_packet(data)["ready_for_short_horizon"] is False


def test_short_horizon_requires_two_authorizations_and_smoke_evidence() -> None:
    data = _load()
    _populate_authorization_request(data)
    data["compute_envelope"]["compute_authorization"] = _terminal(
        "owner-compute-approval", "c", "authorized"
    )
    data["compute_envelope"]["training_authorization"] = _terminal(
        "owner-training-approval", "d", "authorized"
    )
    data["decision"] = validator.assess_packet(data)
    assert data["decision"]["ready_for_short_horizon"] is False

    data["phase_evidence"]["bounded_smoke"] = _terminal("model341-bounded-smoke", "e")
    data["decision"] = validator.assess_packet(data)
    data["status"] = "READY_FOR_SHORT_HORIZON"
    assert data["decision"]["ready_for_short_horizon"] is True
    assert data["decision"]["ready_for_long_training"] is False
    assert validator.validate_packet(data) == []


def test_long_training_requires_terminal_short_horizon_evidence() -> None:
    data = _load()
    _populate_authorization_request(data)
    data["compute_envelope"]["compute_authorization"] = _terminal(
        "owner-compute-approval", "c", "authorized"
    )
    data["compute_envelope"]["training_authorization"] = _terminal(
        "owner-training-approval", "d", "authorized"
    )
    data["phase_evidence"]["bounded_smoke"] = _terminal("model341-bounded-smoke", "e")
    data["decision"] = validator.assess_packet(data)
    data["status"] = "READY_FOR_SHORT_HORIZON"
    assert validator.validate_packet(data) == []

    data["decision"]["ready_for_long_training"] = True
    errors = validator.validate_packet(data)
    assert any("ready_for_long_training" in error for error in errors)

    clean = copy.deepcopy(data)
    clean["phase_evidence"]["short_horizon"] = _terminal("model341-short-horizon", "f")
    clean["decision"] = validator.assess_packet(clean)
    clean["status"] = "READY_FOR_LONG_TRAINING"
    assert clean["decision"]["ready_for_long_training"] is True
    assert validator.validate_packet(clean) == []
