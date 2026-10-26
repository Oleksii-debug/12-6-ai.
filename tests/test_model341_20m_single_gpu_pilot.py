from __future__ import annotations

import json
from pathlib import Path

from twelve_six.model import load_stage_config

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "configs/runs/model341_20m.single_gpu_pilot.experimental.json"
EXPECTED_MODEL_SHA = "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
EXPECTED_INIT_SHA = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"


def _payload() -> dict:
    return json.loads(RUN.read_text(encoding="utf-8"))


def test_model341_20m_pilot_binds_exact_candidate() -> None:
    payload = _payload()
    stage = load_stage_config(ROOT / payload["stage_config"])

    assert stage.stage == "MODEL-341-20M-CANDIDATE-A"
    assert stage.target_parameters == 20_000_000
    assert stage.expected_parameters == 20_613_440
    assert stage.model.parameter_count() == 20_613_440
    assert stage.model.identity_sha256() == EXPECTED_MODEL_SHA
    assert stage.init.identity_sha256() == EXPECTED_INIT_SHA


def test_model341_20m_pilot_is_mechanics_only_and_cannot_buy_compute() -> None:
    payload = _payload()

    assert payload["run_kind"] == "single_gpu_mechanics_pilot"
    assert payload["state"] == "PREPARED_NOT_LAUNCHED"
    assert payload["data_authority"] == "CONTROLLED_SYNTHETIC_MECHANICS_ONLY"
    assert payload["authorization"] == {
        "provision_compute": False,
        "preprovisioned_accelerator_only": True,
        "paid_compute_launch": False,
        "notes": payload["authorization"]["notes"],
    }
    assert "learned language capability" in payload["truth_boundary"]["not_evidence_for"]
    assert "paid-compute authorization" in payload["truth_boundary"]["not_evidence_for"]


def test_model341_20m_pilot_matches_incumbent_runner_contract() -> None:
    payload = _payload()
    trainer = payload["trainer"]
    pilot = payload["pilot"]
    stage = load_stage_config(ROOT / payload["stage_config"])

    assert trainer["max_steps"] == pilot["steps"] == 4
    assert trainer["gradient_accumulation_steps"] == 1
    assert 1 <= pilot["resume_after_step"] < pilot["steps"]
    assert pilot["microbatch_size"] >= 1
    assert 2 <= pilot["sequence_length"] <= stage.model.max_seq_len
    assert trainer["precision"] in {"fp16", "bf16"}
