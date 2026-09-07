from __future__ import annotations

import copy
import json
from pathlib import Path

from twelve_six.learned20m_readiness import assess_learned20m_readiness

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/r01_learned20m_launch_readiness_v1.json"
SHA40 = "a" * 40
SHA64 = "b" * 64


def _authority() -> dict:
    return {
        "repository": "Oleksii-debug/12-6-ai.",
        "git_sha": SHA40,
        "evidence_sha256": SHA64,
        "terminal": True,
        "workflow_run_id": 123,
        "workflow_conclusion": "success",
    }


def _local_ready() -> dict:
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    evidence = data["evidence"]
    evidence["code"]["git_sha"] = SHA40
    evidence["corpus"].update(
        {
            "manifest_sha256": SHA64,
            "split_sha256": SHA64,
            "packing_sha256": SHA64,
            "two_clean_builds_identical": True,
            "authority": _authority(),
        }
    )
    evidence["tokenizer"].update(
        {
            "identity_sha256": SHA64,
            "decision": "BYTE_BASELINE_RETAINED",
            "authority": _authority(),
        }
    )
    evidence["loss_ledger"].update(
        {
            "identity_sha256": SHA64,
            "unique_causal_loss_positions": 1000,
            "authority": _authority(),
            "data_budget_authority": _authority(),
            "data_budget_status": "QUALIFIED",
        }
    )
    evidence["checkpoint_integrity"].update(
        {"authority": _authority(), "status": "PASS"}
    )
    evidence["evaluation"].update(
        {
            "firewall_authority": _authority(),
            "selection_validation_authority": _authority(),
            "status": "PASS",
        }
    )
    evidence["training_recipe"].update(
        {
            "authority": _authority(),
            "status": "QUALIFIED",
            "seed_count": 1,
            "config_sha256": SHA64,
            "stopping_policy_sha256": SHA64,
            "requested_unique_loss_positions": 1000,
            "requested_total_training_exposures": 1000,
            "max_exposures_per_unique_position": 1,
        }
    )
    return data


def test_terminal_candidate_bound_decontamination_allows_local_readiness() -> None:
    result = assess_learned20m_readiness(_local_ready())
    assert result.ready_for_local_free_pilot


def test_missing_decontamination_authority_fails_closed() -> None:
    data = _local_ready()
    data["evidence"]["evaluation"]["decontamination"]["authority"] = None
    result = assess_learned20m_readiness(data)
    assert not result.ready_for_local_free_pilot
    assert "decontamination_authority_missing" in result.local_free_pilot_blockers


def test_decontamination_must_match_exact_predecontamination_candidate() -> None:
    data = _local_ready()
    data["evidence"]["corpus"]["pre_decontamination_identity_sha256"] = SHA64
    result = assess_learned20m_readiness(data)
    assert not result.ready_for_local_free_pilot
    assert "decontamination_training_corpus_identity_mismatch" in result.local_free_pilot_blockers


def test_report_identity_must_match_terminal_authority() -> None:
    data = _local_ready()
    decontam = data["evidence"]["evaluation"]["decontamination"]
    decontam["authority"] = copy.deepcopy(decontam["authority"])
    decontam["authority"]["evidence_sha256"] = SHA64
    result = assess_learned20m_readiness(data)
    assert not result.ready_for_local_free_pilot
    assert "decontamination_report_authority_mismatch" in result.local_free_pilot_blockers


def test_decontamination_report_must_be_hash_only() -> None:
    data = _local_ready()
    data["evidence"]["evaluation"]["decontamination"]["hash_only_evidence"] = False
    result = assess_learned20m_readiness(data)
    assert not result.ready_for_local_free_pilot
    assert "decontamination_report_must_be_hash_only" in result.local_free_pilot_blockers


def test_decontamination_must_not_read_final_test_outcomes() -> None:
    data = _local_ready()
    data["evidence"]["evaluation"]["decontamination"]["final_test_outcomes_read"] = True
    result = assess_learned20m_readiness(data)
    assert not result.ready_for_local_free_pilot
    assert (
        "decontamination_final_test_outcomes_read_must_be_false"
        in result.local_free_pilot_blockers
    )


def test_decontamination_must_not_select_model_or_hyperparameters() -> None:
    data = _local_ready()
    decontam = data["evidence"]["evaluation"]["decontamination"]
    decontam["model_architecture_or_hyperparameters_selected"] = True
    result = assess_learned20m_readiness(data)
    assert not result.ready_for_local_free_pilot
    assert "decontamination_model_selection_must_be_false" in result.local_free_pilot_blockers


def test_decontamination_must_not_train_and_must_be_local_free() -> None:
    for field, value, blocker in (
        ("training_executed", True, "decontamination_training_executed_must_be_false"),
        ("local_free_only", False, "decontamination_local_free_only_must_be_true"),
    ):
        data = _local_ready()
        data["evidence"]["evaluation"]["decontamination"][field] = value
        result = assess_learned20m_readiness(data)
        assert not result.ready_for_local_free_pilot
        assert blocker in result.local_free_pilot_blockers
