from __future__ import annotations

import json
from pathlib import Path

from twelve_six.learned_20m_readiness import (
    MODEL341_MODELSPEC_SHA256,
    MODEL341_PARAMETER_COUNT,
    MODEL341_SHA,
    R01_CAMPAIGN_BLOB_SHA1,
    R01_CAMPAIGN_ID,
    R01_MERGE_SHA,
    REPOSITORY,
    evaluate_learned_20m_readiness,
    git_blob_sha1,
    verify_r01_campaign_bytes,
)

SHA = "1" * 40
ID = "2" * 64
CORPUS_ID = "3" * 64
SPLIT_ID = "4" * 64
PACKING_ID = "5" * 64
TOKENIZER_ID = "6" * 64


def evidence(authority: str) -> dict[str, object]:
    return {
        "authority": authority,
        "repository": REPOSITORY,
        "source_ref": f"TEST_FIXTURE:{authority}",
        "source_sha": SHA,
        "identity_sha256": ID,
        "terminal_state": "PASS",
        "self_asserted": False,
        "superseded": False,
    }


def ready_packet() -> dict[str, object]:
    corpus = evidence("DATA_CORPUS")
    corpus["corpus_identity_sha256"] = CORPUS_ID
    split = evidence("DATA_SPLIT")
    split.update(
        {"corpus_identity_sha256": CORPUS_ID, "split_identity_sha256": SPLIT_ID}
    )
    packing = evidence("D04_PACKING")
    packing.update(
        {"split_identity_sha256": SPLIT_ID, "packing_identity_sha256": PACKING_ID}
    )
    tokenizer = evidence("D04_TOKENIZER")
    tokenizer["tokenizer_identity_sha256"] = TOKENIZER_ID
    data_budget = evidence("R01_DATA_BUDGET")
    d05 = evidence("D05_CHECKPOINT")
    d05["model_sha"] = MODEL341_SHA
    authorities = {
        "corpus": corpus,
        "split": split,
        "packing": packing,
        "tokenizer": tokenizer,
        "data_budget": data_budget,
        "d05_checkpoint": d05,
        "evaluation_firewall": evidence("D06_EVALUATION_FIREWALL"),
        "selection_validation": evidence("D06_SELECTION_VALIDATION"),
    }
    ledger = evidence("DATA_UNIQUE_LOSS_LEDGER")
    ledger.update(
        {
            "unique_causal_loss_positions": 1_000_000,
            "post_pack": True,
            "no_replay": True,
            "non_ignored_targets_only": True,
            "corpus_identity_sha256": CORPUS_ID,
            "split_identity_sha256": SPLIT_ID,
            "packing_identity_sha256": PACKING_ID,
            "tokenizer_identity_sha256": TOKENIZER_ID,
        }
    )
    data_budget.update(
        {
            "qualified_unique_loss_positions": 1_000_000,
            "unique_loss_ledger_identity_sha256": ledger["identity_sha256"],
        }
    )
    recipe = evidence("TRAIN_RECIPE")
    recipe.update(
        {
            "optimizer": "AdamW",
            "scheduler": "preregistered",
            "precision": "fp32-local-test-fixture",
            "stopping_rule": "fixed-budget-or-fail-closed",
            "seeds": [17, 23],
            "max_unique_loss_positions": 900_000,
            "model_sha": MODEL341_SHA,
            "packing_identity_sha256": PACKING_ID,
            "tokenizer_identity_sha256": TOKENIZER_ID,
            "unique_loss_ledger_identity_sha256": ledger["identity_sha256"],
        }
    )
    bounded = evidence("TRAIN_BOUNDED_PILOT")
    bounded.update(
        {
            "numerics_pass": True,
            "checkpoint_resume_pass": True,
            "evaluation_firewall_pass": True,
            "loss_trajectory_observed": True,
            "model_sha": MODEL341_SHA,
            "training_recipe_identity_sha256": recipe["identity_sha256"],
            "unique_loss_ledger_identity_sha256": ledger["identity_sha256"],
        }
    )
    cost = evidence("C01_COST_ENVELOPE")
    cost.update(
        {
            "estimated_flops": 100_000,
            "estimated_wall_clock_seconds": 60,
            "max_cost_usd": 10.0,
            "hardware_profile": "TEST_FIXTURE",
        }
    )
    audit = evidence("INDEPENDENT_AUDIT")
    audit["verdict"] = "PASS"
    compute = evidence("C01_COMPUTE_AUTHORIZATION")
    compute.update(
        {
            "max_cost_usd": 10.0,
            "max_unique_loss_positions": 900_000,
            "cost_envelope_identity_sha256": cost["identity_sha256"],
            "training_recipe_identity_sha256": recipe["identity_sha256"],
        }
    )
    training = evidence("TRAINING_AUTHORIZATION")
    training.update(
        {
            "max_unique_loss_positions": 900_000,
            "long_training_authorized": True,
            "training_recipe_identity_sha256": recipe["identity_sha256"],
            "compute_authorization_identity_sha256": compute["identity_sha256"],
        }
    )
    return {
        "schema_version": 1,
        "repository": REPOSITORY,
        "model": {
            "source_sha": MODEL341_SHA,
            "modelspec_sha256": MODEL341_MODELSPEC_SHA256,
            "parameter_count": MODEL341_PARAMETER_COUNT,
            "canonical_base": "random_init",
        },
        "r01": {
            "merge_sha": R01_MERGE_SHA,
            "campaign_id": R01_CAMPAIGN_ID,
            "campaign_blob_sha1": R01_CAMPAIGN_BLOB_SHA1,
        },
        "authorities": authorities,
        "unique_loss_ledger": ledger,
        "training_recipe": recipe,
        "pilot_plan": {
            "local_free_only": True,
            "material_compute": False,
            "max_optimizer_updates": 10,
            "max_unique_loss_positions": 100_000,
        },
        "bounded_pilot": bounded,
        "cost_envelope": cost,
        "independent_audit": audit,
        "compute_authorization": compute,
        "training_authorization": training,
    }


def test_all_phases_can_become_ready_with_complete_test_fixture() -> None:
    result = evaluate_learned_20m_readiness(ready_packet())
    assert result.local_pilot_ready is True
    assert result.authorization_request_ready is True
    assert result.material_training_authorized is True


def test_missing_corpus_blocks_all_phases() -> None:
    packet = ready_packet()
    del packet["authorities"]["corpus"]  # type: ignore[index]
    result = evaluate_learned_20m_readiness(packet)
    assert result.local_pilot_ready is False
    assert "authority_corpus_missing" in result.local_pilot_blockers
    assert result.authorization_request_ready is False
    assert result.material_training_authorized is False


def test_self_asserted_authority_fails_closed() -> None:
    packet = ready_packet()
    packet["authorities"]["d05_checkpoint"]["self_asserted"] = True  # type: ignore[index]
    result = evaluate_learned_20m_readiness(packet)
    assert result.local_pilot_ready is False
    assert "authority_d05_checkpoint_self_asserted_or_unset" in result.local_pilot_blockers


def test_local_pilot_does_not_imply_compute_authorization() -> None:
    packet = ready_packet()
    packet.pop("bounded_pilot")
    packet.pop("cost_envelope")
    packet.pop("independent_audit")
    packet.pop("compute_authorization")
    packet.pop("training_authorization")
    result = evaluate_learned_20m_readiness(packet)
    assert result.local_pilot_ready is True
    assert result.authorization_request_ready is False
    assert result.material_training_authorized is False


def test_authorization_request_does_not_imply_material_training() -> None:
    packet = ready_packet()
    packet.pop("compute_authorization")
    packet.pop("training_authorization")
    result = evaluate_learned_20m_readiness(packet)
    assert result.local_pilot_ready is True
    assert result.authorization_request_ready is True
    assert result.material_training_authorized is False


def test_recipe_cannot_exceed_unique_loss_ledger() -> None:
    packet = ready_packet()
    packet["training_recipe"]["max_unique_loss_positions"] = 1_000_001  # type: ignore[index]
    result = evaluate_learned_20m_readiness(packet)
    assert "training_recipe_exceeds_unique_loss_ledger" in result.local_pilot_blockers


def test_compute_authorization_must_cover_cost_and_training_budget() -> None:
    packet = ready_packet()
    packet["compute_authorization"]["max_cost_usd"] = 9.99  # type: ignore[index]
    packet["compute_authorization"]["max_unique_loss_positions"] = 899_999  # type: ignore[index]
    result = evaluate_learned_20m_readiness(packet)
    assert result.authorization_request_ready is True
    assert result.material_training_authorized is False
    assert "compute_authorization_below_cost_envelope" in result.material_training_blockers
    assert "compute_authorization_below_training_recipe" in result.material_training_blockers


def test_model_or_r01_drift_blocks_every_phase() -> None:
    packet = ready_packet()
    packet["model"]["parameter_count"] = MODEL341_PARAMETER_COUNT + 1  # type: ignore[index]
    packet["r01"]["merge_sha"] = SHA  # type: ignore[index]
    result = evaluate_learned_20m_readiness(packet)
    assert "model341_parameter_count_mismatch" in result.local_pilot_blockers
    assert "r01_merge_sha_mismatch" in result.local_pilot_blockers
    assert result.material_training_authorized is False


def test_git_blob_identity_matches_git_object_rule() -> None:
    payload = b"hello\n"
    assert git_blob_sha1(payload) == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_r01_campaign_bytes_fail_closed_on_mutation() -> None:
    campaign_path = (
        Path(__file__).resolve().parents[1]
        / "configs/research/r01_20m_to_100m_scaling_campaign_v1.json"
    )
    payload = campaign_path.read_bytes()
    assert git_blob_sha1(payload) == R01_CAMPAIGN_BLOB_SHA1
    assert verify_r01_campaign_bytes(payload) == []
    mutated = json.loads(payload)
    mutated["authority"]["parameter_count"] += 1
    mutated_bytes = (json.dumps(mutated, sort_keys=True) + "\n").encode()
    assert verify_r01_campaign_bytes(mutated_bytes) == ["r01_campaign_blob_mismatch"]
