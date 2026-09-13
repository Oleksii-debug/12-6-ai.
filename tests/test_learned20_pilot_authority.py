from __future__ import annotations

from twelve_six.learned20_pilot_authority import validate_bounded_pilot_authority


def _evidence() -> dict:
    binding = {
        "identity": "launch-v1",
        "code_sha": "a" * 40,
        "config_sha256": "b" * 64,
        "corpus_identity": "corpus-v1",
        "loss_ledger_identity": "ledger-v1",
        "tokenizer_identity": "tokenizer-v1",
        "checkpoint_identity": "checkpoint-v1",
        "evaluation_firewall_identity": "eval-v1",
        "training_recipe_identity": "recipe-v1",
    }
    pilot = {
        "terminal": True,
        "identity": "pilot-v1",
        "launch_binding_identity": binding["identity"],
        "code_sha": binding["code_sha"],
        "config_sha256": binding["config_sha256"],
        "corpus_identity": binding["corpus_identity"],
        "loss_ledger_identity": binding["loss_ledger_identity"],
        "tokenizer_identity": binding["tokenizer_identity"],
        "checkpoint_identity": binding["checkpoint_identity"],
        "evaluation_firewall_identity": binding["evaluation_firewall_identity"],
        "training_recipe_identity": binding["training_recipe_identity"],
    }
    return {"launch_binding": binding, "bounded_pilot": pilot}


def test_terminal_pilot_must_bind_exact_launch_candidate() -> None:
    evidence = _evidence()
    assert validate_bounded_pilot_authority(evidence) == []


def test_stale_terminal_pilot_is_rejected() -> None:
    evidence = _evidence()
    evidence["bounded_pilot"]["config_sha256"] = "c" * 64
    evidence["bounded_pilot"]["checkpoint_identity"] = "stale-checkpoint"
    blockers = validate_bounded_pilot_authority(evidence)
    assert "bounded_pilot.config_sha256_mismatch" in blockers
    assert "bounded_pilot.checkpoint_identity_mismatch" in blockers


def test_terminal_pilot_missing_binding_fields_is_rejected() -> None:
    evidence = _evidence()
    del evidence["bounded_pilot"]["training_recipe_identity"]
    blockers = validate_bounded_pilot_authority(evidence)
    assert "bounded_pilot.training_recipe_identity_missing" in blockers


def test_nonterminal_pilot_does_not_claim_provenance() -> None:
    evidence = _evidence()
    evidence["bounded_pilot"]["terminal"] = False
    assert validate_bounded_pilot_authority(evidence) == []
