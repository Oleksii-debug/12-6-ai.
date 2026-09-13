from __future__ import annotations

from twelve_six.learned20_pilot_authority import (
    validate_bounded_pilot_authority,
    validate_d06_d07_summary_consistency,
)


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


def test_d06_legacy_summary_must_equal_d07_receipt() -> None:
    checkpoint = "a" * 64
    fingerprint = "sha256:" + "b" * 64
    evidence = {
        "bounded_pilot": {
            "terminal": True,
            "d06_evaluation": {
                "inference_probe": {
                    "prompt_suite_identity": "suite-v1",
                    "output_fingerprint": fingerprint,
                    "checkpoint_identity": checkpoint,
                    "fresh_process_receipt": {
                        "prompt_suite_identity": "suite-v1",
                        "output_fingerprint": fingerprint,
                        "checkpoint_identity": checkpoint,
                    },
                }
            },
        }
    }
    assert validate_d06_d07_summary_consistency(evidence) == []

    evidence["bounded_pilot"]["d06_evaluation"]["inference_probe"][
        "output_fingerprint"
    ] = "sha256:" + "c" * 64
    assert validate_d06_d07_summary_consistency(evidence) == [
        "bounded_pilot.d06.inference_probe.output_fingerprint_receipt_mismatch"
    ]


def test_wrapper_never_trusts_d07_expectations_from_contract(monkeypatch) -> None:
    import twelve_six.learned20_pilot_authority as pilot_authority

    observed: list[object] = []

    monkeypatch.setattr(
        pilot_authority,
        "assess_launch_with_checkpoint_provenance",
        lambda contract, evidence, *, material_cost: {
            "pilot_ready": True,
            "long_training_ready": True,
            "pilot_blockers": [],
            "long_training_blockers": [],
        },
    )
    monkeypatch.setattr(
        pilot_authority,
        "validate_evaluation_firewall_provenance",
        lambda evidence: [],
    )
    monkeypatch.setattr(
        pilot_authority,
        "validate_bounded_pilot_authority",
        lambda evidence: [],
    )

    def fake_terminal(
        evidence,
        *,
        fresh_process_expectations=None,
    ):
        observed.append(fresh_process_expectations)
        return []

    monkeypatch.setattr(
        pilot_authority,
        "validate_terminal_pilot_evaluation",
        fake_terminal,
    )

    attacker_controlled = {"receipt_identity": "attacker-resealed"}
    pilot_authority.assess_launch_with_terminal_provenance(
        {"d07_fresh_process_expectations": attacker_controlled},
        {},
        material_cost=False,
    )
    assert observed[-1] is None

    independently_authenticated = {"receipt_identity": "trusted-external-root"}
    pilot_authority.assess_launch_with_terminal_provenance(
        {"d07_fresh_process_expectations": attacker_controlled},
        {},
        material_cost=False,
        trusted_fresh_process_expectations=independently_authenticated,
    )
    assert observed[-1] is independently_authenticated