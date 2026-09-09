from __future__ import annotations

from twelve_six.portable_run_binding import _build_candidate


FINAL_TEST_IDENTITY = "f" * 64
POSTPACK_IDENTITY = "8" * 64


def test_portable_candidate_retains_verified_decontamination_authority() -> None:
    authority = {
        "repository": "Oleksii-debug/12-6-ai.",
        "git_sha": "a" * 40,
        "evidence_sha256": "b" * 64,
        "workflow_run_id": 123,
        "workflow_conclusion": "success",
        "terminal": True,
    }
    final_test_authority = {
        "repository": "Oleksii-debug/12-6-ai.",
        "git_sha": "9" * 40,
        "evidence_sha256": FINAL_TEST_IDENTITY,
        "workflow_run_id": 456,
        "workflow_conclusion": "success",
        "terminal": True,
    }
    postpack_authority = {
        "repository": "Oleksii-debug/12-6-ai.",
        "git_sha": "7" * 40,
        "evidence_sha256": POSTPACK_IDENTITY,
        "workflow_run_id": 789,
        "workflow_conclusion": "success",
        "terminal": True,
    }
    readiness = {
        "campaign_id": "R01-LEARNED-20M-LAUNCH-V1",
        "model_authority": {
            "git_sha": "c" * 40,
            "modelspec_sha256": "d" * 64,
            "canonical_base": "random_init",
            "parameter_count": 20_613_440,
        },
        "evidence": {
            "code": {"git_sha": "e" * 40},
            "corpus": {"authority": authority},
            "tokenizer": {"authority": authority},
            "loss_ledger": {"authority": authority},
            "postpack_proof": {
                "schema_version": "12-6.d04-deterministic-double-pack-proof.v1",
                "authority": postpack_authority,
                "proof_identity_sha256": POSTPACK_IDENTITY,
                "terminal_corpus_authority_identity_sha256": "1" * 64,
                "terminal_record_inventory_digest_sha256": "2" * 64,
                "terminal_payload_inventory_digest_sha256": "3" * 64,
                "stage_bindings": {
                    "normalization": "4" * 64,
                    "evaluation_reservations": "5" * 64,
                    "dedup": "6" * 64,
                    "split": "a" * 64,
                    "packing": "b" * 64,
                },
                "tokenizer_identity_sha256": "c" * 64,
                "packing_identity_sha256": "d" * 64,
                "ledger_identity_sha256": "e" * 64,
                "canonical_build_sha256": "f" * 64,
                "one_pass_unique_nonignored_causal_loss_positions": 1234,
                "independent_builds_byte_identical": True,
                "training_authorized_by_this_proof": False,
            },
            "checkpoint_integrity": {"authority": authority},
            "evaluation": {
                "firewall_authority": authority,
                "final_test_reservation_authority": final_test_authority,
                "decontamination": {
                    "authority": authority,
                    "final_test_identity": FINAL_TEST_IDENTITY,
                },
            },
            "training_recipe": {},
        },
    }
    template = {
        "status": "BLOCKED_TEMPLATE",
        "identities": {},
        "authorities": {},
        "postpack": {},
        "recipe": {},
        "checkpoint": {},
        "evaluation": {},
        "runtime": {},
        "resource": {},
        "output": {},
    }
    overlay = {
        "scientific_bindings": {
            "authorities": {"code": authority, "model": authority, "backend": authority}
        },
        "checkpoint": {},
        "evaluation": {},
        "runtime": {},
        "resource": {},
        "output": {},
    }

    packet = _build_candidate(
        readiness,
        template,
        overlay,
        readiness_sha256="f" * 64,
        overlay_sha256="1" * 64,
    )

    assert packet["authorities"]["decontamination"] == authority
    assert packet["authorities"]["decontamination"] is not authority
    assert packet["authorities"]["final_test_reservation"] == final_test_authority
    assert packet["authorities"]["final_test_reservation"] is not final_test_authority
    assert packet["authorities"]["postpack_proof"] == postpack_authority
    assert packet["authorities"]["postpack_proof"] is not postpack_authority
    assert packet["evaluation"]["final_test_reservation_sha256"] == FINAL_TEST_IDENTITY
    assert packet["postpack"]["proof_identity_sha256"] == POSTPACK_IDENTITY
    assert packet["postpack"]["one_pass_unique_loss_positions"] == 1234
    assert packet["postpack"]["training_authorized_by_proof"] is False
