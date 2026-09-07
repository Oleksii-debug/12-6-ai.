from __future__ import annotations

from twelve_six.portable_run_binding import _build_candidate


FINAL_TEST_IDENTITY = "f" * 64


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
    assert packet["evaluation"]["final_test_reservation_sha256"] == FINAL_TEST_IDENTITY
