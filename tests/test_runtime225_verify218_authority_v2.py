from __future__ import annotations

from copy import deepcopy

import pytest

from twelve_six.inference.verify218_authority import (
    EXPECTED_CORPUS_SHA256,
    EXPECTED_MODEL_SPEC_SHA256,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_TOKENIZER_CONFIG_SHA256,
    EXPECTED_TOKENIZER_VERSION,
    EXPECTED_TOKENIZER_VOCAB_SHA256,
)
from twelve_six.inference.verify218_authority_v2 import (
    EXPECTED_SOURCE_ARTIFACT_DIGEST,
    EXPECTED_SOURCE_ARTIFACT_ID,
    EXPECTED_SOURCE_ARTIFACT_NAME,
    EXPECTED_SOURCE_SHA,
    EXPECTED_SOURCE_WORKFLOW_RUN_ID,
    GATE_SCHEMA,
    Verify218AuthorityV2Error,
    validate_verify218_authority_v2,
)

VERIFY_SHA = "a" * 40
VERIFY_DIGEST = "sha256:" + "c" * 64
CHECKPOINT_ID = "e" * 64


def _manifest() -> dict:
    return {
        "schema": "12-6.verify218-learned-10m-independent.v2",
        "worker_id": "VERIFY-218-LEARNED-10M-INDEPENDENT",
        "status": "VERIFIED_LEARNED_10M",
        "verified_learned_10m": True,
        "foreign_pretrained_weights": False,
        "mechanics_only_checkpoint": False,
        "one_step_smoke": False,
        "gates": {
            "checkpoint_integrity": True,
            "fresh_process_resume": True,
            "finite_first_party_logits": True,
            "heldout_bpb": True,
            "evaluation_non_mutation": True,
            "greedy_generation": True,
            "best_final_role_resolution": True,
        },
        "model": {
            "model_spec_sha256": EXPECTED_MODEL_SPEC_SHA256,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
        },
        "tokenizer": {
            "version": EXPECTED_TOKENIZER_VERSION,
            "config_sha256": EXPECTED_TOKENIZER_CONFIG_SHA256,
            "vocab_sha256": EXPECTED_TOKENIZER_VOCAB_SHA256,
        },
        "corpus_identity_sha256": EXPECTED_CORPUS_SHA256,
        "source": {
            "artifact_id": EXPECTED_SOURCE_ARTIFACT_ID,
            "artifact_name": EXPECTED_SOURCE_ARTIFACT_NAME,
            "artifact_digest": EXPECTED_SOURCE_ARTIFACT_DIGEST,
            "workflow_run_id": EXPECTED_SOURCE_WORKFLOW_RUN_ID,
            "source_sha": EXPECTED_SOURCE_SHA,
        },
        "checkpoint": {
            "role": "best",
            "checkpoint_id": CHECKPOINT_ID,
            "step": 4903,
            "tokens_seen": 1_000_133,
        },
    }


def _artifact() -> dict:
    return {
        "id": 7001,
        "digest": VERIFY_DIGEST,
        "expired": False,
        "workflow_run": {"id": 6001, "head_sha": VERIFY_SHA},
    }


def _run() -> dict:
    return {
        "id": 6001,
        "head_sha": VERIFY_SHA,
        "status": "completed",
        "conclusion": "success",
    }


def _validate(manifest: dict | None = None) -> dict:
    return validate_verify218_authority_v2(
        _manifest() if manifest is None else manifest,
        _artifact(),
        _run(),
        verifier_artifact_id=7001,
        verifier_artifact_digest=VERIFY_DIGEST,
        verifier_run_id=6001,
        verifier_source_sha=VERIFY_SHA,
    )


def test_v2_accepts_exact_terminal_learn217_source() -> None:
    gate = _validate()
    assert gate["schema"] == GATE_SCHEMA
    assert gate["status"] == "PASS"
    assert gate["learned_source"]["artifact_id"] == EXPECTED_SOURCE_ARTIFACT_ID
    assert gate["learned_source"]["artifact_name"] == EXPECTED_SOURCE_ARTIFACT_NAME
    assert gate["learned_source"]["artifact_digest"] == EXPECTED_SOURCE_ARTIFACT_DIGEST
    assert gate["learned_source"]["workflow_run_id"] == EXPECTED_SOURCE_WORKFLOW_RUN_ID
    assert gate["learned_source"]["source_sha"] == EXPECTED_SOURCE_SHA
    assert gate["learned_source"]["checkpoint_id"] == CHECKPOINT_ID
    assert gate["truth_boundary"]["exact_terminal_learn217_source_bound"] is True
    assert len(gate["identity_sha256"]) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("artifact_id", 9602650342),
        ("artifact_name", "scale141-10m-learned-fallback"),
        ("artifact_digest", "sha256:" + "0" * 64),
        ("workflow_run_id", 32952787071),
        ("source_sha", "0" * 40),
    ),
)
def test_v2_rejects_any_learned_source_substitution(field: str, value: object) -> None:
    manifest = deepcopy(_manifest())
    manifest["source"][field] = value
    with pytest.raises(Verify218AuthorityV2Error, match="not exact terminal LEARN-217"):
        _validate(manifest)


def test_v2_retains_v1_scientific_gate_fail_closed_behavior() -> None:
    manifest = deepcopy(_manifest())
    manifest["gates"]["fresh_process_resume"] = False
    with pytest.raises(Exception, match="required gates"):
        _validate(manifest)
