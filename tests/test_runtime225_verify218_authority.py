from __future__ import annotations

from copy import deepcopy

import pytest

from twelve_six.inference.verify218_authority import (
    EXPECTED_CORPUS_SHA256,
    EXPECTED_MODEL_SPEC_SHA256,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_SOURCE_ARTIFACT_NAME,
    EXPECTED_TOKENIZER_CONFIG_SHA256,
    EXPECTED_TOKENIZER_VERSION,
    EXPECTED_TOKENIZER_VOCAB_SHA256,
    Verify218AuthorityError,
    validate_verify218_authority,
)

VERIFY_SHA = "a" * 40
SOURCE_SHA = "b" * 40
VERIFY_DIGEST = "sha256:" + "c" * 64
SOURCE_DIGEST = "sha256:" + "d" * 64
CHECKPOINT_ID = "e" * 64


def _manifest() -> dict:
    return {
        "schema": "12-6.verify218-learned-10m-authority.v1",
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
            "artifact_id": 9001,
            "artifact_name": EXPECTED_SOURCE_ARTIFACT_NAME,
            "artifact_digest": SOURCE_DIGEST,
            "workflow_run_id": 8001,
            "source_sha": SOURCE_SHA,
        },
        "checkpoint": {
            "role": "best",
            "checkpoint_id": CHECKPOINT_ID,
            "step": 200,
            "tokens_seen": 2_000_000,
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


def _validate(manifest: dict | None = None, artifact: dict | None = None, run: dict | None = None):
    return validate_verify218_authority(
        _manifest() if manifest is None else manifest,
        _artifact() if artifact is None else artifact,
        _run() if run is None else run,
        verifier_artifact_id=7001,
        verifier_artifact_digest=VERIFY_DIGEST,
        verifier_run_id=6001,
        verifier_source_sha=VERIFY_SHA,
    )


def test_runtime225_accepts_only_exact_verify218_authority() -> None:
    gate = _validate()
    assert gate["status"] == "PASS"
    assert gate["authority"]["worker_id"] == "VERIFY-218-LEARNED-10M-INDEPENDENT"
    assert gate["learned_source"] == {
        "artifact_id": 9001,
        "artifact_name": EXPECTED_SOURCE_ARTIFACT_NAME,
        "artifact_digest": SOURCE_DIGEST,
        "workflow_run_id": 8001,
        "source_sha": SOURCE_SHA,
        "checkpoint_role": "best",
        "checkpoint_id": CHECKPOINT_ID,
        "checkpoint_step": 200,
        "checkpoint_tokens_seen": 2_000_000,
        "model_spec_sha256": EXPECTED_MODEL_SPEC_SHA256,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "tokenizer_version": EXPECTED_TOKENIZER_VERSION,
        "tokenizer_config_sha256": EXPECTED_TOKENIZER_CONFIG_SHA256,
        "tokenizer_vocab_sha256": EXPECTED_TOKENIZER_VOCAB_SHA256,
        "corpus_identity_sha256": EXPECTED_CORPUS_SHA256,
    }
    assert len(gate["identity_sha256"]) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("status", "BLOCKED_NO_VERIFIED_10M"),
        ("verified_learned_10m", False),
        ("foreign_pretrained_weights", True),
        ("mechanics_only_checkpoint", True),
        ("one_step_smoke", True),
    ),
)
def test_runtime225_rejects_non_authoritative_verify218_states(field: str, value: object) -> None:
    manifest = _manifest()
    manifest[field] = value
    with pytest.raises(Verify218AuthorityError):
        _validate(manifest=manifest)


def test_runtime225_rejects_missing_independent_verification_gate() -> None:
    manifest = _manifest()
    manifest["gates"]["evaluation_non_mutation"] = False
    with pytest.raises(Verify218AuthorityError, match="required gates"):
        _validate(manifest=manifest)


def test_runtime225_rejects_model_tokenizer_corpus_or_artifact_substitution() -> None:
    mutations = (
        ("model", "model_spec_sha256", "0" * 64),
        ("model", "parameter_count", EXPECTED_PARAMETER_COUNT + 1),
        ("tokenizer", "config_sha256", "0" * 64),
        ("source", "artifact_name", "some-other-10m-artifact"),
    )
    for section, field, value in mutations:
        manifest = _manifest()
        manifest[section][field] = value
        with pytest.raises(Verify218AuthorityError):
            _validate(manifest=manifest)

    manifest = _manifest()
    manifest["corpus_identity_sha256"] = "0" * 64
    with pytest.raises(Verify218AuthorityError):
        _validate(manifest=manifest)


def test_runtime225_rejects_unlearned_or_non_best_checkpoint() -> None:
    manifest = _manifest()
    manifest["checkpoint"]["tokens_seen"] = 0
    with pytest.raises(Verify218AuthorityError):
        _validate(manifest=manifest)

    manifest = _manifest()
    manifest["checkpoint"]["role"] = "final"
    with pytest.raises(Verify218AuthorityError):
        _validate(manifest=manifest)


def test_runtime225_rejects_nonterminal_or_mismatched_verifier_transport() -> None:
    run = _run()
    run["conclusion"] = "failure"
    with pytest.raises(Verify218AuthorityError, match="terminal SUCCESS"):
        _validate(run=run)

    artifact = deepcopy(_artifact())
    artifact["workflow_run"]["head_sha"] = "f" * 40
    with pytest.raises(Verify218AuthorityError, match="workflow provenance"):
        _validate(artifact=artifact)
