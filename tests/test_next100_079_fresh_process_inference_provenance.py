from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

from twelve_six.checkpoint import CheckpointIdentity, save_checkpoint
from twelve_six.inference.contracts import GenerationConfig
from twelve_six.inference.fresh_process import (
    _receipt_identity,
    generation_config_identity,
    prompt_payload_identity,
    run_fresh_process_probe,
    validate_fresh_process_receipt,
)
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]
S0_STAGE = ROOT / "configs/stages/s0_10k.json"


def _tiny_verified_checkpoint(tmp_path: Path) -> tuple[Path, str]:
    stage = load_stage_config(S0_STAGE)
    model = TwelveSixDecoder(stage.model, stage.init)
    tokenizer = ByteTokenizer()
    identity = CheckpointIdentity(
        git_sha="1" * 40,
        model_spec=stage.model.to_dict(),
        parameter_count=stage.model.parameter_count(),
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash="2" * 64,
        run_manifest_hash="3" * 64,
        training_config={
            "training": {"context_length": stage.model.max_seq_len},
            "data": {"tokenizer_version": tokenizer.identity.version},
        },
        seed=7,
        precision="float32",
        step=1,
        tokens_seen=16,
        optimizer={"name": "test-only", "lr": 0.0},
        scheduler=None,
    )
    checkpoint = tmp_path / "checkpoint"
    manifest = save_checkpoint(checkpoint, model=model, identity=identity)
    return checkpoint, manifest["checkpoint_id"]


def _expectations(receipt: dict[str, object]) -> dict[str, str]:
    return {
        "expected_receipt_identity": str(receipt["receipt_identity"]),
        "expected_checkpoint_identity": str(receipt["checkpoint_identity"]),
        "expected_prompt_suite_identity": str(receipt["prompt_suite_identity"]),
        "expected_prompt_payload_sha256": str(receipt["prompt_payload_sha256"]),
        "expected_generation_config_identity": str(receipt["generation_config_identity"]),
    }


def test_real_fresh_process_probe_binds_verified_checkpoint(tmp_path: Path) -> None:
    checkpoint, checkpoint_id = _tiny_verified_checkpoint(tmp_path)
    config = GenerationConfig(max_new_tokens=1, sample=False, seed=11)
    receipt = run_fresh_process_probe(
        checkpoint,
        expected_checkpoint_identity=checkpoint_id,
        prompt_token_ids=(65,),
        prompt_suite_identity="test-suite:v1",
        config=config,
        cache_mode="stateless",
    )

    assert receipt["checkpoint_identity"] == checkpoint_id
    assert receipt["child_pid"] != os.getpid()
    assert receipt["parent_pid"] == os.getpid()
    assert receipt["prompt_payload_sha256"] == prompt_payload_identity((65,))
    assert receipt["generation_config_identity"] == generation_config_identity(
        config, cache_mode="stateless"
    )
    assert validate_fresh_process_receipt(receipt, **_expectations(receipt)) == []


def test_wrong_expected_checkpoint_is_rejected_before_receipt_acceptance(tmp_path: Path) -> None:
    checkpoint, checkpoint_id = _tiny_verified_checkpoint(tmp_path)
    with pytest.raises(Exception, match="wrong checkpoint"):
        run_fresh_process_probe(
            checkpoint,
            expected_checkpoint_identity="f" * 64 if checkpoint_id != "f" * 64 else "e" * 64,
            prompt_token_ids=(65,),
            prompt_suite_identity="test-suite:v1",
            config=GenerationConfig(max_new_tokens=1),
        )


def test_receipt_is_closed_world_even_if_resealed(tmp_path: Path) -> None:
    checkpoint, checkpoint_id = _tiny_verified_checkpoint(tmp_path)
    receipt = run_fresh_process_probe(
        checkpoint,
        expected_checkpoint_identity=checkpoint_id,
        prompt_token_ids=(65,),
        prompt_suite_identity="test-suite:v1",
        config=GenerationConfig(max_new_tokens=1),
    )
    forged = copy.deepcopy(receipt)
    forged["caller_asserted_fresh_process"] = True
    forged["receipt_identity"] = _receipt_identity(forged)
    blockers = validate_fresh_process_receipt(
        forged,
        expected_receipt_identity=str(receipt["receipt_identity"]),
        expected_checkpoint_identity=checkpoint_id,
        expected_prompt_suite_identity="test-suite:v1",
        expected_prompt_payload_sha256=str(receipt["prompt_payload_sha256"]),
        expected_generation_config_identity=str(receipt["generation_config_identity"]),
    )
    assert any("receipt_keys_unknown" in blocker for blocker in blockers)


def test_same_process_claim_fails_even_with_coherent_reseal(tmp_path: Path) -> None:
    checkpoint, checkpoint_id = _tiny_verified_checkpoint(tmp_path)
    receipt = run_fresh_process_probe(
        checkpoint,
        expected_checkpoint_identity=checkpoint_id,
        prompt_token_ids=(65,),
        prompt_suite_identity="test-suite:v1",
        config=GenerationConfig(max_new_tokens=1),
    )
    forged = copy.deepcopy(receipt)
    forged["child_pid"] = forged["parent_pid"]
    from twelve_six.inference import fresh_process as fp

    forged["process_run_identity"] = fp._process_run_identity(forged)
    forged["receipt_identity"] = fp._receipt_identity(forged)
    blockers = validate_fresh_process_receipt(
        forged,
        expected_receipt_identity=str(forged["receipt_identity"]),
        expected_checkpoint_identity=checkpoint_id,
        expected_prompt_suite_identity="test-suite:v1",
        expected_prompt_payload_sha256=str(receipt["prompt_payload_sha256"]),
        expected_generation_config_identity=str(receipt["generation_config_identity"]),
    )
    assert "d07.fresh_process.process_not_fresh" in blockers


def test_semantic_substitution_cannot_reuse_authorized_receipt_identity(tmp_path: Path) -> None:
    checkpoint, checkpoint_id = _tiny_verified_checkpoint(tmp_path)
    receipt = run_fresh_process_probe(
        checkpoint,
        expected_checkpoint_identity=checkpoint_id,
        prompt_token_ids=(65,),
        prompt_suite_identity="test-suite:v1",
        config=GenerationConfig(max_new_tokens=1),
    )
    forged = copy.deepcopy(receipt)
    forged["prompt_suite_identity"] = "different-suite:v1"
    forged["receipt_identity"] = _receipt_identity(forged)
    blockers = validate_fresh_process_receipt(
        forged,
        expected_receipt_identity=str(receipt["receipt_identity"]),
        expected_checkpoint_identity=checkpoint_id,
        expected_prompt_suite_identity="test-suite:v1",
        expected_prompt_payload_sha256=str(receipt["prompt_payload_sha256"]),
        expected_generation_config_identity=str(receipt["generation_config_identity"]),
    )
    assert "d07.fresh_process.prompt_suite_identity_mismatch" in blockers
    assert "d07.fresh_process.receipt_identity_expected_mismatch" in blockers


@pytest.mark.parametrize("bad_token", [True, -1, "65"])
def test_prompt_payload_identity_rejects_invalid_tokens(bad_token: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        prompt_payload_identity((bad_token,))  # type: ignore[arg-type]


def test_stochastic_probe_is_rejected() -> None:
    with pytest.raises(ValueError, match="sample=False"):
        generation_config_identity(
            GenerationConfig(max_new_tokens=1, sample=True),
            cache_mode="stateless",
        )
