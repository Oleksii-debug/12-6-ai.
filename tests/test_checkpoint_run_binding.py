from __future__ import annotations

import copy

import pytest

from twelve_six.checkpoint import CheckpointCompatibilityError, bind_checkpoint_identity, hash_json

TOKENIZER_SHA = "b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1"
VOCAB_SHA = "905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571"
DATASET_SHA = "b085a7ab56510575a11a80824fcff3a95a17f237d46d1be820e59d1289f220c2"
GIT_SHA = "a" * 40


def model_spec() -> dict[str, object]:
    return {
        "vocab_size": 256,
        "max_seq_len": 128,
        "d_model": 20,
        "n_layers": 1,
        "n_heads": 2,
        "d_ff": 56,
    }


def tokenizer_identity() -> dict[str, object]:
    return {
        "version": "s0-byte-v1",
        "config_sha256": TOKENIZER_SHA,
        "vocab_sha256": VOCAB_SHA,
        "vocab_size": 256,
        "normalization": "none",
        "encoding": "utf-8-bytes",
        "special_tokens": {},
    }


def run_manifest() -> dict[str, object]:
    spec = model_spec()
    return {
        "schema_version": 1,
        "run_id": "s0-local-cpu-0001",
        "stage": "S0",
        "run_kind": "integrated_training",
        "state": "RUNNING",
        "candidate": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": GIT_SHA,
            "branch_or_tag": "candidate/s0",
            "modelspec_sha256": hash_json(spec),
            "parameter_count": 10140,
        },
        "data": {
            "dataset_manifest_sha256": DATASET_SHA,
            "tokenizer_sha256": TOKENIZER_SHA,
            "tokenizer_vocab_sha256": VOCAB_SHA,
            "tokenizer_version": "s0-byte-v1",
            "split_identity": "s0-tiny-controlled-v1",
        },
        "training": {
            "seed": 17,
            "device": "cpu",
            "precision": "fp32",
            "optimizer": {"name": "AdamW", "lr": 0.001},
            "scheduler": {"name": "constant"},
            "context_length": 128,
            "global_batch_tokens": 256,
            "target_steps": 20,
            "target_tokens": 5120,
            "checkpoint_interval_steps": 5,
        },
    }


def test_binds_exact_run_manifest_identity() -> None:
    spec = model_spec()
    manifest = run_manifest()
    identity = bind_checkpoint_identity(
        run_manifest=manifest,
        model_spec=spec,
        tokenizer_identity=tokenizer_identity(),
        step=5,
        tokens_seen=1280,
        environment_lock_hash="c" * 64,
    )

    assert identity.git_sha == GIT_SHA
    assert identity.model_spec == spec
    assert identity.parameter_count == 10140
    assert identity.tokenizer_hash == TOKENIZER_SHA
    assert identity.tokenizer_vocab_hash == VOCAB_SHA
    assert identity.dataset_manifest_hash == DATASET_SHA
    assert identity.run_manifest_hash == hash_json(manifest)
    assert identity.training_config["run_id"] == "s0-local-cpu-0001"
    assert identity.training_config["run_manifest_sha256"] == hash_json(manifest)
    assert identity.training_config["data"]["tokenizer_version"] == "s0-byte-v1"
    assert identity.training_config["data"]["tokenizer_vocab_sha256"] == VOCAB_SHA
    assert identity.optimizer == {"name": "AdamW", "lr": 0.001}
    assert identity.scheduler == {"name": "constant"}


def test_rejects_model_tokenizer_vocab_drift() -> None:
    tokenizer = tokenizer_identity()
    tokenizer["vocab_size"] = 259
    with pytest.raises(CheckpointCompatibilityError, match="vocab mismatch"):
        bind_checkpoint_identity(
            run_manifest=run_manifest(),
            model_spec=model_spec(),
            tokenizer_identity=tokenizer,
            step=0,
            tokens_seen=0,
        )


def test_rejects_tokenizer_hash_drift() -> None:
    manifest = run_manifest()
    manifest["data"]["tokenizer_sha256"] = "d" * 64
    with pytest.raises(CheckpointCompatibilityError, match="tokenizer SHA"):
        bind_checkpoint_identity(
            run_manifest=manifest,
            model_spec=model_spec(),
            tokenizer_identity=tokenizer_identity(),
            step=0,
            tokens_seen=0,
        )


def test_rejects_tokenizer_vocabulary_hash_drift() -> None:
    manifest = run_manifest()
    manifest["data"]["tokenizer_vocab_sha256"] = "d" * 64
    with pytest.raises(CheckpointCompatibilityError, match="vocabulary SHA"):
        bind_checkpoint_identity(
            run_manifest=manifest,
            model_spec=model_spec(),
            tokenizer_identity=tokenizer_identity(),
            step=0,
            tokens_seen=0,
        )


def test_rejects_modelspec_hash_drift() -> None:
    manifest = run_manifest()
    manifest["candidate"]["modelspec_sha256"] = "e" * 64
    with pytest.raises(CheckpointCompatibilityError, match="ModelSpec hash"):
        bind_checkpoint_identity(
            run_manifest=manifest,
            model_spec=model_spec(),
            tokenizer_identity=tokenizer_identity(),
            step=0,
            tokens_seen=0,
        )


def test_rejects_abbreviated_git_sha_and_unresolved_run() -> None:
    manifest = run_manifest()
    manifest["candidate"]["git_sha"] = "abcdef0"
    with pytest.raises(CheckpointCompatibilityError, match="full 40- or 64-hex Git SHA"):
        bind_checkpoint_identity(
            run_manifest=manifest,
            model_spec=model_spec(),
            tokenizer_identity=tokenizer_identity(),
            step=0,
            tokens_seen=0,
        )

    unresolved = copy.deepcopy(run_manifest())
    unresolved["run_id"] = "UNRESOLVED"
    with pytest.raises(CheckpointCompatibilityError, match="run_id is unresolved"):
        bind_checkpoint_identity(
            run_manifest=unresolved,
            model_spec=model_spec(),
            tokenizer_identity=tokenizer_identity(),
            step=0,
            tokens_seen=0,
        )
