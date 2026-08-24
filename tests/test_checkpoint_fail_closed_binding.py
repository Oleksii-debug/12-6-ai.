from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    CheckpointIntegrityError,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)

INIT_SHA = "6" * 64
PACKING_SHA = "7" * 64
ENV_LOCK_SHA = "8" * 64


class Model:
    def __init__(self, value: float = 1.0):
        self.weights = np.array([value, value + 1.0], dtype=np.float64)
        self.load_calls = 0

    def state_dict(self):
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state, strict=True):
        self.load_calls += 1
        self.weights = state["weights"].copy()


@dataclass
class TrainerState:
    micro_step: int
    optimizer_step: int
    tokens_seen: int
    optimizer: dict[str, object]
    scheduler: dict[str, object] | None
    scaler: dict[str, object] | None
    config: dict[str, object]


class Trainer:
    def __init__(self, marker: int = 3):
        self.marker = marker
        self.load_calls = 0

    def state_dict(self):
        return TrainerState(
            micro_step=self.marker,
            optimizer_step=self.marker,
            tokens_seen=self.marker * 10,
            optimizer={"state": {0: {"momentum": np.array([float(self.marker)])}}},
            scheduler={"last_epoch": self.marker},
            scaler={},
            config={"seed": 11, "precision": "fp32"},
        )

    def load_state_dict(self, state):
        self.load_calls += 1
        self.marker = state["optimizer_step"]


def identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "fail-closed-test", "vocab_size": 256},
        parameter_count=2,
        tokenizer_hash="1" * 64,
        tokenizer_vocab_hash="2" * 64,
        dataset_manifest_hash="3" * 64,
        run_manifest_hash="4" * 64,
        training_config={
            "run_id": "fail-closed-test",
            "run_manifest_sha256": "4" * 64,
            "init_spec_sha256": INIT_SHA,
            "training": {"seed": 11, "precision": "fp32"},
            "data": {
                "split_identity": "train:fixture-sha",
                "packing_sha256": PACKING_SHA,
                "packing_version": "pack-v1",
            },
            "environment": {"lock_sha256": ENV_LOCK_SHA},
        },
        seed=11,
        precision="fp32",
        step=3,
        tokens_seen=30,
        optimizer={"name": "test"},
        scheduler={"name": "test"},
        environment_lock_hash=ENV_LOCK_SHA,
    )


def saved_checkpoint(tmp_path: Path) -> tuple[Path, dict]:
    checkpoint = tmp_path / "source"
    manifest = save_trainer_checkpoint(
        checkpoint,
        model=Model(),
        trainer=Trainer(),
        identity=identity(),
    )
    return checkpoint, manifest


def assert_target_untouched(model: Model, trainer: Trainer) -> None:
    np.testing.assert_array_equal(model.weights, np.array([99.0, 100.0]))
    assert model.load_calls == 0
    assert trainer.marker == 99
    assert trainer.load_calls == 0


@pytest.mark.parametrize(
    "mismatch",
    [
        {"expected_git_sha": "b" * 40},
        {"expected_model_spec_hash": "b" * 64},
        {"expected_init_spec_hash": "b" * 64},
        {"expected_tokenizer_hash": "b" * 64},
        {"expected_tokenizer_vocab_hash": "b" * 64},
        {"expected_dataset_manifest_hash": "b" * 64},
        {"expected_split_identity": "validation:fixture-sha"},
        {"expected_packing_hash": "b" * 64},
        {"expected_packing_version": "pack-v2"},
        {"expected_run_manifest_hash": "b" * 64},
        {"expected_training_config_hash": "b" * 64},
        {"expected_environment_lock_hash": "b" * 64},
        {"expected_seed": 12},
    ],
)
def test_identity_mismatch_fails_before_model_or_trainer_mutation(
    tmp_path: Path,
    mismatch: dict[str, object],
) -> None:
    checkpoint, _ = saved_checkpoint(tmp_path)
    target_model = Model(99.0)
    target_trainer = Trainer(99)

    with pytest.raises(CheckpointCompatibilityError):
        load_trainer_checkpoint(
            checkpoint,
            model=target_model,
            trainer=target_trainer,
            **mismatch,
        )

    assert_target_untouched(target_model, target_trainer)


def test_exact_full_binding_restores_after_preflight(tmp_path: Path) -> None:
    checkpoint, manifest = saved_checkpoint(tmp_path)
    expected = identity()
    target_model = Model(99.0)
    target_trainer = Trainer(99)

    result = load_trainer_checkpoint(
        checkpoint,
        model=target_model,
        trainer=target_trainer,
        restore_rng=False,
        expected_git_sha=expected.git_sha,
        expected_model_spec_hash=hash_json(expected.model_spec),
        expected_init_spec_hash=INIT_SHA,
        expected_tokenizer_hash=expected.tokenizer_hash,
        expected_tokenizer_vocab_hash=expected.tokenizer_vocab_hash,
        expected_dataset_manifest_hash=expected.dataset_manifest_hash,
        expected_split_identity="train:fixture-sha",
        expected_packing_hash=PACKING_SHA,
        expected_packing_version="pack-v1",
        expected_run_manifest_hash=expected.run_manifest_hash,
        expected_training_config_hash=manifest["identity"]["training_config_hash"],
        expected_environment_lock_hash=ENV_LOCK_SHA,
        expected_seed=11,
    )

    assert result.manifest["checkpoint_id"] == manifest["checkpoint_id"]
    assert target_model.load_calls == 1
    assert target_trainer.load_calls == 1
    np.testing.assert_array_equal(target_model.weights, np.array([1.0, 2.0]))
    assert target_trainer.marker == 3


@pytest.mark.parametrize("tamper", ["weights", "manifest", "manifest_checksum"])
def test_corrupt_bundle_fails_before_mutation(
    tmp_path: Path,
    tamper: str,
) -> None:
    source, _ = saved_checkpoint(tmp_path)
    corrupt = tmp_path / f"corrupt-{tamper}"
    shutil.copytree(source, corrupt)

    if tamper == "weights":
        with (corrupt / "weights.safetensors").open("ab") as handle:
            handle.write(b"tamper")
    elif tamper == "manifest":
        with (corrupt / "manifest.json").open("ab") as handle:
            handle.write(b" ")
    else:
        (corrupt / "MANIFEST.sha256").write_text(
            f"{'0' * 64}  manifest.json\n",
            encoding="ascii",
        )

    target_model = Model(99.0)
    target_trainer = Trainer(99)
    with pytest.raises(CheckpointIntegrityError):
        load_trainer_checkpoint(corrupt, model=target_model, trainer=target_trainer)
    assert_target_untouched(target_model, target_trainer)
