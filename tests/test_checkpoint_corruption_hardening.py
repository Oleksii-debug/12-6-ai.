from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    CheckpointIntegrityError,
    canonical_json_bytes if False else hash_json,
    load_checkpoint,
    load_trainer_checkpoint,
    save_checkpoint,
)


class NumpyModel:
    def __init__(self, weights: np.ndarray):
        self.weights = np.asarray(weights).copy()

    def state_dict(self):
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state, strict=True):
        assert not strict or set(state) == {"weights"}
        self.weights = state["weights"].copy()


def identity(*, step: int = 1, tokens_seen: int = 8, parameter_count: int = 3):
    return CheckpointIdentity(
        git_sha="f" * 40,
        model_spec={"kind": "hardening-test", "width": parameter_count},
        parameter_count=parameter_count,
        tokenizer_hash="a" * 64,
        tokenizer_vocab_hash="b" * 64,
        dataset_manifest_hash="c" * 64,
        run_manifest_hash="d" * 64,
        training_config={"batch_size": 1, "max_steps": 8},
        seed=17,
        precision="test",
        step=step,
        tokens_seen=tokens_seen,
        optimizer={"name": "test"},
        scheduler=None,
        environment_lock_hash="e" * 64,
    )


def _rebind_manifest(path: Path, mutate) -> None:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest["checkpoint_id"] = hash_json(
        {"identity": manifest["identity"], "files": manifest["files"]}
    )
    manifest_bytes = (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    manifest_path.write_bytes(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    (path / "MANIFEST.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )


def test_dtype_mismatch_rejected_before_numpy_model_mutation(tmp_path: Path):
    source = NumpyModel(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    checkpoint = tmp_path / "dtype"
    save_checkpoint(checkpoint, model=source, identity=identity())

    target = NumpyModel(np.array([9.0, 9.0, 9.0], dtype=np.float32))
    before = target.weights.copy()
    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)
    np.testing.assert_array_equal(target.weights, before)
    assert target.weights.dtype == np.float32


def test_rebound_negative_manifest_counters_fail_closed(tmp_path: Path):
    source = NumpyModel(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    checkpoint = tmp_path / "negative-counters"
    save_checkpoint(checkpoint, model=source, identity=identity())

    def corrupt(manifest):
        manifest["identity"]["step"] = -1
        manifest["identity"]["tokens_seen"] = -32

    _rebind_manifest(checkpoint, corrupt)
    target = NumpyModel(np.array([7.0, 7.0, 7.0], dtype=np.float64))
    before = target.weights.copy()
    with pytest.raises(CheckpointIntegrityError, match="step.*non-negative"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)
    np.testing.assert_array_equal(target.weights, before)


def test_expected_counter_binding_rejects_before_model_mutation(tmp_path: Path):
    source = NumpyModel(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    checkpoint = tmp_path / "counter-binding"
    save_checkpoint(checkpoint, model=source, identity=identity(step=4, tokens_seen=64))

    target = NumpyModel(np.array([5.0, 5.0, 5.0], dtype=np.float64))
    before = target.weights.copy()
    with pytest.raises(CheckpointCompatibilityError, match="counter identity mismatch"):
        load_checkpoint(
            checkpoint,
            model=target,
            restore_rng=False,
            expected_step=3,
            expected_tokens_seen=64,
        )
    np.testing.assert_array_equal(target.weights, before)


def test_wrong_shaped_sgd_momentum_rejected_before_model_mutation(tmp_path: Path):
    torch = pytest.importorskip("torch")

    source_model = torch.nn.Linear(3, 2)

    class CorruptSGDState:
        def state_dict(self):
            return {
                "state": {
                    0: {"momentum_buffer": torch.zeros(1, dtype=source_model.weight.dtype)},
                },
                "param_groups": [{"params": [0, 1]}],
            }

    checkpoint = tmp_path / "optimizer-shape"
    save_checkpoint(
        checkpoint,
        model=source_model,
        optimizer=CorruptSGDState(),
        identity=identity(parameter_count=sum(p.numel() for p in source_model.parameters())),
    )

    target_model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(target_model.parameters(), lr=0.1, momentum=0.9)
    before = {name: tensor.detach().clone() for name, tensor in target_model.state_dict().items()}

    with pytest.raises(CheckpointCompatibilityError, match="optimizer state.*shape mismatch"):
        load_checkpoint(
            checkpoint,
            model=target_model,
            optimizer=optimizer,
            restore_rng=False,
        )
    for name, tensor in target_model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    assert optimizer.state == {}


@dataclass(frozen=True)
class FakeConfig:
    gradient_accumulation_steps: int = 1
    max_steps: int = 8


class FakeTrainer:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.scheduler = None
        self.config = FakeConfig()
        self.loaded = False

    def load_state_dict(self, state):
        self.loaded = True
        self.optimizer.load_state_dict(state["optimizer"])


def test_nested_trainer_optimizer_corruption_rejected_before_model_mutation(tmp_path: Path):
    torch = pytest.importorskip("torch")
    source_model = torch.nn.Linear(3, 2)
    bad_trainer_state = {
        "micro_step": 1,
        "optimizer_step": 1,
        "tokens_seen": 8,
        "optimizer": {
            "state": {0: {"momentum_buffer": torch.zeros(1)}},
            "param_groups": [{"params": [0, 1]}],
        },
        "scheduler": None,
        "scaler": None,
        "config": {
            "gradient_accumulation_steps": 1,
            "max_steps": 8,
        },
    }
    checkpoint = tmp_path / "trainer-optimizer-shape"
    save_checkpoint(
        checkpoint,
        model=source_model,
        trainer_state=bad_trainer_state,
        identity=identity(parameter_count=sum(p.numel() for p in source_model.parameters())),
    )

    target_model = torch.nn.Linear(3, 2)
    target_optimizer = torch.optim.SGD(target_model.parameters(), lr=0.1, momentum=0.9)
    trainer = FakeTrainer(target_optimizer)
    before = {name: tensor.detach().clone() for name, tensor in target_model.state_dict().items()}

    with pytest.raises(CheckpointCompatibilityError, match="optimizer state.*shape mismatch"):
        load_trainer_checkpoint(
            checkpoint,
            model=target_model,
            trainer=trainer,
            restore_rng=False,
        )
    for name, tensor in target_model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    assert not trainer.loaded
    assert target_optimizer.state == {}
