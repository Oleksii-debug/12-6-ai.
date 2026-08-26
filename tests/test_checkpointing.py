from __future__ import annotations

import copy
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from twelve_six.checkpoint import (
    CheckpointIdentity,
    CheckpointIntegrityError,
    hash_json,
    load_checkpoint,
    save_checkpoint,
    verify_checkpoint,
)


class NumpyModel:
    def __init__(self, weights: np.ndarray):
        self.weights = np.asarray(weights, dtype=np.float64).copy()

    def state_dict(self):
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state, strict=True):
        assert not strict or set(state) == {"weights"}
        self.weights = state["weights"].copy()


class MomentumSGD:
    def __init__(self, model: NumpyModel, lr=0.03, momentum=0.8):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.velocity = np.zeros_like(model.weights)

    def step(self, grad):
        self.velocity = self.momentum * self.velocity + grad
        self.model.weights -= self.lr * self.velocity

    def state_dict(self):
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "velocity": self.velocity.copy(),
        }

    def load_state_dict(self, state):
        self.lr = state["lr"]
        self.momentum = state["momentum"]
        self.velocity = state["velocity"].copy()


class StepScheduler:
    def __init__(self, optimizer: MomentumSGD, gamma=0.95):
        self.optimizer = optimizer
        self.gamma = gamma
        self.steps = 0

    def step(self):
        self.optimizer.lr *= self.gamma
        self.steps += 1

    def state_dict(self):
        return {"gamma": self.gamma, "steps": self.steps}

    def load_state_dict(self, state):
        self.gamma = state["gamma"]
        self.steps = state["steps"]


def identity(step: int, tokens_seen: int) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="f" * 40,
        model_spec={"kind": "numpy-test-model", "width": 3},
        parameter_count=3,
        tokenizer_hash="a" * 64,
        tokenizer_vocab_hash="d" * 64,
        dataset_manifest_hash="b" * 64,
        run_manifest_hash="e" * 64,
        training_config={"batch_size": 1, "max_steps": 8},
        seed=17,
        precision="float64-test",
        step=step,
        tokens_seen=tokens_seen,
        optimizer={"name": "MomentumSGD", "lr": 0.03, "momentum": 0.8},
        scheduler={"name": "StepScheduler", "gamma": 0.95},
        environment_lock_hash="c" * 64,
    )


def train_step(model, optimizer, scheduler):
    x = np.random.normal(size=3)
    target = random.uniform(-1.0, 1.0)
    pred = float(np.dot(model.weights, x))
    grad = 2.0 * (pred - target) * x
    optimizer.step(grad)
    scheduler.step()


def seeded_stack():
    random.seed(17)
    np.random.seed(17)
    model = NumpyModel(np.array([0.1, -0.2, 0.3]))
    optimizer = MomentumSGD(model)
    scheduler = StepScheduler(optimizer)
    return model, optimizer, scheduler


def test_direct_identity_rejects_weak_or_abbreviated_lineage() -> None:
    good = identity(step=0, tokens_seen=0)
    good.validate()

    with pytest.raises(ValueError, match="git_sha"):
        replace(good, git_sha="abcdef0").validate()
    with pytest.raises(ValueError, match="tokenizer_hash"):
        replace(good, tokenizer_hash="tok-hash").validate()
    with pytest.raises(ValueError, match="tokenizer_vocab_hash"):
        replace(good, tokenizer_vocab_hash="vocab-hash").validate()
    with pytest.raises(ValueError, match="dataset_manifest_hash"):
        replace(good, dataset_manifest_hash="data-hash").validate()
    with pytest.raises(ValueError, match="run_manifest_hash"):
        replace(good, run_manifest_hash="run-hash").validate()
    with pytest.raises(ValueError, match="environment_lock_hash"):
        replace(good, environment_lock_hash="lock-hash").validate()


def test_save_load_roundtrip_and_manifest(tmp_path: Path):
    model, optimizer, scheduler = seeded_stack()
    for _ in range(3):
        train_step(model, optimizer, scheduler)
    expected_weights = model.weights.copy()
    expected_velocity = optimizer.velocity.copy()
    ckpt = tmp_path / "ckpt"
    expected_identity = identity(step=3, tokens_seen=24)
    manifest = save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_state={"loss": 1.25, "micro_step": 3},
        identity=expected_identity,
    )
    assert manifest["serialization"]["pickle"] is False
    assert manifest["identity"]["model_spec_hash"] == hash_json(expected_identity.model_spec)
    assert manifest["identity"]["tokenizer_hash"] == expected_identity.tokenizer_hash
    assert manifest["identity"]["tokenizer_vocab_hash"] == expected_identity.tokenizer_vocab_hash
    assert manifest["identity"]["run_manifest_hash"] == expected_identity.run_manifest_hash
    assert verify_checkpoint(ckpt)["checkpoint_id"] == manifest["checkpoint_id"]

    model.weights[:] = 99.0
    optimizer.velocity[:] = -88.0
    result = load_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_model_spec_hash=manifest["identity"]["model_spec_hash"],
        expected_tokenizer_vocab_hash=expected_identity.tokenizer_vocab_hash,
        expected_run_manifest_hash=expected_identity.run_manifest_hash,
    )
    np.testing.assert_array_equal(model.weights, expected_weights)
    np.testing.assert_array_equal(optimizer.velocity, expected_velocity)
    assert result.trainer_state == {"loss": 1.25, "micro_step": 3}
    assert scheduler.steps == 3


def test_checksum_tamper_is_rejected_before_load(tmp_path: Path):
    model, optimizer, scheduler = seeded_stack()
    ckpt = tmp_path / "ckpt"
    save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_state={},
        identity=identity(step=0, tokens_seen=0),
    )
    weights = ckpt / "weights.safetensors"
    with weights.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(CheckpointIntegrityError, match="size mismatch|checksum mismatch"):
        load_checkpoint(ckpt, model=model)


def test_manifest_internal_identity_hash_tamper_is_rejected(tmp_path: Path):
    model, optimizer, scheduler = seeded_stack()
    ckpt = tmp_path / "ckpt"
    save_checkpoint(
        ckpt,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_state={},
        identity=identity(step=0, tokens_seen=0),
    )

    manifest_path = ckpt / "manifest.json"
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["model_spec_hash"] = "0" * 64
    manifest_path.write_text(
        __import__("json").dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest_sha = __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest()
    (ckpt / "MANIFEST.sha256").write_text(f"{manifest_sha}  manifest.json\n", encoding="ascii")

    with pytest.raises(CheckpointIntegrityError, match="model_spec_hash does not match model_spec"):
        verify_checkpoint(ckpt)


def test_interrupted_resume_matches_uninterrupted_training(tmp_path: Path):
    total_steps = 8
    split = 3

    baseline_model, baseline_optimizer, baseline_scheduler = seeded_stack()
    for _ in range(total_steps):
        train_step(baseline_model, baseline_optimizer, baseline_scheduler)
    baseline = {
        "weights": baseline_model.weights.copy(),
        "velocity": baseline_optimizer.velocity.copy(),
        "lr": baseline_optimizer.lr,
        "scheduler_steps": baseline_scheduler.steps,
        "python_rng": copy.deepcopy(random.getstate()),
        "numpy_rng": copy.deepcopy(np.random.get_state()),
    }

    interrupted_model, interrupted_optimizer, interrupted_scheduler = seeded_stack()
    for _ in range(split):
        train_step(interrupted_model, interrupted_optimizer, interrupted_scheduler)
    ckpt = tmp_path / "resume"
    save_checkpoint(
        ckpt,
        model=interrupted_model,
        optimizer=interrupted_optimizer,
        scheduler=interrupted_scheduler,
        trainer_state={"next_step": split},
        identity=identity(step=split, tokens_seen=split * 8),
    )

    random.seed(999)
    np.random.seed(999)
    resumed_model = NumpyModel(np.array([9.0, 9.0, 9.0]))
    resumed_optimizer = MomentumSGD(resumed_model, lr=9.0, momentum=0.0)
    resumed_scheduler = StepScheduler(resumed_optimizer, gamma=0.1)
    result = load_checkpoint(
        ckpt,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        restore_rng=True,
    )
    assert result.trainer_state["next_step"] == split
    for _ in range(result.trainer_state["next_step"], total_steps):
        train_step(resumed_model, resumed_optimizer, resumed_scheduler)

    np.testing.assert_array_equal(resumed_model.weights, baseline["weights"])
    np.testing.assert_array_equal(resumed_optimizer.velocity, baseline["velocity"])
    assert resumed_optimizer.lr == baseline["lr"]
    assert resumed_scheduler.steps == baseline["scheduler_steps"]
    assert random.getstate() == baseline["python_rng"]
    resumed_numpy_rng = np.random.get_state()
    assert resumed_numpy_rng[0] == baseline["numpy_rng"][0]
    np.testing.assert_array_equal(resumed_numpy_rng[1], baseline["numpy_rng"][1])
    assert resumed_numpy_rng[2:] == baseline["numpy_rng"][2:]


def test_torch_state_roundtrip_if_available(tmp_path: Path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(123)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(4, 3)
    loss = model(x).pow(2).mean()
    loss.backward()
    optimizer.step()
    expected = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    ckpt = tmp_path / "torch"
    torch_identity = CheckpointIdentity(
        git_sha="e" * 40,
        model_spec={"kind": "torch-linear-test", "in": 3, "out": 2},
        parameter_count=sum(p.numel() for p in model.parameters()),
        tokenizer_hash="1" * 64,
        tokenizer_vocab_hash="2" * 64,
        dataset_manifest_hash="3" * 64,
        run_manifest_hash="4" * 64,
        training_config={"steps": 1},
        seed=123,
        precision="float32",
        step=1,
        tokens_seen=12,
        optimizer={"name": "AdamW", "lr": 0.01},
        scheduler=None,
    )
    save_checkpoint(ckpt, model=model, optimizer=optimizer, identity=torch_identity)

    optimizer.zero_grad(set_to_none=True)
    continuation_x = torch.randn(4, 3)
    continuation_loss = model(continuation_x).pow(2).mean()
    continuation_loss.backward()
    optimizer.step()
    uninterrupted = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    torch.manual_seed(999)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    load_checkpoint(ckpt, model=model, optimizer=optimizer, restore_rng=True)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)

    optimizer.zero_grad(set_to_none=True)
    resumed_x = torch.randn(4, 3)
    resumed_loss = model(resumed_x).pow(2).mean()
    resumed_loss.backward()
    optimizer.step()
    torch.testing.assert_close(resumed_x, continuation_x, rtol=0, atol=0)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, uninterrupted[name], rtol=0, atol=0)
