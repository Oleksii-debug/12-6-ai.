from __future__ import annotations

import copy
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
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


def canonical_bound_identity() -> CheckpointIdentity:
    run_hash = "e" * 64
    return CheckpointIdentity(
        git_sha="f" * 40,
        model_spec={"kind": "canonical-bound-test", "width": 3},
        parameter_count=3,
        tokenizer_hash="a" * 64,
        tokenizer_vocab_hash="d" * 64,
        dataset_manifest_hash="b" * 64,
        run_manifest_hash=run_hash,
        training_config={
            "run_id": "canonical-test-run",
            "run_manifest_sha256": run_hash,
            "stage": "TEST",
            "run_kind": "integrated_training",
            "init_spec_sha256": "1" * 64,
            "training": {
                "seed": 17,
                "precision": "float64-test",
                "optimizer": "MomentumSGD",
                "scheduler": "StepScheduler",
            },
            "data": {
                "dataset_manifest_sha256": "b" * 64,
                "split_identity": "canonical-test:train",
                "tokenizer_sha256": "a" * 64,
                "tokenizer_vocab_sha256": "d" * 64,
                "tokenizer_version": "test-tokenizer-v1",
                "packing_sha256": "2" * 64,
                "packing_version": "test-pack-v1",
            },
            "environment": {"lock_sha256": "c" * 64},
        },
        seed=17,
        precision="float64-test",
        step=0,
        tokens_seen=0,
        optimizer={"name": "MomentumSGD"},
        scheduler={"name": "StepScheduler"},
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


@pytest.mark.parametrize("scheduler_method", ["state_dict", "load_state_dict"])
def test_noncallable_scheduler_interface_rejected_before_model_mutation(
    tmp_path: Path, scheduler_method: str,
) -> None:
    source_model, source_optimizer, source_scheduler = seeded_stack()
    train_step(source_model, source_optimizer, source_scheduler)
    ckpt = tmp_path / "noncallable-scheduler-interface"
    save_checkpoint(
        ckpt,
        model=source_model,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        trainer_state={},
        identity=identity(step=1, tokens_seen=8),
    )

    target_model, target_optimizer, target_scheduler = seeded_stack()
    target_model.weights[:] = 77.0
    target_optimizer.velocity[:] = -55.0
    weights_before = target_model.weights.copy()
    velocity_before = target_optimizer.velocity.copy()
    setattr(target_scheduler, scheduler_method, None)

    with pytest.raises(
        CheckpointCompatibilityError,
        match="scheduler must provide state_dict/load_state_dict",
    ):
        load_checkpoint(
            ckpt,
            model=target_model,
            optimizer=target_optimizer,
            scheduler=target_scheduler,
            restore_rng=False,
        )

    np.testing.assert_array_equal(target_model.weights, weights_before)
    np.testing.assert_array_equal(target_optimizer.velocity, velocity_before)
    assert target_scheduler.steps == 0


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



def test_checkpoint_save_is_rng_neutral_across_effectful_state_exports(
    tmp_path: Path,
) -> None:
    class EffectfulModel(NumpyModel):
        def state_dict(self):
            random.random()
            np.random.random()
            return super().state_dict()

    class EffectfulOptimizer(MomentumSGD):
        def state_dict(self):
            random.random()
            np.random.random()
            return super().state_dict()

    class EffectfulScheduler(StepScheduler):
        def state_dict(self):
            random.random()
            np.random.random()
            return super().state_dict()

    random.seed(1701)
    np.random.seed(1701)
    model = EffectfulModel(np.array([0.1, -0.2, 0.3]))
    optimizer = EffectfulOptimizer(model)
    scheduler = EffectfulScheduler(optimizer)
    python_before = copy.deepcopy(random.getstate())
    numpy_before = copy.deepcopy(np.random.get_state())

    checkpoint = tmp_path / "rng-neutral-save"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_state={"next_step": 0},
        identity=identity(step=0, tokens_seen=0),
    )

    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]

    random.seed(999)
    np.random.seed(999)
    target_model = NumpyModel(np.array([9.0, 9.0, 9.0]))
    target_optimizer = MomentumSGD(target_model)
    target_scheduler = StepScheduler(target_optimizer)
    load_checkpoint(
        checkpoint,
        model=target_model,
        optimizer=target_optimizer,
        scheduler=target_scheduler,
        restore_rng=True,
    )

    assert random.getstate() == python_before
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_before[0]
    np.testing.assert_array_equal(restored_numpy[1], numpy_before[1])
    assert restored_numpy[2:] == numpy_before[2:]


def test_failed_checkpoint_save_restores_entry_rng_and_publishes_nothing(
    tmp_path: Path,
) -> None:
    class FailingModel(NumpyModel):
        def state_dict(self):
            random.random()
            np.random.random()
            raise RuntimeError("injected model export failure")

    random.seed(1702)
    np.random.seed(1702)
    model = FailingModel(np.array([0.1, -0.2, 0.3]))
    python_before = copy.deepcopy(random.getstate())
    numpy_before = copy.deepcopy(np.random.get_state())
    checkpoint = tmp_path / "failed-rng-neutral-save"

    with pytest.raises(RuntimeError, match="injected model export failure"):
        save_checkpoint(
            checkpoint,
            model=model,
            trainer_state={},
            identity=identity(step=0, tokens_seen=0),
        )

    assert not checkpoint.exists()
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]


def test_checkpoint_save_restores_torch_rng_and_warn_only_policy(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    original_rng = torch.get_rng_state().clone()
    original_policy = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )

    class EffectfulLinear(torch.nn.Linear):
        def state_dict(self, *args, **kwargs):
            torch.rand(1)
            torch.use_deterministic_algorithms(
                torch.are_deterministic_algorithms_enabled(),
                warn_only=not torch.is_deterministic_algorithms_warn_only_enabled(),
            )
            return super().state_dict(*args, **kwargs)

    try:
        torch.manual_seed(1703)
        torch.use_deterministic_algorithms(True, warn_only=True)
        model = EffectfulLinear(3, 2)
        rng_before = torch.get_rng_state().clone()
        policy_before = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        checkpoint = tmp_path / "torch-rng-neutral-save"
        torch_identity = CheckpointIdentity(
            git_sha="e" * 40,
            model_spec={"kind": "effectful-torch-linear", "in": 3, "out": 2},
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            tokenizer_hash="1" * 64,
            tokenizer_vocab_hash="2" * 64,
            dataset_manifest_hash="3" * 64,
            run_manifest_hash="4" * 64,
            training_config={"steps": 0},
            seed=1703,
            precision="float32",
            step=0,
            tokens_seen=0,
            optimizer={"name": "none"},
            scheduler=None,
        )

        save_checkpoint(
            checkpoint,
            model=model,
            trainer_state={},
            identity=torch_identity,
        )

        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
        assert (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        ) == policy_before
    finally:
        torch.set_rng_state(original_rng)
        torch.use_deterministic_algorithms(
            original_policy[0],
            warn_only=original_policy[1],
        )



@pytest.mark.parametrize(
    "fail",
    [False, True],
    ids=["success", "failure"],
)
def test_checkpoint_prepublish_validator_is_rng_neutral(
    tmp_path: Path,
    fail: bool,
) -> None:
    random.seed(1704)
    np.random.seed(1704)
    model = NumpyModel(np.array([0.1, -0.2, 0.3]))
    python_before = copy.deepcopy(random.getstate())
    numpy_before = copy.deepcopy(np.random.get_state())
    checkpoint = tmp_path / f"prepublish-validator-{fail}"
    validator_calls: list[bool] = []

    def validator() -> None:
        validator_calls.append(True)
        random.random()
        np.random.random()
        if fail:
            raise RuntimeError("injected prepublish rejection")

    if fail:
        with pytest.raises(RuntimeError, match="injected prepublish rejection"):
            save_checkpoint(
                checkpoint,
                model=model,
                trainer_state={},
                identity=identity(step=0, tokens_seen=0),
                prepublish_validator=validator,
            )
        assert not checkpoint.exists()
    else:
        save_checkpoint(
            checkpoint,
            model=model,
            trainer_state={},
            identity=identity(step=0, tokens_seen=0),
            prepublish_validator=validator,
        )
        assert checkpoint.is_dir()
        verify_checkpoint(checkpoint)

    assert validator_calls == [True]
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]



@pytest.mark.parametrize(
    "fail",
    [False, True],
    ids=["success", "failure"],
)
def test_post_rng_prepublish_validator_runs_after_rng_restore(
    tmp_path: Path,
    fail: bool,
) -> None:
    random.seed(1705)
    np.random.seed(1705)
    model = NumpyModel(np.array([0.1, -0.2, 0.3]))
    python_before = copy.deepcopy(random.getstate())
    numpy_before = copy.deepcopy(np.random.get_state())
    checkpoint = tmp_path / f"post-rng-validator-{fail}"
    pre_calls: list[bool] = []
    post_calls: list[bool] = []

    def pre_validator() -> None:
        pre_calls.append(True)
        random.random()
        np.random.random()

    def post_validator() -> None:
        post_calls.append(True)
        assert random.getstate() == python_before
        numpy_live = np.random.get_state()
        assert numpy_live[0] == numpy_before[0]
        np.testing.assert_array_equal(numpy_live[1], numpy_before[1])
        assert numpy_live[2:] == numpy_before[2:]
        if fail:
            raise RuntimeError("injected post-RNG rejection")

    if fail:
        with pytest.raises(RuntimeError, match="injected post-RNG rejection"):
            save_checkpoint(
                checkpoint,
                model=model,
                trainer_state={},
                identity=identity(step=0, tokens_seen=0),
                prepublish_validator=pre_validator,
                post_rng_prepublish_validator=post_validator,
            )
        assert not checkpoint.exists()
    else:
        save_checkpoint(
            checkpoint,
            model=model,
            trainer_state={},
            identity=identity(step=0, tokens_seen=0),
            prepublish_validator=pre_validator,
            post_rng_prepublish_validator=post_validator,
        )
        assert checkpoint.is_dir()
        verify_checkpoint(checkpoint)

    assert pre_calls == [True]
    assert post_calls == [True]
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]


def test_canonical_bound_identity_accepts_normalized_string_metadata(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "canonical-bound-positive"
    manifest = save_checkpoint(
        checkpoint,
        model=NumpyModel(np.array([0.1, -0.2, 0.3])),
        trainer_state={},
        identity=canonical_bound_identity(),
    )

    assert checkpoint.is_dir()
    assert verify_checkpoint(checkpoint)["checkpoint_id"] == manifest["checkpoint_id"]


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        ("run_manifest_sha256", "9" * 64),
        ("training.seed", 18),
        ("training.precision", "bf16"),
        ("training.optimizer", "OtherOptimizer"),
        ("training.scheduler", "OtherScheduler"),
        ("data.dataset_manifest_sha256", "8" * 64),
        ("data.tokenizer_sha256", "7" * 64),
        ("data.tokenizer_vocab_sha256", "6" * 64),
        ("environment.lock_sha256", "5" * 64),
    ],
)
def test_canonical_bound_identity_cross_field_contradiction_is_not_published(
    tmp_path: Path,
    path: str,
    bad_value: object,
) -> None:
    identity_value = canonical_bound_identity()
    training_config = copy.deepcopy(dict(identity_value.training_config))
    parts = path.split(".")
    target = training_config
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = bad_value
    bad_identity = replace(identity_value, training_config=training_config)
    checkpoint = tmp_path / f"contradiction-{path.replace('.', '-')}"

    with pytest.raises(CheckpointIntegrityError, match="disagrees"):
        save_checkpoint(
            checkpoint,
            model=NumpyModel(np.array([0.1, -0.2, 0.3])),
            trainer_state={},
            identity=bad_identity,
        )

    assert not checkpoint.exists()



def test_resealed_canonical_bound_contradiction_is_rejected_on_load(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "resealed-canonical-contradiction"
    save_checkpoint(
        checkpoint,
        model=NumpyModel(np.array([0.1, -0.2, 0.3])),
        trainer_state={},
        identity=canonical_bound_identity(),
    )

    manifest_path = checkpoint / "manifest.json"
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["training_config"]["training"]["seed"] = 18
    manifest["identity"]["training_config_hash"] = hash_json(
        manifest["identity"]["training_config"]
    )
    manifest_path.write_text(
        __import__("json").dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n",
        encoding="utf-8",
    )
    manifest_sha = __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest()
    (checkpoint / "MANIFEST.sha256").write_text(
        f"{manifest_sha}  manifest.json\n",
        encoding="ascii",
    )

    with pytest.raises(
        CheckpointIntegrityError,
        match="training seed disagrees with top-level seed",
    ):
        verify_checkpoint(checkpoint)



@pytest.mark.parametrize(
    "path",
    [
        "run_id",
        "stage",
        "run_kind",
        "init_spec_sha256",
        "data.split_identity",
        "data.tokenizer_version",
        "data.packing_sha256",
        "data.packing_version",
        "environment.lock_sha256",
    ],
)
def test_canonical_bound_identity_requires_complete_provenance(
    tmp_path: Path,
    path: str,
) -> None:
    identity_value = canonical_bound_identity()
    training_config = copy.deepcopy(dict(identity_value.training_config))
    parts = path.split(".")
    target = training_config
    for part in parts[:-1]:
        target = target[part]
    target.pop(parts[-1], None)
    bad_identity = replace(identity_value, training_config=training_config)
    checkpoint = tmp_path / f"missing-{path.replace('.', '-')}"

    with pytest.raises(CheckpointIntegrityError):
        save_checkpoint(
            checkpoint,
            model=NumpyModel(np.array([0.1, -0.2, 0.3])),
            trainer_state={},
            identity=bad_identity,
        )

    assert not checkpoint.exists()


def test_canonical_bound_identity_cannot_drop_environment_lock_as_legacy_optional(
    tmp_path: Path,
) -> None:
    identity_value = canonical_bound_identity()
    training_config = copy.deepcopy(dict(identity_value.training_config))
    training_config["environment"]["lock_sha256"] = None
    bad_identity = replace(
        identity_value,
        training_config=training_config,
        environment_lock_hash=None,
    )
    checkpoint = tmp_path / "missing-canonical-environment-lock"

    with pytest.raises(
        CheckpointIntegrityError,
        match="environment.lock_sha256",
    ):
        save_checkpoint(
            checkpoint,
            model=NumpyModel(np.array([0.1, -0.2, 0.3])),
            trainer_state={},
            identity=bad_identity,
        )

    assert not checkpoint.exists()
