from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load as load_safetensors_bytes
from safetensors.numpy import save as save_safetensors_bytes

from twelve_six.checkpoint.core import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    CheckpointIntegrityError,
    hash_json,
    load_checkpoint,
    load_verified_checkpoint,
    prepare_checkpoint_load,
    save_checkpoint,
    verify_checkpoint,
)
from twelve_six.checkpoint.state_tree import pack_state_tree, unpack_state_tree


class NumpyModel:
    def __init__(self, values: list[float]) -> None:
        self.weights = np.asarray(values, dtype=np.float64).copy()
        self.loads = 0

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        assert not strict or set(state) == {"weights"}
        self.loads += 1
        self.weights = state["weights"].copy()


def _identity(
    *,
    parameter_count: int = 3,
    precision: str = "float64",
    optimizer: dict[str, object] | None = None,
    step: int = 1,
    tokens_seen: int = 3,
) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "corruption-regression", "parameters": parameter_count},
        parameter_count=parameter_count,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 2},
        seed=7,
        precision=precision,
        step=step,
        tokens_seen=tokens_seen,
        optimizer=optimizer or {"name": "none"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _rewrite_manifest(checkpoint: Path, payload_names: tuple[str, ...] = ()) -> None:
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in payload_names:
        data = (checkpoint / name).read_bytes()
        manifest["files"][name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    manifest["checkpoint_id"] = hash_json(
        {"identity": manifest["identity"], "files": manifest["files"]}
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (checkpoint / "MANIFEST.sha256").write_text(
        f"{manifest_sha}  manifest.json\n", encoding="ascii"
    )


def test_dtype_corruption_fails_before_numpy_model_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "dtype"
    save_checkpoint(checkpoint, model=NumpyModel([1, 2, 3]), identity=_identity())

    arrays = load_safetensors_bytes((checkpoint / "weights.safetensors").read_bytes())
    arrays["weights"] = arrays["weights"].astype(np.float32)
    (checkpoint / "weights.safetensors").write_bytes(save_safetensors_bytes(arrays))
    _rewrite_manifest(checkpoint, ("weights.safetensors",))

    target = NumpyModel([9, 9, 9])
    before = target.weights.copy()
    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0


class SemanticScheduler:
    def __init__(self, value: float) -> None:
        self.value = value
        self.loads = 0

    def state_dict(self) -> dict[str, float]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.loads += 1
        self.value = state["value"]
        if self.value < 0:
            raise ValueError("negative scheduler value")


@pytest.mark.parametrize("verified_api", [False, True])
@pytest.mark.parametrize("value", [-1.0, 2.0])
def test_direct_scheduler_semantics_are_checked_before_live_mutation(
    tmp_path: Path, verified_api: bool, value: float,
) -> None:
    checkpoint = tmp_path / "semantic-scheduler"
    save_checkpoint(
        checkpoint,
        model=NumpyModel([1, 2, 3]),
        scheduler=SemanticScheduler(value),
        identity=replace(_identity(), scheduler={"name": "semantic-scheduler"}),
    )
    # The checksum-valid checkpoint has the same keys and types in both cases.
    verified = prepare_checkpoint_load(checkpoint)
    target = NumpyModel([9, 9, 9])
    scheduler = SemanticScheduler(1.0)
    loader = load_verified_checkpoint if verified_api else load_checkpoint
    source = verified if verified_api else checkpoint
    if value < 0:
        with pytest.raises(CheckpointCompatibilityError, match="semantic"):
            loader(source, model=target, scheduler=scheduler, restore_rng=False)
        np.testing.assert_array_equal(target.weights, [9, 9, 9])
        assert target.loads == scheduler.loads == 0
        assert scheduler.value == 1.0
    else:
        loader(source, model=target, scheduler=scheduler, restore_rng=False)
        np.testing.assert_array_equal(target.weights, [1, 2, 3])
        assert target.loads == scheduler.loads == 1
        assert scheduler.value == value


@pytest.mark.parametrize("adapter", [False, True])
def test_nested_scheduler_probe_is_isolated_without_cloning_optimizer(adapter: bool) -> None:
    torch = pytest.importorskip("torch")
    from twelve_six.checkpoint import core, trainer_adapter

    class NoOptimizerCopy(torch.optim.SGD):
        def __deepcopy__(self, memo):
            raise AssertionError("model-scale optimizer must not be cloned")

    parameter = torch.nn.Parameter(torch.ones(3))
    optimizer = NoOptimizerCopy([parameter], lr=0.1, momentum=0.9)
    first = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    second = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[first, second], milestones=[5],
    )
    before = copy.deepcopy(scheduler.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    incoming = copy.deepcopy(before)
    incoming["_schedulers"][0]["last_epoch"] += 7
    incoming["_schedulers"][1]["last_epoch"] += 9
    module = trainer_adapter if adapter else core
    module._preflight_stateful_component(scheduler, incoming, label="scheduler")
    assert scheduler.state_dict() == before
    assert optimizer.state_dict() == optimizer_before
    assert scheduler._schedulers == [first, second]
    assert optimizer.param_groups[0]["params"][0] is parameter


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (("step", -1), ("tokens_seen", -1)),
)
def test_negative_manifest_counters_fail_even_after_rebinding(
    tmp_path: Path, field: str, bad_value: int
) -> None:
    checkpoint = tmp_path / field
    save_checkpoint(checkpoint, model=NumpyModel([1, 2, 3]), identity=_identity())

    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"][field] = bad_value
    manifest["checkpoint_id"] = hash_json(
        {"identity": manifest["identity"], "files": manifest["files"]}
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (checkpoint / "MANIFEST.sha256").write_text(
        f"{manifest_sha}  manifest.json\n", encoding="ascii"
    )

    with pytest.raises(CheckpointIntegrityError, match="non-negative integers"):
        verify_checkpoint(checkpoint)


def _torch_checkpoint(tmp_path: Path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(11)
    source = torch.nn.Linear(3, 2, bias=False)
    optimizer = torch.optim.SGD(source.parameters(), lr=0.05, momentum=0.9)
    loss = source(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()

    checkpoint = tmp_path / "torch-sgd"
    save_checkpoint(
        checkpoint,
        model=source,
        optimizer=optimizer,
        identity=_identity(
            parameter_count=6,
            precision="float32",
            optimizer={"name": "SGD", "lr": 0.05, "momentum": 0.9},
        ),
    )
    return torch, checkpoint, source


def test_valid_sgd_momentum_checkpoint_still_loads(tmp_path: Path) -> None:
    torch, checkpoint, source = _torch_checkpoint(tmp_path)
    target = torch.nn.Linear(3, 2, bias=False)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.05, momentum=0.9)

    load_checkpoint(
        checkpoint,
        model=target,
        optimizer=target_optimizer,
        restore_rng=False,
    )

    for source_value, target_value in zip(
        source.state_dict().values(), target.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(target_value, source_value)
    assert target_optimizer.state


def test_wrong_shaped_momentum_fails_before_model_mutation(tmp_path: Path) -> None:
    torch, checkpoint, _source = _torch_checkpoint(tmp_path)

    tree = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    arrays = load_safetensors_bytes((checkpoint / "state.safetensors").read_bytes())
    combined = unpack_state_tree(tree, arrays)
    optimizer_state = combined["optimizer"]["state"]
    parameter_state = next(iter(optimizer_state.values()))
    momentum = parameter_state["momentum_buffer"]
    parameter_state["momentum_buffer"] = momentum.reshape(-1)[:1].clone()
    packed = pack_state_tree(combined)
    (checkpoint / "state.safetensors").write_bytes(save_safetensors_bytes(packed.tensors))
    (checkpoint / "state.json").write_text(
        json.dumps(packed.tree, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rewrite_manifest(checkpoint, ("state.safetensors", "state.json"))

    target = torch.nn.Linear(3, 2, bias=False)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.05, momentum=0.9)
    before = {name: value.detach().clone() for name, value in target.state_dict().items()}

    with pytest.raises(CheckpointCompatibilityError, match="optimizer state tensor shape mismatch"):
        load_checkpoint(
            checkpoint,
            model=target,
            optimizer=target_optimizer,
            restore_rng=False,
        )

    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name])
    assert not target_optimizer.state


class SourceScheduler:
    def state_dict(self) -> dict[str, int]:
        return {"schema_version": 2}


class TargetScheduler:
    def state_dict(self) -> dict[str, int]:
        return {"schema_version": 1, "last_epoch": 0}

    def load_state_dict(self, state: dict[str, int]) -> None:
        if state != {"schema_version": 1, "last_epoch": 0}:
            raise ValueError("scheduler schema mismatch")


def test_direct_scheduler_incompatibility_fails_before_model_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "scheduler-preflight"
    identity = _identity()
    identity = CheckpointIdentity(
        git_sha=identity.git_sha,
        model_spec=identity.model_spec,
        parameter_count=identity.parameter_count,
        tokenizer_hash=identity.tokenizer_hash,
        tokenizer_vocab_hash=identity.tokenizer_vocab_hash,
        dataset_manifest_hash=identity.dataset_manifest_hash,
        run_manifest_hash=identity.run_manifest_hash,
        training_config=identity.training_config,
        seed=identity.seed,
        precision=identity.precision,
        step=identity.step,
        tokens_seen=identity.tokens_seen,
        optimizer=identity.optimizer,
        scheduler={"name": "versioned-scheduler"},
        environment_lock_hash=identity.environment_lock_hash,
    )
    save_checkpoint(
        checkpoint,
        model=NumpyModel([1, 2, 3]),
        scheduler=SourceScheduler(),
        identity=identity,
    )

    target = NumpyModel([9, 9, 9])
    scheduler = TargetScheduler()
    before = target.weights.copy()

    with pytest.raises(CheckpointCompatibilityError, match="scheduler"):
        load_checkpoint(
            checkpoint,
            model=target,
            scheduler=scheduler,
            restore_rng=False,
        )

    np.testing.assert_array_equal(target.weights, before)
    assert target.loads == 0
