from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from safetensors.numpy import load as load_safetensors_bytes
from safetensors.numpy import save as save_safetensors_bytes

from twelve_six.checkpoint import CheckpointCompatibilityError, CheckpointIdentity, hash_json
from twelve_six.checkpoint.state_tree import pack_state_tree, unpack_state_tree
from twelve_six.checkpoint.trainer_adapter import (
    _preflight_trainer_state,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)


class _StateComponentProbe:
    def __init__(self) -> None:
        self.value = 1.0
        self.history = [1.0, 2.0]
        self.loads = 0

    def state_dict(self) -> dict[str, Any]:
        return {"value": self.value, "history": list(self.history)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if set(state) != {"value", "history"}:
            raise ValueError("state keys mismatch")
        if not isinstance(state["history"], list) or len(state["history"]) != 2:
            raise ValueError("state history geometry mismatch")
        self.loads += 1
        self.value = float(state["value"])
        self.history = list(state["history"])


class _TrainerProbe:
    def __init__(
        self,
        model: Any,
        *,
        populated: bool,
        with_state_components: bool = False,
    ) -> None:
        torch = pytest.importorskip("torch")
        self.config = {"gradient_accumulation_steps": 1, "max_steps": 10}
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        self.scheduler = _StateComponentProbe() if with_state_components else None
        self.scaler = _StateComponentProbe() if with_state_components else None
        self.loads = 0
        if populated:
            loss = model(torch.ones(1, 3)).sum()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self) -> dict[str, Any]:
        return {
            "micro_step": 1,
            "optimizer_step": 1,
            "tokens_seen": 3,
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler": (
                None if self.scheduler is None else copy.deepcopy(self.scheduler.state_dict())
            ),
            "scaler": None if self.scaler is None else copy.deepcopy(self.scaler.state_dict()),
            "config": copy.deepcopy(self.config),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.loads += 1
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if self.scaler is not None:
            self.scaler.load_state_dict(state["scaler"])


class _GenericTrainerProbe:
    """Checkpoint-v1 trainer adapter with opaque trainer-owned optimizer state."""

    def __init__(self) -> None:
        self.config = {"seed": 7, "precision": "fp32"}
        self.loads = 0
        self.velocity = [1.0, 2.0]

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["config"] != self.config:
            raise ValueError("config mismatch")
        velocity = state["optimizer"]["velocity"]
        if not isinstance(velocity, list) or len(velocity) != 2:
            raise ValueError("velocity geometry mismatch")
        self.loads += 1
        self.velocity = list(velocity)


def _identity(parameter_count: int) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "trainer-preflight-probe", "parameters": parameter_count},
        parameter_count=parameter_count,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"probe": True},
        seed=17,
        precision="float32",
        step=1,
        tokens_seen=3,
        optimizer={"name": "SGD", "lr": 0.05, "momentum": 0.9},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _rewrite_state(checkpoint: Path, state: object) -> None:
    packed = pack_state_tree(state)
    (checkpoint / "state.safetensors").write_bytes(save_safetensors_bytes(packed.tensors))
    (checkpoint / "state.json").write_text(
        json.dumps(packed.tree, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("state.safetensors", "state.json"):
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
    checksum = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (checkpoint / "MANIFEST.sha256").write_text(
        f"{checksum}  manifest.json\n", encoding="ascii"
    )


def test_generic_trainer_without_optimizer_attribute_preflights_on_detached_copy() -> None:
    trainer = _GenericTrainerProbe()
    state = {
        "micro_step": 4,
        "optimizer_step": 2,
        "tokens_seen": 64,
        "optimizer": {"velocity": [3.0, 4.0]},
        "scheduler": {"last_epoch": 2},
        "scaler": {},
        "config": copy.deepcopy(trainer.config),
    }

    _preflight_trainer_state(trainer, state)

    assert trainer.loads == 0
    assert trainer.velocity == [1.0, 2.0]


@pytest.mark.parametrize(
    ("identity", "message"),
    [
        ({"step": 2, "tokens_seen": 3}, "optimizer_step disagrees"),
        ({"step": 1, "tokens_seen": 4}, "tokens_seen disagrees"),
    ],
)
def test_trainer_progress_must_match_verified_checkpoint_identity(
    identity: dict[str, int],
    message: str,
) -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 2, bias=False)
    trainer = _TrainerProbe(model, populated=False)
    state = trainer.state_dict()

    with pytest.raises(CheckpointCompatibilityError, match=message):
        _preflight_trainer_state(
            trainer,
            state,
            manifest={"identity": identity},
        )


def test_trainer_owned_optimizer_corruption_fails_before_model_mutation(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    source_model = torch.nn.Linear(3, 2, bias=False)
    source_trainer = _TrainerProbe(source_model, populated=True)
    checkpoint = tmp_path / "trainer-owned-corruption"
    save_trainer_checkpoint(
        checkpoint,
        model=source_model,
        trainer=source_trainer,
        identity=_identity(sum(parameter.numel() for parameter in source_model.parameters())),
    )

    tree = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    tensors = load_safetensors_bytes((checkpoint / "state.safetensors").read_bytes())
    state = unpack_state_tree(tree, tensors)
    trainer_state = state["trainer"]
    first_parameter_state = next(iter(trainer_state["optimizer"]["state"].values()))
    momentum = first_parameter_state["momentum_buffer"]
    first_parameter_state["momentum_buffer"] = momentum.reshape(-1)[:1].clone()
    _rewrite_state(checkpoint, state)

    torch.manual_seed(99)
    target_model = torch.nn.Linear(3, 2, bias=False)
    target_trainer = _TrainerProbe(target_model, populated=False)
    before_model = {
        name: tensor.detach().clone() for name, tensor in target_model.state_dict().items()
    }
    before_optimizer = copy.deepcopy(target_trainer.optimizer.state_dict())

    with pytest.raises(CheckpointCompatibilityError, match="optimizer state tensor shape mismatch"):
        load_trainer_checkpoint(
            checkpoint,
            model=target_model,
            trainer=target_trainer,
            restore_rng=False,
        )

    for name, tensor in target_model.state_dict().items():
        torch.testing.assert_close(tensor, before_model[name], rtol=0, atol=0)
    assert target_trainer.optimizer.state_dict() == before_optimizer
    assert target_trainer.loads == 0


@pytest.mark.parametrize("field", ["scheduler", "scaler"])
def test_trainer_owned_component_corruption_fails_before_model_mutation(
    tmp_path: Path,
    field: str,
) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    source_model = torch.nn.Linear(3, 2, bias=False)
    source_trainer = _TrainerProbe(
        source_model,
        populated=True,
        with_state_components=True,
    )
    checkpoint = tmp_path / f"trainer-{field}-corruption"
    save_trainer_checkpoint(
        checkpoint,
        model=source_model,
        trainer=source_trainer,
        identity=_identity(sum(parameter.numel() for parameter in source_model.parameters())),
    )

    tree = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    tensors = load_safetensors_bytes((checkpoint / "state.safetensors").read_bytes())
    state = unpack_state_tree(tree, tensors)
    state["trainer"][field]["history"] = [1.0]
    _rewrite_state(checkpoint, state)

    torch.manual_seed(109)
    target_model = torch.nn.Linear(3, 2, bias=False)
    target_trainer = _TrainerProbe(
        target_model,
        populated=False,
        with_state_components=True,
    )
    before_model = {
        name: tensor.detach().clone() for name, tensor in target_model.state_dict().items()
    }
    before_optimizer = copy.deepcopy(target_trainer.optimizer.state_dict())

    with pytest.raises(
        CheckpointCompatibilityError,
        match=rf"{field} state\.history list geometry mismatch",
    ):
        load_trainer_checkpoint(
            checkpoint,
            model=target_model,
            trainer=target_trainer,
            restore_rng=False,
        )

    for name, tensor in target_model.state_dict().items():
        torch.testing.assert_close(tensor, before_model[name], rtol=0, atol=0)
    assert target_trainer.optimizer.state_dict() == before_optimizer
    assert target_trainer.loads == 0
    assert target_trainer.scheduler is not None
    assert target_trainer.scaler is not None
    assert target_trainer.scheduler.loads == 0
    assert target_trainer.scaler.loads == 0
