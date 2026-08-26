from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors.numpy import load as load_safetensors_bytes
from safetensors.numpy import save as save_safetensors_bytes

import twelve_six.checkpoint.core as checkpoint_core
from twelve_six.checkpoint.core import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    CheckpointIntegrityError,
    hash_json,
    load_checkpoint,
    save_checkpoint,
    verify_checkpoint,
)
from twelve_six.checkpoint.state_tree import pack_state_tree, unpack_state_tree


class TinyModel(torch.nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))


def identity(*, step: int = 1, tokens_seen: int = 3) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "d05-hardening-test", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 2},
        seed=7,
        precision="float32",
        step=step,
        tokens_seen=tokens_seen,
        optimizer={"name": "SGD", "momentum": 0.9},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _rewrite_manifest(checkpoint: Path) -> None:
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("weights.safetensors", "state.safetensors", "state.json"):
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


def _seeded_sgd(model: TinyModel) -> torch.optim.SGD:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer


def test_model_dtype_mismatch_rejects_before_model_mutation(tmp_path: Path) -> None:
    source = TinyModel([1.0, 2.0, 3.0])
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=source, identity=identity())

    arrays = load_safetensors_bytes((checkpoint / "weights.safetensors").read_bytes())
    arrays["weight"] = arrays["weight"].astype("float64")
    (checkpoint / "weights.safetensors").write_bytes(save_safetensors_bytes(arrays))
    _rewrite_manifest(checkpoint)

    target = TinyModel([9.0, 9.0, 9.0])
    before = target.weight.detach().clone()
    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        load_checkpoint(checkpoint, model=target, restore_rng=False)
    torch.testing.assert_close(target.weight.detach(), before)


def test_wrong_shaped_momentum_rejects_before_model_mutation(tmp_path: Path) -> None:
    source = TinyModel([1.0, 2.0, 3.0])
    source_optimizer = _seeded_sgd(source)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model=source,
        optimizer=source_optimizer,
        identity=identity(),
    )

    tree = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    arrays = load_safetensors_bytes((checkpoint / "state.safetensors").read_bytes())
    state = copy.deepcopy(unpack_state_tree(tree, arrays))
    parameter_id = next(iter(state["optimizer"]["state"]))
    state["optimizer"]["state"][parameter_id]["momentum_buffer"] = torch.zeros(
        2, dtype=torch.float32
    )
    packed = pack_state_tree(state)
    (checkpoint / "state.safetensors").write_bytes(save_safetensors_bytes(packed.tensors))
    (checkpoint / "state.json").write_text(
        json.dumps(packed.tree, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rewrite_manifest(checkpoint)

    target = TinyModel([9.0, 9.0, 9.0])
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1, momentum=0.9)
    before = target.weight.detach().clone()
    with pytest.raises(CheckpointCompatibilityError, match="optimizer tensor shape mismatch"):
        load_checkpoint(
            checkpoint,
            model=target,
            optimizer=target_optimizer,
            restore_rng=False,
        )
    torch.testing.assert_close(target.weight.detach(), before)
    assert target_optimizer.state == {}


@pytest.mark.parametrize(("field", "value"), [("step", -1), ("tokens_seen", -3)])
def test_negative_manifest_counters_fail_even_when_manifest_is_rebound(
    tmp_path: Path, field: str, value: int
) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=TinyModel([1.0, 2.0, 3.0]), identity=identity())
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"][field] = value
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

    with pytest.raises(CheckpointIntegrityError, match="non-negative"):
        verify_checkpoint(checkpoint)


def test_expected_resume_counters_bind_before_model_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=TinyModel([1.0, 2.0, 3.0]), identity=identity())
    target = TinyModel([9.0, 9.0, 9.0])
    before = target.weight.detach().clone()

    with pytest.raises(CheckpointCompatibilityError, match="counter identity mismatch"):
        checkpoint_core.load_checkpoint(
            checkpoint,
            model=target,
            restore_rng=False,
            expected_step=2,
            expected_tokens_seen=6,
        )
    torch.testing.assert_close(target.weight.detach(), before)
