from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from twelve_six.checkpoint.core import (
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointIdentity,
    CheckpointIntegrityError,
)
from twelve_six.checkpoint.v2 import (
    ResumeTopology,
    apply_retention_v2,
    begin_async_checkpoint_v2,
    load_checkpoint_v2,
    plan_retention_v2,
    save_checkpoint_v2,
    verify_checkpoint_v2,
)

HASH = "a" * 64
GIT = "b" * 40


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(64, 16)
        self.proj = nn.Linear(16, 64, bias=False)
        self.proj.weight = self.embedding.weight
        self.ff = nn.Linear(16, 16, bias=False)


def _optimizer(model: nn.Module, *, populated: bool) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    if populated:
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return optimizer


def _identity(model: nn.Module, step: int = 0) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha=GIT,
        model_spec={
            "name": "tiny-v2-test",
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        parameter_count=sum(p.numel() for p in model.parameters()),
        tokenizer_hash=HASH,
        tokenizer_vocab_hash=HASH,
        dataset_manifest_hash=HASH,
        run_manifest_hash=HASH,
        training_config={"test": True},
        seed=7,
        precision="fp32",
        step=step,
        tokens_seen=step * 10,
        optimizer={"name": "AdamW"},
        scheduler=None,
    )


def _fingerprint(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_v2_round_trip_populated_optimizer_and_manifest_last(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = TinyModel()
    optimizer = _optimizer(model, populated=True)
    expected = _fingerprint(model)
    root = tmp_path / "checkpoint"
    manifest = save_checkpoint_v2(
        root,
        model=model,
        optimizer=optimizer,
        identity=_identity(model),
        trainer_state={"micro_step": 4},
    )
    assert manifest["format"] == "12-6-checkpoint-v2"
    assert manifest["status"] == "COMPLETE"
    assert manifest["storage"]["training_state"] == "torch-distributed-checkpoint"
    assert manifest["storage"]["control_tensors"] == "safetensors"
    assert verify_checkpoint_v2(root)["checkpoint_id"] == manifest["checkpoint_id"]

    restored = TinyModel()
    restored_optimizer = _optimizer(restored, populated=False)
    result = load_checkpoint_v2(
        root,
        model=restored,
        optimizer=restored_optimizer,
        expected_identity=_identity(model),
    )
    assert result.trainer_state == {"micro_step": 4}
    assert result.rng_restored is True
    assert len(restored_optimizer.state) == len(list(restored.parameters()))
    for name, value in restored.state_dict().items():
        assert torch.equal(value, expected[name])


def test_v2_rejects_payload_tamper_and_incomplete_directory(tmp_path: Path) -> None:
    model = TinyModel()
    root = tmp_path / "checkpoint"
    save_checkpoint_v2(root, model=model, identity=_identity(model))
    shard = next((root / "dcp").glob("*.distcp"))
    shard.write_bytes(shard.read_bytes() + b"tamper")
    with pytest.raises(CheckpointIntegrityError, match="size mismatch|checksum mismatch"):
        verify_checkpoint_v2(root)

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "dcp").mkdir()
    with pytest.raises(CheckpointIntegrityError, match="incomplete"):
        verify_checkpoint_v2(incomplete)


def test_v2_topology_change_is_explicit_and_rng_is_not_resharded(tmp_path: Path) -> None:
    model = TinyModel()
    root = tmp_path / "checkpoint"
    source = ResumeTopology(world_size=1, parallelism={"data": 1})
    save_checkpoint_v2(root, model=model, identity=_identity(model), topology=source)
    target = ResumeTopology(world_size=1, parallelism={"data": 1, "tensor": 1})
    with pytest.raises(CheckpointCompatibilityError, match="allow_reshard"):
        load_checkpoint_v2(root, model=TinyModel(), topology=target)
    with pytest.raises(CheckpointCompatibilityError, match="restore_rng=False"):
        load_checkpoint_v2(
            root,
            model=TinyModel(),
            topology=target,
            allow_reshard=True,
            restore_rng=True,
        )
    result = load_checkpoint_v2(
        root,
        model=TinyModel(),
        topology=target,
        allow_reshard=True,
        restore_rng=False,
    )
    assert result.resharded is True
    assert result.rng_restored is False


def test_v2_async_save_allows_only_one_inflight_and_manifest_commits_on_wait(
    tmp_path: Path,
) -> None:
    model = TinyModel()
    first = tmp_path / "first"
    handle = begin_async_checkpoint_v2(first, model=model, identity=_identity(model))
    assert not (first / "manifest.json").exists()
    with pytest.raises(CheckpointError, match="only one"):
        begin_async_checkpoint_v2(tmp_path / "second", model=model, identity=_identity(model))
    assert handle.wait()["status"] == "COMPLETE"
    assert verify_checkpoint_v2(first)["status"] == "COMPLETE"


def test_v2_retention_only_deletes_verified_complete_checkpoints(tmp_path: Path) -> None:
    for step in range(4):
        model = TinyModel()
        save_checkpoint_v2(
            tmp_path / f"step-{step}",
            model=model,
            identity=_identity(model, step=step),
        )
    incomplete = tmp_path / "step-incomplete"
    incomplete.mkdir()
    plan = plan_retention_v2(tmp_path, keep_last=1, keep_every_n_steps=2)
    assert {path.name for path in plan.keep} == {"step-0", "step-2", "step-3"}
    assert {path.name for path in plan.delete} == {"step-1"}
    apply_retention_v2(plan)
    assert not (tmp_path / "step-1").exists()
    assert incomplete.exists()


def test_v2_checksum_consistent_semantic_relabel_still_fails_expected_identity(
    tmp_path: Path,
) -> None:
    model = TinyModel()
    root = tmp_path / "checkpoint"
    save_checkpoint_v2(root, model=model, identity=_identity(model))
    manifest = verify_checkpoint_v2(root)
    assert manifest["semantic_identity"]["parameter_count"] == sum(
        p.numel() for p in model.parameters()
    )
    wrong = _identity(model, step=1)
    with pytest.raises(CheckpointCompatibilityError, match="semantic identity mismatch"):
        load_checkpoint_v2(root, model=TinyModel(), expected_identity=wrong)
