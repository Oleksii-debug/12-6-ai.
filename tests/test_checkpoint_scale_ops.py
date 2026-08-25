from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import twelve_six.distributed.checkpoint_scale_ops as ops
from twelve_six.distributed.checkpoint_scale_ops import (
    ScaleRetentionPolicy,
    V1ScaleMigrationProvenance,
    apply_scale_checkpoint_retention,
    assert_v1_scale_migration_compatible,
    async_checkpoint_gate,
    capture_trainer_resume_control,
    checkpoint_scale_policy,
    plan_scale_checkpoint_retention,
    restore_trainer_resume_control,
)
from twelve_six.distributed.dcp_checkpoint import ScaleCheckpointIdentity

HASH_A = "a" * 64
HASH_B = "b" * 64
GIT = "c" * 40


class FakeOptimizer:
    def __init__(self) -> None:
        self.value = torch.tensor([3.0])

    def state_dict(self) -> dict[str, Any]:
        return {"state": {0: {"moment": self.value.clone()}}, "param_groups": []}


@dataclass
class FakeTrainerState:
    micro_step: int
    optimizer_step: int
    tokens_seen: int
    scheduler: dict[str, Any]
    scaler: dict[str, Any] | None
    config: dict[str, Any]


class FakeTrainer:
    def __init__(self) -> None:
        self.optimizer = FakeOptimizer()
        self.loaded: dict[str, Any] | None = None

    def assert_checkpoint_safe(self) -> None:
        return None

    def state_dict(self) -> FakeTrainerState:
        return FakeTrainerState(
            micro_step=8,
            optimizer_step=4,
            tokens_seen=1024,
            scheduler={"last_epoch": 4, "base_lrs": [0.001]},
            scaler=None,
            config={"gradient_accumulation_steps": 2, "precision": "fp32"},
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.loaded = state


def _scale_identity() -> ScaleCheckpointIdentity:
    return ScaleCheckpointIdentity(
        git_sha=GIT,
        model_spec_sha256=HASH_A,
        init_spec_sha256=HASH_B,
        tokenizer_config_sha256=HASH_A,
        tokenizer_vocab_sha256=HASH_B,
        data_manifest_sha256=HASH_A,
        packing_sha256=HASH_B,
        training_config_sha256=HASH_A,
        environment_lock_sha256=HASH_B,
        seed=7,
        step=4,
        tokens_seen=1024,
    )


def _v1_manifest() -> dict[str, Any]:
    return {
        "format": "12-6-checkpoint",
        "format_version": 1,
        "checkpoint_id": HASH_A,
        "identity": {
            "git_sha": GIT,
            "model_spec_hash": HASH_A,
            "tokenizer_hash": HASH_A,
            "tokenizer_vocab_hash": HASH_B,
            "dataset_manifest_hash": HASH_A,
            "training_config_hash": HASH_A,
            "environment_lock_hash": HASH_B,
            "seed": 7,
            "step": 4,
            "tokens_seen": 1024,
            "run_manifest_hash": HASH_A,
        },
    }


def test_checkpoint_policy_changes_at_scale_without_replacing_v1_globally() -> None:
    assert checkpoint_scale_policy(10_000).training_resume_format == "CHECKPOINT_V1"
    assert checkpoint_scale_policy(1_000_000).training_resume_format.startswith(
        "DUAL_QUALIFICATION"
    )
    assert checkpoint_scale_policy(10_000_000).training_resume_format.startswith("D18_DCP")
    distributed = checkpoint_scale_policy(1_000_000, distributed=True)
    assert distributed.training_resume_format == "D18_DCP_REQUIRED"
    assert "D18" in distributed.d18_status


def test_async_gate_refuses_fake_background_wrapper() -> None:
    gate = async_checkpoint_gate()
    assert gate.supported is False
    assert gate.max_in_flight == 1
    assert any("async_save" in item for item in gate.required_evidence)
    assert any("peak host" in item for item in gate.required_evidence)


def test_trainer_resume_control_restores_counters_and_rng() -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    trainer = FakeTrainer()
    control = capture_trainer_resume_control(trainer)
    expected = (random.random(), float(np.random.random()), float(torch.rand(1).item()))

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restored, rng_restored = restore_trainer_resume_control(trainer, control)
    actual = (random.random(), float(np.random.random()), float(torch.rand(1).item()))

    assert restored is True
    assert rng_restored is True
    assert actual == expected
    assert trainer.loaded is not None
    assert trainer.loaded["micro_step"] == 8
    assert trainer.loaded["optimizer_step"] == 4
    assert trainer.loaded["tokens_seen"] == 1024
    assert trainer.loaded["scheduler"] == {"last_epoch": 4, "base_lrs": [0.001]}
    assert torch.equal(trainer.loaded["optimizer"]["state"][0]["moment"], torch.tensor([3.0]))


def test_v1_migration_guard_preserves_every_overlapping_identity() -> None:
    provenance = V1ScaleMigrationProvenance(
        source_run_manifest_sha256=HASH_A,
        init_spec_sha256=HASH_B,
        packing_sha256=HASH_B,
    )
    assert_v1_scale_migration_compatible(
        _v1_manifest(),
        scale_identity=_scale_identity(),
        provenance=provenance,
    )
    tampered = _v1_manifest()
    tampered["identity"]["model_spec_hash"] = HASH_B
    with pytest.raises(ValueError, match="migration identity mismatch"):
        assert_v1_scale_migration_compatible(
            tampered,
            scale_identity=_scale_identity(),
            provenance=provenance,
        )


def test_retention_keeps_verified_generations_and_never_auto_deletes_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifests: dict[Path, dict[str, Any]] = {}
    for step in range(4):
        path = tmp_path / f"step-{step}"
        path.mkdir()
        (path / "COMMITTED").write_text(HASH_A + "\n", encoding="ascii")
        manifests[path] = {
            "identity": {"step": step, "tokens_seen": step * 100},
            "aggregate_checkpoint_sha256": HASH_A,
        }
    staging = tmp_path / ".step-5.dcp-staging-deadbeef"
    staging.mkdir()

    def fake_verify(path: str | Path) -> dict[str, Any]:
        return manifests[Path(path)]

    monkeypatch.setattr(ops, "verify_scale_checkpoint", fake_verify)
    plan = plan_scale_checkpoint_retention(
        tmp_path,
        policy=ScaleRetentionPolicy(keep_last=1, keep_every_n_steps=2),
    )
    assert {item.path.name for item in plan.keep} == {"step-0", "step-2", "step-3"}
    assert {item.path.name for item in plan.delete} == {"step-1"}
    assert plan.uncommitted_staging == (staging,)

    apply_scale_checkpoint_retention(plan)
    assert not (tmp_path / "step-1").exists()
    assert staging.exists()
