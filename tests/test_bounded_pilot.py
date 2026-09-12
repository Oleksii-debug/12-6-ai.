from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from typing import NoReturn

import pytest
import torch
from torch import nn

from twelve_six.data.deterministic_exposure_order import (
    build_deterministic_exposure_plan,
    ordered_next_exposure_identity,
)
from twelve_six.data.identity_safe_exposure_guard import IdentitySafeExposureReplayGuard
from twelve_six.data.unique_loss_ledger_v2 import build_ledger
from twelve_six.portable_run_binding import PortableRunBinding, canonical_sha256
from twelve_six.training.bounded_pilot import (
    BoundedPilotAuthorizationError,
    BoundedPilotRecoveryRequiredError,
    BoundedPilotStepRunner,
)
from twelve_six.training.config import TrainerConfig
from twelve_six.training.single_gpu import SingleDeviceStepRunner
from twelve_six.training.trainer import Trainer


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _identity(value: dict, field: str) -> str:
    payload = deepcopy(value)
    payload.pop(field, None)
    return hashlib.sha256(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def _ledger(loss_positions: int) -> dict:
    token_count = loss_positions + 1
    materialization = {
        "schema_version": "12-6.postpack-loss-materialization.v2",
        "stage_bindings": {
            "normalization": _sha("normalization"),
            "evaluation_reservations": _sha("reservations"),
            "dedup": _sha("dedup"),
            "split": _sha("split"),
            "packing": _sha("packing-stage"),
        },
        "tokenizer": {
            "name": "s0-byte-v1",
            "identity_sha256": _sha("tokenizer"),
            "source_bytes_are_loss_positions": False,
        },
        "documents": [
            {
                "document_id": "doc",
                "language": "uk",
                "modality": "text",
                "family_id": "family.uk",
                "normalized_payload_sha256": _sha("payload"),
                "source_bytes": token_count,
                "token_count": token_count,
                "split": "train",
                "dedup_cluster_id": "cluster-doc",
                "retained_after_dedup": True,
                "evaluation_reserved": False,
                "reserved_target_ranges": [],
                "eligible_target_ranges": [[1, token_count]],
            }
        ],
        "packing": {
            "identity_sha256": _sha("packing"),
            "complete_one_pass": True,
            "packs": [
                {
                    "pack_id": "p0",
                    "token_count": token_count,
                    "loss_spans": [
                        {
                            "document_id": "doc",
                            "target_start": 1,
                            "target_end": token_count,
                            "pack_target_start": 1,
                        }
                    ],
                }
            ],
        },
    }
    materialization["materialization_identity_sha256"] = _identity(
        materialization,
        "materialization_identity_sha256",
    )
    return build_ledger(materialization)


def _guard_and_plan(
    *,
    batch_count: int = 1,
) -> tuple[IdentitySafeExposureReplayGuard, dict]:
    loss_positions = 2 * batch_count
    ledger = _ledger(loss_positions)
    guard = IdentitySafeExposureReplayGuard(
        ledger,
        expected_ledger_identity_sha256=ledger["ledger_identity_sha256"],
        authorized_budget=loss_positions,
        trainer_state_binding={
            "checkpoint_generation": "g000",
            "checkpoint_manifest_sha256": _sha("checkpoint"),
            "optimizer_step": 0,
            "trainer_nonignored_target_count": 0,
        },
    )
    segment_id = ledger["segments"][0]["segment_identity_sha256"]
    batches = []
    for batch_index in range(batch_count):
        offset_start = 2 * batch_index
        batches.append(
            {
                "global_batch_index": batch_index,
                "shard_index": 0,
                "worker_id": 0,
                "claims": [
                    {
                        "segment_identity_sha256": segment_id,
                        "offset_start": offset_start,
                        "offset_end": offset_start + 2,
                    }
                ],
                "actual_nonignored_targets": 2,
            }
        )
    plan = build_deterministic_exposure_plan(
        batches,
        num_workers=1,
        batches_per_shard=batch_count,
        shard_count=1,
    )
    return guard, plan


class _Identity:
    def __init__(self, label: str, parameters: int | None = None) -> None:
        self.label = label
        self.parameters = parameters

    def identity_sha256(self) -> str:
        return _sha(self.label)

    def parameter_count(self) -> int:
        assert self.parameters is not None
        return self.parameters


class _TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.projection = nn.Linear(4, 8)
        count = sum(parameter.numel() for parameter in self.parameters())
        self.spec = _Identity("model", count)
        self.init_spec = _Identity("init")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(input_ids))


def _trainer(
    *,
    learning_rate: float = 1e-3,
    max_steps: int = 1,
) -> Trainer:
    return Trainer(
        _TinyLM(),
        TrainerConfig(
            learning_rate=learning_rate,
            max_steps=max_steps,
            gradient_accumulation_steps=1,
            gradient_clip_norm=1.0,
            precision="fp32",
            seed=1333,
            deterministic_algorithms=True,
        ),
        device="cpu",
    )


def _overlay_projection(trainer: Trainer) -> dict:
    return {
        "optimizer": trainer.optimizer.__class__.__name__,
        "scheduler": trainer.config.scheduler,
        "precision": trainer.config.precision,
    }


def _binding(
    guard: IdentitySafeExposureReplayGuard,
    trainer: Trainer,
    *,
    overlay_projection: dict | None = None,
) -> PortableRunBinding:
    budget = guard.authorized_budget
    packet = {
        "identities": {
            "canonical_base": "random_init",
            "modelspec_sha256": trainer.model.spec.identity_sha256(),
            "initspec_sha256": trainer.model.init_spec.identity_sha256(),
            "unique_loss_ledger_sha256": guard.ledger_identity_sha256,
        },
        "recipe": {
            "training_config_sha256": _sha("session"),
            "target_unique_loss_positions": budget,
            "maximum_total_exposures": budget,
            "available_unique_loss_positions": budget,
            "max_exposures_per_unique_position": 1,
            "seed": trainer.config.seed,
            "optimizer_scheduler_precision": (
                _overlay_projection(trainer)
                if overlay_projection is None
                else overlay_projection
            ),
        },
        "resource": {
            "resource_class": "LOCAL_FREE",
            "maximum_cost_usd": 0,
            "materially_paid": False,
        },
        "truth_boundary": {"final_test_payload_accessed": False},
    }
    packet_sha256 = canonical_sha256(packet)
    return PortableRunBinding(
        binding_ready=True,
        mode="FRESH_START",
        readiness_ready=True,
        overlay_contract_valid=True,
        packet_contract_valid=True,
        blockers=(),
        readiness_sha256=_sha("readiness"),
        overlay_sha256=_sha("overlay"),
        packet_sha256=packet_sha256,
        packet=packet,
    )


def _batch() -> dict[str, torch.Tensor]:
    return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}


def _gate(
    runner: SingleDeviceStepRunner,
    binding: PortableRunBinding,
    guard: IdentitySafeExposureReplayGuard,
    plan: dict,
) -> BoundedPilotStepRunner:
    assert binding.packet_sha256 is not None
    return BoundedPilotStepRunner(
        runner,
        binding=binding,
        expected_packet_sha256=binding.packet_sha256,
        replay_guard=guard,
        exposure_plan=plan,
        expected_plan_identity_sha256=plan["plan_identity_sha256"],
    )


class _PostStepFailureRunner(SingleDeviceStepRunner):
    def train_microbatch(self, batch: dict[str, torch.Tensor]) -> NoReturn:
        super().train_microbatch(batch)
        raise RuntimeError("synthetic post-step synchronization failure")


def test_exact_d04_handoff_authorizes_one_optimizer_step_and_rebinds_state() -> None:
    guard, plan = _guard_and_plan()
    trainer = _trainer()
    binding = _binding(guard, trainer)
    gate = _gate(SingleDeviceStepRunner(trainer), binding, guard, plan)
    expected = ordered_next_exposure_identity(
        guard,
        plan,
        batch_index=0,
        expected_plan_identity_sha256=plan["plan_identity_sha256"],
    )
    metrics, receipt = gate.train_authorized_microbatch(
        _batch(),
        batch_index=0,
        expected_next_exposure_identity_sha256=expected,
    )
    gate.close()
    assert metrics.trainer.optimizer_stepped is True
    assert trainer.optimizer_step == 1
    assert trainer.tokens_seen == 2
    assert guard.consumed_loss_positions == 2
    assert guard.trainer_state_binding["optimizer_step"] == 1
    assert guard.trainer_state_binding["trainer_nonignored_target_count"] == 2
    assert receipt.optimizer_step_before == 0
    assert receipt.optimizer_step_after == 1
    assert receipt.exposure_identity_sha256 == expected
    assert receipt.packet_sha256 == binding.packet_sha256
    assert receipt.modelspec_sha256 == trainer.model.spec.identity_sha256()


def test_two_steps_consume_distinct_exposures_with_live_rebinding() -> None:
    guard, plan = _guard_and_plan(batch_count=2)
    trainer = _trainer(max_steps=2)
    binding = _binding(guard, trainer)
    gate = _gate(SingleDeviceStepRunner(trainer), binding, guard, plan)
    observed = []
    for batch_index in range(2):
        expected = ordered_next_exposure_identity(
            guard,
            plan,
            batch_index=batch_index,
            expected_plan_identity_sha256=plan["plan_identity_sha256"],
        )
        _, receipt = gate.train_authorized_microbatch(
            _batch(),
            batch_index=batch_index,
            expected_next_exposure_identity_sha256=expected,
        )
        observed.append(receipt.exposure_identity_sha256)
    gate.close()
    assert observed[0] != observed[1]
    assert trainer.optimizer_step == 2
    assert trainer.tokens_seen == 4
    assert guard.consumed_loss_positions == 4
    assert guard.trainer_state_binding["optimizer_step"] == 2
    assert guard.trainer_state_binding["trainer_nonignored_target_count"] == 4


def test_wrong_exposure_identity_blocks_before_trainer_mutation() -> None:
    guard, plan = _guard_and_plan()
    trainer = _trainer()
    binding = _binding(guard, trainer)
    gate = _gate(SingleDeviceStepRunner(trainer), binding, guard, plan)
    before = guard.state_dict()
    with pytest.raises(BoundedPilotAuthorizationError, match="external handoff"):
        gate.train_authorized_microbatch(
            _batch(),
            batch_index=0,
            expected_next_exposure_identity_sha256=_sha("wrong"),
        )
    gate.close()
    assert gate.poisoned is False
    assert trainer.optimizer_step == 0
    assert trainer.tokens_seen == 0
    assert guard.state_dict() == before


def test_self_resealed_packet_is_blocked_by_external_packet_root() -> None:
    guard, plan = _guard_and_plan()
    trainer = _trainer()
    binding = _binding(guard, trainer)
    assert binding.packet is not None
    assert binding.packet_sha256 is not None
    tampered_packet = deepcopy(binding.packet)
    tampered_packet["recipe"]["seed"] = trainer.config.seed + 1
    tampered = replace(
        binding,
        packet=tampered_packet,
        packet_sha256=canonical_sha256(tampered_packet),
    )
    with pytest.raises(BoundedPilotAuthorizationError, match="external authority"):
        BoundedPilotStepRunner(
            SingleDeviceStepRunner(trainer),
            binding=tampered,
            expected_packet_sha256=binding.packet_sha256,
            replay_guard=guard,
            exposure_plan=plan,
            expected_plan_identity_sha256=plan["plan_identity_sha256"],
        )
    assert trainer.optimizer_step == 0


def test_live_optimizer_drift_is_blocked_before_trainer_mutation() -> None:
    guard, plan = _guard_and_plan()
    trainer = _trainer()
    binding = _binding(guard, trainer)
    gate = _gate(SingleDeviceStepRunner(trainer), binding, guard, plan)
    trainer.optimizer.param_groups[0]["lr"] = 2e-3
    expected = ordered_next_exposure_identity(
        guard,
        plan,
        batch_index=0,
        expected_plan_identity_sha256=plan["plan_identity_sha256"],
    )
    with pytest.raises(BoundedPilotAuthorizationError, match="live optimizer hyperparameters"):
        gate.train_authorized_microbatch(
            _batch(),
            batch_index=0,
            expected_next_exposure_identity_sha256=expected,
        )
    gate.close()
    assert gate.poisoned is False
    assert trainer.optimizer_step == 0
    assert trainer.tokens_seen == 0
    assert guard.consumed_loss_positions == 0


def test_post_authorization_failure_never_rewinds_exposure_and_poison_gate() -> None:
    guard, plan = _guard_and_plan()
    trainer = _trainer()
    binding = _binding(guard, trainer)
    gate = _gate(_PostStepFailureRunner(trainer), binding, guard, plan)
    expected = ordered_next_exposure_identity(
        guard,
        plan,
        batch_index=0,
        expected_plan_identity_sha256=plan["plan_identity_sha256"],
    )
    with pytest.raises(BoundedPilotRecoveryRequiredError, match="fresh verified recovery"):
        gate.train_authorized_microbatch(
            _batch(),
            batch_index=0,
            expected_next_exposure_identity_sha256=expected,
        )
    assert gate.poisoned is True
    assert trainer.optimizer_step == 1
    assert guard.consumed_loss_positions == 2
    with pytest.raises(BoundedPilotRecoveryRequiredError, match="fresh verified recovery"):
        gate.train_authorized_microbatch(
            _batch(),
            batch_index=0,
            expected_next_exposure_identity_sha256=expected,
        )
    gate.close()
    assert trainer.optimizer_step == 1
    assert guard.consumed_loss_positions == 2


def test_actual_model_identity_substitution_is_blocked_before_step_1() -> None:
    guard, plan = _guard_and_plan()
    trainer = _trainer()
    binding = _binding(guard, trainer)
    assert binding.packet is not None
    tampered_packet = deepcopy(binding.packet)
    tampered_packet["identities"]["modelspec_sha256"] = _sha("other-model")
    tampered_root = canonical_sha256(tampered_packet)
    tampered = replace(
        binding,
        packet=tampered_packet,
        packet_sha256=tampered_root,
    )
    with pytest.raises(BoundedPilotAuthorizationError, match="actual ModelSpec differs"):
        BoundedPilotStepRunner(
            SingleDeviceStepRunner(trainer),
            binding=tampered,
            expected_packet_sha256=tampered_root,
            replay_guard=guard,
            exposure_plan=plan,
            expected_plan_identity_sha256=plan["plan_identity_sha256"],
        )
    assert trainer.optimizer_step == 0


def test_zero_authority_binding_cannot_reach_optimizer() -> None:
    guard, plan = _guard_and_plan()
    trainer = _trainer()
    blocked = PortableRunBinding(
        binding_ready=False,
        mode="FRESH_START",
        readiness_ready=False,
        overlay_contract_valid=True,
        packet_contract_valid=True,
        blockers=("readiness:loss_ledger_unique_positions_zero",),
        readiness_sha256=_sha("readiness"),
        overlay_sha256=_sha("overlay"),
        packet_sha256=None,
        packet=None,
    )
    with pytest.raises(BoundedPilotAuthorizationError, match="BLOCKED_PRE_STEP_1"):
        BoundedPilotStepRunner(
            SingleDeviceStepRunner(trainer),
            binding=blocked,
            expected_packet_sha256=_sha("external-packet"),
            replay_guard=guard,
            exposure_plan=plan,
            expected_plan_identity_sha256=plan["plan_identity_sha256"],
        )
    assert trainer.optimizer_step == 0
    assert guard.consumed_loss_positions == 0
