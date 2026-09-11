from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest
import torch
from torch import nn

from twelve_six.data.deterministic_exposure_order import (
    build_deterministic_exposure_plan,
    ordered_next_exposure_identity,
)
from twelve_six.data.identity_safe_exposure_guard import IdentitySafeExposureReplayGuard
from twelve_six.data.unique_loss_ledger_v2 import build_ledger
from twelve_six.portable_run_binding import PortableRunBinding
from twelve_six.training.bounded_pilot import (
    BoundedPilotAuthorizationError,
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
        (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
    ).hexdigest()


def _ledger() -> dict:
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
                "source_bytes": 3,
                "token_count": 3,
                "split": "train",
                "dedup_cluster_id": "cluster-doc",
                "retained_after_dedup": True,
                "evaluation_reserved": False,
                "reserved_target_ranges": [],
                "eligible_target_ranges": [[1, 3]],
            }
        ],
        "packing": {
            "identity_sha256": _sha("packing"),
            "complete_one_pass": True,
            "packs": [
                {
                    "pack_id": "p0",
                    "token_count": 3,
                    "loss_spans": [
                        {
                            "document_id": "doc",
                            "target_start": 1,
                            "target_end": 3,
                            "pack_target_start": 1,
                        }
                    ],
                }
            ],
        },
    }
    materialization["materialization_identity_sha256"] = _identity(
        materialization, "materialization_identity_sha256"
    )
    return build_ledger(materialization)


def _guard_and_plan() -> tuple[IdentitySafeExposureReplayGuard, dict]:
    ledger = _ledger()
    guard = IdentitySafeExposureReplayGuard(
        ledger,
        expected_ledger_identity_sha256=ledger["ledger_identity_sha256"],
        authorized_budget=2,
        trainer_state_binding={
            "checkpoint_generation": "g000",
            "checkpoint_manifest_sha256": _sha("checkpoint"),
            "optimizer_step": 0,
            "trainer_nonignored_target_count": 0,
        },
    )
    segment_id = ledger["segments"][0]["segment_identity_sha256"]
    plan = build_deterministic_exposure_plan(
        [
            {
                "global_batch_index": 0,
                "shard_index": 0,
                "worker_id": 0,
                "claims": [
                    {
                        "segment_identity_sha256": segment_id,
                        "offset_start": 0,
                        "offset_end": 2,
                    }
                ],
                "actual_nonignored_targets": 2,
            }
        ],
        num_workers=1,
        batches_per_shard=1,
        shard_count=1,
    )
    return guard, plan


def _binding(guard: IdentitySafeExposureReplayGuard) -> PortableRunBinding:
    packet = {
        "identities": {
            "canonical_base": "random_init",
            "unique_loss_ledger_sha256": guard.ledger_identity_sha256,
        },
        "recipe": {
            "target_unique_loss_positions": 2,
            "maximum_total_exposures": 2,
            "available_unique_loss_positions": 2,
            "max_exposures_per_unique_position": 1,
        },
        "resource": {
            "resource_class": "LOCAL_FREE",
            "maximum_cost_usd": 0,
            "materially_paid": False,
        },
        "truth_boundary": {"final_test_payload_accessed": False},
    }
    return PortableRunBinding(
        binding_ready=True,
        mode="FRESH_START",
        readiness_ready=True,
        overlay_contract_valid=True,
        packet_contract_valid=True,
        blockers=(),
        readiness_sha256=_sha("readiness"),
        overlay_sha256=_sha("overlay"),
        packet_sha256=_sha("packet"),
        packet=packet,
    )


class _TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.projection = nn.Linear(4, 8)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(input_ids))


def _trainer() -> Trainer:
    return Trainer(
        _TinyLM(),
        TrainerConfig(
            learning_rate=1e-3,
            max_steps=1,
            gradient_accumulation_steps=1,
            gradient_clip_norm=1.0,
            precision="fp32",
            seed=1333,
            deterministic_algorithms=True,
        ),
        device="cpu",
    )


def _batch() -> dict[str, torch.Tensor]:
    return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}


def test_exact_d04_handoff_authorizes_one_optimizer_step() -> None:
    guard, plan = _guard_and_plan()
    trainer = _trainer()
    gate = BoundedPilotStepRunner(
        SingleDeviceStepRunner(trainer),
        binding=_binding(guard),
        replay_guard=guard,
        exposure_plan=plan,
        expected_plan_identity_sha256=plan["plan_identity_sha256"],
    )
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
    assert guard.consumed_loss_positions == 2
    assert receipt.optimizer_step_before == 0
    assert receipt.optimizer_step_after == 1
    assert receipt.actual_nonignored_targets == 2
    assert receipt.exposure_identity_sha256 == expected


def test_wrong_exposure_identity_blocks_optimizer_step_and_rolls_back_guard() -> None:
    guard, plan = _guard_and_plan()
    trainer = _trainer()
    gate = BoundedPilotStepRunner(
        SingleDeviceStepRunner(trainer),
        binding=_binding(guard),
        replay_guard=guard,
        exposure_plan=plan,
        expected_plan_identity_sha256=plan["plan_identity_sha256"],
    )
    before = guard.state_dict()

    with pytest.raises(BoundedPilotAuthorizationError, match="external handoff"):
        gate.train_authorized_microbatch(
            _batch(),
            batch_index=0,
            expected_next_exposure_identity_sha256=_sha("wrong-exposure"),
        )
    gate.close()

    assert trainer.optimizer_step == 0
    assert guard.state_dict() == before


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
            replay_guard=guard,
            exposure_plan=plan,
            expected_plan_identity_sha256=plan["plan_identity_sha256"],
        )

    assert trainer.optimizer_step == 0
    assert guard.consumed_loss_positions == 0
