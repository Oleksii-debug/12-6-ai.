from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest
import torch

from twelve_six import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.data.deterministic_exposure_order import (
    build_deterministic_exposure_plan,
    ordered_next_exposure_identity,
)
from twelve_six.data.identity_safe_exposure_guard import IdentitySafeExposureReplayGuard
from twelve_six.data.loss_bearing_content_binding_v1 import (
    build_loss_bearing_content_manifest,
)
from twelve_six.data.unique_loss_ledger_v2 import build_ledger
from twelve_six.portable_run_binding import PortableRunBinding, canonical_sha256
from twelve_six.training.bounded_pilot import (
    BoundedPilotAuthorizationError,
    BoundedPilotRecoveryRequiredError,
    BoundedPilotStepRunner,
)
from twelve_six.training.config import TrainerConfig
from twelve_six.training.single_gpu import SingleDeviceStepRunner
from twelve_six.training.trainer import Trainer, build_optimizer


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _authority_sha(value: dict) -> str:
    return hashlib.sha256(
        (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
    ).hexdigest()


def _rehash(value: dict, field: str) -> str:
    body = deepcopy(value)
    body.pop(field, None)
    return _authority_sha(body)


def _spec() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=32,
        max_seq_len=8,
        d_model=8,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        head_dim=4,
        d_ff=16,
        rope_rotary_dim=4,
    )


def _trainer(*, max_steps: int = 2, seed: int = 1333) -> Trainer:
    torch.manual_seed(seed)
    model = TwelveSixDecoder(_spec(), InitSpec())
    return Trainer(
        model,
        TrainerConfig(
            learning_rate=1e-3,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            eps=1e-8,
            max_steps=max_steps,
            warmup_steps=0,
            scheduler="constant",
            gradient_accumulation_steps=1,
            gradient_clip_norm=1.0,
            precision="fp32",
            seed=seed,
            deterministic_algorithms=True,
        ),
        device="cpu",
    )


def _materialization() -> dict:
    value = {
        "schema_version": "12-6.postpack-loss-materialization.v2",
        "terminal_corpus_authority_identity_sha256": _sha("terminal-corpus"),
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
                "token_count": 5,
                "split": "train",
                "dedup_cluster_id": "cluster-doc",
                "retained_after_dedup": True,
                "evaluation_reserved": False,
                "reserved_target_ranges": [],
                "eligible_target_ranges": [[1, 5]],
            }
        ],
        "packing": {
            "identity_sha256": _sha("packing"),
            "complete_one_pass": True,
            "packs": [
                {
                    "pack_id": "p0",
                    "token_count": 5,
                    "token_ids": [1, 2, 3, 4, 5],
                    "loss_spans": [
                        {
                            "document_id": "doc",
                            "target_start": 1,
                            "target_end": 5,
                            "pack_target_start": 1,
                        }
                    ],
                }
            ],
        },
    }
    value["materialization_identity_sha256"] = _rehash(
        value, "materialization_identity_sha256"
    )
    return value


def _authority(batch_count: int = 2) -> tuple[IdentitySafeExposureReplayGuard, dict, dict]:
    materialization = _materialization()
    ledger = build_ledger(materialization)
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
    manifest = build_loss_bearing_content_manifest(
        materialization,
        ledger,
        plan,
        expected_materialization_identity_sha256=materialization[
            "materialization_identity_sha256"
        ],
        expected_ledger_identity_sha256=ledger["ledger_identity_sha256"],
        expected_plan_identity_sha256=plan["plan_identity_sha256"],
    )
    guard = IdentitySafeExposureReplayGuard(
        ledger,
        expected_ledger_identity_sha256=ledger["ledger_identity_sha256"],
        authorized_budget=2 * batch_count,
        trainer_state_binding={
            "checkpoint_generation": "g000",
            "checkpoint_manifest_sha256": _sha("checkpoint"),
            "optimizer_step": 0,
            "trainer_nonignored_target_count": 0,
        },
        loss_bearing_content_manifest=manifest,
        expected_loss_bearing_manifest_identity_sha256=manifest[
            "manifest_identity_sha256"
        ],
    )
    return guard, plan, manifest


def _launch_authority(trainer: Trainer, guard: IdentitySafeExposureReplayGuard) -> dict:
    value = {
        "schema_version": "12-6.learned20m-launch-input-authority.v2",
        "binding_status": "READY_FOR_READINESS_BINDING",
        "data_spine": {
            "terminal_corpus_authority_identity_sha256": _sha("corpus"),
            "stage_bindings": {"d04": _sha("d04")},
            "deterministic_double_pack_proof_identity_sha256": _sha("double-pack"),
            "terminal_record_inventory_digest_sha256": _sha("records"),
            "terminal_payload_inventory_digest_sha256": _sha("payloads"),
            "terminal_split_application_identity_sha256": _sha("split-app"),
            "terminal_split_spec_identity_sha256": _sha("split-spec"),
            "terminal_split_train_record_membership_sha256": _sha("membership"),
            "canonical_build_sha256": _sha("build"),
            "two_clean_proof_identity_sha256": _sha("two-clean"),
            "two_clean_input_packet_identity_sha256": _sha("two-clean-input"),
            "two_clean_runtime_identity_sha256": _sha("two-clean-runtime"),
            "materialization_identity_sha256": _sha("materialization"),
            "unique_loss_ledger_identity_sha256": guard.ledger_identity_sha256,
            "tokenizer_identity_sha256": _sha("tokenizer-root"),
            "packing_identity_sha256": _sha("pack-root"),
            "one_pass_unique_nonignored_causal_loss_positions": guard.one_pass_maximum,
            "requested_unique_loss_positions": guard.authorized_budget,
        },
        "carrier": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": _sha("git-sha")[:40],
            "modelspec_sha256": trainer.model.spec.identity_sha256(),
            "initialization_identity_sha256": trainer.model.init_spec.identity_sha256(),
            "canonical_base": "random_init",
            "foreign_pretrained_weights_used": False,
            "terminal": True,
            "workflow_run_id": "fixture-run",
            "workflow_status": "completed",
            "workflow_conclusion": "success",
            "workflow_head_sha": _sha("head")[:40],
            "evidence_sha256": _sha("evidence"),
        },
        "claim_boundary": {
            "contains_source_text": False,
            "final_test_payload_consumed": False,
            "authorizes_training": False,
            "authorizes_compute": False,
            "authorized_optimized_target_exposure": 0,
            "replay_padding_or_replacement_can_increase_unique_capacity": False,
        },
    }
    body = deepcopy(value)
    body.pop("authority_identity_sha256", None)
    value["authority_identity_sha256"] = _authority_sha(body)
    return value


def _binding(
    trainer: Trainer,
    guard: IdentitySafeExposureReplayGuard,
    launch_root: str,
    *,
    mode: str = "FRESH_START",
    overlay_projection: dict | None = None,
) -> tuple[PortableRunBinding, str]:
    portable_execution = _sha("portable-execution")
    packet = {
        "identities": {
            "canonical_base": "random_init",
            "modelspec_sha256": trainer.model.spec.identity_sha256(),
            "initspec_sha256": trainer.model.init_spec.identity_sha256(),
            "unique_loss_ledger_sha256": guard.ledger_identity_sha256,
        },
        "recipe": {
            "training_config_sha256": _sha("session"),
            "target_unique_loss_positions": guard.authorized_budget,
            "maximum_total_exposures": guard.authorized_budget,
            "available_unique_loss_positions": guard.one_pass_maximum,
            "max_exposures_per_unique_position": 1,
            "seed": trainer.config.seed,
            "optimizer_scheduler_precision": (
                {
                    "optimizer": trainer.optimizer.__class__.__name__,
                    "scheduler": trainer.config.scheduler,
                    "precision": trainer.config.precision,
                }
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
        "binding": {
            "portable_execution_sha256": portable_execution,
            "launch_input_authority_identity_sha256": launch_root,
        },
    }
    packet_sha = canonical_sha256(packet)
    binding = PortableRunBinding(
        binding_ready=True,
        mode=mode,
        readiness_ready=True,
        overlay_contract_valid=True,
        packet_contract_valid=True,
        blockers=(),
        readiness_sha256=_sha("readiness"),
        overlay_sha256=_sha("overlay"),
        packet_sha256=packet_sha,
        packet=packet,
    )
    return binding, portable_execution


def _batch(batch_index: int) -> dict[str, torch.Tensor]:
    if batch_index == 0:
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "target_ids": torch.tensor([[2, 3]], dtype=torch.long),
        }
    if batch_index == 1:
        return {
            "input_ids": torch.tensor([[3, 4]], dtype=torch.long),
            "target_ids": torch.tensor([[4, 5]], dtype=torch.long),
        }
    raise AssertionError("test batch index outside fixture")


def _next(guard: IdentitySafeExposureReplayGuard, plan: dict, index: int) -> str:
    return ordered_next_exposure_identity(
        guard,
        plan,
        batch_index=index,
        expected_plan_identity_sha256=plan["plan_identity_sha256"],
    )


def _gate(
    trainer: Trainer,
    guard: IdentitySafeExposureReplayGuard,
    plan: dict,
    manifest: dict,
    *,
    mode: str = "FRESH_START",
    runner: SingleDeviceStepRunner | None = None,
    launch_root_override: str | None = None,
) -> BoundedPilotStepRunner:
    launch = _launch_authority(trainer, guard)
    launch_root = launch["authority_identity_sha256"]
    binding, portable_execution = _binding(trainer, guard, launch_root, mode=mode)
    assert binding.packet_sha256 is not None
    return BoundedPilotStepRunner(
        runner or SingleDeviceStepRunner(trainer),
        binding=binding,
        expected_packet_sha256=binding.packet_sha256,
        expected_portable_execution_sha256=portable_execution,
        launch_input_authority=launch,
        expected_launch_input_authority_identity_sha256=(
            launch_root if launch_root_override is None else launch_root_override
        ),
        replay_guard=guard,
        loss_bearing_content_manifest=manifest,
        expected_loss_bearing_manifest_identity_sha256=manifest["manifest_identity_sha256"],
        exposure_plan=plan,
        expected_plan_identity_sha256=plan["plan_identity_sha256"],
    )


def test_exact_live_content_authorizes_two_real_optimizer_steps() -> None:
    guard, plan, manifest = _authority(batch_count=2)
    trainer = _trainer(max_steps=2)
    gate = _gate(trainer, guard, plan, manifest)
    receipts = []
    for batch_index in range(2):
        expected = _next(guard, plan, batch_index)
        metrics, receipt = gate.train_authorized_microbatch(
            _batch(batch_index),
            batch_index=batch_index,
            expected_next_exposure_identity_sha256=expected,
        )
        assert metrics.trainer.optimizer_stepped is True
        receipts.append(receipt)
    gate.close()
    assert trainer.optimizer_step == 2
    assert trainer.tokens_seen == 4
    assert guard.consumed_loss_positions == 4
    assert receipts[0].exposure_identity_sha256 != receipts[1].exposure_identity_sha256


@pytest.mark.parametrize("surface", ["input", "target"])
def test_same_cardinality_live_content_substitution_blocks_before_gradient(surface: str) -> None:
    guard, plan, manifest = _authority(batch_count=1)
    trainer = _trainer(max_steps=1)
    gate = _gate(trainer, guard, plan, manifest)
    batch = _batch(0)
    key = "input_ids" if surface == "input" else "target_ids"
    batch[key] = batch[key].clone()
    batch[key][0, 0] = 17
    before = [parameter.detach().clone() for parameter in trainer.model.parameters()]
    with pytest.raises(BoundedPilotAuthorizationError, match="live loss-bearing content rejected"):
        gate.train_authorized_microbatch(
            batch,
            batch_index=0,
            expected_next_exposure_identity_sha256=_next(guard, plan, 0),
        )
    gate.close()
    assert trainer.optimizer_step == 0
    assert trainer.tokens_seen == 0
    assert guard.consumed_loss_positions == 0
    assert all(
        torch.equal(left, right)
        for left, right in zip(before, trainer.model.parameters(), strict=True)
    )


def test_fresh_start_rejects_same_architecture_mutated_weights() -> None:
    guard, plan, manifest = _authority(batch_count=1)
    trainer = _trainer(max_steps=1)
    with torch.no_grad():
        next(trainer.model.parameters()).view(1)[0].add_(0.25)
    with pytest.raises(BoundedPilotAuthorizationError, match="canonical random initialization"):
        _gate(trainer, guard, plan, manifest)


def test_out_of_band_model_mutation_between_steps_fails_closed() -> None:
    guard, plan, manifest = _authority()
    trainer = _trainer(max_steps=2)
    gate = _gate(trainer, guard, plan, manifest)
    gate.train_authorized_microbatch(
        _batch(0),
        batch_index=0,
        expected_next_exposure_identity_sha256=_next(guard, plan, 0),
    )
    with torch.no_grad():
        next(trainer.model.parameters()).view(-1)[0].add_(0.25)
    with pytest.raises(BoundedPilotAuthorizationError, match="outside authorized optimizer chain"):
        gate.train_authorized_microbatch(
            _batch(1),
            batch_index=1,
            expected_next_exposure_identity_sha256=_next(guard, plan, 1),
        )
    gate.close()
    assert trainer.optimizer_step == 1
    assert guard.consumed_loss_positions == 2


def test_optimizer_object_replacement_cannot_bypass_original_pre_hook() -> None:
    guard, plan, manifest = _authority(batch_count=1)
    trainer = _trainer(max_steps=1)
    gate = _gate(trainer, guard, plan, manifest)
    trainer.optimizer = build_optimizer(trainer.model, trainer.config)
    with pytest.raises(BoundedPilotAuthorizationError, match="optimizer object replaced"):
        gate.train_authorized_microbatch(
            _batch(0),
            batch_index=0,
            expected_next_exposure_identity_sha256=_next(guard, plan, 0),
        )
    gate.close()
    assert trainer.optimizer_step == 0
    assert guard.consumed_loss_positions == 0


def test_noncanonical_adamw_behavior_flag_fails_closed() -> None:
    guard, plan, manifest = _authority(batch_count=1)
    trainer = _trainer(max_steps=1)
    gate = _gate(trainer, guard, plan, manifest)
    trainer.optimizer.param_groups[0]["maximize"] = True
    with pytest.raises(BoundedPilotAuthorizationError, match="group semantics"):
        gate.train_authorized_microbatch(
            _batch(0),
            batch_index=0,
            expected_next_exposure_identity_sha256=_next(guard, plan, 0),
        )
    gate.close()
    assert trainer.optimizer_step == 0


def test_stale_live_trainer_replay_state_is_rechecked_per_step() -> None:
    guard, plan, manifest = _authority(batch_count=1)
    trainer = _trainer(max_steps=1)
    gate = _gate(trainer, guard, plan, manifest)
    trainer.optimizer_step = 1
    with pytest.raises(BoundedPilotAuthorizationError, match="replay optimizer state stale"):
        gate.train_authorized_microbatch(
            _batch(0),
            batch_index=0,
            expected_next_exposure_identity_sha256=_next(guard, plan, 0),
        )
    gate.close()
    assert guard.consumed_loss_positions == 0


def test_wrong_external_launch_root_fails_before_step_1() -> None:
    guard, plan, manifest = _authority(batch_count=1)
    trainer = _trainer(max_steps=1)
    launch = _launch_authority(trainer, guard)
    launch_root = launch["authority_identity_sha256"]
    binding, portable_execution = _binding(trainer, guard, launch_root)
    assert binding.packet_sha256 is not None
    with pytest.raises(
        BoundedPilotAuthorizationError,
        match="D10 launch-input authority identity mismatch",
    ):
        BoundedPilotStepRunner(
            SingleDeviceStepRunner(trainer),
            binding=binding,
            expected_packet_sha256=binding.packet_sha256,
            expected_portable_execution_sha256=portable_execution,
            launch_input_authority=launch,
            expected_launch_input_authority_identity_sha256=_sha("wrong-launch"),
            replay_guard=guard,
            loss_bearing_content_manifest=manifest,
            expected_loss_bearing_manifest_identity_sha256=manifest[
                "manifest_identity_sha256"
            ],
            exposure_plan=plan,
            expected_plan_identity_sha256=plan["plan_identity_sha256"],
        )
    assert trainer.optimizer_step == 0


def test_resume_mode_is_explicitly_blocked_until_checkpoint_root_is_consumed() -> None:
    guard, plan, manifest = _authority(batch_count=1)
    trainer = _trainer(max_steps=1)
    with pytest.raises(BoundedPilotAuthorizationError, match="FRESH_START only"):
        _gate(trainer, guard, plan, manifest, mode="RESUME")


def test_post_commit_failure_never_rewinds_consumed_exposure() -> None:
    guard, plan, manifest = _authority(batch_count=1)
    trainer = _trainer(max_steps=1)
    runner = SingleDeviceStepRunner(trainer)
    synchronize_calls = 0

    def fail_after_trainer_commit() -> None:
        nonlocal synchronize_calls
        synchronize_calls += 1
        if synchronize_calls == 3:
            raise RuntimeError("synthetic post-step synchronization failure")

    runner._synchronize = fail_after_trainer_commit  # type: ignore[method-assign]
    gate = _gate(trainer, guard, plan, manifest, runner=runner)
    expected = _next(guard, plan, 0)
    with pytest.raises(BoundedPilotRecoveryRequiredError, match="fresh verified recovery"):
        gate.train_authorized_microbatch(
            _batch(0),
            batch_index=0,
            expected_next_exposure_identity_sha256=expected,
        )
    assert gate.poisoned is True
    assert trainer.optimizer_step == 1
    assert guard.consumed_loss_positions == 2
    with pytest.raises(BoundedPilotRecoveryRequiredError, match="fresh verified recovery"):
        gate.train_authorized_microbatch(
            _batch(0),
            batch_index=0,
            expected_next_exposure_identity_sha256=expected,
        )
    gate.close()
    assert trainer.optimizer_step == 1
    assert guard.consumed_loss_positions == 2
