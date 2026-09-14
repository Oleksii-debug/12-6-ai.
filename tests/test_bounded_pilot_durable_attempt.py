from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from test_bounded_pilot import (
    _authority,
    _batch,
    _binding,
    _launch_authority,
    _next,
)

from twelve_six import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.learned20m_recipe import SESSION_SCHEMA, identity_sha256
from twelve_six.portable_run_binding import PortableRunBinding, canonical_sha256
from twelve_six.training import bounded_pilot as bounded_pilot_module
from twelve_six.training.bounded_pilot import (
    BoundedPilotAuthorizationError,
    BoundedPilotRecoveryRequiredError,
    BoundedPilotStepRunner,
    build_bounded_start_projection,
)
from twelve_six.training.config import TrainerConfig
from twelve_six.training.resilience import RecoveryPolicy, RecoveryStore
from twelve_six.training.single_gpu import SingleDeviceStepRunner
from twelve_six.training.trainer import Trainer

_POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "research"
    / "r01_learned20m_recipe_authority_v1.json"
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _policy() -> dict[str, Any]:
    return json.loads(_POLICY_PATH.read_text(encoding="utf-8"))


def _trainer(
    *,
    max_steps: int = 1,
    config_overrides: dict[str, Any] | None = None,
) -> Trainer:
    seed = 20260826
    torch.manual_seed(seed)
    model = TwelveSixDecoder(
        ModelSpec(
            schema_version=1,
            vocab_size=32,
            max_seq_len=128,
            d_model=8,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            head_dim=4,
            d_ff=16,
            rope_rotary_dim=4,
        ),
        InitSpec(),
    )
    config_values: dict[str, Any] = {
        "learning_rate": 0.00022,
        "weight_decay": 0.1,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "max_steps": max_steps,
        "warmup_steps": 0,
        "scheduler": "constant",
        "gradient_accumulation_steps": 1,
        "gradient_clip_norm": 1.0,
        "precision": "fp32",
        "seed": seed,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
    }
    if config_overrides:
        config_values.update(config_overrides)
    return Trainer(
        model,
        TrainerConfig(**config_values),
        device="cpu",
    )


def _qualified_session(policy: dict[str, Any], *, budget: int) -> dict[str, Any]:
    core = {
        "schema": SESSION_SCHEMA,
        "status": "QUALIFIED_RECIPE_ONLY",
        "policy_identity_sha256": policy["policy_identity_sha256"],
        "bindings_identity_sha256": _sha("pytest-learn345-bindings"),
        "trusted_authorities_identity_sha256": _sha("pytest-learn345-trusted"),
        "qualified_runtime_unique_loss_positions": budget,
        "training_recipe_status": "QUALIFIED",
        "training_authorized": False,
        "compute_authorized": False,
        "authorized_optimized_targets": 0,
        "optimizer_updates_executed": 0,
    }
    return {**core, "session_identity_sha256": identity_sha256(core)}


def _binding_for_session(
    trainer: Trainer,
    guard,
    launch_root: str,
    session: dict[str, Any],
) -> tuple[PortableRunBinding, str]:
    legacy, portable_execution = _binding(trainer, guard, launch_root)
    assert legacy.packet is not None
    packet = copy.deepcopy(legacy.packet)
    packet["recipe"]["training_config_sha256"] = session["session_identity_sha256"]
    packet_sha = canonical_sha256(packet)
    return (
        PortableRunBinding(
            binding_ready=legacy.binding_ready,
            mode=legacy.mode,
            readiness_ready=legacy.readiness_ready,
            overlay_contract_valid=legacy.overlay_contract_valid,
            packet_contract_valid=legacy.packet_contract_valid,
            blockers=legacy.blockers,
            readiness_sha256=legacy.readiness_sha256,
            overlay_sha256=legacy.overlay_sha256,
            packet_sha256=packet_sha,
            packet=packet,
        ),
        portable_execution,
    )


def _constructor_inputs(
    root: Path,
    trainer: Trainer,
    guard,
    plan: dict[str, Any],
    manifest: dict[str, Any],
    *,
    runner: SingleDeviceStepRunner | None = None,
    run_id: str = "explicit-authority-bounded-pilot",
    policy: dict[str, Any] | None = None,
) -> tuple[SingleDeviceStepRunner, dict[str, Any], RecoveryStore]:
    selected_policy = copy.deepcopy(policy if policy is not None else _policy())
    session = _qualified_session(selected_policy, budget=guard.authorized_budget)
    launch = _launch_authority(trainer, guard)
    launch_root = launch["authority_identity_sha256"]
    binding, portable_execution = _binding_for_session(
        trainer,
        guard,
        launch_root,
        session,
    )
    assert binding.packet_sha256 is not None

    state = guard.state_dict()
    state_root = state["state_identity_sha256"]
    assert isinstance(state_root, str)
    projection = build_bounded_start_projection(
        packet_sha256=binding.packet_sha256,
        modelspec_sha256=trainer.model.spec.identity_sha256(),
        initspec_sha256=trainer.model.init_spec.identity_sha256(),
        ledger_identity_sha256=guard.ledger_identity_sha256,
        loss_bearing_manifest_identity_sha256=manifest["manifest_identity_sha256"],
        exposure_plan_identity_sha256=plan["plan_identity_sha256"],
        exposure_state_identity_sha256=state_root,
        authorized_unique_loss_positions=guard.authorized_budget,
        training_config_sha256=session["session_identity_sha256"],
        training_recipe_policy_identity_sha256=selected_policy[
            "policy_identity_sha256"
        ],
    )
    run_manifest = {
        "schema_version": "12-6.pytest-bounded-pilot-run-manifest.v2",
        "run_id": run_id,
        "candidate": {"git_sha": "2" * 40},
        "recovery": {
            "topology": {
                "backend": "single-process-cpu",
                "world_size": 1,
                "rank_count": 1,
                "resume_policy": "exact_topology",
            }
        },
        "bounded_pilot": projection,
    }
    store = RecoveryStore(
        root,
        run_manifest=run_manifest,
        policy=RecoveryPolicy(
            checkpoint_every_steps=1,
            retain_last=2,
            max_restarts=0,
            max_preemptions=0,
        ),
    )
    kwargs = {
        "binding": binding,
        "expected_packet_sha256": binding.packet_sha256,
        "expected_portable_execution_sha256": portable_execution,
        "launch_input_authority": launch,
        "expected_launch_input_authority_identity_sha256": launch_root,
        "replay_guard": guard,
        "loss_bearing_content_manifest": manifest,
        "expected_loss_bearing_manifest_identity_sha256": manifest[
            "manifest_identity_sha256"
        ],
        "exposure_plan": plan,
        "expected_plan_identity_sha256": plan["plan_identity_sha256"],
        "recovery_store": store,
        "expected_run_manifest_sha256": store.run_manifest_sha256,
        "expected_run_id": store.run_id,
        "training_recipe_policy": selected_policy,
        "training_recipe_session": session,
        "expected_training_config_sha256": session["session_identity_sha256"],
    }
    return runner or SingleDeviceStepRunner(trainer), kwargs, store


def _gate(
    root: Path,
    trainer: Trainer,
    guard,
    plan: dict[str, Any],
    manifest: dict[str, Any],
    *,
    runner: SingleDeviceStepRunner | None = None,
    run_id: str = "explicit-authority-bounded-pilot",
    policy: dict[str, Any] | None = None,
) -> tuple[BoundedPilotStepRunner, RecoveryStore]:
    selected_runner, kwargs, store = _constructor_inputs(
        root,
        trainer,
        guard,
        plan,
        manifest,
        runner=runner,
        run_id=run_id,
        policy=policy,
    )
    return BoundedPilotStepRunner(selected_runner, **kwargs), store


def test_fresh_process_same_run_cannot_replay_ambiguous_committed_attempt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "durable-run"
    run_id = "fresh-process-replay-regression"
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
    gate, _store = _gate(
        root,
        trainer,
        guard,
        plan,
        manifest,
        runner=runner,
        run_id=run_id,
    )
    with pytest.raises(BoundedPilotRecoveryRequiredError):
        gate.train_authorized_microbatch(
            _batch(0),
            batch_index=0,
            expected_next_exposure_identity_sha256=_next(guard, plan, 0),
        )
    assert trainer.optimizer_step == 1
    assert guard.consumed_loss_positions == 2
    gate.close()

    # Simulate a new Python process: rebuild every live object from stale zero state,
    # but point at the exact same immutable TRAIN39 run manifest and attempt directory.
    fresh_guard, fresh_plan, fresh_manifest = _authority(batch_count=1)
    fresh_trainer = _trainer(max_steps=1)
    fresh_gate, _fresh_store = _gate(
        root,
        fresh_trainer,
        fresh_guard,
        fresh_plan,
        fresh_manifest,
        run_id=run_id,
    )
    before = [parameter.detach().clone() for parameter in fresh_trainer.model.parameters()]
    with pytest.raises(
        BoundedPilotRecoveryRequiredError,
        match="already has durable attempt/history",
    ):
        fresh_gate.train_authorized_microbatch(
            _batch(0),
            batch_index=0,
            expected_next_exposure_identity_sha256=_next(fresh_guard, fresh_plan, 0),
        )
    fresh_gate.close()
    assert fresh_trainer.optimizer_step == 0
    assert fresh_guard.consumed_loss_positions == 0
    assert all(
        left.equal(right)
        for left, right in zip(before, fresh_trainer.model.parameters(), strict=True)
    )


def test_durable_receipt_binds_run_attempt_and_recipe_roots(tmp_path: Path) -> None:
    guard, plan, manifest = _authority(batch_count=1)
    trainer = _trainer(max_steps=1)
    gate, store = _gate(tmp_path / "receipt", trainer, guard, plan, manifest)
    _metrics, receipt = gate.train_authorized_microbatch(
        _batch(0),
        batch_index=0,
        expected_next_exposure_identity_sha256=_next(guard, plan, 0),
    )
    gate.close()
    assert receipt.run_id == store.run_id
    assert receipt.attempt == 1
    assert receipt.run_manifest_sha256 == store.run_manifest_sha256
    assert len(receipt.attempt_state_sha256) == 64
    assert len(receipt.training_config_sha256) == 64
    assert receipt.training_recipe_policy_identity_sha256 == _policy()[
        "policy_identity_sha256"
    ]


def test_production_facade_has_no_test_recovery_factory() -> None:
    assert not hasattr(bounded_pilot_module, "_set_test_recovery_store_factory")
    assert not hasattr(bounded_pilot_module, "_TEST_RECOVERY_STORE_FACTORY")


def test_production_constructor_requires_explicit_durable_authority(tmp_path: Path) -> None:
    guard, plan, manifest = _authority(batch_count=1)
    trainer = _trainer(max_steps=1)
    runner, kwargs, _store = _constructor_inputs(
        tmp_path / "required",
        trainer,
        guard,
        plan,
        manifest,
    )
    kwargs.pop("recovery_store")
    kwargs.pop("expected_run_manifest_sha256")
    kwargs.pop("expected_run_id")
    with pytest.raises(TypeError, match="recovery_store"):
        BoundedPilotStepRunner(runner, **kwargs)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("optimizer", "SGD"),
        ("learning_rate", 0.00021),
        ("betas", [0.9, 0.94]),
        ("eps", 1e-7),
        ("weight_decay", 0.0),
        ("gradient_clip_norm", 0.5),
        ("scheduler", "cosine"),
        ("warmup_steps", 1),
        ("sequence_length", 64),
        ("micro_batch_size", 2),
        ("gradient_accumulation_steps", 2),
        ("precision", "bf16"),
        (
            "seed_vector",
            {
                "model_init": 20260827,
                "data_order": 20260826,
                "dataloader": 20260826,
            },
        ),
    ],
)
def test_rehashed_recipe_policy_mutation_is_rejected_before_step(
    tmp_path: Path,
    field: str,
    replacement: Any,
) -> None:
    policy = _policy()
    policy["recipe"][field] = replacement
    body = {key: value for key, value in policy.items() if key != "policy_identity_sha256"}
    policy["policy_identity_sha256"] = identity_sha256(body)

    guard, plan, manifest = _authority(batch_count=1)
    trainer = _trainer(max_steps=1)
    runner, kwargs, store = _constructor_inputs(
        tmp_path / f"policy-{field}",
        trainer,
        guard,
        plan,
        manifest,
        policy=policy,
    )
    with pytest.raises(
        BoundedPilotAuthorizationError,
        match="canonical LEARN-345 policy rejected",
    ):
        BoundedPilotStepRunner(runner, **kwargs)
    assert trainer.optimizer_step == 0
    assert guard.consumed_loss_positions == 0
    assert store.open()["attempt"] == 0


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("learning_rate", 0.00021),
        ("weight_decay", 0.0),
        ("betas", (0.9, 0.94)),
        ("eps", 1e-7),
        ("gradient_clip_norm", 0.5),
        ("scheduler", "cosine"),
        ("warmup_steps", 1),
        ("seed", 20260827),
        ("max_steps", 2),
    ],
)
def test_self_consistent_initial_trainer_drift_rejected_by_authenticated_recipe(
    tmp_path: Path,
    field: str,
    replacement: Any,
) -> None:
    guard, plan, manifest = _authority(batch_count=1)
    overrides = {field: replacement}
    trainer = _trainer(max_steps=1, config_overrides=overrides)
    runner, kwargs, store = _constructor_inputs(
        tmp_path / f"self-consistent-{field}",
        trainer,
        guard,
        plan,
        manifest,
    )
    with pytest.raises(BoundedPilotAuthorizationError):
        BoundedPilotStepRunner(runner, **kwargs)
    assert trainer.optimizer_step == 0
    assert guard.consumed_loss_positions == 0
    assert store.open()["attempt"] == 0


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("learning_rate", 0.00021),
        ("weight_decay", 0.0),
        ("betas", (0.9, 0.94)),
        ("eps", 1e-7),
        ("gradient_clip_norm", 0.5),
        ("scheduler", "cosine"),
        ("warmup_steps", 1),
        ("gradient_accumulation_steps", 2),
        ("precision", "bf16"),
        ("seed", 20260827),
        ("deterministic_algorithms", False),
        ("deterministic_warn_only", True),
        ("max_steps", 2),
    ],
)
def test_live_trainer_recipe_drift_is_rejected_before_durable_attempt(
    tmp_path: Path,
    field: str,
    replacement: Any,
) -> None:
    guard, plan, manifest = _authority(batch_count=1)
    trainer = _trainer(max_steps=1)
    gate, store = _gate(
        tmp_path / f"live-{field}",
        trainer,
        guard,
        plan,
        manifest,
    )
    object.__setattr__(trainer.config, field, replacement)

    with pytest.raises(BoundedPilotAuthorizationError):
        gate.train_authorized_microbatch(
            _batch(0),
            batch_index=0,
            expected_next_exposure_identity_sha256=_next(guard, plan, 0),
        )
    gate.close()
    assert trainer.optimizer_step == 0
    assert guard.consumed_loss_positions == 0
    assert store.open()["attempt"] == 0
