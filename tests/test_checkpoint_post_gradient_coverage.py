"""D05 must recheck optimizer ownership after effectful gradient inspection.

Synthetic CPU coverage only; this grants no corpus, training, or learned-weight evidence.
"""
from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from twelve_six.checkpoint import CheckpointIdentity, core
from twelve_six.checkpoint import progress_trainer, trainer_adapter
from twelve_six.training import Trainer, TrainerConfig


class _TinyLogits(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight.reshape(1, 1, 3).expand(*input_ids.shape, 3)


_BATCH = {
    "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
    "target_ids": torch.tensor([[1, 2]], dtype=torch.long),
}


@pytest.fixture(autouse=True)
def preserve_process_state():
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    policy_before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        yield
    finally:
        random.setstate(py_before)
        np.random.set_state(np_before)
        torch.set_rng_state(torch_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state_all(cuda_before)
        torch.use_deterministic_algorithms(
            policy_before[0],
            warn_only=policy_before[1],
        )


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "post-gradient-coverage", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 3},
        seed=811,
        precision="fp32",
        step=1,
        tokens_seen=2,
        optimizer={"name": "AdamW"},
        scheduler={"name": "cosine"},
        environment_lock_hash="f" * 64,
    )


def _source() -> Trainer:
    source = Trainer(
        _TinyLogits(),
        TrainerConfig(seed=811, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    assert source.train_microbatch(_BATCH).optimizer_stepped
    return source


@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "restore_rng",
    [False, True],
    ids=["opt-out", "exact-rng"],
)
def test_final_gradient_scan_cannot_invalidate_optimizer_coverage_before_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    restore_rng: bool,
) -> None:
    source = _source()
    checkpoint = tmp_path / "post-gradient-coverage-дані з пробілами"
    core.save_checkpoint(
        checkpoint,
        model=source.model,
        trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(checkpoint)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    initial_weights = target.model.weight.detach().clone()
    assert not torch.equal(initial_weights, source.model.weight.detach())

    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    policy_before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )

    original_preflight = loader._preflight_trainer_state
    original_bind = loader._bind_model_state_loader
    original_parameters = target.model.parameters
    foreign_parameter = torch.nn.Parameter(torch.zeros_like(target.model.weight))
    preflight_calls = 0
    parameter_calls = 0
    model_applications: list[bool] = []

    def bind_tracked_model_loader(model: Any, strict: bool):
        apply = original_bind(model, strict)

        def tracked_apply(materialized: Any) -> Any:
            model_applications.append(True)
            return apply(materialized)

        return tracked_apply

    def effectful_parameters(*args: Any, **kwargs: Any):
        nonlocal parameter_calls
        parameter_calls += 1
        if parameter_calls == 2:
            target.optimizer.param_groups[0]["params"][0] = foreign_parameter
        return original_parameters(*args, **kwargs)

    def preflight_then_arm(*args: Any, **kwargs: Any) -> Any:
        nonlocal preflight_calls
        result = original_preflight(*args, **kwargs)
        preflight_calls += 1
        if preflight_calls == 3:
            monkeypatch.setattr(target.model, "parameters", effectful_parameters)
        return result

    monkeypatch.setattr(loader, "_bind_model_state_loader", bind_tracked_model_loader)
    monkeypatch.setattr(loader, "_preflight_trainer_state", preflight_then_arm)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="stable optimizer ownership",
    ):
        loader.load_trainer_checkpoint(
            checkpoint,
            model=target.model,
            trainer=target,
            strict_model=False,
            restore_rng=restore_rng,
            **extra,
        )

    assert preflight_calls == 3
    assert parameter_calls >= 3
    assert model_applications == []
    assert target.optimizer.param_groups[0]["params"][0] is foreign_parameter
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)

    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)
    assert (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    ) == policy_before

@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "restore_rng",
    [False, True],
    ids=["opt-out", "exact-rng"],
)
def test_final_authority_lookup_cannot_change_canonical_restore_policy_before_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    restore_rng: bool,
) -> None:
    source = _source()
    checkpoint = tmp_path / "restore-policy-drift-дані з пробілами"
    core.save_checkpoint(
        checkpoint,
        model=source.model,
        trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(checkpoint)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    initial_weights = target.model.weight.detach().clone()
    expected_policy = dict(target._canonical_default_optimizer_options)

    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    policy_before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )

    original_preflight = loader._preflight_trainer_state
    original_bind = loader._bind_model_state_loader
    original_coverage = Trainer.__dict__["_require_optimizer_parameter_coverage"]
    preflight_calls = 0
    model_applications: list[bool] = []

    class EffectfulCoverage:
        def __get__(self, instance: Any, owner: type[Any]) -> Any:
            if instance is target:
                target._canonical_default_optimizer_options["drifted"] = True
            return original_coverage.__get__(instance, owner)

    def bind_tracked_model_loader(model: Any, strict: bool):
        apply = original_bind(model, strict)

        def tracked_apply(materialized: Any) -> Any:
            model_applications.append(True)
            return apply(materialized)

        return tracked_apply

    def preflight_then_arm(*args: Any, **kwargs: Any) -> Any:
        nonlocal preflight_calls
        result = original_preflight(*args, **kwargs)
        preflight_calls += 1
        if preflight_calls == 3:
            monkeypatch.setattr(
                Trainer,
                "_require_optimizer_parameter_coverage",
                EffectfulCoverage(),
            )
        return result

    monkeypatch.setattr(loader, "_bind_model_state_loader", bind_tracked_model_loader)
    monkeypatch.setattr(loader, "_preflight_trainer_state", preflight_then_arm)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="_canonical_default_optimizer_options policy changed during checkpoint restore",
    ):
        loader.load_trainer_checkpoint(
            checkpoint,
            model=target.model,
            trainer=target,
            strict_model=False,
            restore_rng=restore_rng,
            **extra,
        )

    assert preflight_calls == 3
    assert model_applications == []
    assert target._canonical_default_optimizer_options != expected_policy
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)

    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)
    assert (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    ) == policy_before

@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
def test_restore_policy_snapshot_rejects_exotic_objects_without_running_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
) -> None:
    target = Trainer(
        _TinyLogits(),
        TrainerConfig(seed=811, max_steps=3, scheduler="cosine"),
        device="cpu",
    )
    hook_calls: list[str] = []
    checkpoint_reads: list[bool] = []

    class EffectfulPolicy:
        def __deepcopy__(self, memo: Any) -> Any:
            del memo
            hook_calls.append("deepcopy")
            random.random()
            np.random.random()
            torch.rand(1)
            return self

        def __eq__(self, other: Any) -> bool:
            del other
            hook_calls.append("eq")
            random.random()
            np.random.random()
            torch.rand(1)
            return True

    target._canonical_default_optimizer_options = EffectfulPolicy()
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    def forbid_checkpoint_read(*args: Any, **kwargs: Any) -> Any:
        checkpoint_reads.append(True)
        raise AssertionError("invalid restore policy reached checkpoint I/O")

    monkeypatch.setattr(loader, "prepare_checkpoint_load", forbid_checkpoint_read)
    extra = (
        {"expected_step": 0, "expected_tokens_seen": 0}
        if loader is progress_trainer else {}
    )

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="unsupported restore-contract data",
    ):
        loader.load_trainer_checkpoint(
            tmp_path / "must-not-open",
            model=target.model,
            trainer=target,
            restore_rng=False,
            **extra,
        )

    assert hook_calls == []
    assert checkpoint_reads == []
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)

@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "restore_rng",
    [False, True],
    ids=["opt-out", "exact-rng"],
)
def test_final_authority_lookup_cannot_mutate_config_in_place_before_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    restore_rng: bool,
) -> None:
    source = _source()
    checkpoint = tmp_path / "config-in-place-drift-дані з пробілами"
    core.save_checkpoint(
        checkpoint,
        model=source.model,
        trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(checkpoint)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    initial_weights = target.model.weight.detach().clone()
    initial_max_steps = target.config.max_steps

    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    policy_before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )

    original_preflight = loader._preflight_trainer_state
    original_bind = loader._bind_model_state_loader
    original_coverage = Trainer.__dict__["_require_optimizer_parameter_coverage"]
    preflight_calls = 0
    model_applications: list[bool] = []

    class EffectfulCoverage:
        def __get__(self, instance: Any, owner: type[Any]) -> Any:
            if instance is target:
                object.__setattr__(
                    target.config,
                    "max_steps",
                    target.config.max_steps + 1,
                )
            return original_coverage.__get__(instance, owner)

    def bind_tracked_model_loader(model: Any, strict: bool):
        apply = original_bind(model, strict)

        def tracked_apply(materialized: Any) -> Any:
            model_applications.append(True)
            return apply(materialized)

        return tracked_apply

    def preflight_then_arm(*args: Any, **kwargs: Any) -> Any:
        nonlocal preflight_calls
        result = original_preflight(*args, **kwargs)
        preflight_calls += 1
        if preflight_calls == 3:
            monkeypatch.setattr(
                Trainer,
                "_require_optimizer_parameter_coverage",
                EffectfulCoverage(),
            )
        return result

    monkeypatch.setattr(loader, "_bind_model_state_loader", bind_tracked_model_loader)
    monkeypatch.setattr(loader, "_preflight_trainer_state", preflight_then_arm)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="target config changed during preflight",
    ):
        loader.load_trainer_checkpoint(
            checkpoint,
            model=target.model,
            trainer=target,
            strict_model=False,
            restore_rng=restore_rng,
            **extra,
        )

    assert preflight_calls == 3
    assert model_applications == []
    assert target.config.max_steps == initial_max_steps + 1
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)

    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)
    assert (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    ) == policy_before

@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "restore_rng",
    [False, True],
    ids=["opt-out", "exact-rng"],
)
def test_final_authority_lookup_cannot_rebind_trainer_device_before_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    restore_rng: bool,
) -> None:
    source = _source()
    checkpoint = tmp_path / "device-rebind-дані з пробілами"
    core.save_checkpoint(
        checkpoint,
        model=source.model,
        trainer_state=asdict(source.state_dict()),
        identity=_identity(),
    )
    core.verify_checkpoint(checkpoint)

    target = Trainer(_TinyLogits(), source.config, device="cpu")
    original_device = target.device
    initial_weights = target.model.weight.detach().clone()

    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    policy_before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )

    original_preflight = loader._preflight_trainer_state
    original_bind = loader._bind_model_state_loader
    original_coverage = Trainer.__dict__["_require_optimizer_parameter_coverage"]
    preflight_calls = 0
    model_applications: list[bool] = []

    class EffectfulCoverage:
        def __get__(self, instance: Any, owner: type[Any]) -> Any:
            if instance is target:
                target.device = torch.device("meta")
            return original_coverage.__get__(instance, owner)

    def bind_tracked_model_loader(model: Any, strict: bool):
        apply = original_bind(model, strict)

        def tracked_apply(materialized: Any) -> Any:
            model_applications.append(True)
            return apply(materialized)

        return tracked_apply

    def preflight_then_arm(*args: Any, **kwargs: Any) -> Any:
        nonlocal preflight_calls
        result = original_preflight(*args, **kwargs)
        preflight_calls += 1
        if preflight_calls == 3:
            monkeypatch.setattr(
                Trainer,
                "_require_optimizer_parameter_coverage",
                EffectfulCoverage(),
            )
        return result

    monkeypatch.setattr(loader, "_bind_model_state_loader", bind_tracked_model_loader)
    monkeypatch.setattr(loader, "_preflight_trainer_state", preflight_then_arm)
    extra = (
        {"expected_step": 1, "expected_tokens_seen": 2}
        if loader is progress_trainer else {}
    )

    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="device binding changed during checkpoint restore",
    ):
        loader.load_trainer_checkpoint(
            checkpoint,
            model=target.model,
            trainer=target,
            strict_model=False,
            restore_rng=restore_rng,
            **extra,
        )

    assert preflight_calls == 3
    assert model_applications == []
    assert target.device.type == "meta"
    assert target.device != original_device
    assert (target.micro_step, target.optimizer_step, target.tokens_seen) == (0, 0, 0)
    torch.testing.assert_close(target.model.weight, initial_weights, rtol=0, atol=0)

    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    np.testing.assert_array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)
    assert (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    ) == policy_before

