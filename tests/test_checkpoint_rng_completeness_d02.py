"""Integrity-valid incomplete-RNG rejection for the D05/D02 restore boundary.

The independent regression cases were adopted into PR #2628 alongside the
canonical same-lineage fail-closed repair; execution still requires CI.
No real corpus, tokenizer fitting or trained-model evidence is used.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    core,
    progress_trainer,
    trainer_adapter,
)
from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import Trainer


@pytest.fixture
def checkpoint_identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "d02-incomplete-rng-regression", "width": 3},
        parameter_count=12,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 10},
        seed=703,
        precision="fp32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "AdamW"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _seal_checkpoint(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity: CheckpointIdentity,
    missing: str | None,
) -> tuple[dict[str, Any], TrainerConfig]:
    config = TrainerConfig(max_steps=10, seed=703)
    source_model = torch.nn.Linear(3, 3)
    source = Trainer(source_model, config)
    captured = core.capture_rng_state()
    deliberately_incomplete = dict(captured)
    if missing == "cuda":
        deliberately_incomplete["torch"] = dict(captured["torch"])
        deliberately_incomplete["torch"].pop("cuda")
    elif missing == "cuda_environment":
        deliberately_incomplete["torch"] = dict(captured["torch"])
        deliberately_incomplete["torch"].pop("cuda_environment")
    elif missing == "warn_only":
        deliberately_incomplete["torch"] = dict(captured["torch"])
        deliberately_incomplete["torch"].pop("deterministic_warn_only")
    elif missing in {
        "default_dtype",
        "matmul_precision",
        "cudnn_tf32",
        "cudnn_enabled",
        "cudnn_deterministic",
        "cudnn_benchmark",
    }:
        deliberately_incomplete["torch"] = dict(captured["torch"])
        field = {
            "default_dtype": "default_dtype",
            "matmul_precision": "float32_matmul_precision",
            "cudnn_tf32": "cudnn_allow_tf32",
            "cudnn_enabled": "cudnn_enabled",
            "cudnn_deterministic": "cudnn_deterministic",
            "cudnn_benchmark": "cudnn_benchmark",
        }[missing]
        deliberately_incomplete["torch"].pop(field)
    elif missing is not None:
        deliberately_incomplete.pop(missing)
    # Produce a completely re-signed checkpoint through the production writer.
    # There is no checksum tampering, mocked decoder or unverified file input.
    with monkeypatch.context() as patch:
        patch.setattr(core, "capture_rng_state", lambda: deliberately_incomplete)
        trainer_adapter.save_trainer_checkpoint(
            path, model=source_model, trainer=source, identity=identity,
        )
    core.verify_checkpoint(path)
    _, decoded = core._decode_verified_state(core.prepare_checkpoint_load(path))
    if missing == "cuda":
        assert "cuda" not in decoded["rng"]["torch"]
    elif missing == "cuda_environment":
        assert "cuda_environment" not in decoded["rng"]["torch"]
    elif missing == "warn_only":
        assert "deterministic_warn_only" not in decoded["rng"]["torch"]
    elif missing in {
        "default_dtype",
        "matmul_precision",
        "cudnn_tf32",
        "cudnn_enabled",
        "cudnn_deterministic",
        "cudnn_benchmark",
    }:
        field = {
            "default_dtype": "default_dtype",
            "matmul_precision": "float32_matmul_precision",
            "cudnn_tf32": "cudnn_allow_tf32",
            "cudnn_enabled": "cudnn_enabled",
            "cudnn_deterministic": "cudnn_deterministic",
            "cudnn_benchmark": "cudnn_benchmark",
        }[missing]
        assert field not in decoded["rng"]["torch"]
    elif missing is not None:
        assert missing not in decoded["rng"]
    return captured, config


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "missing",
    [
        "python",
        "numpy",
        "cuda",
        "cuda_environment",
        "default_dtype",
        "matmul_precision",
        "cudnn_tf32",
        "cudnn_enabled",
        "cudnn_deterministic",
        "cudnn_benchmark",
    ],
)
def test_incomplete_but_verified_rng_rejected_before_model_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
    missing: str,
) -> None:
    checkpoint = tmp_path / "sealed-incomplete-rng"
    _, config = _seal_checkpoint(checkpoint, monkeypatch, checkpoint_identity, missing)
    model = torch.nn.Linear(3, 3)
    trainer = Trainer(model, config)
    before = [p.detach().clone() for p in model.parameters()]

    def forbidden_materialization(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("missing RNG stream must reject before model materialization")

    monkeypatch.setattr(loader, "_prepare_model_weights", forbidden_materialization)
    with pytest.raises(CheckpointCompatibilityError, match="RNG|rng|Python|NumPy"):
        loader.load_trainer_checkpoint(
            checkpoint, model=model, trainer=trainer, restore_rng=True,
        )
    for current, initial in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(current.detach(), initial, rtol=0, atol=0)
    assert trainer.optimizer_step == 0
    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "missing",
    [
        "python",
        "numpy",
        "cuda",
        "cuda_environment",
        "default_dtype",
        "matmul_precision",
        "cudnn_tf32",
        "cudnn_enabled",
        "cudnn_deterministic",
        "cudnn_benchmark",
    ],
)
def test_explicit_rng_opt_out_retains_existing_checkpoint_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
    missing: str,
) -> None:
    checkpoint = tmp_path / "sealed-opt-out"
    _, config = _seal_checkpoint(checkpoint, monkeypatch, checkpoint_identity, missing)
    model = torch.nn.Linear(3, 3)
    trainer = Trainer(model, config)
    result = loader.load_trainer_checkpoint(
        checkpoint, model=model, trainer=trainer, restore_rng=False,
    )
    assert result.manifest["identity"]["step"] == 0
    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
def test_rng_opt_out_rejects_preflight_cuda_environment_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
) -> None:
    key = "CUBLAS_WORKSPACE_CONFIG"
    ambient = core.capture_rng_state()
    initial = ambient["torch"]["cuda_environment"][key]
    replacement = ":16:8" if initial != ":16:8" else ":4096:8"
    checkpoint = tmp_path / "opt-out-preflight-environment-drift"
    config = TrainerConfig(max_steps=10, seed=703)
    source_model = torch.nn.Linear(3, 3)
    source = Trainer(source_model, config)
    trainer_adapter.save_trainer_checkpoint(
        checkpoint,
        model=source_model,
        trainer=source,
        identity=checkpoint_identity,
    )

    target_model = torch.nn.Linear(3, 3)
    target = Trainer(target_model, config)
    original_preflight = trainer_adapter._preflight_optimizer_state

    def drifting_preflight(*args: Any, **kwargs: Any) -> Any:
        result = original_preflight(*args, **kwargs)
        monkeypatch.setenv(key, replacement)
        return result

    def forbidden_materialization(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("ambient drift must reject before model materialization")

    monkeypatch.setattr(
        trainer_adapter,
        "_preflight_optimizer_state",
        drifting_preflight,
    )
    monkeypatch.setattr(loader, "_prepare_model_weights", forbidden_materialization)

    with pytest.raises(
        CheckpointCompatibilityError,
        match="live ambient torch/CUDA process state changed",
    ):
        loader.load_trainer_checkpoint(
            checkpoint,
            model=target_model,
            trainer=target,
            restore_rng=False,
        )

    assert target._failure_reason == "checkpoint_preflight_rng_rollback_failed"
    assert target._update_incomplete is True


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    ("phase", "mutation"),
    [
        ("model_apply", "default_dtype"),
        ("model_apply", "cudnn_benchmark"),
        ("model_apply", "cuda_environment"),
        ("trainer_apply", "cuda_environment"),
    ],
)
def test_rng_opt_out_rejects_application_ambient_process_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
    phase: str,
    mutation: str,
) -> None:
    key = "CUBLAS_WORKSPACE_CONFIG"
    checkpoint = tmp_path / f"opt-out-{phase}-{mutation}"
    config = TrainerConfig(max_steps=10, seed=703)

    def mutate_process_state() -> None:
        if mutation == "default_dtype":
            replacement = (
                torch.float64
                if torch.get_default_dtype() is not torch.float64
                else torch.float32
            )
            torch.set_default_dtype(replacement)
        elif mutation == "cudnn_benchmark":
            torch.backends.cudnn.benchmark = not torch.backends.cudnn.benchmark
        elif mutation == "cuda_environment":
            current = core.capture_rng_state()["torch"]["cuda_environment"][key]
            replacement = ":16:8" if current != ":16:8" else ":4096:8"
            monkeypatch.setenv(key, replacement)
        else:
            raise AssertionError(f"unknown mutation: {mutation}")

    class DriftLinear(torch.nn.Linear):
        def load_state_dict(self, state_dict: Any, *args: Any, **kwargs: Any):
            result = super().load_state_dict(state_dict, *args, **kwargs)
            if phase == "model_apply":
                mutate_process_state()
            return result

    source_model = DriftLinear(3, 3)
    source = Trainer(source_model, config)
    trainer_adapter.save_trainer_checkpoint(
        checkpoint,
        model=source_model,
        trainer=source,
        identity=checkpoint_identity,
    )

    target_model = DriftLinear(3, 3)
    target = Trainer(target_model, config)
    ambient = core.capture_rng_state()

    if phase == "trainer_apply":
        original_bind = loader._bind_trainer_state_loader

        def bind_drifting_loader(trainer: Any) -> Any:
            apply_state = original_bind(trainer)

            def drifting_apply(state: Any) -> Any:
                result = apply_state(state)
                mutate_process_state()
                return result

            return drifting_apply

        monkeypatch.setattr(
            loader,
            "_bind_trainer_state_loader",
            bind_drifting_loader,
        )

    with pytest.raises(
        CheckpointCompatibilityError,
        match="live ambient torch/CUDA process state changed",
    ):
        loader.load_trainer_checkpoint(
            checkpoint,
            model=target_model,
            trainer=target,
            restore_rng=False,
        )

    assert target._failure_reason == "checkpoint_restore_apply_failed"
    assert target._update_incomplete is True
    if mutation == "default_dtype":
        assert str(torch.get_default_dtype()) == ambient["torch"]["default_dtype"]
    elif mutation == "cudnn_benchmark":
        assert (
            bool(torch.backends.cudnn.benchmark)
            == ambient["torch"]["cudnn_benchmark"]
        )


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "replay"])
@pytest.mark.parametrize(
    ("phase", "mutation"),
    [
        ("preapply", "grad_enabled"),
        ("model_apply", "grad_enabled"),
        ("trainer_apply", "grad_enabled"),
        ("model_apply", "inference_mode"),
    ],
)
def test_public_restore_rejects_autograd_execution_mode_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
    restore_rng: bool,
    phase: str,
    mutation: str,
) -> None:
    checkpoint = tmp_path / f"autograd-drift-{phase}-{mutation}-{restore_rng}"
    config = TrainerConfig(max_steps=10, seed=703)
    leaked_contexts: list[Any] = []

    def mutate_execution_mode() -> None:
        if mutation == "grad_enabled":
            torch.set_grad_enabled(not torch.is_grad_enabled())
        elif mutation == "inference_mode":
            context = torch.inference_mode(
                not torch.is_inference_mode_enabled()
            )
            context.__enter__()
            leaked_contexts.append(context)
        else:
            raise AssertionError(f"unknown execution-mode mutation: {mutation}")

    class DriftLinear(torch.nn.Linear):
        def load_state_dict(self, state_dict: Any, *args: Any, **kwargs: Any):
            result = super().load_state_dict(state_dict, *args, **kwargs)
            if phase == "model_apply":
                mutate_execution_mode()
            return result

    source_model = DriftLinear(3, 3)
    source = Trainer(source_model, config)
    trainer_adapter.save_trainer_checkpoint(
        checkpoint,
        model=source_model,
        trainer=source,
        identity=checkpoint_identity,
    )

    target_model = DriftLinear(3, 3)
    target = Trainer(target_model, config)
    entry_grad_enabled = torch.is_grad_enabled()

    if phase == "preapply":
        original_prepare = loader._prepare_model_weights

        def drifting_prepare(*args: Any, **kwargs: Any) -> Any:
            result = original_prepare(*args, **kwargs)
            mutate_execution_mode()
            return result

        monkeypatch.setattr(loader, "_prepare_model_weights", drifting_prepare)
    elif phase == "trainer_apply":
        original_bind = loader._bind_trainer_state_loader

        def bind_drifting_loader(trainer: Any) -> Any:
            apply_state = original_bind(trainer)

            def drifting_apply(state: Any) -> Any:
                result = apply_state(state)
                mutate_execution_mode()
                return result

            return drifting_apply

        monkeypatch.setattr(
            loader,
            "_bind_trainer_state_loader",
            bind_drifting_loader,
        )

    try:
        with pytest.raises(
            CheckpointCompatibilityError,
            match="live torch autograd/inference mode changed",
        ):
            loader.load_trainer_checkpoint(
                checkpoint,
                model=target_model,
                trainer=target,
                restore_rng=restore_rng,
            )
    finally:
        for context in reversed(leaked_contexts):
            context.__exit__(None, None, None)
        torch.set_grad_enabled(entry_grad_enabled)

    expected_reason = (
        "checkpoint_preapply_rng_rollback_failed"
        if phase == "preapply"
        else "checkpoint_restore_apply_failed"
    )
    assert target._failure_reason == expected_reason
    assert target._update_incomplete is True


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
@pytest.mark.parametrize("restore_rng", [False, True], ids=["opt-out", "replay"])
def test_public_restore_preserves_caller_no_grad_mode(
    tmp_path: Path,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
    restore_rng: bool,
) -> None:
    checkpoint = tmp_path / f"caller-no-grad-{restore_rng}"
    config = TrainerConfig(max_steps=10, seed=703)
    source_model = torch.nn.Linear(3, 3)
    source = Trainer(source_model, config)
    trainer_adapter.save_trainer_checkpoint(
        checkpoint,
        model=source_model,
        trainer=source,
        identity=checkpoint_identity,
    )
    target_model = torch.nn.Linear(3, 3)
    target = Trainer(target_model, config)

    with torch.no_grad():
        expected = (
            torch.is_grad_enabled(),
            torch.is_inference_mode_enabled(),
        )
        result = loader.load_trainer_checkpoint(
            checkpoint,
            model=target_model,
            trainer=target,
            restore_rng=restore_rng,
        )
        assert (
            torch.is_grad_enabled(),
            torch.is_inference_mode_enabled(),
        ) == expected

    assert result.manifest["identity"]["step"] == 0
    assert target._failure_reason is None
    assert target._update_incomplete is False


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
def test_complete_verified_checkpoint_replays_python_numpy_and_torch_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
) -> None:
    ambient = core.capture_rng_state()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        checkpoint = tmp_path / "sealed-complete"
        saved, config = _seal_checkpoint(checkpoint, monkeypatch, checkpoint_identity, None)
        py_probe = random.Random()
        py_probe.setstate(saved["python"])
        np_probe = np.random.RandomState()
        np_probe.set_state(saved["numpy"])
        torch_probe = torch.Generator(device="cpu")
        torch_probe.set_state(saved["torch"]["cpu"])
        expected = (
            py_probe.random(), np_probe.random_sample(),
            torch.rand((), generator=torch_probe).item(),
        )
        model = torch.nn.Linear(3, 3)
        trainer = Trainer(model, config)
        random.seed(1199)
        np.random.seed(1199)
        torch.manual_seed(1199)
        loader.load_trainer_checkpoint(
            checkpoint, model=model, trainer=trainer, restore_rng=True,
        )
        actual = (random.random(), np.random.random_sample(), torch.rand(()).item())
        assert actual == expected
    finally:
        core.restore_rng_state(ambient)
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    ("environment_key", "checkpoint_value", "live_value"),
    [
        ("CUBLAS_WORKSPACE_CONFIG", ":16:8", ":4096:2"),
        ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0", "1"),
        ("NVIDIA_TF32_OVERRIDE", "0", None),
    ],
)
def test_cuda_process_environment_mismatch_rejected_before_model_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
    environment_key: str,
    checkpoint_value: str,
    live_value: str | None,
) -> None:
    checkpoint = tmp_path / "sealed-cuda-environment"
    with monkeypatch.context() as source_env:
        source_env.setenv(environment_key, checkpoint_value)
        config = TrainerConfig(max_steps=10, seed=703)
        source_model = torch.nn.Linear(3, 3)
        source = Trainer(source_model, config)
        trainer_adapter.save_trainer_checkpoint(
            checkpoint,
            model=source_model,
            trainer=source,
            identity=checkpoint_identity,
        )
    core.verify_checkpoint(checkpoint)

    with monkeypatch.context() as live_env:
        if live_value is None:
            live_env.delenv(environment_key, raising=False)
        else:
            live_env.setenv(environment_key, live_value)
        target_model = torch.nn.Linear(3, 3)
        target = Trainer(target_model, config)
        before = [parameter.detach().clone() for parameter in target_model.parameters()]

        def forbidden_materialization(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError(
                "environment mismatch must reject before model materialization"
            )

        live_env.setattr(loader, "_prepare_model_weights", forbidden_materialization)
        with pytest.raises(
            CheckpointCompatibilityError,
            match="CUDA process environment differs",
        ):
            loader.load_trainer_checkpoint(
                checkpoint,
                model=target_model,
                trainer=target,
                restore_rng=True,
            )

        for current, initial in zip(target_model.parameters(), before, strict=True):
            torch.testing.assert_close(current.detach(), initial, rtol=0, atol=0)
        assert target.optimizer_step == 0
        assert target._failure_reason is None
        assert target._update_incomplete is False


@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
def test_cuda_process_environment_mismatch_explicit_opt_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
) -> None:
    checkpoint = tmp_path / "sealed-cuda-environment-opt-out"
    with monkeypatch.context() as source_env:
        source_env.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        config = TrainerConfig(max_steps=10, seed=703)
        source_model = torch.nn.Linear(3, 3)
        source = Trainer(source_model, config)
        trainer_adapter.save_trainer_checkpoint(
            checkpoint,
            model=source_model,
            trainer=source,
            identity=checkpoint_identity,
        )

    with monkeypatch.context() as live_env:
        live_env.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:2")
        target_model = torch.nn.Linear(3, 3)
        target = Trainer(target_model, config)
        result = loader.load_trainer_checkpoint(
            checkpoint,
            model=target_model,
            trainer=target,
            restore_rng=False,
        )
        assert result.manifest["identity"]["step"] == 0
        assert target._failure_reason is None
        assert target._update_incomplete is False


@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "phase",
    ["preapply", "model_apply", "rng_replay"],
)
def test_cuda_process_environment_drift_during_restore_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
    phase: str,
) -> None:
    key = "CUBLAS_WORKSPACE_CONFIG"
    monkeypatch.delenv(key, raising=False)
    checkpoint = tmp_path / f"cuda-environment-drift-{phase}"
    config = TrainerConfig(max_steps=10, seed=703)

    class EnvironmentDriftLinear(torch.nn.Linear):
        def load_state_dict(self, state_dict: Any, *args: Any, **kwargs: Any):
            result = super().load_state_dict(state_dict, *args, **kwargs)
            if phase == "model_apply":
                monkeypatch.setenv(key, ":16:8")
            return result

    source_model = EnvironmentDriftLinear(3, 3)
    source = Trainer(source_model, config)
    trainer_adapter.save_trainer_checkpoint(
        checkpoint,
        model=source_model,
        trainer=source,
        identity=checkpoint_identity,
    )

    target_model = EnvironmentDriftLinear(3, 3)
    target = Trainer(target_model, config)

    if phase == "preapply":
        original_prepare = loader._prepare_model_weights

        def drifting_prepare(*args: Any, **kwargs: Any):
            result = original_prepare(*args, **kwargs)
            monkeypatch.setenv(key, ":16:8")
            return result

        monkeypatch.setattr(loader, "_prepare_model_weights", drifting_prepare)
    elif phase == "rng_replay":
        original_restore = loader.restore_rng_state

        def drifting_restore(state: Any):
            result = original_restore(state)
            monkeypatch.setenv(key, ":16:8")
            return result

        monkeypatch.setattr(loader, "restore_rng_state", drifting_restore)

    with pytest.raises(
        CheckpointCompatibilityError,
        match="CUDA process environment differs",
    ):
        loader.load_trainer_checkpoint(
            checkpoint,
            model=target_model,
            trainer=target,
            restore_rng=True,
        )

    assert target._update_incomplete is True
    if phase == "preapply":
        assert target._failure_reason == "checkpoint_preapply_process_environment_drift"
    else:
        assert target._failure_reason == "checkpoint_restore_apply_failed"


@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
@pytest.mark.parametrize(
    "phase",
    ["model_apply", "rng_replay"],
)
def test_numeric_policy_drift_during_restore_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
    phase: str,
) -> None:
    ambient = core.capture_rng_state()
    checkpoint = tmp_path / f"numeric-policy-drift-{phase}"
    config = TrainerConfig(max_steps=10, seed=703)

    class NumericDriftLinear(torch.nn.Linear):
        def load_state_dict(self, state_dict: Any, *args: Any, **kwargs: Any):
            result = super().load_state_dict(state_dict, *args, **kwargs)
            if phase == "model_apply":
                torch.backends.cudnn.benchmark = (
                    not torch.backends.cudnn.benchmark
                )
            return result

    try:
        source_model = NumericDriftLinear(3, 3)
        source = Trainer(source_model, config)
        trainer_adapter.save_trainer_checkpoint(
            checkpoint,
            model=source_model,
            trainer=source,
            identity=checkpoint_identity,
        )

        target_model = NumericDriftLinear(3, 3)
        target = Trainer(target_model, config)

        if phase == "rng_replay":
            original_restore = loader.restore_rng_state

            def drifting_restore(state: Any):
                result = original_restore(state)
                torch.backends.cudnn.benchmark = (
                    not torch.backends.cudnn.benchmark
                )
                return result

            monkeypatch.setattr(loader, "restore_rng_state", drifting_restore)

        with pytest.raises(
            CheckpointCompatibilityError,
            match="numeric policy differs",
        ):
            loader.load_trainer_checkpoint(
                checkpoint,
                model=target_model,
                trainer=target,
                restore_rng=True,
            )

        assert target._failure_reason == "checkpoint_restore_apply_failed"
        assert target._update_incomplete is True
    finally:
        core.restore_rng_state(ambient)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"CUBLAS_WORKSPACE_CONFIG": None}, "fields differ"),
        (
            {
                "CUBLAS_WORKSPACE_CONFIG": None,
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": None,
                "NVIDIA_TF32_OVERRIDE": 0,
            },
            "NVIDIA_TF32_OVERRIDE",
        ),
    ],
)
def test_invalid_cuda_process_environment_rejected_before_torch_rng_mutation(
    mutation: dict[str, Any],
    message: str,
) -> None:
    ambient = core.capture_rng_state()
    state = core.capture_rng_state()
    state["torch"] = dict(state["torch"])
    state["torch"]["cuda_environment"] = mutation
    cpu_before = torch.get_rng_state().clone()

    try:
        with pytest.raises(CheckpointCompatibilityError, match=message):
            core.restore_rng_state(state)
        torch.testing.assert_close(torch.get_rng_state(), cpu_before, rtol=0, atol=0)
    finally:
        core.restore_rng_state(ambient)


@pytest.mark.parametrize(
    "loader",
    [trainer_adapter, progress_trainer],
    ids=["adapter", "progress"],
)
def test_complete_checkpoint_replays_torch_numeric_policy(
    tmp_path: Path,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
) -> None:
    ambient = core.capture_rng_state()

    class PolicyObservingAdamW(torch.optim.AdamW):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.observed_numeric_policy: tuple[
                str, str, bool, bool, bool, bool
            ] | None = None

        def load_state_dict(self, state_dict: Any):
            self.observed_numeric_policy = (
                str(torch.get_default_dtype()),
                torch.get_float32_matmul_precision(),
                bool(torch.backends.cudnn.allow_tf32),
                bool(torch.backends.cudnn.enabled),
                bool(torch.backends.cudnn.deterministic),
                bool(torch.backends.cudnn.benchmark),
            )
            return super().load_state_dict(state_dict)

    try:
        torch.set_default_dtype(torch.float64)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        config = TrainerConfig(max_steps=10, seed=703)
        source_model = torch.nn.Linear(3, 3, dtype=torch.float32)
        source = Trainer(source_model, config)
        checkpoint = tmp_path / "sealed-numeric-policy"
        trainer_adapter.save_trainer_checkpoint(
            checkpoint,
            model=source_model,
            trainer=source,
            identity=checkpoint_identity,
        )
        core.verify_checkpoint(checkpoint)

        torch.set_default_dtype(torch.float32)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        target_model = torch.nn.Linear(3, 3, dtype=torch.float32)
        target_optimizer = PolicyObservingAdamW(
            target_model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        target = Trainer(
            target_model,
            config,
            optimizer=target_optimizer,
        )
        loader.load_trainer_checkpoint(
            checkpoint,
            model=target_model,
            trainer=target,
            restore_rng=True,
        )

        assert target_optimizer.observed_numeric_policy == (
            "torch.float64",
            "high",
            False,
            False,
            True,
            False,
        )
        assert torch.get_default_dtype() is torch.float64
        assert torch.get_float32_matmul_precision() == "high"
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cudnn.enabled is False
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert target._failure_reason is None
        assert target._update_incomplete is False
    finally:
        core.restore_rng_state(ambient)


def test_checkpoint_save_is_numeric_policy_neutral(
    tmp_path: Path,
    checkpoint_identity: CheckpointIdentity,
) -> None:
    ambient = core.capture_rng_state()

    class EffectfulLinear(torch.nn.Linear):
        def state_dict(self, *args: Any, **kwargs: Any):
            torch.set_default_dtype(torch.float32)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            return super().state_dict(*args, **kwargs)

    try:
        torch.set_default_dtype(torch.float64)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        model = EffectfulLinear(3, 3, dtype=torch.float32)
        checkpoint = tmp_path / "numeric-policy-neutral-save"

        def final_validator() -> None:
            torch.set_default_dtype(torch.float32)
            torch.set_float32_matmul_precision("medium")
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

        core.save_checkpoint(
            checkpoint,
            model=model,
            trainer_state={},
            identity=checkpoint_identity,
            post_rng_prepublish_validator=final_validator,
        )

        assert torch.get_default_dtype() is torch.float64
        assert torch.get_float32_matmul_precision() == "high"
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cudnn.enabled is False
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        verified = core.prepare_checkpoint_load(checkpoint)
        _, decoded = core._decode_verified_state(verified)
        assert decoded["rng"]["torch"]["default_dtype"] == "torch.float64"
        assert decoded["rng"]["torch"]["float32_matmul_precision"] == "high"
        assert decoded["rng"]["torch"]["cudnn_allow_tf32"] is False
        assert decoded["rng"]["torch"]["cudnn_enabled"] is False
        assert decoded["rng"]["torch"]["cudnn_deterministic"] is True
        assert decoded["rng"]["torch"]["cudnn_benchmark"] is False
    finally:
        core.restore_rng_state(ambient)


def test_generic_legacy_rng_restore_preserves_live_numeric_policy() -> None:
    ambient = core.capture_rng_state()
    try:
        legacy = core.capture_rng_state()
        legacy["torch"] = dict(legacy["torch"])
        legacy["torch"].pop("default_dtype")
        legacy["torch"].pop("float32_matmul_precision")
        legacy["torch"].pop("cudnn_allow_tf32")
        legacy["torch"].pop("cudnn_enabled")
        legacy["torch"].pop("cudnn_deterministic")
        legacy["torch"].pop("cudnn_benchmark")
        legacy["torch"].pop("cuda_environment")

        torch.set_default_dtype(torch.float64)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        core.restore_rng_state(legacy)

        assert torch.get_default_dtype() is torch.float64
        assert torch.get_float32_matmul_precision() == "high"
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cudnn.enabled is False
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        core.restore_rng_state(ambient)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("default_dtype", "torch.int32", "default_dtype"),
        ("float32_matmul_precision", "fastest", "float32_matmul_precision"),
        ("cudnn_allow_tf32", "yes", "cudnn_allow_tf32"),
        ("cudnn_enabled", 1, "cudnn_enabled"),
        ("cudnn_deterministic", 1, "cudnn_deterministic"),
        ("cudnn_benchmark", 1, "cudnn_benchmark"),
    ],
)
def test_invalid_numeric_policy_rejected_before_torch_rng_mutation(
    field: str,
    value: str,
    message: str,
) -> None:
    ambient = core.capture_rng_state()
    state = core.capture_rng_state()
    state["torch"] = dict(state["torch"])
    state["torch"][field] = value
    cpu_before = torch.get_rng_state().clone()
    dtype_before = torch.get_default_dtype()
    precision_before = torch.get_float32_matmul_precision()
    cudnn_tf32_before = torch.backends.cudnn.allow_tf32
    cudnn_enabled_before = torch.backends.cudnn.enabled
    cudnn_deterministic_before = torch.backends.cudnn.deterministic
    cudnn_benchmark_before = torch.backends.cudnn.benchmark

    try:
        with pytest.raises(CheckpointCompatibilityError, match=message):
            core.restore_rng_state(state)

        torch.testing.assert_close(torch.get_rng_state(), cpu_before, rtol=0, atol=0)
        assert torch.get_default_dtype() is dtype_before
        assert torch.get_float32_matmul_precision() == precision_before
        assert torch.backends.cudnn.allow_tf32 is cudnn_tf32_before
        assert torch.backends.cudnn.enabled is cudnn_enabled_before
        assert torch.backends.cudnn.deterministic is cudnn_deterministic_before
        assert torch.backends.cudnn.benchmark is cudnn_benchmark_before
    finally:
        core.restore_rng_state(ambient)


@pytest.mark.parametrize(
    "fault",
    ["default_dtype", "matmul_precision"],
)
def test_failed_apply_numeric_policy_rollback_isolates_setter_fault(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    ambient = core.capture_rng_state()
    try:
        torch.set_default_dtype(torch.float64)
        torch.set_float32_matmul_precision("high")
        expected = core.capture_rng_state()

        torch.set_default_dtype(torch.float32)
        torch.set_float32_matmul_precision("highest")
        primary = RuntimeError("injected checkpoint apply failure")

        def fail_default_dtype(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("injected default-dtype rollback failure")

        def fail_matmul_precision(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("injected matmul-precision rollback failure")

        with monkeypatch.context() as patch:
            if fault == "default_dtype":
                patch.setattr(
                    core,
                    "_restore_torch_default_dtype",
                    fail_default_dtype,
                )
            else:
                patch.setattr(
                    core,
                    "_restore_torch_matmul_precision",
                    fail_matmul_precision,
                )
            trainer_adapter._restore_ambient_rng_after_failed_apply(
                expected,
                primary,
            )

        notes = getattr(primary, "__notes__", ())
        if fault == "default_dtype":
            assert torch.get_default_dtype() is torch.float32
            assert torch.get_float32_matmul_precision() == "high"
            assert any("default-dtype rollback" in note for note in notes)
        else:
            assert torch.get_default_dtype() is torch.float64
            assert torch.get_float32_matmul_precision() == "highest"
            assert any("float32-matmul-precision rollback" in note for note in notes)
    finally:
        core.restore_rng_state(ambient)


def test_failed_apply_cudnn_tf32_rollback_fault_keeps_other_numeric_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ambient = core.capture_rng_state()
    try:
        torch.set_default_dtype(torch.float64)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = False
        expected = core.capture_rng_state()

        torch.set_default_dtype(torch.float32)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = True
        primary = RuntimeError("injected checkpoint apply failure")

        def fail_cudnn_tf32(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("injected cuDNN TF32 rollback failure")

        with monkeypatch.context() as patch:
            patch.setattr(
                core,
                "_restore_torch_cudnn_allow_tf32",
                fail_cudnn_tf32,
            )
            trainer_adapter._restore_ambient_rng_after_failed_apply(
                expected,
                primary,
            )

        assert torch.get_default_dtype() is torch.float64
        assert torch.get_float32_matmul_precision() == "high"
        assert torch.backends.cudnn.allow_tf32 is True
        assert any(
            "cuDNN TF32 rollback" in note
            for note in getattr(primary, "__notes__", ())
        )
    finally:
        core.restore_rng_state(ambient)


@pytest.mark.parametrize(
    ("fault", "note_fragment"),
    [
        ("enabled", "cuDNN enabled rollback"),
        ("deterministic", "cuDNN deterministic rollback"),
        ("benchmark", "cuDNN benchmark rollback"),
    ],
)
def test_failed_apply_cudnn_algorithm_policy_rollback_isolates_setter_fault(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    note_fragment: str,
) -> None:
    ambient = core.capture_rng_state()
    try:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        expected = core.capture_rng_state()

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        primary = RuntimeError("injected checkpoint apply failure")

        def fail_backend_policy(*_args: Any, **_kwargs: Any) -> None:
            raise OSError(f"injected cuDNN {fault} rollback failure")

        helper_name = {
            "enabled": "_restore_torch_cudnn_enabled",
            "deterministic": "_restore_torch_cudnn_deterministic",
            "benchmark": "_restore_torch_cudnn_benchmark",
        }[fault]
        with monkeypatch.context() as patch:
            patch.setattr(core, helper_name, fail_backend_policy)
            trainer_adapter._restore_ambient_rng_after_failed_apply(
                expected,
                primary,
            )

        if fault == "enabled":
            assert torch.backends.cudnn.enabled is True
        else:
            assert torch.backends.cudnn.enabled is False
        if fault == "deterministic":
            assert torch.backends.cudnn.deterministic is False
        else:
            assert torch.backends.cudnn.deterministic is True
        if fault == "benchmark":
            assert torch.backends.cudnn.benchmark is True
        else:
            assert torch.backends.cudnn.benchmark is False
        assert any(
            note_fragment in note
            for note in getattr(primary, "__notes__", ())
        )
    finally:
        core.restore_rng_state(ambient)


def _seal_warn_only_mismatch(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity: CheckpointIdentity,
) -> TrainerConfig:
    config = TrainerConfig(max_steps=10, seed=703)
    source_model = torch.nn.Linear(3, 3)
    source = Trainer(source_model, config)
    captured = core.capture_rng_state()
    mismatched = dict(captured)
    mismatched["torch"] = dict(captured["torch"])
    mismatched["torch"]["deterministic_warn_only"] = (
        not config.deterministic_warn_only
    )
    with monkeypatch.context() as patch:
        patch.setattr(core, "capture_rng_state", lambda: mismatched)
        trainer_adapter.save_trainer_checkpoint(
            path, model=source_model, trainer=source, identity=identity,
        )
    core.verify_checkpoint(path)
    return config


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
def test_warn_only_mismatch_rejected_before_model_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
) -> None:
    checkpoint = tmp_path / "sealed-warn-only-mismatch"
    config = _seal_warn_only_mismatch(checkpoint, monkeypatch, checkpoint_identity)
    model = torch.nn.Linear(3, 3)
    trainer = Trainer(model, config)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    def forbidden_materialization(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError(
            "warn-only mismatch must reject before model materialization"
        )

    monkeypatch.setattr(loader, "_prepare_model_weights", forbidden_materialization)
    with pytest.raises(
        CheckpointCompatibilityError, match="deterministic_warn_only",
    ):
        loader.load_trainer_checkpoint(
            checkpoint, model=model, trainer=trainer, restore_rng=True,
        )

    for current, initial in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(current.detach(), initial, rtol=0, atol=0)
    assert trainer.optimizer_step == 0
    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False


@pytest.mark.parametrize(
    "loader", [trainer_adapter, progress_trainer], ids=["adapter", "progress"],
)
def test_legacy_warn_only_omission_remains_canonical_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_identity: CheckpointIdentity,
    loader: Any,
) -> None:
    checkpoint = tmp_path / "sealed-legacy-warn-only"
    _, config = _seal_checkpoint(
        checkpoint, monkeypatch, checkpoint_identity, "warn_only",
    )
    model = torch.nn.Linear(3, 3)
    trainer = Trainer(model, config)

    result = loader.load_trainer_checkpoint(
        checkpoint, model=model, trainer=trainer, restore_rng=True,
    )

    assert result.manifest["identity"]["step"] == 0
    assert trainer._failure_reason is None
    assert trainer._update_incomplete is False
