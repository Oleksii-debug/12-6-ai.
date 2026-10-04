"""CPU-simulated CUDA rollback faults; does not qualify physical CUDA.

D05's per-device fallback must continue after a CUDA batch setter or one
individual device setter fails. No GPU or training data is required.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from twelve_six.checkpoint import core, trainer_adapter


@pytest.mark.parametrize(
    "failed_family",
    ["cuda_batch", "cuda_0", "cuda_1", "torch_cpu", "python", "numpy"],
)
def test_partial_cuda_rollback_recovers_independent_devices_and_streams(
    monkeypatch: pytest.MonkeyPatch, failed_family: str,
) -> None:
    """Exercise the real helper with fake CUDA setters, not fake RNG logic."""

    original = core.capture_rng_state()
    original_enabled = torch.are_deterministic_algorithms_enabled()
    original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        ambient = core.capture_rng_state()
        cuda_states = [
            torch.Generator(device="cpu").manual_seed(seed).get_state()
            for seed in (3111, 3222)
        ]
        ambient["torch"]["cuda"] = cuda_states
        fake_devices = {
            index: torch.Generator(device="cpu").manual_seed(4000 + index).get_state()
            for index in range(2)
        }
        expected_python = random.Random()
        expected_python.setstate(ambient["python"])
        expected_numpy = np.random.RandomState()
        expected_numpy.set_state(ambient["numpy"])
        expected_torch = torch.Generator(device="cpu")
        expected_torch.set_state(ambient["torch"]["cpu"])
        expected_next = (
            expected_python.random(),
            expected_numpy.random_sample(),
            torch.rand((), generator=expected_torch).item(),
        )
        random.random()
        np.random.random_sample()
        torch.rand(())
        original_error = RuntimeError("original failed checkpoint application")
        device_attempts: list[int] = []

        def failed_batch(states: list[torch.Tensor]) -> None:
            # Model a partially applied CUDA batch restore that raises.
            fake_devices[0] = states[0].clone()
            raise OSError("injected CUDA batch setter failure")

        def restore_device(state: torch.Tensor, *, device: int) -> None:
            device_attempts.append(device)
            if failed_family == f"cuda_{device}":
                raise OSError(f"injected CUDA device {device} setter failure")
            fake_devices[device] = state.clone()

        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda, "is_available", lambda: True)
            patch.setattr(torch.cuda, "device_count", lambda: 2)
            patch.setattr(torch.cuda, "set_rng_state_all", failed_batch)
            patch.setattr(torch.cuda, "set_rng_state", restore_device)
            if failed_family == "torch_cpu":
                patch.setattr(
                    torch, "set_rng_state",
                    lambda _state: (_ for _ in ()).throw(
                        OSError("injected CPU torch RNG setter failure")
                    ),
                )
            if failed_family == "python":
                patch.setattr(
                    random, "setstate",
                    lambda _state: (_ for _ in ()).throw(
                        OSError("injected Python RNG setter failure")
                    ),
                )
            if failed_family == "numpy":
                patch.setattr(
                    np.random, "set_state",
                    lambda _state: (_ for _ in ()).throw(
                        OSError("injected NumPy RNG setter failure")
                    ),
                )
            trainer_adapter._restore_ambient_rng_after_failed_apply(
                ambient, original_error,
            )

        assert device_attempts == [0, 1], "both CUDA devices need individual recovery"
        assert str(original_error) == "original failed checkpoint application"
        assert any("rollback" in note.lower() for note in original_error.__notes__)
        for index, wanted in enumerate(cuda_states):
            if failed_family == f"cuda_{index}":
                # A failed device may have recovered during the batch attempt.
                # Still require individual retry and an explicit diagnostic.
                assert any(
                    f"device {index}" in note for note in original_error.__notes__
                )
            else:
                assert torch.equal(fake_devices[index], wanted)
        actual_next = (random.random(), np.random.random_sample(), torch.rand(()).item())
        for index, stream in enumerate(("python", "numpy", "torch_cpu")):
            if failed_family != stream:
                assert actual_next[index] == expected_next[index], stream
    finally:
        random.setstate(original["python"])
        np.random.set_state(original["numpy"])
        torch.set_rng_state(original["torch"]["cpu"])
        torch.use_deterministic_algorithms(
            original_enabled, warn_only=original_warn_only,
        )
