"""Fail-closed D05 ambient RNG rollback must attempt independent stream families.

Non-owning regression handoff for PR #2628. These checks intentionally fail until
D05 continues rollback after an individual stream restore raises. They do not
assert CUDA hardware qualification or grant training/data authority.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from twelve_six.checkpoint import core, trainer_adapter


@pytest.mark.parametrize("failed_family", ["python", "numpy"])
def test_ambient_rng_rollback_continues_after_single_stream_failure(
    monkeypatch: pytest.MonkeyPatch, failed_family: str,
) -> None:
    """Other RNG streams must recover even if one setter fails once."""

    initial = core.capture_rng_state()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        random.seed(1701)
        np.random.seed(1702)
        torch.manual_seed(1703)
        ambient = core.capture_rng_state()
        python_probe = random.Random()
        python_probe.setstate(ambient["python"])
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(ambient["numpy"])
        torch_probe = torch.Generator(device="cpu")
        torch_probe.set_state(ambient["torch"]["cpu"])
        expected_python = python_probe.random()
        expected_numpy = numpy_probe.random_sample()
        expected_torch = torch.rand((), generator=torch_probe).item()

        # Simulate a partially applied model/trainer loader that consumes every
        # process-global stream before failing. Only the rollback is under test.
        random.seed(2701)
        np.random.seed(2702)
        torch.manual_seed(2703)
        primary = RuntimeError("original checkpoint apply failure")
        with monkeypatch.context() as patch:
            if failed_family == "python":
                setter = random.setstate

                def fail_python(state: object) -> None:
                    patch.setattr(random, "setstate", setter)
                    raise OSError("injected Python stream rollback failure")

                patch.setattr(random, "setstate", fail_python)
            else:
                setter = np.random.set_state

                def fail_numpy(state: object) -> None:
                    patch.setattr(np.random, "set_state", setter)
                    raise OSError("injected NumPy stream rollback failure")

                patch.setattr(np.random, "set_state", fail_numpy)
            trainer_adapter._restore_ambient_rng_after_failed_apply(ambient, primary)

        notes = getattr(primary, "__notes__", ())
        assert any("rollback" in note.lower() for note in notes)
        assert str(primary) == "original checkpoint apply failure"
        if failed_family != "python":
            assert random.random() == expected_python
        if failed_family != "numpy":
            assert np.random.random_sample() == expected_numpy
        # This assertion is RED on #2628 HEAD ef7d035 when either earlier
        # setter fails: core.restore_rng_state aborts before reaching torch.
        assert torch.rand(()).item() == expected_torch
    finally:
        core.restore_rng_state(initial)
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
