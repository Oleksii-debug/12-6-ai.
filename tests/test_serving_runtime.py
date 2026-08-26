from __future__ import annotations

import threading
import time
from collections.abc import Sequence

import pytest

from twelve_six.inference.serving_runtime import (
    ServingOverloadedError,
    ServingRequestTimeoutError,
    ServingRuntime,
    ServingRuntimeError,
    ServingUnavailableError,
)


class BlockingBackend:
    eos_token_id = None
    max_context_tokens = 8

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def encode(self, text: str) -> list[int]:
        return [0] if text else []

    def decode(self, token_ids: Sequence[int]) -> str:
        return "A" * len(token_ids)

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        del input_ids
        self.calls += 1
        self.started.set()
        assert self.release.wait(timeout=5)
        return [0.0, 10.0]

    def diagnostics(self) -> dict[str, object]:
        return {
            "backend": "blocking-test",
            "checkpoint_id": "a" * 64,
            "git_sha": "b" * 40,
            "secret_prompt": "must-not-escape",
        }


class ImmediateBackend(BlockingBackend):
    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        del input_ids
        self.calls += 1
        return [0.0, 10.0]


_PAYLOAD = {"prompt": "x", "temperature": 0, "max_tokens": 1}


def _submit(
    runtime: ServingRuntime,
    results: list[dict[str, object]],
    errors: list[Exception],
    *,
    timeout_seconds: float = 2.0,
) -> None:
    try:
        results.append(
            runtime.submit(
                _PAYLOAD,
                response_id="cmpl-test",
                created=0,
                timeout_seconds=timeout_seconds,
            )
        )
    except (TypeError, ValueError, RuntimeError, OSError) as exc:
        errors.append(exc)


def _wait_for_queue_depth(runtime: ServingRuntime, expected: int) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if runtime.status()["queue_depth"] == expected:
            return
        time.sleep(0.001)
    raise AssertionError(f"queue depth did not reach {expected}")


def test_bounded_queue_serializes_model_execution_and_rejects_overload() -> None:
    backend = BlockingBackend()
    runtime = ServingRuntime(backend, model_name="s0-test", max_queue_depth=1)
    results: list[dict[str, object]] = []
    errors: list[Exception] = []

    first = threading.Thread(target=_submit, args=(runtime, results, errors))
    first.start()
    assert backend.started.wait(timeout=2)

    second = threading.Thread(target=_submit, args=(runtime, results, errors))
    second.start()
    _wait_for_queue_depth(runtime, 1)

    with pytest.raises(ServingOverloadedError, match="queue is full"):
        runtime.submit(
            _PAYLOAD,
            response_id="cmpl-overload",
            created=0,
            timeout_seconds=1,
        )

    backend.release.set()
    first.join(timeout=3)
    second.join(timeout=3)
    runtime.close()

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert len(results) == 2
    assert backend.calls == 2
    status = runtime.status()
    assert status["state"] == "stopped"
    assert status["queue_high_watermark"] == 1
    assert status["rejected_overload"] == 1
    assert status["completed_requests"] == 2


def test_timeout_cancels_queued_work_before_model_execution() -> None:
    backend = BlockingBackend()
    runtime = ServingRuntime(backend, model_name="s0-test", max_queue_depth=1)
    results: list[dict[str, object]] = []
    errors: list[Exception] = []

    first = threading.Thread(target=_submit, args=(runtime, results, errors))
    first.start()
    assert backend.started.wait(timeout=2)

    with pytest.raises(ServingRequestTimeoutError) as caught:
        runtime.submit(
            _PAYLOAD,
            response_id="cmpl-queued-timeout",
            created=0,
            timeout_seconds=0.05,
        )
    assert caught.value.execution_started is False

    backend.release.set()
    first.join(timeout=3)
    runtime.close()

    assert errors == []
    assert backend.calls == 1
    status = runtime.status()
    assert status["timed_out_before_start"] == 1
    assert status["cancelled_before_start"] == 1


def test_timeout_does_not_claim_preemption_after_model_execution_started() -> None:
    backend = BlockingBackend()
    runtime = ServingRuntime(backend, model_name="s0-test", max_queue_depth=0)
    results: list[dict[str, object]] = []
    errors: list[Exception] = []

    submitter = threading.Thread(
        target=_submit,
        args=(runtime, results, errors),
        kwargs={"timeout_seconds": 0.2},
    )
    submitter.start()
    assert backend.started.wait(timeout=2)
    submitter.join(timeout=1)

    assert not submitter.is_alive()
    assert results == []
    assert len(errors) == 1
    assert isinstance(errors[0], ServingRequestTimeoutError)
    assert errors[0].execution_started is True

    backend.release.set()
    runtime.close()
    status = runtime.status()
    assert status["timed_out_after_start"] == 1
    assert status["completed_requests"] == 1


def test_loading_lifecycle_and_identity_are_fail_closed_and_privacy_safe() -> None:
    runtime = ServingRuntime(None, model_name="s0-test", max_queue_depth=0)
    assert runtime.state == "loading"
    assert runtime.ready is False

    with pytest.raises(ServingUnavailableError, match="not ready"):
        runtime.submit(
            _PAYLOAD,
            response_id="cmpl-not-ready",
            created=0,
            timeout_seconds=1,
        )

    backend = ImmediateBackend()
    runtime.install_backend(backend)
    assert runtime.ready is True
    identity = runtime.model_identity()
    assert identity["backend"] == "blocking-test"
    assert identity["checkpoint_id"] == "a" * 64
    assert "secret_prompt" not in identity

    with pytest.raises(ServingRuntimeError, match="installation is allowed only"):
        runtime.install_backend(backend)

    runtime.begin_draining()
    assert runtime.ready is False
    with pytest.raises(ServingUnavailableError, match="not ready"):
        runtime.submit(
            _PAYLOAD,
            response_id="cmpl-draining",
            created=0,
            timeout_seconds=1,
        )
    runtime.close()
    assert runtime.state == "stopped"


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_completion_timeout_must_be_finite_positive(value: object) -> None:
    runtime = ServingRuntime(ImmediateBackend(), model_name="s0-test")
    with pytest.raises(ValueError, match="completion_timeout_seconds"):
        runtime.submit(
            _PAYLOAD,
            response_id="cmpl-invalid-timeout",
            created=0,
            timeout_seconds=value,  # type: ignore[arg-type]
        )
    runtime.close()
