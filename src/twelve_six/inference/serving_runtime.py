"""Bounded single-lane execution runtime for local raw-Base serving."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Mapping
from concurrent.futures import (
    CancelledError,
    Future,
    ThreadPoolExecutor,
)
from concurrent.futures import (
    TimeoutError as FutureTimeoutError,
)
from typing import Any, Self

from .contracts import InferenceBackend
from .openai_compat import completion_response

DEFAULT_MAX_QUEUE_DEPTH = 8
DEFAULT_COMPLETION_TIMEOUT_SECONDS = 120.0
_EXECUTION_LANES = 1
_IDENTITY_KEYS = frozenset(
    {
        "backend",
        "checkpoint_id",
        "git_sha",
        "model_spec_sha256",
        "parameter_count",
        "vocab_size",
        "max_context_tokens",
        "tokenizer_version",
        "tokenizer_config_sha256",
        "tokenizer_vocab_sha256",
        "dataset_manifest_sha256",
        "run_manifest_sha256",
        "step",
        "tokens_seen",
        "device",
    }
)


class ServingRuntimeError(RuntimeError):
    """Base class for stable local-serving lifecycle failures."""


class ServingUnavailableError(ServingRuntimeError):
    """The model execution lane is not accepting new work."""


class ServingOverloadedError(ServingRuntimeError):
    """The bounded execution admission capacity is exhausted."""


class ServingRequestTimeoutError(ServingRuntimeError):
    """The HTTP-facing wait budget expired before completion returned."""

    def __init__(self, *, execution_started: bool) -> None:
        super().__init__("completion exceeded configured server wait timeout")
        self.execution_started = execution_started


def _require_positive_finite(value: object, *, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be finite and > 0")
    return float(value)


class ServingRuntime:
    """Own one model execution lane behind bounded concurrent admission.

    Network handlers may run concurrently, but canonical first-party inference
    is intentionally serialized until backend concurrent-safety or a maintained
    batching/runtime backend is proved. ``max_queue_depth`` bounds requests
    waiting behind the one reserved execution lane.
    """

    def __init__(
        self,
        backend: InferenceBackend | None,
        *,
        model_name: str,
        max_queue_depth: int = DEFAULT_MAX_QUEUE_DEPTH,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must not be empty")
        if (
            not isinstance(max_queue_depth, int)
            or isinstance(max_queue_depth, bool)
            or max_queue_depth < 0
        ):
            raise ValueError("max_queue_depth must be a non-negative integer")

        self.model_name = model_name
        self.max_queue_depth = max_queue_depth
        self._backend = backend
        self._state = "ready" if backend is not None else "loading"
        self._loading_error_type: str | None = None
        self._identity = self._safe_backend_identity(backend)
        self._lock = threading.Lock()
        self._capacity = threading.BoundedSemaphore(
            value=_EXECUTION_LANES + max_queue_depth
        )
        self._executor = ThreadPoolExecutor(
            max_workers=_EXECUTION_LANES,
            thread_name_prefix="12-6-serving-model",
        )
        self._futures: set[Future[dict[str, object]]] = set()
        self._pending = 0
        self._active = 0
        self._accepted = 0
        self._completed = 0
        self._failed = 0
        self._rejected_overload = 0
        self._rejected_unavailable = 0
        self._timed_out_before_start = 0
        self._timed_out_after_start = 0
        self._cancelled_before_start = 0
        self._queue_high_watermark = 0
        self._model_execution_seconds = 0.0
        self._closed = False

    @staticmethod
    def _safe_backend_identity(
        backend: InferenceBackend | None,
    ) -> dict[str, object]:
        if backend is None:
            return {}
        diagnostics = getattr(backend, "diagnostics", None)
        if not callable(diagnostics):
            return {}
        try:
            raw = diagnostics()
        except (TypeError, ValueError, RuntimeError, OSError):
            return {}
        if not isinstance(raw, Mapping):
            return {}
        return {key: raw[key] for key in _IDENTITY_KEYS if key in raw}

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._state == "ready" and not self._closed

    def install_backend(self, backend: InferenceBackend) -> None:
        """Complete an explicit loading lifecycle without replacing a live model."""

        if backend is None:
            raise TypeError("backend must not be None")
        identity = self._safe_backend_identity(backend)
        with self._lock:
            if self._closed:
                raise ServingUnavailableError("serving runtime is stopped")
            if self._state != "loading" or self._backend is not None:
                raise ServingRuntimeError(
                    "backend installation is allowed only while runtime is loading"
                )
            self._backend = backend
            self._identity = identity
            self._loading_error_type = None
            self._state = "ready"

    def fail_loading(self, exc: BaseException) -> None:
        """Record only the failure class; never persist arbitrary exception text."""

        with self._lock:
            if self._closed:
                return
            if self._state != "loading" or self._backend is not None:
                raise ServingRuntimeError(
                    "loading failure can be recorded only while runtime is loading"
                )
            self._loading_error_type = type(exc).__name__
            self._state = "failed"

    def begin_draining(self) -> None:
        with self._lock:
            if self._closed or self._state == "stopped":
                return
            self._state = "draining"

    def model_identity(self) -> dict[str, object]:
        with self._lock:
            identity = {
                "id": self.model_name,
                "runtime_state": self._state,
                "execution_lanes": _EXECUTION_LANES,
            }
            identity.update(self._identity)
            return identity

    def status(self) -> dict[str, object]:
        with self._lock:
            queue_depth = max(0, self._pending - _EXECUTION_LANES)
            return {
                "state": self._state,
                "ready": self._state == "ready" and not self._closed,
                "model": self.model_name,
                "execution_lanes": _EXECUTION_LANES,
                "queue_depth": queue_depth,
                "max_queue_depth": self.max_queue_depth,
                "active_requests": self._active,
                "queue_high_watermark": self._queue_high_watermark,
                "accepted_requests": self._accepted,
                "completed_requests": self._completed,
                "failed_requests": self._failed,
                "rejected_overload": self._rejected_overload,
                "rejected_unavailable": self._rejected_unavailable,
                "timed_out_before_start": self._timed_out_before_start,
                "timed_out_after_start": self._timed_out_after_start,
                "cancelled_before_start": self._cancelled_before_start,
                "model_execution_seconds": self._model_execution_seconds,
                "loading_error_type": self._loading_error_type,
            }

    def submit(
        self,
        payload: Mapping[str, Any],
        *,
        response_id: str,
        created: int,
        timeout_seconds: float = DEFAULT_COMPLETION_TIMEOUT_SECONDS,
    ) -> dict[str, object]:
        timeout = _require_positive_finite(
            timeout_seconds,
            name="completion_timeout_seconds",
        )

        with self._lock:
            if self._closed or self._state != "ready" or self._backend is None:
                self._rejected_unavailable += 1
                raise ServingUnavailableError(
                    f"serving runtime is not ready (state={self._state})"
                )

        if not self._capacity.acquire(blocking=False):
            with self._lock:
                self._rejected_overload += 1
            raise ServingOverloadedError("serving request queue is full")

        with self._lock:
            if self._closed or self._state != "ready" or self._backend is None:
                self._rejected_unavailable += 1
                self._capacity.release()
                raise ServingUnavailableError(
                    f"serving runtime is not ready (state={self._state})"
                )
            backend = self._backend
            self._pending += 1
            self._accepted += 1
            queue_depth = max(0, self._pending - _EXECUTION_LANES)
            self._queue_high_watermark = max(
                self._queue_high_watermark,
                queue_depth,
            )

        try:
            future = self._executor.submit(
                self._execute,
                backend,
                dict(payload),
                response_id,
                created,
            )
        except RuntimeError as exc:
            self._capacity.release()
            with self._lock:
                self._pending -= 1
                self._accepted -= 1
                self._rejected_unavailable += 1
            raise ServingUnavailableError("serving runtime is stopping") from exc

        with self._lock:
            self._futures.add(future)
        future.add_done_callback(self._on_done)

        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            cancelled = future.cancel()
            with self._lock:
                if cancelled:
                    self._timed_out_before_start += 1
                else:
                    self._timed_out_after_start += 1
            raise ServingRequestTimeoutError(
                execution_started=not cancelled
            ) from exc
        except CancelledError as exc:
            raise ServingUnavailableError(
                "serving request was cancelled during shutdown"
            ) from exc

    def _execute(
        self,
        backend: InferenceBackend,
        payload: Mapping[str, Any],
        response_id: str,
        created: int,
    ) -> dict[str, object]:
        started = time.monotonic()
        with self._lock:
            self._active += 1
        try:
            return completion_response(
                backend,
                payload,
                response_id=response_id,
                created=created,
                model_name=self.model_name,
            )
        finally:
            elapsed = time.monotonic() - started
            with self._lock:
                self._active -= 1
                self._model_execution_seconds += elapsed

    def _on_done(self, future: Future[dict[str, object]]) -> None:
        with self._lock:
            self._futures.discard(future)
            self._pending -= 1
            if future.cancelled():
                self._cancelled_before_start += 1
            else:
                try:
                    error = future.exception()
                except CancelledError:
                    self._cancelled_before_start += 1
                else:
                    if error is None:
                        self._completed += 1
                    else:
                        self._failed += 1
        self._capacity.release()

    def close(self, *, drain: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._state = "draining"
        self._executor.shutdown(wait=True, cancel_futures=not drain)
        with self._lock:
            self._state = "stopped"

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
