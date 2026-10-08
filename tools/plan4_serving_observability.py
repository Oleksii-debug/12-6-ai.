"""Plan 4 Section 9: measured, bounded telemetry around accepted S8 serving.

No model, scheduler, provider, evaluation or checkpoint authority is created.
Host-only readiness probes are not quality evaluation or training evidence.
"""
from __future__ import annotations

import re
import threading
import time
from collections import Counter
from typing import Any, Callable

from tools.plan4_multimodel_serving import MultiModelServing, SCHEMA as SERVING_SCHEMA

SCHEMA = "12-6.plan4-serving-observability.v1"
_REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_ERROR_CODES = frozenset({
    "ALL_ROUTES_FAILED", "BACKEND_FAILURE", "BACKEND_REJECTED",
    "IDENTITY_DRIFT", "INVALID_BACKEND", "INVALID_REQUEST",
    "OVERLOADED", "PERMISSION_DENIED", "QUEUE_TIMEOUT",
    "RESOURCE_LIMIT", "SERVING_FAILURE", "UNKNOWN_ROUTE",
    "UNKNOWN_SESSION", "UNSUPPORTED_OP",
})


class ObservabilityError(ValueError):
    """Refused host configuration or non-authoritative measurement."""


class ServingObserver:
    """Bounded admission and process-local measurements for one S8 plane."""

    def __init__(self, serving: MultiModelServing, *, max_inflight: int = 8,
                 max_queue: int = 0, queue_timeout_s: float = 0.0,
                 memory_probe: Callable[[], int | None] | None = None) -> None:
        if not isinstance(serving, MultiModelServing):
            raise ObservabilityError("canonical S8 serving plane required")
        if type(max_inflight) is not int or not 1 <= max_inflight <= 128:
            raise ObservabilityError("invalid in-flight admission budget")
        if type(max_queue) is not int or not 0 <= max_queue <= 1024:
            raise ObservabilityError("invalid queue admission budget")
        if type(queue_timeout_s) not in (int, float) or not (
                0 <= queue_timeout_s <= 60):
            raise ObservabilityError("invalid bounded queue timeout")
        if memory_probe is not None and not callable(memory_probe):
            raise ObservabilityError("memory probe must be host-provided")
        self._serving = serving
        self._max_inflight = max_inflight
        self._max_queue = max_queue
        self._queue_timeout_s = float(queue_timeout_s)
        self._memory_probe = memory_probe
        self._condition = threading.Condition(threading.RLock())
        self._started_ns = time.monotonic_ns()
        self._inflight = 0
        self._queued = 0
        self._peak_queue = 0
        self._completed = 0
        self._success = 0
        self._errors: Counter[str] = Counter()
        self._cancellations = 0
        self._rejected = 0
        self._latency_total_ns = 0
        self._latency_max_ns = 0
        self._queue_total_ns = 0
        self._load_count = 0
        self._load_failures = 0
        self._load_total_ns = 0
        self._canary_count = 0
        self._canary_ready = 0

    @staticmethod
    def _safe_error(code: object) -> str:
        return code if type(code) is str and code in _ERROR_CODES else "OTHER"

    @staticmethod
    def _rejection(request: object, code: str) -> dict[str, Any]:
        request_id = None
        if type(request) is dict and type(request.get("args")) is dict:
            maybe = request["args"].get("request_id")
            if type(maybe) is str and _REQUEST_ID.fullmatch(maybe):
                request_id = maybe
        return {
            "schema": SERVING_SCHEMA, "ok": False, "request_id": request_id,
            "attempts": [],
            "error": {"code": code, "message": "serving admission rejected"},
        }

    def dispatch(self, request: object) -> dict[str, Any]:
        """Preserve S8 responses; never retry failed or ambiguous effects."""
        start = time.monotonic_ns()
        with self._condition:
            if self._inflight >= self._max_inflight:
                if self._max_queue == 0 or self._queued >= self._max_queue:
                    self._rejected += 1
                    self._errors["OVERLOADED"] += 1
                    return self._rejection(request, "OVERLOADED")
                self._queued += 1
                self._peak_queue = max(self._queued, self._peak_queue)
                deadline = time.monotonic() + self._queue_timeout_s
                try:
                    while self._inflight >= self._max_inflight:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            self._rejected += 1
                            self._errors["QUEUE_TIMEOUT"] += 1
                            return self._rejection(request, "QUEUE_TIMEOUT")
                        self._condition.wait(remaining)
                finally:
                    self._queued -= 1
                    self._queue_total_ns += max(0, time.monotonic_ns() - start)
            self._inflight += 1
        begin = time.monotonic_ns()
        try:
            response = self._serving.dispatch(request)
            if type(response) is not dict or response.get("schema") != SERVING_SCHEMA:
                raise ObservabilityError("S8 returned invalid envelope")
        except Exception:
            # Exception outcome might follow execution: never replay or reroute it.
            response = self._rejection(request, "SERVING_FAILURE")
        finally:
            elapsed = max(0, time.monotonic_ns() - begin)
            with self._condition:
                self._inflight -= 1
                self._condition.notify()
                self._completed += 1
                self._latency_total_ns += elapsed
                self._latency_max_ns = max(elapsed, self._latency_max_ns)
                if response.get("ok") is True:
                    self._success += 1
                    if (type(request) is dict
                            and request.get("op") == "stream.cancel"):
                        self._cancellations += 1
                else:
                    raw = response.get("error")
                    code = raw.get("code") if type(raw) is dict else None
                    self._errors[self._safe_error(code)] += 1
        return response

    def record_model_load(self, operation: Callable[[], Any]) -> Any:
        """Time one explicitly host-issued existing S6 load, no load authority."""
        if not callable(operation):
            raise ObservabilityError("trusted host operation required")
        start = time.monotonic_ns()
        try:
            return operation()
        except BaseException:
            with self._condition:
                self._load_failures += 1
            raise
        finally:
            with self._condition:
                self._load_count += 1
                self._load_total_ns += max(0, time.monotonic_ns() - start)

    def metrics(self) -> dict[str, Any]:
        """Metadata only; missing process memory is UNKNOWN, not 0."""
        observed_memory = None
        if self._memory_probe is not None:
            try:
                raw = self._memory_probe()
                if raw is not None and type(raw) is int and raw >= 0:
                    observed_memory = raw
            except Exception:
                pass
        catalog_reply = self._serving.dispatch({
            "schema": SERVING_SCHEMA, "op": "catalog", "args": {},
        })
        with self._condition:
            errors = dict(sorted(self._errors.items()))
            elapsed = max(1, time.monotonic_ns() - self._started_ns)
            local = {
                "schema": SCHEMA, "elapsed_ns": elapsed,
                "completed_requests": self._completed,
                "successful_requests": self._success,
                "errors_by_code": errors, "rejected_requests": self._rejected,
                "cancellations": self._cancellations,
                "throughput_rps": self._completed * 1e9 / elapsed,
                "latency_mean_ns": (self._latency_total_ns / self._completed
                                    if self._completed else None),
                "latency_max_ns": (self._latency_max_ns if self._completed else None),
                "inflight": self._inflight, "max_inflight": self._max_inflight,
                "queued": self._queued, "max_queue": self._max_queue,
                "peak_queue": self._peak_queue,
                "queue_total_wait_ns": self._queue_total_ns,
                "model_load_count": self._load_count,
                "model_load_failures": self._load_failures,
                "model_load_total_ns": self._load_total_ns,
                "canary_count": self._canary_count,
                "canary_ready": self._canary_ready,
                "process_memory_bytes": observed_memory,
                "process_memory_source": (
                    "host_probe" if observed_memory is not None else "UNAVAILABLE"
                ),
            }
        if type(catalog_reply) is dict and catalog_reply.get("ok") is True:
            c = catalog_reply["result"]
            local["capacity"] = {
                "budget_bytes_declared": c["budget_bytes"],
                "reserved_bytes_declared": c["reserved_bytes"],
                "available_bytes_declared": c["available_bytes"],
                "active_sessions": c["active_sessions"],
                "loaded_slot_count": len(c["slots"]),
                "actual_gpu_memory_bytes": None,
            }
        else:
            local["capacity"] = None
        return local

    def canary(self, *, role: str, allow_external: bool = False) -> dict[str, Any]:
        """Host-only readiness smoke check, never benchmark/evaluation evidence."""
        if type(role) is not str or _REQUEST_ID.fullmatch(role) is None:
            raise ObservabilityError("invalid canary role")
        if type(allow_external) is not bool:
            raise ObservabilityError("explicit external policy must be boolean")
        catalog = self._serving.dispatch({
            "schema": SERVING_SCHEMA, "op": "catalog", "args": {},
        })
        slots = catalog.get("result", {}).get("slots", []) if catalog.get("ok") else []
        ordered = sorted((s for s in slots if s["role"] == role),
                         key=lambda s: (-s["priority"], s["slot_id"]))
        with self._condition:
            self._canary_count += 1
        if not ordered:
            return {"schema": SCHEMA, "ready": False, "reason": "NO_ROLE",
                    "evaluation": False}
        if ordered[0]["transport"] == "external" and not allow_external:
            return {"schema": SCHEMA, "ready": False, "reason": "EXTERNAL_DENIED",
                    "evaluation": False}
        check = self.dispatch({
            "schema": SERVING_SCHEMA, "op": "generate",
            "args": {"request_id": "canary", "role": role, "prompt": "ready",
                     "config": {"max_new_tokens": 1, "strategy": "greedy"},
                     "failover": False},
        })
        ready = check.get("ok") is True
        with self._condition:
            self._canary_ready += int(ready)
        return {
            "schema": SCHEMA, "ready": ready, "evaluation": False,
            "model_identity": (check["selected"]["identity"] if ready else None),
            "reason": None if ready else self._safe_error(
                check.get("error", {}).get("code")),
        }
