"""Plan 4 Section 8: bounded multi-model serving over the canonical ModelGateway.

Host-only admission/eviction/replacement; never an agent scheduler or tool authority.
Every inference/stream reply binds request, selected model and attempted routes.
"""
from __future__ import annotations

import re
import secrets
import threading
from dataclasses import dataclass
from typing import Any

from tools.plan4_model_gateway import ModelGateway

SCHEMA = "12-6.plan4-multimodel-serving.v1"
_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
# Only a proven pre-dispatch identity rejection can safely reroute.
# An ambiguous backend/transport failure may follow generation execution.
# Never automatically repeat the same request across providers after that.
_RETRYABLE = frozenset({"IDENTITY_DRIFT"})


class ServingError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _name(value: object) -> str:
    if type(value) is not str or _NAME.fullmatch(value) is None:
        raise ServingError("INVALID_REQUEST", "invalid serving identifier")
    return value


def _fields(value: object, expected: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ServingError("INVALID_REQUEST", "missing or unexpected fields")
    return value


@dataclass(frozen=True)
class _Slot:
    slot_id: str
    role: str
    provider: str
    model: str
    transport: str
    priority: int
    reserved_bytes: int
    identity: tuple[tuple[str, object], ...]
    capabilities: tuple[str, ...]
    gateway: ModelGateway


@dataclass(frozen=True)
class _Session:
    slot_id: str
    gateway_session_id: str
    request_id: str
    role: str


class MultiModelServing:
    """One serving/routing plane, without scheduling, training or tool permissions."""

    def __init__(self, *, budget_bytes: int, max_sessions: int = 32,
                 failover_roles: tuple[str, ...] = ()) -> None:
        if type(budget_bytes) is not int or not 1 <= budget_bytes <= 2**50:
            raise ServingError("INVALID_REQUEST", "invalid capacity budget")
        if type(max_sessions) is not int or not 1 <= max_sessions <= 128:
            raise ServingError("INVALID_REQUEST", "invalid stream capacity")
        if type(failover_roles) is not tuple or any(
                type(x) is not str or _NAME.fullmatch(x) is None
                for x in failover_roles) or len(set(failover_roles)) != len(failover_roles):
            raise ServingError("INVALID_REQUEST", "invalid trusted failover policy")
        self._lock = threading.RLock()
        self._budget_bytes = budget_bytes
        self._max_sessions = max_sessions
        self._failover_roles = frozenset(failover_roles)
        self._slots: dict[str, _Slot] = {}
        self._sessions: dict[str, _Session] = {}
        self._retired: list[dict[str, Any]] = []

    @staticmethod
    def _snapshot(slot: _Slot) -> dict[str, Any]:
        return {
            "slot_id": slot.slot_id, "role": slot.role,
            "provider": slot.provider, "model": slot.model,
            "transport": slot.transport, "priority": slot.priority,
            "reserved_bytes": slot.reserved_bytes,
            "identity": dict(slot.identity),
            "capabilities": list(slot.capabilities),
        }

    @staticmethod
    def _prepare(slot_id: str, role: str, provider: str, model: str,
                 transport: str, client: Any, priority: int, reserved_bytes: int,
                 authorize_external: bool, capabilities: tuple[str, ...]) -> _Slot:
        for value in (slot_id, role, provider, model):
            _name(value)
        if type(priority) is not int or not -100 <= priority <= 100:
            raise ServingError("INVALID_REQUEST", "invalid role priority")
        if type(reserved_bytes) is not int or not 1 <= reserved_bytes <= 2**50:
            raise ServingError("INVALID_REQUEST", "invalid resource reservation")
        gateway = ModelGateway(max_sessions=128)
        gateway.register(provider, model, transport, client,
                         authorize_external=authorize_external, capabilities=capabilities)
        reply = gateway.dispatch({
            "schema": "12-6.plan4-model-gateway.v1",
            "op": "describe", "args": {"provider": provider, "model": model},
        })
        if reply.get("ok") is not True:
            raise ServingError("INVALID_BACKEND", "gateway admission failed")
        desc = reply["result"]
        return _Slot(slot_id, role, provider, model, transport, priority,
                     reserved_bytes, tuple(sorted(desc["identity"].items())),
                     tuple(desc["capabilities"]), gateway)

    def _used(self) -> int:
        return sum(slot.reserved_bytes for slot in self._slots.values())

    def _busy(self, slot_id: str) -> bool:
        return any(s.slot_id == slot_id for s in self._sessions.values())

    def publish(self, *, slot_id: str, role: str, provider: str, model: str,
                transport: str, client: Any, priority: int, reserved_bytes: int,
                authorize_external: bool = False,
                capabilities: tuple[str, ...] = ("generate", "stream")) -> dict[str, Any]:
        """Trusted host-only transaction; client dispatch cannot register providers."""
        with self._lock:
            if slot_id in self._slots:
                raise ServingError("CONFLICT", "slot identifier already published")
            candidate = self._prepare(slot_id, role, provider, model, transport,
                                      client, priority, reserved_bytes, authorize_external,
                                      capabilities)
            if self._used() + reserved_bytes > self._budget_bytes:
                raise ServingError("RESOURCE_LIMIT", "insufficient serving reservation")
            self._slots[slot_id] = candidate
            return self._snapshot(candidate)

    def evict(self, slot_id: str) -> dict[str, Any]:
        """No implicit eviction of active streams; old identity remains in receipts."""
        with self._lock:
            if slot_id not in self._slots:
                raise ServingError("UNKNOWN_ROUTE", "unknown serving slot")
            if self._busy(slot_id):
                raise ServingError("CONFLICT", "active stream prevents eviction")
            old = self._slots.pop(slot_id)
            receipt = self._snapshot(old)
            self._retired.append(receipt)
            return receipt

    def replace(self, old_slot_id: str, *, slot_id: str, role: str, provider: str,
                model: str, transport: str, client: Any, priority: int,
                reserved_bytes: int, authorize_external: bool = False,
                capabilities: tuple[str, ...] = ("generate", "stream")) -> dict[str, Any]:
        """Prepare/verify successor first; atomically swap or keep old slot."""
        with self._lock:
            if old_slot_id not in self._slots or slot_id in self._slots:
                raise ServingError("CONFLICT", "missing incumbent or reused slot ID")
            if self._busy(old_slot_id):
                raise ServingError("CONFLICT", "active stream prevents replacement")
            candidate = self._prepare(slot_id, role, provider, model, transport, client,
                                      priority, reserved_bytes, authorize_external,
                                      capabilities)
            old = self._slots[old_slot_id]
            if self._used() - old.reserved_bytes + reserved_bytes > self._budget_bytes:
                raise ServingError("RESOURCE_LIMIT", "replacement exceeds capacity")
            del self._slots[old_slot_id]
            self._slots[slot_id] = candidate
            self._retired.append(self._snapshot(old))
            return {"retired": self._snapshot(old), "active": self._snapshot(candidate)}

    def _catalog(self) -> dict[str, Any]:
        return {
            "budget_bytes": self._budget_bytes,
            "reserved_bytes": self._used(),
            "available_bytes": self._budget_bytes - self._used(),
            "active_sessions": len(self._sessions),
            "slots": [self._snapshot(s) for s in
                      sorted(self._slots.values(), key=lambda x: x.slot_id)],
            "retired": list(self._retired),
        }

    def _route(self, op: str, args: dict[str, Any]) -> dict[str, Any]:
        _fields(args, {"request_id", "role", "prompt", "config", "failover"})
        request_id, role = _name(args["request_id"]), _name(args["role"])
        failover = args["failover"]
        if type(failover) is not bool:
            raise ServingError("INVALID_REQUEST", "failover must be explicit boolean")
        if failover and role not in self._failover_roles:
            raise ServingError("PERMISSION_DENIED", "role has no host-approved failover")
        if op == "stream.start" and len(self._sessions) >= self._max_sessions:
            raise ServingError("RESOURCE_LIMIT", "serving stream capacity exhausted")
        candidates = sorted((s for s in self._slots.values() if s.role == role),
                            key=lambda s: (-s.priority, s.slot_id))
        if not candidates:
            raise ServingError("UNKNOWN_ROUTE", "role has no admitted models")
        if not failover:
            candidates = candidates[:1]
        attempts: list[dict[str, Any]] = []
        for slot in candidates:
            gw = slot.gateway.dispatch({
                "schema": "12-6.plan4-model-gateway.v1",
                "op": op,
                "args": {"provider": slot.provider, "model": slot.model,
                         "prompt": args["prompt"], "config": args["config"]},
            })
            attempt = {"route": self._snapshot(slot), "error_code": None}
            if gw.get("ok") is True:
                attempts.append(attempt)
                result = gw["result"]
                if op == "stream.start":
                    remote = result.get("data", {}).get("session_id")
                    if type(remote) is not str:
                        return self._error("INVALID_BACKEND", "malformed stream",
                                           request_id, attempts)
                    sid = secrets.token_hex(16)
                    self._sessions[sid] = _Session(slot.slot_id, remote, request_id, role)
                    result = {**result, "data": {**result["data"], "session_id": sid}}
                return {"schema": SCHEMA, "ok": True, "request_id": request_id,
                        "role": role, "selected": self._snapshot(slot),
                        "attempts": attempts, "result": result}
            code = gw.get("error", {}).get("code", "BACKEND_FAILURE")
            attempt["error_code"] = code
            attempts.append(attempt)
            if not failover or code not in _RETRYABLE:
                break
        return self._error("ALL_ROUTES_FAILED", "no admitted route completed",
                           request_id, attempts)

    @staticmethod
    def _error(code: str, message: str, request_id: str | None = None,
               attempts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {"schema": SCHEMA, "ok": False, "request_id": request_id,
                "attempts": attempts or [], "error": {"code": code, "message": message}}

    def _stream(self, op: str, args: dict[str, Any]) -> dict[str, Any]:
        expected = {"session_id", "cursor"} if op == "stream.next" else {"session_id"}
        _fields(args, expected)
        sid = args["session_id"]
        if type(sid) is not str or sid not in self._sessions:
            raise ServingError("UNKNOWN_SESSION", "unknown serving session")
        session = self._sessions[sid]
        if session.slot_id not in self._slots:
            raise ServingError("IDENTITY_DRIFT", "session route was retired")
        slot = self._slots[session.slot_id]
        forwarded = {"session_id": session.gateway_session_id}
        if op == "stream.next":
            forwarded["cursor"] = args["cursor"]
        gw = slot.gateway.dispatch({
            "schema": "12-6.plan4-model-gateway.v1", "op": op, "args": forwarded,
        })
        attempt = {"route": self._snapshot(slot),
                   "error_code": None if gw.get("ok") is True else
                   gw.get("error", {}).get("code", "BACKEND_FAILURE")}
        if gw.get("ok") is not True:
            return self._error(attempt["error_code"], "pinned stream rejected",
                               session.request_id, [attempt])
        result = gw["result"]
        if op == "stream.close":
            del self._sessions[sid]
        if "data" in result:
            result = {**result, "data": {**result["data"], "session_id": sid}}
        return {"schema": SCHEMA, "ok": True, "request_id": session.request_id,
                "role": session.role, "selected": self._snapshot(slot),
                "attempts": [attempt], "result": result}

    def dispatch(self, request: object) -> dict[str, Any]:
        with self._lock:
            try:
                req = _fields(request, {"schema", "op", "args"})
                if req["schema"] != SCHEMA or type(req["op"]) is not str:
                    raise ServingError("INVALID_REQUEST", "unknown serving protocol")
                op = req["op"]
                if op == "catalog":
                    _fields(req["args"], set())
                    return {"schema": SCHEMA, "ok": True, "result": self._catalog()}
                if op in {"generate", "stream.start"}:
                    return self._route(op, req["args"])
                if op in {"stream.next", "stream.cancel", "stream.close"}:
                    return self._stream(op, req["args"])
                raise ServingError("UNSUPPORTED_OP", "unknown serving operation")
            except ServingError as exc:
                return self._error(exc.code, str(exc))
            except Exception:
                return self._error("SERVING_FAILURE", "serving plane failed closed")
