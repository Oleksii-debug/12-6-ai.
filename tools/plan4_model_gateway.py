"""Plan 4 Section 7: host-pinned ModelGateway over the canonical service v1.

The gateway neither loads weights nor grants routing, tool or provider permissions.
Only trusted host registration creates immutable (provider, model) routes.
"""
from __future__ import annotations

import re
import secrets
import threading
from dataclasses import dataclass
from typing import Any, Protocol

from tools.plan4_model_service import SCHEMA as SERVICE_SCHEMA

SCHEMA = "12-6.plan4-model-gateway.v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_CAPABILITIES = frozenset({"generate", "stream"})
_TRANSPORTS = frozenset({"local", "server", "external"})


class GatewayError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ServiceClient(Protocol):
    def call(self, request: object) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Route:
    provider: str
    model: str
    transport: str
    capabilities: tuple[str, ...]
    identity: tuple[tuple[str, object], ...]


@dataclass
class _Stream:
    route: Route
    remote_id: str


def _fields(value: object, required: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != required:
        raise GatewayError("INVALID_REQUEST", "unknown or missing request fields")
    return value


def _name(value: object) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise GatewayError("INVALID_REQUEST", "invalid provider or model identity")
    return value


def _identity(value: object, model: str) -> tuple[tuple[str, object], ...]:
    if type(value) is not dict:
        raise GatewayError("INVALID_BACKEND", "invalid model identity response")
    required = {
        "model", "epoch", "model_spec_sha256", "model_weights_sha256",
        "tokenizer_sha256", "backend",
    }
    if set(value) != required or value["model"] != model:
        raise GatewayError("INVALID_BACKEND", "model identity mismatch")
    if type(value["epoch"]) is not int or value["epoch"] < 0:
        raise GatewayError("INVALID_BACKEND", "invalid service epoch")
    if type(value["backend"]) is not str or not value["backend"]:
        raise GatewayError("INVALID_BACKEND", "invalid backend type")
    for field in ("model_spec_sha256", "model_weights_sha256", "tokenizer_sha256"):
        if type(value[field]) is not str or _HASH.fullmatch(value[field]) is None:
            raise GatewayError("INVALID_BACKEND", "invalid model identity digest")
    return tuple(sorted(value.items()))


def _usage(result: object) -> dict[str, int] | None:
    if type(result) is not dict:
        return None
    source = result.get("generation", result.get("event", {}).get("result", {}))
    if type(source) is not dict:
        return None
    prompt = source.get("prompt_token_count")
    output = source.get("output_token_ids")
    if type(prompt) is not int or prompt < 0 or type(output) not in (list, tuple):
        return None
    if any(type(token) is not int or token < 0 for token in output):
        return None
    return {"input_tokens": prompt, "output_tokens": len(output)}


class ModelGateway:
    """Replaceable cognitive core with exact, immutable, host-authorized routes."""

    def __init__(self, *, max_sessions: int = 32) -> None:
        if type(max_sessions) is not int or not 1 <= max_sessions <= 128:
            raise GatewayError("INVALID_REQUEST", "invalid session limit")
        self._lock = threading.RLock()
        self._max_sessions = max_sessions
        self._routes: dict[tuple[str, str], tuple[Route, ServiceClient]] = {}
        self._streams: dict[str, _Stream] = {}

    @staticmethod
    def _service(client: ServiceClient, op: str, args: dict[str, Any]) -> Any:
        try:
            response = client.call({"schema": SERVICE_SCHEMA, "op": op, "args": args})
        except Exception:
            raise GatewayError("BACKEND_FAILURE", "service transport failed") from None
        if type(response) is not dict or response.get("schema") != SERVICE_SCHEMA:
            raise GatewayError("INVALID_BACKEND", "wrong service contract")
        if response.get("ok") is False:
            error = response.get("error")
            if type(error) is not dict or type(error.get("code")) is not str:
                raise GatewayError("INVALID_BACKEND", "malformed backend error")
            raise GatewayError("BACKEND_REJECTED", "service rejected: " + error["code"])
        if response.get("ok") is not True or type(response.get("result")) is not dict:
            raise GatewayError("INVALID_BACKEND", "malformed backend result")
        return response["result"]

    def register(
        self,
        provider: str,
        model: str,
        transport: str,
        client: ServiceClient,
        *,
        capabilities: tuple[str, ...] = ("generate", "stream"),
        authorize_external: bool = False,
    ) -> None:
        """Host-only registration; requests cannot register or change a route."""
        with self._lock:
            provider, model = _name(provider), _name(model)
            if transport not in _TRANSPORTS:
                raise GatewayError("INVALID_REQUEST", "unsupported transport")
            if transport == "external" and authorize_external is not True:
                raise GatewayError("PERMISSION_DENIED", "external adapter not authorized")
            if type(capabilities) is not tuple or not capabilities:
                raise GatewayError("INVALID_REQUEST", "capabilities must be explicit")
            known = type(capabilities) is tuple and all(type(c) is str for c in capabilities)
            if not known or len(set(capabilities)) != len(capabilities):
                raise GatewayError("INVALID_REQUEST", "invalid capabilities")
            if not set(capabilities) <= _CAPABILITIES:
                raise GatewayError("INVALID_REQUEST", "unsupported capabilities")
            key = (provider, model)
            if key in self._routes:
                raise GatewayError("CONFLICT", "route is immutable")
            observed = self._service(client, "identity", {})
            pinned = _identity(observed, model)
            route = Route(provider, model, transport, tuple(sorted(capabilities)), pinned)
            self._routes[key] = (route, client)

    def _registered(self, provider: object, model: object) -> tuple[Route, ServiceClient]:
        key = (_name(provider), _name(model))
        if key not in self._routes:
            raise GatewayError("UNKNOWN_ROUTE", "provider and model not registered")
        route, client = self._routes[key]
        live = _identity(self._service(client, "identity", {}), route.model)
        if live != route.identity:
            raise GatewayError("IDENTITY_DRIFT", "service identity changed; rebind by host")
        return route, client

    @staticmethod
    def _envelope(route: Route, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "provider": route.provider,
            "model": route.model,
            "transport": route.transport,
            "identity": dict(route.identity),
            "capabilities": list(route.capabilities),
            "usage": _usage(result),
            "data": result,
        }

    def _execute(self, op: str, args: dict[str, Any]) -> dict[str, Any]:
        if op == "describe":
            _fields(args, {"provider", "model"})
            route, _ = self._registered(args["provider"], args["model"])
            return self._envelope(route, {})
        if op in ("generate", "stream.start"):
            _fields(args, {"provider", "model", "prompt", "config"})
            route, client = self._registered(args["provider"], args["model"])
            capability = "generate" if op == "generate" else "stream"
            if capability not in route.capabilities:
                raise GatewayError("UNSUPPORTED_FEATURE", "route lacks capability")
            if op == "stream.start" and len(self._streams) >= self._max_sessions:
                raise GatewayError("RESOURCE_LIMIT", "gateway sessions exhausted")
            result = self._service(client, op, {
                "prompt": args["prompt"], "config": args["config"],
            })
            self._registered(route.provider, route.model)
            if op == "stream.start":
                remote_id = result.get("session_id")
                if type(remote_id) is not str or not remote_id:
                    raise GatewayError("INVALID_BACKEND", "missing stream identity")
                gateway_id = secrets.token_hex(16)
                self._streams[gateway_id] = _Stream(route, remote_id)
                result = {**result, "session_id": gateway_id}
            return self._envelope(route, result)
        if op in ("stream.next", "stream.cancel", "stream.close"):
            required = {"session_id", "cursor"} if op == "stream.next" else {"session_id"}
            _fields(args, required)
            sid = args["session_id"]
            if type(sid) is not str or sid not in self._streams:
                raise GatewayError("UNKNOWN_SESSION", "unknown gateway session")
            stream = self._streams[sid]
            route, client = self._registered(stream.route.provider, stream.route.model)
            if route != stream.route:
                raise GatewayError("IDENTITY_DRIFT", "gateway route replaced")
            forwarded: dict[str, Any] = {"session_id": stream.remote_id}
            if op == "stream.next":
                forwarded["cursor"] = args["cursor"]
            result = self._service(client, op, forwarded)
            self._registered(route.provider, route.model)
            if op == "stream.close":
                del self._streams[sid]
            result = {**result, "session_id": sid}
            return self._envelope(route, result)
        raise GatewayError("UNSUPPORTED_OP", "unsupported gateway operation")

    def dispatch(self, request: object) -> dict[str, Any]:
        with self._lock:
            try:
                req = _fields(request, {"schema", "op", "args"})
                if req["schema"] != SCHEMA or type(req["op"]) is not str:
                    raise GatewayError("INVALID_REQUEST", "unknown gateway schema")
                if type(req["args"]) is not dict:
                    raise GatewayError("INVALID_REQUEST", "arguments must be an object")
                result = self._execute(req["op"], req["args"])
                return {"schema": SCHEMA, "ok": True, "result": result}
            except GatewayError as exc:
                return {"schema": SCHEMA, "ok": False,
                        "error": {"code": exc.code, "message": str(exc)}}
            except Exception:
                return {"schema": SCHEMA, "ok": False,
                        "error": {"code": "BACKEND_FAILURE",
                                  "message": "gateway failed closed"}}
