"""Plan 4 Section 6: one fail-closed local/loopback-HTTP model service contract.

Consumes already-verified ReferenceInference adapters; it never deserializes
weights, replaces checkpoint authority, or permits unverified external effects.
HTTP and in-process clients use precisely the same dispatch state machine.
"""
from __future__ import annotations

import hmac
import json
import re
import secrets
import threading
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from tools.inference_runtime import (
    GenerationConfig,
    GenerationEvent,
    InferenceError,
    InferenceSession,
    ReferenceInference,
    _model_weights_sha,
    _object_sha,
)

SCHEMA = "12-6.plan4-model-service.v1"
MAX_HTTP_BYTES = 65536
MODEL_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
OPERATIONS = {
    "load", "unload", "swap", "restart", "health", "identity", "generate",
    "stream.start", "stream.next", "stream.cancel", "stream.close",
}


class ServiceError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _check_fields(value: object, keys: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ServiceError("INVALID_REQUEST", "invalid request fields")
    return value


def _validate_identity(runtime: ReferenceInference) -> None:
    if not isinstance(runtime, ReferenceInference):
        raise ServiceError("INVALID_BACKEND", "trusted reference adapter required")
    if runtime.model.training:
        raise ServiceError("INVALID_BACKEND", "reference model must remain in eval mode")
    if _model_weights_sha(runtime.model) != runtime.model_weights_sha256:
        raise ServiceError("MODEL_DRIFT", "model weights changed since admission")
    if _object_sha(runtime.tokenizer.identity.to_dict()) != runtime.tokenizer_sha256:
        raise ServiceError("MODEL_DRIFT", "tokenizer identity changed")


def _config(raw: object) -> GenerationConfig:
    if type(raw) is not dict:
        raise ServiceError("INVALID_CONFIG", "config object required")
    allowed = {
        "max_new_tokens", "strategy", "temperature", "top_k", "seed", "decode_errors",
    }
    if "max_new_tokens" not in raw or set(raw) - allowed:
        raise ServiceError("INVALID_CONFIG", "missing or unknown generation setting")
    try:
        return GenerationConfig(**raw)
    except (TypeError, ValueError) as exc:
        raise ServiceError("INVALID_CONFIG", "invalid generation settings") from exc


def _event_dict(event: GenerationEvent) -> dict[str, Any]:
    return {
        "kind": event.kind, "sequence": event.sequence,
        "token_id": event.token_id, "result": asdict(event.result),
    }


@dataclass
class _LiveSession:
    source: InferenceSession
    events: Iterator[GenerationEvent]
    epoch: int
    cursor: int = 0
    last: dict[str, Any] | None = None
    terminal: bool = False


class ModelService:
    """Bounded single-process service; only a trusted host may register adapters."""

    def __init__(self, *, max_sessions: int = 16) -> None:
        if type(max_sessions) is not int or not 1 <= max_sessions <= 128:
            raise ServiceError("INVALID_REQUEST", "invalid session capacity")
        self._lock = threading.RLock()
        self._registered: dict[str, ReferenceInference] = {}
        self._selected: str | None = None
        self._status = "UNLOADED"
        self._epoch = 0
        self._sessions: dict[str, _LiveSession] = {}
        self._max_sessions = max_sessions

    def register(self, name: str, runtime: ReferenceInference) -> None:
        """Trusted in-process registration, never a remote file upload/unpickle."""
        with self._lock:
            if type(name) is not str or MODEL_NAME.fullmatch(name) is None:
                raise ServiceError("INVALID_REQUEST", "invalid model alias")
            if name in self._registered:
                raise ServiceError("ALREADY_EXISTS", "model aliases are immutable")
            _validate_identity(runtime)
            self._registered[name] = runtime

    def _get_runtime(self) -> ReferenceInference:
        if self._status != "READY" or self._selected is None:
            raise ServiceError("NOT_READY", "model service is not ready")
        runtime = self._registered[self._selected]
        try:
            _validate_identity(runtime)
        except (ServiceError, InferenceError, ValueError):
            self._status = "FAULTED"
            self._sessions.clear()
            raise ServiceError("MODEL_DRIFT", "selected model is no longer trusted") from None
        return runtime

    def _active(self) -> int:
        return sum(not value.terminal for value in self._sessions.values())

    def _health(self) -> dict[str, Any]:
        if self._status == "READY":
            try:
                self._get_runtime()
            except ServiceError:
                pass
        return {
            "status": self._status, "ready": self._status == "READY",
            "epoch": self._epoch, "model": self._selected,
            "active_sessions": self._active(), "capacity": self._max_sessions,
            "backend": "ReferenceInference" if self._status == "READY" else None,
        }

    def _identity(self) -> dict[str, Any]:
        runtime = self._get_runtime()
        return {
            "model": self._selected, "epoch": self._epoch,
            "model_spec_sha256": runtime.model_spec_sha256,
            "model_weights_sha256": runtime.model_weights_sha256,
            "tokenizer_sha256": runtime.tokenizer_sha256,
            "backend": "reference-torch-fp32",
        }

    def _select(self, alias: str, *, swap: bool) -> dict[str, Any]:
        if type(alias) is not str or alias not in self._registered:
            raise ServiceError("UNKNOWN_MODEL", "unregistered model alias")
        if self._status == "FAULTED":
            raise ServiceError("NOT_READY", "restart or unload faulted service first")
        if swap and self._status != "READY":
            raise ServiceError("NOT_READY", "swap needs a ready current model")
        if not swap and self._status != "UNLOADED":
            raise ServiceError("CONFLICT", "load requires unloaded service")
        if self._active():
            raise ServiceError("CONFLICT", "finish or cancel streams before model swap")
        _validate_identity(self._registered[alias])
        self._selected = alias
        self._status = "READY"
        self._epoch += 1
        self._sessions.clear()
        return self._health()

    def _unload(self) -> dict[str, Any]:
        if self._active():
            raise ServiceError("CONFLICT", "cannot unload a running stream")
        self._selected = None
        self._status = "UNLOADED"
        self._epoch += 1
        self._sessions.clear()
        return self._health()

    def _restart(self) -> dict[str, Any]:
        # A real process restart makes all old session IDs unavailable as well.
        self._selected = None
        self._status = "UNLOADED"
        self._epoch += 1
        self._sessions.clear()
        return self._health()

    def _session(self, session_id: object) -> _LiveSession:
        if type(session_id) is not str or session_id not in self._sessions:
            raise ServiceError("UNKNOWN_SESSION", "unknown or expired stream")
        session = self._sessions[session_id]
        if session.epoch != self._epoch:
            raise ServiceError("STALE_SESSION", "session belongs to previous model epoch")
        return session

    def _start(self, prompt: object, config: object) -> dict[str, Any]:
        runtime = self._get_runtime()
        if len(self._sessions) >= self._max_sessions:
            raise ServiceError("RESOURCE_LIMIT", "session capacity reached; close old sessions")
        if type(prompt) is not str or not prompt:
            raise ServiceError("INVALID_REQUEST", "nonempty prompt required")
        session = runtime.stream(prompt, _config(config))
        id_ = secrets.token_hex(16)
        self._sessions[id_] = _LiveSession(source=session, events=iter(session), epoch=self._epoch)
        return {"session_id": id_, "cursor": 0, "epoch": self._epoch,
                "model_weights_sha256": runtime.model_weights_sha256}

    def _next(self, session_id: object, cursor: object) -> dict[str, Any]:
        self._get_runtime()
        session = self._session(session_id)
        if type(cursor) is not int or cursor < 0:
            raise ServiceError("INVALID_REQUEST", "nonnegative integer cursor required")
        if cursor == session.cursor - 1 and session.last is not None:
            return session.last
        if cursor != session.cursor:
            raise ServiceError("STALE_CURSOR", "stream cursor is stale or out of order")
        if session.terminal:
            raise ServiceError("TERMINAL_STREAM", "stream has already terminated")
        try:
            event = next(session.events)
        except StopIteration as exc:
            raise ServiceError("RUNTIME_FAILURE", "stream ended without terminal event") from exc
        session.cursor += 1
        session.terminal = event.kind in ("COMPLETED", "CANCELLED")
        session.last = {
            "session_id": session_id, "cursor": session.cursor,
            "terminal": session.terminal, "event": _event_dict(event),
        }
        return session.last

    def _cancel(self, session_id: object) -> dict[str, Any]:
        self._get_runtime()
        session = self._session(session_id)
        if session.terminal:
            return {"already_terminal": True, "cursor": session.cursor}
        session.source.cancel()
        return {"already_terminal": False, "cursor": session.cursor}

    def _close(self, session_id: object) -> dict[str, Any]:
        session = self._session(session_id)
        if not session.terminal:
            raise ServiceError("CONFLICT", "cancel and drain a stream before closing")
        del self._sessions[session_id]
        return {"closed": True}

    def _operate(self, op: str, args: dict[str, Any]) -> dict[str, Any]:
        if op == "health":
            _check_fields(args, set())
            return self._health()
        if op == "identity":
            _check_fields(args, set())
            return self._identity()
        if op in ("load", "swap"):
            _check_fields(args, {"model"})
            return self._select(args["model"], swap=op == "swap")
        if op == "unload":
            _check_fields(args, set())
            return self._unload()
        if op == "restart":
            _check_fields(args, set())
            return self._restart()
        if op == "generate":
            _check_fields(args, {"prompt", "config"})
            runtime = self._get_runtime()
            if type(args["prompt"]) is not str or not args["prompt"]:
                raise ServiceError("INVALID_REQUEST", "nonempty prompt required")
            result = runtime.generate(args["prompt"], _config(args["config"]))
            return {"generation": asdict(result), "epoch": self._epoch}
        if op == "stream.start":
            _check_fields(args, {"prompt", "config"})
            return self._start(args["prompt"], args["config"])
        if op == "stream.next":
            _check_fields(args, {"session_id", "cursor"})
            return self._next(args["session_id"], args["cursor"])
        if op == "stream.cancel":
            _check_fields(args, {"session_id"})
            return self._cancel(args["session_id"])
        if op == "stream.close":
            _check_fields(args, {"session_id"})
            return self._close(args["session_id"])
        raise ServiceError("UNSUPPORTED_OP", "unknown or unsupported operation")

    def dispatch(self, request: object) -> dict[str, Any]:
        """Canonical JSON-serializable transport-independent dispatch contract."""
        with self._lock:
            try:
                req = _check_fields(request, {"schema", "op", "args"})
                if req["schema"] != SCHEMA or type(req["op"]) is not str:
                    raise ServiceError("INVALID_REQUEST", "unsupported service protocol")
                if req["op"] not in OPERATIONS:
                    raise ServiceError("UNSUPPORTED_OP", "unsupported service operation")
                args = req["args"]
                if type(args) is not dict:
                    raise ServiceError("INVALID_REQUEST", "operation arguments must be object")
                result = self._operate(req["op"], args)
                # Normalize tuples before either transport sees the response.
                # Local and HTTP clients must observe identical JSON value types.
                transport_result = json.loads(json.dumps(result, ensure_ascii=False,
                                                        allow_nan=False))
                return {"schema": SCHEMA, "ok": True, "result": transport_result}
            except ServiceError as exc:
                return {"schema": SCHEMA, "ok": False,
                        "error": {"code": exc.code, "message": str(exc)}}
            except (InferenceError, ValueError, TypeError):
                return {"schema": SCHEMA, "ok": False,
                        "error": {"code": "INFERENCE_REJECTED",
                                  "message": "canonical inference admission rejected"}}
            except Exception:
                self._status = "FAULTED"
                self._sessions.clear()
                return {"schema": SCHEMA, "ok": False,
                        "error": {"code": "RUNTIME_FAILURE",
                                  "message": "service failed closed"}}


class LocalClient:
    def __init__(self, service: ModelService) -> None:
        self.service = service

    def call(self, request: object) -> dict[str, Any]:
        return self.service.dispatch(request)


def _no_duplicates(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _strict_json(data: bytes) -> object:
    return json.loads(
        data.decode("utf-8"), object_pairs_hook=_no_duplicates,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
    )


def serve_loopback(service: ModelService, *, bearer_token: str) -> ThreadingHTTPServer:
    """Return an unstarted loopback HTTP server; caller controls its lifecycle."""
    if not isinstance(service, ModelService):
        raise ServiceError("INVALID_REQUEST", "service required")
    if type(bearer_token) is not str or len(bearer_token) < 16:
        raise ServiceError("INVALID_REQUEST", "strong bearer token required")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass  # Never log prompts, access credentials, or model internals.

        def _send(self, status: int, obj: object) -> None:
            blob = json.dumps(obj, ensure_ascii=False, allow_nan=False,
                              separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(blob)

        def do_POST(self) -> None:
            if self.path != "/v1/dispatch":
                self._send(404, {"error": "unknown endpoint"})
                return
            if not hmac.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + bearer_token,
            ):
                self._send(403, {"error": "forbidden"})
                return
            try:
                declared = self.headers.get("Content-Length")
                if declared is None or not declared.isdigit():
                    raise ValueError("invalid payload length")
                length = int(declared)
                if not 0 < length <= MAX_HTTP_BYTES:
                    raise ValueError("oversized or empty payload")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("JSON content type required")
                request = _strict_json(self.rfile.read(length))
            except (UnicodeError, ValueError, json.JSONDecodeError):
                self._send(400, {"error": "invalid JSON request"})
                return
            self._send(200, service.dispatch(request))

    return ThreadingHTTPServer(("127.0.0.1", 0), Handler)


class HttpClient:
    """Local-only, bearer-authenticated transport with LocalClient call semantics."""

    def __init__(self, endpoint: str, *, bearer_token: str) -> None:
        parsed = urlsplit(endpoint)
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
                or parsed.path != "/v1/dispatch" or parsed.query or parsed.fragment):
            raise ServiceError("INVALID_REQUEST", "only exact loopback service URL allowed")
        if type(bearer_token) is not str or len(bearer_token) < 16:
            raise ServiceError("INVALID_REQUEST", "strong bearer token required")
        self.endpoint = endpoint
        self.bearer_token = bearer_token

    def call(self, request: object) -> dict[str, Any]:
        body = json.dumps(request, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
        if not 0 < len(body) <= MAX_HTTP_BYTES:
            raise ServiceError("INVALID_REQUEST", "invalid bounded JSON request")
        query = Request(self.endpoint, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.bearer_token,
        })
        with urlopen(query, timeout=10) as response:
            if response.status != 200:
                raise ServiceError("TRANSPORT_REJECTED", "HTTP transport denied request")
            result = _strict_json(response.read(MAX_HTTP_BYTES + 1))
        if type(result) is not dict or result.get("schema") != SCHEMA:
            raise ServiceError("TRANSPORT_REJECTED", "invalid service response")
        return result
