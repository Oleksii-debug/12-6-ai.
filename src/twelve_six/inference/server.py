"""Local HTTP server for raw 12-6 Base text completions."""

from __future__ import annotations

import argparse
import json
import math
import secrets
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .contracts import InferenceBackend
from .serving_runtime import (
    DEFAULT_COMPLETION_TIMEOUT_SECONDS,
    DEFAULT_MAX_QUEUE_DEPTH,
    ServingOverloadedError,
    ServingRequestTimeoutError,
    ServingRuntime,
    ServingUnavailableError,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MAX_REQUEST_BYTES = 1_048_576
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})
_KNOWN_LOG_ENDPOINTS = frozenset(
    {
        "/healthz",
        "/readyz",
        "/statusz",
        "/v1/models",
        "/v1/completions",
        "/v1/chat/completions",
    }
)
_ALLOWED_COMPLETION_FIELDS = frozenset(
    {
        "prompt",
        "messages",
        "model",
        "max_tokens",
        "temperature",
        "top_p",
        "seed",
        "stop",
        "n",
        "stream",
        "echo",
        "logprobs",
    }
)


def _require_positive_finite(value: object, *, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be finite and > 0")
    return float(value)


class CompletionHTTPServer(ThreadingHTTPServer):
    """Concurrent local HTTP transport with one bounded model execution lane."""

    daemon_threads = False
    block_on_close = True

    def __init__(
        self,
        server_address: tuple[str, int],
        backend: InferenceBackend | None,
        *,
        model_name: str,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        completion_timeout_seconds: float = DEFAULT_COMPLETION_TIMEOUT_SECONDS,
        max_queue_depth: int = DEFAULT_MAX_QUEUE_DEPTH,
    ) -> None:
        if (
            not isinstance(max_request_bytes, int)
            or isinstance(max_request_bytes, bool)
            or max_request_bytes < 1
        ):
            raise ValueError("max_request_bytes must be a positive integer")
        if (
            not isinstance(max_queue_depth, int)
            or isinstance(max_queue_depth, bool)
            or max_queue_depth < 0
        ):
            raise ValueError("max_queue_depth must be a non-negative integer")
        if not model_name:
            raise ValueError("model_name must not be empty")
        self.model_name = model_name
        self.max_request_bytes = max_request_bytes
        self.request_timeout_seconds = _require_positive_finite(
            request_timeout_seconds,
            name="request_timeout_seconds",
        )
        self.completion_timeout_seconds = _require_positive_finite(
            completion_timeout_seconds,
            name="completion_timeout_seconds",
        )
        self.backend = backend
        super().__init__(server_address, CompletionRequestHandler)
        self.runtime = ServingRuntime(
            backend,
            model_name=model_name,
            max_queue_depth=max_queue_depth,
        )

    def get_request(self):  # type: ignore[no-untyped-def]
        """Bound socket I/O for each accepted client independently."""

        request, client_address = super().get_request()
        request.settimeout(self.request_timeout_seconds)
        return request, client_address

    def install_backend(self, backend: InferenceBackend) -> None:
        """Transition a loading server to ready exactly once."""

        self.runtime.install_backend(backend)
        self.backend = backend

    def fail_loading(self, exc: BaseException) -> None:
        self.runtime.fail_loading(exc)

    def server_close(self) -> None:
        """Stop admissions, drain accepted model work, then close client threads."""

        self.runtime.close(drain=True)
        super().server_close()


class CompletionRequestHandler(BaseHTTPRequestHandler):
    """OpenAI text-completions subset without chat or hidden prompt semantics."""

    server: CompletionHTTPServer
    server_version = "12-6-base-local/1"
    sys_version = ""

    def _log_endpoint(self) -> str:
        path = getattr(self, "path", "").split("?", 1)[0].split("#", 1)[0]
        return path if path in _KNOWN_LOG_ENDPOINTS else "other"

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        """Log only endpoint class, never arbitrary URL/query/request text."""

        method = getattr(self, "command", None)
        safe_method = method if method in {"GET", "POST", "HEAD"} else "other"
        print(
            "12-6-server "
            f"client={self.client_address[0]} method={safe_method} "
            f"endpoint={self._log_endpoint()} status={code} size={size}",
            file=sys.stderr,
        )

    def log_error(self, format: str, *args: object) -> None:
        # BaseHTTPRequestHandler may include the raw request line in protocol
        # error details. Drop those attacker-controlled values from logs.
        del format, args
        print(
            f"12-6-server client={self.client_address[0]} protocol_error",
            file=sys.stderr,
        )

    def log_message(self, format: str, *args: object) -> None:
        # Defensive fallback for any stdlib path not routed through log_request.
        del format, args
        print(
            f"12-6-server client={self.client_address[0]} protocol_event",
            file=sys.stderr,
        )

    def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _error(
        self,
        status: HTTPStatus,
        message: str,
        error_type: str,
        *,
        code: str | None = None,
    ) -> None:
        self._write_json(
            status,
            {
                "error": {
                    "message": message,
                    "type": error_type,
                    "param": None,
                    "code": code,
                }
            },
        )

    def do_GET(self) -> None:
        if self.path == "/healthz":
            # Liveness deliberately does not imply model readiness. Keep the
            # historical response shape for existing local integrations.
            self._write_json(
                HTTPStatus.OK,
                {"status": "ok", "model": self.server.model_name},
            )
            return
        if self.path == "/readyz":
            runtime_status = self.server.runtime.status()
            ready = bool(runtime_status["ready"])
            self._write_json(
                HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "status": "ready" if ready else "not_ready",
                    "model": self.server.model_name,
                    "runtime_state": runtime_status["state"],
                },
            )
            return
        if self.path == "/statusz":
            self._write_json(HTTPStatus.OK, self.server.runtime.status())
            return
        if self.path == "/v1/models":
            self._write_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.model_name,
                            "object": "model",
                            "created": 0,
                            "owned_by": "12-6-ai",
                            "metadata": self.server.runtime.model_identity(),
                        }
                    ],
                },
            )
            return
        self._error(
            HTTPStatus.NOT_FOUND,
            "unknown endpoint",
            "invalid_request_error",
            code="unknown_endpoint",
        )

    def do_POST(self) -> None:
        if self.path != "/v1/completions":
            if self.path == "/v1/chat/completions":
                self._error(
                    HTTPStatus.NOT_FOUND,
                    "chat completions are not supported by raw canonical Base",
                    "invalid_request_error",
                    code="chat_not_supported",
                )
            else:
                self._error(
                    HTTPStatus.NOT_FOUND,
                    "unknown endpoint",
                    "invalid_request_error",
                    code="unknown_endpoint",
                )
            return

        content_type = self.headers.get("Content-Type", "")
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            self._error(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "Content-Type must be application/json",
                "invalid_request_error",
                code="invalid_content_type",
            )
            return

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._error(
                HTTPStatus.LENGTH_REQUIRED,
                "Content-Length is required",
                "invalid_request_error",
                code="content_length_required",
            )
            return
        try:
            content_length = int(raw_length)
        except ValueError:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "Content-Length must be an integer",
                "invalid_request_error",
                code="invalid_content_length",
            )
            return
        if content_length < 1:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "request body must not be empty",
                "invalid_request_error",
                code="empty_request_body",
            )
            return
        if content_length > self.server.max_request_bytes:
            self._error(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "request body exceeds configured byte limit",
                "invalid_request_error",
                code="request_too_large",
            )
            return

        raw_body = self.rfile.read(content_length)
        if len(raw_body) != content_length:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "request body ended before Content-Length bytes were received",
                "invalid_request_error",
                code="incomplete_request_body",
            )
            return
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(
                HTTPStatus.BAD_REQUEST,
                "request body must be valid UTF-8 JSON",
                "invalid_request_error",
                code="invalid_json",
            )
            return
        if not isinstance(payload, dict):
            self._error(
                HTTPStatus.BAD_REQUEST,
                "request JSON must be an object",
                "invalid_request_error",
                code="invalid_request_object",
            )
            return

        unsupported = sorted(set(payload) - _ALLOWED_COMPLETION_FIELDS)
        if unsupported:
            self._error(
                HTTPStatus.BAD_REQUEST,
                f"unsupported request field(s): {', '.join(unsupported)}",
                "invalid_request_error",
                code="unsupported_request_field",
            )
            return
        requested_model = payload.get("model")
        if requested_model is not None and requested_model != self.server.model_name:
            self._error(
                HTTPStatus.BAD_REQUEST,
                f"requested model must equal {self.server.model_name!r}",
                "invalid_request_error",
                code="model_mismatch",
            )
            return

        try:
            response = self.server.runtime.submit(
                payload,
                response_id=f"cmpl-{secrets.token_hex(12)}",
                created=int(time.time()),
                timeout_seconds=self.server.completion_timeout_seconds,
            )
        except ServingUnavailableError:
            self._error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "model is not ready to accept completions",
                "server_error",
                code="model_not_ready",
            )
            return
        except ServingOverloadedError:
            self._error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "serving request queue is full",
                "server_error",
                code="queue_full",
            )
            return
        except ServingRequestTimeoutError as exc:
            message = (
                "completion timed out while executing; active model work cannot "
                "yet be preempted safely"
                if exc.execution_started
                else "completion timed out before model execution started"
            )
            self._error(
                HTTPStatus.GATEWAY_TIMEOUT,
                message,
                "server_error",
                code="completion_timeout",
            )
            return
        except (TypeError, ValueError) as exc:
            self._error(
                HTTPStatus.BAD_REQUEST,
                str(exc),
                "invalid_request_error",
                code="invalid_completion_request",
            )
            return
        except (RuntimeError, OSError) as exc:  # pragma: no cover - backend/system failure
            print(
                f"12-6-server internal_error={type(exc).__name__}",
                file=sys.stderr,
            )
            self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "internal server error",
                "server_error",
                code="internal_error",
            )
            return
        self._write_json(HTTPStatus.OK, response)


def _validate_bind(host: str, port: int, *, allow_non_loopback: bool) -> None:
    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer in [0, 65535]")
    if host not in _LOOPBACK_HOSTS and not allow_non_loopback:
        raise ValueError(
            "non-loopback bind rejected; use --allow-non-loopback for an explicit remote bind"
        )


def make_server(
    backend: InferenceBackend,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    model_name: str = "12-6-base",
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    completion_timeout_seconds: float = DEFAULT_COMPLETION_TIMEOUT_SECONDS,
    max_queue_depth: int = DEFAULT_MAX_QUEUE_DEPTH,
    allow_non_loopback: bool = False,
) -> CompletionHTTPServer:
    """Construct a ready local server around an already-verified backend."""

    _validate_bind(host, port, allow_non_loopback=allow_non_loopback)
    return CompletionHTTPServer(
        (host, port),
        backend,
        model_name=model_name,
        max_request_bytes=max_request_bytes,
        request_timeout_seconds=request_timeout_seconds,
        completion_timeout_seconds=completion_timeout_seconds,
        max_queue_depth=max_queue_depth,
    )


def make_loading_server(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    model_name: str = "12-6-base",
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    completion_timeout_seconds: float = DEFAULT_COMPLETION_TIMEOUT_SECONDS,
    max_queue_depth: int = DEFAULT_MAX_QUEUE_DEPTH,
    allow_non_loopback: bool = False,
) -> CompletionHTTPServer:
    """Construct a liveness-only server that starts in explicit loading state."""

    _validate_bind(host, port, allow_non_loopback=allow_non_loopback)
    return CompletionHTTPServer(
        (host, port),
        None,
        model_name=model_name,
        max_request_bytes=max_request_bytes,
        request_timeout_seconds=request_timeout_seconds,
        completion_timeout_seconds=completion_timeout_seconds,
        max_queue_depth=max_queue_depth,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve verified 12-6 Base raw text completions on a local HTTP endpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model-name", default="12-6-base")
    parser.add_argument("--max-request-bytes", type=int, default=DEFAULT_MAX_REQUEST_BYTES)
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="Per-client socket I/O timeout.",
    )
    parser.add_argument(
        "--completion-timeout-seconds",
        type=float,
        default=DEFAULT_COMPLETION_TIMEOUT_SECONDS,
        help=(
            "Maximum HTTP wait for one accepted completion. Queued work is cancelled "
            "when possible; already-running model work is not force-preempted."
        ),
    )
    parser.add_argument(
        "--max-queue-depth",
        type=int,
        default=DEFAULT_MAX_QUEUE_DEPTH,
        help="Maximum accepted completion requests waiting behind the model execution lane.",
    )
    parser.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="Explicitly allow binding to an address other than localhost/loopback.",
    )
    parser.add_argument(
        "--json-diagnostics",
        action="store_true",
        help="Emit one JSON startup diagnostic object on stderr.",
    )
    return parser


def _startup_diagnostics(
    backend: InferenceBackend,
    *,
    host: str,
    port: int,
    model_name: str,
    serving: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "event": "server_ready",
        "host": host,
        "port": port,
        "model": model_name,
        "completion_endpoint": "/v1/completions",
        "readiness_endpoint": "/readyz",
        "status_endpoint": "/statusz",
        "chat_semantics": False,
        "streaming": False,
    }
    diagnostics = getattr(backend, "diagnostics", None)
    if callable(diagnostics):
        payload["backend"] = diagnostics()
    if serving is not None:
        payload["serving"] = serving
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    # The canonical CLI keeps the stronger existing property: verify/load the
    # checkpoint before binding. ``make_loading_server`` exists for embedding
    # hosts that deliberately want liveness while an out-of-band loader runs.
    from .first_party import load_first_party_backend

    backend = load_first_party_backend(args.checkpoint)
    server = make_server(
        backend,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        max_request_bytes=args.max_request_bytes,
        request_timeout_seconds=args.request_timeout_seconds,
        completion_timeout_seconds=args.completion_timeout_seconds,
        max_queue_depth=args.max_queue_depth,
        allow_non_loopback=args.allow_non_loopback,
    )
    bound_host, bound_port = server.server_address[:2]
    diagnostics = _startup_diagnostics(
        backend,
        host=str(bound_host),
        port=int(bound_port),
        model_name=args.model_name,
        serving=server.runtime.status(),
    )
    if args.json_diagnostics:
        print(json.dumps(diagnostics, sort_keys=True, allow_nan=False), file=sys.stderr)
    else:
        print(
            f"12-6-server ready http://{bound_host}:{bound_port}/v1/completions "
            f"model={args.model_name}",
            file=sys.stderr,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("12-6-server draining", file=sys.stderr)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
