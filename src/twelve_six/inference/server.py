"""Minimal local HTTP server for raw 12-6 Base text completions."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import time
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .contracts import InferenceBackend
from .openai_compat import completion_response

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MAX_REQUEST_BYTES = 1_048_576
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})
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
_SERVING_IDENTITY_FIELDS = (
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
)
_SHA256_HEX = frozenset("0123456789abcdef")


def _capture_serving_identity(
    backend: InferenceBackend,
) -> tuple[dict[str, object] | None, str | None]:
    """Capture one privacy-safe immutable backend identity for server lifetime."""

    diagnostics = getattr(backend, "diagnostics", None)
    if not callable(diagnostics):
        return None, None
    raw = diagnostics()
    if not isinstance(raw, Mapping):
        raise TypeError("backend diagnostics must return a mapping")

    selected = {
        field: raw[field]
        for field in _SERVING_IDENTITY_FIELDS
        if field in raw
    }
    if not selected:
        return None, None
    try:
        encoded = json.dumps(
            selected,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError("backend diagnostics must be canonical JSON values") from exc
    identity = json.loads(encoded.decode("utf-8"))
    return identity, hashlib.sha256(encoded).hexdigest()


def _checkpoint_header(identity: Mapping[str, object] | None) -> str | None:
    if identity is None:
        return None
    value = identity.get("checkpoint_id")
    if (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and set(value) <= _SHA256_HEX
    ):
        return value
    return None


class CompletionHTTPServer(HTTPServer):
    """Serialized S0 HTTP server with one already-verified inference backend."""

    def __init__(
        self,
        server_address: tuple[str, int],
        backend: InferenceBackend,
        *,
        model_name: str,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    ) -> None:
        if (
            not isinstance(max_request_bytes, int)
            or isinstance(max_request_bytes, bool)
            or max_request_bytes < 1
        ):
            raise ValueError("max_request_bytes must be a positive integer")
        self.backend = backend
        self.model_name = model_name
        self.max_request_bytes = max_request_bytes
        self.serving_identity, self.serving_fingerprint = _capture_serving_identity(backend)
        self.checkpoint_id = _checkpoint_header(self.serving_identity)
        super().__init__(server_address, CompletionRequestHandler)


class CompletionRequestHandler(BaseHTTPRequestHandler):
    """OpenAI text-completions subset without chat or hidden prompt semantics."""

    server: CompletionHTTPServer
    server_version = "12-6-base-local/0"
    sys_version = ""

    def log_message(self, format: str, *args: object) -> None:
        # Keep logs text-only and never include request bodies/prompts.
        message = format % args
        print(f"12-6-server {self.client_address[0]} {message}", file=sys.stderr)

    def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        if self.server.serving_fingerprint is not None:
            self.send_header(
                "X-12-6-Serving-Fingerprint",
                self.server.serving_fingerprint,
            )
        if self.server.checkpoint_id is not None:
            self.send_header("X-12-6-Checkpoint-ID", self.server.checkpoint_id)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str, error_type: str) -> None:
        self._write_json(
            status,
            {
                "error": {
                    "message": message,
                    "type": error_type,
                    "param": None,
                    "code": None,
                }
            },
        )

    def do_GET(self) -> None:
        if self.path == "/healthz":
            payload: dict[str, object] = {
                "status": "ok",
                "model": self.server.model_name,
            }
            if self.server.serving_fingerprint is not None:
                payload["serving_fingerprint"] = self.server.serving_fingerprint
            if self.server.serving_identity is not None:
                payload["serving_identity"] = self.server.serving_identity
            self._write_json(HTTPStatus.OK, payload)
            return
        if self.path == "/v1/models":
            model: dict[str, object] = {
                "id": self.server.model_name,
                "object": "model",
                "created": 0,
                "owned_by": "12-6-ai",
            }
            if self.server.serving_fingerprint is not None:
                metadata: dict[str, object] = {
                    "serving_fingerprint": self.server.serving_fingerprint,
                }
                if self.server.checkpoint_id is not None:
                    metadata["checkpoint_id"] = self.server.checkpoint_id
                model["metadata"] = metadata
            self._write_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [model],
                },
            )
            return
        self._error(HTTPStatus.NOT_FOUND, "unknown endpoint", "invalid_request_error")

    def do_POST(self) -> None:
        if self.path != "/v1/completions":
            if self.path == "/v1/chat/completions":
                self._error(
                    HTTPStatus.NOT_FOUND,
                    "chat completions are not supported by raw canonical Base",
                    "invalid_request_error",
                )
            else:
                self._error(HTTPStatus.NOT_FOUND, "unknown endpoint", "invalid_request_error")
            return

        content_type = self.headers.get("Content-Type", "")
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            self._error(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "Content-Type must be application/json",
                "invalid_request_error",
            )
            return

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._error(
                HTTPStatus.LENGTH_REQUIRED,
                "Content-Length is required",
                "invalid_request_error",
            )
            return
        try:
            content_length = int(raw_length)
        except ValueError:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "Content-Length must be an integer",
                "invalid_request_error",
            )
            return
        if content_length < 1:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "request body must not be empty",
                "invalid_request_error",
            )
            return
        if content_length > self.server.max_request_bytes:
            self._error(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "request body exceeds configured byte limit",
                "invalid_request_error",
            )
            return

        raw_body = self.rfile.read(content_length)
        if len(raw_body) != content_length:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "request body ended before Content-Length bytes were received",
                "invalid_request_error",
            )
            return
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(
                HTTPStatus.BAD_REQUEST,
                "request body must be valid UTF-8 JSON",
                "invalid_request_error",
            )
            return
        if not isinstance(payload, dict):
            self._error(
                HTTPStatus.BAD_REQUEST,
                "request JSON must be an object",
                "invalid_request_error",
            )
            return

        unsupported = sorted(set(payload) - _ALLOWED_COMPLETION_FIELDS)
        if unsupported:
            self._error(
                HTTPStatus.BAD_REQUEST,
                f"unsupported request field(s): {', '.join(unsupported)}",
                "invalid_request_error",
            )
            return
        requested_model = payload.get("model")
        if requested_model is not None and requested_model != self.server.model_name:
            self._error(
                HTTPStatus.BAD_REQUEST,
                f"requested model must equal {self.server.model_name!r}",
                "invalid_request_error",
            )
            return

        try:
            response = completion_response(
                self.server.backend,
                payload,
                response_id=f"cmpl-{secrets.token_hex(12)}",
                created=int(time.time()),
                model_name=self.server.model_name,
            )
        except (TypeError, ValueError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc), "invalid_request_error")
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
            )
            return
        self._write_json(HTTPStatus.OK, response)


def make_server(
    backend: InferenceBackend,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    model_name: str = "12-6-base",
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    allow_non_loopback: bool = False,
) -> CompletionHTTPServer:
    """Construct a local server without loading or mutating model/checkpoint state."""

    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer in [0, 65535]")
    if not model_name:
        raise ValueError("model_name must not be empty")
    if host not in _LOOPBACK_HOSTS and not allow_non_loopback:
        raise ValueError(
            "non-loopback bind rejected; use --allow-non-loopback for an explicit remote bind"
        )
    return CompletionHTTPServer(
        (host, port),
        backend,
        model_name=model_name,
        max_request_bytes=max_request_bytes,
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
    server: CompletionHTTPServer,
    *,
    host: str,
    port: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "event": "server_ready",
        "host": host,
        "port": port,
        "model": server.model_name,
        "completion_endpoint": "/v1/completions",
        "chat_semantics": False,
    }
    if server.serving_fingerprint is not None:
        payload["serving_fingerprint"] = server.serving_fingerprint
    if server.serving_identity is not None:
        payload["serving_identity"] = server.serving_identity
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    # Import/load before binding the listening socket: corrupt or incompatible
    # checkpoints fail closed before the server becomes reachable.
    from .first_party import load_first_party_backend

    backend = load_first_party_backend(args.checkpoint)
    server = make_server(
        backend,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        max_request_bytes=args.max_request_bytes,
        allow_non_loopback=args.allow_non_loopback,
    )
    bound_host, bound_port = server.server_address[:2]
    diagnostics = _startup_diagnostics(
        server,
        host=str(bound_host),
        port=int(bound_port),
    )
    if args.json_diagnostics:
        print(json.dumps(diagnostics, sort_keys=True, allow_nan=False), file=sys.stderr)
    else:
        fingerprint = (
            f" fingerprint={server.serving_fingerprint}"
            if server.serving_fingerprint is not None
            else ""
        )
        print(
            f"12-6-server ready http://{bound_host}:{bound_port}/v1/completions "
            f"model={args.model_name}{fingerprint}",
            file=sys.stderr,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("12-6-server stopped", file=sys.stderr)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
