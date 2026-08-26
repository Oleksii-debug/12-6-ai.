from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from http.client import HTTPConnection

import pytest

from twelve_six.inference.server import make_loading_server, make_server


class TinyBackend:
    eos_token_id = None
    max_context_tokens = 8

    def encode(self, text: str) -> list[int]:
        return [0] if text else []

    def decode(self, token_ids: Sequence[int]) -> str:
        return "A" * len(token_ids)

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        del input_ids
        return [0.0, 10.0]

    def diagnostics(self) -> dict[str, object]:
        return {
            "backend": "tiny-test",
            "checkpoint_id": "a" * 64,
            "secret_prompt": "never-expose-me",
        }


@contextmanager
def running_server(
    *,
    request_timeout_seconds: float = 1.0,
) -> Iterator[tuple[str, int]]:
    server = make_server(
        TinyBackend(),
        host="127.0.0.1",
        port=0,
        model_name="s0-hardening-test",
        request_timeout_seconds=request_timeout_seconds,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield str(host), int(port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@contextmanager
def loading_server() -> Iterator[tuple[object, tuple[str, int]]]:
    server = make_loading_server(
        host="127.0.0.1",
        port=0,
        model_name="s0-loading-test",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield server, (str(host), int(port))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def _json_request(
    address: tuple[str, int],
    method: str,
    path: str,
    payload: object | None = None,
) -> tuple[int, dict[str, object]]:
    connection = HTTPConnection(*address, timeout=3)
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if body is None else {"Content-Type": "application/json"}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    parsed = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, parsed


def _json_get(address: tuple[str, int], path: str) -> tuple[int, dict[str, object]]:
    return _json_request(address, "GET", path)


def test_slow_partial_body_does_not_starve_liveness_endpoint() -> None:
    with running_server(request_timeout_seconds=0.1) as address:
        slow = socket.create_connection(address, timeout=2)
        slow.settimeout(2)
        slow.sendall(
            b"POST /v1/completions HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: 100\r\n"
            b"\r\n"
            b"{"
        )

        status, payload = _json_get(address, "/healthz")
        assert status == 200
        assert payload["status"] == "ok"
        assert slow.recv(1) == b""
        slow.close()


def test_loading_readiness_identity_and_draining_are_separate_from_liveness() -> None:
    with loading_server() as (server, address):
        health_status, health = _json_get(address, "/healthz")
        loading_status, loading = _json_get(address, "/readyz")
        models_status, models = _json_get(address, "/v1/models")

        assert health_status == 200
        assert health == {"status": "ok", "model": "s0-loading-test"}
        assert loading_status == 503
        assert loading["runtime_state"] == "loading"
        assert models_status == 200
        assert models["data"][0]["metadata"]["runtime_state"] == "loading"  # type: ignore[index]

        server.install_backend(TinyBackend())  # type: ignore[attr-defined]
        ready_status, ready = _json_get(address, "/readyz")
        models_status, models = _json_get(address, "/v1/models")
        assert ready_status == 200
        assert ready["status"] == "ready"
        assert models_status == 200
        metadata = models["data"][0]["metadata"]  # type: ignore[index]
        assert metadata["runtime_state"] == "ready"
        assert metadata["checkpoint_id"] == "a" * 64
        assert "secret_prompt" not in metadata

        server.runtime.begin_draining()  # type: ignore[attr-defined]
        draining_status, draining = _json_get(address, "/readyz")
        completion_status, completion = _json_request(
            address,
            "POST",
            "/v1/completions",
            {"prompt": "x", "temperature": 0, "max_tokens": 1},
        )
        assert draining_status == 503
        assert draining["runtime_state"] == "draining"
        assert completion_status == 503
        assert completion["error"]["code"] == "model_not_ready"  # type: ignore[index]


def test_status_endpoint_exposes_counters_without_prompt_content() -> None:
    secret = "STATUS_SECRET_72cda9"
    with running_server() as address:
        status, completion = _json_request(
            address,
            "POST",
            "/v1/completions",
            {"prompt": secret, "temperature": 0, "max_tokens": 1},
        )
        assert status == 200
        status_code, runtime_status = _json_get(address, "/statusz")

    assert status_code == 200
    assert runtime_status["accepted_requests"] == 1
    assert runtime_status["completed_requests"] == 1
    assert runtime_status["execution_lanes"] == 1
    assert secret not in json.dumps(runtime_status)
    assert secret not in json.dumps(completion)


def test_http_logs_never_echo_query_or_malformed_request_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    query_secret = "PROMPT_QUERY_SECRET_8f9971"
    malformed_secret = "MALFORMED_SECRET_d1476a"

    with running_server() as address:
        status, _ = _json_get(address, f"/unknown?prompt={query_secret}")
        assert status == 404

        raw = socket.create_connection(address, timeout=2)
        raw.settimeout(2)
        raw.sendall(
            f"BOGUS /unknown?prompt={malformed_secret} HTTP/1.1\r\n"
            "Host: localhost\r\n\r\n".encode()
        )
        while raw.recv(4096):
            pass
        raw.close()

    captured = capsys.readouterr().err
    assert query_secret not in captured
    assert malformed_secret not in captured
    assert "endpoint=other" in captured
    assert "protocol_error" in captured


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_request_timeout_must_be_finite_positive(value: object) -> None:
    with pytest.raises(ValueError, match="request_timeout_seconds"):
        make_server(
            TinyBackend(),
            host="127.0.0.1",
            port=0,
            request_timeout_seconds=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_server_completion_timeout_must_be_finite_positive(value: object) -> None:
    with pytest.raises(ValueError, match="completion_timeout_seconds"):
        make_server(
            TinyBackend(),
            host="127.0.0.1",
            port=0,
            completion_timeout_seconds=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_server_queue_depth_must_be_non_negative_integer(value: object) -> None:
    with pytest.raises(ValueError, match="max_queue_depth"):
        make_server(
            TinyBackend(),
            host="127.0.0.1",
            port=0,
            max_queue_depth=value,  # type: ignore[arg-type]
        )
