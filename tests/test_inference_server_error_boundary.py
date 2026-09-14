from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from http.client import HTTPConnection

import pytest

from twelve_six.inference.server import CompletionHTTPServer, make_server


_INTERNAL_SECRET = "BACKEND_INTERNAL_SECRET_5f8d67"


class SecretFailingBackend:
    eos_token_id = None
    max_context_tokens = 8

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, text: str) -> list[int]:
        return [0] if text else []

    def decode(self, token_ids: Sequence[int]) -> str:
        return "A" * len(token_ids)

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        del input_ids
        self.calls += 1
        raise ValueError(_INTERNAL_SECRET)

    def diagnostics(self) -> dict[str, object]:
        return {"backend": "secret-failing-test"}


@contextmanager
def running_server(
    backend: SecretFailingBackend,
) -> Iterator[tuple[CompletionHTTPServer, tuple[str, int]]]:
    server = make_server(
        backend,
        host="127.0.0.1",
        port=0,
        model_name="model341-error-boundary-test",
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
    payload: dict[str, object],
) -> tuple[int, dict[str, object]]:
    connection = HTTPConnection(*address, timeout=3)
    body = json.dumps(payload).encode("utf-8")
    connection.request(
        "POST",
        "/v1/completions",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    parsed = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, parsed


def test_backend_value_error_is_sanitized_as_internal_server_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    backend = SecretFailingBackend()
    with running_server(backend) as (_server, address):
        status, payload = _json_request(
            address,
            {"prompt": "x", "temperature": 0, "max_tokens": 1},
        )

    captured = capsys.readouterr().err
    assert status == 500
    assert payload["error"]["code"] == "internal_error"  # type: ignore[index]
    assert payload["error"]["message"] == "internal server error"  # type: ignore[index]
    assert _INTERNAL_SECRET not in json.dumps(payload)
    assert _INTERNAL_SECRET not in captured
    assert "internal_error=ValueError" in captured
    assert backend.calls == 1


def test_invalid_client_request_is_rejected_before_runtime_admission() -> None:
    backend = SecretFailingBackend()
    with running_server(backend) as (server, address):
        status, payload = _json_request(
            address,
            {"prompt": "x", "max_tokens": "not-an-integer"},
        )
        runtime_status = server.runtime.status()

    assert status == 400
    assert payload["error"]["code"] == "invalid_completion_request"  # type: ignore[index]
    assert payload["error"]["message"] == "max_tokens must be an integer"  # type: ignore[index]
    assert runtime_status["accepted_requests"] == 0
    assert runtime_status["failed_requests"] == 0
    assert backend.calls == 0
