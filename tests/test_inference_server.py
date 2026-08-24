from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from http.client import HTTPConnection

import pytest

from twelve_six.inference.server import make_server


class RecordingBackend:
    eos_token_id = 3
    max_context_tokens = 4

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.prompts.append(text)
        return [0] if text else []

    def decode(self, token_ids: Sequence[int]) -> str:
        pieces = {0: "", 1: "A", 2: "B", 3: ""}
        return "".join(pieces[token_id] for token_id in token_ids)

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        generated_count = len(input_ids) - 1
        next_ids = [1, 2, 3]
        next_id = next_ids[min(generated_count, len(next_ids) - 1)]
        logits = [0.0, 0.0, 0.0, 0.0]
        logits[next_id] = 10.0
        return logits


@contextmanager
def running_server(
    backend: RecordingBackend,
    *,
    max_request_bytes: int = 1_048_576,
) -> Iterator[tuple[str, int]]:
    server = make_server(
        backend,
        host="127.0.0.1",
        port=0,
        model_name="s0-test",
        max_request_bytes=max_request_bytes,
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


def request(
    address: tuple[str, int],
    method: str,
    path: str,
    payload: object | None = None,
    *,
    raw_body: bytes | None = None,
    content_type: str = "application/json",
) -> tuple[int, dict[str, object]]:
    connection = HTTPConnection(*address, timeout=5)
    body = raw_body if raw_body is not None else (
        None if payload is None else json.dumps(payload).encode("utf-8")
    )
    headers = {} if body is None else {"Content-Type": content_type}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    parsed = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, parsed


def test_http_completion_preserves_raw_prompt_and_openai_shape() -> None:
    backend = RecordingBackend()
    with running_server(backend) as address:
        status, payload = request(
            address,
            "POST",
            "/v1/completions",
            {"prompt": "raw prompt", "temperature": 0, "max_tokens": 8},
        )

    assert status == 200
    assert backend.prompts == ["raw prompt"]
    assert payload["object"] == "text_completion"
    assert payload["model"] == "s0-test"
    assert payload["choices"][0]["text"] == "AB"  # type: ignore[index]
    assert payload["choices"][0]["finish_reason"] == "stop"  # type: ignore[index]
    assert payload["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 3,
        "total_tokens": 4,
    }


def test_http_seed_stop_and_context_semantics_are_deterministic() -> None:
    backend = RecordingBackend()
    with running_server(backend) as address:
        body = {
            "prompt": "x",
            "temperature": 1.0,
            "top_p": 1.0,
            "seed": 17,
            "max_tokens": 2,
        }
        first_status, first = request(address, "POST", "/v1/completions", body)
        second_status, second = request(address, "POST", "/v1/completions", body)
        stop_status, stopped = request(
            address,
            "POST",
            "/v1/completions",
            {"prompt": "x", "temperature": 0, "max_tokens": 8, "stop": "B"},
        )

    assert first_status == second_status == stop_status == 200
    assert first["choices"][0]["text"] == second["choices"][0]["text"]  # type: ignore[index]
    assert stopped["choices"][0]["text"] == "A"  # type: ignore[index]
    assert stopped["choices"][0]["finish_reason"] == "stop"  # type: ignore[index]


def test_chat_messages_and_streaming_fail_closed() -> None:
    backend = RecordingBackend()
    with running_server(backend) as address:
        chat_status, chat = request(
            address,
            "POST",
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "x"}]},
        )
        messages_status, messages = request(
            address,
            "POST",
            "/v1/completions",
            {"messages": [], "prompt": "x"},
        )
        stream_status, stream = request(
            address,
            "POST",
            "/v1/completions",
            {"prompt": "x", "stream": True},
        )

    assert chat_status == 404
    assert "chat completions" in chat["error"]["message"]  # type: ignore[index]
    assert messages_status == stream_status == 400
    assert "chat/messages" in messages["error"]["message"]  # type: ignore[index]
    assert "stream=true" in stream["error"]["message"]  # type: ignore[index]


def test_protocol_rejects_bad_json_media_type_and_oversized_body() -> None:
    backend = RecordingBackend()
    with running_server(backend, max_request_bytes=32) as address:
        bad_json_status, _ = request(
            address,
            "POST",
            "/v1/completions",
            raw_body=b"{",
        )
        media_status, _ = request(
            address,
            "POST",
            "/v1/completions",
            {"prompt": "x"},
            content_type="text/plain",
        )
        large_status, large = request(
            address,
            "POST",
            "/v1/completions",
            raw_body=b"{" + b"x" * 64 + b"}",
        )

    assert bad_json_status == 400
    assert media_status == 415
    assert large_status == 413
    assert "byte limit" in large["error"]["message"]  # type: ignore[index]


def test_health_models_and_unknown_endpoint() -> None:
    backend = RecordingBackend()
    with running_server(backend) as address:
        health_status, health = request(address, "GET", "/healthz")
        models_status, models = request(address, "GET", "/v1/models")
        missing_status, missing = request(address, "GET", "/nope")

    assert health_status == 200
    assert health == {"status": "ok", "model": "s0-test"}
    assert models_status == 200
    assert models["data"][0]["id"] == "s0-test"  # type: ignore[index]
    assert missing_status == 404
    assert missing["error"]["type"] == "invalid_request_error"  # type: ignore[index]


def test_non_loopback_bind_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="non-loopback"):
        make_server(RecordingBackend(), host="0.0.0.0", port=0)
