from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from http.client import HTTPConnection

import pytest

from twelve_six.inference.server import make_server


class DiagnosticBackend:
    eos_token_id = 1
    max_context_tokens = 8

    def __init__(self) -> None:
        self.diagnostic_payload: dict[str, object] = {
            "backend": "first_party_torch",
            "checkpoint_id": "a" * 64,
            "git_sha": "b" * 40,
            "model_spec_sha256": "c" * 64,
            "parameter_count": 10_140,
            "vocab_size": 256,
            "max_context_tokens": 8,
            "tokenizer_version": "s0-byte-v1",
            "tokenizer_config_sha256": "d" * 64,
            "tokenizer_vocab_sha256": "e" * 64,
            "dataset_manifest_sha256": "1" * 64,
            "run_manifest_sha256": "2" * 64,
            "step": 40,
            "tokens_seen": 10_833,
            "device": "cpu",
            "private_note": "must-not-be-served",
        }

    def diagnostics(self) -> dict[str, object]:
        return dict(self.diagnostic_payload)

    def encode(self, text: str) -> list[int]:
        return [0] if text else []

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join("" if token_id == 1 else "x" for token_id in token_ids)

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        return [0.0, 10.0]


@contextmanager
def running_server(
    backend: DiagnosticBackend,
) -> Iterator[tuple[object, tuple[str, int]]]:
    server = make_server(
        backend,
        host="127.0.0.1",
        port=0,
        model_name="s0-bound",
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


def request(
    address: tuple[str, int],
    method: str,
    path: str,
    payload: object | None = None,
) -> tuple[int, dict[str, object], dict[str, str]]:
    connection = HTTPConnection(*address, timeout=5)
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if body is None else {"Content-Type": "application/json"}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    parsed = json.loads(response.read().decode("utf-8"))
    response_headers = {name.lower(): value for name, value in response.getheaders()}
    status = response.status
    connection.close()
    return status, parsed, response_headers


def test_http_surface_is_bound_to_one_server_lifetime_identity() -> None:
    backend = DiagnosticBackend()
    original_checkpoint_id = str(backend.diagnostic_payload["checkpoint_id"])

    with running_server(backend) as (server, address):
        fingerprint = server.serving_fingerprint
        assert isinstance(fingerprint, str) and len(fingerprint) == 64

        # Mutating a backend's later diagnostics cannot silently change the
        # identity advertised by an already-running server.
        backend.diagnostic_payload["checkpoint_id"] = "f" * 64
        backend.diagnostic_payload["git_sha"] = "9" * 40
        backend.diagnostic_payload["private_note"] = "changed-secret"

        health_status, health, health_headers = request(address, "GET", "/healthz")
        models_status, models, model_headers = request(address, "GET", "/v1/models")
        completion_status, completion, completion_headers = request(
            address,
            "POST",
            "/v1/completions",
            {"prompt": "x", "temperature": 0, "max_tokens": 1},
        )
        missing_status, _, missing_headers = request(address, "GET", "/missing")

    assert health_status == models_status == completion_status == 200
    assert missing_status == 404

    identity = health["serving_identity"]
    assert isinstance(identity, dict)
    assert identity["checkpoint_id"] == original_checkpoint_id
    assert identity["git_sha"] == "b" * 40
    assert "private_note" not in identity
    assert health["serving_fingerprint"] == fingerprint

    model = models["data"][0]  # type: ignore[index]
    assert model["metadata"]["serving_fingerprint"] == fingerprint  # type: ignore[index]
    assert model["metadata"]["checkpoint_id"] == original_checkpoint_id  # type: ignore[index]
    assert completion["model"] == "s0-bound"

    for headers in (
        health_headers,
        model_headers,
        completion_headers,
        missing_headers,
    ):
        assert headers["x-12-6-serving-fingerprint"] == fingerprint
        assert headers["x-12-6-checkpoint-id"] == original_checkpoint_id


def test_server_fails_closed_on_non_mapping_diagnostics() -> None:
    class BadDiagnosticsBackend(DiagnosticBackend):
        def diagnostics(self) -> list[str]:  # type: ignore[override]
            return ["not", "a", "mapping"]

    with pytest.raises(TypeError, match="diagnostics must return a mapping"):
        make_server(BadDiagnosticsBackend(), host="127.0.0.1", port=0)


def test_server_without_diagnostics_keeps_generic_http_contract() -> None:
    class GenericBackend:
        eos_token_id = 1
        max_context_tokens = 8

        def encode(self, text: str) -> list[int]:
            return [0] if text else []

        def decode(self, token_ids: Sequence[int]) -> str:
            return ""

        def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
            return [0.0, 10.0]

    server = make_server(GenericBackend(), host="127.0.0.1", port=0)
    try:
        assert server.serving_identity is None
        assert server.serving_fingerprint is None
        assert server.checkpoint_id is None
    finally:
        server.server_close()
