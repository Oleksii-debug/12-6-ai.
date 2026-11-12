"""The exact Nika Core provider request over the incumbent local transport."""
from __future__ import annotations

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from test_plan4_model_gateway import TOKEN, make_service, svc

from tools.inference_runtime import GenerationConfig
from tools.plan4_model_gateway import ModelGateway
from tools.plan10_nika_loopback import serve_nika_loopback


def _request(server, payload, *, token=TOKEN):
    request = Request(
        f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.load(response)
    except HTTPError as response:
        with response:
            return response.code, json.load(response)


def _configuration():
    _, client = make_service(17)
    gateway = ModelGateway()
    gateway.register("nika-local", "base", "local", client)
    weights = svc(client, "identity")["model_weights_sha256"]
    return gateway, client, weights


@pytest.fixture
def endpoint():
    gateway, client, weights = _configuration()
    observed = []

    def render(messages):
        observed.append(messages)
        return messages[-1][1]

    with serve_nika_loopback(
        gateway, provider="nika-local", model="base", expected_weights_sha256=weights,
        render_messages=render, generation_config=GenerationConfig(2), bearer_token=TOKEN,
    ) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield server, observed, client
        finally:
            server.shutdown()
            worker.join(timeout=5)


def _payload():
    return {
        "model": "base",
        "messages": [
            {"role": "system", "content": "Speak Ukrainian."},
            {"role": "user", "content": "Привіт"},
        ],
        "temperature": 0,
    }


def test_local_peer_shape_uses_pinned_gateway_identity_and_counts(endpoint):
    server, observed, client = endpoint
    status, response = _request(server, _payload())
    direct = svc(client, "generate", prompt="Привіт", config={"max_new_tokens": 2})["generation"]

    assert status == 200
    assert observed == [(("system", "Speak Ukrainian."), ("user", "Привіт"))]
    assert response["model"] == "base"
    assert response["choices"][0]["message"] == {"role": "assistant", "content": direct["text"]}
    assert response["twelve_six"] == {
        "identity": svc(client, "identity"),
        "result_sha256": direct["result_sha256"],
    }
    assert response["usage"] == {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14}


def test_invalid_route_or_protocol_never_reaches_renderer(endpoint):
    server, observed, _ = endpoint
    valid = _payload()
    for payload, expected in [
        ({**valid, "model": "other"}, 404),
        ({**valid, "model": []}, 400),
        ({**valid, "temperature": True}, 400),
        ({**valid, "temperature": 0.7}, 400),
        ({**valid, "stream": True}, 400),
        ({**valid, "messages": [{"role": "operator", "content": "x"}]}, 400),
    ]:
        assert _request(server, payload)[0] == expected
    assert _request(server, valid, token="wrong")[0] == 403
    assert observed == []


def test_wrong_checkpoint_pin_cannot_be_served():
    gateway, _, _ = _configuration()
    with pytest.raises(ValueError, match="selected local generation model"):
        serve_nika_loopback(
            gateway, provider="nika-local", model="base", expected_weights_sha256="0" * 64,
            render_messages=lambda messages: messages[-1][1],
            generation_config=GenerationConfig(1), bearer_token=TOKEN,
        )


def test_backend_cannot_return_another_checkpoint_under_pinned_alias(endpoint, monkeypatch):
    server, _, client = endpoint
    call = client.call

    def foreign_generation(request):
        response = call(request)
        if request["op"] == "generate":
            response["result"]["generation"]["model_weights_sha256"] = "0" * 64
        return response

    monkeypatch.setattr(client, "call", foreign_generation)
    assert _request(server, _payload()) == (503, {"error": {"code": "MODEL_IDENTITY_MISMATCH"}})
