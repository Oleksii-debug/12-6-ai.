"""Plan 4 Section 7: ModelGateway conformance, transport, trust and restart tests."""
from __future__ import annotations

import threading

import pytest
import torch

from twelve_six.model import ModelSpec, TwelveSixDecoder
from twelve_six.tokenization.byte import ByteTokenizer
from tools.inference_runtime import ReferenceInference
from tools.plan4_model_gateway import (
    SCHEMA,
    GatewayError,
    ModelGateway,
)
from tools.plan4_model_service import (
    HttpClient,
    LocalClient,
    ModelService,
    SCHEMA as SERVICE_SCHEMA,
    serve_loopback,
)

TOKEN = "plan4-gateway-fixture-secret-at-least-16"


def packet(op, **args):
    return {"schema": SCHEMA, "op": op, "args": args}


def ok(gateway, op, **args):
    result = gateway.dispatch(packet(op, **args))
    assert result["schema"] == SCHEMA and result["ok"], result
    return result["result"]


def fail(gateway, op, code, **args):
    result = gateway.dispatch(packet(op, **args))
    assert result["schema"] == SCHEMA and not result["ok"], result
    assert result["error"]["code"] == code, result
    return result


def svc(client, op, **args):
    reply = client.call({"schema": SERVICE_SCHEMA, "op": op, "args": args})
    assert reply["ok"], reply
    return reply["result"]


def make_service(seed, alias="base"):
    model_spec = ModelSpec(
        schema_version=1, vocab_size=256, max_seq_len=32,
        d_model=16, n_layers=1, n_heads=2, n_kv_heads=1,
        head_dim=8, d_ff=32, rope_rotary_dim=8,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TwelveSixDecoder(model_spec)
    model.eval()
    service = ModelService(max_sessions=4)
    service.register(alias, ReferenceInference(model, ByteTokenizer()))
    client = LocalClient(service)
    svc(client, "load", model=alias)
    return service, client


@pytest.fixture
def bound():
    service, client = make_service(17)
    gateway = ModelGateway(max_sessions=3)
    gateway.register("local", "base", "local", client)
    return gateway, service, client


def test_versioned_generation_keeps_exact_model_provider_usage_and_capabilities(bound):
    gateway, _, client = bound
    desc = ok(gateway, "describe", provider="local", model="base")
    assert desc["identity"] == svc(client, "identity")
    assert desc["provider"] == "local" and desc["model"] == "base"
    assert desc["transport"] == "local"
    assert desc["capabilities"] == ["generate", "stream"]
    got = ok(
        gateway, "generate", provider="local", model="base",
        prompt="Україна", config={"max_new_tokens": 2},
    )
    direct = svc(client, "generate", prompt="Україна", config={"max_new_tokens": 2})
    assert got["data"] == direct
    assert got["identity"] == desc["identity"]
    assert got["usage"] == {
        "input_tokens": len("Україна".encode("utf-8")), "output_tokens": 2,
    }


def test_multiple_providers_are_never_substituted_or_hidden(bound):
    gateway, _, _ = bound
    _, alternative = make_service(79)
    gateway.register("alternative", "base", "external", alternative, authorize_external=True)
    a = ok(gateway, "generate", provider="local", model="base",
           prompt="A", config={"max_new_tokens": 1})
    b = ok(gateway, "generate", provider="alternative", model="base",
           prompt="A", config={"max_new_tokens": 1})
    assert a["provider"] == "local" and b["provider"] == "alternative"
    assert a["identity"]["model_weights_sha256"] != b["identity"]["model_weights_sha256"]
    fail(gateway, "describe", "UNKNOWN_ROUTE", provider="unknown", model="base")
    fail(gateway, "generate", "UNKNOWN_ROUTE", provider="unknown", model="base",
         prompt="A", config={"max_new_tokens": 1})


def test_capabilities_and_external_transport_cannot_grant_permissions(bound):
    gateway, _, client = bound
    gateway.register("restricted", "base", "local", client, capabilities=("generate",))
    fail(gateway, "stream.start", "UNSUPPORTED_FEATURE",
         provider="restricted", model="base",
         prompt="A", config={"max_new_tokens": 1})
    with pytest.raises(GatewayError, match="not authorized"):
        gateway.register("untrusted", "base", "external", client)
    with pytest.raises(GatewayError, match="immutable"):
        gateway.register("local", "base", "local", client)
    with pytest.raises(GatewayError, match="invalid capabilities"):
        gateway.register("broken", "base", "local", client, capabilities=(["generate"],))
    with pytest.raises(GatewayError, match="unsupported capabilities"):
        gateway.register("tool", "base", "local", client, capabilities=("execute-tool",))
    with pytest.raises(GatewayError, match="unsupported transport"):
        gateway.register("unknown", "base", "provider-autoswitch", client)


def test_stream_tokens_cursor_retry_and_terminal_identity(bound):
    gateway, _, client = bound
    stream = ok(gateway, "stream.start", provider="local", model="base",
                prompt="A", config={"max_new_tokens": 2})
    sid = stream["data"]["session_id"]
    assert len(sid) == 32 and sid != stream["data"].get("remote_id")
    seen = []
    for cursor in range(3):
        got = ok(gateway, "stream.next", session_id=sid, cursor=cursor)
        assert got == ok(gateway, "stream.next", session_id=sid, cursor=cursor)
        assert got["data"]["cursor"] == cursor + 1
        assert got["provider"] == "local" and got["model"] == "base"
        if got["data"]["event"]["kind"] == "TOKEN":
            seen.append(got["data"]["event"]["token_id"])
    assert got["data"]["event"]["kind"] == "COMPLETED"
    expected = svc(client, "generate", prompt="A", config={"max_new_tokens": 2})
    assert seen == expected["generation"]["output_token_ids"]
    assert got["usage"]["output_tokens"] == 2
    assert ok(gateway, "stream.close", session_id=sid)["data"]["closed"]
    fail(gateway, "stream.next", "UNKNOWN_SESSION", session_id=sid, cursor=3)


def test_cancel_before_next_token_is_not_completion(bound):
    gateway, _, _ = bound
    begin = ok(gateway, "stream.start", provider="local", model="base",
               prompt="x", config={"max_new_tokens": 3})
    sid = begin["data"]["session_id"]
    assert ok(gateway, "stream.cancel", session_id=sid)["data"]["already_terminal"] is False
    end = ok(gateway, "stream.next", session_id=sid, cursor=0)
    assert end["data"]["event"]["kind"] == "CANCELLED"
    assert end["data"]["event"]["result"]["status"] == "CANCELLED"
    ok(gateway, "stream.close", session_id=sid)


def test_session_overflow_does_not_silently_evict(bound):
    gateway, _, _ = bound
    ids = [
        ok(gateway, "stream.start", provider="local", model="base",
           prompt="x", config={"max_new_tokens": 0})["data"]["session_id"]
        for _ in range(3)
    ]
    fail(gateway, "stream.start", "RESOURCE_LIMIT",
         provider="local", model="base", prompt="x", config={"max_new_tokens": 0})
    for sid in ids:
        ok(gateway, "stream.next", session_id=sid, cursor=0)
        ok(gateway, "stream.close", session_id=sid)


def test_restart_changes_epoch_and_refuses_stale_bindings_and_sessions(bound):
    gateway, _, client = bound
    start = ok(gateway, "stream.start", provider="local", model="base",
               prompt="x", config={"max_new_tokens": 2})
    original = ok(gateway, "describe", provider="local", model="base")
    svc(client, "restart")
    fail(gateway, "describe", "BACKEND_REJECTED", provider="local", model="base")
    svc(client, "load", model="base")
    fail(gateway, "describe", "IDENTITY_DRIFT", provider="local", model="base")
    fail(gateway, "stream.next", "IDENTITY_DRIFT",
         session_id=start["data"]["session_id"], cursor=0)
    assert original["identity"]["epoch"] != svc(client, "identity")["epoch"]


def test_weight_drift_fails_without_redirecting_to_other_provider(bound):
    gateway, service, _ = bound
    _, alternative = make_service(31)
    gateway.register("other", "base", "external", alternative, authorize_external=True)
    with torch.no_grad():
        next(service._registered["base"].model.parameters()).reshape(-1)[0].add_(0.2)
    fail(gateway, "generate", "BACKEND_REJECTED", provider="local", model="base",
         prompt="x", config={"max_new_tokens": 2})
    assert ok(gateway, "describe", provider="other", model="base")["provider"] == "other"


def test_invalid_protocol_unknown_operation_and_routing_override_are_rejected(bound):
    gateway, _, _ = bound
    for invalid in (
        {"schema": "v0", "op": "describe", "args": {"provider": "local", "model": "base"}},
        {"schema": SCHEMA, "op": "describe", "args": {"provider": "local"}},
        {"schema": SCHEMA, "op": "describe", "args": {
            "provider": "local", "model": "base", "permissions": ["execute"],
        }},
        {"schema": SCHEMA, "op": "load", "args": {"model": "base"}},
        {"schema": SCHEMA, "op": "generate", "args": {
            "provider": "local", "model": "base", "prompt": "x",
            "config": {"max_new_tokens": 1}, "fallback_provider": "other",
        }},
        {"schema": SCHEMA, "op": "describe", "args": ["local", "base"]},
    ):
        assert gateway.dispatch(invalid)["ok"] is False


def test_real_authenticated_loopback_server_adapter_matches_local_contract(bound):
    gateway, service, _ = bound
    server = serve_loopback(service, bearer_token=TOKEN)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1/dispatch"
        gateway.register("server", "base", "server", HttpClient(endpoint, bearer_token=TOKEN))
        a = ok(gateway, "generate", provider="local", model="base",
               prompt="hello", config={"max_new_tokens": 2})
        b = ok(gateway, "generate", provider="server", model="base",
               prompt="hello", config={"max_new_tokens": 2})
        assert a["data"] == b["data"] and a["identity"] == b["identity"]
        assert a["usage"] == b["usage"]
        assert a["provider"] == "local" and b["provider"] == "server"
        assert a["transport"] == "local" and b["transport"] == "server"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_broken_adapter_and_malformed_response_fail_closed(bound):
    gateway, _, _ = bound

    class Broken:
        def call(self, request):
            return {"schema": "malicious", "ok": True, "result": {"model": "base"}}

    with pytest.raises(GatewayError, match="wrong service contract"):
        gateway.register("broken", "base", "local", Broken())

    class Raises:
        def call(self, request):
            raise RuntimeError("secret credentials: never expose")

    result = gateway.dispatch(packet(
        "generate", provider="local", model="base",
        prompt="x", config={"max_new_tokens": 1},
    ))
    assert result["ok"] is True
    with pytest.raises(GatewayError, match="service"):
        gateway.register("raising", "base", "local", Raises()) if False else (
            gateway.register("raising", "base", "local", Broken())
        )


def test_session_limit_and_registered_alias_validation(bound):
    _, _, client = bound
    with pytest.raises(GatewayError, match="session limit"):
        ModelGateway(max_sessions=True)
    with pytest.raises(GatewayError, match="invalid provider"):
        ModelGateway().register("../other", "base", "local", client)
