"""Plan 4 S6: exact local/HTTP service semantics, recovery and adversarial tests."""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import torch

from twelve_six.model import ModelSpec, TwelveSixDecoder
from twelve_six.tokenization.byte import ByteTokenizer
from tools.inference_runtime import GenerationConfig, ReferenceInference
from tools.plan4_model_service import (
    SCHEMA,
    HttpClient,
    LocalClient,
    ModelService,
    ServiceError,
    serve_loopback,
)

TOKEN = "local-qualification-secret-32bytes"


def request(op, **args):
    return {"schema": SCHEMA, "op": op, "args": args}


@pytest.fixture
def service():
    srv = ModelService(max_sessions=4)
    spec = ModelSpec(
        schema_version=1, vocab_size=256, max_seq_len=32,
        d_model=16, n_layers=1, n_heads=2, n_kv_heads=1,
        head_dim=8, d_ff=32, rope_rotary_dim=8,
    )
    for alias, seed in (("alpha", 42), ("beta", 43)):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = TwelveSixDecoder(spec)
        model.eval()
        srv.register(alias, ReferenceInference(model, ByteTokenizer()))
    return srv


def ok(client, op, **kwargs):
    response = client.call(request(op, **kwargs))
    assert response["schema"] == SCHEMA and response["ok"] is True, response
    return response["result"]


def denied(client, op, code, **kwargs):
    response = client.call(request(op, **kwargs))
    assert response["schema"] == SCHEMA and response["ok"] is False
    assert response["error"]["code"] == code, response
    return response


def test_health_identity_load_generate_unload_and_reuse_canonical_decoder(service):
    client = LocalClient(service)
    assert ok(client, "health")["status"] == "UNLOADED"
    denied(client, "identity", "NOT_READY")
    denied(client, "generate", "NOT_READY", prompt="A", config={"max_new_tokens": 2})
    assert ok(client, "load", model="alpha")["status"] == "READY"
    identity = ok(client, "identity")
    assert identity["model"] == "alpha"
    assert identity["model_weights_sha256"] == service._registered["alpha"].model_weights_sha256
    got = ok(client, "generate", prompt="A", config={"max_new_tokens": 2})
    direct = service._registered["alpha"].generate("A", GenerationConfig(2))
    assert got["generation"]["output_token_ids"] == direct.output_token_ids
    assert got["generation"]["request_id"] == direct.request_id
    assert got["generation"]["result_sha256"] == direct.result_sha256
    assert ok(client, "unload")["status"] == "UNLOADED"
    denied(client, "identity", "NOT_READY")


def test_exact_stream_cursor_idempotence_and_terminal_close(service):
    client = LocalClient(service)
    ok(client, "load", model="alpha")
    start = ok(client, "stream.start", prompt="A", config={"max_new_tokens": 3})
    sid = start["session_id"]
    cursor = 0
    tokens = []
    for _ in range(4):
        ev = ok(client, "stream.next", session_id=sid, cursor=cursor)
        assert ev["cursor"] == cursor + 1
        assert ok(client, "stream.next", session_id=sid, cursor=cursor) == ev
        cursor += 1
        if not ev["terminal"]:
            tokens.append(ev["event"]["token_id"])
    assert ev["terminal"] and ev["event"]["kind"] == "COMPLETED"
    expected = ok(client, "generate", prompt="A", config={"max_new_tokens": 3})
    assert tuple(tokens) == tuple(expected["generation"]["output_token_ids"])
    assert ev["event"]["result"]["result_sha256"] == expected["generation"]["result_sha256"]
    denied(client, "stream.next", "TERMINAL_STREAM", session_id=sid, cursor=cursor)
    assert ok(client, "stream.close", session_id=sid)["closed"]
    denied(client, "stream.next", "UNKNOWN_SESSION", session_id=sid, cursor=cursor)


def test_cancel_before_and_after_token_never_forges_completion(service):
    client = LocalClient(service)
    ok(client, "load", model="alpha")
    first = ok(client, "stream.start", prompt="x", config={"max_new_tokens": 3})
    sid = first["session_id"]
    assert ok(client, "stream.cancel", session_id=sid)["already_terminal"] is False
    cancelled = ok(client, "stream.next", session_id=sid, cursor=0)
    assert cancelled["terminal"] is True
    assert cancelled["event"]["kind"] == "CANCELLED"
    assert cancelled["event"]["result"]["output_token_ids"] == ()
    assert ok(client, "stream.cancel", session_id=sid)["already_terminal"] is True
    ok(client, "stream.close", session_id=sid)
    second = ok(client, "stream.start", prompt="x", config={"max_new_tokens": 3})
    sid = second["session_id"]
    token = ok(client, "stream.next", session_id=sid, cursor=0)
    assert token["event"]["kind"] == "TOKEN"
    ok(client, "stream.cancel", session_id=sid)
    cancelled = ok(client, "stream.next", session_id=sid, cursor=1)
    assert cancelled["event"]["kind"] == "CANCELLED"
    assert cancelled["event"]["result"]["output_token_ids"] == [token["event"]["token_id"]]


def test_active_stream_blocks_unload_and_swap_until_terminal(service):
    client = LocalClient(service)
    ok(client, "load", model="alpha")
    start = ok(client, "stream.start", prompt="z", config={"max_new_tokens": 1})
    denied(client, "unload", "CONFLICT")
    denied(client, "swap", "CONFLICT", model="beta")
    assert ok(client, "health")["active_sessions"] == 1
    ok(client, "stream.next", session_id=start["session_id"], cursor=0)
    ok(client, "stream.next", session_id=start["session_id"], cursor=1)
    assert ok(client, "swap", model="beta")["model"] == "beta"
    assert ok(client, "identity")["model_weights_sha256"] != start["model_weights_sha256"]
    denied(client, "stream.next", "UNKNOWN_SESSION",
           session_id=start["session_id"], cursor=0)


def test_restart_invalidates_stream_and_unloads_model(service):
    client = LocalClient(service)
    ok(client, "load", model="alpha")
    old = ok(client, "stream.start", prompt="x", config={"max_new_tokens": 2})
    epoch = ok(client, "health")["epoch"]
    restarted = ok(client, "restart")
    assert restarted["status"] == "UNLOADED" and restarted["epoch"] == epoch + 1
    denied(client, "stream.next", "NOT_READY", session_id=old["session_id"], cursor=0)
    ok(client, "load", model="alpha")
    denied(client, "stream.next", "UNKNOWN_SESSION",
           session_id=old["session_id"], cursor=0)
    assert ModelService().dispatch(request("health"))["result"]["ready"] is False


def test_model_drift_fails_closed_until_explicit_unload_and_restore(service):
    client = LocalClient(service)
    ok(client, "load", model="alpha")
    model = service._registered["alpha"].model
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        next(model.parameters()).reshape(-1)[0].add_(0.1)
    assert ok(client, "health")["status"] == "FAULTED"
    denied(client, "generate", "NOT_READY", prompt="x", config={"max_new_tokens": 1})
    denied(client, "load", "NOT_READY", model="alpha")
    with torch.no_grad():
        model.load_state_dict(original)
    assert ok(client, "unload")["status"] == "UNLOADED"
    assert ok(client, "load", model="alpha")["status"] == "READY"


@pytest.mark.parametrize("bad", [
    {"schema": "v0", "op": "health", "args": {}},
    {"schema": SCHEMA, "op": "eval", "args": {}},
    {"schema": SCHEMA, "op": "health", "args": {"extra": True}},
    {"schema": SCHEMA, "op": "health"},
    {"schema": SCHEMA, "op": "health", "args": {}, "backend": "vllm"},
    ["health"],
])
def test_fail_closed_protocol_validation(service, bad):
    result = LocalClient(service).call(bad)
    assert not result["ok"] and result["error"]["code"] in {
        "INVALID_REQUEST", "UNSUPPORTED_OP",
    }


def test_invalid_config_unknown_model_and_cursor_fail_closed(service):
    client = LocalClient(service)
    denied(client, "load", "UNKNOWN_MODEL", model="unverified-vllm")
    ok(client, "load", model="alpha")
    denied(client, "load", "CONFLICT", model="alpha")
    denied(client, "generate", "INVALID_CONFIG", prompt="x", config={"max_new_tokens": True})
    denied(client, "generate", "INVALID_CONFIG", prompt="x",
           config={"max_new_tokens": 1, "unknown": "switch"})
    denied(client, "generate", "INVALID_CONFIG", prompt="x",
           config={"max_new_tokens": 1, "strategy": "sample"})
    start = ok(client, "stream.start", prompt="x", config={"max_new_tokens": 2})
    denied(client, "stream.next", "INVALID_REQUEST", session_id=start["session_id"],
           cursor=True)
    denied(client, "stream.next", "STALE_CURSOR", session_id=start["session_id"],
           cursor=3)
    denied(client, "stream.close", "CONFLICT", session_id=start["session_id"])


def test_duplicate_alias_and_noncanonical_untrusted_registration(service):
    with pytest.raises(ServiceError, match="immutable"):
        service.register("alpha", service._registered["alpha"])
    with pytest.raises(ServiceError, match="invalid model alias"):
        service.register("../escape", service._registered["alpha"])
    with pytest.raises(ServiceError, match="trusted reference"):
        service.register("evil", object())
    with pytest.raises(ServiceError, match="session capacity"):
        ModelService(max_sessions=True)


def test_threaded_in_process_requests_share_a_single_serial_state(service):
    client = LocalClient(service)
    ok(client, "load", model="alpha")
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(
            lambda _: client.call(request("generate", prompt="A", config={"max_new_tokens": 2})),
            range(8),
        ))
    assert all(response["ok"] for response in outcomes)
    assert len({x["result"]["generation"]["result_sha256"] for x in outcomes}) == 1


def test_local_and_loopback_http_share_identical_backend_neutral_contract(service):
    local = LocalClient(service)
    server = serve_loopback(service, bearer_token=TOKEN)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1/dispatch"
        remote = HttpClient(url, bearer_token=TOKEN)
        assert local.call(request("health")) == remote.call(request("health"))
        ok(remote, "load", model="alpha")
        assert local.call(request("identity")) == remote.call(request("identity"))
        query = request("generate", prompt="Україна", config={"max_new_tokens": 2})
        assert local.call(query) == remote.call(query)
        stream = ok(remote, "stream.start", prompt="x", config={"max_new_tokens": 2})
        tokens = []
        for cursor in range(3):
            reply = ok(remote, "stream.next", session_id=stream["session_id"],
                       cursor=cursor)
            if reply["event"]["kind"] == "TOKEN":
                tokens.append(reply["event"]["token_id"])
        assert tuple(tokens) == tuple(
            ok(local, "generate", prompt="x", config={"max_new_tokens": 2})[
                "generation"]["output_token_ids"]
        )
        assert ok(remote, "stream.close", session_id=stream["session_id"])["closed"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_loopback_http_rejects_unauthenticated_duplicate_nonfinite_and_oversize(service):
    server = serve_loopback(service, bearer_token=TOKEN)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1/dispatch"

        def post(payload, token=TOKEN):
            req = Request(endpoint, data=payload, method="POST", headers={
                "Content-Type": "application/json", "Authorization": "Bearer " + token,
            })
            with pytest.raises(HTTPError) as caught:
                urlopen(req, timeout=5)
            return caught.value.code

        assert post(b"{}", "incorrect-token-00000000") == 403
        assert post(b'{"schema":"x","schema":"y"}') == 400
        assert post(b'{"number":NaN}') == 400
        assert post(b"{", TOKEN) == 400
        assert post(b"x" * 65537) == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    with pytest.raises(ServiceError, match="loopback"):
        HttpClient("http://example.com/v1/dispatch", bearer_token=TOKEN)
    with pytest.raises(ServiceError, match="strong bearer"):
        serve_loopback(service, bearer_token="short")


def test_closed_completed_generation_has_no_active_sessions(service):
    client = LocalClient(service)
    ok(client, "load", model="alpha")
    stream = ok(client, "stream.start", prompt="x", config={"max_new_tokens": 0})
    result = ok(client, "stream.next", session_id=stream["session_id"], cursor=0)
    assert result["terminal"] and result["event"]["kind"] == "COMPLETED"
    assert ok(client, "health")["active_sessions"] == 0
    ok(client, "stream.close", session_id=stream["session_id"])
    assert ok(client, "unload")["status"] == "UNLOADED"
