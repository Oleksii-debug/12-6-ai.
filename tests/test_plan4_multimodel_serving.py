"""Plan 4 Section 8: multiroute isolation, failover and recovery conformance."""
import pytest

from tools.plan4_multimodel_serving import MultiModelServing, SCHEMA, ServingError
from tools.plan4_model_service import SCHEMA as SERVICE_SCHEMA


class FixtureClient:
    """Bounded deterministic versioned model service, never paid or networked."""

    def __init__(self, model, fingerprint="a", broken=None):
        self.model = model
        self.fingerprint = fingerprint
        self.epoch = 1
        self.broken = broken
        self.sessions = {}
        self.calls = []

    def call(self, request):
        op, args = request["op"], request["args"]
        self.calls.append(op)
        if op == "identity":
            data = {
                "model": self.model, "epoch": self.epoch,
                "model_spec_sha256": "1" * 64,
                "model_weights_sha256": self.fingerprint * 64,
                "tokenizer_sha256": "3" * 64,
                "backend": "reference-torch-fp32",
            }
        elif self.broken:
            return {"schema": SERVICE_SCHEMA, "ok": False,
                    "error": {"code": self.broken}}
        elif op == "generate":
            data = {"generation": {
                "prompt_token_count": len(args["prompt"]),
                "output_token_ids": [ord(self.fingerprint)],
            }}
        elif op == "stream.start":
            remote = f"remote-{len(self.sessions) + 1}"
            self.sessions[remote] = {"cursor": 0, "cancelled": False}
            data = {"session_id": remote}
        elif op == "stream.cancel":
            self.sessions[args["session_id"]]["cancelled"] = True
            data = {"already_terminal": False}
        elif op == "stream.next":
            session = self.sessions[args["session_id"]]
            if args["cursor"] > session["cursor"]:
                return {"schema": SERVICE_SCHEMA, "ok": False,
                        "error": {"code": "INVALID_CURSOR"}}
            if args["cursor"] == session["cursor"]:
                session["cursor"] += 1
            data = {"cursor": session["cursor"], "event": {
                "kind": "CANCELLED" if session["cancelled"] else "COMPLETED",
                "result": {"prompt_token_count": 1, "output_token_ids": []},
            }}
        elif op == "stream.close":
            del self.sessions[args["session_id"]]
            data = {"closed": True}
        else:
            raise ValueError("unsupported fixture service operation")
        return {"schema": SERVICE_SCHEMA, "ok": True, "result": data}


def packet(op, **kwargs):
    return {"schema": SCHEMA, "op": op, "args": kwargs}


def publish(plane, slot, model, client, *, role="answer", priority=0, reserve=5):
    return plane.publish(
        slot_id=slot, role=role, provider="local", model=model,
        transport="local", client=client, priority=priority, reserved_bytes=reserve,
    )


def ask(plane, *, request_id="q1", role="answer", failover=False, op="generate"):
    return plane.dispatch(packet(
        op, request_id=request_id, role=role, prompt="x", config={"max_new_tokens": 1},
        failover=failover,
    ))


def test_two_models_isolated_by_role_with_exact_usage_and_identity():
    plane = MultiModelServing(budget_bytes=20)
    a, b = FixtureClient("alpha", "a"), FixtureClient("beta", "b")
    publish(plane, "slot-a", "alpha", a, role="answer", priority=2)
    publish(plane, "slot-b", "beta", b, role="teacher", priority=2)
    result = ask(plane)
    assert result["ok"] and result["selected"]["model"] == "alpha"
    assert result["result"]["usage"]["output_tokens"] == 1
    assert result["attempts"][0]["route"]["identity"]["model_weights_sha256"] == "a" * 64
    assert "generate" not in b.calls
    teach = ask(plane, role="teacher", request_id="q2")
    assert teach["selected"]["model"] == "beta"
    assert result["request_id"] != teach["request_id"]


def test_resource_admission_denies_without_eviction():
    plane = MultiModelServing(budget_bytes=5)
    publish(plane, "a", "alpha", FixtureClient("alpha"))
    with pytest.raises(ServingError, match="insufficient"):
        publish(plane, "b", "beta", FixtureClient("beta"))
    state = plane.dispatch(packet("catalog", **{}))["result"]
    assert state["reserved_bytes"] == 5
    assert [s["slot_id"] for s in state["slots"]] == ["a"]


def test_no_silent_failover_even_if_second_route_is_healthy():
    plane = MultiModelServing(budget_bytes=20, failover_roles=("answer",))
    a, b = FixtureClient("alpha", "a", "NOT_READY"), FixtureClient("beta", "b")
    publish(plane, "a", "alpha", a, priority=9)
    publish(plane, "b", "beta", b, priority=1)
    result = ask(plane, failover=False)
    assert not result["ok"] and len(result["attempts"]) == 1
    assert result["attempts"][0]["route"]["model"] == "alpha"
    assert "generate" not in b.calls


def test_authorized_failover_preserves_both_model_attempts():
    plane = MultiModelServing(budget_bytes=20, failover_roles=("answer",))
    a, b = FixtureClient("alpha", "a", "NOT_READY"), FixtureClient("beta", "b")
    publish(plane, "a", "alpha", a, priority=9)
    publish(plane, "b", "beta", b, priority=1)
    result = ask(plane, failover=True)
    assert result["ok"] and result["selected"]["model"] == "beta"
    assert [t["route"]["model"] for t in result["attempts"]] == ["alpha", "beta"]
    assert result["attempts"][0]["error_code"] == "BACKEND_REJECTED"
    assert result["attempts"][1]["error_code"] is None
    assert result["attempts"][0]["route"]["identity"]["model_weights_sha256"] == "a" * 64


def test_unapproved_failover_fails_closed_without_calling_second_model():
    plane = MultiModelServing(budget_bytes=20)
    a, b = FixtureClient("alpha", "a"), FixtureClient("beta", "b")
    publish(plane, "a", "alpha", a, priority=3)
    publish(plane, "b", "beta", b, priority=1)
    result = ask(plane, failover=True)
    assert result["error"]["code"] == "PERMISSION_DENIED"
    assert "generate" not in a.calls and "generate" not in b.calls


def test_restart_identity_drift_is_explicit_and_may_failover_by_host_policy():
    plane = MultiModelServing(budget_bytes=20, failover_roles=("answer",))
    a, b = FixtureClient("alpha", "a"), FixtureClient("beta", "b")
    publish(plane, "a", "alpha", a, priority=9)
    publish(plane, "b", "beta", b, priority=1)
    a.epoch += 1
    result = ask(plane, failover=True)
    assert result["ok"] and result["selected"]["model"] == "beta"
    assert result["attempts"][0]["error_code"] == "IDENTITY_DRIFT"
    assert result["attempts"][0]["route"]["identity"]["epoch"] == 1


def test_stream_stays_on_pinned_model_and_prevents_eviction():
    plane = MultiModelServing(budget_bytes=20, max_sessions=1)
    a = FixtureClient("alpha")
    publish(plane, "a", "alpha", a)
    start = ask(plane, op="stream.start")
    sid = start["result"]["data"]["session_id"]
    assert start["ok"] and len(sid) == 32
    assert not ask(plane, op="stream.start", request_id="q2")["ok"]
    with pytest.raises(ServingError, match="active stream"):
        plane.evict("a")
    assert plane.dispatch(packet("stream.cancel", session_id=sid))["ok"]
    nxt = plane.dispatch(packet("stream.next", session_id=sid, cursor=0))
    assert nxt["result"]["data"]["event"]["kind"] == "CANCELLED"
    assert nxt["selected"]["model"] == "alpha"
    assert nxt["request_id"] == "q1"
    assert plane.dispatch(packet("stream.close", session_id=sid))["ok"]
    assert plane.evict("a")["model"] == "alpha"


def test_atomic_model_reload_refuses_failed_successor():
    plane = MultiModelServing(budget_bytes=6)
    publish(plane, "v1", "alpha", FixtureClient("alpha"), reserve=5)
    with pytest.raises(ServingError, match="capacity"):
        plane.replace(
            "v1", slot_id="v2", role="answer", provider="local", model="beta",
            transport="local", client=FixtureClient("beta"), priority=0,
            reserved_bytes=7,
        )
    assert plane.dispatch(packet("catalog"))["result"]["slots"][0]["slot_id"] == "v1"
    out = plane.replace(
        "v1", slot_id="v2", role="answer", provider="local", model="beta",
        transport="local", client=FixtureClient("beta"), priority=1,
        reserved_bytes=5,
    )
    assert out["retired"]["model"] == "alpha"
    assert out["active"]["model"] == "beta"
    assert ask(plane)["selected"]["slot_id"] == "v2"


def test_invalid_or_duplicate_registration_and_external_admission_denied():
    plane = MultiModelServing(budget_bytes=10)
    publish(plane, "a", "alpha", FixtureClient("alpha"))
    with pytest.raises(ServingError, match="already"):
        publish(plane, "a", "alpha", FixtureClient("alpha"))
    with pytest.raises(ValueError, match="not authorized"):
        plane.publish(
            slot_id="external", role="answer", provider="outside", model="alpha",
            transport="external", client=FixtureClient("alpha"), priority=0,
            reserved_bytes=2,
        )
    for bad in (True, 0, -1):
        with pytest.raises(ServingError):
            MultiModelServing(budget_bytes=bad)


def test_invalid_protocol_and_remote_mutation_blocked():
    plane = MultiModelServing(budget_bytes=10)
    assert plane.dispatch(packet("publish"))["error"]["code"] == "UNSUPPORTED_OP"
    assert plane.dispatch({"schema": SCHEMA, "op": "catalog",
                           "args": {"role": "attack"}})["error"]["code"] == "INVALID_REQUEST"
    assert plane.dispatch({"schema": SCHEMA, "op": "generate",
                           "args": {"role": "answer"}})["error"]["code"] == "INVALID_REQUEST"
    assert plane.dispatch(packet("generate", request_id="x", role="answer",
                                 prompt="x", config={}, failover=1))[
                                     "error"]["code"] == "INVALID_REQUEST"


def test_stale_stream_after_backend_restart_fails_closed_with_original_identity():
    plane = MultiModelServing(budget_bytes=10)
    a = FixtureClient("alpha")
    publish(plane, "a", "alpha", a)
    start = ask(plane, op="stream.start")
    sid = start["result"]["data"]["session_id"]
    a.epoch += 1
    later = plane.dispatch(packet("stream.next", session_id=sid, cursor=0))
    assert later["error"]["code"] == "IDENTITY_DRIFT"
    assert later["attempts"][0]["route"]["identity"]["epoch"] == 1


def test_deterministic_priority_tie_uses_slot_identifier():
    plane = MultiModelServing(budget_bytes=20)
    publish(plane, "b", "beta", FixtureClient("beta"))
    publish(plane, "a", "alpha", FixtureClient("alpha"))
    assert ask(plane)["selected"]["slot_id"] == "a"
