"""LOCAL_FREE physical-through-fixture composition of routing and Plan4 serving."""
from dataclasses import replace

import pytest

from tools.owner_adaptive_research_serving_bridge import (
    VerifiedNextStep,
    route_local_generation,
)
from tools.plan4_model_service import SCHEMA as SERVICE_SCHEMA
from tools.plan4_multimodel_serving import MultiModelServing
from twelve_six_agent_runtime.adaptive_research_routing import (
    ResourceReading, RouteError, RoutePolicy, RouteState, Specialist,
)


class LocalClient:
    def __init__(self, model, fingerprint, *, fail=False):
        self.model, self.fingerprint, self.fail = model, fingerprint, fail
        self.calls = []

    def call(self, request):
        op = request["op"]
        self.calls.append(op)
        if op == "identity":
            data = {"model": self.model, "epoch": 1, "model_spec_sha256": "1" * 64,
                    "model_weights_sha256": self.fingerprint * 64,
                    "tokenizer_sha256": "2" * 64, "backend": "fixture"}
            return {"schema": SERVICE_SCHEMA, "ok": True, "result": data}
        if op == "generate":
            if self.fail:
                raise OSError("possibly executed but no receipt")
            return {"schema": SERVICE_SCHEMA, "ok": True,
                    "result": {"generation": {
                        "prompt_token_count": len(request["args"]["prompt"]),
                        "output_token_ids": [ord(self.fingerprint)]}}}
        raise AssertionError("unexpected operation")


def member(node_id, cap, successors=(), *, fingerprint="a", provider="local", charge=0):
    return Specialist(
        node_id, (cap,), provider, fingerprint * 64, 100, 1, charge, 20, tuple(successors),
    )


def policy(**changes):
    fields = {"max_calls": 8, "max_visits_per_node": 4,
              "max_no_progress": 3, "cost_limit_microunits": 0}
    fields.update(changes)
    return RoutePolicy(**fields)


def boot(specs):
    serving = MultiModelServing(budget_bytes=100)
    clients = {}
    for node_id, fingerprint in specs:
        client = LocalClient(node_id, fingerprint)
        clients[node_id] = client
        serving.publish(slot_id=node_id, role=node_id, provider="local", model=node_id,
                        transport="local", client=client, priority=0, reserved_bytes=5)
    return serving, clients


def call(state, members, serving, verifier=None, *, ram=300, settings=None):
    return route_local_generation(
        state, members, settings or policy(), ResourceReading(ram, 4), serving,
        prompt="evaluate bounded hypothesis", max_new_tokens=3,
        verify=verifier if verifier is not None else
        (lambda _: VerifiedNextStep("verify", True)),
    )


def test_real_gateway_two_stage_dynamic_route_with_pinned_identities():
    serving, clients = boot([("reason", "a"), ("audit", "b")])
    nodes = (member("reason", "logic", ("audit",), fingerprint="a"),
             member("audit", "verify", ("reason",), fingerprint="b"))
    state = RouteState("study", "logic")
    first = call(state, nodes, serving)
    assert first.status == "VERIFIED_ROUTE_ADVANCED"
    assert first.reply["selected"]["model"] == "reason"
    assert first.state.history == ("reason",)
    second = call(first.state, nodes, serving,
                  lambda _: VerifiedNextStep("logic", True))
    assert second.status == "VERIFIED_ROUTE_ADVANCED"
    assert second.state.history == ("reason", "audit")
    assert second.state.receipt_ids == ("receipt-" + "r" +
           __import__("hashlib").sha256(b'{"revision":0,"task":"study"}').hexdigest()[:28],
           "receipt-" + "r" +
           __import__("hashlib").sha256(b'{"revision":1,"task":"study"}').hexdigest()[:28])
    assert clients["reason"].calls.count("generate") == 1
    assert clients["audit"].calls.count("generate") == 1


def test_low_ram_wait_does_not_call_model():
    serving, clients = boot([("reason", "a")])
    nodes = (member("reason", "logic", (), fingerprint="a"),)
    decision = call(RouteState("study", "logic"), nodes, serving, ram=50)
    assert decision.status == "WAIT_RESOURCES"
    assert clients["reason"].calls.count("generate") == 0


def test_identity_mismatch_is_fail_closed_before_generation():
    serving, clients = boot([("reason", "a")])
    nodes = (member("reason", "logic", (), fingerprint="b"),)
    with pytest.raises(RouteError, match="identity drift"):
        call(RouteState("study", "logic"), nodes, serving)
    assert clients["reason"].calls.count("generate") == 0


def test_unverified_verdict_does_not_advance_or_forge_learning():
    serving, clients = boot([("reason", "a")])
    nodes = (member("reason", "logic", (), fingerprint="a"),)
    start = RouteState("study", "logic")
    out = call(start, nodes, serving, lambda _: None)
    assert out.status == "UNVERIFIED" and out.state == start
    assert clients["reason"].calls.count("generate") == 1
    with pytest.raises(RouteError, match="untrusted verifier verdict"):
        call(start, nodes, serving, lambda _: "verified")


def test_failed_provider_is_not_retried():
    serving, clients = boot([("reason", "a")])
    clients["reason"].fail = True
    nodes = (member("reason", "logic", ("reason",), fingerprint="a"),)
    state = RouteState("study", "logic")
    out = call(state, nodes, serving)
    assert out.status == "SERVING_FAILED_NO_RETRY" and out.state == state
    assert clients["reason"].calls.count("generate") == 1


def test_paid_or_external_specialists_are_not_invoked():
    serving, clients = boot([("reason", "a")])
    state = RouteState("study", "logic")
    nodes = (member("reason", "logic", (), charge=1),)
    with pytest.raises(RouteError, match="only authorized free local"):
        call(state, nodes, serving,
             settings=policy(cost_limit_microunits=5, allow_paid=True))
    nodes = (replace(member("reason", "logic"), provider="external"),)
    with pytest.raises(RouteError, match="only authorized free local"):
        call(state, nodes, serving, settings=policy(allow_external=True))
    assert clients["reason"].calls.count("generate") == 0


def test_incompatible_serving_role_and_bad_boundaries():
    serving, clients = boot([("other", "a")])
    nodes = (member("reason", "logic", (), fingerprint="a"),)
    with pytest.raises(RouteError, match="exactly one"):
        call(RouteState("study", "logic"), nodes, serving)
    assert clients["other"].calls.count("generate") == 0
    with pytest.raises(RouteError, match="bounded"):
        route_local_generation(RouteState("study", "logic"), nodes, policy(),
                               ResourceReading(200, 4), serving, prompt="hi",
                               max_new_tokens=True, verify=lambda _: None)
