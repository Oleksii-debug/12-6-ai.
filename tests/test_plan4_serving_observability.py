"""Plan 4 Section 9: fixture-only observability, capacity, negative/recovery tests."""
from __future__ import annotations

import threading

import pytest

from tools.plan4_model_service import SCHEMA as SERVICE_SCHEMA
from tools.plan4_multimodel_serving import MultiModelServing, SCHEMA as SERVING_SCHEMA
from tools.plan4_serving_observability import (
    SCHEMA, ObservabilityError, ServingObserver,
)


class ModelFixture:
    def __init__(self, model="tiny", broken=None):
        self.model = model
        self.epoch = 1
        self.broken = broken
        self.calls = []
        self.entered = None
        self.release = None
        self.sessions = {}

    def call(self, request):
        op, args = request["op"], request["args"]
        self.calls.append(op)
        if op == "identity":
            data = {
                "model": self.model, "epoch": self.epoch,
                "model_spec_sha256": "1" * 64,
                "model_weights_sha256": "2" * 64,
                "tokenizer_sha256": "3" * 64,
                "backend": "reference-torch-fp32",
            }
        elif self.broken:
            return {"schema": SERVICE_SCHEMA, "ok": False,
                    "error": {"code": self.broken}}
        elif op == "generate":
            if self.entered is not None:
                self.entered.set()
                assert self.release.wait(2)
            data = {"generation": {"prompt_token_count": 1,
                                   "output_token_ids": [1]}}
        elif op == "stream.start":
            sid = f"remote-{len(self.sessions) + 1}"
            self.sessions[sid] = True
            data = {"session_id": sid}
        elif op == "stream.cancel":
            data = {"already_terminal": False}
        elif op == "stream.next":
            data = {"cursor": 1, "terminal": True, "event": {
                "kind": "CANCELLED",
                "result": {"prompt_token_count": 1, "output_token_ids": []},
            }}
        elif op == "stream.close":
            data = {"closed": True}
        else:
            raise ValueError("unknown fixture operation")
        return {"schema": SERVICE_SCHEMA, "ok": True, "result": data}


def make_plane(*, role="answer", transport="local", broken=None):
    plane = MultiModelServing(budget_bytes=40, max_sessions=4)
    fixture = ModelFixture(broken=broken)
    plane.publish(slot_id="one", role=role, provider="fixture",
                  model="tiny", transport=transport, client=fixture,
                  authorize_external=(transport == "external"),
                  priority=1, reserved_bytes=5)
    return plane, fixture


def request(op="generate", *, prompt="sensitive text", request_id="req1"):
    args = {
        "request_id": request_id, "role": "answer",
        "prompt": prompt, "config": {"max_new_tokens": 1},
        "failover": False,
    }
    return {"schema": SERVING_SCHEMA, "op": op, "args": args}


def test_pass_through_identity_and_bounded_latency_throughput():
    plane, client = make_plane()
    monitor = ServingObserver(plane, memory_probe=lambda: 1024)
    original = monitor.dispatch(request())
    assert original["ok"] and original["selected"]["model"] == "tiny"
    assert "generate" in client.calls
    stats = monitor.metrics()
    assert stats["schema"] == SCHEMA
    assert stats["completed_requests"] == 1
    assert stats["successful_requests"] == 1
    assert stats["latency_mean_ns"] >= 0
    assert stats["throughput_rps"] > 0
    assert stats["capacity"]["budget_bytes_declared"] == 40
    assert stats["capacity"]["reserved_bytes_declared"] == 5
    assert stats["capacity"]["actual_gpu_memory_bytes"] is None
    assert stats["process_memory_bytes"] == 1024


def test_overload_fails_without_second_generation_and_retains_safe_id():
    plane, client = make_plane()
    entered, release = threading.Event(), threading.Event()
    client.entered, client.release = entered, release
    monitor = ServingObserver(plane, max_inflight=1)
    outcomes = []
    worker = threading.Thread(target=lambda: outcomes.append(monitor.dispatch(request())))
    worker.start()
    assert entered.wait(2)
    denied = monitor.dispatch(request(request_id="other"))
    assert denied["error"]["code"] == "OVERLOADED"
    assert denied["request_id"] == "other"
    release.set()
    worker.join(2)
    assert not worker.is_alive() and outcomes[0]["ok"]
    assert client.calls.count("generate") == 1
    assert monitor.metrics()["rejected_requests"] == 1


def test_queue_timeout_is_bounded_and_no_implicit_retry():
    plane, client = make_plane()
    entered, release = threading.Event(), threading.Event()
    client.entered, client.release = entered, release
    monitor = ServingObserver(plane, max_inflight=1, max_queue=1,
                              queue_timeout_s=0.001)
    worker = threading.Thread(target=lambda: monitor.dispatch(request()))
    worker.start()
    assert entered.wait(2)
    denied = monitor.dispatch(request(request_id="queued"))
    assert denied["error"]["code"] == "QUEUE_TIMEOUT"
    release.set()
    worker.join(2)
    assert monitor.metrics()["queued"] == 0
    assert monitor.metrics()["peak_queue"] == 1
    assert monitor.metrics()["errors_by_code"]["QUEUE_TIMEOUT"] == 1


def test_backend_failure_keeps_provenance_and_does_not_repeat():
    plane, client = make_plane(broken="NOT_READY")
    monitor = ServingObserver(plane)
    got = monitor.dispatch(request())
    assert not got["ok"] and got["attempts"][0]["route"]["model"] == "tiny"
    assert client.calls.count("generate") == 1
    assert monitor.metrics()["errors_by_code"]["ALL_ROUTES_FAILED"] == 1


def test_unknown_and_malformed_errors_are_bounded_taxonomy():
    plane, _ = make_plane()
    monitor = ServingObserver(plane)
    malformed = monitor.dispatch({"schema": SERVING_SCHEMA, "op": "unknown", "args": {}})
    assert malformed["error"]["code"] == "UNSUPPORTED_OP"
    assert monitor.metrics()["errors_by_code"]["UNSUPPORTED_OP"] == 1


def test_canary_is_readiness_only_and_never_discloses_generated_text():
    plane, fixture = make_plane()
    monitor = ServingObserver(plane)
    status = monitor.canary(role="answer")
    assert status["ready"] is True and status["evaluation"] is False
    assert status["model_identity"]["model_weights_sha256"] == "2" * 64
    assert "generation" not in status and "prompt" not in str(status)
    assert "generate" in fixture.calls
    assert monitor.metrics()["canary_ready"] == 1


def test_canary_denies_missing_role_and_external_default_without_inference():
    plane, fixture = make_plane(transport="external")
    monitor = ServingObserver(plane)
    missing = monitor.canary(role="teacher")
    denied = monitor.canary(role="answer")
    assert missing["reason"] == "NO_ROLE"
    assert denied["reason"] == "EXTERNAL_DENIED"
    assert fixture.calls.count("generate") == 0
    assert monitor.metrics()["canary_count"] == 2


def test_canary_backend_failure_never_claims_quality_evaluation():
    plane, _ = make_plane(broken="NOT_READY")
    monitor = ServingObserver(plane)
    result = monitor.canary(role="answer")
    assert result["ready"] is False
    assert result["evaluation"] is False
    assert monitor.metrics()["canary_ready"] == 0


def test_host_model_load_success_and_failure_both_measured():
    plane, _ = make_plane()
    monitor = ServingObserver(plane)
    assert monitor.record_model_load(lambda: "loaded") == "loaded"
    with pytest.raises(RuntimeError, match="load unavailable"):
        monitor.record_model_load(
            lambda: (_ for _ in ()).throw(RuntimeError("load unavailable")))
    stats = monitor.metrics()
    assert stats["model_load_count"] == 2
    assert stats["model_load_failures"] == 1
    assert stats["model_load_total_ns"] >= 0


@pytest.mark.parametrize("bad", [True, -1, -1.2, "1024", float("nan")])
def test_memory_probe_untrusted_results_are_unknown_not_zero(bad):
    plane, _ = make_plane()
    monitor = ServingObserver(plane, memory_probe=lambda: bad)
    stats = monitor.metrics()
    assert stats["process_memory_bytes"] is None
    assert stats["process_memory_source"] == "UNAVAILABLE"


def test_memory_probe_exception_is_unsupported_not_green():
    plane, _ = make_plane()

    def unavailable():
        raise RuntimeError("host denies memory")

    stats = ServingObserver(plane, memory_probe=unavailable).metrics()
    assert stats["process_memory_bytes"] is None


def test_cancel_receipts_are_counted_without_opening_model_authority():
    plane, _ = make_plane()
    monitor = ServingObserver(plane)
    first = monitor.dispatch(request("stream.start"))
    sid = first["result"]["data"]["session_id"]
    second = monitor.dispatch({
        "schema": SERVING_SCHEMA, "op": "stream.cancel",
        "args": {"session_id": sid},
    })
    assert second["ok"]
    assert monitor.metrics()["cancellations"] == 1


def test_restart_resets_local_metrics_but_not_serving_model_identity():
    plane, _ = make_plane()
    monitor = ServingObserver(plane)
    assert monitor.dispatch(request())["ok"]
    restarted = ServingObserver(plane)
    stats = restarted.metrics()
    assert stats["completed_requests"] == 0
    assert stats["capacity"]["loaded_slot_count"] == 1
    assert restarted.dispatch(request(request_id="after_restart"))["ok"]


@pytest.mark.parametrize("field,value", [
    ("max_inflight", True), ("max_inflight", 0),
    ("max_queue", -1), ("queue_timeout_s", -1.0),
    ("queue_timeout_s", float("nan")), ("queue_timeout_s", 100),
    ("memory_probe", object()),
])
def test_invalid_host_capacity_and_probe_denied(field, value):
    plane, _ = make_plane()
    with pytest.raises(ObservabilityError):
        ServingObserver(plane, **{field: value})


def test_cannot_use_inference_observer_as_permission_or_provider_scheduler():
    plane, _ = make_plane()
    monitor = ServingObserver(plane)
    assert not hasattr(monitor, "publish")
    assert not hasattr(monitor, "train")
    assert monitor.canary(role="answer", allow_external=False)["ready"]


def test_sensitive_request_not_reflected_in_operational_metrics():
    plane, _ = make_plane()
    monitor = ServingObserver(plane)
    secret = "private-prompt-shall-not-be-logged"
    assert monitor.dispatch(request(prompt=secret))["ok"]
    assert secret not in str(monitor.metrics())


def test_canary_denies_server_route_without_explicit_host_permission():
    plane, fixture = make_plane(transport="server")
    monitor = ServingObserver(plane)
    result = monitor.canary(role="answer")
    assert result["ready"] is False
    assert result["reason"] == "EXTERNAL_DENIED"
    assert "generate" not in fixture.calls


def test_canary_is_atomic_against_host_replacement():
    # A catalog precheck alone is vulnerable to a local-to-external swap.
    # The S8 host lock must span route inspection and the bounded call.
    plane, local = make_plane()
    remote = ModelFixture(model="remote")
    catalog_read = threading.Event()
    replacement_finished = threading.Event()
    original_dispatch = plane.dispatch

    def intercepted(request):
        result = original_dispatch(request)
        if type(request) is dict and request.get("op") == "catalog":
            catalog_read.set()
            replacement_finished.wait(0.1)
        return result

    plane.dispatch = intercepted
    errors = []

    def replace_host_route():
        if not catalog_read.wait(2):
            errors.append("catalog not read")
            return
        try:
            plane.replace(
                "one", slot_id="external", role="answer", provider="remote",
                model="remote", transport="external", client=remote,
                priority=1, reserved_bytes=5, authorize_external=True,
            )
        except Exception as exc:
            errors.append(str(exc))
        finally:
            replacement_finished.set()

    thread = threading.Thread(target=replace_host_route)
    thread.start()
    result = ServingObserver(plane).canary(role="answer")
    thread.join(2)
    assert not thread.is_alive()
    assert not errors
    assert result["ready"] is True
    assert result["model_identity"]["model"] == "tiny"
    assert local.calls.count("generate") == 1
    assert "generate" not in remote.calls
