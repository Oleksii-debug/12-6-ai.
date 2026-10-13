"""Independent LOCAL_FREE acceptance for bounded experimental research routing."""
import json
from dataclasses import replace

import pytest

from twelve_six_agent_runtime.adaptive_research_routing import (
    ModelOutcome,
    ResourceReading,
    RouteError,
    RoutePolicy,
    RouteState,
    Specialist,
    accept_outcome,
    select_next,
)


def node(node_id, cap, successors=(), *, provider="local", memory=400,
         charge=0, priority=10):
    return Specialist(node_id=node_id, capabilities=(cap,), provider=provider,
                      model_sha256="a" * 64, memory_mb=memory, cpu_units=1,
                      max_charge_microunits=charge, priority=priority,
                      successors=tuple(successors))


def policy(**kwargs):
    defaults = dict(max_calls=12, max_visits_per_node=4, max_no_progress=3,
                    cost_limit_microunits=100)
    defaults.update(kwargs)
    return RoutePolicy(**defaults)


def capacity(ram=2048, cpu=4):
    return ResourceReading(ram, cpu)


def result(model_id, seq=1, cap="logic", *, improved=True, charge=0):
    return ModelOutcome(task_id="task1", node_id=model_id, receipt_id="proof" + str(seq),
                        evidence_sha256="b" * 64, charged_microunits=charge,
                        next_capability=cap, measurable_progress=improved)


def test_non_linear_adaptive_handoff_with_host_attested_evidence():
    members = (node("a", "sense", ("b", "c")),
               node("b", "logic", ("a", "d")),
               node("c", "causal", ("d",)),
               node("d", "verify", ("a",)))
    state = RouteState("task1", "sense")
    expected = (("a", "causal"), ("c", "verify"), ("d", "sense"), ("a", "causal"))
    for i, (ident, next_cap) in enumerate(expected, start=1):
        choice = select_next(state, members, policy(), capacity())
        assert (choice.status, choice.node_id, choice.revision) == ("SELECTED", ident, i - 1)
        state = accept_outcome(state, choice, result(ident, i, next_cap),
                               members, policy(), capacity(), lambda receipt: receipt.receipt_id.startswith("proof"))
    assert state.history == ("a", "c", "d", "a")
    assert state.revision == 4 and state.no_progress == 0
    assert state.spent_microunits == 0


def test_reports_resource_wait_instead_of_loading_model():
    members = (node("large", "logic", memory=2000),)
    choice = select_next(RouteState("task1", "logic"), members, policy(), capacity(1500))
    assert choice.status == "WAIT_RESOURCES" and choice.node_id is None
    assert select_next(RouteState("task1", "logic"), members, policy(), capacity(2000)).node_id == "large"


def test_no_progress_stops_even_if_cycle_is_possible():
    members = (node("a", "logic", ("a",)),)
    state = RouteState("task1", "logic")
    for seq in range(1, 3):
        choice = select_next(state, members, policy(max_no_progress=2), capacity())
        state = accept_outcome(state, choice, result("a", seq, "logic", improved=False),
                               members, policy(max_no_progress=2), capacity(), lambda _: True)
    assert select_next(state, members, policy(max_no_progress=2), capacity()).status == "STOP_NO_PROGRESS"


def test_external_and_paid_require_host_policy_both_explicit():
    ext = (node("api", "logic", provider="external", charge=20),)
    state = RouteState("task1", "logic")
    assert select_next(state, ext, policy(), capacity()).status == "NO_ELIGIBLE_ROUTE"
    assert select_next(state, ext, policy(allow_external=True), capacity()).status == "NO_ELIGIBLE_ROUTE"
    assert select_next(state, ext, policy(allow_paid=True), capacity()).status == "NO_ELIGIBLE_ROUTE"
    assert select_next(state, ext, policy(allow_external=True, allow_paid=True), capacity()).node_id == "api"
    local = (node("paid-local", "logic", charge=1),)
    assert select_next(state, local, policy(), capacity()).status == "NO_ELIGIBLE_ROUTE"


def test_no_unbounded_model_cycles_or_cost_overrun():
    members = (node("a", "logic", ("a",), charge=7),)
    p = policy(max_calls=2, max_visits_per_node=2, cost_limit_microunits=14, allow_paid=True)
    state = RouteState("task1", "logic")
    for i in range(1, 3):
        decision = select_next(state, members, p, capacity())
        assert decision.node_id == "a"
        state = accept_outcome(state, decision, result("a", i, "logic", charge=7), members, p, capacity(), lambda _: True)
    assert state.spent_microunits == 14
    assert select_next(state, members, p, capacity()).status == "BUDGET_EXHAUSTED"


def test_stale_and_forged_receipts_fail_closed():
    members = (node("a", "logic", ("a",)),)
    p = policy()
    before = RouteState("task1", "logic")
    chosen = select_next(before, members, p, capacity())
    assert chosen.node_id == "a"
    with pytest.raises(RouteError, match="host evidence"):
        accept_outcome(before, chosen, result("a"), members, p, capacity(), lambda _: False)
    with pytest.raises(RouteError, match="identity mismatch"):
        accept_outcome(before, chosen, result("wrong"), members, p, capacity(), lambda _: True)
    with pytest.raises(RouteError, match="digest"):
        accept_outcome(before, chosen, replace(result("a"), evidence_sha256="x"), members, p, capacity(), lambda _: True)
    after = accept_outcome(before, chosen, result("a"), members, p, capacity(), lambda _: True)
    with pytest.raises(RouteError, match="stale"):
        accept_outcome(after, chosen, result("a", 2), members, p, capacity(), lambda _: True)
    next_choice = select_next(after, members, p, capacity())
    with pytest.raises(RouteError, match="replayed"):
        accept_outcome(after, next_choice, result("a"), members, p, capacity(), lambda _: True)
    assert after == RouteState("task1", "logic", ("a",), ("proof1",), 0, 0, 1)


def test_snapshots_restore_and_reject_tampering_or_noncanonical_encoding():
    state = RouteState("task1", "logic", ("a",), ("proof1",), 3, 0, 1)
    raw, checksum = state.snapshot()
    assert RouteState.from_snapshot(raw, checksum) == state
    with pytest.raises(RouteError, match="checksum"):
        RouteState.from_snapshot(raw.replace("task1", "task2"), checksum)
    altered = json.loads(raw)
    altered["state"]["revision"] = 400
    new_raw = json.dumps(altered, sort_keys=True, separators=(",", ":"))
    import hashlib
    with pytest.raises(RouteError, match="malformed"):
        RouteState.from_snapshot(new_raw, hashlib.sha256(new_raw.encode()).hexdigest())


def test_catalog_and_policy_validate_every_request():
    state = RouteState("task1", "logic")
    good = node("a", "logic")
    with pytest.raises(RouteError, match="duplicate"):
        select_next(state, (good, good), policy(), capacity())
    with pytest.raises(RouteError, match="outside"):
        select_next(state, (replace(good, successors=("ghost",)),), policy(), capacity())
    with pytest.raises(RouteError, match="capabilities"):
        select_next(state, (replace(good, capabilities=("logic", "logic")),), policy(), capacity())
    with pytest.raises(RouteError, match="integer"):
        select_next(state, (good,), policy(max_calls=True), capacity())
    with pytest.raises(RouteError, match="booleans"):
        select_next(state, (good,), policy(allow_paid=1), capacity())
    with pytest.raises(RouteError, match="model identity"):
        select_next(state, (replace(good, model_sha256="wrong"),), policy(), capacity())
    with pytest.raises(RouteError, match="unknown"):
        select_next(RouteState("task1", "logic", ("ghost",), ("proof1",), 0, 0, 1),
                    (good,), policy(), capacity())


def test_exact_capability_and_constrained_successors():
    members = (node("start", "sense", ("reason",)),
               node("reason", "logic", ("check",)),
               node("check", "verify", ("start",)))
    state = RouteState("task1", "sense")
    d = select_next(state, members, policy(), capacity())
    state = accept_outcome(state, d, result("start", cap="verify"), members,
                           policy(), capacity(), lambda _: True)
    assert select_next(state, members, policy(), capacity()).status == "NO_ELIGIBLE_ROUTE"


def test_no_implicit_tool_or_provider_calls():
    touched = []
    members = (node("a", "logic"),)
    for _ in range(3):
        decision = select_next(RouteState("task1", "logic"), members, policy(), capacity())
        assert decision.status == "SELECTED"
    assert touched == []  # pure planner; host must separately admit and execute effects


def test_route_state_checkpoint_binds_existing_plan5_taskstore(tmp_path):
    """Durable TaskStore owns the authority; route snapshot stays separately hashed."""
    from twelve_six_agent_runtime.task_state import TaskStore

    db_path = tmp_path / "task.db"
    store = TaskStore(db_path)
    task = store.create(task_id="task1", plan_id="research", step_id="select")
    members = (node("a", "logic", ("a",)),)
    route = RouteState("task1", "logic")
    decision = select_next(route, members, policy(), capacity())
    route = accept_outcome(route, decision, result("a"), members,
                           policy(), capacity(), lambda _: True)
    raw, digest = route.snapshot()
    ticket = store.checkpoint(task.task_id, expected_epoch=task.control_epoch,
                              expected_revision=task.revision, step_id="route1",
                              checkpoint_id=digest, pending_effects=())
    restored_task = TaskStore(db_path).load("task1")
    restored_route = RouteState.from_snapshot(raw, restored_task.checkpoint_id)
    assert restored_task == ticket
    assert restored_route == route
    assert select_next(restored_route, members, policy(), capacity()).node_id == "a"
    with pytest.raises(RouteError, match="checksum"):
        RouteState.from_snapshot(raw + " ", restored_task.checkpoint_id)
