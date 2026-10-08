"""Plan 5 S7: descriptor policy, TaskStore replay and ambiguous outcome tests."""
import pytest

from twelve_six_agent_runtime.task_state import PendingEffect, TaskStore
from twelve_six_agent_runtime.tool_registry import (
    Invocation, Receipt, ToolDescriptor, ToolError, ToolRegistry,
)


def obj(properties=None, required=None):
    return {"type": "object", "properties": properties or {},
            "required": required or [], "additionalProperties": False}


def descriptor(**overrides):
    values = dict(name="fixture.read", version=1,
                  input_schema=obj({"key": {"type": "string", "maxLength": 10}}, ["key"]),
                  output_schema=obj({"value": {"type": "integer", "minimum": 0,
                                               "maximum": 100}}, ["value"]),
                  capabilities=("fixture",), permissions=("read",), effect="read")
    values.update(overrides)
    return ToolDescriptor(**values)


def reserved(tmp_path):
    tasks = TaskStore(tmp_path / "tasks.db")
    initial = tasks.create(task_id="task", plan_id="plan-5", step_id="start")
    s = tasks.checkpoint("task", expected_epoch=initial.control_epoch,
                         expected_revision=initial.revision, step_id="tool",
                         checkpoint_id="cp", pending_effects=(PendingEffect("fx", "fixture"),))
    return tasks, s


def prepare(registry, tasks, state, **kw):
    opts = dict(task_id="task", effect_id="fx", epoch=state.control_epoch,
                revision=state.revision, tool="fixture.read", arguments={"key": "abc"},
                host_permissions=frozenset(("read",)))
    opts.update(kw)
    return registry.prepare(tasks, **opts)


def receipt(call, output=None):
    return Receipt(call.call_identity, "receipt-v1", output or {"value": 2}, ("mock:verified",))


def test_registry_versioned_and_discovery_is_not_permission():
    registry = ToolRegistry((descriptor(),))
    assert registry.discover()[0]["version"] == 1
    assert ToolRegistry((descriptor(version=2),)).discover() != registry.discover()
    with pytest.raises(ToolError, match="duplicate"):
        ToolRegistry((descriptor(), descriptor()))


def test_success_integrates_with_only_existing_taskstore(tmp_path):
    tasks, state = reserved(tmp_path)
    registry = ToolRegistry((descriptor(),))
    call = prepare(registry, tasks, state)
    evidence = registry.invoke(tasks, call, executor=lambda tool, args: receipt(call))
    assert evidence.evidence == ("mock:verified",)
    assert tasks.load("task").pending_effects[0].receipt_id == "receipt-v1"
    with pytest.raises(ToolError, match="reconcile"):
        registry.invoke(tasks, call, executor=lambda tool, args: receipt(call))


def test_denied_unknown_unavailable_and_unsupported_are_fail_closed(tmp_path):
    tasks, state = reserved(tmp_path)
    registry = ToolRegistry((descriptor(),))
    for opts in ({"host_permissions": frozenset()}, {"tool": "bad"},
                 {"arguments": {"key": "abc", "admin": True}},
                 {"arguments": {"key": 2}}):
        with pytest.raises(ToolError):
            prepare(registry, tasks, state, **opts)
    with pytest.raises(ToolError, match="unavailable"):
        prepare(ToolRegistry((descriptor(available=False),)), tasks, state)
    assert tasks.load("task").pending_effects[0].status == "pending"


def test_irreversible_requires_host_bound_confirmation(tmp_path):
    tasks, state = reserved(tmp_path)
    reg = ToolRegistry((descriptor(effect="irreversible"),))
    with pytest.raises(ToolError, match="confirmation"):
        prepare(reg, tasks, state)
    with pytest.raises(ToolError, match="confirmation"):
        prepare(reg, tasks, state, confirmed=lambda digest: False)
    assert prepare(reg, tasks, state, confirmed=lambda digest: len(digest) == 64)


def test_stale_epoch_and_revision_rejected(tmp_path):
    tasks, state = reserved(tmp_path)
    reg = ToolRegistry((descriptor(),))
    old = prepare(reg, tasks, state)
    tasks.resume("task")
    with pytest.raises(ToolError, match="stale"):
        prepare(reg, tasks, state)
    with pytest.raises(ToolError, match="reconcile"):
        reg.invoke(tasks, old, executor=lambda tool, args: receipt(old))


def test_crash_failure_unknown_never_automatic_replay(tmp_path):
    tasks, state = reserved(tmp_path)
    reg = ToolRegistry((descriptor(),))
    call = prepare(reg, tasks, state)

    def crash(*args):
        raise RuntimeError("process stopped after issue")

    with pytest.raises(RuntimeError):
        reg.invoke(tasks, call, executor=crash)
    restarted = TaskStore(tasks.path)
    assert restarted.load("task").pending_effects[0].status == "unknown"
    with pytest.raises(ToolError):
        reg.invoke(restarted, call, executor=lambda tool, args: receipt(call))


def test_output_receipt_tamper_remains_unknown(tmp_path):
    tasks, state = reserved(tmp_path)
    reg = ToolRegistry((descriptor(),))
    call = prepare(reg, tasks, state)
    with pytest.raises(ToolError, match="unbound"):
        reg.invoke(tasks, call, executor=lambda tool, args:
                   Receipt("bad", "receipt", {"value": 1}, ("evidence",)))
    assert TaskStore(tasks.path).load("task").pending_effects[0].status == "unknown"


def test_schema_and_descriptor_mutation_rejected(tmp_path):
    for bad in (
        descriptor(input_schema={"type": "object", "properties": {}}),
        descriptor(input_schema=obj({"x": {"type": "array", "items":
                                            {"type": "string", "maxLength": 2}}})),
        descriptor(permissions=("write", "read")),
        descriptor(version=True),
    ):
        with pytest.raises(ToolError):
            ToolRegistry((bad,))
    tasks, state = reserved(tmp_path)
    desc = descriptor()
    reg = ToolRegistry((desc,))
    desc.input_schema["properties"]["key"]["maxLength"] = 0
    with pytest.raises(ToolError, match="drift"):
        prepare(reg, tasks, state)


def test_invocation_tamper_fails_before_issuing(tmp_path):
    tasks, state = reserved(tmp_path)
    reg = ToolRegistry((descriptor(),))
    good = prepare(reg, tasks, state)
    tampered = Invocation(good.task_id, good.effect_id, good.control_epoch,
                          good.task_revision, good.tool, good.tool_identity,
                          {"key": "other"}, good.call_identity)
    with pytest.raises(ToolError, match="identity"):
        reg.invoke(tasks, tampered, executor=lambda tool, args: receipt(good))
    assert tasks.load("task").pending_effects[0].status == "pending"
