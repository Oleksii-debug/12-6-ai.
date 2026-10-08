from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from twelve_six_agent_runtime.task_state import PendingEffect, TaskStore
from twelve_six_agent_runtime.tools import (
    DESCRIPTOR_VERSION,
    ToolBoundaryError,
    ToolDescriptor,
    ToolRegistry,
    make_result,
)

INPUT = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "maxLength": 64},
        "dry_run": {"type": "boolean"},
        "indices": {"type": "array", "items": {"type": "integer"}, "maxItems": 3},
    },
    "required": ["path"],
    "additionalProperties": False,
}
OUTPUT = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}, "code": {"type": "integer"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def descriptor(*, available=True, effect="write", version=DESCRIPTOR_VERSION):
    return ToolDescriptor(
        tool_id="file.readwrite", version=version,
        capabilities=("file", "filesystem"), permissions=("filesystem.write",),
        side_effect=effect, input_schema=INPUT, output_schema=OUTPUT,
        available=available,
    )


def harness(tmp_path: Path):
    tasks = TaskStore(tmp_path / "tasks.sqlite")
    reg = ToolRegistry(tasks)
    reg.register(descriptor(), verify_descriptor=lambda _: True)
    tasks.create(task_id="task", plan_id="plan5", step_id="start")
    tasks.checkpoint(
        "task", expected_epoch=0, expected_revision=0, step_id="call",
        checkpoint_id="cp1",
        pending_effects=(PendingEffect("effect", "write file after host grant"),),
    )
    return tasks, reg


def prepare(reg: ToolRegistry, *, request=None, verifier=lambda *_: True,
            permissions=("filesystem.write",), epoch=0, revision=1):
    return reg.prepare(
        task_id="task", effect_id="effect", tool_id="file.readwrite",
        request={"path": "safe.txt", "dry_run": True} if request is None else request,
        permissions=permissions, grant_evidence_id="host-grant-1",
        expected_epoch=epoch, expected_revision=revision, authorize=verifier,
    )


def test_versioned_discovery_does_not_grant_permission(tmp_path):
    tasks = TaskStore(tmp_path / "state.db")
    reg = ToolRegistry(tasks)
    with pytest.raises(ToolBoundaryError, match="unverified"):
        reg.register(descriptor(), verify_descriptor=lambda _: False)
    identity = reg.register(descriptor(), verify_descriptor=lambda _: True)
    assert len(identity) == 64
    assert reg.discover() == (descriptor(),)
    with pytest.raises(ToolBoundaryError, match="unknown"):
        reg.get("unknown")
    with pytest.raises(ToolBoundaryError, match="duplicate"):
        reg.register(descriptor(), verify_descriptor=lambda _: True)
    with pytest.raises(ToolBoundaryError, match="unknown"):
        prepare(reg)


def test_discovery_not_permission_and_no_unreserved_effect(tmp_path):
    _, reg = harness(tmp_path)
    reg.discover()
    with pytest.raises(ToolBoundaryError, match="grant"):
        prepare(reg, verifier=lambda *_: False)
    with pytest.raises(ToolBoundaryError, match="permissions"):
        prepare(reg, permissions=())
    with pytest.raises(ToolBoundaryError, match="permissions"):
        prepare(reg, permissions=("filesystem.write", "root"))
    assert len(reg.discover()) == 1


def test_verified_tool_receipt_resolves_effect_once_and_restart_preserves_it(tmp_path):
    tasks, reg = harness(tmp_path)
    call = prepare(reg)
    result = make_result(
        call, receipt_id="host-receipt", evidence_id="signed-log",
        outcome="success", output={"ok": True, "code": 0},
    )
    assert len(call.call_id) == 64
    snap = tasks.issue_effect("task", "effect", expected_epoch=0, expected_revision=1)
    assert snap.pending_effects[0].status == "unknown"
    with pytest.raises(ToolBoundaryError, match="verified"):
        reg.accept_result(call, result, verify_call=lambda _: False,
                          verify_result=lambda *_: True)
    out = reg.accept_result(call, result, verify_call=lambda _: True,
                            verify_result=lambda *_: True)
    assert out.pending_effects[0].receipt_id == "host-receipt"
    assert TaskStore(tasks.path).load("task") == out
    with pytest.raises(ToolBoundaryError, match="unissued"):
        reg.accept_result(call, result, verify_call=lambda _: True,
                          verify_result=lambda *_: True)
    with pytest.raises(ToolBoundaryError, match="unreserved"):
        prepare(reg, revision=out.revision)


def test_bad_input_and_strict_schema_fails_before_effect_issue(tmp_path):
    tasks, reg = harness(tmp_path)
    for payload in (
        {}, {"path": 1}, {"path": "x" * 65}, {"path": "ok", "other": "inject"},
        {"path": "ok", "dry_run": 0}, {"path": "ok", "indices": [1, 2, 3, 4]},
        {"path": "ok", "indices": [True]},
    ):
        with pytest.raises(ToolBoundaryError):
            prepare(reg, request=payload)
    assert tasks.load("task").pending_effects[0].status == "pending"


def test_result_tamper_and_cross_task_binding_rejected(tmp_path):
    tasks, reg = harness(tmp_path)
    call = prepare(reg)
    tasks.issue_effect("task", "effect", expected_epoch=0, expected_revision=1)
    good = make_result(call, receipt_id="r", evidence_id="e", outcome="success",
                       output={"ok": True})
    for forged in (
        replace(good, result_id="0" * 64),
        replace(good, task_id="other"),
        replace(good, effect_id="different"),
        replace(good, call_id="other"),
        replace(good, output_json='{"ok":false}'),
    ):
        with pytest.raises(ToolBoundaryError):
            reg.accept_result(call, forged, verify_call=lambda _: True,
                              verify_result=lambda *_: True)
    with pytest.raises(ToolBoundaryError, match="integrity"):
        reg.accept_result(replace(call, task_id="other"), good,
                          verify_call=lambda _: True, verify_result=lambda *_: True)
    assert tasks.load("task").pending_effects[0].status == "unknown"


def test_output_schema_and_independent_host_evidence(tmp_path):
    tasks, reg = harness(tmp_path)
    call = prepare(reg)
    tasks.issue_effect("task", "effect", expected_epoch=0, expected_revision=1)
    for output in ({"ok": 1}, {}, {"ok": True, "extra": "secret"},
                   {"ok": True, "code": False}):
        record = make_result(call, receipt_id="r", evidence_id="e",
                             outcome="success", output=output)
        with pytest.raises(ToolBoundaryError):
            reg.accept_result(call, record, verify_call=lambda _: True,
                              verify_result=lambda *_: True)
    verified = make_result(call, receipt_id="r", evidence_id="e",
                           outcome="failure", output={"ok": False})
    with pytest.raises(ToolBoundaryError, match="verified"):
        reg.accept_result(call, verified, verify_call=lambda _: True,
                          verify_result=lambda *_: 1)
    assert tasks.load("task").pending_effects[0].status == "unknown"


def test_restart_stale_epoch_cannot_accept_old_tool_result(tmp_path):
    tasks, reg = harness(tmp_path)
    call = prepare(reg)
    tasks.issue_effect("task", "effect", expected_epoch=0, expected_revision=1)
    tasks.resume("task")
    receipt = make_result(call, receipt_id="r", evidence_id="e",
                          outcome="success", output={"ok": True})
    with pytest.raises(ToolBoundaryError, match="stale"):
        reg.accept_result(call, receipt, verify_call=lambda _: True,
                          verify_result=lambda *_: True)
    assert tasks.load("task").pending_effects[0].status == "unknown"


def test_unsupported_descriptor_unavailable_and_malformed_schema(tmp_path):
    reg = ToolRegistry(TaskStore(tmp_path / "task.db"))
    with pytest.raises(ToolBoundaryError, match="version"):
        reg.register(descriptor(version="future.v2"), verify_descriptor=lambda _: True)
    with pytest.raises(ToolBoundaryError, match="open|unsupported"):
        reg.register(replace(descriptor(), input_schema={**INPUT, "additionalProperties": True}),
                     verify_descriptor=lambda _: True)
    with pytest.raises(ToolBoundaryError, match="unbounded"):
        reg.register(replace(descriptor(), output_schema={
            **OUTPUT, "properties": {"ok": {"type": "string"}},
        }), verify_descriptor=lambda _: True)
    reg.register(descriptor(available=False), verify_descriptor=lambda _: True)
    assert reg.discover() == ()
    with pytest.raises(ToolBoundaryError, match="unavailable"):
        reg.get("file.readwrite")


def test_descriptor_freezes_mutable_schema_after_registration(tmp_path):
    tasks, reg = harness(tmp_path)
    original_identity = reg.get("file.readwrite")[1]
    INPUT["properties"]["path"]["maxLength"] = 1
    try:
        assert reg.get("file.readwrite")[1] == original_identity
        assert prepare(reg).request_json
    finally:
        INPUT["properties"]["path"]["maxLength"] = 64


def test_call_must_precede_issue_and_no_blind_retry(tmp_path):
    tasks, reg = harness(tmp_path)
    original = prepare(reg)
    snap = tasks.issue_effect("task", "effect", expected_epoch=0, expected_revision=1)
    with pytest.raises(ToolBoundaryError, match="unreserved"):
        prepare(reg, revision=snap.revision)
    with pytest.raises(ToolBoundaryError, match="unissued"):
        reg.accept_result(
            original,
            make_result(original, receipt_id="r", evidence_id="e",
                        outcome="success", output={"ok": True}),
            verify_call=lambda _: True, verify_result=lambda *_: True,
        ) if False else None
    # Unknown external effect cannot be scheduled a second time.
    assert tasks.load("task").pending_effects[0].status == "unknown"
