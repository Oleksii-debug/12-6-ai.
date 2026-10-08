"""Plan 5 S8: semantic browser/computer reference contract qualification."""
from dataclasses import replace

import pytest

from twelve_six_agent_runtime.task_state import PendingEffect, StateError, TaskStore
from twelve_six_agent_runtime.tools import ToolBoundaryError, ToolRegistry
from twelve_six_agent_runtime.ui_control import (
    SemanticNode, SemanticUIControl, UIControlError, UISnapshot, ui_tool_descriptor,
)


def snapshot(*, backend="dom", generation=1, nodes=None):
    return UISnapshot(
        backend=backend, application_id="browser-app", view_id="tab-17",
        generation=generation, tree_digest="b" * 64, evidence_id="host-tree-receipt",
        nodes=tuple(nodes if nodes is not None else (
            SemanticNode("n1", "button", "Submit", "idle"),
            SemanticNode("n2", "textbox", "Search", "empty"),
        )),
    )


def harness(tmp_path):
    task_store = TaskStore(tmp_path / "canonical-tasks.sqlite")
    task_store.create(task_id="task-a", plan_id="plan5", step_id="ui")
    reg = ToolRegistry(task_store)
    reg.register(ui_tool_descriptor(), verify_descriptor=lambda _: True)
    ui = SemanticUIControl(reg)
    return task_store, reg, ui


def reserve(tasks):
    initial = tasks.load("task-a")
    return tasks.checkpoint(
        "task-a", expected_epoch=initial.control_epoch,
        expected_revision=initial.revision, step_id="ui-action",
        checkpoint_id="reserve-ui-1",
        pending_effects=(PendingEffect("effect-1", "UI external effect"),),
    )


def prepare(ui, surface=None, **overrides):
    kw = dict(
        role="button", name="Submit", action="activate", expected_state="clicked",
        task_id="task-a", effect_id="effect-1", expected_epoch=0, expected_revision=1,
        grant_evidence_id="trusted-grant", verify_snapshot=lambda _: True,
        authorize=lambda *_: True,
    )
    kw.update(overrides)
    return ui.prepare(snapshot() if surface is None else surface, **kw)


def issue(ui, intent, current=None):
    return ui.issue(intent, intent.snapshot if current is None else current,
                    verify_current=lambda _: True)


def complete(ui, intent, **overrides):
    kw = dict(
        observed_state="clicked", receipt_id="external-receipt",
        evidence_id="postcondition-attestation",
        verify_call=lambda _: True,
        verify_result=lambda *_: True,
        verify_postcondition=lambda *_: True,
    )
    kw.update(overrides)
    return ui.complete(intent, **kw)


def test_browser_dom_semantic_action_and_exact_receipt(tmp_path):
    tasks, _, ui = harness(tmp_path)
    reserve(tasks)
    intent = prepare(ui)
    assert intent.target.node_id == "n1"
    assert intent.call.request_json.startswith('{"action":"activate"')
    issued = issue(ui, intent)
    assert issued.pending_effects[0].status == "unknown"
    result = complete(ui, intent)
    assert result.pending_effects[0].receipt_id == "external-receipt"
    assert TaskStore(tasks.path).load("task-a") == result
    with pytest.raises((UIControlError, StateError, ToolBoundaryError)):
        issue(ui, intent)
    with pytest.raises((UIControlError, StateError, ToolBoundaryError)):
        complete(ui, intent)


def test_ax_uia_reference_surfaces_use_same_identity_and_tool_boundary(tmp_path):
    for backend in ("ax", "uia"):
        # Use distinct temporary directories: TaskStore is the shared authority per case.
        location = tmp_path / backend
        location.mkdir()
        tasks, _, ui = harness(location)
        reserve(tasks)
        intent = prepare(ui, snapshot(backend=backend))
        assert intent.snapshot.backend == backend
        assert issue(ui, intent).pending_effects[0].status == "unknown"
        assert complete(ui, intent).pending_effects[0].status == "resolved"


def test_ambiguous_and_missing_nodes_never_select_arbitrarily(tmp_path):
    tasks, _, ui = harness(tmp_path)
    reserve(tasks)
    duplicate = snapshot(nodes=(
        SemanticNode("a", "button", "Submit", "idle"),
        SemanticNode("b", "button", "Submit", "idle"),
    ))
    with pytest.raises(UIControlError, match="ambiguous"):
        prepare(ui, duplicate)
    assert prepare(ui, duplicate, node_id="a").target.node_id == "a"
    with pytest.raises(UIControlError, match="ambiguous"):
        prepare(ui, duplicate, node_id="absent")
    assert tasks.load("task-a").pending_effects[0].status == "pending"


def test_stale_target_and_snapshot_generation_rejected_before_issue(tmp_path):
    tasks, _, ui = harness(tmp_path)
    reserve(tasks)
    intent = prepare(ui)
    with pytest.raises(UIControlError, match="stale"):
        issue(ui, intent, snapshot(generation=2))
    with pytest.raises(UIControlError, match="stale"):
        ui.issue(intent, intent.snapshot, verify_current=lambda _: False)
    assert tasks.load("task-a").pending_effects[0].status == "pending"
    assert issue(ui, intent).pending_effects[0].status == "unknown"


def test_coordinate_and_vision_fallback_requires_explicit_host_proof(tmp_path):
    tasks, _, ui = harness(tmp_path)
    reserve(tasks)
    for method in ("coordinate", "vision"):
        with pytest.raises(UIControlError, match="fallback"):
            prepare(ui, method=method)
        with pytest.raises(UIControlError, match="fallback"):
            prepare(ui, method=method, fallback_evidence_id="visual-attestation",
                    verify_fallback=lambda *_: False)
        intent = prepare(ui, method=method, fallback_evidence_id="visual-attestation",
                         verify_fallback=lambda *_: True)
        assert intent.target.node_id == "n1"
    with pytest.raises(UIControlError, match="fallback"):
        prepare(ui, method="semantic", fallback_evidence_id="untrusted")
    with pytest.raises(UIControlError, match="unsupported"):
        prepare(ui, method="raw-screen-coordinate")
    assert tasks.load("task-a").pending_effects[0].status == "pending"


def test_fallback_evidence_cannot_be_substituted_after_grant(tmp_path):
    tasks, _, ui = harness(tmp_path)
    reserve(tasks)
    intent = prepare(ui, method="vision", fallback_evidence_id="verified-fallback",
                     verify_fallback=lambda *_: True)
    with pytest.raises(UIControlError, match="binding"):
        issue(ui, replace(intent, fallback_evidence_id="forged-fallback"))
    assert tasks.load("task-a").pending_effects[0].status == "pending"
    assert issue(ui, intent).pending_effects[0].status == "unknown"


def test_unverified_observation_grant_and_actions_fail_closed(tmp_path):
    tasks, _, ui = harness(tmp_path)
    reserve(tasks)
    with pytest.raises(UIControlError, match="unverified"):
        prepare(ui, verify_snapshot=lambda _: False)
    with pytest.raises(UIControlError, match="unsupported"):
        prepare(ui, action="delete_machine")
    with pytest.raises(ValueError, match="grant"):
        prepare(ui, authorize=lambda *_: False)
    assert tasks.load("task-a").pending_effects[0].status == "pending"


def test_unknown_effect_restart_requires_reconciliation_not_blind_retry(tmp_path):
    tasks, _, ui = harness(tmp_path)
    reserve(tasks)
    intent = prepare(ui)
    issue(ui, intent)
    restarted = TaskStore(tasks.path)
    epoch = restarted.resume("task-a")
    assert epoch.control_epoch == 1
    with pytest.raises((UIControlError, StateError, ToolBoundaryError)):
        complete(ui, intent)
    with pytest.raises(UIControlError, match="evidence"):
        ui.reconcile_unknown(intent, observed_state="clicked", receipt_id="r",
                             evidence_id="host-recovery", verify_recovery=lambda *_: False)
    assert restarted.load("task-a").pending_effects[0].status == "unknown"
    recovered = ui.reconcile_unknown(
        intent, observed_state="clicked", receipt_id="recovered-receipt",
        evidence_id="host-recovery", verify_recovery=lambda *_: True,
    )
    assert recovered.pending_effects[0].status == "resolved"
    assert TaskStore(tasks.path).load("task-a") == recovered
    with pytest.raises(UIControlError, match="already reconciled"):
        ui.reconcile_unknown(intent, observed_state="clicked", receipt_id="r",
                             evidence_id="e", verify_recovery=lambda *_: True)


def test_postcondition_mismatch_and_unverified_result_remain_unknown(tmp_path):
    tasks, _, ui = harness(tmp_path)
    reserve(tasks)
    intent = prepare(ui)
    issue(ui, intent)
    with pytest.raises(UIControlError, match="postcondition"):
        complete(ui, intent, observed_state="idle")
    with pytest.raises(UIControlError, match="proof"):
        complete(ui, intent, verify_postcondition=lambda *_: False)
    with pytest.raises(ValueError, match="verified"):
        complete(ui, intent, verify_result=lambda *_: False)
    assert tasks.load("task-a").pending_effects[0].status == "unknown"
    with pytest.raises(UIControlError, match="successful"):
        ui.reconcile_unknown(intent, observed_state="idle", receipt_id="r",
                             evidence_id="e", verify_recovery=lambda *_: True)


def test_forged_intent_and_incompatible_tool_contract_rejected(tmp_path):
    tasks, _, ui = harness(tmp_path)
    reserve(tasks)
    intent = prepare(ui)
    for forged in (
        replace(intent, target=SemanticNode("x", "button", "Submit", "idle")),
        replace(intent, expected_state="other"),
        replace(intent, action="focus"),
        replace(intent, method="vision"),
        replace(intent, snapshot=snapshot(generation=3)),
    ):
        with pytest.raises(UIControlError):
            issue(ui, forged)
    assert tasks.load("task-a").pending_effects[0].status == "pending"
    reg = ToolRegistry(tasks)
    bad = replace(ui_tool_descriptor(), side_effect="read")
    reg.register(bad, verify_descriptor=lambda _: True)
    with pytest.raises(UIControlError, match="descriptor"):
        SemanticUIControl(reg)


def test_malformed_tree_and_duplicate_node_identity_fail_closed(tmp_path):
    tasks, _, ui = harness(tmp_path)
    reserve(tasks)
    for malformed in (
        snapshot(backend="coordinate"),
        snapshot(nodes=(SemanticNode("a", "button", "Submit", "idle"),) * 2),
        replace(snapshot(), tree_digest="not-a-sha"),
        replace(snapshot(), generation=-1),
        replace(snapshot(), nodes=("not-a-typed-node",)),
        replace(snapshot(), nodes=tuple(SemanticNode(str(n), "button", "A", "idle")
                                        for n in range(1025))),
    ):
        with pytest.raises((ValueError, TypeError)):
            prepare(ui, malformed)
    assert tasks.load("task-a").pending_effects[0].status == "pending"
