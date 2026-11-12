"""Plan 5 Section 11: ModelGateway fixture E2E and adverse restart tests."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from twelve_six_agent_runtime.fixture_integration import (
    EchoGatewayFixture,
    FakeHostTool,
    FixtureError,
    FixtureRuntime,
    GatewayProposal,
    Scenario,
)
from twelve_six_agent_runtime.memory import MemoryRecord
from twelve_six_agent_runtime.scheduler import SchedulerStore, WorkSpec
from twelve_six_agent_runtime.task_state import TaskStore

FIXTURE = (
    Path(__file__).parent / "fixtures" / "plan5_agent_runtime_scenario_v1.json"
)


def scenario():
    return Scenario.from_json(FIXTURE.read_text(encoding="utf-8").strip())


def harness(tmp_path, *, gateway=None, tool=None, verify_host=None, preset=None):
    sample = scenario() if preset is None else preset
    model = gateway if gateway is not None else EchoGatewayFixture(sample.observed_value)
    adapter = tool if tool is not None else FakeHostTool()
    runtime = FixtureRuntime(
        tmp_path, sample, model, adapter, verify_host=verify_host,
    )
    return runtime, model, adapter


def test_full_mock_model_tool_world_memory_task_e2e(tmp_path):
    runtime, model, adapter = harness(tmp_path)
    state = runtime.run()
    assert state.step_id == "fixture-done"
    assert state.pending_effects[0].status == "resolved"
    assert adapter.dispatches == 1
    assert model.calls == 1
    observed = runtime.world.lookup("fixture-environment", "state", now=2)
    assert observed.state == "observed"
    assert observed.value == "inventory-ready"
    memory = runtime.memory.retrieve(now=3, kind="episodic")
    assert len(memory) == 1
    assert memory[0].authority == "memory_only"
    assert memory[0].requires_live_verification
    assert memory[0].record.content == "Observed: inventory-ready"
    assert TaskStore(runtime.tasks.path).load(state.task_id) == state


def test_duplicate_invocation_no_model_replay_or_double_tool(tmp_path):
    runtime, model, tool = harness(tmp_path)
    first = runtime.run()
    assert runtime.run() == first
    assert model.calls == 1
    assert tool.dispatches == 1
    assert runtime.memory.revision() == 1


class NoReplayGateway:
    def propose(self, _request):
        raise AssertionError("model must not execute after durable reservation")


def test_restart_after_reservation_reuses_exact_state_and_new_epoch(tmp_path):
    runtime, model, _ = harness(tmp_path)
    original = runtime.run(stop_after="reserved")
    assert original.pending_effects[0].status == "pending"
    assert model.calls == 1
    resumed_runtime, _, adapter = harness(tmp_path, gateway=NoReplayGateway())
    resumed = resumed_runtime.tasks.resume(scenario().scenario_id)
    assert resumed.control_epoch == original.control_epoch + 1
    final = resumed_runtime.run()
    assert final.step_id == "fixture-done"
    assert adapter.dispatches == 1
    assert resumed_runtime.memory.revision() == 1


def test_issued_unknown_restart_blocks_replay_until_verified_receipt(tmp_path):
    runtime, model, adapter = harness(tmp_path)
    issued = runtime.run(stop_after="issued")
    assert issued.pending_effects[0].status == "unknown"
    assert adapter.dispatches == 0
    recovery, _, recovered_tool = harness(tmp_path, gateway=NoReplayGateway())
    after_restart = recovery.tasks.resume(scenario().scenario_id)
    with pytest.raises(FixtureError, match="no retry"):
        recovery.run()
    for receipt, verifier in (
        ("forged", lambda _: True),
        ("fixture-verified-receipt", lambda _: False),
    ):
        with pytest.raises(FixtureError, match="receipt"):
            recovery.reconcile_unknown(
                receipt_id=receipt, verify_external_receipt=verifier,
            )
    assert recovery.tasks.load(scenario().scenario_id) == after_restart
    recovered = recovery.reconcile_unknown(
        receipt_id="fixture-verified-receipt",
        verify_external_receipt=lambda _: True,
    )
    assert recovered.pending_effects[0].status == "resolved"
    assert recovery.run().step_id == "fixture-done"
    assert recovered_tool.dispatches == 0
    assert model.calls == 1


def test_crash_after_resolved_effect_before_memory_replays_no_tool(tmp_path):
    runtime, _, adapter = harness(tmp_path)
    state = runtime.run(stop_after="resolved")
    assert state.pending_effects[0].status == "resolved"
    assert runtime.memory.revision() == 0
    assert adapter.dispatches == 1
    recovered, _, new_adapter = harness(tmp_path, gateway=NoReplayGateway())
    recovered.tasks.resume(scenario().scenario_id)
    assert recovered.run().step_id == "fixture-done"
    assert recovered.memory.revision() == 1
    assert new_adapter.dispatches == 0


class InjectionGateway:
    def __init__(self, proposal):
        self.proposal = proposal

    def propose(self, _request):
        return self.proposal


@pytest.mark.parametrize("proposal", [
    GatewayProposal("owner.shell", {"value": "inventory-ready"}),
    GatewayProposal("fixture.observe", {"value": "vault:secrets"}),
    GatewayProposal("fixture.observe", {"value": "changed observation"}),
    GatewayProposal("fixture.observe", {"value": "inventory-ready", "api_key": "raw"}),
])
def test_untrusted_model_proposal_never_controls_permissions(tmp_path, proposal):
    runtime, _, adapter = harness(tmp_path, gateway=InjectionGateway(proposal))
    with pytest.raises(FixtureError, match="model proposal"):
        runtime.run()
    assert runtime.tasks.load(scenario().scenario_id).revision == 0
    assert adapter.dispatches == 0


def test_denied_host_grant_and_payload_never_issue_effect(tmp_path):
    runtime, _, adapter = harness(tmp_path, verify_host=lambda name: name != "grant")
    with pytest.raises(ValueError):
        runtime.run()
    assert adapter.dispatches == 0
    state = runtime.tasks.load(scenario().scenario_id)
    assert state.pending_effects[0].status == "pending"


class MaliciousFakeTool(FakeHostTool):
    def dispatch(self, request):
        super().dispatch(request)
        return {"ok": True, "value": "forged-result"}


def test_invalid_tool_result_leaves_unknown_not_success(tmp_path):
    fake = MaliciousFakeTool()
    runtime, _, _ = harness(tmp_path, tool=fake)
    with pytest.raises(ValueError):
        runtime.run()
    assert fake.dispatches == 1
    assert runtime.tasks.load(scenario().scenario_id).pending_effects[0].status == "unknown"
    assert runtime.memory.revision() == 0
    restarted, _, new_host = harness(tmp_path, gateway=NoReplayGateway())
    with pytest.raises(FixtureError, match="no retry"):
        restarted.run()
    assert new_host.dispatches == 0


def test_scenario_replacement_cannot_rebind_reserved_effect(tmp_path):
    runtime, _, _ = harness(tmp_path)
    runtime.run(stop_after="reserved")
    changed = replace(scenario(), observed_value="forged-world")
    other, _, fake = harness(tmp_path, preset=changed)
    with pytest.raises(FixtureError, match="identity"):
        other.run()
    assert fake.dispatches == 0


def test_memory_correction_is_durable_and_not_world_observation(tmp_path):
    runtime, _, _ = harness(tmp_path)
    runtime.run()
    prior = runtime.memory.history()[0]
    corrected = MemoryRecord(
        memory_id="correction-1", kind="episodic",
        content="Correction: verify before external use",
        source_id="verified-host", source_kind="tool", evidence_id="correction-receipt",
        confidence_ppm=900_000, created_at=4, action="correct",
        replaces_id=prior.memory_id,
    )
    runtime.memory.append(corrected, expected_revision=1)
    after, _, _ = harness(tmp_path, gateway=NoReplayGateway())
    hits = after.memory.retrieve(now=5)
    assert len(hits) == 1 and hits[0].record.memory_id == "correction-1"
    assert hits[0].requires_live_verification
    assert after.world.lookup("fixture-environment", "state", now=5).state == "observed"
    assert after.run().step_id == "fixture-done"


def test_deterministic_model_replacement_preserves_task_memory_semantics(tmp_path):
    scenario()
    outcomes = []
    for label in ("gateway-a", "gateway-b"):
        root = tmp_path / label
        root.mkdir()
        runtime, model, host = harness(root)
        completed = runtime.run()
        outcomes.append((
            completed.encode(), tuple(runtime.memory.history()),
            runtime.world.export(), model.calls, host.dispatches,
        ))
    assert outcomes[0] == outcomes[1]


def test_resource_scheduler_fixture_remains_independent_authority(tmp_path):
    tasks = TaskStore(tmp_path / "scheduler-tasks.sqlite")
    scheduler = SchedulerStore(tmp_path / "scheduler.sqlite", tasks)
    work = WorkSpec(
        task_id="scheduled-fixture", plan_id="plan5", goal_id="fixture-goal",
        priority=60, deadline_tick=20, max_steps=3, cpu_units=1, memory_mb=16,
    )
    scheduler.register(work)
    assert scheduler.acquire(
        tick=1, available_cpu=1, available_memory_mb=16, pressure_ppm=950_000,
    ) is None
    lease = scheduler.acquire(tick=2, available_cpu=1, available_memory_mb=16)
    assert lease is not None
    assert scheduler.complete(lease, used_steps=1).status == "done"
    assert scheduler.acquire(tick=3, available_cpu=1, available_memory_mb=16) is None


def test_scenario_schema_and_invalid_inputs_fail_closed():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for bad in (
        {**payload, "schema": "legacy"},
        {**payload, "extra": "forbidden"},
        {**payload, "observed_value": ""},
        {**payload, "scenario_id": "../../escape"},
    ):
        with pytest.raises(FixtureError):
            Scenario.from_json(json.dumps(bad, ensure_ascii=False, sort_keys=True,
                                          separators=(",", ":")))
    with pytest.raises(FixtureError):
        Scenario.from_json('{"schema":"12-6.plan5-agent-fixture.v1","schema":"duplicate"}')

@pytest.mark.parametrize("plan_id,step_id", [
    ("other-plan", "fixture-start"),
    ("plan5", "foreign-start"),
])
def test_foreign_existing_task_is_rejected_before_any_mutation_or_model_call(
    tmp_path, plan_id, step_id,
):
    store = TaskStore(tmp_path / "tasks.sqlite")
    original = store.create(
        task_id=scenario().scenario_id, plan_id=plan_id, step_id=step_id,
    )
    runtime, model, adapter = harness(tmp_path)
    with pytest.raises(FixtureError, match="identity"):
        runtime.run()
    assert store.load(scenario().scenario_id) == original
    assert model.calls == 0
    assert adapter.dispatches == 0

