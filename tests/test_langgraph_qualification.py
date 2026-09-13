from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twelve_six.langgraph_qualification import (  # noqa: E402
    ContractError,
    atomic_write_json,
    benchmark_project_checkpoint,
    project_transition,
    read_json,
    validate_task_state,
)


def fixture():
    return {
        "schema_version": 1,
        "task_id": "demo",
        "status": "READY",
        "goal": "test",
        "completed_steps": [],
        "pending_steps": ["one", "two"],
        "evidence": [],
        "checkpoint_seq": 0,
    }


def test_state_transition_and_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    state = project_transition(fixture())
    atomic_write_json(path, state)
    assert read_json(path) == state
    assert state["completed_steps"] == ["one"]
    assert state["checkpoint_seq"] == 1


def test_schema_rejects_unknown_fields():
    state = fixture()
    state["attacker"] = "x"
    with pytest.raises(ContractError):
        validate_task_state(state)


def test_schema_rejects_duplicate_or_overlapping_steps():
    state = fixture()
    state["completed_steps"] = ["one", "one"]
    with pytest.raises(ContractError):
        validate_task_state(state)
    state = fixture()
    state["completed_steps"] = ["one"]
    with pytest.raises(ContractError):
        validate_task_state(state)


def test_corrupt_json_fails(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"schema_version":1}', encoding="utf-8")
    with pytest.raises(Exception):
        read_json(path)


def test_unsafe_task_id_fails():
    state = fixture()
    state["task_id"] = "../escape"
    with pytest.raises(ContractError):
        validate_task_state(state)


def test_checkpoint_is_deterministic_for_same_input(tmp_path):
    path = tmp_path / "state.json"
    a = fixture()
    b = fixture()
    assert atomic_write_json(path, a) == atomic_write_json(path, b)


def test_benchmark_is_real_project_io_only():
    result = benchmark_project_checkpoint(10)
    assert result["iterations"] == 10
    assert result["ops_per_second"] > 0
