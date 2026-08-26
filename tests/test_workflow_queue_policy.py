from __future__ import annotations

from twelve_six.integration.workflow_queue_policy import validate_workflow_text, workflow_triggers


def test_added_pull_request_workflow_is_rejected_even_with_concurrency() -> None:
    text = """name: dedicated
on:
  pull_request:
concurrency:
  group: dedicated-${{ github.ref }}
  cancel-in-progress: true
jobs:
  test:
    runs-on: ubuntu-24.04
    timeout-minutes: 5
    steps: []
"""
    violations = validate_workflow_text(".github/workflows/dedicated.yml", "A", text)
    assert any("may not auto-trigger" in item for item in violations)


def test_added_manual_workflow_is_allowed_with_timeout() -> None:
    text = """name: manual
on:
  workflow_dispatch:
jobs:
  test:
    runs-on: ubuntu-24.04
    timeout-minutes: 5
    steps: []
"""
    assert validate_workflow_text(".github/workflows/manual.yml", "A", text) == ()


def test_modified_automatic_workflow_requires_concurrency() -> None:
    text = """name: existing
on:
  pull_request:
jobs:
  test:
    runs-on: ubuntu-24.04
    timeout-minutes: 5
    steps: []
"""
    violations = validate_workflow_text(".github/workflows/existing.yml", "M", text)
    assert any("top-level concurrency" in item for item in violations)
    assert any("cancel-in-progress" in item for item in violations)


def test_modified_automatic_workflow_with_cancellation_is_allowed() -> None:
    text = """name: existing
on:
  pull_request:
concurrency:
  group: existing-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true
jobs:
  test:
    runs-on: ubuntu-24.04
    timeout-minutes: 5
    steps: []
"""
    assert validate_workflow_text(".github/workflows/existing.yml", "M", text) == ()


def test_inline_automatic_trigger_is_detected() -> None:
    text = """name: inline
on: [push, workflow_dispatch]
jobs:
  test:
    runs-on: ubuntu-24.04
    timeout-minutes: 5
    steps: []
"""
    assert workflow_triggers(text) == frozenset({"push", "workflow_dispatch"})
    violations = validate_workflow_text(".github/workflows/inline.yaml", "A", text)
    assert any("may not auto-trigger" in item for item in violations)


def test_new_runnable_workflow_requires_timeout() -> None:
    text = """name: manual
on:
  workflow_dispatch:
jobs:
  test:
    runs-on: ubuntu-24.04
    steps: []
"""
    violations = validate_workflow_text(".github/workflows/manual.yml", "A", text)
    assert violations == (".github/workflows/manual.yml: new runnable workflow requires timeout-minutes",)
