from pathlib import Path

import pytest


WORKFLOWS = (
    Path(".github/workflows/d02-s0-real-training.yml"),
    Path(".github/workflows/d02-s0-repeatability.yml"),
    Path(".github/workflows/d02-s1-numerical-preflight.yml"),
)


@pytest.mark.parametrize("workflow", WORKFLOWS)
def test_d02_workflow_is_scoped_manual_and_cancellable(workflow: Path) -> None:
    text = workflow.read_text(encoding="utf-8")

    assert "  pull_request:\n    paths:\n" in text
    assert "  workflow_dispatch:\n" in text
    assert "concurrency:\n" in text
    assert "cancel-in-progress: true" in text
    assert "github.event.pull_request.head.sha || github.sha" in text
    assert f'- ".github/workflows/{workflow.name}"' in text


@pytest.mark.parametrize("workflow", WORKFLOWS)
def test_d02_workflow_tracks_shared_scientific_inputs(workflow: Path) -> None:
    text = workflow.read_text(encoding="utf-8")

    required_paths = (
        '      - "src/twelve_six/model.py"',
        '      - "src/twelve_six/training/**"',
        '      - "src/twelve_six/tokenization/**"',
        '      - "src/twelve_six/packing/**"',
        '      - "data/s0/**"',
        '      - "requirements/locks/**"',
        '      - "pyproject.toml"',
        '      - "tools/verify_locked_environment.py"',
    )
    for required_path in required_paths:
        assert required_path in text


def test_s0_workflows_track_exact_s0_stage_authority() -> None:
    for workflow in WORKFLOWS[:2]:
        text = workflow.read_text(encoding="utf-8")
        assert '      - "configs/stages/s0_10k.json"' in text


def test_s1_preflight_tracks_exact_s1_stage_authority() -> None:
    text = WORKFLOWS[2].read_text(encoding="utf-8")
    assert '      - "configs/stages/s1_100k.json"' in text
