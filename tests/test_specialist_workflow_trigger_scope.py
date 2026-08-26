from pathlib import Path


SCOPED_SPECIALIST_WORKFLOWS = (
    ".github/workflows/d02-s0-real-training.yml",
    ".github/workflows/d02-s0-repeatability.yml",
    ".github/workflows/d02-s1-numerical-preflight.yml",
    ".github/workflows/scale02-s2-1m-executable.yml",
    ".github/workflows/d08-purpose-environments.yml",
    ".github/workflows/train29-s1-observability.yml",
)

DIRECT_INPUTS_BY_WORKFLOW = {
    ".github/workflows/d02-s0-real-training.yml": (
        "data/s0/packaged/**",
        "requirements/locks/index.json",
    ),
    ".github/workflows/d02-s0-repeatability.yml": (
        "data/s0/packaged/**",
        "requirements/locks/index.json",
    ),
    ".github/workflows/d02-s1-numerical-preflight.yml": (
        "data/s0/packaged/**",
        "requirements/locks/index.json",
    ),
    ".github/workflows/scale02-s2-1m-executable.yml": (
        "configs/stages/alternatives/s2_1m_byte_gqa.candidate.json",
        "data/s0/packaged/**",
        "requirements/locks/index.json",
    ),
    ".github/workflows/train29-s1-observability.yml": (
        "data/s0/packaged/**",
        "requirements/locks/index.json",
    ),
}


def test_specialist_workflows_are_scoped_cancellable_and_dispatchable() -> None:
    root = Path(__file__).resolve().parents[1]
    manual_exact_head = "${{ github.event.pull_request.head.sha || github.sha }}"

    for workflow_path in SCOPED_SPECIALIST_WORKFLOWS:
        text = (root / workflow_path).read_text(encoding="utf-8")
        assert "pull_request:\n    paths:" in text, workflow_path
        assert f'- "{workflow_path}"' in text, workflow_path
        assert "\n  workflow_dispatch:\n" in text, workflow_path
        assert "\nconcurrency:\n" in text, workflow_path
        assert "cancel-in-progress: true" in text, workflow_path
        assert manual_exact_head in text, workflow_path


def test_specialist_scopes_keep_direct_runtime_inputs() -> None:
    root = Path(__file__).resolve().parents[1]

    for workflow_path, direct_inputs in DIRECT_INPUTS_BY_WORKFLOW.items():
        text = (root / workflow_path).read_text(encoding="utf-8")
        for direct_input in direct_inputs:
            assert f'- "{direct_input}"' in text, f"{workflow_path}: {direct_input}"
