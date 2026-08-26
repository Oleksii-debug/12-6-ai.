from pathlib import Path


SCOPED_SPECIALIST_WORKFLOWS = (
    ".github/workflows/d02-s0-real-training.yml",
    ".github/workflows/d02-s0-repeatability.yml",
    ".github/workflows/d02-s1-numerical-preflight.yml",
    ".github/workflows/scale02-s2-1m-executable.yml",
    ".github/workflows/d08-purpose-environments.yml",
    ".github/workflows/train29-s1-observability.yml",
)


def test_specialist_workflows_are_path_scoped_and_cancel_superseded_runs():
    root = Path(__file__).resolve().parents[1]

    for workflow_path in SCOPED_SPECIALIST_WORKFLOWS:
        text = (root / workflow_path).read_text(encoding="utf-8")
        assert "pull_request:\n    paths:" in text, workflow_path
        assert f'- "{workflow_path}"' in text, workflow_path
        assert "\nconcurrency:\n" in text, workflow_path
        assert "cancel-in-progress: true" in text, workflow_path
