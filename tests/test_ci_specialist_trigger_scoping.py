from pathlib import Path


WORKFLOWS = (
    "d02-s0-real-training.yml",
    "d02-s0-repeatability.yml",
    "d02-s1-numerical-preflight.yml",
    "d08-purpose-environments.yml",
    "data21-22-external-source-intake.yml",
    "scale02-s2-1m-executable.yml",
    "train29-s1-observability.yml",
)


def _workflow_text(name: str) -> str:
    path = Path(".github/workflows") / name
    return path.read_text(encoding="utf-8")


def test_specialist_workflows_are_path_scoped_manual_and_cancellable() -> None:
    for name in WORKFLOWS:
        text = _workflow_text(name)
        assert "pull_request:\n    paths:" in text, name
        assert "workflow_dispatch:" in text, name
        assert "\nconcurrency:\n" in text, name
        assert "cancel-in-progress: true" in text, name
        assert f'      - ".github/workflows/{name}"' in text, name


def test_specialist_workflows_support_exact_manual_head_fallback() -> None:
    for name in WORKFLOWS:
        text = _workflow_text(name)
        assert "github.event.pull_request.head.sha || github.sha" in text, name


def test_specialist_workflows_do_not_use_global_source_or_test_globs() -> None:
    for name in WORKFLOWS:
        text = _workflow_text(name)
        assert '- "src/**"' not in text, name
        assert '- "tests/**"' not in text, name
        assert '- "**"' not in text, name
