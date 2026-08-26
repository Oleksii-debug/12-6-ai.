from __future__ import annotations

from pathlib import Path


ROUTED_WORKFLOWS = {
    ".github/workflows/d02-s0-real-training.yml": "src/twelve_six/training/trainer.py",
    ".github/workflows/d02-s0-repeatability.yml": "src/twelve_six/training/trainer.py",
    ".github/workflows/d02-s1-numerical-preflight.yml": "src/twelve_six/training/s1_preflight.py",
    ".github/workflows/data21-22-external-source-intake.yml": "src/twelve_six/data/source_intake.py",
    ".github/workflows/d08-purpose-environments.yml": "requirements/profiles/linux-x86_64-runtime.json",
    ".github/workflows/scale02-s2-1m-executable.yml": "src/twelve_six/checkpoint/trainer_adapter.py",
    ".github/workflows/train29-s1-observability.yml": "tests/test_training_observability.py",
}

CONTROL_ONLY_CHANGES = (
    ".github/workflows/ci.yml",
    "src/twelve_six/ci_workflow_policy.py",
    "configs/control/20m_readiness_controller_v1.json",
    "docs/CI_CONTROL_PLANE.md",
)


def _pull_request_paths(text: str) -> tuple[str, ...]:
    lines = text.splitlines()
    try:
        pull_request_index = next(
            index for index, line in enumerate(lines) if line == "  pull_request:"
        )
    except StopIteration as exc:
        raise AssertionError("workflow is missing top-level pull_request trigger") from exc

    paths_index: int | None = None
    for index in range(pull_request_index + 1, len(lines)):
        line = lines[index]
        if line and not line.startswith("    "):
            break
        if line == "    paths:":
            paths_index = index
            break
    if paths_index is None:
        raise AssertionError("pull_request trigger is missing paths filter")

    paths: list[str] = []
    for line in lines[paths_index + 1 :]:
        if not line.startswith("      - "):
            break
        value = line.removeprefix("      - ").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        paths.append(value)
    if not paths:
        raise AssertionError("pull_request.paths is empty")
    return tuple(paths)


def _matches_owned_path(pattern: str, path: str) -> bool:
    if pattern.endswith("/**"):
        return path.startswith(pattern[:-3] + "/")
    return pattern == path


def test_legacy_specialist_workflows_are_scoped_and_cancellable() -> None:
    for workflow_path, relevant_probe in ROUTED_WORKFLOWS.items():
        text = Path(workflow_path).read_text(encoding="utf-8")
        paths = _pull_request_paths(text)

        assert "  workflow_dispatch:" in text, workflow_path
        assert "concurrency:" in text, workflow_path
        assert "cancel-in-progress: true" in text, workflow_path
        assert any(_matches_owned_path(pattern, relevant_probe) for pattern in paths), workflow_path

        for control_path in CONTROL_ONLY_CHANGES:
            assert not any(
                _matches_owned_path(pattern, control_path) for pattern in paths
            ), f"{workflow_path} should not auto-run for control-only change {control_path}"


def test_manual_dispatch_keeps_exact_source_binding() -> None:
    fallback = "github.event.pull_request.head.sha || github.sha"
    for workflow_path in ROUTED_WORKFLOWS:
        text = Path(workflow_path).read_text(encoding="utf-8")
        assert fallback in text, workflow_path


def test_each_specialist_workflow_self_change_requalifies_it() -> None:
    for workflow_path in ROUTED_WORKFLOWS:
        paths = _pull_request_paths(Path(workflow_path).read_text(encoding="utf-8"))
        assert workflow_path in paths
