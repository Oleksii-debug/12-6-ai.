from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    ROOT / ".github/workflows/d02-s0-real-training.yml",
    ROOT / ".github/workflows/d02-s0-repeatability.yml",
    ROOT / ".github/workflows/d02-s1-numerical-preflight.yml",
)


def test_d02_specialist_workflows_are_scoped_and_cancellable() -> None:
    for path in WORKFLOWS:
        text = path.read_text(encoding="utf-8")
        assert "pull_request:\n    paths:" in text, path
        assert "workflow_dispatch:" in text, path
        assert "cancel-in-progress: true" in text, path
        assert "github.event.pull_request.head.sha || github.sha" in text, path
        assert "--run-repo-checks" not in text, path


def test_d02_specialist_workflows_keep_owned_dependency_surfaces() -> None:
    common = (
        '"src/twelve_six/model.py"',
        '"src/twelve_six/training/**"',
        '"src/twelve_six/packing/**"',
        '"src/twelve_six/tokenization/**"',
        '"data/s0/packaged/**"',
        '"tools/verify_locked_environment.py"',
        '"requirements/locks/linux-x86_64/**"',
        '"pyproject.toml"',
    )
    for path in WORKFLOWS:
        text = path.read_text(encoding="utf-8")
        for dependency in common:
            assert dependency in text, (path, dependency)

    assert '"configs/stages/s0_10k.json"' in WORKFLOWS[0].read_text(encoding="utf-8")
    assert '"configs/stages/s0_10k.json"' in WORKFLOWS[1].read_text(encoding="utf-8")
    assert '"configs/stages/s1_100k.json"' in WORKFLOWS[2].read_text(encoding="utf-8")
