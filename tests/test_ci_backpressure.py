from pathlib import Path

from twelve_six.ci_backpressure import inspect_workflow, validate_inventory

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "configs" / "ci" / "legacy_workflow_backpressure_v1.json"


def test_live_workflow_inventory_is_complete_and_backpressured():
    assert validate_inventory(ROOT, INVENTORY) == []


def test_inspector_detects_unprotected_broad_pull_request_workflow(tmp_path: Path):
    workflow = tmp_path / ".github" / "workflows" / "expensive.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "name: Expensive\n"
        "on:\n"
        "  pull_request:\n"
        "permissions:\n"
        "  contents: read\n"
        "jobs:\n"
        "  test:\n"
        "    runs-on: ubuntu-24.04\n",
        encoding="utf-8",
    )

    observed = inspect_workflow(workflow, repo_root=tmp_path)
    assert observed.pull_request is True
    assert observed.paths_scoped is False
    assert observed.concurrency is False
    assert observed.cancel_in_progress is False
    assert observed.pr_scoped_group is False


def test_inspector_detects_path_scope_and_pr_backpressure(tmp_path: Path):
    workflow = tmp_path / ".github" / "workflows" / "focused.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "name: Focused\n"
        "on:\n"
        "  pull_request:\n"
        "    paths:\n"
        "      - 'src/**'\n"
        "  workflow_dispatch:\n"
        "concurrency:\n"
        "  group: focused-${{ github.event.pull_request.number || github.ref }}\n"
        "  cancel-in-progress: true\n"
        "jobs:\n"
        "  test:\n"
        "    runs-on: ubuntu-24.04\n",
        encoding="utf-8",
    )

    observed = inspect_workflow(workflow, repo_root=tmp_path)
    assert observed.pull_request is True
    assert observed.paths_scoped is True
    assert observed.concurrency is True
    assert observed.cancel_in_progress is True
    assert observed.pr_scoped_group is True
