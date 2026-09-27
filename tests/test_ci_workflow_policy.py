from pathlib import Path

from twelve_six.ci_workflow_policy import parse_name_status, policy_violations


def test_allows_normal_source_changes():
    changes = parse_name_status("M\tsrc/twelve_six/model.py\nA\ttests/test_model.py\n")
    assert policy_violations(changes) == []


def test_allows_modifying_existing_workflow():
    changes = parse_name_status("M\t.github/workflows/legacy.yml\n")
    assert policy_violations(changes) == []


def test_blocks_new_dedicated_workflow():
    changes = parse_name_status("A\t.github/workflows/worker-123.yml\n")
    assert policy_violations(changes) == [
        "new dedicated workflow is prohibited: .github/workflows/worker-123.yml"
    ]


def test_blocks_copy_or_rename_into_new_workflow():
    changes = parse_name_status(
        "C100\t.github/workflows/ci.yml\t.github/workflows/copy.yml\n"
        "R100\told.yml\t.github/workflows/renamed.yml\n"
    )
    assert len(policy_violations(changes)) == 2


def test_blocks_removing_canonical_ci():
    deletion = parse_name_status("D\t.github/workflows/ci.yml\n")
    rename = parse_name_status(
        "R100\t.github/workflows/ci.yml\t.github/workflows/ci-renamed.yml\n"
    )
    assert policy_violations(deletion) == ["canonical CI workflow may not be deleted"]
    assert "canonical CI workflow may not be renamed" in policy_violations(rename)


def test_d03_artifact_job_is_same_repo_and_pr_pinned():
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    job_start = workflow.index("  d03-current-clean-physical:\n")
    job = workflow[job_start:]
    assert "github.event.pull_request.number == 2211" in job
    assert (
        "github.event.pull_request.head.repo.full_name == github.repository"
        in job
    )
    assert (
        "github.head_ref == 'd03/2023-clean-current-main-execution-2210'"
        in job
    )
    assert "pull_request_target" not in job
