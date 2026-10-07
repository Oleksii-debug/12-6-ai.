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
    assert (
        'type(composition.get("authorized_optimized_target_exposure")) is not int'
        in job
    )
    assert (
        'type(composition.get("optimizer_updates_executed_on_real_targets")) is not int'
        in job
    )


def test_d03_selection_reconstruction_is_exact_and_retention_independent():
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    job_start = workflow.index("  d03-current-clean-physical:\n")
    job = workflow[job_start:]
    compact_job = " ".join(job.replace(chr(92) + "\n", " ").split())

    assert 'EVAL290_HEAD_SHA: "029514654829cebc149cff6fc1fea2a8ba4fa566"' in job
    assert 'EVAL291_HEAD_SHA: "fb268061300127b62cc2a262664b30c614559dac"' in job
    assert "git fetch --no-tags origin refs/pull/402/head" in job
    assert 'test "$(git rev-parse FETCH_HEAD)" = "$EVAL290_HEAD_SHA"' in job
    assert "git fetch --no-tags origin refs/pull/395/head" in job
    assert 'test "$(git rev-parse FETCH_HEAD)" = "$EVAL291_HEAD_SHA"' in job
    assert 'eval290_out_a="$RUNNER_TEMP/eval290-out-a"' in job
    assert 'eval290_out_b="$RUNNER_TEMP/eval290-out-b"' in job
    assert 'diff -ru "$eval290_out_a" "$eval290_out_b"' in job
    assert (
        'python "$eval290_src/tools/execution_bootstrap.py" bootstrap '
        '--repo-root "$eval290_src" --capabilities "runtime,tests" '
        '--venv "$eval290_venv" --manifest "$eval290_environment" '
        '--command "python -m pytest -q tests/test_eval290_ua_selection.py"'
        in compact_job
    )
    assert (
        'PYTHONPATH="$eval290_src/src" "$eval290_venv/bin/python" -m '
        'twelve_six.eval290_ua_selection build --repo-root "$eval290_src"'
        in compact_job
    )
    assert (
        'PYTHONPATH="$eval290_src/src" "$eval290_venv/bin/python" -m '
        'twelve_six.eval290_ua_selection verify --repo-root "$eval290_src"'
        in compact_job
    )
    assert 'numpy.__version__ == "2.4.6"' in job
    assert 'pyarrow.__version__ == "25.0.1"' in job
    assert 'eval291_venv="$RUNNER_TEMP/eval291-venv"' in job
    assert 'eval291_environment="$RUNNER_TEMP/eval291-environment.json"' in job
    assert (
        'python "$eval291_src/tools/execution_bootstrap.py" bootstrap '
        '--repo-root "$eval291_src" --capabilities "runtime,tests" '
        '--venv "$eval291_venv" --manifest "$eval291_environment" '
        '--command "python -m pytest -q tests/test_eval291_en_selection_validation.py"'
        in compact_job
    )
    assert (
        'PYTHONPATH="$eval291_src/src" "$eval291_venv/bin/python" -m '
        'twelve_six.eval291_en_selection_validation build --repo-root "$eval291_src"'
        in compact_job
    )
    assert (
        'PYTHONPATH="$eval291_src/src" "$eval291_venv/bin/python" -m '
        'twelve_six.eval291_en_selection_validation verify --repo-root "$eval291_src"'
        in compact_job
    )
    assert (
        'PYTHONPATH="$eval291_src/src" python -m '
        'twelve_six.eval291_en_selection_validation'
        not in compact_job
    )
    assert 'cmp "$RUNNER_TEMP/eval291-en.jsonl"' in job
    assert "resolve_eval303_selection_payloads_from_reconstructed" in job
    assert "Fetch immutable selection-validation payload artifacts" not in job
    assert "EVAL290_ARTIFACT_ID:" not in job
    assert "EVAL291_ARTIFACT_ID:" not in job


def test_d03_rada_fresh_snapshot_job_is_same_repo_and_claim_pinned():
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    job_start = workflow.index("  d03-rada-fresh-snapshot-v2:\n")
    job = workflow[job_start:]

    assert "github.event_name == 'pull_request'" in job
    assert "github.event.pull_request.head.repo.full_name == github.repository" in job
    assert "github.head_ref == 'swarm/2477-rada-laws-fresh-snapshot-v2'" in job
    assert "pull_request_target" not in job
    assert job.count("--accept-current-upstream") == 2
    assert job.count("--archive-output") == 2
    assert "--rights-policy configs/data/d03_rada_bulk_fresh_snapshot_rights_v2.json" in job
    assert '--attribution-output "$RUNNER_TEMP/ATTRIBUTION.txt"' in job
    assert "expected_current_observation" not in job
    assert 'cmp "$RUNNER_TEMP/rada-source-a.zip" "$RUNNER_TEMP/rada-source-b.zip"' in job
    assert 'test -s "$RUNNER_TEMP/ATTRIBUTION.txt"' in job
    assert "Upload attributed exact Rada snapshot and qualification evidence" in job
    assert "Upload Rada probe metadata on qualification failure" in job
    assert "if: failure()" in job
    assert "if: always()" not in job
    success_upload = job.index(
        "Upload attributed exact Rada snapshot and qualification evidence"
    )
    failure_upload = job.index("Upload Rada probe metadata on qualification failure")
    success_section = job[success_upload:failure_upload]
    failure_section = job[failure_upload:]
    assert "rada-source-a.zip" in success_section
    assert "ATTRIBUTION.txt" in success_section
    assert "rada-source-b.zip" not in success_section
    assert "rada-source-a.zip" not in failure_section
    assert "rada-source-b.zip" not in failure_section
    assert "if-no-files-found: error" in success_section
    assert "retention-days: 90" in success_section
    assert "if-no-files-found: warn" in failure_section
    assert "retention-days: 30" in failure_section


def test_d03_rada_physical_success_requires_exact_run_successor_pin():
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    job = workflow[workflow.index("  d03-rada-fresh-snapshot-v2:\n"):]
    compile_step = job.index("Bind exact Rada fresh-snapshot checkout")
    qualification = job.index("Qualify two-clean exact snapshot and attributed retention")
    pin = job.index("Verify and pin retained Rada snapshot to exact run")
    upload = job.index("Upload attributed exact Rada snapshot and qualification evidence")
    failure = job.index("Upload Rada probe metadata on qualification failure")

    assert compile_step < qualification < pin < upload < failure
    assert "tools/pin_d03_rada_bulk_fresh_snapshot_v2.py" in job[
        compile_step:qualification
    ]
    pin_step = job[pin:upload]
    for expected in (
        "python tools/pin_d03_rada_bulk_fresh_snapshot_v2.py",
        '--archive "$RUNNER_TEMP/rada-source-a.zip"',
        '--probe-report "$RUNNER_TEMP/rada-probe-a.json"',
        '--probe-b "$RUNNER_TEMP/rada-probe-b.json"',
        '--qualification "$RUNNER_TEMP/rada-fresh-snapshot-qualification-v2.json"',
        '--attribution "$RUNNER_TEMP/ATTRIBUTION.txt"',
        "--config configs/data/d03_rada_bulk_fresh_snapshot_v2.json",
        "--rights-policy configs/data/d03_rada_bulk_fresh_snapshot_rights_v2.json",
        '--execution-head-sha "$RADA_CAPTURE_SHA"',
        '--output "$RUNNER_TEMP/rada-successor-pin-v2.json"',
        'test -s "$RUNNER_TEMP/rada-successor-pin-v2.json"',
    ):
        assert expected in pin_step
    success_section = job[upload:failure]
    failure_section = job[failure:]
    assert "${{ runner.temp }}/rada-successor-pin-v2.json" in success_section
    assert "rada-successor-pin-v2.json" not in failure_section
    assert "rada-source-a.zip" not in failure_section
    assert "if: failure()" in failure_section
