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
