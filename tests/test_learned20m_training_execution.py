import json
from pathlib import Path

import pytest

from twelve_six.learned20m_training_execution import (
    ExecutionContext,
    _read_repo_json,
    assess_training_execution,
)

SHA = "a" * 40
H = "b" * 64


def _manifest(**overrides):
    manifest = {
        "schema_version": 1,
        "manifest_id": "R01-LEARNED20M-LAUNCH-MANIFEST-V1",
        "stage": "LEARNED_20M",
        "identities": {
            "source_git_sha": SHA,
            "modelspec_sha256": H,
            "initspec_sha256": H,
            "tokenizer_sha256": H,
            "corpus_manifest_sha256": H,
            "split_sha256": H,
            "packing_sha256": H,
            "unique_loss_ledger_sha256": H,
            "training_config_sha256": H,
            "portable_run_packet_sha256": H,
            "portable_run_binding_sha256": H,
        },
        "recipe": {
            "optimizer_scheduler_precision": "AdamW|cosine|fp32",
            "seed": 1,
            "target_unique_loss_positions": 100,
            "maximum_total_exposures": 100,
        },
        "checkpoint": {
            "lineage": "FRESH_START",
            "checkpoint_contract_sha256": H,
        },
        "evaluation": {
            "firewall_sha256": H,
            "final_test_payload_access": False,
        },
        "resource": {
            "resource_class": "GITHUB_HOSTED_FREE",
            "maximum_cost_usd": 0,
            "materially_paid": False,
        },
        "authorities": {
            "training": {
                "reference": "issue:#2035",
                "evidence_sha256": H,
            },
            "compute": {
                "reference": "LOCAL_FREE:GITHUB_HOSTED_FREE",
                "evidence_sha256": H,
            },
        },
        "execution_backend": "PROJECT_NATIVE_PYTORCH",
    }
    manifest.update(overrides)
    return manifest


def _context(**overrides):
    values = {
        "event_name": "workflow_dispatch",
        "repository": "Oleksii-debug/12-6-ai.",
        "ref": "refs/heads/main",
        "github_sha": SHA,
        "requested_source_sha": SHA,
    }
    values.update(overrides)
    return ExecutionContext(**values)


def test_valid_carrier_context_still_blocks_until_external_authorities_integrate():
    result = assess_training_execution(_manifest(), _context())
    assert result.context_valid is True
    assert result.manifest_valid is True
    assert result.github_hosted_free_bound is True
    assert result.optimizer_start_permitted is False
    assert result.scientific_truth_changed is False
    assert set(result.blockers) == {
        "global_cross_runner_training_lease_not_integrated",
        "terminal_prestep_pass_not_integrated",
        "real_target_training_worker_not_integrated",
    }


@pytest.mark.parametrize(
    ("change", "blocker"),
    [
        ({"event_name": "push"}, "event_must_be_workflow_dispatch"),
        ({"repository": "attacker/repo"}, "repository_mismatch"),
        ({"ref": "refs/heads/feature"}, "ref_must_be_main"),
        ({"requested_source_sha": "c" * 40}, "requested_source_sha_not_current_checkout"),
    ],
)
def test_dispatch_context_is_closed_world(change, blocker):
    result = assess_training_execution(_manifest(), _context(**change))
    assert blocker in result.blockers
    assert result.optimizer_start_permitted is False


def test_manifest_source_must_match_checkout():
    manifest = _manifest(identities={"source_git_sha": "c" * 40})
    result = assess_training_execution(manifest, _context())
    assert "manifest_source_git_sha_not_current_checkout" in result.blockers


def test_paid_or_wrong_resource_never_qualifies():
    manifest = _manifest(
        resource={
            "resource_class": "LOCAL_FREE",
            "maximum_cost_usd": 1,
            "materially_paid": True,
        }
    )
    result = assess_training_execution(manifest, _context())
    assert result.github_hosted_free_bound is False
    assert "manifest_not_bound_to_github_hosted_free" in result.blockers


def test_manifest_contract_error_is_preserved_fail_closed():
    manifest = _manifest(schema_version=2)
    result = assess_training_execution(manifest, _context())
    assert result.manifest_valid is False
    assert "schema_version_mismatch" in result.contract_errors
    assert "launch_manifest_contract_invalid" in result.blockers


def test_manifest_locator_cannot_escape_repo(tmp_path: Path):
    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes_repository"):
        _read_repo_json(tmp_path, "../outside.json")


def test_manifest_locator_reads_repo_relative_object(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    assert _read_repo_json(tmp_path, "manifest.json") == {"ok": True}


def test_single_ci_workflow_contains_manual_only_training_lane():
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "learned20m-authorized-training:" in workflow
    assert "if: github.event_name == 'workflow_dispatch'" in workflow
    assert "needs: bootstrap" in workflow
    assert "cancel-in-progress: ${{ github.event_name != 'workflow_dispatch' }}" in workflow
    assert "python -m twelve_six.learned20m_training_execution" in workflow
    assert "permissions:\n      contents: read" in workflow
    assert "schedule:" not in workflow
