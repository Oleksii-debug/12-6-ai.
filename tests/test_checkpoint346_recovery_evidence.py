from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "checkpoint346"
V5_WORKFLOW = ROOT / ".github" / "workflows" / "checkpoint346-20m-recovery-qualification.yml"


def _load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def _sha256(name: str) -> str:
    return hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()


def _assert_recovery_result(result: dict, receipt: dict) -> None:
    assert result["verdict"] == "PASS_MODEL341_RECOVERY_MECHANICS"
    assert result["identity_sha256"] == receipt["result_identity_sha256"]
    assert result["carrier"]["git_sha"] == receipt["model341_carrier_sha"]
    assert result["carrier"]["parameter_count"] == 20_613_440
    assert result["carrier"]["canonical_base"] == "random_init"

    execution = result["execution"]
    assert execution["local_free"] is True
    assert execution["synthetic_mechanics_only"] is True
    assert execution["optimizer_updates_total"] == 3
    assert execution["fresh_process_distinct"] is True
    assert execution["same_next_step_equal"] is True
    assert execution["binding_mismatch_failed_closed"] is True
    assert execution["rng_scope_restored"] == {
        "numpy": True,
        "python": True,
        "torch_cpu": True,
    }

    for key in (
        "loss",
        "update_loss",
        "optimizer_step",
        "micro_step",
        "tokens_seen",
        "model_state_sha256",
        "optimizer_state_sha256",
        "trainer_state_sha256",
        "rng_probe_before_step",
    ):
        assert result["baseline_step2"][key] == result["resumed_step2"][key]

    credit = result["scientific_credit"]
    assert credit == {
        "final_test_read": False,
        "foreign_pretrained_weights": False,
        "learned_weights_created": False,
        "model_training_credit": False,
        "optimized_target_exposure": 0,
        "paid_compute_used": False,
        "real_corpus_used": False,
        "training_authorized_corpus_used": False,
    }
    assert receipt["scientific_credit"] == credit


def test_checkpoint346_v4_remains_immutable_historical_evidence() -> None:
    receipt = _load("model341_recovery_execution_receipt_v4.json")
    result = _load("model341_recovery_result_v4.json")
    environment = _load("model341_recovery_environment_v4.json")

    assert receipt["workflow_run_id"] == 34536431404
    assert receipt["workflow_job_id"] == 103069022528
    assert receipt["executed_head_sha"] == "cda305f06b90b3d17a28b3c928771f726b03eec3"
    assert receipt["model341_carrier_sha"] == "133867d21a94637920b7a24dfc046dc09371ab5c"
    assert receipt["runner_path"] == "tools/run_checkpoint346_model341_recovery.py"
    assert receipt["runner_git_blob_sha"] == "524ae13a6f9ad51de50157295b7d3fb525dcbc5b"

    assert _sha256("model341_recovery_result_v4.json") == receipt["artifact"]["result_file_sha256"]
    assert _sha256("model341_recovery_environment_v4.json") == receipt["artifact"]["environment_file_sha256"]
    assert receipt["artifact"]["zip_sha256"] == (
        "ab5f91407cb4a98470dca143b35c91cbb2dbdc7897c441e14cab08d213452636"
    )
    assert environment["identity_sha256"] == receipt["environment_identity_sha256"]
    _assert_recovery_result(result, receipt)


def test_checkpoint346_v5_binds_current_runner_and_current_model341_carrier() -> None:
    receipt = _load("model341_recovery_execution_receipt_v5.json")
    result = _load("model341_recovery_result_v5.json")
    environment = _load("model341_recovery_environment_v5.json")

    assert receipt["workflow_run_id"] == 34538615652
    assert receipt["workflow_job_id"] == 103075876817
    assert receipt["executed_head_sha"] == "c49bea6caf2858a4bf63203f6d85f05ca9aa8af7"
    assert receipt["executed_pr"] == 426
    assert receipt["model341_carrier_sha"] == "82c43005bb5db153482ae5b20a31d59240faaebc"

    assert _sha256("model341_recovery_result_v5.json") == receipt["artifact"]["result_file_sha256"]
    assert _sha256("model341_recovery_environment_v5.json") == receipt["artifact"]["environment_file_sha256"]
    assert receipt["artifact"]["id"] == 10176452651
    assert receipt["artifact"]["zip_sha256"] == (
        "a29e2944def1ed11940f16ea740ca0d81a8ac5610906f1f199572c41cd63b268"
    )

    runner_blob = subprocess.check_output(
        ["git", "hash-object", receipt["runner_path"]],
        cwd=ROOT,
        text=True,
    ).strip()
    assert runner_blob == receipt["runner_git_blob_sha"]
    assert runner_blob == "2d0098e6914a1ec90091f1455736d42d8107d7ed"

    assert receipt["carrier_git_blobs"] == {
        "candidate": "69e3cbd5f5c83c9d3d529a2a6376db3055979c40",
        "checkpoint_core": "8bff5f2cad79f64532f60dce3b413aa5e385d97a",
        "model": "d0823aa666883ddb5c445a438730043ef8b50ff1",
        "trainer": "049394ae119a5a5ac301ceded6ae6dfcae66c020",
    }
    assert receipt["execution_workflow_path"] == (
        ".github/workflows/checkpoint346-20m-recovery-qualification.yml"
    )
    assert receipt["execution_workflow_git_blob_sha"] == (
        "30a92252f57d9407b846ee35326b83a06a419059"
    )
    assert receipt["execution_workflow_removed_from_final_head"] is True
    assert not V5_WORKFLOW.exists()
    assert receipt["historical_v4_immutable"] is True

    assert environment["identity_sha256"] == receipt["environment_identity_sha256"]
    _assert_recovery_result(result, receipt)

    v4 = _load("model341_recovery_execution_receipt_v4.json")
    assert v4["workflow_run_id"] != receipt["workflow_run_id"]
    assert v4["executed_head_sha"] != receipt["executed_head_sha"]
    assert v4["model341_carrier_sha"] != receipt["model341_carrier_sha"]
    assert v4["runner_git_blob_sha"] != receipt["runner_git_blob_sha"]
