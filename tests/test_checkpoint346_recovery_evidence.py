from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "checkpoint346"


def _load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def _sha256(name: str) -> str:
    return hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()


def test_checkpoint346_execution_receipt_binds_real_run_to_current_runner() -> None:
    receipt = _load("model341_recovery_execution_receipt_v4.json")
    result = _load("model341_recovery_result_v4.json")
    environment = _load("model341_recovery_environment_v4.json")

    assert receipt["workflow_run_id"] == 34536431404
    assert receipt["workflow_job_id"] == 103069022528
    assert receipt["executed_head_sha"] == "cda305f06b90b3d17a28b3c928771f726b03eec3"
    assert receipt["model341_carrier_sha"] == "133867d21a94637920b7a24dfc046dc09371ab5c"

    assert _sha256("model341_recovery_result_v4.json") == receipt["artifact"]["result_file_sha256"]
    assert _sha256("model341_recovery_environment_v4.json") == receipt["artifact"]["environment_file_sha256"]
    assert receipt["artifact"]["zip_sha256"] == "ab5f91407cb4a98470dca143b35c91cbb2dbdc7897c441e14cab08d213452636"

    runner_blob = subprocess.check_output(
        ["git", "hash-object", receipt["runner_path"]],
        cwd=ROOT,
        text=True,
    ).strip()
    assert runner_blob == receipt["runner_git_blob_sha"]
    assert runner_blob == "524ae13a6f9ad51de50157295b7d3fb525dcbc5b"

    assert result["verdict"] == "PASS_MODEL341_RECOVERY_MECHANICS"
    assert result["identity_sha256"] == receipt["result_identity_sha256"]
    assert environment["identity_sha256"] == receipt["environment_identity_sha256"]
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
