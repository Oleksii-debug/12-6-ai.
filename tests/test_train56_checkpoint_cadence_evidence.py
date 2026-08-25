from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_train56_measured_cadence_and_resume_equivalence(tmp_path: Path) -> None:
    report_path = tmp_path / "train56-checkpoint-cadence.json"
    completed = subprocess.run(
        [
            sys.executable,
            "tools/run_checkpoint_cadence_experiment.py",
            "--output",
            str(report_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "TRAIN56_REPORT_BEGIN" in completed.stdout
    assert "TRAIN56_REPORT_END" in completed.stdout

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["checkpoint_format"] == "incumbent 12-6-checkpoint v1"
    assert [stage["label"] for stage in report["stages"]] == ["~100K", "~1M"]
    assert [stage["actual_parameters"] for stage in report["stages"]] == [95_568, 1_037_696]
    for stage in report["stages"]:
        assert stage["optimizer_step_median_s"] > 0.0
        assert stage["checkpoint_save_verify_median_s"] > 0.0
        assert stage["checkpoint_explicit_verify_median_s"] > 0.0
        assert stage["fresh_load_median_s"] > 0.0
        assert stage["checkpoint_bytes"] > 0
        assert stage["valid_causal_tokens_per_step"] == 256
        assert len(stage["cadence_targets"]) >= 3
        assert stage["selected_cadence"]["interval_steps"] >= 1
        assert stage["selected_cadence"]["lost_work_seconds"] > 0.0

    equivalence = report["interrupted_equivalence"]
    assert equivalence["control_optimizer_step"] == equivalence["resumed_optimizer_step"]
    assert equivalence["control_tokens_seen"] == equivalence["resumed_tokens_seen"]
    assert equivalence["model_state_exact_equal"] is True
    assert equivalence["trainer_state_exact_equal"] is True
    assert equivalence["control_model_sha256"] == equivalence["resumed_model_sha256"]
