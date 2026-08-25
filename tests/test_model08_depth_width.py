from __future__ import annotations

import subprocess
from pathlib import Path

from twelve_six.model08_depth_width import run_model08_candidate


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_repo_root(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_small_model08_run_retains_exact_token_weighted_train_loss(tmp_path: Path) -> None:
    report = run_model08_candidate(
        repo_root=_repo_root(),
        source_sha=_head(),
        candidate_id="balanced_d48_l3",
        output_path=tmp_path / "model08.json",
        checkpoint_dir=tmp_path / "checkpoints",
        token_budgets=(13, 31),
        batch_size=2,
        sequence_length=8,
        torch_threads=1,
    )
    telemetry = report["training_loss_telemetry"]
    assert telemetry["overall"]["optimized_tokens"] == 31
    assert telemetry["overall"]["steps"] == report["trace_steps"]
    assert [segment["optimized_tokens"] for segment in telemetry["segments"]] == [13, 18]
    assert telemetry["segments"][0]["segment_end_optimized_tokens"] == 13
    assert telemetry["segments"][1]["segment_end_optimized_tokens"] == 31
    assert telemetry["overall"]["token_weighted_mean"] is not None
    assert report["resume_exercised"] is True
