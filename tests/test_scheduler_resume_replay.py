from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace


def _load_evidence_module(repo_root: Path):
    path = repo_root / "tools/run_s1_scheduler_resume_evidence.py"
    spec = importlib.util.spec_from_file_location("train51_scheduler_resume", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s1_scheduler_resume_exact_replay(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_evidence_module(repo_root)
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    args = SimpleNamespace(
        source_sha=source_sha,
        seed=20260825,
        max_steps=48,
        warmup_steps=6,
        sequence_length=32,
        workspace=str(tmp_path / "scheduler-resume"),
    )

    evidence = module._parent_run(args, repo_root)

    assert evidence["identity"]["parameter_count"] == 107856
    assert evidence["identity"]["precision"] == "fp32"
    assert evidence["scheduler_semantics"]["off_by_one_lr_application_detected"] is False
    assert evidence["scheduler_semantics"]["post_final_next_lr"] == 0.0
    assert [item["split_after_optimizer_step"] for item in evidence["interrupted_replays"]] == [
        5,
        6,
        26,
        27,
        46,
        47,
    ]
    assert all(item["exact_lr_sequence_match"] for item in evidence["interrupted_replays"])
    assert all(item["exact_counter_sequence_match"] for item in evidence["interrupted_replays"])
    assert all(item["exact_final_model_match"] for item in evidence["interrupted_replays"])
    assert all(item["exact_final_trainer_match"] for item in evidence["interrupted_replays"])

    fallback = evidence["corrupt_latest_fallback"]
    assert fallback["older_verified_checkpoint_step"] == 46
    assert fallback["newer_corrupted_checkpoint_step"] == 47
    assert fallback["selected_fallback_step"] == 46
    assert fallback["discarded_update_47_lr"] == fallback["replayed_update_47_lr"]
    assert fallback["exact_lr_sequence_match_after_rollback"] is True
    assert fallback["exact_counter_sequence_match_after_rollback"] is True
    assert fallback["exact_final_model_match"] is True
    assert fallback["exact_final_trainer_match"] is True
    assert fallback["final_recovery_phase"] == "COMPLETED"
