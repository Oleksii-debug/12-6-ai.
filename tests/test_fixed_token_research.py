from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch

from twelve_six.fixed_token_research import (
    _aligned_batch,
    _steps_for_budgets,
    _validate_candidate,
    candidate_specs,
    config_payload,
    depth_width_specs,
    run_candidate,
    scaling_specs,
)


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


def test_depth_width_candidates_are_iso_parameter_and_identity_pinned() -> None:
    specs = depth_width_specs()
    assert list(specs) == [
        "wide_shallow_d64_l2",
        "balanced_d48_l3",
        "deeper_d40_l4",
        "deep_d32_l6",
        "very_deep_narrow_d28_l8",
    ]
    assert [spec.parameter_count() for spec in specs.values()] == [
        100_160,
        99_888,
        99_560,
        100_000,
        99_932,
    ]
    assert max(spec.parameter_count() for spec in specs.values()) / min(
        spec.parameter_count() for spec in specs.values()
    ) < 1.01
    assert {spec.vocab_size for spec in specs.values()} == {256}
    assert {spec.max_seq_len for spec in specs.values()} == {256}
    assert {spec.activation for spec in specs.values()} == {"swiglu"}
    assert {spec.norm_kind for spec in specs.values()} == {"rmsnorm"}
    assert {spec.position_embedding for spec in specs.values()} == {"rope"}


def test_research41_scaling_family_is_reused_exactly() -> None:
    specs = scaling_specs()
    assert [spec.parameter_count() for spec in specs.values()] == [
        95_568,
        267_912,
        467_808,
        1_037_696,
    ]
    assert candidate_specs("scaling") == specs


def test_aligned_batch_counts_only_requested_valid_causal_targets() -> None:
    raw = torch.arange(2 * 8, dtype=torch.long).reshape(2, 8)
    batch = _aligned_batch(raw, 9)
    assert batch["input_ids"].shape == (2, 8)
    assert int(batch["loss_mask"].sum().item()) == 9
    valid = batch["target_ids"].ne(-100) & batch["loss_mask"].bool()
    assert int(valid.sum().item()) == 9
    assert torch.equal(batch["target_ids"][:, :-1], raw[:, 1:])
    assert torch.all(batch["target_ids"][:, -1].eq(-100))
    assert torch.all(batch["loss_mask"][:, -1].eq(0))


def test_arbitrary_token_budgets_have_explicit_partial_steps() -> None:
    # 4 x (64-1) = 252 valid causal targets on a full RESEARCH41 batch.
    assert _steps_for_budgets((16_384, 65_536), 252) == 262
    # The exact runner must land on 16,384 and 65,536, never the old 16,632/65,772 overshoot.


def test_committed_depth_width_config_matches_generator() -> None:
    committed = json.loads(
        (_repo_root() / "configs/experiments/model08_depth_width_100k.v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert committed == config_payload()


def test_small_real_run_lands_exactly_and_fresh_resumes(tmp_path: Path) -> None:
    report = run_candidate(
        repo_root=_repo_root(),
        source_sha=_head(),
        family="scaling",
        candidate_id="scale_0095568",
        output_path=tmp_path / "candidate.json",
        checkpoint_dir=tmp_path / "checkpoints",
        token_budgets=(13, 31),
        batch_size=2,
        sequence_length=8,
        torch_threads=1,
        exercise_resume=True,
    )
    _validate_candidate(report, expected_source_sha=_head())
    assert [point["optimized_tokens"] for point in report["checkpoints"]] == [13, 31]
    assert [point["evaluation_optimized_tokens"] for point in report["checkpoints"]] == [0, 0]
    assert report["resume_exercised"] is True
    assert report["resume_events"][0]["fresh_objects"] is True
    assert report["trace_steps"] == 3
    assert report["final_validation_improvement"] == (
        report["initial_validation_loss"] - report["checkpoints"][-1]["validation_loss"]
    )
