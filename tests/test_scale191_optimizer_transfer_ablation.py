from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "scale191_optimizer_transfer_ablation.py"


def test_scale191_plan_is_full_factorial_and_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "plan", "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["schema"] == "12-6.scale191.optimizer-transfer.v1"
    assert plan["varied_only"] == ["learning_rate", "gradient_clip_norm"]
    assert plan["fixed_identity"]["parameter_count"] == 3_221_184
    assert plan["optimized_token_checkpoints"] == [16_632, 65_772, 131_292]
    assert plan["seeds"] == [1337, 1338]
    assert len(plan["trials"]) == 9
    pairs = {(row["lr_factor"], row["clip_factor"]) for row in plan["trials"]}
    assert pairs == {(lr, clip) for lr in (0.5, 1.0, 2.0) for clip in (0.5, 1.0, 2.0)}
    baselines = [row for row in plan["trials"] if row["is_scale190_baseline"]]
    assert len(baselines) == 1
    assert plan["truth_boundary"]["stage_promotion"] is False
    assert plan["truth_boundary"]["paid_compute"] is False


def _synthetic_result(lr: float, clip: float, seed: int, final_bpb: float) -> dict:
    is_baseline = lr == 1.0 and clip == 1.0
    expected = {
        1337: [3.63065595487007, 2.8779210625587925, 3.6269801849983665],
        1338: [3.6899818981527384, 2.9178113452297696, 4.0184479394803585],
    }
    if is_baseline:
        trajectory_bpb = expected[seed]
        final_bpb = trajectory_bpb[-1]
        parity = {"pass": True}
    else:
        trajectory_bpb = [3.5, 2.8, final_bpb]
        parity = None
    return {
        "trial": {
            "trial_id": f"lr-{lr:g}x_clip-{clip:g}x",
            "lr_factor": lr,
            "clip_factor": clip,
            "learning_rate": 3e-4 * lr,
            "gradient_clip_norm": clip,
            "is_scale190_baseline": is_baseline,
        },
        "seed": seed,
        "trajectory": [{"heldout_bpb": value} for value in trajectory_bpb],
        "late_delta_bpb_65772_to_131292": trajectory_bpb[-1] - trajectory_bpb[1],
        "final_clip_fraction": 0.5,
        "baseline_parity": parity,
    }


def test_scale191_aggregate_ranks_complete_two_seed_grid(tmp_path: Path) -> None:
    inputs = []
    for lr in (0.5, 1.0, 2.0):
        for clip in (0.5, 1.0, 2.0):
            for seed in (1337, 1338):
                final_bpb = 2.0 + abs(lr - 0.5) + abs(clip - 2.0)
                payload = _synthetic_result(lr, clip, seed, final_bpb)
                path = tmp_path / f"lr-{lr}-clip-{clip}-seed-{seed}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                inputs.append(path)
    output = tmp_path / "aggregate.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "aggregate",
            "--inputs",
            *[str(path) for path in inputs],
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    aggregate = json.loads(output.read_text(encoding="utf-8"))
    assert aggregate["baseline_parity_pass"] is True
    assert aggregate["winner"]["trial_id"] == "lr-0.5x_clip-2x"
    assert len(aggregate["ranking"]) == 9
    assert [row["rank"] for row in aggregate["ranking"]] == list(range(1, 10))
