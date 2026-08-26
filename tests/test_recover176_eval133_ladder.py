from __future__ import annotations

from pathlib import Path

from twelve_six.eval_reservations import training_text_collisions
from twelve_six.milestone150_learned_base_ladder import SCALE_ORDER
from twelve_six.recover176_eval133_ladder import (
    EXPECTED_PARAMS,
    M150_SOURCE_SHA,
    _execution_environment,
    _role_steps,
    model_spec,
    validate_immutable_eval133,
)

ROOT = Path(__file__).resolve().parents[1]


def test_recovery_preserves_exact_eval133_blobs_and_suite_identity() -> None:
    identity = validate_immutable_eval133(ROOT)
    assert identity["suite_id"] == "eval133-en-raw-v1"
    assert identity["suite_version"] == "1.0.0"
    assert identity["items"] == 32
    assert len(identity["immutable_git_blobs"]) == 7


def test_recovery_targets_exact_m150_comparable_family() -> None:
    assert len(M150_SOURCE_SHA) == 40
    assert tuple(SCALE_ORDER) == ("100k", "500k", "1m")
    for scale in SCALE_ORDER:
        assert model_spec(scale).parameter_count() == EXPECTED_PARAMS[scale]


def test_random_best_final_role_selection_deduplicates_same_checkpoint() -> None:
    report = {"evaluation": {"best_step": 750}}
    assert _role_steps(report) == [
        (0, ["random_init"]),
        (750, ["best"]),
        (1000, ["final"]),
    ]
    report = {"evaluation": {"best_step": 1000}}
    assert _role_steps(report) == [
        (0, ["random_init"]),
        (1000, ["best", "final"]),
    ]


def test_reserved_scan_detects_suite_material_and_accepts_benign_text() -> None:
    leaked = "prefix The brass key beside the notes fits the lock. suffix"
    assert training_text_collisions(ROOT, [leaked])
    benign = (
        "Example 999: this English passage checks deterministic corpus identity "
        "and preserves audit provenance without recycling evaluation sentences."
    )
    assert training_text_collisions(ROOT, [benign]) == []


def test_full_locked_cpu_environment_includes_dev_test_lock() -> None:
    env = _execution_environment(ROOT)
    assert env["hash_locked"] is True
    assert env["tests_installed_from_dev_lock"] is True
    assert set(env["files"]) == {
        "requirements/locks/linux-x86_64/toolchain.lock.txt",
        "requirements/locks/linux-x86_64/runtime.lock.txt",
        "requirements/locks/linux-x86_64/dev.lock.txt",
    }
