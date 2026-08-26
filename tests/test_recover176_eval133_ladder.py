from __future__ import annotations

from pathlib import Path

import pytest

from twelve_six.eval_reservations import training_text_collisions
from twelve_six.milestone150_learned_base_ladder import SCALE_ORDER
from twelve_six.recover176_eval133_ladder import (
    EXPECTED_PARAMS,
    M150_ARTIFACT_DIGEST,
    M150_ARTIFACT_ID,
    M150_CORPUS_ID,
    M150_EVALUATION_ID,
    M150_LADDER_REPORT_SHA256,
    M150_RUN_ID,
    M150_SOURCE_SHA,
    RecoveryError,
    _execution_environment,
    _scale_model_spec_sha,
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


def test_recovery_pins_terminal_frozen_m150_incumbent() -> None:
    assert M150_SOURCE_SHA == "5838cd16869dcfcf762368d8673eddf52d51b7e3"
    assert M150_RUN_ID == 32937411703
    assert M150_ARTIFACT_ID == 9595677772
    assert M150_ARTIFACT_DIGEST.startswith("sha256:")
    assert M150_LADDER_REPORT_SHA256 == "1f8350bed574a7b78778f0ebb7854ca5311173006820ec27110122f8965c9a5a"
    assert M150_CORPUS_ID == "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
    assert M150_EVALUATION_ID == "7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113"
    assert tuple(SCALE_ORDER) == ("100k", "500k", "1m")


def test_recovery_targets_exact_m150_comparable_family() -> None:
    for scale in SCALE_ORDER:
        assert model_spec(scale).parameter_count() == EXPECTED_PARAMS[scale]


def test_regression_m150_scale_report_uses_model_spec_sha_field() -> None:
    expected = model_spec("100k").identity_sha256()
    assert _scale_model_spec_sha({"model": {"spec_sha256": expected}}) == expected
    with pytest.raises(RecoveryError, match="model.spec_sha256 missing"):
        _scale_model_spec_sha({"model": {"model_spec_sha256": expected}})


def test_reserved_scan_detects_suite_material_and_accepts_benign_text() -> None:
    leaked = "prefix The brass key beside the notes fits the lock. suffix"
    assert training_text_collisions(ROOT, [leaked])
    benign = (
        "Example 999: this English passage checks deterministic corpus identity "
        "and preserves audit provenance without recycling evaluation sentences."
    )
    assert training_text_collisions(ROOT, [benign]) == []


def test_universal_locked_cpu_environment_includes_dev_test_lock() -> None:
    env = _execution_environment(ROOT)
    assert env["hash_locked"] is True
    assert env["tests_installed_from_dev_lock"] is True
    assert set(env["files"]) == {
        "requirements/locks/linux-x86_64/toolchain.lock.txt",
        "requirements/locks/linux-x86_64/runtime.lock.txt",
        "requirements/locks/linux-x86_64/dev.lock.txt",
    }
