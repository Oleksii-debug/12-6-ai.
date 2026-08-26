from __future__ import annotations

import json
from pathlib import Path

from twelve_six.checkpoint import hash_json
from twelve_six.data185_scientific_candidate import (
    AUTHORITY,
    DATA25_EXPECTED_ID,
    MIN_REAL_TRAIN_SHARE,
    RESEARCH140_HEAD,
    SCHEMA,
    SOURCE_FAMILY_MIN_BY_STRATUM,
    Data185Error,
    _blocked_report,
    _research140_single_pair,
    validate,
)


def test_scientific_thresholds_are_explicit_and_cross_modal() -> None:
    assert MIN_REAL_TRAIN_SHARE == 0.01
    assert SOURCE_FAMILY_MIN_BY_STRATUM == {"uk": 2, "en": 2, "code": 1}


def test_single_fixed_pair_uses_research140_fail_closed_semantics() -> None:
    result = _research140_single_pair(4.20, 4.10)
    assert result["methodology_upstream_head"] == RESEARCH140_HEAD
    assert result["metric"] == "common_data25_validation_bits_per_byte"
    assert result["paired_repeats"] == 1
    assert result["decision"] == "INSUFFICIENT_REPEATS"
    assert result["winner"] is None
    assert result["oriented_delta_baseline_minus_candidate_bpb"] > 0


def test_validate_accepts_only_bound_machine_report(tmp_path: Path) -> None:
    report = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {"git_sha": "a" * 40},
        "status": "RETEST_REQUIRED",
        "fixed_approx_1m_comparison": {
            "previous_dataset_identity_sha256": DATA25_EXPECTED_ID
        },
        "truth_boundary": {
            "local_free": True,
            "foreign_pretrained_weights": False,
        },
    }
    report["report_sha256"] = hash_json(report)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    checked = validate(path, "a" * 40)
    assert checked["status"] == "RETEST_REQUIRED"


def test_blocked_report_omits_unsupported_comparison(tmp_path: Path) -> None:
    blocked = _blocked_report("c" * 40, Data185Error("missing evidence"))
    assert blocked["status"] == "BLOCKED"
    assert "fixed_approx_1m_comparison" not in blocked
    assert "candidate" not in blocked
    path = tmp_path / "blocked.json"
    path.write_text(json.dumps(blocked), encoding="utf-8")
    checked = validate(path, "c" * 40)
    assert checked["status"] == "BLOCKED"


def test_validate_rejects_self_hash_drift(tmp_path: Path) -> None:
    report = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {"git_sha": "b" * 40},
        "status": "RETEST_REQUIRED",
        "fixed_approx_1m_comparison": {
            "previous_dataset_identity_sha256": DATA25_EXPECTED_ID
        },
        "truth_boundary": {
            "local_free": True,
            "foreign_pretrained_weights": False,
        },
        "report_sha256": "0" * 64,
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    try:
        validate(path, "b" * 40)
    except RuntimeError as exc:
        assert "self-hash" in str(exc)
    else:
        raise AssertionError("expected report self-hash validation failure")
