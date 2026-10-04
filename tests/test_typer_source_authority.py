from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.typer_source_authority import (
    CONFIG_PATH,
    RECEIPT_FILE_SHA256,
    validate_typer_source_authority,
    validate_typer_source_authority_files,
)

ROOT = Path(__file__).resolve().parents[1]


def _load() -> tuple[dict, bytes]:
    config = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    receipt = (ROOT / config["historical_execution"]["receipt_path"]).read_bytes()
    return config, receipt


def test_current_main_typer_source_authority_is_zero_credit_and_valid() -> None:
    config, receipt = _load()
    assert validate_typer_source_authority(config, receipt) == []
    assert validate_typer_source_authority_files(ROOT) == []
    assert config["current_composition"]["canonical_capacity_credit_bytes"] == 0
    assert config["truth_boundary"]["authorized_optimized_target_exposure"] == 0


def test_exact_historical_receipt_is_cryptographically_bound() -> None:
    config, receipt = _load()
    assert config["historical_execution"]["receipt_file_sha256"] == RECEIPT_FILE_SHA256
    tampered = receipt.replace(b'"verdict": "ADMIT"', b'"verdict": "REJECT"', 1)
    assert "historical_receipt_file_sha256_mismatch" in validate_typer_source_authority(
        config, tampered
    )


def test_source_identity_drift_fails_closed() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["bounded_source"]["raw_sha256"] = "0" * 64
    assert "bounded_source_identity_mismatch" in validate_typer_source_authority(
        mutated, receipt
    )


def test_lineage_exclusion_cannot_be_removed() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    del mutated["lineage_exclusions"]["typer/_click/**"]
    assert "lineage_exclusions_mismatch" in validate_typer_source_authority(
        mutated, receipt
    )


def test_historical_dedup_cannot_be_promoted_to_current_authority() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["historical_dedup"]["current_global_authority"] = True
    assert "historical_dedup_boundary_mismatch" in validate_typer_source_authority(
        mutated, receipt
    )


def test_current_global_dedup_cannot_be_fabricated_as_complete() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["current_composition"]["global_dedup"] = "PASS"
    assert "current_composition_global_dedup_mismatch" in validate_typer_source_authority(
        mutated, receipt
    )


def test_capacity_credit_cannot_be_created_by_source_authority() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["current_composition"]["canonical_capacity_credit_bytes"] = 7599
    blockers = validate_typer_source_authority(mutated, receipt)
    assert "current_composition_canonical_capacity_credit_bytes_must_be_exact_int_zero" in blockers


def test_evaluation_use_cannot_be_silently_admitted() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["license"]["evaluation"] = "ALLOWED"
    assert "license_identity_or_use_boundary_mismatch" in validate_typer_source_authority(
        mutated, receipt
    )


def test_truth_boundary_rejects_bool_masquerading_as_zero() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["truth_boundary"]["authorized_optimized_target_exposure"] = False
    blockers = validate_typer_source_authority(mutated, receipt)
    assert "truth_boundary_authorized_optimized_target_exposure_must_be_exact_int_zero" in blockers


def test_historical_success_cannot_remove_fresh_audit_requirement() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["project_authority"]["fresh_current_head_audit_required"] = False
    assert "project_authority_mismatch" in validate_typer_source_authority(mutated, receipt)


def test_receipt_lineage_metrics_are_bound_not_just_empty_failures() -> None:
    config, receipt = _load()
    payload = json.loads(receipt)
    payload["historical_dedup"]["click"]["max_skeleton_containment"] = 0.99
    forged = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    blockers = validate_typer_source_authority(config, forged)
    assert "historical_receipt_file_sha256_mismatch" in blockers


def test_receipt_scientific_boundary_is_zero_and_false() -> None:
    config, receipt = _load()
    payload = json.loads(receipt)
    payload["scientific_boundary"]["training_executed"] = True
    forged = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    blockers = validate_typer_source_authority(config, forged)
    assert "historical_receipt_file_sha256_mismatch" in blockers


def test_unknown_top_level_authority_is_rejected() -> None:
    config, receipt = _load()
    mutated = copy.deepcopy(config)
    mutated["legacy_registry_authority"] = True
    assert "config_top_level_keys_mismatch" in validate_typer_source_authority(
        mutated, receipt
    )


@pytest.mark.parametrize(
    ("path", "alias", "blocker"),
    [
        (("bounded_source", "repository_id"), 229937405.0, "bounded_source_identity_mismatch"),
        (("bounded_source", "size_bytes"), 7599.0, "bounded_source_identity_mismatch"),
        (("license", "size_bytes"), 1086.0, "license_identity_or_use_boundary_mismatch"),
        (("project_authority", "source_pr"), 476.0, "project_authority_mismatch"),
        (
            ("project_authority", "fresh_current_head_audit_required"),
            1, "project_authority_mismatch",
        ),
        (("historical_execution", "run_id"), 32999414943.0, "historical_run_id_mismatch"),
        (("historical_dedup", "historical_only"), 1, "historical_dedup_boundary_mismatch"),
        (
            ("historical_dedup", "near_duplicate_thresholds", "python_skeleton_5gram_jaccard"),
            "0.85", "historical_dedup_boundary_mismatch",
        ),
    ],
)
def test_nested_authority_numeric_and_boolean_aliases_fail_closed(
    path: tuple[str, ...], alias: object, blocker: str,
) -> None:
    config, receipt = _load()
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = alias
    assert blocker in validate_typer_source_authority(config, receipt)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"a":{"b":1,"b":2}}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":1e9999}',
        b'{"a":1e-9999}',
        br'{"\ud800":"invalid"}',
        b'{"a":"\xff"}',
        b" " * 1_048_577,
        b'{"a":' + b"[" * 65 + b"0" + b"]" * 65 + b"}",
    ],
)
def test_file_loader_rejects_untrusted_json_before_receipt_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: bytes,
) -> None:
    (tmp_path / CONFIG_PATH).parent.mkdir(parents=True)
    (tmp_path / CONFIG_PATH).write_bytes(raw)
    opened: list[Path] = []
    original = Path.open

    def track(path: Path, *args: object, **kwargs: object):
        opened.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track)
    assert validate_typer_source_authority_files(tmp_path) == ["config_json_invalid"]
    assert opened == [tmp_path / CONFIG_PATH]


@pytest.mark.parametrize(
    "attack_path",
    ["../../attack.json", "/tmp/attack.json", "C:\\attack.json"],
)
def test_receipt_path_never_selects_filesystem_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack_path: str,
) -> None:
    config, receipt = _load()
    config["historical_execution"]["receipt_path"] = attack_path
    (tmp_path / CONFIG_PATH).parent.mkdir(parents=True)
    (tmp_path / CONFIG_PATH).write_text(
        json.dumps(config, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / "evidence/data").mkdir(parents=True)
    expected = tmp_path / "evidence/data/next100_052_typer_terminal_source_receipt_v1.json"
    expected.write_bytes(receipt)

    opened: list[Path] = []
    original = Path.open

    def track(path: Path, *args: object, **kwargs: object):
        opened.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track)
    blockers = validate_typer_source_authority_files(tmp_path)
    assert "historical_receipt_path_mismatch" in blockers
    assert opened == [tmp_path / CONFIG_PATH, expected]


def test_malformed_historical_execution_cannot_control_receipt_path(
    tmp_path: Path,
) -> None:
    config, receipt = _load()
    config["historical_execution"] = []
    (tmp_path / CONFIG_PATH).parent.mkdir(parents=True)
    (tmp_path / CONFIG_PATH).write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "evidence/data").mkdir(parents=True)
    expected = tmp_path / "evidence/data/next100_052_typer_terminal_source_receipt_v1.json"
    expected.write_bytes(receipt)
    assert "historical_execution_missing" in validate_typer_source_authority_files(
        tmp_path
    )
