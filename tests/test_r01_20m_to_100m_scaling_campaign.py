from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/research/r01_20m_to_100m_scaling_campaign_v1.json"
VALIDATOR_PATH = ROOT / "tools/validate_r01_20m_to_100m_scaling_campaign.py"

spec = importlib.util.spec_from_file_location("r01_scaling_validator", VALIDATOR_PATH)
assert spec is not None and spec.loader is not None
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def _load() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_campaign_is_valid() -> None:
    assert validator.validate_campaign(_load()) == []


def test_paid_compute_cannot_be_silently_authorized() -> None:
    data = _load()
    data["hard_boundaries"]["paid_compute_authorized"] = True
    errors = validator.validate_campaign(data)
    assert any("paid_compute_authorized" in error for error in errors)


def test_long_training_experiment_cannot_be_authorized_now() -> None:
    data = _load()
    experiment = next(item for item in data["experiment_matrix"] if item["id"] == "R01-E20")
    experiment["authorized_now"] = True
    errors = validator.validate_campaign(data)
    assert any("R01-E20" in error for error in errors)


def test_100m_modelspec_cannot_be_frozen_before_evidence() -> None:
    data = _load()
    experiment = next(item for item in data["experiment_matrix"] if item["id"] == "R01-E30")
    experiment["freeze_100m_modelspec_now"] = True
    errors = validator.validate_campaign(data)
    assert any("100M ModelSpec" in error for error in errors)


def test_model341_authority_drift_fails_closed() -> None:
    data = _load()
    data["authority"]["model341_sha"] = "0" * 40
    errors = validator.validate_campaign(data)
    assert any("authority.model341_sha" in error for error in errors)


def test_cross_tokenizer_metric_cannot_drop_bpb() -> None:
    data = _load()
    data["scientific_principles"]["cross_tokenizer_primary_metric"] = "perplexity"
    errors = validator.validate_campaign(data)
    assert any("cross-tokenizer primary metric" in error for error in errors)


def test_required_promotion_gate_cannot_be_removed() -> None:
    data = _load()
    data["promotion_gates"].remove("reserved_evaluation_decontamination")
    errors = validator.validate_campaign(data)
    assert any("promotion gate set" in error for error in errors)


def test_mutating_copy_does_not_change_control() -> None:
    original = _load()
    mutated = copy.deepcopy(original)
    mutated["baseline_model"]["n_layers"] = 17
    assert validator.validate_campaign(original) == []
    errors = validator.validate_campaign(mutated)
    assert any("baseline_model.n_layers" in error for error in errors)


@pytest.mark.parametrize(
    "raw",
    [
        '{"status":"a","status":"b"}',
        '{"outer":{"status":"a","status":"b"}}',
    ],
)
def test_strict_loader_rejects_duplicate_members(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "campaign.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match=r"^duplicate object member$"):
        validator._load_campaign(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_strict_loader_rejects_nonfinite_constants(
    tmp_path: Path,
    constant: str,
) -> None:
    path = tmp_path / "campaign.json"
    path.write_text('{"value":' + constant + "}", encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        validator._load_campaign(path)


@pytest.mark.parametrize("number", ["1e400", "-1e400"])
def test_strict_loader_rejects_float_overflow(tmp_path: Path, number: str) -> None:
    path = tmp_path / "campaign.json"
    path.write_text('{"value":' + number + "}", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON number is not finite"):
        validator._load_campaign(path)


def test_validator_rejects_oversized_campaign_before_decode(tmp_path: Path) -> None:
    path = tmp_path / "campaign.json"
    path.write_bytes(b" " * (validator.MAX_INPUT_BYTES + 1))
    with pytest.raises(ValueError, match="input byte limit"):
        validator._load_campaign(path)
    assert validator.main(["validate", str(path)]) == 2


@pytest.mark.parametrize("kind", ["array", "object"])
def test_validator_rejects_excessive_nesting_without_traceback(
    tmp_path: Path,
    kind: str,
) -> None:
    depth = 10_000
    raw = (
        "[" * depth + "0" + "]" * depth
        if kind == "array"
        else '{"item":' * depth + "0" + "}" * depth
    )
    path = tmp_path / "campaign.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="JSON nesting limit exceeded"):
        validator._load_campaign(path)
    assert validator.main(["validate", str(path)]) == 2


def test_validator_rejects_extra_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    assert validator.main(["validate", "one.json", "two.json"]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.strip() == (
        "FAIL: invalid arguments: expected at most one campaign path"
    )


def test_default_campaign_works_outside_repository_cwd(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout.strip().startswith("PASS: R01 20M -> 100M")
