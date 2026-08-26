from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "data_bulk_code1",
    ROOT / "tools" / "materialize_data_bulk_code1_permissive_python_bundle.py",
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)
CONFIG_PATH = ROOT / "configs/data/data_bulk_code1_permissive_python_bundle_v1.json"


def test_contract_is_exact_and_fail_closed() -> None:
    config = MOD.load_contract(CONFIG_PATH)
    assert config["contract_identity_sha256"] == MOD.EXPECTED_CONTRACT_IDENTITY
    assert len(config["sources"]) == 6
    assert len({source["family_id"] for source in config["sources"]}) == 6
    assert config["truth_boundary"]["authorized_training_exposure"] == 0
    assert config["rights_boundary"]["automatic_canonical_capacity_credit"] is False


def test_contract_rejects_training_or_capacity_promotion(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["truth_boundary"]["authorized_training_exposure"] = 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(MOD.MaterializationError):
        MOD.load_contract(path)


def test_eligible_python_file_records_exact_bytes_and_ast(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    root = repo / "src" / "pkg"
    root.mkdir(parents=True)
    path = root / "module.py"
    raw = b"def add(a, b):\n    return a + b\n"
    path.write_bytes(raw)
    config = MOD.load_contract(CONFIG_PATH)
    record, exclusion = MOD._eligible_file_record(repo, root, path, config["selection_policy"])
    assert exclusion is None
    assert record == {
        "path": "src/pkg/module.py",
        "sha256": MOD._sha256(raw),
        "utf8_bytes": len(raw),
    }


def test_credential_pattern_is_excluded(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    root = repo / "src" / "pkg"
    root.mkdir(parents=True)
    path = root / "secret.py"
    path.write_bytes(b"TOKEN = 'AKIAABCDEFGHIJKLMNOP'\n")
    config = MOD.load_contract(CONFIG_PATH)
    record, exclusion = MOD._eligible_file_record(repo, root, path, config["selection_policy"])
    assert record is None
    assert exclusion is not None
    assert exclusion["reason"] == "credential_pattern"
    assert "aws_access_key" in exclusion["patterns"]


def test_invalid_python_is_excluded(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    root = repo / "src" / "pkg"
    root.mkdir(parents=True)
    path = root / "broken.py"
    path.write_text("def broken(:\n", encoding="utf-8")
    config = MOD.load_contract(CONFIG_PATH)
    record, exclusion = MOD._eligible_file_record(repo, root, path, config["selection_policy"])
    assert record is None
    assert exclusion is not None
    assert exclusion["reason"] == "ast_parse_failure"


def test_excluded_directory_receives_no_credit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    root = repo / "src" / "pkg"
    path = root / "tests" / "test_example.py"
    path.parent.mkdir(parents=True)
    path.write_text("assert True\n", encoding="utf-8")
    config = MOD.load_contract(CONFIG_PATH)
    record, exclusion = MOD._eligible_file_record(repo, root, path, config["selection_policy"])
    assert record is None
    assert exclusion == {"path": "src/pkg/tests/test_example.py", "reason": "excluded_directory_component"}
