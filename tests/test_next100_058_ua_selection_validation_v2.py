from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "validate_next100_058_ua_selection_validation_v2.py"
SPEC = importlib.util.spec_from_file_location("next100_058_validator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_next100_058_authority_validates_fail_closed() -> None:
    report = MODULE.validate(ROOT)
    assert report["status"] == MODULE.BLOCKED_STATUS
    assert report["preserved_eval290_records"] == 8
    assert report["preserved_eval290_families"] == [
        "kubernetes.website.docs",
        "lang-uk.perestoroha-ocr",
    ]
    assert report["new_records_admitted"] == 0
    assert report["new_families_admitted"] == 0
    assert report["exact_training_overlap_count"] == 0
    assert report["near_copy_training_cluster_proof_present"] is False
    assert report["final_test_payload_read"] is False
    assert report["final_test_outcomes_read"] is False
    assert report["training_executed"] is False


def test_authority_self_hash_is_immutable() -> None:
    config = json.loads((ROOT / MODULE.CONFIG_PATH).read_text(encoding="utf-8"))
    assert MODULE._authority_identity(config) == MODULE.EXPECTED_AUTHORITY_ID
    assert config["authority_identity_sha256"] == MODULE.EXPECTED_AUTHORITY_ID


def test_release_cannot_be_claimed_without_near_copy_proof() -> None:
    config = json.loads((ROOT / MODULE.CONFIG_PATH).read_text(encoding="utf-8"))
    assert config["training_exclusion_boundary"]["near_copy_cluster_proof"] == "ABSENT"
    assert config["status"] == MODULE.BLOCKED_STATUS
    exact_proof = json.loads((ROOT / MODULE.EXACT_PROOF_PATH).read_text(encoding="utf-8"))
    assert exact_proof["verdict"]["near_copy_or_dedup_cluster_scan_claimed"] is False
    assert exact_proof["verdict"]["wave3_data300_g07_g08_still_required"] is True


def test_late_bound_candidates_receive_no_illegal_credit() -> None:
    config = json.loads((ROOT / MODULE.CONFIG_PATH).read_text(encoding="utf-8"))
    candidates = config["late_bound_candidates"]
    assert {item["pull_request"] for item in candidates} == {446, 449, 455, 462}
    assert all(item["admitted_to_v2"] is False for item in candidates)
    assert all(item["new_family_credit"] == 0 for item in candidates)
    assert all(item["blockers"] for item in candidates)


def test_validator_rejects_weakening_training_firewall(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = json.loads((ROOT / MODULE.CONFIG_PATH).read_text(encoding="utf-8"))
    config["hard_requirements"]["future_training_prohibited"] = False
    target = tmp_path / MODULE.CONFIG_PATH
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.setattr(MODULE, "CONFIG_PATH", MODULE.CONFIG_PATH)
    with pytest.raises(MODULE.AuthorityError):
        MODULE.validate(tmp_path)
