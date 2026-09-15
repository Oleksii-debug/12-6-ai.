from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.d03_languk_postexecution_rights import (
    CONFIG_PATH,
    LangUKPostExecutionRightsError,
    validate_languk_postexecution_rights,
)

ROOT = Path(__file__).resolve().parents[1]


def _base() -> dict:
    return json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _reject(tmp_path: Path, payload: dict) -> None:
    with pytest.raises(LangUKPostExecutionRightsError):
        validate_languk_postexecution_rights(ROOT, _write(tmp_path, payload))


def test_exact_current_authorities_validate() -> None:
    result = validate_languk_postexecution_rights(ROOT)
    assert result["status"] == "SOURCE_RIGHTS_POSTEXEC_ADMISSION_EXECUTED_ZERO_CREDIT"
    assert result["truth_boundary"]["source_rights_postexecution_admitted"] is True
    assert result["truth_boundary"]["source_admitted_candidate_bytes"] == 2_809_632
    assert result["truth_boundary"]["training_authorized_bytes"] == 0
    assert result["truth_boundary"]["authorized_optimized_target_exposure"] == 0
    assert result["truth_boundary"]["optimizer_updates"] == 0
    assert result["truth_boundary"]["learned_weights_created"] is False


def test_unknown_root_key_fails_closed(tmp_path: Path) -> None:
    payload = _base()
    payload["self_consistent_reseal"] = "forbidden"
    _reject(tmp_path, payload)


def test_rights_head_substitution_fails_closed(tmp_path: Path) -> None:
    payload = _base()
    payload["rights_authority"]["historical_head_sha"] = "0" * 40
    _reject(tmp_path, payload)


def test_rights_blob_reseal_fails_closed(tmp_path: Path) -> None:
    payload = _base()
    payload["rights_authority"]["manifest_blob_sha1"] = "1" * 40
    payload["rights_authority"]["evidence_identity_sha256"] = "2" * 64
    _reject(tmp_path, payload)


def test_execution_head_substitution_fails_closed(tmp_path: Path) -> None:
    payload = _base()
    payload["execution_authority"]["product_final_head_sha"] = "3" * 40
    payload["execution_authority"]["execution_head_sha"] = "4" * 40
    _reject(tmp_path, payload)


def test_source_revision_and_hash_substitution_fails_closed(tmp_path: Path) -> None:
    payload = _base()
    payload["source_scope"]["revision"] = "5" * 40
    payload["source_scope"]["sha256"] = "6" * 64
    _reject(tmp_path, payload)


def test_deanonymized_file_cannot_be_admitted(tmp_path: Path) -> None:
    payload = _base()
    payload["source_scope"]["file"] = payload["source_scope"]["excluded_file"]
    payload["proof"]["deanonymized_file_admitted"] = True
    _reject(tmp_path, payload)


def test_other_languk_families_cannot_be_widened(tmp_path: Path) -> None:
    payload = _base()
    payload["proof"]["other_languk_families_admitted"] = True
    _reject(tmp_path, payload)


def test_old_pre_repair_payload_cannot_replace_repaired_bytes(tmp_path: Path) -> None:
    payload = _base()
    payload["admitted_rights_candidate"]["retained_normalized_bytes"] = 2_813_635
    payload["truth_boundary"]["source_admitted_candidate_bytes"] = 2_813_635
    _reject(tmp_path, payload)


def test_repaired_candidate_hash_cannot_be_resealed(tmp_path: Path) -> None:
    payload = _base()
    payload["admitted_rights_candidate"]["retained_jsonl_sha256"] = "7" * 64
    payload["admitted_rights_candidate"]["report_identity_sha256"] = "8" * 64
    payload["admitted_rights_candidate"]["config_identity_sha256"] = "9" * 64
    _reject(tmp_path, payload)


def test_training_authority_cannot_be_widened(tmp_path: Path) -> None:
    for key, value in (
        ("canonical_capacity_credited", 1),
        ("training_authorized_bytes", 1),
        ("family_credit_added", 1),
        ("authorized_unique_loss_positions", 1),
        ("authorized_optimized_target_exposure", 1),
        ("optimizer_updates", 1),
    ):
        payload = _base()
        payload["truth_boundary"][key] = value
        _reject(tmp_path, payload)


def test_bool_int_aliases_fail_closed(tmp_path: Path) -> None:
    payload = _base()
    payload["truth_boundary"]["training_authorized_bytes"] = False
    _reject(tmp_path, payload)

    payload = _base()
    payload["truth_boundary"]["model_training_executed"] = 0
    _reject(tmp_path, payload)


def test_model_training_and_learned_weights_cannot_be_claimed(tmp_path: Path) -> None:
    payload = _base()
    payload["truth_boundary"]["model_training_executed"] = True
    payload["truth_boundary"]["learned_weights_created"] = True
    _reject(tmp_path, payload)


def test_final_test_and_paid_compute_cannot_be_claimed(tmp_path: Path) -> None:
    payload = _base()
    payload["truth_boundary"]["final_test_payload_accessed"] = True
    payload["truth_boundary"]["final_test_outcomes_read"] = True
    payload["truth_boundary"]["paid_compute_used"] = True
    _reject(tmp_path, payload)


def test_proof_cannot_be_rewritten_as_training_admission(tmp_path: Path) -> None:
    payload = _base()
    payload["proof"]["training_source_admitted"] = True
    _reject(tmp_path, payload)


def test_downstream_gates_cannot_be_dropped(tmp_path: Path) -> None:
    payload = _base()
    payload["downstream_required"] = payload["downstream_required"][:-1]
    _reject(tmp_path, payload)


def test_nested_unknown_key_fails_closed(tmp_path: Path) -> None:
    payload = copy.deepcopy(_base())
    payload["execution_authority"]["invented_authority"] = True
    _reject(tmp_path, payload)
