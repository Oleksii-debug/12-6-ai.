from __future__ import annotations

import copy
import json
from pathlib import Path

from twelve_six.d03_arxiv_execution_receipt import validate_d03_arxiv_receipt

REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = REPO_ROOT / "evidence" / "d03-arxiv-real-execution-v1.json"

COMPARE_MANIFEST = {
    "claim_boundary": {
        "authorized_optimized_target_exposure": 0,
        "canonical_capacity_credited": 0,
        "final_test_accessed": False,
        "foreign_pretrained_weights_used": False,
        "learned_weights_created": False,
        "optimizer_updates": 0,
        "paid_compute_used": False,
        "training_authorized_bytes": 0,
        "training_executed": False,
    },
    "deterministic_evidence": {
        "candidate_sha256": "21304338039306b2df175bbb71a1aed9a443c2416f3858f3cc63ca208e9c7327",
        "claim_boundary": {
            "authorized_unique_loss_positions": 0,
            "canonical_capacity_credited": 0,
            "family_credit_added": 0,
            "final_test_accessed": False,
            "foreign_pretrained_weights_used": False,
            "model_training_executed": False,
            "optimizer_updates": 0,
            "paid_compute_used": False,
            "tokenizer_fit_authorized": False,
            "training_authorized_bytes": 0,
        },
        "disposition_sha256": "e308ae28167c9fe92abc5de1c22f53852336877d677623ce6aab075b67e0ed17",
        "report_file_sha256": "1b94e8ea09f90f53fe86188e62e69f0469e96ef08d042c2fa1b88c69c24869c2",
        "report_identity_sha256": "4251b0b593500e3f91fd4c534873bbecd05992efca60d0d43d4e46c83edf6a2a",
        "retained_normalized_bytes": 1139552,
        "retained_records": 1024,
        "rows_scanned": 4096,
        "schema_version": "12-6.d03-arxiv-run-bound-evidence.v1",
        "source": {
            "bytes": 232616926,
            "dataset": "common-pile/arxiv_abstracts",
            "file": "00003_arxiv-abstracts.jsonl.gz",
            "revision": "46de78c48636c0b46f60049dfd1c5a3710d233f9",
            "sha256": "3d781bf7617fd9a6840211227c6c8831280dba457a6d9c79a2a5a357e025a8b7",
        },
    },
    "execution_a_artifact": {
        "digest": "6e3be902b1ea414ca6d916a2a1c027737408c64b1a6738a9cdc01dd9b98f2283",
        "evidence_sha256": "fce52b5adaa1504d601d447951720d52651817a979471b9954e96f0cbbfbcc95",
        "id": 10185121514,
    },
    "execution_b_artifact": {
        "digest": "3d1449f1b8355a2abc0e87921d0d32dbaa258099daacdd1b3957fdd1d11620c1",
        "evidence_sha256": "5435a33fcc9bd4e75ef29656e0ab05c033750ced041e6e0502cd7026dddafc18",
        "id": 10185121867,
    },
    "execution_head_sha": "37289b5af224c58f0b169f55f4e0f8b466b90f6d",
    "run_attempt": 1,
    "run_id": 34563405053,
    "schema_version": "12-6.d03-arxiv-run-bound-compare.v1",
}


def _receipt() -> dict:
    return json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))


def test_committed_receipt_matches_retained_compare_manifest() -> None:
    assert validate_d03_arxiv_receipt(_receipt(), compare_manifest=COMPARE_MANIFEST) == []


def test_rejects_artifact_identity_drift() -> None:
    receipt = _receipt()
    receipt["run_bound_artifacts"]["execution_a"]["id"] += 1
    errors = validate_d03_arxiv_receipt(receipt, compare_manifest=COMPARE_MANIFEST)
    assert "artifact_execution_a_id_mismatch" in errors


def test_rejects_bool_alias_for_zero_credit_integer() -> None:
    receipt = _receipt()
    receipt["claim_boundary"]["canonical_capacity_credited"] = False
    errors = validate_d03_arxiv_receipt(receipt, compare_manifest=COMPARE_MANIFEST)
    assert "canonical_capacity_credited_must_be_exact_int_zero" in errors


def test_rejects_source_identity_drift() -> None:
    receipt = _receipt()
    receipt["source"]["revision"] = "0" * 40
    errors = validate_d03_arxiv_receipt(receipt, compare_manifest=COMPARE_MANIFEST)
    assert "source_identity_mismatch" in errors


def test_rejects_output_identity_drift() -> None:
    receipt = _receipt()
    receipt["independent_execution_b"]["candidate_sha256"] = "0" * 64
    errors = validate_d03_arxiv_receipt(receipt, compare_manifest=COMPARE_MANIFEST)
    assert "independent_execution_b_mismatch" in errors


def test_rejects_compare_manifest_artifact_drift() -> None:
    manifest = copy.deepcopy(COMPARE_MANIFEST)
    manifest["execution_b_artifact"]["digest"] = "0" * 64
    errors = validate_d03_arxiv_receipt(_receipt(), compare_manifest=manifest)
    assert "compare_manifest_execution_b_artifact_digest_mismatch" in errors
    assert "compare_manifest_content_sha256_mismatch" in errors
