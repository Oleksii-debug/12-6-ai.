from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "verify_learned_ladder_independent_v1.py"
SPEC = importlib.util.spec_from_file_location("learned_ladder_verify", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _contract() -> dict:
    return json.loads(
        (ROOT / "configs" / "eval" / "learned_ladder_independent_verify_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_contract_is_fail_closed_and_preserves_unmatched_budgets() -> None:
    contract = _contract()
    MODULE.validate_contract(contract)
    assert contract["comparison_boundary"]["matched_optimized_budget"] is False
    assert contract["comparison_boundary"]["direct_scale_ranking_authorized"] is False
    assert contract["comparison_boundary"]["three_m_actual_optimized_tokens"] == 131_938
    assert contract["comparison_boundary"]["ten_m_actual_optimized_tokens"] == 2_000_060


def test_contract_rejects_matched_budget_rewrite() -> None:
    contract = _contract()
    contract["comparison_boundary"]["matched_optimized_budget"] = True
    with pytest.raises(MODULE.VerificationError, match="matched optimized budget"):
        MODULE.validate_contract(contract)


def test_contract_rejects_wrong_verification_ids() -> None:
    contract = _contract()
    contract["verifications"][0]["verify_id"] = "VERIFY-FAKE"
    with pytest.raises(MODULE.VerificationError, match="verification IDs"):
        MODULE.validate_contract(contract)


def test_exact_value_helper_rejects_identity_drift() -> None:
    with pytest.raises(MODULE.VerificationError, match="source_sha"):
        MODULE._require_equal("drifted", "expected", "source_sha")


def test_true_false_helpers_fail_closed() -> None:
    with pytest.raises(MODULE.VerificationError):
        MODULE._require_true(False, "required pass")
    with pytest.raises(MODULE.VerificationError):
        MODULE._require_false(True, "forbidden authority")


def test_artifact_digest_drift_fails_before_evidence_is_trusted(tmp_path: Path) -> None:
    contract = _contract()["verifications"][0]
    artifact = tmp_path / "tampered.zip"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("unexpected.txt", "tampered")
    with pytest.raises(MODULE.VerificationError, match="artifact sha256"):
        MODULE.verify_learn191(artifact, contract)


def test_checkpoint_payload_hash_corruption_fails_closed(tmp_path: Path) -> None:
    archive_path = tmp_path / "checkpoint.zip"
    manifest = {
        "checkpoint_id": "checkpoint-id",
        "identity": {
            "git_sha": "source",
            "model_spec_hash": "model",
            "parameter_count": 7,
            "dataset_manifest_hash": "corpus",
            "tokenizer_hash": "tok-config",
            "tokenizer_vocab_hash": "tok-vocab",
            "tokens_seen": 11,
        },
        "files": {
            "state.json": {
                "bytes": len(b"expected"),
                "sha256": MODULE._sha256_bytes(b"expected"),
            }
        },
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("cp/manifest.json", json.dumps(manifest))
        archive.writestr("cp/state.json", b"corrupt!")
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(MODULE.VerificationError, match="sha256"):
            MODULE._verify_checkpoint_directory(
                archive,
                "cp/",
                expected_source_sha="source",
                expected_model_spec_sha256="model",
                expected_parameter_count=7,
                expected_corpus_sha256="corpus",
                expected_tokenizer_config_sha256="tok-config",
                expected_tokenizer_vocab_sha256="tok-vocab",
                expected_checkpoint_id="checkpoint-id",
                expected_tokens_seen=11,
            )


def test_contract_rejects_direct_scale_ranking_rewrite() -> None:
    contract = _contract()
    contract["comparison_boundary"]["direct_scale_ranking_authorized"] = True
    with pytest.raises(MODULE.VerificationError, match="direct scale ranking"):
        MODULE.validate_contract(contract)
