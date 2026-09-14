from __future__ import annotations

from twelve_six.learned20_checkpoint_authority import validate_checkpoint_authority
from twelve_six.learned20_launch_gate import MODELSPEC_SHA256, PARAMETER_COUNT


def _bound_evidence() -> dict:
    return {
        "corpus": {
            "terminal": True,
            "identity": "corpus-authority-v1",
            "corpus_identity": "corpus-v1",
            "packing_identity": "packing-v1",
        },
        "tokenizer": {
            "terminal": True,
            "identity": "tokenizer-v1",
        },
        "checkpoint": {
            "terminal": True,
            "identity": "checkpoint-authority-v1",
            "source_sha": "a" * 40,
            "workflow_run_id": 12345,
            "workflow_conclusion": "success",
            "artifact_id": 67890,
            "artifact_digest": "sha256:" + "b" * 64,
            "modelspec_sha256": MODELSPEC_SHA256,
            "parameter_count": PARAMETER_COUNT,
            "corruption_matrix_passed": True,
            "corruption_matrix_cases_passed": 13,
            "corruption_matrix_failures": 0,
            "fresh_resume_equivalence": True,
            "fresh_process_reload": True,
            "model_state_exact": True,
            "trainer_state_exact": True,
            "optimizer_state_exact": True,
            "scheduler_state_exact": True,
            "scaler_state_exact": True,
            "counter_state_exact": True,
            "rng_state_exact": True,
            "dataloader_state_exact": True,
            "dataset_identity": "corpus-v1",
            "packing_identity": "packing-v1",
            "tokenizer_identity": "tokenizer-v1",
            "next_batch_identity": "next-batch-v1",
        },
    }


def test_terminal_checkpoint_requires_exact_provenance() -> None:
    evidence = _bound_evidence()
    checkpoint = evidence["checkpoint"]
    for key in ("source_sha", "workflow_run_id", "artifact_id", "artifact_digest"):
        original = checkpoint.pop(key)
        blockers = validate_checkpoint_authority(evidence)
        assert any(blocker.startswith(f"checkpoint.{key}") for blocker in blockers)
        checkpoint[key] = original


def test_terminal_checkpoint_requires_full_resume_state() -> None:
    evidence = _bound_evidence()
    checkpoint = evidence["checkpoint"]
    for key in (
        "fresh_process_reload",
        "optimizer_state_exact",
        "scheduler_state_exact",
        "scaler_state_exact",
        "counter_state_exact",
        "rng_state_exact",
        "dataloader_state_exact",
    ):
        checkpoint[key] = False
        assert f"checkpoint.{key}_not_proven" in validate_checkpoint_authority(evidence)
        checkpoint[key] = True


def test_terminal_checkpoint_rejects_cross_identity_mismatch() -> None:
    evidence = _bound_evidence()
    evidence["checkpoint"]["dataset_identity"] = "other-corpus"
    evidence["checkpoint"]["packing_identity"] = "other-packing"
    evidence["checkpoint"]["tokenizer_identity"] = "other-tokenizer"
    blockers = validate_checkpoint_authority(evidence)
    assert "checkpoint.dataset_identity_mismatch" in blockers
    assert "checkpoint.packing_identity_mismatch" in blockers
    assert "checkpoint.tokenizer_identity_mismatch" in blockers


def test_terminal_checkpoint_requires_exact_model341_binding() -> None:
    evidence = _bound_evidence()
    evidence["checkpoint"]["modelspec_sha256"] = "c" * 64
    evidence["checkpoint"]["parameter_count"] = PARAMETER_COUNT - 1
    blockers = validate_checkpoint_authority(evidence)
    assert "checkpoint.modelspec_sha256_mismatch" in blockers
    assert "checkpoint.parameter_count_mismatch" in blockers


def test_complete_checkpoint_authority_is_accepted() -> None:
    assert validate_checkpoint_authority(_bound_evidence()) == []


def test_nonterminal_checkpoint_is_not_promoted_or_reclassified_here() -> None:
    evidence = _bound_evidence()
    evidence["checkpoint"]["terminal"] = False
    assert validate_checkpoint_authority(evidence) == []
