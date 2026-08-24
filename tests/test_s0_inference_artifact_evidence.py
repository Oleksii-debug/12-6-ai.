from __future__ import annotations

import copy
from pathlib import Path

import pytest

from twelve_six.checkpoint import CheckpointIntegrityError
from twelve_six.checkpoint.s0_inference_artifact_evidence import (
    S0InferenceArtifactEvidenceError,
    build_s0_inference_artifact_evidence,
    validate_s0_inference_artifact_evidence,
)


def test_real_trained_checkpoint_artifact_reload_generation_server_and_negatives(
    tmp_path: Path,
) -> None:
    source_sha = "a" * 40
    output_dir = tmp_path / "artifact"
    report = build_s0_inference_artifact_evidence(
        Path.cwd(),
        source_sha=source_sha,
        output_dir=output_dir,
        seed=17,
        max_steps=4,
        batch_size=3,
        verify_checkout=False,
    )

    validated = validate_s0_inference_artifact_evidence(
        report,
        checkpoint_dir=output_dir / "checkpoint",
        expected_source_sha=source_sha,
    )
    assert validated["checkpoint"]["verified_after_save"] is True
    assert validated["generation"]["seeded_sampling"]["repeat_exact"] is True
    assert validated["generation"]["token_stop"]["stop_reason"] == "stop_token"
    assert validated["server"]["completion_matches_greedy"] is True
    assert validated["server"]["chat_semantics_rejected"] is True
    assert validated["training"]["validation_optimized_tokens"] == 0

    with pytest.raises(S0InferenceArtifactEvidenceError, match="source SHA mismatch"):
        validate_s0_inference_artifact_evidence(
            report,
            expected_source_sha="b" * 40,
        )

    tampered = copy.deepcopy(report)
    tampered["claims"]["candidate_or_stable_promotion"] = True
    with pytest.raises(S0InferenceArtifactEvidenceError, match="self-hash mismatch"):
        validate_s0_inference_artifact_evidence(tampered)

    state_path = output_dir / "checkpoint" / "state.json"
    state_path.write_bytes(state_path.read_bytes() + b"\n")
    with pytest.raises(CheckpointIntegrityError):
        validate_s0_inference_artifact_evidence(
            report,
            checkpoint_dir=output_dir / "checkpoint",
            expected_source_sha=source_sha,
        )
