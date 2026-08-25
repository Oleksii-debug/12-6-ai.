from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest

from twelve_six.checkpoint import CheckpointIntegrityError
from twelve_six.inference.s0_artifact import (
    S0GenerationArtifactError,
    build_s0_generation_artifact,
    validate_s0_generation_artifact,
)


@pytest.fixture(scope="module")
def built_artifact(tmp_path_factory: pytest.TempPathFactory):
    root = Path(__file__).resolve().parents[1]
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    temp = tmp_path_factory.mktemp("s0-generation-artifact")
    checkpoint = temp / "checkpoint"
    evidence = build_s0_generation_artifact(
        root,
        candidate_sha=source_sha,
        checkpoint_out=checkpoint,
        train_steps=4,
        seed=20260824,
    )
    return checkpoint, evidence


def test_retained_artifact_matches_strict_candidate_and_exact_parity(built_artifact) -> None:
    checkpoint, evidence = built_artifact
    validate_s0_generation_artifact(evidence, checkpoint_path=checkpoint)

    assert evidence["checkpoint"]["matches_strict_d04_final_checkpoint_id"] is True
    assert evidence["parity"]["passed"] is True
    assert evidence["parity"]["atol"] == 0.0
    assert evidence["parity"]["rtol"] == 0.0
    assert evidence["parity"]["max_abs_error"] == 0.0
    assert evidence["parity"]["max_rel_error"] == 0.0
    assert evidence["generation"]["seeded_sampling_repeatable"] is True
    assert evidence["generation"]["token_stop_verified"] is True
    assert evidence["generation"]["context_limit_verified"] is True
    assert evidence["generation"]["over_context_rejected"] is True
    assert evidence["claims"]["promotion_authority"] is False


def test_evidence_mutation_fails_closed(built_artifact) -> None:
    _, evidence = built_artifact
    mutated = copy.deepcopy(evidence)
    mutated["claims"]["promotion_authority"] = True

    with pytest.raises(S0GenerationArtifactError, match="promotion authority"):
        validate_s0_generation_artifact(mutated)


def test_checkpoint_byte_corruption_fails_closed(built_artifact, tmp_path: Path) -> None:
    checkpoint, evidence = built_artifact
    corrupt = tmp_path / "corrupt-checkpoint"
    import shutil

    shutil.copytree(checkpoint, corrupt)
    weights = corrupt / "weights.safetensors"
    payload = bytearray(weights.read_bytes())
    payload[-1] ^= 0x01
    weights.write_bytes(bytes(payload))

    with pytest.raises(CheckpointIntegrityError):
        validate_s0_generation_artifact(evidence, checkpoint_path=corrupt)


def test_cross_identity_evidence_fails_closed(built_artifact) -> None:
    _, evidence = built_artifact
    mutated = copy.deepcopy(evidence)
    mutated["checkpoint"]["git_sha"] = "0" * 40

    with pytest.raises(S0GenerationArtifactError, match="checkpoint Git SHA mismatch"):
        validate_s0_generation_artifact(mutated)
