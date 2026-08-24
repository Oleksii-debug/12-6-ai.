from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.inference.s0_evidence import (
    S0InferenceEvidenceError,
    collect_s0_trained_inference_evidence,
    validate_s0_trained_inference_evidence,
)


def test_real_trained_checkpoint_inference_evidence(tmp_path: Path) -> None:
    output_dir = tmp_path / "evidence"
    evidence = collect_s0_trained_inference_evidence(
        Path("."),
        source_sha="a" * 40,
        output_dir=output_dir,
        seed=1337,
        max_steps=4,
        batch_size=3,
        verify_checkout=False,
    )

    result = validate_s0_trained_inference_evidence(
        evidence,
        checkpoint=output_dir / "checkpoint",
    )
    assert result["status"] == "PASS"
    assert result["zero_tolerance_parity"] is True
    assert evidence["checkpoint"]["retained"] is True
    assert evidence["parity"]["prompts_compared"] == 3
    assert evidence["parity"]["max_abs_error"] == 0.0
    assert evidence["generation"]["seeded_sampling"]["repeatable"] is True
    assert (
        evidence["generation"]["stop_semantics"]["token_stop"]["stop_reason"]
        == "stop_token"
    )
    assert (
        evidence["generation"]["stop_semantics"]["text_stop"]["stop_reason"]
        == "stop_string"
    )
    assert (
        evidence["generation"]["context_semantics"]["exact_context_stop_reason"]
        == "context_limit"
    )
    assert evidence["cli"]["prompt_json"]["exit_code"] == 0
    assert evidence["cli"]["stdin_plain"]["exit_code"] == 0
    assert (
        evidence["openai_compatible_handoff"]["raw_base_completion_matches_canonical"]
        is True
    )

    persisted = json.loads(
        (output_dir / "s0-trained-inference-evidence.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted == evidence


def test_trained_inference_evidence_rejects_tamper(tmp_path: Path) -> None:
    output_dir = tmp_path / "evidence"
    evidence = collect_s0_trained_inference_evidence(
        Path("."),
        source_sha="b" * 40,
        output_dir=output_dir,
        max_steps=1,
        verify_checkout=False,
    )
    tampered = copy.deepcopy(evidence)
    tampered["parity"]["passed"] = False

    with pytest.raises(S0InferenceEvidenceError, match="self-hash"):
        validate_s0_trained_inference_evidence(tampered)
