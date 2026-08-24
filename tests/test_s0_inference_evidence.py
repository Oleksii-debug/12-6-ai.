from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence

import pytest

from twelve_six.inference.evidence import (
    InferenceEvidenceError,
    collect_first_party_inference_evidence,
    validate_first_party_inference_evidence,
)


class _DeterministicByteBackend:
    eos_token_id = None
    max_context_tokens = 128

    def __init__(self, *, drift: bool = False) -> None:
        self._drift = drift

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: Sequence[int]) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")

    def next_token_logits(self, input_ids: Sequence[int]) -> list[float]:
        pivot = sum(input_ids) % 256
        logits = [-abs(index - pivot) / 17.0 for index in range(256)]
        if self._drift:
            logits[(pivot + 1) % 256] += 0.125
        return logits


class _CandidateBackend(_DeterministicByteBackend):
    def diagnostics(self) -> dict[str, object]:
        return {
            "backend": "first_party_torch",
            "checkpoint_id": "a" * 64,
            "git_sha": "b" * 40,
            "model_spec_sha256": "c" * 64,
            "parameter_count": 10_140,
            "vocab_size": 256,
            "max_context_tokens": 128,
            "tokenizer_version": "s0-byte-v1",
            "tokenizer_config_sha256": "d" * 64,
            "tokenizer_vocab_sha256": "e" * 64,
            "dataset_manifest_sha256": "f" * 64,
            "run_manifest_sha256": "1" * 64,
            "step": 40,
            "tokens_seen": 10_833,
            "device": "cpu",
        }


def _rehash(payload: dict[str, object]) -> None:
    material = copy.deepcopy(payload)
    material.pop("evidence_sha256", None)
    raw = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["evidence_sha256"] = hashlib.sha256(raw).hexdigest()


def test_evidence_records_exact_logits_tokens_decode_sampling_stops_and_context() -> None:
    evidence = collect_first_party_inference_evidence(
        _DeterministicByteBackend(),
        _CandidateBackend(),
        prompts=("12-6", "Base"),
        seed=17,
        max_new_tokens=4,
    )
    validate_first_party_inference_evidence(evidence)

    assert evidence["backend_diagnostics"]["checkpoint_id"] == "a" * 64
    assert len(evidence["prompts"]) == 2
    for report in evidence["prompts"]:
        assert report["parity_trace"]["all_logits_exact"] is True
        assert report["parity_trace"]["steps_compared"] > 0
        assert report["greedy"]["reference"] == report["greedy"]["candidate"]
        sampled = report["seeded_sampling"]
        assert sampled["reference"] == sampled["candidate_first"] == sampled["candidate_second"]
    assert evidence["stop_semantics"]["stop_token_result"]["stop_reason"] == "stop_token"
    assert evidence["stop_semantics"]["stop_string_result"]["stop_reason"] == "stop_string"
    assert evidence["context_semantics"]["at_limit_result"]["stop_reason"] == "context_limit"
    assert evidence["context_semantics"]["over_context_rejected"] is True


def test_collector_fails_on_exact_logit_drift() -> None:
    with pytest.raises(InferenceEvidenceError, match="logits diverged"):
        collect_first_party_inference_evidence(
            _DeterministicByteBackend(),
            _CandidateBackend(drift=True),
            prompts=("12-6",),
            max_new_tokens=2,
        )


def test_validator_rejects_byte_tamper_and_semantic_rewrite_even_when_rehashed() -> None:
    evidence = collect_first_party_inference_evidence(
        _DeterministicByteBackend(),
        _CandidateBackend(),
        prompts=("12-6",),
        max_new_tokens=2,
    )
    byte_tampered = copy.deepcopy(evidence)
    byte_tampered["backend_diagnostics"]["tokens_seen"] += 1
    with pytest.raises(InferenceEvidenceError, match="self-hash mismatch"):
        validate_first_party_inference_evidence(byte_tampered)

    semantic_tampered = copy.deepcopy(evidence)
    semantic_tampered["prompts"][0]["seeded_sampling"]["repeatable"] = False
    _rehash(semantic_tampered)
    with pytest.raises(InferenceEvidenceError, match="sampling proof is incomplete"):
        validate_first_party_inference_evidence(semantic_tampered)


def test_validator_rejects_truth_boundary_widening() -> None:
    evidence = collect_first_party_inference_evidence(
        _DeterministicByteBackend(),
        _CandidateBackend(),
        prompts=("12-6",),
        max_new_tokens=2,
    )
    evidence["claims"]["candidate_or_stable_promotion"] = True
    _rehash(evidence)
    with pytest.raises(InferenceEvidenceError, match="truth boundary"):
        validate_first_party_inference_evidence(evidence)
