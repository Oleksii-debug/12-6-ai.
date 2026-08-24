from __future__ import annotations

import copy
import json
from collections.abc import Sequence

import pytest

from twelve_six.inference.contracts import GenerationConfig
from twelve_six.inference.evidence import (
    _sha256,
    build_evidence,
    replay_evidence,
    validate_evidence,
)


class DeterministicBackend:
    eos_token_id = None
    max_context_tokens = 24

    def __init__(self, *, logit_bias: float = 0.0) -> None:
        self.logit_bias = logit_bias

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: Sequence[int]) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        pivot = (sum(input_ids) + len(input_ids)) % 256
        logits = [-20.0 - abs(index - pivot) for index in range(256)]
        logits[pivot] = 10.0 + self.logit_bias
        logits[(pivot + 1) % 256] = 9.0
        return logits

    def diagnostics(self) -> dict[str, object]:
        return {
            "backend": "first_party_torch",
            "checkpoint_id": "1" * 64,
            "git_sha": "2" * 40,
            "model_spec_sha256": "3" * 64,
            "parameter_count": 10140,
            "vocab_size": 256,
            "max_context_tokens": self.max_context_tokens,
            "tokenizer_version": "s0-byte-v1",
            "tokenizer_config_sha256": "4" * 64,
            "tokenizer_vocab_sha256": "5" * 64,
            "dataset_manifest_sha256": "6" * 64,
            "run_manifest_sha256": "7" * 64,
            "step": 40,
            "tokens_seen": 10833,
            "device": "cpu",
        }


def _probes() -> list[tuple[str, str, GenerationConfig]]:
    return [
        (
            "greedy",
            "12-6",
            GenerationConfig(max_new_tokens=4, sample=False, seed=11),
        ),
        (
            "sampled",
            "base",
            GenerationConfig(
                max_new_tokens=5,
                sample=True,
                temperature=0.8,
                top_k=8,
                top_p=0.95,
                seed=23,
            ),
        ),
    ]


def _rehash(payload: dict[str, object]) -> None:
    payload.pop("evidence_sha256", None)
    payload["evidence_sha256"] = _sha256(payload)


def test_evidence_is_deterministic_self_hashed_and_replayable() -> None:
    backend = DeterministicBackend()
    first = build_evidence(backend, _probes())
    second = build_evidence(backend, _probes())

    assert first == second
    assert validate_evidence(first) == first
    report = replay_evidence(first, backend)
    assert report["passed"] is True
    assert report["probes_replayed"] == 2
    assert report["checkpoint_id"] == "1" * 64

    rendered = json.dumps(first, ensure_ascii=False, sort_keys=True)
    assert '"12-6"' not in rendered
    assert '"base"' not in rendered
    assert first["claims"] == {
        "raw_base_completion_only": True,
        "instruction_or_chat_semantics": False,
        "cross_hardware_bitwise_reproducibility": False,
        "promotion_authority": False,
    }


def test_evidence_rejects_direct_tamper() -> None:
    payload = build_evidence(DeterministicBackend(), _probes())
    tampered = copy.deepcopy(payload)
    tampered["probes"][0]["generated_token_ids"][0] ^= 1

    with pytest.raises(ValueError, match="self-hash mismatch"):
        validate_evidence(tampered)


def test_evidence_rejects_rehashed_truth_boundary_weakening() -> None:
    payload = build_evidence(DeterministicBackend(), _probes())
    tampered = copy.deepcopy(payload)
    tampered["claims"]["promotion_authority"] = True
    _rehash(tampered)

    with pytest.raises(ValueError, match="truth-boundary"):
        validate_evidence(tampered)


def test_replay_rejects_checkpoint_identity_substitution() -> None:
    payload = build_evidence(DeterministicBackend(), _probes())
    tampered = copy.deepcopy(payload)
    tampered["checkpoint"]["checkpoint_id"] = "8" * 64
    _rehash(tampered)

    with pytest.raises(ValueError, match="backend identity"):
        replay_evidence(tampered, DeterministicBackend())


def test_replay_detects_logit_drift_with_same_declared_identity() -> None:
    payload = build_evidence(DeterministicBackend(), _probes())

    with pytest.raises(ValueError, match="replay diverged"):
        replay_evidence(payload, DeterministicBackend(logit_bias=-3.0))


def test_validate_rejects_unknown_generation_config_field_even_if_rehashed() -> None:
    payload = build_evidence(DeterministicBackend(), _probes())
    tampered = copy.deepcopy(payload)
    probe = tampered["probes"][0]
    probe["config"]["system_prompt"] = "forbidden"
    probe.pop("record_sha256")
    probe["record_sha256"] = _sha256(probe)
    _rehash(tampered)

    with pytest.raises(ValueError, match="missing or unknown fields"):
        validate_evidence(tampered)
