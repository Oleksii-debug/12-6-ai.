from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from twelve_six.checkpoint import CheckpointIdentity, hash_json, save_checkpoint, sha256_file
from twelve_six.inference.acceptance import (
    ACCEPTANCE_SCHEMA,
    InferenceAcceptanceError,
    collect_backend_acceptance,
    collect_checkpoint_acceptance,
    validate_acceptance_report,
)
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]


class DeterministicBackend:
    eos_token_id = None
    max_context_tokens = 16

    def __init__(self, *, checkpoint_id: str = "1" * 64, logit_bias: float = 0.0) -> None:
        self.checkpoint_id = checkpoint_id
        self.logit_bias = logit_bias

    def encode(self, text: str) -> list[int]:
        return [0] * len(text.encode("utf-8"))

    def decode(self, token_ids: Sequence[int]) -> str:
        pieces = {0: "", 1: "A", 2: "B", 3: "C"}
        return "".join(pieces[token_id] for token_id in token_ids)

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        return [0.0, 4.0 + self.logit_bias, 2.0, 1.0]

    def diagnostics(self) -> dict[str, object]:
        return {
            "backend": "deterministic-test",
            "checkpoint_id": self.checkpoint_id,
            "git_sha": "2" * 40,
            "model_spec_sha256": "3" * 64,
            "parameter_count": 4,
            "vocab_size": 4,
            "max_context_tokens": self.max_context_tokens,
            "tokenizer_version": "test-byte-v1",
            "tokenizer_config_sha256": "4" * 64,
            "tokenizer_vocab_sha256": "5" * 64,
            "dataset_manifest_sha256": "6" * 64,
            "run_manifest_sha256": "7" * 64,
            "step": 3,
            "tokens_seen": 48,
            "device": "cpu",
        }


def test_acceptance_report_is_exact_privacy_safe_and_http_bound() -> None:
    prompt = "sensitive prompt"
    report = collect_backend_acceptance(
        DeterministicBackend(),
        DeterministicBackend(),
        (prompt, "12-6"),
        seed=17,
        max_new_tokens=2,
        model_name="s0-test",
    )

    assert report["schema"] == ACCEPTANCE_SCHEMA
    assert report["passed"] is True
    assert report["reload_parity"]["passed"] is True  # type: ignore[index]
    assert report["context"]["passed"] is True  # type: ignore[index]
    assert report["http"]["health_identity"] is True  # type: ignore[index]
    assert report["http"]["model_identity"] is True  # type: ignore[index]
    assert report["http"]["wrong_model_rejected"] is True  # type: ignore[index]
    assert report["http"]["chat_semantics_rejected"] is True  # type: ignore[index]
    assert report["http"]["oversized_context_rejected"] is True  # type: ignore[index]
    assert report["claims"]["promotion_authority"] is False  # type: ignore[index]
    assert report["claims"]["windows_nvda_live_execution"] is False  # type: ignore[index]

    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    assert prompt not in serialized
    assert "12-6" not in serialized
    validate_acceptance_report(report)


def test_acceptance_rejects_reload_identity_or_logit_divergence() -> None:
    with pytest.raises(InferenceAcceptanceError, match="identity mismatch"):
        collect_backend_acceptance(
            DeterministicBackend(),
            DeterministicBackend(checkpoint_id="8" * 64),
            ("probe",),
            max_new_tokens=2,
        )

    with pytest.raises(InferenceAcceptanceError, match="parity failed"):
        collect_backend_acceptance(
            DeterministicBackend(),
            DeterministicBackend(logit_bias=0.25),
            ("probe",),
            max_new_tokens=2,
        )


def test_acceptance_report_hash_rejects_tampering() -> None:
    report = collect_backend_acceptance(
        DeterministicBackend(),
        DeterministicBackend(),
        ("probe",),
        max_new_tokens=1,
    )
    tampered = copy.deepcopy(report)
    tampered["claims"]["promotion_authority"] = True  # type: ignore[index]
    with pytest.raises(InferenceAcceptanceError, match="SHA-256 mismatch"):
        validate_acceptance_report(tampered)


def test_real_first_party_checkpoint_is_loaded_twice_and_accepted(tmp_path: Path) -> None:
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    model = TwelveSixDecoder(stage.model, stage.init)
    checkpoint = tmp_path / "real-first-party-checkpoint"
    identity = CheckpointIdentity(
        git_sha="c" * 40,
        model_spec=stage.model.to_dict(),
        parameter_count=stage.expected_parameters,
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash=sha256_file(ROOT / "data/s0/packaged/manifest.json"),
        run_manifest_hash=hash_json(
            {
                "kind": "d05-d07-inference-acceptance-test",
                "canonical_base": "random_init_pretraining_only",
            }
        ),
        training_config={
            "training": {"context_length": stage.model.max_seq_len},
            "data": {"tokenizer_version": tokenizer.identity.version},
        },
        seed=17,
        precision="fp32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "test-no-optimizer-state"},
        scheduler=None,
        environment_lock_hash=sha256_file(ROOT / "requirements/locks/index.json"),
    )
    manifest = save_checkpoint(checkpoint, model=model, identity=identity)

    report = collect_checkpoint_acceptance(
        checkpoint,
        ("12-6",),
        seed=17,
        max_new_tokens=2,
        model_name="12-6-s0-test",
    )

    assert report["passed"] is True
    assert report["checkpoint"]["checkpoint_id"] == manifest["checkpoint_id"]  # type: ignore[index]
    assert report["checkpoint"]["git_sha"] == identity.git_sha  # type: ignore[index]
    assert report["checkpoint"]["model_spec_sha256"] == hash_json(stage.model.to_dict())  # type: ignore[index]
    assert report["checkpoint"]["tokenizer_config_sha256"] == tokenizer.identity.config_sha256  # type: ignore[index]
    assert report["reload_parity"]["max_abs_error"] == 0.0  # type: ignore[index]
    assert report["reload_parity"]["max_rel_error"] == 0.0  # type: ignore[index]
    validate_acceptance_report(report)
