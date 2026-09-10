from __future__ import annotations

import copy

import pytest

from twelve_six.tokenization.byte import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
)
from twelve_six.tokenization.decision_authority import (
    DECISION,
    STATUS,
    TokenizerDecisionError,
    authority_sha256,
    bind_byte_baseline_decision,
    verify_byte_baseline_decision,
)


def _authority(role: str, status: str) -> dict[str, object]:
    return {
        "schema": "test-authority.v1",
        "role": role,
        "status": status,
        "identity_seed": f"{role}-exact",
    }


def _bind() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    corpus = _authority("corpus", "TERMINAL_CORPUS")
    split = _authority("split", "TERMINAL_SPLIT")
    report = bind_byte_baseline_decision(
        corpus,
        split,
        expected_corpus_sha256=authority_sha256(corpus),
        expected_split_sha256=authority_sha256(split),
        corpus_terminal_status="TERMINAL_CORPUS",
        split_terminal_status="TERMINAL_SPLIT",
    )
    return corpus, split, report


def _verify(
    report: dict[str, object],
    corpus: dict[str, object],
    split: dict[str, object],
) -> None:
    verify_byte_baseline_decision(
        report,
        corpus,
        split,
        expected_corpus_sha256=authority_sha256(corpus),
        expected_split_sha256=authority_sha256(split),
        corpus_terminal_status="TERMINAL_CORPUS",
        split_terminal_status="TERMINAL_SPLIT",
    )


def test_terminal_decision_binds_canonical_byte_tokenizer_and_zero_authority() -> None:
    corpus, split, report = _bind()

    assert report["status"] == STATUS
    assert report["decision"] == DECISION
    assert report["tokenizer_version"] == BYTE_TOKENIZER_VERSION
    assert report["tokenizer_config_sha256"] == BYTE_TOKENIZER_HASH
    assert report["tokenizer_vocab_sha256"] == BYTE_VOCAB_HASH
    assert report["vocab_size"] == 256
    assert report["tokenizer_fit_executed"] is False
    assert report["training_authorized_by_this_report"] is False
    assert report["compute_authorized_by_this_report"] is False
    assert report["authorized_optimized_target_exposure"] == 0
    _verify(report, corpus, split)


def test_decision_is_deterministic_for_same_exact_upstreams() -> None:
    corpus, split, first = _bind()
    second = bind_byte_baseline_decision(
        corpus,
        split,
        expected_corpus_sha256=authority_sha256(corpus),
        expected_split_sha256=authority_sha256(split),
        corpus_terminal_status="TERMINAL_CORPUS",
        split_terminal_status="TERMINAL_SPLIT",
    )
    assert first == second


@pytest.mark.parametrize("role", ["corpus", "split"])
def test_substituted_upstream_cannot_teach_its_own_expected_identity(role: str) -> None:
    corpus = _authority("corpus", "TERMINAL_CORPUS")
    split = _authority("split", "TERMINAL_SPLIT")
    expected_corpus = authority_sha256(corpus)
    expected_split = authority_sha256(split)

    target = corpus if role == "corpus" else split
    target["identity_seed"] = "forged"

    with pytest.raises(TokenizerDecisionError, match="identity mismatch"):
        bind_byte_baseline_decision(
            corpus,
            split,
            expected_corpus_sha256=expected_corpus,
            expected_split_sha256=expected_split,
            corpus_terminal_status="TERMINAL_CORPUS",
            split_terminal_status="TERMINAL_SPLIT",
        )


@pytest.mark.parametrize(
    ("role", "replacement"),
    [("corpus", "RUNNING"), ("split", "BLOCKED")],
)
def test_nonterminal_upstream_fails_closed(role: str, replacement: str) -> None:
    corpus = _authority("corpus", "TERMINAL_CORPUS")
    split = _authority("split", "TERMINAL_SPLIT")
    target = corpus if role == "corpus" else split
    target["status"] = replacement

    with pytest.raises(TokenizerDecisionError, match="not terminal"):
        bind_byte_baseline_decision(
            corpus,
            split,
            expected_corpus_sha256=authority_sha256(corpus),
            expected_split_sha256=authority_sha256(split),
            corpus_terminal_status="TERMINAL_CORPUS",
            split_terminal_status="TERMINAL_SPLIT",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("decision", "FIT_BPE"),
        ("tokenizer_version", "forged"),
        ("tokenizer_config_sha256", "0" * 64),
        ("tokenizer_vocab_sha256", "1" * 64),
        ("vocab_size", 257),
        ("tokenizer_fit_executed", True),
        ("training_authorized_by_this_report", True),
        ("compute_authorized_by_this_report", True),
        ("authorized_optimized_target_exposure", 1),
        ("authorized_optimized_target_exposure", False),
    ],
)
def test_report_tamper_fails_closed(field: str, value: object) -> None:
    corpus, split, report = _bind()
    tampered = copy.deepcopy(report)
    tampered[field] = value

    with pytest.raises(TokenizerDecisionError):
        _verify(tampered, corpus, split)


def test_report_self_hash_tamper_fails_closed() -> None:
    corpus, split, report = _bind()
    report["decision_identity_sha256"] = "f" * 64

    with pytest.raises(TokenizerDecisionError, match="identity mismatch"):
        _verify(report, corpus, split)


def test_unknown_report_field_fails_closed_even_with_rehashed_identity() -> None:
    corpus, split, report = _bind()
    report["training_authorized"] = True
    core = {key: value for key, value in report.items() if key != "decision_identity_sha256"}
    report["decision_identity_sha256"] = authority_sha256(core)

    with pytest.raises(TokenizerDecisionError, match="closed-world"):
        _verify(report, corpus, split)


@pytest.mark.parametrize("bad_hash", ["A" * 64, "0" * 63, "g" * 64, True, None])
def test_expected_authority_hash_must_be_lowercase_sha256(bad_hash: object) -> None:
    corpus = _authority("corpus", "TERMINAL_CORPUS")
    split = _authority("split", "TERMINAL_SPLIT")

    with pytest.raises(TokenizerDecisionError, match="lowercase SHA-256"):
        bind_byte_baseline_decision(
            corpus,
            split,
            expected_corpus_sha256=bad_hash,  # type: ignore[arg-type]
            expected_split_sha256=authority_sha256(split),
            corpus_terminal_status="TERMINAL_CORPUS",
            split_terminal_status="TERMINAL_SPLIT",
        )
