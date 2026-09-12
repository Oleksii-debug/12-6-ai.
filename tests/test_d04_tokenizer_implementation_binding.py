from __future__ import annotations

import pytest

from twelve_six.tokenization import decision_authority
from twelve_six.tokenization.byte import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
)

_UPSTREAMS = {
    "retained_inventory_identity_sha256": "1" * 64,
    "decontamination_authority_sha256": "2" * 64,
    "dedup_authority_sha256": "3" * 64,
    "balance_policy_identity_sha256": "4" * 64,
    "balance_result_identity_sha256": "5" * 64,
}


def _stub_upstreams(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        decision_authority,
        "_bind_upstreams",
        lambda *args, **kwargs: ("a" * 64, "b" * 64),
    )


def _bind_with_stub(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    _stub_upstreams(monkeypatch)
    return decision_authority.bind_byte_baseline_decision(
        _UPSTREAMS,
        {},
        expected_selection_identity_sha256="a" * 64,
        expected_application_identity_sha256="b" * 64,
        expected_retained_inventory_identity_sha256="1" * 64,
        expected_decontamination_authority_sha256="2" * 64,
        expected_dedup_authority_sha256="3" * 64,
        expected_balance_policy_identity_sha256="4" * 64,
        expected_balance_result_identity_sha256="5" * 64,
    )


def test_decision_binds_independent_byte_implementation_git_blob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _bind_with_stub(monkeypatch)

    assert report["canonical_byte_tokenizer_git_blob_sha1"] == (
        decision_authority.CANONICAL_BYTE_TOKENIZER_GIT_BLOB_SHA1
    )
    assert report["tokenizer_version"] == BYTE_TOKENIZER_VERSION
    assert report["tokenizer_config_sha256"] == BYTE_TOKENIZER_HASH
    assert report["tokenizer_vocab_sha256"] == BYTE_VOCAB_HASH


def test_encode_implementation_only_drift_fails_closed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_source = decision_authority._BYTE_TOKENIZER_SOURCE_PATH.read_bytes()
    needle = b"return list(text.encode(self.encoding))"
    assert needle in canonical_source

    drifted_source = canonical_source.replace(
        needle,
        b"return []  # implementation-only drift",
        1,
    )
    drifted_path = tmp_path / "byte.py"
    drifted_path.write_bytes(drifted_source)
    monkeypatch.setattr(decision_authority, "_BYTE_TOKENIZER_SOURCE_PATH", drifted_path)
    _stub_upstreams(monkeypatch)

    assert BYTE_TOKENIZER_VERSION == "s0-byte-v1"
    assert BYTE_TOKENIZER_HASH == "b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1"
    assert BYTE_VOCAB_HASH == "905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571"
    with pytest.raises(
        decision_authority.TokenizerDecisionError,
        match="implementation identity drift",
    ):
        decision_authority.bind_byte_baseline_decision(
            _UPSTREAMS,
            {},
            expected_selection_identity_sha256="a" * 64,
            expected_application_identity_sha256="b" * 64,
            expected_retained_inventory_identity_sha256="1" * 64,
            expected_decontamination_authority_sha256="2" * 64,
            expected_dedup_authority_sha256="3" * 64,
            expected_balance_policy_identity_sha256="4" * 64,
            expected_balance_result_identity_sha256="5" * 64,
        )
