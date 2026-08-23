from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.tokenization import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    ByteTokenizer,
    TokenizerCompatibilityError,
    canonical_config_json,
    require_tokenizer_identity,
    tokenizer_config_hash,
)


def test_frozen_raw_byte_token_id_semantics_match_s0_model_vocab() -> None:
    tokenizer = ByteTokenizer()
    assert BYTE_TOKENIZER_VERSION == "s0-byte-v1"
    assert tokenizer.special_tokens == {}
    assert tokenizer.pad_id is tokenizer.bos_id is tokenizer.eos_id is None
    assert tokenizer.encode("\x00A") == [0, 65]
    assert tokenizer.vocab_size == 256


def test_unicode_roundtrip_is_lossless_without_normalization() -> None:
    tokenizer = ByteTokenizer()
    samples = ["", "ASCII", "Україна", "e\u0301 != é", "🙂\n\t𐍈"]
    for text in samples:
        ids = tokenizer.encode(text)
        assert tokenizer.decode(ids) == text
        assert tokenizer.oov_count(text) == 0


def test_s0_byte_tokenizer_rejects_missing_special_token_requests() -> None:
    tokenizer = ByteTokenizer()
    with pytest.raises(ValueError, match="BOS"):
        tokenizer.encode("x", add_bos=True)
    with pytest.raises(ValueError, match="EOS"):
        tokenizer.encode("x", add_eos=True)


def test_invalid_ids_fail_closed() -> None:
    tokenizer = ByteTokenizer()
    with pytest.raises(ValueError):
        tokenizer.decode([256])
    with pytest.raises(TypeError):
        tokenizer.decode(["3"])  # type: ignore[list-item]


def test_identity_matches_repository_config() -> None:
    tokenizer = ByteTokenizer()
    config_path = Path(__file__).parents[1] / "configs" / "s0" / "tokenizer_byte_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repo_canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(repo_canonical.encode("utf-8")).hexdigest()

    assert canonical_config_json() == repo_canonical
    assert tokenizer_config_hash() == BYTE_TOKENIZER_HASH == digest
    assert BYTE_TOKENIZER_HASH == "b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1"
    assert tokenizer.identity.to_dict()["config_sha256"] == BYTE_TOKENIZER_HASH


def test_fertility_counts_utf8_bytes_per_code_point() -> None:
    tokenizer = ByteTokenizer()
    assert tokenizer.fertility("") == 0.0
    assert tokenizer.fertility("abc") == 1.0
    assert tokenizer.fertility("🙂") == 4.0


def test_checkpoint_identity_guard_fails_closed() -> None:
    tokenizer = ByteTokenizer()
    require_tokenizer_identity(
        tokenizer,
        expected_version=BYTE_TOKENIZER_VERSION,
        expected_config_sha256=BYTE_TOKENIZER_HASH,
        expected_vocab_size=256,
    )
    with pytest.raises(TokenizerCompatibilityError):
        require_tokenizer_identity(
            tokenizer,
            expected_version="other",
            expected_config_sha256=BYTE_TOKENIZER_HASH,
            expected_vocab_size=256,
        )
