from __future__ import annotations

from twelve_six.milestone150_learned_base_ladder import (
    EXPECTED_CORPUS_ID,
    SCALE_ORDER,
    SCALE_SPECS,
    evaluation_identity,
    init_spec,
    model_spec,
)
from twelve_six.tokenization import ByteTokenizer


def test_milestone150_scale_family_is_exact_and_byte_native() -> None:
    expected = {
        "100k": (95_568, "4f1aaa6821360f0d22033356e011843646c8c14a6b4d20a3ad5b2ad125867470"),
        "500k": (467_808, "208ac8ca113388e76f280d0154cae815785bee7705546f4d854d9447b9dd1f4a"),
        "1m": (1_037_696, "ff3cee542a1f75bb4e1eff8d7d24d72533af8f4f3d82bd064fb1cbfeba8c8d07"),
    }
    assert SCALE_ORDER == ("100k", "500k", "1m")
    assert set(SCALE_SPECS) == set(expected)
    assert "10m" not in SCALE_SPECS
    for scale, (parameters, identity) in expected.items():
        spec = model_spec(scale)
        assert spec.vocab_size == 256
        assert spec.max_seq_len == 256
        assert spec.parameter_count() == parameters
        assert spec.identity_sha256() == identity


def test_milestone150_init_and_common_evaluation_identity_are_frozen() -> None:
    init = init_spec()
    assert init.identity_sha256() == "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"

    tok = ByteTokenizer()
    manifest = {"corpus_identity_sha256": EXPECTED_CORPUS_ID}
    first = evaluation_identity(tok, manifest)
    second = evaluation_identity(tok, manifest)
    assert first == second
    assert first["corpus_identity_sha256"] == EXPECTED_CORPUS_ID
    assert first["split"] == "validation"
    assert first["strata_order"] == ["uk", "en", "code"]
    assert first["tokenizer"]["version"] == "s0-byte-v1"
    assert first["tokenizer"]["vocab_size"] == 256
    assert first["tokenizer"]["special_tokens"] == {}
    assert first["packing"]["sequence_length"] == 128
    assert first["packing"]["cross_document"] is False
