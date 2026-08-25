from __future__ import annotations

from twelve_six.context_scaling import ContextPackingSpec, context_probe_spec
from twelve_six.model import load_stage_config
from twelve_six.packing import DEFAULT_SEQUENCE_LENGTH, PACKING_CONFIG_HASH, PACKING_VERSION
from twelve_six.tokenization import BYTE_TOKENIZER_HASH


def test_model17_contexts_change_identity_not_parameters() -> None:
    stage = load_stage_config("configs/stages/s1_100k.json")
    c128 = context_probe_spec(stage.model, max_seq_len=128)
    c256 = context_probe_spec(stage.model, max_seq_len=256)
    assert stage.model.max_seq_len == 256
    assert c128.parameter_count() == c256.parameter_count() == 107_856
    assert c128.identity_sha256() != c256.identity_sha256()
    assert c256.identity_sha256() == stage.model.identity_sha256()
    a = ContextPackingSpec(sequence_length=128)
    b = ContextPackingSpec(sequence_length=256)
    assert a.identity_sha256(tokenizer_config_sha256=BYTE_TOKENIZER_HASH) != b.identity_sha256(
        tokenizer_config_sha256=BYTE_TOKENIZER_HASH
    )


def test_model17_preserves_canonical_s0_context_identity() -> None:
    assert DEFAULT_SEQUENCE_LENGTH == 128
    assert PACKING_VERSION == "s0-byte-pack-v1"
    assert PACKING_CONFIG_HASH == "23a695b807f3e3f5c61d19c34968bcd88fafc6a45346dc08673d7a494219f285"
