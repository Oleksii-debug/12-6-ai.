from pathlib import Path

from twelve_six.milestone100_first_learned import (
    BATCH,
    MAX_STEPS,
    MIXTURE,
    SEQ,
    _model,
    _steps_by_stratum,
)
from twelve_six.packing import TextRecord, iter_packed_examples
from twelve_six.tokenization import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]


def test_byte_bound_s1_is_exact_95568_parameters():
    spec, init, provenance = _model(ROOT)
    assert spec.vocab_size == 256
    assert spec.parameter_count() == 95_568
    assert init.identity_sha256()
    assert provenance["only_geometry_change"] == (
        "vocab_size:512->256 to bind canonical s0-byte-v1"
    )


def test_1000_step_schedule_is_exact_45_35_20_by_batch():
    assert len(MIXTURE) == 20
    assert MAX_STEPS == 1000
    assert BATCH == 8
    assert _steps_by_stratum(MAX_STEPS) == {"uk": 450, "en": 350, "code": 200}


def test_incumbent_byte_packing_preserves_document_pairs():
    tokenizer = ByteTokenizer()
    records = [TextRecord("fixture", "abc", "train")]
    packed = list(
        iter_packed_examples(
            records,
            tokenizer,
            expected_split="train",
            sequence_length=SEQ,
            cross_document=False,
        )
    )
    assert len(packed) == 1
    assert packed[0].record_ids == ("fixture",)
    assert packed[0].num_loss_tokens == 2
