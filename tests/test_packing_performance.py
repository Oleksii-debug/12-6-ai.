from __future__ import annotations

import torch

from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.packing import (
    TextRecord,
    collate_right_trimmed_rows,
    document_window_spans,
    iter_packed_examples,
    valid_causal_pairs,
)
from twelve_six.tokenization import ByteTokenizer


def test_document_spans_cover_each_valid_pair_exactly_once_and_restart_exactly() -> None:
    spans = document_window_spans("r", 13, sequence_length=5)
    assert [(s.source_start, s.source_end) for s in spans] == [(0, 5), (4, 9), (8, 13)]

    pairs = [pair for span in spans for pair in valid_causal_pairs(span)]
    assert pairs == [(i, i + 1) for i in range(12)]
    assert len(set(pairs)) == 12
    assert spans[0].source_offset_for_packed_position(4) == 4

    cursor = 1
    rebuilt = document_window_spans("r", 13, sequence_length=5)
    assert rebuilt[cursor:] == spans[cursor:]


def test_source_mapping_marks_only_right_padding_as_unmapped() -> None:
    span = document_window_spans("r", 7, sequence_length=5)[-1]
    assert (span.source_start, span.source_end, span.actual_length) == (4, 7, 3)
    assert [span.source_offset_for_packed_position(i) for i in range(5)] == [4, 5, 6, None, None]


def test_right_trim_preserves_incumbent_example_order_and_targets() -> None:
    tok = ByteTokenizer()
    records = (
        TextRecord("a", "abcdef", "train"),
        TextRecord("b", "xyz", "train"),
    )
    examples = tuple(
        iter_packed_examples(records, tok, expected_split="train", sequence_length=8)
    )
    assert len(examples) == 2

    trimmed = collate_right_trimmed_rows(examples, target_mode="labels")
    assert len(trimmed["input_ids"][0]) == 6
    assert trimmed["input_ids"][0] == examples[0].input_ids[:6]
    assert trimmed["input_ids"][1] == examples[1].input_ids[:6]
    assert trimmed["labels"][0] == examples[0].labels[:6]
    assert trimmed["labels"][1] == examples[1].labels[:6]
    assert sum(value != -100 for row in trimmed["labels"] for value in row[1:]) == 7


def test_right_trim_leaves_causal_prefix_logits_unchanged() -> None:
    torch.manual_seed(7)
    spec = ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=16,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        head_dim=8,
        d_ff=32,
        rope_rotary_dim=8,
    )
    model = TwelveSixDecoder(spec, InitSpec()).eval()
    tok = ByteTokenizer()
    examples = tuple(
        iter_packed_examples(
            (TextRecord("a", "abcdef", "train"), TextRecord("b", "xyz", "train")),
            tok,
            expected_split="train",
            sequence_length=16,
        )
    )
    fixed = torch.tensor([x.input_ids for x in examples], dtype=torch.long)
    rows = collate_right_trimmed_rows(examples)
    trimmed = torch.tensor(rows["input_ids"], dtype=torch.long)

    with torch.no_grad():
        fixed_logits = model(fixed).logits[:, : trimmed.shape[1], :]
        trimmed_logits = model(trimmed).logits
    torch.testing.assert_close(fixed_logits, trimmed_logits, rtol=0.0, atol=0.0)
