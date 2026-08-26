from __future__ import annotations

from twelve_six import scale141_10m_continuation as core
from twelve_six.packing import TextRecord, iter_packed_examples
from twelve_six.scale141_10m_runtime_v2 import (
    BATCH,
    EVAL_TOKEN_TARGETS,
    MAX_OPTIMIZER_STEPS,
    MAX_TOKEN_OVERSHOOT,
    RESUME_TOKEN_TARGET,
    SEQ,
    TARGET_OPTIMIZED_TOKENS,
    _install_runtime_contract,
    _should_capture_update,
)
from twelve_six.tokenization import ByteTokenizer


def test_runtime_budget_is_actual_tokens_and_less_than_one_corpus() -> None:
    assert TARGET_OPTIMIZED_TOKENS == 2_000_000
    assert RESUME_TOKEN_TARGET == 1_000_000
    assert TARGET_OPTIMIZED_TOKENS < core.TRAIN_CORPUS_BYTES
    assert 0.099 < TARGET_OPTIMIZED_TOKENS / core.TRAIN_CORPUS_BYTES < 0.101
    assert EVAL_TOKEN_TARGETS == (0, 500_000, 1_000_000, 1_500_000, 2_000_000)
    assert MAX_OPTIMIZER_STEPS == 20_000


def test_document_isolated_padding_does_not_count_as_optimized_tokens() -> None:
    tok = ByteTokenizer()
    examples = list(
        iter_packed_examples(
            [TextRecord("r1", "abc", "train")],
            tok,
            expected_split="train",
            sequence_length=SEQ,
            cross_document=False,
        )
    )
    assert len(examples) == 1
    assert len(examples[0].input_ids) == SEQ
    assert examples[0].num_loss_tokens == 2
    assert examples[0].num_loss_tokens != SEQ - 1


def test_runtime_contract_uses_cpu_efficient_sequence_without_changing_model_max() -> None:
    _install_runtime_contract()
    assert core.SEQ == 256
    assert core.BATCH == BATCH == 1
    assert core.EXPECTED_OPTIMIZED_TOKENS == TARGET_OPTIMIZED_TOKENS
    assert core.EXPECTED_TOKENS_PER_STEP == 0
    spec, _, _ = core._model(__import__("pathlib").Path(__file__).resolve().parents[1])
    assert spec.max_seq_len == 1024
    assert SEQ <= spec.max_seq_len


def test_update_snapshot_is_requested_only_near_token_threshold() -> None:
    assert _should_capture_update(499_900, 500_000) is True
    assert _should_capture_update(499_700, 500_000) is False
    assert _should_capture_update(500_000, 500_000) is False
    assert _should_capture_update(0, None) is False
    assert MAX_TOKEN_OVERSHOOT == SEQ - 2
