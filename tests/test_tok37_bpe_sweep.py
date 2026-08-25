from __future__ import annotations

from twelve_six.tokenization.bpe_sweep import (
    MODEL_PROBE_OPTIMIZED_TOKENS,
    MODEL_PROBE_STEPS,
    MODEL_PROBE_BATCH,
    MODEL_PROBE_SEQUENCE,
    REQUESTED_GRID,
    _corpus_contract,
    _parameter_tax,
    _purpose_environment_contract,
    _rebalance_100k,
)


def test_grid_keeps_256_as_fail_closed_control_and_adds_legal_floor() -> None:
    assert REQUESTED_GRID == (256, 257, 320, 384, 512, 768, 1024)


def test_current_corpus_truth_boundary_is_not_representative() -> None:
    corpus = _corpus_contract()
    assert corpus["sha256"] == "059f04e01d6fc6b8224b373b08efbb37f09d546de35ed510afdb4587ebdb6012"
    assert corpus["bytes"] == 1454
    assert corpus["records"] == 9
    assert corpus["representative_corpus"] is False


def test_purpose_environment_is_exact_d08_tokenizer_profile() -> None:
    environment = _purpose_environment_contract()
    assert environment["profile_id"] == "linux-x86_64-tokenizer-experiment"
    assert environment["profile_semantic_sha256"] == (
        "e368fa4c9fb2fc924482de32d5057837959111e958649663813cb46dddf6b5e4"
    )
    assert environment["overlay_sha256"] == (
        "11f27613ee7c15585796af39accde71b1e7c2791c24ff98d74c395262ee68544"
    )
    assert environment["tokenizers"] == "0.23.1"


def test_embedding_tax_uses_controlled_100k_to_1m_widths() -> None:
    tax = _parameter_tax(472)
    assert tax["100K"]["d_model"] == 48
    assert tax["250K"]["d_model"] == 72
    assert tax["500K"]["d_model"] == 96
    assert tax["1M"]["d_model"] == 128
    assert tax["100K"]["embedding_parameters"] == 472 * 48
    assert tax["1M"]["incremental_parameters_vs_byte_vocab"] == (472 - 256) * 128


def test_100k_probe_rebalances_ffn_without_changing_attention_geometry() -> None:
    spec = _rebalance_100k(472)
    assert spec.vocab_size == 472
    assert spec.d_model == 48
    assert spec.n_layers == 3
    assert spec.n_heads == 4
    assert spec.n_kv_heads == 4
    assert spec.d_ff % 8 == 0
    assert abs(spec.parameter_count() - 100_000) < 2_000


def test_model_probe_budget_is_exact() -> None:
    assert (
        MODEL_PROBE_STEPS * MODEL_PROBE_BATCH * (MODEL_PROBE_SEQUENCE - 1)
        == MODEL_PROBE_OPTIMIZED_TOKENS
        == 1_024
    )
