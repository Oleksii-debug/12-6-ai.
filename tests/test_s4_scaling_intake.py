from twelve_six.scaling import DenseScalingTemplate, solve_dense_scaling_candidates


def test_current_byte_mha_template_solves_to_selected_s4_candidate() -> None:
    template = DenseScalingTemplate(
        vocab_size=256,
        max_seq_len=4096,
        d_model=768,
        n_layers=13,
        n_heads=12,
        n_kv_heads=12,
        head_dim=64,
        d_ff_multiple=128,
    )
    candidates = solve_dense_scaling_candidates(100_000_000, (template,))
    assert candidates
    best = candidates[0]
    assert best.spec.d_ff == 2304
    assert best.exact_parameters == 99_897_600
    assert best.model_identity_sha256 == (
        "6103d0d457e25206c11871f09aef1f2e23860329c060379c9f956b3851740170"
    )
