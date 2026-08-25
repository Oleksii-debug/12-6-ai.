from twelve_six.milestone100_one_m import (
    EXPECTED_PARAMETERS,
    RESEARCH41_HEAD,
    _one_m_model,
)


def test_milestone_uses_exact_research41_one_m_geometry():
    spec, init, provenance = _one_m_model(None)
    assert spec.parameter_count() == EXPECTED_PARAMETERS == 1_037_696
    assert spec.vocab_size == 256
    assert spec.max_seq_len == 256
    assert spec.d_model == 128
    assert spec.n_layers == 5
    assert spec.n_heads == 8
    assert spec.n_kv_heads == 8
    assert spec.head_dim == 16
    assert spec.d_ff == 352
    assert provenance["incumbent_head_sha"] == RESEARCH41_HEAD
    assert provenance["geometry_changes_by_milestone"] == "NONE"
    assert init.identity_sha256()
