from twelve_six.milestone100_one_m import (
    EXPECTED_PARAMETERS,
    RESEARCH41_HEAD,
    _one_m_model,
    json_normalize,
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


def test_fresh_process_run_manifest_representation_is_json_stable():
    in_process = {
        "trainer_config": {"betas": (0.9, 0.95), "precision": "fp32"},
        "checkpoint_steps": (0, 250, 500, 750, 1000),
        "identity_sha256": "representation-only-sentinel",
    }
    normalized = json_normalize(in_process)
    assert normalized["trainer_config"]["betas"] == [0.9, 0.95]
    assert normalized["checkpoint_steps"] == [0, 250, 500, 750, 1000]
    assert normalized["identity_sha256"] == in_process["identity_sha256"]
    assert json_normalize(normalized) == normalized
