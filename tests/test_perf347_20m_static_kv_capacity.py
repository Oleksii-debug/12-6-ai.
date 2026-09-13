from tools.qualify_perf347_20m_static_kv import qualify


def test_perf347_primary_20m_static_kv_capacity() -> None:
    report = qualify()

    assert report["qualification"] == "PASS"
    assert report["pass"] is True
    assert report["execution"]["device"] == "cpu"
    assert report["execution"]["cuda_build"] is None
    assert report["execution"]["cuda_available"] is False
    assert report["execution"]["long_training_performed"] is False

    assert report["authority"]["primary_model_worker"] == "MODEL-341-20M-CANDIDATE-A"
    assert (
        report["authority"]["model_spec_sha256"]
        == "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
    )
    assert (
        report["authority"]["static_kv_blob"]
        == "e4e8cf3746cbc7fc1e43f7c08b088b7df12e268b"
    )
    assert report["model"]["parameters"] == 20_613_440

    accounting = report["cache_accounting"]
    assert accounting["batch1_full_static_bytes"] == 8_388_608
    assert accounting["batch2_full_static_bytes"] == 16_777_216
    assert accounting["batch1_dynamic_growth_per_token_bytes"] == 8_192
    assert accounting["batch2_dynamic_growth_per_token_bytes"] == 16_384

    direct = report["checks"]["stateless_dynamic_static_parity_and_capacity"]
    assert direct["max_abs_static_vs_dynamic"] <= 1e-6
    assert direct["max_abs_static_vs_stateless"] <= 1e-6
    assert direct["static_physical_growth_bytes"] == 0
    assert direct["dynamic_growth_bytes"] == 131_072
    assert direct["storage_stable"] is True

    generation = report["checks"]["generation_greedy_sampling_stop_eos"]
    assert generation["greedy_parity"] is True
    assert generation["sampling_parity"] is True
    assert generation["stop_token_parity"] is True
    assert generation["eos_parity"] is True

    batching = report["checks"]["batching"]
    assert batching["max_abs_static_vs_dynamic"] <= 1e-6
    assert batching["static_physical_growth_bytes"] == 0
    assert batching["dynamic_growth_bytes"] == 65_536
    assert batching["storage_stable"] is True
    assert batching["partial_completion_generation_parity"] is True
    assert batching["sampled_batch_row_parity"] is True

    reset = report["checks"]["reset_reuse"]
    assert reset["storage_stable"] is True
    assert reset["physical_growth_bytes"] == 0
    assert reset["reset_sequence_length"] == 2

    context = report["checks"]["maximum_context"]
    assert context["maximum_context_tokens"] == 1024
    assert context["final_legal_token_parity"] is True
    assert context["max_abs_static_vs_dynamic"] <= 1e-6
    assert context["max_abs_static_vs_stateless"] <= 1e-6
    assert context["static_physical_growth_bytes"] == 0
    assert context["overflow_rejected_before_static_length_mutation"] is True
