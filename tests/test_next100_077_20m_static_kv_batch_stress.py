from tools.qualify_next100_077_20m_static_kv_batch_stress import qualify


def test_next100_077_primary_20m_static_kv_batch_stress() -> None:
    report = qualify()

    assert report["worker"] == "NEXT100-077-20M-STATICKV-BATCH-STRESS"
    assert report["qualification"] == "PASS"
    assert report["pass"] is True
    assert report["execution"]["profile"] == "LOCAL_FREE"
    assert report["execution"]["device"] == "cpu"
    assert report["execution"]["cuda_build"] is None
    assert report["execution"]["cuda_available"] is False
    assert report["execution"]["training_performed"] is False
    assert report["execution"]["paid_compute"] is False

    authority = report["authority"]
    assert authority["model341_head"] == "e4ff486fd90802fc123bebf60eed4e59196a98df"
    assert (
        authority["next100_009_terminal_head"]
        == "7e3fc17aa204f647e4493861ce0817a3e7a19e98"
    )
    assert (
        authority["model_spec_sha256"]
        == "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
    )
    assert authority["static_kv_blob"] == "e4e8cf3746cbc7fc1e43f7c08b088b7df12e268b"
    assert authority["model_parameters"] == 20_613_440

    accounting = report["cache_accounting"]
    assert accounting["physical_bytes_by_batch"] == {
        "1": 8_388_608,
        "2": 16_777_216,
        "4": 33_554_432,
    }
    assert accounting["dynamic_growth_per_decode_step_bytes_by_batch"] == {
        "1": 8_192,
        "2": 16_384,
        "4": 32_768,
    }

    requirements = report["requirements"]
    assert requirements["batch_sizes_exercised"] == [1, 2, 4]
    for requirement in (
        "stateless_parity",
        "dynamic_parity",
        "storage_identity_stability",
        "partial_row_completion",
        "eos_per_row",
        "reset",
        "reuse",
        "overflow_isolation",
        "seeded_sampling_determinism",
        "zero_decode_backing_growth",
    ):
        assert requirements[requirement] is True

    scheduler = report["scheduler_boundary"]
    assert scheduler["continuous_batching_added"] is False
    assert scheduler["production_scheduler_modified"] is False

    for batch_size, expected_bytes in ((1, 8_388_608), (2, 16_777_216), (4, 33_554_432)):
        batch = report["batches"][str(batch_size)]
        assert batch["batch_size"] == batch_size
        assert batch["exact_physical_static_bytes"] == expected_bytes

        parity = batch["parity_and_backing"]
        assert parity["stateless_parity"] is True
        assert parity["dynamic_parity"] is True
        assert parity["max_abs_static_vs_dynamic"] <= 1e-6
        assert parity["max_abs_static_vs_stateless"] <= 1e-6
        assert parity["static_physical_bytes"] == expected_bytes
        assert parity["static_physical_bytes_after_decode"] == expected_bytes
        assert parity["static_backing_growth_bytes"] == 0
        assert parity["storage_identity_stable"] is True
        assert (
            parity["dynamic_growth_per_decode_step_bytes"]
            == 8_192 * batch_size
        )

        reset_reuse = batch["reset_reuse"]
        assert reset_reuse["reset"] is True
        assert reset_reuse["reuse"] is True
        assert reset_reuse["storage_identity_stable"] is True
        assert reset_reuse["physical_bytes"] == expected_bytes
        assert reset_reuse["physical_growth_bytes"] == 0
        assert reset_reuse["max_abs_static_vs_dynamic_after_reuse"] <= 1e-6
        assert reset_reuse["max_abs_static_vs_stateless_after_reuse"] <= 1e-6

        overflow = batch["overflow_isolation"]
        assert overflow["overflow_isolated"] is True
        assert overflow["valid_lengths_unchanged"] is True
        assert overflow["storage_identity_stable"] is True
        assert overflow["backing_bytes_unchanged"] is True
        assert overflow["backing_contents_unchanged"] is True

        sampling = batch["partial_completion_and_sampling"]
        assert sampling["partial_row_completion"] is True
        assert sampling["seeded_sampling_deterministic"] is True
        assert sampling["static_repeat_parity"] is True
        assert sampling["static_dynamic_generation_parity"] is True
        assert sampling["static_stateless_generation_parity"] is True
        assert sampling["peak_static_cache_bytes"] == expected_bytes
        if batch_size == 1:
            assert sampling["retired_row_decode_positions"] == 0
        else:
            assert sampling["retired_row_decode_positions"] > 0

        eos = batch["eos"]
        assert eos["eos_per_row"] is True
        assert eos["static_dynamic_stateless_parity"] is True
        assert eos["stop_reasons"][0] == "eos"
        if batch_size > 1:
            assert any(reason != "eos" for reason in eos["stop_reasons"][1:])

    claim = report["claim_boundary"]
    assert claim["zero_total_tensor_allocation_claimed"] is False
    assert claim["learned_quality_claimed"] is False
    assert claim["hardware_extrapolation_claimed"] is False
