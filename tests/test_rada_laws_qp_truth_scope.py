from twelve_six.data.rada_laws_qp_dedup_adapter import _receipt


def test_receipt_scopes_external_llm_truth_to_upstream_source() -> None:
    receipt = _receipt(
        projection_identity_sha256="0" * 64,
        source_count=101_559,
        payload_bytes=192_078_166,
        duplicate_hashes=764,
        encoding_counts={"utf-8": 79_960, "windows-1251": 21_599},
    )
    truth = receipt["truth_boundary"]

    assert "external_llm_or_api_used_for_data_or_intelligence" not in truth
    assert truth["upstream_source_evidence_external_llm_or_api_used"] is False
    assert truth["current_corpus_external_llm_free_claimed_by_this_adapter"] is False

    assert truth["canonical_capacity_credited"] == 0
    assert truth["training_authorized_bytes"] == 0
    assert truth["authorized_optimized_target_exposure"] == 0
    assert truth["tokenizer_fit_authorized"] is False
    assert truth["optimizer_updates_executed_on_real_targets"] == 0
    assert truth["training_executed"] is False
    assert truth["learned_weights_created"] is False
    assert truth["final_test_outcomes_read"] is False
    assert truth["paid_compute_used"] is False
    assert truth["foreign_pretrained_weights"] is False
