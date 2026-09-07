from pathlib import Path

from tools.validate_eval303_selection_validation_composite import EXPECTED, load_json, verify


def test_eval303_composite_verifies():
    result = verify(Path('.').resolve())
    assert result['status'] == 'PASS'
    assert result['selection_identity_sha256'] == EXPECTED['selection_identity_sha256']
    assert result['documents'] == 10


def test_selection_only_and_code_fail_closed():
    manifest = load_json(Path('configs/evaluation/eval303_selection_validation_composite_v1.json'))
    usage = manifest['usage_contract']
    assert usage['selection_only'] is True
    assert usage['may_train'] is False
    assert usage['may_fit_tokenizer'] is False
    assert usage['may_update_model'] is False
    assert usage['may_report_final_test'] is False
    assert manifest['strata']['code']['documents'] == 0
    assert manifest['strata']['code']['selection_eligible'] is False


def test_data300_exact_exclusion_is_hash_bound_not_family_washed():
    proof = load_json(Path('evidence/eval303/data300-exact-exclusion-proof-v1.json'))
    assert proof['comparisons']['selected_content_vs_training_raw_or_normalized_sha256_overlap'] == []
    assert proof['comparisons']['selected_git_blob_vs_training_git_blob_overlap'] == []
    assert proof['comparisons']['selected_source_family_vs_training_source_family_overlap'] == [
        'github:encode/httpx', 'github:psf/requests'
    ]
    assert len(proof['comparisons']['same_family_distinct_object_evidence']) == 2
    assert proof['verdict']['near_copy_or_dedup_cluster_scan_claimed'] is False


def test_final_test_outcomes_and_payload_not_consumed_by_eval303():
    manifest = load_json(Path('configs/evaluation/eval303_selection_validation_composite_v1.json'))
    firewall = manifest['final_test_firewall']
    assert firewall['outcomes_read_by_eval303'] is False
    assert firewall['final_test_payload_read_by_eval303'] is False
    assert firewall['final_test_bytes_copied_into_composite'] is False
