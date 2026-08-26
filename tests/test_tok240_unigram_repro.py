from __future__ import annotations

import json

import pytest

from twelve_six.tok240_unigram_repro import DECISION, Tok240Error, summarize_runs, validate_report


def _run(*, json_hash: str, vocab_hash: str, semantic_hash: str, probe: int) -> dict[str, object]:
    return {
        "artifact": {
            "algorithm": "unigram",
            "tokenizers_version": "0.23.1",
            "training_manifest_sha256": "1" * 64,
            "tokenizer_json_sha256": json_hash,
            "vocab_sha256": vocab_hash,
            "vocab_size": 497,
            "config_sha256": "2" * 64,
            "special_tokens": {"<unk>": 0},
        },
        "canonical_semantic_sha256": semantic_hash,
        "probe_ids": [[probe]],
        "strict_roundtrip_all": True,
        "unknown_tokens": 0,
        "serialization": {
            "same_object_to_str_exact": True,
            "serialize_reload_serialize_exact": False,
        },
        "training": {
            "train_sha256": "3" * 64,
            "training_manifest_sha256": "1" * 64,
            "ordered_texts_sha256": "4" * 64,
            "records": 10,
        },
        "runtime": {
            "python": "3.11.16",
            "tokenizers": "0.23.1",
            "pythonhashseed": "0",
            "tokenizers_parallelism": "false",
            "rayon_num_threads": "1",
        },
        "seed_probe": {"public_seed_argument_supported": False},
    }


def test_summary_marks_semantic_drift_ineligible() -> None:
    runs = [
        _run(json_hash="a" * 64, vocab_hash="b" * 64, semantic_hash="c" * 64, probe=1),
        _run(json_hash="d" * 64, vocab_hash="e" * 64, semantic_hash="f" * 64, probe=2),
    ]
    summary = summarize_runs(runs)
    assert summary["independent_runs"] == 2
    assert summary["byte_identical_tokenizer_json"] is False
    assert summary["ordered_token_id_vocabulary_equal"] is False
    assert summary["canonical_semantic_identity_equal"] is False
    assert summary["probe_encoding_equal"] is False
    assert summary["special_token_metadata_equal"] is True
    assert summary["eligible_under_reproducibility_contract"] is False


def test_summary_requires_two_independent_artifacts() -> None:
    with pytest.raises(Tok240Error, match="at least two"):
        summarize_runs([_run(json_hash="a" * 64, vocab_hash="b" * 64, semantic_hash="c" * 64, probe=1)])


def test_validator_rejects_model_comparison_reenable() -> None:
    report = {
        "schema": "12-6.tok240-unigram-reproducibility-final.v1",
        "source": {"source_sha": "1" * 40},
        "reproducibility": {
            "independent_runs": 2,
            "eligible_under_reproducibility_contract": False,
        },
        "decision": {
            "status": DECISION,
            "research_selection_eligible": False,
            "stop_model_comparisons": False,
            "tok241_may_compare_unigram": False,
        },
    }
    import hashlib

    body = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    report["evidence_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    with pytest.raises(Tok240Error, match="model comparisons must stop"):
        validate_report(report)
