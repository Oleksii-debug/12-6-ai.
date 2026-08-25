from __future__ import annotations

import copy

import pytest

from twelve_six.tokenization import unigram_repro as repro


def _run(*, artifact_suffix: str, vocab_suffix: str, probe_ids: list[list[int]]) -> dict[str, object]:
    return {
        "artifact": {
            "algorithm": "unigram",
            "tokenizers_version": "0.23.1",
            "training_manifest_sha256": "1" * 64,
            "tokenizer_json_sha256": artifact_suffix * 64,
            "vocab_sha256": vocab_suffix * 64,
            "vocab_size": 497,
            "config_sha256": artifact_suffix * 64,
            "special_tokens": {"<unk>": 0},
        },
        "internals": {
            "ordered_model_vocab_sha256": vocab_suffix * 64,
            "serialization_repeat_exact": True,
            "serialize_reload_serialize_exact": True,
        },
        "held_out": {
            "probe_ids": probe_ids,
            "strict_round_trip_all": True,
            "unknown_tokens": 0,
        },
    }


def _report() -> dict[str, object]:
    report: dict[str, object] = {
        "schema": repro.SCHEMA,
        "source": {
            "repository": "Oleksii-debug/12-6-ai.",
            "source_sha": "a" * 40,
            "observed_head_sha": "a" * 40,
        },
        "root_cause": {
            "supported_seed_control": False,
        },
        "semantic_equivalence": {
            "safe_for_checkpoint_identity": False,
            "ordered_vocab_drift": True,
            "held_out_encoding_drift": True,
        },
        "decision": {
            "status": repro.DECISION_REJECT,
            "canonical_use": "FAIL",
            "semantic_equivalence_identity_allowed": False,
            "canonical_s0_unchanged": True,
        },
    }
    _rehash(report)
    return report


def _rehash(report: dict[str, object]) -> None:
    report.pop("evidence_sha256", None)
    report["evidence_sha256"] = repro._sha256_text(repro._canonical_json(report))


def test_regime_summary_separates_training_drift_from_serialization() -> None:
    first = _run(artifact_suffix="1", vocab_suffix="2", probe_ids=[[1, 2], [3]])
    second = _run(artifact_suffix="3", vocab_suffix="4", probe_ids=[[5], [6, 7]])
    summary = repro._regime_summary([first, second])

    assert summary["exact_artifact_identity_equal"] is False
    assert summary["tokenizer_json_identity_equal"] is False
    assert summary["ordered_vocab_identity_equal"] is False
    assert summary["held_out_encoding_equal"] is False
    assert summary["serialization_repeat_exact_all"] is True
    assert summary["serialize_reload_serialize_exact_all"] is True
    assert summary["strict_round_trip_all"] is True
    assert summary["zero_unknown_tokens_all"] is True


def test_serial_child_environment_disables_parallelism_and_python_hash_randomness() -> None:
    env = repro._child_environment(serial=True)
    assert env["PYTHONHASHSEED"] == "0"
    assert env["TOKENIZERS_PARALLELISM"] == "false"
    assert env["RAYON_NUM_THREADS"] == "1"


def test_truthful_reject_report_validates() -> None:
    report = _report()
    repro.validate_report(report, expected_source_sha="a" * 40)


def test_validator_rejects_promoting_known_unigram_drift() -> None:
    report = _report()
    report["decision"]["canonical_use"] = "PASS"
    _rehash(report)
    with pytest.raises(repro.UnigramReproError, match="cannot be promoted"):
        repro.validate_report(report)


def test_validator_rejects_semantic_equivalence_override() -> None:
    report = _report()
    report["decision"]["semantic_equivalence_identity_allowed"] = True
    report["semantic_equivalence"]["safe_for_checkpoint_identity"] = True
    _rehash(report)
    with pytest.raises(repro.UnigramReproError, match="semantic-equivalence"):
        repro.validate_report(report)


def test_rehash_detects_tampering() -> None:
    report = _report()
    tampered = copy.deepcopy(report)
    tampered["decision"]["status"] = "PASS"
    with pytest.raises(repro.UnigramReproError, match="self-hash"):
        repro.validate_report(tampered)
