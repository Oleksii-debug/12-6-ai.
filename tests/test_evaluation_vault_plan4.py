from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.evaluation_vault import EvaluationBoundaryError, EvaluationVault


def setup(tmp_path: Path):
    vault = EvaluationVault(tmp_path / "eval", training_roots=(tmp_path / "train",))
    data = b'{"id":"a","answer":"yes"}\n{"id":"b","answer":"no"}\n'
    ref = vault.reserve(dataset=data, dataset_version="fixture-v1")
    identities = {"model_sha256": "1" * 64, "config_sha256": "2" * 64,
                  "evaluator_sha256": "3" * 64}
    return vault, ref, identities


def test_sealed_scoring_with_terminal_release_and_restart(tmp_path):
    vault, ref, identity = setup(tmp_path)
    receipt = vault.evaluate(dataset_ref=ref, predictions={"a": "yes", "b": "wrong"}, **identity)
    assert set(receipt) == {"schema_version", "evaluation_id", "state"}
    assert "yes" not in json.dumps(receipt) and "accuracy" not in receipt
    with pytest.raises(EvaluationBoundaryError, match="unverified"):
        vault.terminal_report(sealed=receipt, trusted_terminal_ids=frozenset())
    restarted = EvaluationVault(tmp_path / "eval", training_roots=(tmp_path / "train",))
    out = restarted.terminal_report(
        sealed=receipt, trusted_terminal_ids=frozenset({receipt["evaluation_id"]})
    )
    assert out["sample_count"] == 2 and out["accuracy"] == .5
    assert out["model_sha256"] == identity["model_sha256"]
    assert restarted.evaluate(dataset_ref=ref, predictions={"a": "yes", "b": "wrong"},
                              **identity) == receipt


@pytest.mark.parametrize("paths", [("train", "train/eval"), ("train/eval", "train"),
                                  ("train", "train")])
def test_training_evaluation_overlap_rejected(tmp_path, paths):
    with pytest.raises(EvaluationBoundaryError, match="overlap"):
        EvaluationVault(tmp_path / paths[1], training_roots=(tmp_path / paths[0],))


def test_symlink_escape_rejected(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(EvaluationBoundaryError, match="symlink"):
        EvaluationVault(tmp_path / "link" / "eval", training_roots=(tmp_path / "train",))


def test_preterminal_errors_do_not_reveal_answers(tmp_path):
    vault, ref, identity = setup(tmp_path)
    with pytest.raises(EvaluationBoundaryError, match="candidate ID mismatch") as exc:
        vault.evaluate(dataset_ref=ref, predictions={"a": "yes"}, **identity)
    assert "no" not in str(exc.value)
    with pytest.raises(EvaluationBoundaryError, match="identities"):
        vault.evaluate(dataset_ref=ref, predictions={"a": "yes", "b": "no"},
                       **{**identity, "model_sha256": "invalid"})


def test_corruption_and_untrusted_result_fail_closed(tmp_path):
    vault, ref, identity = setup(tmp_path)
    receipt = vault.evaluate(dataset_ref=ref, predictions={"a": "yes", "b": "no"}, **identity)
    file = vault.results / (receipt["evaluation_id"] + ".json")
    file.write_text('{"evaluation_id":"forged"}')
    with pytest.raises(EvaluationBoundaryError, match="verification failed"):
        vault.terminal_report(sealed=receipt,
                              trusted_terminal_ids=frozenset({receipt["evaluation_id"]}))
    with pytest.raises(EvaluationBoundaryError, match="immutable result mismatch"):
        vault.evaluate(dataset_ref=ref, predictions={"a": "yes", "b": "no"}, **identity)


def test_reserved_dataset_tamper_and_duplicate_ids_fail_closed(tmp_path):
    vault, ref, identity = setup(tmp_path)
    path = vault.reserved / (ref["dataset_sha256"] + ".jsonl")
    path.write_bytes(b"corrupted")
    with pytest.raises(EvaluationBoundaryError, match="digest mismatch"):
        vault.evaluate(dataset_ref=ref, predictions={"a": "yes", "b": "no"}, **identity)
    with pytest.raises(EvaluationBoundaryError, match="duplicate"):
        vault.reserve(dataset=b'{"id":"a","answer":"1"}\n{"id":"a","answer":"2"}\n',
                      dataset_version="test")

def test_version_and_result_digest_bindings_fail_closed(tmp_path):
    vault, ref, identity = setup(tmp_path)
    with pytest.raises(EvaluationBoundaryError, match="dataset version binding"):
        vault.evaluate(dataset_ref={**ref, "dataset_version": "forged"},
                       predictions={"a": "yes", "b": "no"}, **identity)
    with pytest.raises(EvaluationBoundaryError, match="immutable dataset version"):
        vault.reserve(dataset=b'{"id":"a","answer":"yes"}\n{"id":"b","answer":"no"}\n',
                      dataset_version="changed")
    sealed = vault.evaluate(dataset_ref=ref, predictions={"a": "yes", "b": "no"}, **identity)
    result = vault.results / (sealed["evaluation_id"] + ".json")
    payload = json.loads(result.read_text())
    payload["accuracy"] = 0.0
    result.write_text(json.dumps(payload))
    with pytest.raises(EvaluationBoundaryError, match="verification failed"):
        vault.terminal_report(sealed=sealed,
                              trusted_terminal_ids=frozenset({sealed["evaluation_id"]}))

def test_duplicate_json_object_fields_rejected(tmp_path):
    vault, _, _ = setup(tmp_path)
    with pytest.raises(EvaluationBoundaryError, match="duplicate JSON field"):
        vault.reserve(dataset=b'{"id":"x","answer":"one","answer":"two"}\n',
                      dataset_version="fixture-v2")
