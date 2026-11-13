import hashlib
import json
from pathlib import Path

import pytest

from tools import plan9_cyber_independent_eval_v1 as cyber_eval
from tools import plan9_cyber_triage_eval_v1 as triage
from tools import plan9_prepare_cyber_candidate_v1 as corpus
from tools.plan9_scale500m_launch_preparation_v1 import LaunchPreparationDenied, _audit_cyber_shards

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "plan9_cyber_candidate"


def test_committed_receipts_bind_exact_cyber_majority_and_eval_only():
    raw = (EVIDENCE / "corpus_receipt.json").read_bytes()
    candidate_sha = hashlib.sha256(raw).hexdigest()
    candidate = json.loads(raw)
    holdout = json.loads((EVIDENCE / "holdout_receipt.json").read_bytes())
    independent = json.loads((EVIDENCE / "attack_eval_receipt.json").read_bytes())
    exposure = json.loads((EVIDENCE / "exposure_receipt.json").read_bytes())
    assert candidate_sha == holdout["candidate_receipt_sha256"] == independent["candidate_receipt_sha256"]
    assert exposure["candidate_receipt_sha256"] == candidate_sha
    assert independent["verified_training_sha256"] == candidate["files"]["train.jsonl"]
    cyber = candidate["causal_targets"]["cyber"]
    total = sum(candidate["causal_targets"].values())
    assert 100 * cyber >= corpus.CYBER_PERCENT * total
    assert candidate["cyber_share_ppm"] == cyber * 1_000_000 // total
    assert exposure["targets"] == total and exposure["blocks"] == candidate["blocks"]
    assert exposure["raw_training_text_included"] is exposure["evaluation_payload_opened"] is False
    assert exposure["campaign_authorized"] is exposure["external_data_transfer_authorized"] is False
    assert candidate["public_redistribution_authorized"] is candidate["campaign_authorized"] is False
    assert independent["evaluation_only"] is True and independent["training_uses_evaluation"] is False


def test_gpu_inventory_checks_exact_shard_bytes_and_actual_masks(tmp_path):
    (tmp_path / "shards").mkdir()
    name = "shards/shard-000000.json"
    member = tmp_path / name

    def stage(mask):
        raw = corpus.canonical({
            "shard_index": 0, "block_size": 3,
            "blocks": [{"loss_mask": mask, "target_count": 2}],
        })
        member.write_bytes(raw)
        return {"block_size": 3, "shards": [{
            "path": name, "byte_count": len(raw), "sha256": corpus.sha(raw), "block_count": 1, "target_count": 2,
        }]}

    valid = stage([1, 1, 0])
    assert _audit_cyber_shards(tmp_path, valid) == (1, 1, 2)
    member.write_bytes(member.read_bytes().replace(b"[1,1,0]", b"[1,0,0]"))
    with pytest.raises(LaunchPreparationDenied, match="shard bytes changed"):
        _audit_cyber_shards(tmp_path, valid)
    bad_count = stage([1, 0, 0])
    with pytest.raises(LaunchPreparationDenied, match="block target count changed"):
        _audit_cyber_shards(tmp_path, bad_count)


def test_triage_development_cases_match_receipt_and_score_both_languages(tmp_path):
    output = tmp_path / "triage"
    assert triage.build(output) == json.loads((EVIDENCE / "triage_receipt.json").read_bytes())
    key = output / "answer-key.jsonl"
    rows = triage.read_rows(key.read_bytes())
    first = next(case_id for case_id, row in rows.items() if row["answer"]["priority"] == "P1")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_bytes(b"".join(triage.canonical({
        "id": case_id,
        "prediction": json.dumps({**row["answer"], "priority": "P2"} if case_id == first else row["answer"]),
    }) for case_id, row in rows.items()))
    result = triage.score(key, triage.sha(key.read_bytes()), predictions)
    assert result["metrics"]["exact"] == 39
    assert result["metrics"]["confirmed_p1_undertriage"] == 1
    assert result["paired_exact"] == 19


def test_mitigation_scorer_checks_ids_and_reports_macro_set_metrics(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_bytes(b"".join(map(cyber_eval.canonical, (
        {"id": "one", "category": "test", "expected_mitigation_ids": ["M1000", "M1001"]},
        {"id": "two", "category": "test", "expected_mitigation_ids": ["M2000"]},
    ))))
    expected = cyber_eval.digest(cases.read_bytes())
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_bytes(b'{"id":"one","prediction":"M1000 m1000 M1002"}\n{"id":"two","prediction":""}\n')
    result = cyber_eval.score(cases, predictions, expected)
    assert result["all"] == {"cases": 2, "precision": 0.25, "recall": 0.25, "f1": 0.25, "exact_match_rate": 0}
    assert result["predictions_sha256"] == cyber_eval.digest(predictions.read_bytes())
    predictions.write_bytes(b'{"id":"one","id":"two","prediction":"M2000"}\n')
    with pytest.raises(ValueError, match="duplicate JSON field"):
        cyber_eval.score(cases, predictions, expected)
    predictions.write_bytes(b'{"id":"one","prediction":"M1000"}\n')
    with pytest.raises(ValueError, match="exactly match benchmark case IDs"):
        cyber_eval.score(cases, predictions, expected)
