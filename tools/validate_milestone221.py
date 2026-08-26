from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "evidence/milestone221/learned-base-ladder-v3.json"

EXPECTED_RANKING = ["1m", "500k", "100k"]
EXPECTED_PARAMS = {"100k": 95_568, "500k": 467_808, "1m": 1_037_696}
EXPECTED_BPB = {
    "100k": 1.8529853170496395,
    "500k": 0.2645455968814711,
    "1m": 0.12651757096387536,
}
M150_SOURCE = "5838cd16869dcfcf762368d8673eddf52d51b7e3"
RECOVER178_SOURCE = "fc4b3a1ed39216ee8e4cc938283ece2bd44f4d68"
LEARN191_SOURCE = "a75920cef8bde37a8c590e34095be83c97b75f1d"
DATA25 = "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
INITSPEC = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_record(record: dict) -> None:
    assert record["schema"] == "12-6.milestone221.learned-base-ladder-v3.v1"
    assert record["repository"] == "Oleksii-debug/12-6-ai."
    assert record["shared_m150_contract"]["producer_source_sha"] == M150_SOURCE
    assert record["shared_m150_contract"]["workflow_conclusion"] == "success"
    assert record["shared_m150_contract"]["corpus"]["identity_sha256"] == DATA25
    assert record["shared_m150_contract"]["optimized_tokens"] == 948_504
    assert record["shared_m150_contract"]["init_spec_sha256"] == INITSPEC

    rungs = record["rungs"]
    assert set(rungs) == set(EXPECTED_PARAMS)
    for scale, params in EXPECTED_PARAMS.items():
        rung = rungs[scale]
        assert rung["status"] == "ADMITTED_TERMINAL_LEARNED"
        assert rung["parameters"] == params
        assert rung["optimized_tokens"] == 948_504
        assert rung["corpus_identity_sha256"] == DATA25
        assert rung["init_spec_sha256"] == INITSPEC
        assert rung["fresh_reload_resume"]["passed"] is True
        assert rung["fresh_reload_resume"]["final_checkpoint_reload"] is True
        assert rung["first_party_logits_verified"] is True
        assert rung["evaluation_non_mutation"] is True
        assert rung["best_checkpoint"]["aggregate_bpb"] == EXPECTED_BPB[scale]
        assert rung["final_checkpoint"]["aggregate_bpb"] == EXPECTED_BPB[scale]
        assert rung["memorization"]["status"] == "TERMINAL_DEDICATED_AUTHORITY_AVAILABLE"
        assert rung["memorization"]["workflow_run_id"] == 32938943596

    ranking = record["directly_comparable_same_recipe_ranking"]
    assert [item["scale"] for item in ranking] == EXPECTED_RANKING
    assert all(item["optimized_tokens"] == 948_504 for item in ranking)
    assert [item["best_bpb"] for item in ranking] == sorted(
        item["best_bpb"] for item in ranking
    )

    mem = record["terminal_memorization_authority"]
    assert mem["source_sha"] == RECOVER178_SOURCE
    assert mem["workflow_run_id"] == 32938943596
    assert mem["conclusion"] == "success"
    assert set(mem["scope"]) == set(EXPECTED_PARAMS)

    evidence_3m = record["different_token_budget_learned_evidence"]["3m"]
    assert evidence_3m["producer_source_sha"] == LEARN191_SOURCE
    assert evidence_3m["workflow_run_id"] == 32940842372
    assert evidence_3m["workflow_conclusion"] == "success"
    assert evidence_3m["parameters"] == 3_213_120
    assert evidence_3m["admission_status"] == "NOT_ADMITTED_PENDING_VERIFY219"
    assert evidence_3m["preregistered_optimized_token_targets"] == [16_632, 65_772, 131_292]

    evidence_10m = record["different_token_budget_learned_evidence"]["10m"]
    assert evidence_10m["admission_status"].startswith("NOT_ADMITTED")
    assert evidence_10m["checkpoint211"]["full_10m_retraining_performed"] is False

    assert record["external_real_corpus_evidence"]["included"] is False
    assert all(value is False for value in record["prohibitions"].values())


def validate_m150(report: dict) -> None:
    assert report["source_sha"] == M150_SOURCE
    assert report["corpus"]["identity_sha256"] == DATA25
    scales = report["scales"]
    for scale, params in EXPECTED_PARAMS.items():
        item = scales[scale]
        assert item["parameter_count"] == params
        assert item["final_optimized_tokens"] == 948_504
        assert item["best_checkpoint"]["selection_validation"]["bits_per_byte"] == EXPECTED_BPB[scale]


def find_key(obj: object, key: str) -> list[object]:
    found: list[object] = []
    if isinstance(obj, dict):
        for k, value in obj.items():
            if k == key:
                found.append(value)
            found.extend(find_key(value, key))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(find_key(value, key))
    return found


def validate_recover178(report: dict) -> None:
    corpus_ids = find_key(report, "corpus_identity_sha256") + find_key(report, "corpus_identity")
    assert DATA25 in corpus_ids
    source_ids = find_key(report, "source_sha")
    assert RECOVER178_SOURCE in source_ids
    non_mutation = find_key(report, "evaluation_non_mutation")
    assert non_mutation and all(value is True for value in non_mutation if isinstance(value, bool))


def validate_learn191(report: dict) -> None:
    assert report["source_sha"] == LEARN191_SOURCE
    assert report["model"]["parameter_count"] == 3_213_120
    assert report["corpus_identity_sha256"] == DATA25
    assert report["evaluations"]
    assert report["fresh_process_resume"]
    assert report["best_checkpoint"]
    assert report["final_checkpoint"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m150-report", type=Path)
    parser.add_argument("--recover178-report", type=Path)
    parser.add_argument("--learn191-report", type=Path)
    args = parser.parse_args()

    record = load(RECORD)
    validate_record(record)
    if args.m150_report:
        validate_m150(load(args.m150_report))
    if args.recover178_report:
        validate_recover178(load(args.recover178_report))
    if args.learn191_report:
        validate_learn191(load(args.learn191_report))
    print("MILESTONE221_VALIDATION_PASS")


if __name__ == "__main__":
    main()
