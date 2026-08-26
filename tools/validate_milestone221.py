from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "evidence/milestone221/learned-base-ladder-v3.json"
EXPECTED_RANKING = ["1m", "500k", "100k"]
EXPECTED_PARAMS = {"100k": 95_568, "500k": 467_808, "1m": 1_037_696}
EXPECTED_BPB = {"100k": 1.8529853170496395, "500k": 0.2645455968814711, "1m": 0.12651757096387536}
M150_SOURCE = "5838cd16869dcfcf762368d8673eddf52d51b7e3"
RECOVER178_SOURCE = "fc4b3a1ed39216ee8e4cc938283ece2bd44f4d68"
LEARN191_SOURCE = "a75920cef8bde37a8c590e34095be83c97b75f1d"
DATA25 = "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
INITSPEC = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_record(record: dict) -> None:
    assert record["schema"] == "12-6.milestone221.learned-base-ladder-v3.v1"
    common = record["shared_m150_contract"]
    assert common["producer_source_sha"] == M150_SOURCE
    assert common["workflow_conclusion"] == "success"
    assert common["corpus"]["identity_sha256"] == DATA25
    assert common["optimized_tokens"] == 948_504
    assert common["init_spec_sha256"] == INITSPEC
    assert set(record["rungs"]) == set(EXPECTED_PARAMS)
    for scale, params in EXPECTED_PARAMS.items():
        rung = record["rungs"][scale]
        assert rung["status"] == "ADMITTED_TERMINAL_LEARNED"
        assert rung["parameters"] == params
        assert rung["optimized_tokens"] == 948_504
        assert rung["corpus_identity_sha256"] == DATA25
        assert rung["init_spec_sha256"] == INITSPEC
        assert rung["fresh_reload_resume"]["passed"] is True
        assert rung["fresh_reload_resume"]["final_checkpoint_reload"] is True
        assert rung["best_checkpoint"]["aggregate_bpb"] == EXPECTED_BPB[scale]
        assert rung["final_checkpoint"]["aggregate_bpb"] == EXPECTED_BPB[scale]
        assert rung["first_party_logits_verified"] is True
        assert rung["evaluation_non_mutation"] is True
        assert rung["memorization"]["authority"] == "RECOVER-178"
        assert rung["memorization"]["diagnostic_stop"] is True
        assert rung["memorization"]["privacy_claim"] == "NONE"
    ranking = record["directly_comparable_same_recipe_ranking"]
    assert [x["scale"] for x in ranking] == EXPECTED_RANKING
    assert all(x["optimized_tokens"] == 948_504 for x in ranking)
    assert [x["best_bpb"] for x in ranking] == sorted(x["best_bpb"] for x in ranking)
    mem = record["terminal_memorization_authority"]
    assert mem["source_sha"] == RECOVER178_SOURCE and mem["conclusion"] == "success"
    three = record["different_token_budget_learned_evidence"]["3m"]
    assert three["admission_status"] == "NOT_ADMITTED_PENDING_VERIFY219"
    assert three["producer_source_sha"] == LEARN191_SOURCE
    assert three["workflow_conclusion"] == "success"
    assert three["parameters"] == 3_213_120
    assert three["actual_checkpoints"][-1]["actual"] == 131_938
    assert three["best_checkpoint"]["id"] == three["final_checkpoint"]["id"]
    assert three["best_checkpoint"]["aggregate_bpb"] == 2.2859499700392583
    assert three["fresh_reload_resume"]["final_fresh_load_passed"] is True
    assert three["first_party_logits_verification"]["status"] == "NOT_RETAINED_IN_LEARN191_ARTIFACT"
    assert record["different_token_budget_learned_evidence"]["10m"]["admission_status"] == "NOT_ADMITTED_NO_VERIFY218"
    assert record["external_real_corpus_evidence"]["included"] is False
    assert all(v is False for v in record["prohibitions"].values())


def validate_m150(report: dict, record: dict) -> None:
    assert report["source"]["git_sha"] == M150_SOURCE
    assert report["truth_model"]["corpus_identity_sha256"] == DATA25
    for scale, params in EXPECTED_PARAMS.items():
        src = report["scales"][scale]
        rung = record["rungs"][scale]
        assert src["model"]["parameter_count"] == params
        assert src["model"]["spec_sha256"] == rung["model_spec"]["sha256"]
        assert src["training"]["optimized_tokens"] == 948_504
        assert src["checkpoints"]["best_checkpoint_id"] == rung["best_checkpoint"]["id"]
        assert src["checkpoints"]["final_checkpoint_id"] == rung["final_checkpoint"]["id"]
        assert src["fresh_verification"]["status"] == "PASS"


def validate_recover178(report: dict) -> None:
    assert report["source_sha"] == RECOVER178_SOURCE
    assert report["corpus_identity_sha256"] == DATA25
    assert set(report["scale_order"]) == set(EXPECTED_PARAMS)
    for scale in EXPECTED_PARAMS:
        item = report["scales"][scale]
        assert item["parameter_count"] == EXPECTED_PARAMS[scale]
        assert item["all_evaluation_non_mutating"] is True
        assert item["checkpoint_optimizer_steps"] == [0, 250, 500, 750, 1000]
        assert max(item["final_observed_exposures"].values()) == 160


def validate_learn191(report: dict) -> None:
    assert report["source_sha"] == LEARN191_SOURCE
    assert report["model"]["parameter_count"] == 3_213_120
    assert report["model"]["spec_sha256"] == "462c85da80a3c0d7d6a4f1a570b87d208b1847d8a57b12a4d9be7e36846b65dc"
    assert report["corpus_identity_sha256"] == DATA25
    assert [x["actual_optimized_tokens"] for x in report["checkpoints"]] == [17_125, 66_417, 131_938]
    assert report["best_checkpoint"]["selection_validation_bpb"] == 2.2859499700392583
    assert report["final_checkpoint"]["selection_validation_bpb"] == 2.2859499700392583
    assert report["fresh_process_resume"]["passed"] is True
    assert report["generation"]["backend_diagnostics"]["backend"] == "first_party_torch"
    assert all(x["selection_validation"]["non_mutation_passed"] for x in report["evaluations"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m150-report", type=Path)
    parser.add_argument("--recover178-report", type=Path)
    parser.add_argument("--learn191-report", type=Path)
    args = parser.parse_args()
    record = load(RECORD)
    validate_record(record)
    if args.m150_report: validate_m150(load(args.m150_report), record)
    if args.recover178_report: validate_recover178(load(args.recover178_report))
    if args.learn191_report: validate_learn191(load(args.learn191_report))
    print("MILESTONE221_VALIDATION_PASS")


if __name__ == "__main__":
    main()
