from pathlib import Path

from tools.validate_learn318_authority_gate import validate


def test_learn318_independently_reconstructs_and_fails_closed() -> None:
    report = validate(
        Path("evidence/learn318/authority-gate.json"),
        Path("configs/data/data300_corpus_v03_frozen_build_contract_v2.json"),
    )
    assert report["worker_id"] == "LEARN-318-EXTERNAL-REAL-1M-V2"
    assert report["execution"]["training_started"] is False
    assert report["execution"]["optimizer_updates"] == 0
    assert report["budget_preregistration"]["realized_optimized_target_budget"] == 0
    assert (
        report["independent_contract_reconstruction"]["depends_on_learn317_runtime"]
        is False
    )
    assert report["checkpoint_and_evaluation_protocol"][
        "retain_best_and_final_separately"
    ] is True
