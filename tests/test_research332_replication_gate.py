import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_research332_frozen_controls_match_learn318() -> None:
    reference = _load("evidence/learn318/authority-gate.json")
    replication = _load("evidence/research332/replication-gate.json")

    ref = reference["independent_contract_reconstruction"]
    frozen = replication["frozen_controls"]

    assert replication["status"] == (
        "BLOCKED_REFERENCE_RUN_NEVER_AUTHORIZED_NO_NONZERO_FROZEN_TRAINING_BUDGET"
    )
    assert replication["execution_profile"] == "LOCAL_FREE"
    assert replication["reference_authority"]["model_parameters"] == ref["model_1m"]["parameters"]
    assert replication["reference_authority"]["model_spec_sha256"] == ref["model_1m"]["model_spec_sha256"]

    assert frozen["corpus_contract_identity_sha256"] == ref["frozen_corpus_contract"]["contract_identity_sha256"]
    assert frozen["corpus_source_sha"] == ref["frozen_corpus_contract"]["source_sha"]
    assert frozen["current_authorized_optimized_target_budget"] == reference["budget_preregistration"]["realized_optimized_target_budget"] == 0

    assert frozen["tokenizer"]["id"] == ref["tokenizer"]["tokenizer_id"]
    assert frozen["tokenizer"]["vocab_size"] == ref["tokenizer"]["vocab_size"]
    assert frozen["tokenizer"]["config_sha256"] == ref["tokenizer"]["config_sha256"]
    assert frozen["tokenizer"]["vocab_sha256"] == ref["tokenizer"]["vocab_sha256"]

    optimizer = frozen["optimizer"]
    reference_optimizer = ref["optimizer"]
    for key in (
        "name",
        "learning_rate",
        "betas",
        "eps",
        "weight_decay",
        "schedule",
        "warmup_steps",
        "gradient_clip_norm",
        "precision",
        "sequence_length",
        "batch_size",
        "document_isolated",
    ):
        assert optimizer[key] == reference_optimizer[key]


def test_research332_seed_panel_is_preregistered_but_not_executed() -> None:
    replication = _load("evidence/research332/replication-gate.json")

    assert replication["seed_panel"]["seeds"] == [1337, 2027, 4099, 7919, 104729]
    assert replication["seed_panel"]["seed_is_only_intended_experimental_factor"] is True
    assert replication["seed_panel"]["data_order_fixed_independently_of_initialization_seed"] is True
    assert replication["execution"]["training_started"] is False
    assert replication["execution"]["completed_seed_count"] == 0
    assert replication["execution"]["optimizer_updates_total"] == 0
    assert replication["execution"]["bpb_metrics_available"] is False
    assert replication["execution"]["variance_estimable"] is False

    results = replication["seed_results"]
    assert len(results) == 5
    assert all(row["status"] == "BLOCKED" for row in results)
    assert all(row["optimizer_updates"] == 0 for row in results)
    assert all(row["aggregate_bpb"] is None for row in results)


def test_research332_never_turns_zero_budget_into_zero_bpb() -> None:
    replication = _load("evidence/research332/replication-gate.json")

    assert replication["frozen_controls"]["current_authorized_optimized_target_budget"] == 0
    assert replication["variance_report"]["status"] == "NOT_ESTIMABLE_NO_COMPLETED_SEEDS"
    for row in replication["seed_results"]:
        assert row["aggregate_bpb"] is None
        assert row["ua_bpb"] is None
        assert row["en_bpb"] is None
        assert row["code_bpb"] is None
        assert row["source_family_bpb"] is None
