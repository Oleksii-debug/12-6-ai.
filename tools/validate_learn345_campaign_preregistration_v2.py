from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

DEFAULT = Path("evidence/learn345/20m_campaign_preregistration_v2.json")


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def canonical_without_identity(data: dict[str, object]) -> bytes:
    body = dict(data)
    body.pop("evidence_identity_sha256", None)
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def validate(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = hashlib.sha256(canonical_without_identity(data)).hexdigest()
    if data.get("evidence_identity_sha256") != expected:
        fail("evidence identity mismatch")

    if data.get("schema") != "12-6.learn345.20m-campaign-preregistration.v2":
        fail("schema drift")
    if data.get("execution_profile") != "LOCAL_FREE":
        fail("execution profile drift")
    if data.get("activation_state") != (
        "BLOCKED_MISSING_TERMINAL_CORPUS_PACKED_EXPOSURE_D05_D06_RECIPE"
    ):
        fail("activation state must remain fail-closed")

    authorities = data["observed_authorities"]
    model = authorities["primary_20m_model"]
    if model != {
        "authority_sha": "e4ff486fd90802fc123bebf60eed4e59196a98df",
        "parameter_count": 20_613_440,
        "model_spec_sha256": "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441",
        "init_spec_sha256": "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5",
        "random_init": True,
        "mechanics_authority_available": True,
    }:
        fail("MODEL-341 authority drift")

    optimizer = authorities["optimizer_mechanics"]
    if optimizer["contract_identity_sha256"] != (
        "f212d051e80d65b9c731ea38dbe08b3e48d930f32caeb3da7becf68030613f21"
    ):
        fail("TRAIN-344B identity drift")
    if optimizer["scoped_ci_conclusion"] != "success":
        fail("TRAIN-344B scoped mechanics evidence must be terminal success")
    if optimizer["status"] != "MECHANICS_ONLY_NO_LR_SELECTION_AUTHORITY":
        fail("optimizer mechanics must not select the learned-campaign LR")

    packed = authorities["d04_packed_exposure"]
    if packed["real_postpack_unique_loss_positions"] != 0:
        fail("current real post-pack exposure must remain zero")
    if packed["status"] != "NOT_MATERIALIZED":
        fail("must not fabricate packed exposure")

    recipe = data["recipe_contract"]
    if recipe["optimizer_name"] != "AdamW":
        fail("AdamW control drift")
    if recipe["mechanics_candidate_learning_rates"] != [0.00016, 0.00022, 0.00026]:
        fail("mechanics LR candidate set drift")
    if recipe["learning_rate_selection_authority"] != (
        "REQUIRED_SEPARATE_TERMINAL_RECIPE_AUTHORITY_BEFORE_STEP_1"
    ):
        fail("synthetic mechanics may not self-select learned-campaign LR")
    required = set(recipe["required_before_step_1"])
    for item in {
        "exact_seed_vector",
        "exact_learning_rate",
        "exact_scheduler",
        "exact_precision",
        "exact_train_trace_identity",
        "exact_next_exposure_identity",
    }:
        if item not in required:
            fail(f"missing step-1 recipe binding: {item}")

    campaign = data["campaign"]
    if campaign["requested_optimized_target_budget"] != 20_000_000:
        fail("campaign budget drift")
    if campaign["meaningful_minimum_optimized_targets"] != 10_000_000:
        fail("meaningful floor drift")
    for key in ("replay_allowed", "replacement_sampling_allowed", "padding_counts_as_data"):
        if campaign[key] is not False:
            fail(f"capacity firewall weakened: {key}")

    resume = data["checkpoint_and_resume"]
    if resume["mandatory_fresh_process_resume_fraction"] != "0.50":
        fail("fresh-process resume boundary drift")
    if resume["equivalence_gate"] != (
        "uninterrupted_N_equals_K_checkpoint_fresh_resume_to_N_under_exact_train_trace"
    ):
        fail("resume equivalence gate drift")
    if "next_exposure_identity" not in resume["mandatory_state"]:
        fail("resume must bind exact next exposure")

    evaluation = data["evaluation"]
    if evaluation["per_stratum_required"] != ["UA", "EN", "code"]:
        fail("held-out stratum coverage drift")
    if evaluation["primary_selection_metric"] != "immutable_selection_validation_aggregate_BPB":
        fail("selection metric drift")
    if evaluation["final_test_sealed_until_selection_lock"] is not True:
        fail("final-test firewall weakened")
    if evaluation["final_test_may_influence_selection"] is not False:
        fail("final test may not influence selection")

    truth = data["truth_boundary"]
    if truth != {
        "authorized_optimized_targets": 0,
        "optimizer_updates_executed": 0,
        "campaign_runnable_now": False,
        "long_training_started": False,
        "paid_compute_authorized": False,
        "final_test_payload_accessed": False,
    }:
        fail("truth boundary weakened")

    return expected


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    print(f"PASS LEARN-345-V2 {validate(path)}")


if __name__ == "__main__":
    main()
