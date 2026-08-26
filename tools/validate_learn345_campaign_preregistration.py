#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

DEFAULT = Path("evidence/learn345/20m_campaign_preregistration_v1.json")


def canonical_without_identity(data: dict) -> bytes:
    body = dict(data)
    body.pop("evidence_identity_sha256", None)
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def validate(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))

    expected = hashlib.sha256(canonical_without_identity(data)).hexdigest()
    if data.get("evidence_identity_sha256") != expected:
        fail("evidence identity mismatch")

    if data["worker_id"] != "LEARN-345-20M-CAMPAIGN-PREREGISTRATION":
        fail("worker id drift")
    if data["execution_profile"] != "LOCAL_FREE":
        fail("execution profile drift")
    if data["long_campaign_executed"] is not False:
        fail("long campaign must not execute")
    if data["optimizer_updates_executed"] != 0:
        fail("optimizer updates must remain zero")

    state = data["activation_state"]
    if not state.startswith("BLOCKED_"):
        fail("current preregistration must fail closed")

    d301 = data["observed_authorities"]["data301_terminal_build"]
    if d301["status"] != "TERMINAL_BLOCKED":
        fail("DATA-301 status drift")
    if d301["corpus_identity"] is not None or d301["shard_identity"] is not None:
        fail("must not fabricate terminal corpus/shard identity")
    if d301["authorized_balanced_no_replay_capacity"] != 0:
        fail("current no-replay authority must remain zero")

    model = data["observed_authorities"]["primary_20m_architecture"]
    if model["terminal_authority_found"] is not False:
        fail("primary 20M authority unexpectedly marked terminal")
    if model["modelspec_identity_sha256"] is not None:
        fail("must not invent ModelSpec identity")

    opt = data["observed_authorities"]["optimizer_transfer"]
    if opt["terminal_authority_found"] is not False:
        fail("TRAIN-344 unexpectedly marked terminal")
    if opt["optimizer_contract_identity_sha256"] is not None:
        fail("must not invent optimizer identity")

    campaign = data["campaign"]
    if campaign["requested_optimized_target_budget"] != 20_000_000:
        fail("requested budget drift")
    if campaign["meaningful_minimum_optimized_targets"] != 10_000_000:
        fail("meaningful floor drift")
    if campaign["first_campaign_upper_guardrail"] != 40_000_000:
        fail("planning guardrail drift")
    if campaign["replay_allowed"] is not False:
        fail("replay must remain forbidden")
    if campaign["replacement_sampling_allowed"] is not False:
        fail("replacement sampling must remain forbidden")
    if campaign["padding_counts_as_data"] is not False:
        fail("padding must not count as data")

    fractions = data["schedule"]["boundary_fractions"]
    if fractions != ["0", "0.10", "0.25", "0.50", "0.75", "0.90", "1.00"]:
        fail("schedule drift")
    if data["schedule"]["mandatory_fresh_process_resume_after_fraction"] != "0.50":
        fail("fresh-process resume boundary drift")

    select = data["selection_rule"]
    if select["primary_metric"] != "immutable_selection_validation_aggregate_BPB":
        fail("selection metric drift")
    if select["metric_direction"] != "minimize":
        fail("selection direction drift")
    if select["final_test_may_influence_selection"] is not False:
        fail("final test must not influence selection")

    fw = data["final_test_firewall"]
    if fw["sealed_during_training_and_selection"] is not True:
        fail("final-test firewall weakened")
    if fw["final_test_payload_read_before_selection_lock"] is not False:
        fail("final-test payload exposure")
    if fw["final_test_outcomes_read_before_selection_lock"] is not False:
        fail("final-test outcome exposure")

    truth = data["truth_boundary"]
    if any(
        truth[k]
        for k in (
            "research_corpus_v1_consumed",
            "primary_20m_model_consumed",
            "optimizer_transfer_consumed",
            "campaign_runnable_now",
            "long_training_started",
        )
    ):
        fail("truth boundary weakened")

    return expected


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    identity = validate(path)
    print(f"PASS LEARN-345 {identity}")


if __name__ == "__main__":
    main()
