from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_WORKER = "MODEL-248-10M-GQA-TRANSFER-V2"
EXPECTED_STATUS = "BLOCKED_MISSING_PREREQUISITE_AUTHORITY"


def validate(payload: dict) -> list[str]:
    errors: list[str] = []
    if payload.get("worker_id") != EXPECTED_WORKER:
        errors.append("worker_id")
    if payload.get("status") != EXPECTED_STATUS:
        errors.append("status")
    if payload.get("training_executed") is not False:
        errors.append("training_executed")
    if payload.get("optimizer_updates") != 0:
        errors.append("optimizer_updates")
    if payload.get("paired_replacement_runs") != 0:
        errors.append("paired_replacement_runs")

    decision = payload.get("decision", {})
    if decision.get("replace_8q2kv_with_8q4kv") is not False:
        errors.append("replacement_must_be_false_while_blocked")
    if decision.get("current_10m_default") != "8Q/2KV":
        errors.append("incumbent_default")
    if decision.get("mha_required_now") is not False:
        errors.append("mha_scope")
    if decision.get("paid_compute") is not False:
        errors.append("paid_compute")

    pm = payload.get("parameter_match", {})
    if pm.get("incumbent_parameters") != 10_000_640:
        errors.append("incumbent_parameters")
    if pm.get("candidate_parameters") != 9_997_568:
        errors.append("candidate_parameters")
    if pm.get("delta") != -3_072:
        errors.append("parameter_delta")
    if pm.get("strict_match") is not False:
        errors.append("strict_parameter_match")
    if pm.get("artificial_trainable_ballast_allowed") is not False:
        errors.append("trainable_ballast")

    missing = payload.get("live_missing_or_nonterminal", {})
    for key in ("model180", "train243", "current_corpus_decision", "current_tokenizer_decision"):
        if not missing.get(key):
            errors.append(f"missing_authority_marker:{key}")

    blockers = payload.get("hard_blockers", [])
    if len(blockers) < 4:
        errors.append("hard_blockers")

    kv = payload.get("unexpanded_kv_cache_reference", {})
    if kv.get("incumbent_8q2kv_bytes") != 3_145_728:
        errors.append("incumbent_kv_bytes")
    if kv.get("candidate_8q4kv_bytes") != 6_291_456:
        errors.append("candidate_kv_bytes")
    if kv.get("candidate_multiplier") != 2.0:
        errors.append("kv_multiplier")

    prior = payload.get("prior_quality_signal", {})
    if prior.get("candidate_wins") != 3 or prior.get("candidate_losses") != 0:
        errors.append("model142_pairing")
    if prior.get("promotion_valid") is not False:
        errors.append("model142_promotion_boundary")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evidence/model248/readiness_20260826.json"),
    )
    args = parser.parse_args()
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    errors = validate(payload)
    result = {
        "worker_id": EXPECTED_WORKER,
        "validation": "PASS" if not errors else "FAIL",
        "runnable": False,
        "errors": errors,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
