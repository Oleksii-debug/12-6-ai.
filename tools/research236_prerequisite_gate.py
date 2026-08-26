from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

WORKER_ID = "RESEARCH-236-CORPUS-ORIGIN-ABLATION"
REPO = "Oleksii-debug/12-6-ai."
DATA25_ID = "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
TOKENIZER_ID = "s0-byte-v1"
EVAL_ID = "7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113"
MATCHED_OPTIMIZED_TOKENS = 131_938
PAIRED_SEEDS = (1337, 1338, 7331)
MODEL_SPECS = {
    "500k": {
        "parameters": 467_808,
        "model_spec_sha256": "208ac8ca113388e76f280d0154cae815785bee7705546f4d854d9447b9dd1f4a",
    },
    "1m": {
        "parameters": 1_037_696,
        "model_spec_sha256": "ff3cee542a1f75bb4e1eff8d7d24d72533af8f4f3d82bd064fb1cbfeba8c8d07",
    },
}


def _load(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _terminal_ok(payload: dict[str, Any] | None, worker_id: str) -> tuple[bool, list[str]]:
    if payload is None:
        return False, [f"missing:{worker_id}"]
    reasons: list[str] = []
    observed_worker = payload.get("worker_id") or payload.get("swarm_worker_id")
    if observed_worker != worker_id:
        reasons.append(f"worker_id:{observed_worker!r}")
    status = str(payload.get("status") or payload.get("decision") or "").upper()
    if status not in {"PASS", "SUCCESS", "TERMINAL_SUCCESS", "CANDIDATE", "READY"}:
        reasons.append(f"terminal_status:{status or 'MISSING'}")
    return not reasons, reasons


def build_gate_report(
    *,
    data230: dict[str, Any] | None,
    eval233: dict[str, Any] | None,
    source_sha: str,
) -> dict[str, Any]:
    blockers: list[str] = []
    data_ok, data_reasons = _terminal_ok(data230, "DATA-230-CORPUS-V03-EXTERNAL-REAL")
    eval_ok, eval_reasons = _terminal_ok(eval233, "EVAL-233-REAL-HOLDOUT-V2")
    if not data_ok:
        blockers.extend(f"data230:{r}" for r in data_reasons)
    if not eval_ok:
        blockers.extend(f"eval233:{r}" for r in eval_reasons)

    if data_ok:
        origin_classes = set(data230.get("origin_classes", []))
        if "EXTERNAL_REAL" not in origin_classes:
            blockers.append("data230:no_external_real_origin")
        if data230.get("deterministic_two_builds_identical") is not True:
            blockers.append("data230:determinism_not_proven")
        supply = int(data230.get("train_loss_token_supply", 0) or 0)
        if supply < MATCHED_OPTIMIZED_TOKENS:
            blockers.append("data230:insufficient_one_pass_loss_token_supply")
        if data230.get("artificial_repetition") not in {False, None}:
            blockers.append("data230:artificial_repetition_detected")

    if eval_ok:
        purposes = set(eval233.get("purposes", []))
        if not {"selection-validation", "final-test"}.issubset(purposes):
            blockers.append("eval233:purpose_separation_missing")
        if eval233.get("final_test_exposed_to_selection") is True:
            blockers.append("eval233:final_test_selection_leak")

    report: dict[str, Any] = {
        "schema": "12-6.research236-prerequisite-gate.v1",
        "worker_id": WORKER_ID,
        "repository": REPO,
        "source_sha": source_sha,
        "status": "READY_TO_EXECUTE" if not blockers else "BLOCKED_MISSING_OR_INVALID_AUTHORITY",
        "blockers": blockers,
        "frozen": {
            "data25_corpus_identity": DATA25_ID,
            "tokenizer": TOKENIZER_ID,
            "evaluation_identity": EVAL_ID,
            "matched_actual_optimized_tokens": MATCHED_OPTIMIZED_TOKENS,
            "paired_seeds": list(PAIRED_SEEDS),
            "model_specs": MODEL_SPECS,
            "sequence_length": 128,
            "batch_size": 8,
            "optimizer": {
                "name": "AdamW",
                "lr": 3e-4,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "schedule": "constant",
                "warmup_steps": 0,
                "grad_clip": 1.0,
                "precision": "fp32_deterministic",
            },
            "exposure_accounting": "actual_unique_nonignored_causal_source_loss_tokens_only",
            "padded_tensor_positions_count_as_exposure": False,
            "source_loss_token_repetition_allowed": False,
        },
        "authorities": {
            "data230_present": data230 is not None,
            "eval233_present": eval233 is not None,
        },
        "claim_boundary": {
            "external_real_automatically_better": False,
            "numerical_ablation_claim_permitted": not blockers,
            "paid_compute": False,
            "foreign_pretrained_weights": False,
        },
    }
    report["report_sha256"] = _sha256_json(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data230")
    parser.add_argument("--eval233")
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_gate_report(
        data230=_load(args.data230),
        eval233=_load(args.eval233),
        source_sha=args.source_sha,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["status"])
    for blocker in report["blockers"]:
        print(blocker)
    return 0 if report["status"] == "READY_TO_EXECUTE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
