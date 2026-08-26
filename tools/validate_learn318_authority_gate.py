#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

EXPECTED_SCHEMA = "12-6.learn318-external-real-1m-v2.authority-gate.v1"
EXPECTED_WORKER = "LEARN-318-EXTERNAL-REAL-1M-V2"
EXPECTED_STATUS = "BLOCKED_FROZEN_CORPUS_NOT_TERMINAL_AND_NO_REPLAY_BUDGET_ZERO"
EXPECTED_DATA300_SHA = "8ea7f830e50a23754d189dd4134f4afad76a7ee9"
EXPECTED_DATA300_ID = "07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5"
EXPECTED_TOKENIZER_CONFIG = "b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1"
EXPECTED_TOKENIZER_VOCAB = "905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571"
EXPECTED_1M_SPEC = "ff3cee542a1f75bb4e1eff8d7d24d72533af8f4f3d82bd064fb1cbfeba8c8d07"
EXPECTED_1M_PARAMETERS = 1_037_696
EXPECTED_BLOCKERS = {
    "G05_QUALITY",
    "G06_PRIVACY",
    "G09_BALANCE_DIVERSITY",
    "G10_SELECTION_VALIDATION",
    "G12_UNIQUE_LOSS",
    "G14_TWO_CLEAN_BUILDS",
}
EXPECTED_BPB = {
    "aggregate_BPB",
    "UA_BPB",
    "EN_BPB",
    "code_BPB",
    "source_family_BPB",
}


def canonical_identity(payload: dict) -> str:
    body = dict(payload)
    claimed = body.pop("gate_identity_sha256", None)
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError("missing gate_identity_sha256")
    raw = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate(gate_path: Path, corpus_contract_path: Path | None = None) -> dict:
    report = json.loads(gate_path.read_text(encoding="utf-8"))

    if report.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("wrong LEARN-318 schema")
    if report.get("worker_id") != EXPECTED_WORKER:
        raise ValueError("wrong worker id")
    if report.get("status") != EXPECTED_STATUS:
        raise ValueError("LEARN-318 must fail closed at this cutoff")
    if canonical_identity(report) != report.get("gate_identity_sha256"):
        raise ValueError("LEARN-318 gate identity mismatch")

    reconstruction = report["independent_contract_reconstruction"]
    if reconstruction["depends_on_learn317_runtime"]:
        raise ValueError("LEARN-318 illegally depends on LEARN-317 runtime")
    if reconstruction["learn317_availability_required"]:
        raise ValueError("LEARN-317 availability became a prerequisite")

    corpus = reconstruction["frozen_corpus_contract"]
    if corpus["source_sha"] != EXPECTED_DATA300_SHA:
        raise ValueError("DATA-300 source SHA drift")
    if corpus["contract_identity_sha256"] != EXPECTED_DATA300_ID:
        raise ValueError("DATA-300 contract identity drift")
    if corpus["contract_state"] != "FROZEN_EXECUTABLE_CONTRACT":
        raise ValueError("DATA-300 contract is not frozen")
    if corpus["corpus_state"] != "NOT_BUILT_NOT_FROZEN_NOT_TERMINAL":
        raise ValueError("unexpected DATA-300 corpus state")
    if corpus["family_constrained_no_replay_budget"] != 0:
        raise ValueError("expected frozen family-constrained no-replay budget to be zero")
    if set(corpus["blocking_gates"]) != EXPECTED_BLOCKERS:
        raise ValueError("blocking gate set drift")

    tokenizer = reconstruction["tokenizer"]
    if tokenizer["tokenizer_id"] != "s0-byte-v1":
        raise ValueError("tokenizer id drift")
    if tokenizer["config_sha256"] != EXPECTED_TOKENIZER_CONFIG:
        raise ValueError("tokenizer config identity drift")
    if tokenizer["vocab_sha256"] != EXPECTED_TOKENIZER_VOCAB:
        raise ValueError("tokenizer vocab identity drift")
    if tokenizer["vocab_size"] != 256 or tokenizer["special_tokens"] != []:
        raise ValueError("canonical byte tokenizer contract drift")

    optimizer = reconstruction["optimizer"]
    expected_optimizer = {
        "name": "AdamW",
        "learning_rate": 0.0003,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "schedule": "constant",
        "warmup_steps": 0,
        "gradient_clip_norm": 1.0,
        "precision": "fp32",
        "sequence_length": 128,
        "batch_size": 8,
        "seed": 1337,
        "document_isolated": True,
    }
    if optimizer != expected_optimizer:
        raise ValueError("optimizer/packing contract drift")

    model = reconstruction["model_1m"]
    if model["parameters"] != EXPECTED_1M_PARAMETERS:
        raise ValueError("1M parameter count drift")
    if model["model_spec_sha256"] != EXPECTED_1M_SPEC:
        raise ValueError("1M ModelSpec identity drift")
    if not model["random_initialization_only"] or model["foreign_pretrained_weights"]:
        raise ValueError("scratch Base requirement lost")

    budget = report["budget_preregistration"]
    if budget["realized_optimized_target_budget"] != 0:
        raise ValueError("realized no-replay budget must remain zero")
    forbidden_repetition = (
        budget["artificial_repetition_allowed"],
        budget["document_replication_allowed"],
        budget["sampling_with_replacement_allowed"],
        budget["recycle_to_hit_budget_allowed"],
        budget["padding_counts_as_data"],
    )
    if any(forbidden_repetition):
        raise ValueError("artificial repetition was enabled")
    if budget["optimizer_step_1_authorized"]:
        raise ValueError("optimizer step 1 cannot be authorized at budget zero")

    protocol = report["checkpoint_and_evaluation_protocol"]
    if not protocol["retain_best_and_final_separately"]:
        raise ValueError("best/final separation lost")
    if not protocol["fresh_process_resume_required"]:
        raise ValueError("fresh-process resume requirement lost")
    if protocol["evaluation_model_state_mutation_allowed"]:
        raise ValueError("evaluation model mutation enabled")
    if protocol["evaluation_trainer_state_mutation_allowed"]:
        raise ValueError("evaluation trainer mutation enabled")
    if not protocol["selection_validation_only_for_best_checkpoint"]:
        raise ValueError("best checkpoint selector may use non-selection data")
    if protocol["final_test_before_selection_freeze_allowed"]:
        raise ValueError("final test exposure before selection freeze enabled")
    if set(protocol["required_bpb_reports_when_unblocked"]) != EXPECTED_BPB:
        raise ValueError("required BPB report set drift")

    execution = report["execution"]
    if execution["execution_profile"] != "LOCAL_FREE" or execution["paid_compute"]:
        raise ValueError("LOCAL_FREE execution boundary violated")
    if execution["training_started"] or execution["optimizer_updates"] != 0:
        raise ValueError("training occurred despite zero authorized budget")
    if execution["checkpoints_written"] or execution["bpb_metrics_available"]:
        raise ValueError("training outputs claimed despite blocked run")

    if corpus_contract_path is not None:
        source = json.loads(corpus_contract_path.read_text(encoding="utf-8"))
        if source["contract_identity_sha256"] != EXPECTED_DATA300_ID:
            raise ValueError("repository DATA-300 identity does not match LEARN-318 reconstruction")
        if source["contract_state"] != corpus["contract_state"]:
            raise ValueError("repository DATA-300 contract state mismatch")
        if source["corpus_state"] != corpus["corpus_state"]:
            raise ValueError("repository DATA-300 corpus state mismatch")
        if source["terminal_component_lock"]["balance"]["current_family_constrained_no_replay_budget"] != 0:
            raise ValueError("repository DATA-300 no-replay budget is no longer zero")
        if set(source["current_candidate_status"]["blocking_gates"]) != EXPECTED_BLOCKERS:
            raise ValueError("repository DATA-300 blocker set mismatch")

    return report


def main() -> int:
    gate = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evidence/learn318/authority-gate.json")
    source = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path("configs/data/data300_corpus_v03_frozen_build_contract_v2.json")
    )
    report = validate(gate, source)
    print(
        json.dumps(
            {
                "validation": "PASS",
                "status": report["status"],
                "training_started": report["execution"]["training_started"],
                "optimizer_updates": report["execution"]["optimizer_updates"],
                "realized_no_replay_budget": report["budget_preregistration"][
                    "realized_optimized_target_budget"
                ],
                "gate_identity_sha256": report["gate_identity_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
