#!/usr/bin/env python3
"""Fail-closed validator for the exact MODEL-341 TRAIN-344B mechanics contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CONTRACT_PATH = Path("configs/experiments/train344b_model341_optimizer_mechanics_v2.json")


class ContractError(RuntimeError):
    pass


def canonical_identity(value: dict) -> str:
    unsigned = dict(value)
    unsigned.pop("identity_sha256", None)
    raw = json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def load_contract(repo: Path) -> dict:
    value = json.loads((repo / CONTRACT_PATH).read_text(encoding="utf-8"))
    if value.get("schema") != "12-6.train344b-model341-optimizer-mechanics.v2":
        raise ContractError("contract schema drift")
    if value.get("worker_id") != "TRAIN-344B-MODEL341-OPTIMIZER-MECHANICS":
        raise ContractError("worker identity drift")
    if value.get("identity_sha256") != canonical_identity(value):
        raise ContractError("contract self-hash mismatch")
    return value


def validate_frozen_optimizer(contract: dict) -> None:
    opt = contract["optimizer"]
    expected = {
        "name": "AdamW",
        "learning_rate_candidates": [0.00016, 0.00022, 0.00026],
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.1,
        "gradient_clip_norm": 1.0,
        "scheduler": "constant",
        "warmup_steps": 0,
        "sequence_length": 256,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "precision": "fp32",
        "deterministic_algorithms": True,
        "selection_authority": "NONE",
    }
    if opt != expected:
        raise ContractError("frozen TRAIN-344 optimizer semantics drift")
    probe = contract["bounded_probe"]
    if probe["optimizer_steps_per_lr"] != 32:
        raise ContractError("probe step budget drift")
    if probe["optimized_causal_targets_per_step"] != 255:
        raise ContractError("per-step causal target count drift")
    if probe["optimized_causal_targets_per_lr"] != 8160:
        raise ContractError("per-arm target budget drift")
    if probe["total_optimized_causal_targets"] != 24480:
        raise ContractError("total target budget drift")
    if not probe["same_initial_weights_and_batch_trace_across_arms"]:
        raise ContractError("paired-arm invariant disabled")


def validate_dependency_firewall(contract: dict) -> None:
    gate = contract["dependency_firewall"]
    if gate["learned_corpus_optimizer_updates_authorized"] != 0:
        raise ContractError("learned-corpus optimizer updates must remain zero")
    if gate["long_training_authorized"] or gate["paid_compute_authorized"]:
        raise ContractError("TRAIN-344B may not authorize long or paid training")
    if not gate["rerun_or_refresh_required_after_data_and_d05_terminalization"]:
        raise ContractError("terminal dependency refresh requirement was removed")


def validate_exact_model(repo: Path, contract: dict) -> dict:
    target = contract["target_model"]
    path = (repo / target["config_path"]).resolve()
    if not path.is_file() or repo not in path.parents:
        raise ContractError("exact MODEL-341 config path missing or outside repository")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != target["config_content_sha256"]:
        raise ContractError("MODEL-341 config byte SHA-256 drift")
    if git_blob_sha1(payload) != target["config_git_blob_sha1"]:
        raise ContractError("MODEL-341 config Git blob identity drift")

    config = json.loads(payload.decode("utf-8"))
    if config.get("stage") != "MODEL-341-20M-CANDIDATE-A":
        raise ContractError("MODEL-341 stage identity drift")
    if config.get("canonical_base") != "random_init":
        raise ContractError("canonical Base is no longer random-init")
    if config.get("expected_parameters") != target["parameter_count"]:
        raise ContractError("config expected parameter count drift")
    if config.get("expected_model_identity_sha256") != target["model_spec_sha256"]:
        raise ContractError("config expected ModelSpec identity drift")
    if config.get("expected_init_identity_sha256") != target["init_spec_sha256"]:
        raise ContractError("config expected InitSpec identity drift")

    from twelve_six.model import load_stage_config

    stage = load_stage_config(path)
    parameter_count = stage.model.parameter_count()
    model_identity = stage.model.identity_sha256()
    init_identity = stage.init.identity_sha256()
    if parameter_count != target["parameter_count"]:
        raise ContractError(f"runtime parameter count drift: {parameter_count}")
    if model_identity != target["model_spec_sha256"]:
        raise ContractError("runtime ModelSpec identity drift")
    if init_identity != target["init_spec_sha256"]:
        raise ContractError("runtime InitSpec identity drift")
    if stage.model.vocab_size != target["vocab_size"]:
        raise ContractError("vocabulary identity drift")
    if stage.model.max_seq_len != target["context_length"]:
        raise ContractError("context identity drift")
    geometry = target["geometry"]
    for field, expected in geometry.items():
        if getattr(stage.model, field) != expected:
            raise ContractError(f"MODEL-341 geometry drift at {field}")
    return {
        "status": "READY_FOR_BOUNDED_SYNTHETIC_MECHANICS",
        "config_path": target["config_path"],
        "config_content_sha256": target["config_content_sha256"],
        "config_git_blob_sha1": target["config_git_blob_sha1"],
        "parameter_count": parameter_count,
        "model_spec_sha256": model_identity,
        "init_spec_sha256": init_identity,
        "optimizer_updates_authorized_for_synthetic_probe": 96,
        "learned_corpus_optimizer_updates_authorized": 0,
    }


def build_report(repo: Path) -> dict:
    contract = load_contract(repo)
    validate_frozen_optimizer(contract)
    validate_dependency_firewall(contract)
    model_gate = validate_exact_model(repo, contract)
    return {
        "schema": "12-6.train344b-model341-readiness.v2",
        "worker_id": contract["worker_id"],
        "contract_identity_sha256": contract["identity_sha256"],
        "source_model_authority_sha": contract["target_model"]["authority_sha"],
        "model_gate": model_gate,
        "dependency_firewall": contract["dependency_firewall"],
        "local_free": True,
        "paid_compute": False,
        "long_training": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("evidence/train344b/readiness.json"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    report = build_report(repo)
    output = (repo / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
