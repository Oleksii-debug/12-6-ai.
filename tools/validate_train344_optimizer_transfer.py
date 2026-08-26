#!/usr/bin/env python3
"""Validate TRAIN-344 preregistration and fail closed without an exact ~20M ModelSpec."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CONTRACT = Path("configs/experiments/train344_20m_optimizer_transfer_contract.json")


class ContractError(RuntimeError):
    pass


def _canonical_sha(value: dict) -> str:
    unsigned = dict(value)
    unsigned.pop("identity_sha256", None)
    raw = json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(value: dict) -> None:
    if value.get("worker_id") != "TRAIN-344-20M-OPTIMIZER-TRANSFER-CONTRACT":
        raise ContractError("worker identity drift")
    if value.get("identity_sha256") != _canonical_sha(value):
        raise ContractError("contract self-hash mismatch")
    opt = value["preregistered_optimizer"]
    if opt["lr_transfer"]["candidates"] != [0.00016, 0.00022, 0.00026]:
        raise ContractError("LR grid drift")
    if opt["betas"] != [0.9, 0.95] or opt["beta_sweep"]:
        raise ContractError("beta contract drift")
    if opt["weight_decay"] != 0.1 or opt["weight_decay_sweep"]:
        raise ContractError("weight-decay contract drift")
    if opt["gradient_clip_norm"] != 1.0 or opt["clipping_sweep"]:
        raise ContractError("clipping contract drift")
    if (opt["micro_batch_size"], opt["sequence_length"], opt["gradient_accumulation_steps"]) != (1, 256, 1):
        raise ContractError("batch/sequence contract drift")
    probe = value["bounded_stability_probe"]
    if probe["optimizer_steps_per_lr"] != 32 or probe["optimized_tokens_per_lr"] != 8160:
        raise ContractError("bounded probe budget drift")
    if probe["total_optimized_tokens_all_lr_arms"] != 24480:
        raise ContractError("total probe budget drift")
    if probe["selection_authority"] != "NONE":
        raise ContractError("stability probe gained selection authority")
    budget = value["learned_transfer_budget"]
    if budget["authorized_unique_nonignored_causal_loss_positions_now"] != 0:
        raise ContractError("learned-corpus budget must remain zero under RESEARCH-313")
    if value["scope"]["paid_compute"] or value["scope"]["huge_sweep"]:
        raise ContractError("scope drift")


def validate_model_config(repo: Path, model_config: str | None, contract: dict) -> dict:
    if not model_config:
        return {
            "status": "BLOCKED_MISSING_20M_MODELSPEC",
            "optimizer_updates_authorized": 0,
            "reason": "No exact mechanically qualified ~20M stage config was supplied; TRAIN-344 does not invent geometry.",
        }
    path = (repo / model_config).resolve()
    if not path.is_file() or repo.resolve() not in path.parents:
        raise ContractError("model config must be an existing repository file")
    from twelve_six.model import load_stage_config
    stage = load_stage_config(path)
    count = stage.model.parameter_count()
    low, high = contract["target_20m_gate"]["allowed_parameter_window"]
    if not low <= count <= high:
        raise ContractError(f"model parameter count {count} outside preregistered [{low}, {high}]")
    return {
        "status": "READY_FOR_BOUNDED_STABILITY_PROBE",
        "optimizer_updates_authorized": 32 * 3,
        "model_config": model_config,
        "parameter_count": count,
        "model_spec_sha256": stage.model.identity_sha256(),
        "init_spec_sha256": stage.init.identity_sha256(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--model-config")
    parser.add_argument("--output", type=Path, default=Path("evidence/train344/readiness.json"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    contract = _load(repo / CONTRACT)
    validate_contract(contract)
    gate = validate_model_config(repo, args.model_config, contract)
    report = {
        "schema": "12-6.train344-readiness.v1",
        "worker_id": contract["worker_id"],
        "contract_identity_sha256": contract["identity_sha256"],
        "gate": gate,
        "learned_transfer_budget": contract["learned_transfer_budget"],
        "local_free": True,
        "paid_compute": False,
    }
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
