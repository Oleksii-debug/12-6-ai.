"""RESEARCH-212 recovery authority for the RESEARCH-192 frozen scale contract.

This module does not train models.  It resolves the current executable identities,
compares them with one explicit frozen scientific contract, and provides a cheap
DATA-25 ledger/model-construction preflight before any scale matrix is authorized.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from twelve_six import milestone100_first_learned as m100
from twelve_six import milestone150_learned_base_ladder as m150
from twelve_six import research192_scaling_transfer as r192
from twelve_six.checkpoint import hash_json
from twelve_six.model import ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer

SCHEMA = "12-6.research212-fixed-scale-contract-recovery.v1"
WORKER_ID = "RESEARCH-212-FIXED-SCALE-CONTRACT-RECOVERY"
HISTORICAL_RUN_ID = 32941405721
HISTORICAL_HEAD_SHA = "08db052c228e6dd10f59e21d9fbee0d4d77e06d6"

FROZEN_CONTRACT: dict[str, Any] = {
    "schema": "12-6.research192-frozen-scientific-contract.v3",
    "corpus": {
        "id": "DATA-25",
        "identity_sha256": "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8",
    },
    "tokenizer": {
        "version": "s0-byte-v1",
        "config_sha256": "b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1",
        "vocab_sha256": "905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571",
        "vocab_size": 256,
        "special_tokens": {},
    },
    "evaluation_identity_sha256": "7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113",
    "init_spec_sha256": "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5",
    "packing": {
        "version": "s0-byte-pack-v1",
        "sequence_length": 128,
        "cross_document": False,
        "batch_size": 8,
        "mixture_pattern": list(m100.MIXTURE),
    },
    "optimizer_recipe": {
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "warmup_steps": 0,
        "scheduler": "constant",
        "gradient_accumulation_steps": 1,
        "gradient_clip_norm": 1.0,
        "precision": "fp32",
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
    },
    "scales": {
        "1m": {
            "parameters": 1_037_696,
            "model_spec_sha256": "ff3cee542a1f75bb4e1eff8d7d24d72533af8f4f3d82bd064fb1cbfeba8c8d07",
            "geometry": {"d_model": 128, "n_layers": 5, "n_heads": 8, "n_kv_heads": 8, "head_dim": 16, "d_ff": 352},
            "authority": "MILESTONE-150",
        },
        "3m": {
            "parameters": 3_213_120,
            "model_spec_sha256": "462c85da80a3c0d7d6a4f1a570b87d208b1847d8a57b12a4d9be7e36846b65dc",
            "geometry": {"d_model": 192, "n_layers": 7, "n_heads": 12, "n_kv_heads": 12, "head_dim": 16, "d_ff": 528},
            "authority": "LEARN-191@a75920cef8bde37a8c590e34095be83c97b75f1d",
        },
        "10m": {
            "parameters": 10_000_640,
            "model_spec_sha256": "f01cf22d3a44bd72be74691ca4b4a75b093851f45fc2b252c5116eb72370dc53",
            "geometry": {"d_model": 256, "n_layers": 12, "n_heads": 16, "n_kv_heads": 16, "head_dim": 16, "d_ff": 736},
            "authority": "RESEARCH-192 MHA/context-256 fixed-control continuation",
        },
    },
    "checkpoint_steps": [18, 70, 139],
    "optimized_token_budgets": {"18": 17_125, "70": 66_417, "139": 131_938},
    "midpoint_step": 70,
    "final_step": 139,
    "source_exposure_ceiling": 0.01,
    "m150_producer": {
        "source_sha": "5838cd16869dcfcf762368d8673eddf52d51b7e3",
        "workflow_run_id": 32937411703,
        "artifact_id": 9595677772,
        "artifact_name": "milestone150-learned-base-ladder-v1",
        "artifact_sha256": "c00b7e9006320f8916c739a3311e8cc47ad0d0b16957f8ebd7d19233fd9f1c71",
        "ladder_report_sha256": "1f8350bed574a7b78778f0ebb7854ca5311173006820ec27110122f8965c9a5a",
        "one_m_report_identity_sha256": "1b63e8f5096c43b9a36923ddd9d4b8d8a8d1705559f63080c0a287c5520fc738",
    },
}

# Exact stale expectations exercised by run 32941405721 after the final source-only
# convergence commit.  This is a regression fixture, never an eligible contract.
HISTORICAL_RUN_32941405721_CONTRACT = copy.deepcopy(FROZEN_CONTRACT)
HISTORICAL_RUN_32941405721_CONTRACT["scales"]["3m"] = {
    "parameters": 3_221_184,
    "model_spec_sha256": "3255ebffea76d17e59a19b4de50be616b27e85593a6eebec0db935d7efebb5ea",
    "geometry": {"d_model": 192, "n_layers": 7, "n_heads": 12, "n_kv_heads": 12, "head_dim": 16, "d_ff": 530},
    "authority": "stale pre-LEARN-191 RESEARCH-192 interpolation",
}
HISTORICAL_RUN_32941405721_CONTRACT["checkpoint_steps"] = [500, 1000]
HISTORICAL_RUN_32941405721_CONTRACT["optimized_token_budgets"] = {
    "500": 474_377,
    "1000": 948_504,
}
HISTORICAL_RUN_32941405721_CONTRACT["midpoint_step"] = 500
HISTORICAL_RUN_32941405721_CONTRACT["final_step"] = 1000


class ContractRecoveryError(RuntimeError):
    pass


def _optimizer_recipe() -> dict[str, Any]:
    cfg = asdict(r192.trainer_config(1337))
    return {
        key: cfg[key]
        for key in (
            "learning_rate",
            "weight_decay",
            "betas",
            "eps",
            "warmup_steps",
            "scheduler",
            "gradient_accumulation_steps",
            "gradient_clip_norm",
            "precision",
            "deterministic_algorithms",
            "deterministic_warn_only",
        )
    } | {"betas": list(cfg["betas"])}


def _scale_record(scale: str) -> dict[str, Any]:
    spec = ModelSpec.from_dict(dict(r192.SCALE_SPECS[scale]["model"]))
    geometry = {
        "d_model": spec.d_model,
        "n_layers": spec.n_layers,
        "n_heads": spec.n_heads,
        "n_kv_heads": spec.n_kv_heads,
        "head_dim": spec.head_dim,
        "d_ff": spec.d_ff,
    }
    authority = {
        "1m": "MILESTONE-150",
        "3m": f"LEARN-191@{r192.LEARN191_GEOMETRY['source_sha']}",
        "10m": "RESEARCH-192 MHA/context-256 fixed-control continuation",
    }[scale]
    return {
        "parameters": spec.parameter_count(),
        "model_spec_sha256": spec.identity_sha256(),
        "geometry": geometry,
        "authority": authority,
    }


def resolved_contract() -> dict[str, Any]:
    tok = ByteTokenizer()
    return {
        "schema": FROZEN_CONTRACT["schema"],
        "corpus": {"id": "DATA-25", "identity_sha256": r192.EXPECTED_CORPUS_ID},
        "tokenizer": {
            "version": tok.identity.version,
            "config_sha256": tok.identity.config_sha256,
            "vocab_sha256": tok.identity.vocab_sha256,
            "vocab_size": tok.identity.vocab_size,
            "special_tokens": dict(tok.identity.special_tokens),
        },
        "evaluation_identity_sha256": r192.EXPECTED_EVALUATION_ID,
        "init_spec_sha256": r192.init_spec().identity_sha256(),
        "packing": {
            "version": m100.PACKING_VERSION,
            "sequence_length": m100.SEQ,
            "cross_document": False,
            "batch_size": m100.BATCH,
            "mixture_pattern": list(m100.MIXTURE),
        },
        "optimizer_recipe": _optimizer_recipe(),
        "scales": {scale: _scale_record(scale) for scale in ("1m", "3m", "10m")},
        "checkpoint_steps": list(r192.CHECKPOINT_STEPS),
        "optimized_token_budgets": {str(k): v for k, v in r192.EXPECTED_TOKEN_BUDGETS.items()},
        "midpoint_step": r192.MIDPOINT_STEP,
        "final_step": r192.FINAL_STEP,
        "source_exposure_ceiling": 0.01,
        "m150_producer": dict(r192.M150_PRODUCER),
    }


def _reason_code(path: str) -> str:
    if path.startswith("scales.") and path.endswith("parameters"):
        return "SCALE_PARAMETER_COUNT_MISMATCH"
    if path.startswith("scales.") and path.endswith("model_spec_sha256"):
        return "SCALE_MODEL_SPEC_IDENTITY_MISMATCH"
    if path.startswith("scales.") and ".geometry." in path:
        return "SCALE_GEOMETRY_MISMATCH"
    if path == "checkpoint_steps":
        return "CHECKPOINT_STEPS_MISMATCH"
    if path.startswith("optimized_token_budgets"):
        return "OPTIMIZED_TOKEN_BUDGET_MISMATCH"
    if path.startswith("optimizer_recipe"):
        return "OPTIMIZER_RECIPE_MISMATCH"
    if path.startswith("packing"):
        return "PACKING_OR_BATCH_GEOMETRY_MISMATCH"
    if path.startswith("tokenizer"):
        return "TOKENIZER_IDENTITY_MISMATCH"
    if path.startswith("corpus"):
        return "CORPUS_IDENTITY_MISMATCH"
    if path.startswith("evaluation_identity_sha256"):
        return "EVALUATION_IDENTITY_MISMATCH"
    if path.startswith("init_spec_sha256"):
        return "INIT_SPEC_IDENTITY_MISMATCH"
    if path.startswith("m150_producer"):
        return "M150_PRODUCER_IDENTITY_MISMATCH"
    return "FROZEN_CONTRACT_MISMATCH"


def _diff(expected: Any, actual: Any, path: str, reasons: list[dict[str, Any]]) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}" if path else str(key)
            if key not in expected:
                reasons.append({"code": _reason_code(child), "path": child, "frozen": None, "actual": actual[key],
                                "message": f"{child}: unexpected live field {actual[key]!r}"})
            elif key not in actual:
                reasons.append({"code": _reason_code(child), "path": child, "frozen": expected[key], "actual": None,
                                "message": f"{child}: frozen field {expected[key]!r} is missing from live contract"})
            else:
                _diff(expected[key], actual[key], child, reasons)
        return
    if expected != actual:
        reasons.append({
            "code": _reason_code(path),
            "path": path,
            "frozen": expected,
            "actual": actual,
            "message": f"{path}: frozen {expected!r} != live incumbent {actual!r}",
        })


def diagnose_contract(candidate: dict[str, Any]) -> dict[str, Any]:
    actual = resolved_contract()
    reasons: list[dict[str, Any]] = []
    _diff(candidate, actual, "", reasons)
    return {
        "schema": SCHEMA,
        "status": "PASS" if not reasons else "FAIL",
        "reason_count": len(reasons),
        "reasons": reasons,
        "candidate_identity_sha256": hash_json(candidate),
        "resolved_identity_sha256": hash_json(actual),
    }


def require_frozen_contract() -> dict[str, Any]:
    r192.validate_static_contract()
    result = diagnose_contract(FROZEN_CONTRACT)
    if result["status"] != "PASS":
        summary = "; ".join(str(reason["message"]) for reason in result["reasons"])
        raise ContractRecoveryError(f"frozen scientific contract mismatch: {summary}")
    m150_one_m = m150.SCALE_SPECS["1m"]
    frozen_one_m = FROZEN_CONTRACT["scales"]["1m"]
    if m150_one_m["expected_parameters"] != frozen_one_m["parameters"]:
        raise ContractRecoveryError("M150 1M parameter authority diverged from frozen contract")
    if m150_one_m["expected_model_spec_sha256"] != frozen_one_m["model_spec_sha256"]:
        raise ContractRecoveryError("M150 1M ModelSpec authority diverged from frozen contract")
    return result


def _optimized_token_ledger(corpus: Path, manifest: dict[str, Any], tok: ByteTokenizer) -> dict[int, int]:
    iterators = m100._train_iters(corpus, manifest, tok, 0)
    batches = {stratum: m100._batches(iterator) for stratum, iterator in iterators.items()}
    cumulative = 0
    ledger: dict[int, int] = {}
    checkpoints = set(r192.CHECKPOINT_STEPS)
    for index in range(r192.FINAL_STEP):
        stratum = m100.MIXTURE[index % len(m100.MIXTURE)]
        batch = next(batches[stratum])
        cumulative += int(batch["labels"][:, 1:].ne(-100).sum().item())
        step = index + 1
        if step in checkpoints:
            ledger[step] = cumulative
    return ledger


def preflight(repo: Path, source_sha: str, out: Path) -> dict[str, Any]:
    contract_result = require_frozen_contract()
    out.mkdir(parents=True, exist_ok=True)
    manifest, tok, eval_id = r192.common(repo, source_sha, out, True)
    if eval_id["identity_sha256"] != FROZEN_CONTRACT["evaluation_identity_sha256"]:
        raise ContractRecoveryError("rebuilt M150 evaluation identity differs from frozen contract")

    ledger = _optimized_token_ledger(out / "corpus-a", manifest, tok)
    if ledger != r192.EXPECTED_TOKEN_BUDGETS:
        raise ContractRecoveryError(
            f"optimized-token ledger mismatch: frozen {r192.EXPECTED_TOKEN_BUDGETS!r} != rebuilt {ledger!r}"
        )

    constructions = []
    init = r192.init_spec()
    for scale in ("1m", "3m", "10m"):
        spec = r192.spec_for(scale)
        torch.manual_seed(0)
        model = TwelveSixDecoder(spec, init)
        constructed = sum(parameter.numel() for parameter in model.parameters())
        if constructed != spec.parameter_count():
            raise ContractRecoveryError(
                f"{scale} constructor parameter mismatch: spec {spec.parameter_count()} != module {constructed}"
            )
        constructions.append({
            "scale": scale,
            "parameter_count": constructed,
            "model_spec_sha256": spec.identity_sha256(),
            "construction_only": True,
            "forward_executed": False,
            "optimizer_created": False,
            "training_executed": False,
        })
        del model

    report = {
        "schema": SCHEMA,
        "worker_id": WORKER_ID,
        "status": "PASS",
        "source_sha": source_sha,
        "frozen_contract_identity_sha256": hash_json(FROZEN_CONTRACT),
        "resolved_contract_identity_sha256": contract_result["resolved_identity_sha256"],
        "corpus_identity_sha256": manifest["corpus_identity_sha256"],
        "evaluation_identity_sha256": eval_id["identity_sha256"],
        "optimized_token_ledger": {str(k): v for k, v in ledger.items()},
        "model_construction": constructions,
        "source_exposure_fraction_final": ledger[r192.FINAL_STEP] / int(manifest["by_split"]["train"]["byte_tokens"]),
        "historical_failure": {
            "run_id": HISTORICAL_RUN_ID,
            "head_sha": HISTORICAL_HEAD_SHA,
            "classification": "STALE_TEST_AND_DOCUMENT_CONTRACT_AFTER_INTENTIONAL_SOURCE_CONVERGENCE",
            "training_started": False,
            "comparison_started": False,
        },
        "truth_boundary": {
            "model_result_claims": False,
            "forward_executed": False,
            "optimizer_updates": 0,
            "training_matrix_executed": False,
            "paid_compute": False,
            "foreign_pretrained_weights": False,
        },
    }
    report["identity_sha256"] = hash_json(report)
    (out / "contract-preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-static")
    q = sub.add_parser("diagnose-historical")
    q.add_argument("--json", action="store_true")
    q = sub.add_parser("preflight")
    q.add_argument("--repo", type=Path, default=Path("."))
    q.add_argument("--source-sha", required=True)
    q.add_argument("--out", type=Path, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "validate-static":
        result = require_frozen_contract()
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "diagnose-historical":
        result = diagnose_contract(HISTORICAL_RUN_32941405721_CONTRACT)
        if args.json:
            print(json.dumps(result, sort_keys=True))
        else:
            for reason in result["reasons"]:
                print(f"{reason['code']}: {reason['message']}")
        return 0 if result["status"] == "FAIL" and result["reasons"] else 1
    if args.command == "preflight":
        result = preflight(args.repo, args.source_sha, args.out)
        print(json.dumps({"status": result["status"], "identity_sha256": result["identity_sha256"]}, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
