"""RECOVER-175 execution bridge: frozen EVAL-132 over the verified M150 Base ladder."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from twelve_six import evaluation_ua_v1 as ua
from twelve_six import milestone150_learned_base_ladder as m150
from twelve_six.checkpoint import hash_json, verify_checkpoint
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.model import TwelveSixDecoder

SCHEMA = "12-6.learned-base-ladder-v1.eval132-ua-convergence.v1"
DIAGNOSTIC_SCHEMA = "12-6.eval132-ua-ladder-diagnostic.v1"
AUTHORITY = "LOCAL_FREE_RAW_BASE_DIAGNOSTIC_NOT_STAGE_PROMOTION"
DEFAULT_M150_SHA = "8344085ddd0b52e4b698c3344a1d0482153525dc"
EXPECTED_EVAL132_BLOBS = {
    "data/evaluation/benchmark_registry.json": "170d80f90b871481ceae6b3d651656130c1892f7",
    "data/evaluation/ua_raw_base_v1/manifest.json": "2db0bf5c6f93838852b8a730584946986fed530e",
    "data/evaluation/ua_raw_base_v1/source_rows.json": "068c5b515cfafef1f5dbd1ab8b44348e3e7e00f9",
    "data/external/reserved_fingerprints.json": "19e4dbebd8a34c165d6c20eb23f0ca79cbd02b53",
    "src/twelve_six/evaluation_ua_v1.py": "0f87477a9382281e63ee64ea85c07b7ce194d118",
    "tests/test_eval132_ua_suite.py": "5aef96e4d602adc0c41717610b47fc73f39e554e",
}


class Recover175Error(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Recover175Error(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _self_hashed(value: dict[str, Any]) -> dict[str, Any]:
    out = dict(value)
    out["report_sha256"] = hash_json(out)
    return out


def _verify_eval_registration(repo: Path) -> dict[str, Any]:
    reserved = ua.validate_reserved_registry(repo / "data/external/reserved_fingerprints.json")
    registry = _read_json(repo / "data/evaluation/benchmark_registry.json")
    if registry.get("manifest_sha256") != ua.D06_REGISTRY_SHA256:
        raise Recover175Error("D06 EVAL-132 benchmark registry identity mismatch")
    matching = [row for row in registry.get("benchmarks", []) if row.get("benchmark_id") == "eval132-ua-raw-base"]
    if len(matching) != 1:
        raise Recover175Error("EVAL-132 D06 registration missing or duplicated")
    entry = matching[0]
    if entry.get("source_id") != ua.SOURCE_ID or entry.get("held_out") is not True:
        raise Recover175Error("EVAL-132 D06 held-out/source binding mismatch")
    if entry.get("allowed_uses") != ["evaluation"]:
        raise Recover175Error("EVAL-132 D06 use policy is not evaluation-only")
    manifest = _read_json(repo / "data/evaluation/ua_raw_base_v1/manifest.json")
    contamination = manifest.get("contamination", {})
    interpretation = manifest.get("interpretation", {})
    task = manifest.get("task", {})
    if manifest.get("dataset_sha256") != ua.DATASET_SHA256 or manifest.get("item_count") != 216:
        raise Recover175Error("EVAL-132 diagnostic manifest identity mismatch")
    if contamination.get("held_out") is not True or contamination.get("future_training_allowed") is not False:
        raise Recover175Error("EVAL-132 manifest is not held-out/evaluation-only")
    if contamination.get("reserved_variant_count") != 432:
        raise Recover175Error("EVAL-132 reserved count mismatch")
    if interpretation.get("proficiency_claim_authorized") is not False:
        raise Recover175Error("EVAL-132 proficiency boundary weakened")
    if task.get("instruction_following") is not False:
        raise Recover175Error("EVAL-132 instruction-following boundary weakened")
    return {
        "status": "PASS",
        "dataset_id": ua.DATASET_ID,
        "dataset_sha256": ua.DATASET_SHA256,
        "source_id": ua.SOURCE_ID,
        "source_identity_sha256": ua.SOURCE_IDENTITY_SHA256,
        "d06_registry_sha256": ua.D06_REGISTRY_SHA256,
        "reserved_registry_identity_sha256": reserved["registry_identity_sha256"],
        "reserved_variant_count": 432,
        "held_out": True,
        "allowed_uses": ["evaluation"],
        "future_training_allowed": False,
    }


def _normalized_likelihood_diagnostics(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(evaluation.get("items", []))
    if len(rows) != 216:
        raise Recover175Error("EVAL-132 scorer did not return 216 item rows")
    result: dict[str, Any] = {}
    for side in ("preferred", "contrast"):
        total_nats = math.fsum(float(row[side]["logprob_nats"]) for row in rows)
        source_bytes = sum(int(row[side]["source_bytes"]) for row in rows)
        byte_tokens = sum(int(row[side]["byte_tokens"]) for row in rows)
        if source_bytes <= 0 or byte_tokens != source_bytes:
            raise Recover175Error("byte-tokenizer source-byte normalization invariant failed")
        result[side] = {
            "joint_logprob_nats": total_nats,
            "source_bytes": source_bytes,
            "byte_tokens": byte_tokens,
            "tokens_per_source_byte": byte_tokens / source_bytes,
            "conditional_bpb": -total_nats / (math.log(2.0) * source_bytes),
        }
    result["pair_margin"] = dict(evaluation["overall"])
    result["tokenizer_length_artifact"] = {
        "tokenizer": "s0-byte-v1",
        "byte_tokenizer_native": True,
        "tokens_per_source_byte": 1.0,
        "inter_scale_tokenizer_identity_constant": True,
        "length_artifact_confounded": False,
        "interpretation": "All compared rungs use the same byte tokenizer; inter-scale deltas are not attributable to tokenizer token-length differences.",
    }
    return result


def _slim_eval(evaluation: dict[str, Any]) -> dict[str, Any]:
    out = dict(evaluation)
    out.pop("items", None)
    out["tokenizer_normalized"] = _normalized_likelihood_diagnostics(evaluation)
    return out


def _random_init(scale: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    spec = m150.model_spec(scale)
    init = m150.init_spec()
    torch.manual_seed(m150.SEED)
    model = TwelveSixDecoder(spec, init)
    source = {
        "kind": "architecture_matched_random_init",
        "scale": scale,
        "seed": m150.SEED,
        "model_spec_sha256": spec.identity_sha256(),
        "init_spec_sha256": init.identity_sha256(),
        "foreign_pretrained_weights": False,
    }
    return _slim_eval(ua.evaluate_model(model, label=f"random-init-{scale}", source=source, items=items, include_item_rows=True))


def _learned(scale: str, checkpoint: Path, scale_report: Mapping[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = verify_checkpoint(checkpoint)
    backend = load_first_party_backend(checkpoint)
    diagnostics = backend.diagnostics()
    spec = m150.model_spec(scale)
    if diagnostics["model_spec_sha256"] != spec.identity_sha256():
        raise Recover175Error(f"{scale} learned checkpoint ModelSpec mismatch")
    if int(diagnostics["parameter_count"]) != spec.parameter_count():
        raise Recover175Error(f"{scale} learned checkpoint parameter count mismatch")
    if diagnostics["git_sha"] != DEFAULT_M150_SHA:
        raise Recover175Error(f"{scale} learned checkpoint source SHA mismatch")
    checkpoints = scale_report.get("checkpoints", {})
    expected_id = checkpoints.get("best_checkpoint_id")
    if expected_id and diagnostics["checkpoint_id"] != expected_id:
        raise Recover175Error(f"{scale} retained best checkpoint identity mismatch")
    if scale_report.get("fresh_verification", {}).get("status") != "PASS":
        raise Recover175Error(f"{scale} upstream M150 fresh verification not PASS")
    source = {
        "kind": "learned_retained_best_checkpoint",
        "scale": scale,
        "checkpoint_id": diagnostics["checkpoint_id"],
        "checkpoint_step": diagnostics["step"],
        "optimized_tokens": diagnostics["tokens_seen"],
        "git_sha": diagnostics["git_sha"],
        "model_spec_sha256": diagnostics["model_spec_sha256"],
        "parameter_count": diagnostics["parameter_count"],
        "tokenizer_version": diagnostics["tokenizer_version"],
        "tokenizer_config_sha256": diagnostics["tokenizer_config_sha256"],
        "tokenizer_vocab_sha256": diagnostics["tokenizer_vocab_sha256"],
        "dataset_manifest_sha256": diagnostics["dataset_manifest_sha256"],
        "checkpoint_verified": True,
        "manifest_checkpoint_id": manifest["checkpoint_id"],
    }
    return _slim_eval(ua.evaluate_model(backend.model, label=f"learned-best-{scale}", source=source, items=items, include_item_rows=True))


def _delta(after: Mapping[str, Any], before: Mapping[str, Any]) -> dict[str, float]:
    a = after["overall"]
    b = before["overall"]
    an = after["tokenizer_normalized"]
    bn = before["tokenizer_normalized"]
    return {
        "accuracy": float(a["accuracy"]) - float(b["accuracy"]),
        "mean_margin_nats_per_source_byte": float(a["mean_margin_nats_per_source_byte"]) - float(b["mean_margin_nats_per_source_byte"]),
        "median_margin_nats_per_source_byte": float(a["median_margin_nats_per_source_byte"]) - float(b["median_margin_nats_per_source_byte"]),
        "preferred_conditional_bpb": float(an["preferred"]["conditional_bpb"]) - float(bn["preferred"]["conditional_bpb"]),
        "contrast_conditional_bpb": float(an["contrast"]["conditional_bpb"]) - float(bn["contrast"]["conditional_bpb"]),
    }


def _phenomenon_deltas(after: Mapping[str, Any], before: Mapping[str, Any]) -> dict[str, Any]:
    return {
        phenomenon: {
            "accuracy": float(after["by_phenomenon"][phenomenon]["accuracy"]) - float(before["by_phenomenon"][phenomenon]["accuracy"]),
            "mean_margin_nats_per_source_byte": float(after["by_phenomenon"][phenomenon]["mean_margin_nats_per_source_byte"]) - float(before["by_phenomenon"][phenomenon]["mean_margin_nats_per_source_byte"]),
        }
        for phenomenon in ua.PHENOMENA
    }


def execute(repo: Path, m150_evidence: Path, output: Path, expected_m150_sha: str = DEFAULT_M150_SHA) -> dict[str, Any]:
    if expected_m150_sha != DEFAULT_M150_SHA:
        raise Recover175Error("RECOVER-175 is bound to the verified M150 source SHA")
    registration = _verify_eval_registration(repo)
    items = ua.generate_items(repo / "data/evaluation/ua_raw_base_v1/source_rows.json")
    ladder_path = m150_evidence / "ladder-report.json"
    base_ladder = m150.validate_ladder(ladder_path, expected_m150_sha)
    if base_ladder.get("ten_million", {}).get("included_in_rankings") is not False:
        raise Recover175Error("unexpected 10M inclusion in M150 V1")

    scales: dict[str, Any] = {}
    for scale in m150.SCALE_ORDER:
        scale_report = base_ladder["scales"][scale]
        random_eval = _random_init(scale, items)
        learned_eval = _learned(scale, m150_evidence / "retained" / scale / "best", scale_report, items)
        scales[scale] = {
            "model_spec_sha256": m150.model_spec(scale).identity_sha256(),
            "parameter_count": m150.model_spec(scale).parameter_count(),
            "random_init": random_eval,
            "learned_best": learned_eval,
            "learned_minus_random_init": _delta(learned_eval, random_eval),
            "phenomenon_deltas_learned_minus_random": _phenomenon_deltas(learned_eval, random_eval),
        }

    adjacent: dict[str, Any] = {}
    for left, right in zip(m150.SCALE_ORDER, m150.SCALE_ORDER[1:]):
        adjacent[f"{left}_to_{right}"] = _delta(scales[right]["learned_best"], scales[left]["learned_best"])

    diagnostic = _self_hashed({
        "schema": DIAGNOSTIC_SCHEMA,
        "authority": AUTHORITY,
        "suite_registration": registration,
        "suite": {
            "dataset_id": ua.DATASET_ID,
            "dataset_sha256": ua.DATASET_SHA256,
            "item_count": len(items),
            "phenomena": list(ua.PHENOMENA),
            "items_per_phenomenon": 24,
            "scoring": "conditional_mean_logprob_per_utf8_byte",
        },
        "common_truth": {
            "m150_source_sha": expected_m150_sha,
            "m150_report_sha256": base_ladder["report_sha256"],
            "corpus_identity_sha256": base_ladder["truth_model"]["corpus_identity_sha256"],
            "tokenizer": base_ladder["truth_model"]["tokenizer"],
            "evaluation_non_mutation_required": True,
            "first_party_checkpoint_loader": True,
        },
        "scales": scales,
        "adjacent_learned_scale_deltas": adjacent,
        "tokenizer_length_artifact_separation": {
            "status": "PASS",
            "tokenizer_identity_common_across_rungs": True,
            "tokens_per_source_byte": 1.0,
            "note": "Reported inter-scale likelihood deltas are byte-normalized and are not caused by tokenizer token-length differences.",
        },
        "ten_million": {
            "status": "INCOMPLETE_NO_VERIFIED_LEARNED_CHECKPOINT",
            "evaluated": False,
            "reason": "SCALE-141 failed contract tests before phase-1 training; no learned 10M checkpoint was produced by that run.",
        },
        "claims": {
            "broad_ukrainian_proficiency": False,
            "instruction_following": False,
            "alignment": False,
            "production_readiness": False,
            "intelligence": False,
            "foreign_pretrained_weights": False,
            "sft": False,
            "rlhf": False,
            "dpo": False,
            "paid_compute": False,
        },
    })
    _write_json(output / "eval132-ua-report.json", diagnostic)

    convergence = _self_hashed({
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "base_ladder": base_ladder,
        "ukrainian_raw_lm_diagnostic": diagnostic,
        "rankings": base_ladder["rankings"],
        "minimum_comparable_ladder_complete": base_ladder["minimum_comparable_ladder_complete"],
        "ten_million": diagnostic["ten_million"],
        "unsupported_claims_absent": [
            "intelligence", "production_readiness", "alignment", "instruction_following", "broad_ukrainian_proficiency"
        ],
    })
    _write_json(output / "learned-base-ladder-v1.json", convergence)
    return convergence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--m150-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-m150-sha", default=DEFAULT_M150_SHA)
    args = parser.parse_args()
    report = execute(args.repo_root.resolve(), args.m150_evidence.resolve(), args.output.resolve(), args.expected_m150_sha)
    diag = report["ukrainian_raw_lm_diagnostic"]
    summary = {
        "validation": "PASS",
        "report_sha256": report["report_sha256"],
        "m150_report_sha256": report["base_ladder"]["report_sha256"],
        "ten_million": report["ten_million"]["status"],
        "scales": {
            scale: {
                "accuracy": diag["scales"][scale]["learned_best"]["overall"]["accuracy"],
                "mean_margin": diag["scales"][scale]["learned_best"]["overall"]["mean_margin_nats_per_source_byte"],
                "preferred_conditional_bpb": diag["scales"][scale]["learned_best"]["tokenizer_normalized"]["preferred"]["conditional_bpb"],
            }
            for scale in m150.SCALE_ORDER
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
