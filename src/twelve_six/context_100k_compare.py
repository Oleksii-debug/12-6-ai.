"""Combine MODEL-17 128/256 candidate evidence and make a bounded recommendation."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import torch

from .checkpoint import detect_git_sha
from .context_100k_candidate import TARGET_OPTIMIZED_TOKENS, shared_trainer_config

SCHEMA = "12-6.model17-context-100k.v1"
AUTHORITY = "LOCAL_FREE_CONTEXT_RESEARCH_NOT_S0_IDENTITY_CHANGE"


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def combine(repo_root: Path, source_sha: str, c128_path: Path, c256_path: Path, output: Path):
    if detect_git_sha(repo_root) != source_sha:
        raise RuntimeError("exact-checkout mismatch")
    c128 = json.loads(c128_path.read_text(encoding="utf-8"))
    c256 = json.loads(c256_path.read_text(encoding="utf-8"))
    if (c128["context"], c256["context"]) != (128, 256):
        raise ValueError("candidate inputs must be 128 then 256")
    for candidate in (c128, c256):
        if candidate["source_sha"] != source_sha or candidate["runtime"]["paid_compute"] is not False:
            raise ValueError("candidate provenance mismatch")
        if candidate["training"]["optimized_tokens"] != TARGET_OPTIMIZED_TOKENS:
            raise ValueError("candidate token budget mismatch")
    if c128["trainable_parameters"] != c256["trainable_parameters"] != 95_568:
        raise ValueError("parameter mismatch")
    if c128["trainable_parameters"] != 95_568:
        raise ValueError("unexpected research parameter count")
    if c128["initial_weight_digest_sha256"] != c256["initial_weight_digest_sha256"]:
        raise ValueError("initial trainable tensors differ")
    if c128["controls"]["optimizer"] != c256["controls"]["optimizer"]:
        raise ValueError("optimizer control differs")
    if c128["controls"]["dataset_manifest_sha256"] != c256["controls"]["dataset_manifest_sha256"]:
        raise ValueError("corpus control differs")
    if c128["controls"]["tokenizer_config_sha256"] != c256["controls"]["tokenizer_config_sha256"]:
        raise ValueError("tokenizer control differs")

    native_delta = c256["held_out"]["final_native"]["bpb"] - c128["held_out"]["final_native"]["bpb"]
    long_history_gain = c256["held_out"]["final_common_128"]["bpb"] - c256["held_out"]["final_native"]["bpb"]
    common128_regime_delta = c256["held_out"]["final_common_128"]["bpb"] - c128["held_out"]["final_common_128"]["bpb"]
    compute_ratio = c256["training"]["seconds_per_optimized_token"] / c128["training"]["seconds_per_optimized_token"]
    util128 = c128["packing"]["train"]["causal_token_utilization"]
    util256 = c256["packing"]["train"]["causal_token_utilization"]

    promote = native_delta < 0.0 and long_history_gain > 0.0
    recommendation = "256" if promote else "128"
    rationale = (
        "Promote 256: it lowers native held-out BPB at equal optimized tokens and the same 256-trained weights gain when evaluation can use history beyond 128."
        if promote else
        "Keep 128: 256 did not satisfy both required signals—lower native held-out BPB at equal optimized tokens and a positive within-model gain from history beyond 128."
    )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {"repository": "Oleksii-debug/12-6-ai.", "git_sha": source_sha, "context_framework": "MODEL-36 ContextPackingSpec/context_scaling"},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "paid_compute": False, "candidate_process_isolation": True},
        "controls": {
            "trainable_parameters": 95_568,
            "optimized_tokens_per_condition": TARGET_OPTIMIZED_TOKENS,
            "optimizer": c128["controls"]["optimizer"],
            "init_spec": c128["controls"]["init_spec"],
            "seed": c128["controls"]["seed"],
            "tokenizer_config_sha256": c128["controls"]["tokenizer_config_sha256"],
            "tokenizer_vocab_sha256": c128["controls"]["tokenizer_vocab_sha256"],
            "dataset_manifest_sha256": c128["controls"]["dataset_manifest_sha256"],
            "initial_weight_digest_sha256": c128["initial_weight_digest_sha256"],
            "only_modelspec_difference": "max_seq_len 128 vs 256",
            "canonical_s0_packing_identity_modified": False,
        },
        "conditions": [c128, c256],
        "separation": {
            "longer_dependency_access": {
                "bpb_gain_from_allowing_256_history": long_history_gain,
                "definition": "256-trained common-128 held-out BPB minus the same model's native-256 BPB; positive means >128 history helps",
            },
            "packing_efficiency": {
                "train_causal_utilization_128": util128,
                "train_causal_utilization_256": util256,
                "utilization_256_minus_128": util256 - util128,
                "padding_token_slots_128": c128["packing"]["train"]["padding_token_slots"],
                "padding_token_slots_256": c256["packing"]["train"]["padding_token_slots"],
                "tail_pair_waste_128": c128["packing"]["train"]["tail_pair_waste"],
                "tail_pair_waste_256": c256["packing"]["train"]["tail_pair_waste"],
                "documents_exceeding_context_128": c128["packing"]["train"]["documents_exceeding_context"],
                "documents_exceeding_context_256": c256["packing"]["train"]["documents_exceeding_context"],
                "documents_hard_truncated_128": c128["packing"]["train"]["documents_hard_truncated"],
                "documents_hard_truncated_256": c256["packing"]["train"]["documents_hard_truncated"],
            },
            "compute_per_token": {
                "seconds_per_token_128": c128["training"]["seconds_per_optimized_token"],
                "seconds_per_token_256": c256["training"]["seconds_per_optimized_token"],
                "ratio_256_over_128": compute_ratio,
                "peak_rss_bytes_128": c128["training"]["peak_rss_bytes"],
                "peak_rss_bytes_256": c256["training"]["peak_rss_bytes"],
            },
            "training_regime_at_common_history": {
                "bpb_256trained_minus_128trained_at_context128": common128_regime_delta,
            },
            "overall_native_heldout": {
                "bpb_128": c128["held_out"]["final_native"]["bpb"],
                "bpb_256": c256["held_out"]["final_native"]["bpb"],
                "bpb_256_minus_128": native_delta,
            },
        },
        "recommendation": {
            "primary_100k_research_context": recommendation,
            "rationale": rationale,
            "preserve_s0_context_identity": True,
            "canonical_architecture_freeze": False,
            "truth_boundary": "One deterministic LOCAL_FREE seed on the tiny project-authored S0 fixture; this is a primary research-context decision, not a broad long-context capability claim.",
        },
    }
    report["report_sha256"] = _hash(report)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def validate(report: dict[str, Any], expected_source_sha: str | None = None) -> None:
    if report.get("schema") != SCHEMA or report.get("runtime", {}).get("paid_compute") is not False:
        raise ValueError("invalid report schema/runtime")
    if expected_source_sha and report["source"]["git_sha"] != expected_source_sha:
        raise ValueError("source SHA mismatch")
    conditions = report.get("conditions", [])
    if [item.get("context") for item in conditions] != [128, 256]:
        raise ValueError("condition order mismatch")
    if conditions[0]["training"]["optimized_tokens"] != conditions[1]["training"]["optimized_tokens"]:
        raise ValueError("optimized-token mismatch")
    if conditions[0]["initial_weight_digest_sha256"] != conditions[1]["initial_weight_digest_sha256"]:
        raise ValueError("initial-weight mismatch")
    if report["controls"]["canonical_s0_packing_identity_modified"] is not False:
        raise ValueError("S0 identity changed")
    claimed = report["report_sha256"]
    payload = dict(report); payload.pop("report_sha256")
    if claimed != _hash(payload):
        raise ValueError("report hash mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    combine_cmd = sub.add_parser("combine")
    combine_cmd.add_argument("--repo-root", type=Path, default=Path("."))
    combine_cmd.add_argument("--source-sha", required=True)
    combine_cmd.add_argument("--context-128", type=Path, required=True)
    combine_cmd.add_argument("--context-256", type=Path, required=True)
    combine_cmd.add_argument("--output", type=Path, required=True)
    validate_cmd = sub.add_parser("validate")
    validate_cmd.add_argument("report", type=Path)
    validate_cmd.add_argument("--expected-source-sha")
    args = parser.parse_args()
    if args.command == "combine":
        report = combine(args.repo_root.resolve(), args.source_sha, args.context_128, args.context_256, args.output)
        print(json.dumps({"separation": report["separation"], "recommendation": report["recommendation"], "report_sha256": report["report_sha256"]}, indent=2, sort_keys=True))
        return 0
    report = json.loads(args.report.read_text(encoding="utf-8"))
    validate(report, args.expected_source_sha)
    print("MODEL-17 context evidence validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
