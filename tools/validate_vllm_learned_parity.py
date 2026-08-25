from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.inference.parity import compare_backends
from twelve_six.inference.sampling import greedy_token
from twelve_six.inference.vllm_native_llama import (
    VllmNativeLlamaBackend,
    verify_vllm_llama_directory,
)
from twelve_six.model import ModelSpec

SCHEMA = "12-6.vllm-native-llama-learned-parity.v1"
DEFAULT_PROBES = (
    "Hello from 12-6.",
    "Привіт від 12-6.",
    "def square(x): return x * x",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_value(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _finite_logits(values: Sequence[float], *, side: str) -> list[float]:
    result: list[float] = []
    for index, value in enumerate(values):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{side} raw logit {index} is not finite")
        result.append(number)
    if not result:
        raise ValueError(f"{side} raw logits are empty")
    return result


def _context_probe(model_dir: Path) -> str:
    binding = verify_vllm_llama_directory(model_dir)
    spec = ModelSpec.from_dict(dict(binding["source_model_spec"]))
    if spec.max_seq_len < 2:
        raise ValueError("ModelSpec max_seq_len must be at least 2 for context parity")
    return "x" * (spec.max_seq_len - 1)


def _collect_raw_trace(reference, candidate, prompts: Sequence[str], max_new_tokens: int):
    traces: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        reference_prompt = list(reference.encode(prompt))
        candidate_prompt = list(candidate.encode(prompt))
        if reference_prompt != candidate_prompt:
            raise ValueError(f"prompt {prompt_index} canonical token IDs differ")
        if not reference_prompt:
            raise ValueError(f"prompt {prompt_index} encoded to zero tokens")
        if len(reference_prompt) > reference.max_context_tokens:
            raise ValueError(f"prompt {prompt_index} exceeds context")

        history = list(reference_prompt)
        reference_generated: list[int] = []
        vllm_generated: list[int] = []
        steps: list[dict[str, Any]] = []
        for step_index in range(max_new_tokens):
            if len(history) >= reference.max_context_tokens:
                break
            reference_logits = _finite_logits(
                reference.next_token_logits(history), side="reference"
            )
            vllm_logits = _finite_logits(candidate.next_token_logits(history), side="vllm")
            if len(reference_logits) != len(vllm_logits):
                raise ValueError("raw-logit vocabulary size mismatch during trace")
            reference_token = int(greedy_token(reference_logits))
            vllm_token = int(greedy_token(vllm_logits))
            steps.append(
                {
                    "step_index": step_index,
                    "context_tokens": len(history),
                    "reference_raw_logits": reference_logits,
                    "vllm_raw_logits": vllm_logits,
                    "reference_raw_logits_sha256": _sha256_value(reference_logits),
                    "vllm_raw_logits_sha256": _sha256_value(vllm_logits),
                    "reference_greedy_token": reference_token,
                    "vllm_greedy_token": vllm_token,
                }
            )
            reference_generated.append(reference_token)
            vllm_generated.append(vllm_token)
            if reference_token != vllm_token:
                break
            history.append(reference_token)

        reference_decoded = reference.decode(reference_generated)
        vllm_decoded = candidate.decode(vllm_generated)
        traces.append(
            {
                "prompt_index": prompt_index,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_token_ids": reference_prompt,
                "prompt_tokens": len(reference_prompt),
                "reference_greedy_token_sequence": reference_generated,
                "vllm_greedy_token_sequence": vllm_generated,
                "reference_decoded_continuation": reference_decoded,
                "vllm_decoded_continuation": vllm_decoded,
                "stopped_at_context_boundary": len(history) >= reference.max_context_tokens,
                "steps": steps,
            }
        )
    return traces


def collect(args: argparse.Namespace) -> dict[str, Any]:
    prompts = list(args.prompt or DEFAULT_PROBES)
    boundary = _context_probe(args.model_dir)
    if boundary not in prompts:
        prompts.append(boundary)

    reference = load_first_party_backend(args.checkpoint)
    binding = verify_vllm_llama_directory(args.model_dir)
    diagnostics = reference.diagnostics()
    if binding["checkpoint_id"] != diagnostics["checkpoint_id"]:
        raise ValueError("standard-Llama export checkpoint ID does not match first-party checkpoint")
    if binding["model_spec_sha256"] != diagnostics["model_spec_sha256"]:
        raise ValueError("standard-Llama export ModelSpec does not match first-party checkpoint")

    candidate = VllmNativeLlamaBackend(
        args.model_dir,
        tokenizer=reference.tokenizer,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    report = compare_backends(
        reference,
        candidate,
        tuple(prompts),
        max_new_tokens=args.max_new_tokens,
        atol=args.atol,
        rtol=args.rtol,
    )

    raw_trace: list[dict[str, Any]] = []
    if report.passed:
        raw_trace = _collect_raw_trace(reference, candidate, prompts, args.max_new_tokens)

    trace_passed = bool(raw_trace) and all(
        trace["reference_greedy_token_sequence"] == trace["vllm_greedy_token_sequence"]
        and trace["reference_decoded_continuation"] == trace["vllm_decoded_continuation"]
        for trace in raw_trace
    )
    evidence: dict[str, Any] = {
        "schema": SCHEMA,
        "passed": bool(report.passed and trace_passed),
        "checkpoint": {
            **diagnostics,
            "model_spec": reference.model.spec.to_dict(),
        },
        "learned_source_artifact": {
            "artifact_id": args.source_artifact_id,
            "artifact_digest": args.source_artifact_digest,
            "head_sha": args.source_artifact_head_sha,
        },
        "standard_llama_export": {
            "source_export": binding["source_export"],
            "runtime_config_sha256": binding["target"]["config_sha256"],
            "runtime_weights_sha256": binding["target"]["weights_sha256"],
            "architecture": binding["target"]["architecture"],
            "model_type": binding["target"]["model_type"],
            "qk_rope_basis_conversion": "INCUMBENT_D07_EXACT",
        },
        "tokenizer": {
            "owner": "12-6_CANONICAL_OUTSIDE_VLLM",
            "id": reference.tokenizer.identity.version,
            "config_sha256": reference.tokenizer.identity.config_sha256,
            "vocab_sha256": reference.tokenizer.identity.vocab_sha256,
            "vocab_size": reference.tokenizer.vocab_size,
            "vllm_tokenizer_initialized": False,
            "chat_template_used": False,
        },
        "vllm": {
            "version": candidate.vllm_version,
            "implementation": "BUILTIN_LLAMA",
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "logprobs_mode": "raw_logits",
            "max_logprobs": -1,
        },
        "parity": report.to_dict(),
        "raw_logit_and_greedy_trace": raw_trace,
        "context_behavior": {
            "max_context_tokens": reference.max_context_tokens,
            "near_limit_probe_tokens": reference.max_context_tokens - 1,
            "near_limit_probe_executed": any(
                trace["prompt_tokens"] == reference.max_context_tokens - 1 for trace in raw_trace
            ),
            "over_context_rejected_before_vllm": True,
        },
        "tolerance_basis": (
            "Q/K RoPE basis conversion is exact. FP32 atol/rtol remain 1e-5 by default only "
            "for maintained kernel reduction-order differences; tolerance is not widened to pass."
        ),
        "foreign_pretrained_weights_used": False,
        "chat_template_used": False,
        "paid_compute": False,
    }
    evidence["evidence_sha256"] = _sha256_value(evidence)
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute learned-checkpoint parity through the incumbent native-vLLM Llama adapter "
            "and retain exact raw-logit vectors and greedy token sequences."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-artifact-id", type=int, required=True)
    parser.add_argument("--source-artifact-digest", required=True)
    parser.add_argument("--source-artifact-head-sha", required=True)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evidence = collect(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(evidence, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        print(args.output)
        return 0 if evidence["passed"] else 1
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
