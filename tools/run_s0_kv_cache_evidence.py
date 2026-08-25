from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch

from twelve_six.inference.sampling import greedy_token
from twelve_six.integration.s0_runtime import S0TorchInferenceBackend
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer

SCHEMA = "12-6.s0-kv-cache-evidence.v1"
REPOSITORY = "Oleksii-debug/12-6-ai."


def _canonical_hash(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _git_head(repo_root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("KV-cache evidence requires a Git checkout") from exc
    return value


def _validate_source_sha(value: str) -> None:
    if len(value) not in {40, 64} or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("source SHA must be a full lowercase Git object id")


def collect_evidence(
    repo_root: Path,
    *,
    source_sha: str,
    steps: int = 16,
    seed: int = 20260825,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    _validate_source_sha(source_sha)
    if _git_head(repo_root) != source_sha:
        raise ValueError("source SHA does not equal checkout HEAD")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 1:
        raise ValueError("steps must be an integer greater than one")

    stage = load_stage_config(repo_root / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    backend = S0TorchInferenceBackend(model, tokenizer)
    prompt_ids = tokenizer.encode("12-6")

    max_abs_error = 0.0
    exact_token_choices = True
    prefix = list(prompt_ids)
    session = backend.begin_generation(prompt_ids)
    try:
        initial_cache_bytes = session.cache_bytes
        for step_index in range(steps):
            cached_logits = [float(value) for value in session.next_token_logits()]
            stateless_logits = [float(value) for value in backend.next_token_logits(prefix)]
            if len(cached_logits) != len(stateless_logits):
                raise RuntimeError("cached/stateless logit vocabulary size mismatch")
            step_error = max(
                abs(cached - stateless)
                for cached, stateless in zip(cached_logits, stateless_logits, strict=True)
            )
            if not math.isfinite(step_error):
                raise FloatingPointError("non-finite cached/stateless logit error")
            max_abs_error = max(max_abs_error, step_error)
            cached_token = greedy_token(cached_logits)
            stateless_token = greedy_token(stateless_logits)
            if cached_token != stateless_token:
                exact_token_choices = False
                raise RuntimeError("KV-cache greedy token diverged from stateless full-prefix decode")
            prefix.append(stateless_token)
            if step_index + 1 < steps:
                session.append(stateless_token)

        cached_token_work = session.tokens_processed
        stateless_token_work = sum(len(prompt_ids) + step for step in range(steps))
        final_cache_bytes = session.cache_bytes
        final_cache_sequence_length = session.sequence_length
    finally:
        session.close()

    if max_abs_error > 1e-6:
        raise RuntimeError(f"KV-cache logits exceeded tolerance: {max_abs_error:.12g}")
    if cached_token_work >= stateless_token_work:
        raise RuntimeError("KV-cache did not reduce decoder input-position work")

    report: dict[str, object] = {
        "schema": SCHEMA,
        "authority": "LOCAL_FREE_INFERENCE_MECHANICS_NOT_PROMOTION",
        "repository": REPOSITORY,
        "source_sha": source_sha,
        "stage": "S0",
        "canonical_base": stage.canonical_base,
        "model_spec_sha256": stage.model.identity_sha256(),
        "init_spec_sha256": stage.init.identity_sha256(),
        "parameter_count": stage.expected_parameters,
        "seed": seed,
        "prompt_tokens": len(prompt_ids),
        "decode_logits_steps": steps,
        "max_abs_logit_error": max_abs_error,
        "logit_tolerance": 1e-6,
        "greedy_token_choices_exact": exact_token_choices,
        "decoder_input_positions": {
            "cached": cached_token_work,
            "stateless_full_prefix": stateless_token_work,
            "reduction_fraction": 1.0 - (cached_token_work / stateless_token_work),
        },
        "cache": {
            "layers": stage.model.n_layers,
            "stored_kv_heads_per_layer": stage.model.n_kv_heads,
            "head_dim": stage.model.head_dim,
            "initial_bytes": initial_cache_bytes,
            "final_bytes": final_cache_bytes,
            "final_sequence_length": final_cache_sequence_length,
            "ephemeral_not_checkpointed": True,
        },
        "truth_boundary": {
            "local_free_cpu": True,
            "runtime_latency_benchmark": False,
            "gpu_or_distributed": False,
            "paid_compute": False,
            "promotion_authority": False,
            "base_behavior_changed": False,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect exact S0 KV-cache inference evidence.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {args.output}")
    report = collect_evidence(
        args.repo_root,
        source_sha=args.source_sha,
        steps=args.steps,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "kv-cache-evidence: PASS "
        f"max_abs_error={report['max_abs_logit_error']} "
        f"cached_positions={report['decoder_input_positions']['cached']} "
        f"stateless_positions={report['decoder_input_positions']['stateless_full_prefix']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
