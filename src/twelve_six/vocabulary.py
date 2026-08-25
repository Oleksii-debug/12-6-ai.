"""Vocabulary/parameter allocation analysis for non-frozen stage candidates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .model import ModelSpec, load_stage_config


class VocabularyAllocationError(ValueError):
    """Fail-closed vocabulary allocation input error."""


@dataclass(frozen=True, slots=True)
class VocabularyCost:
    vocab_size: int
    d_model: int
    tied_lm_head: bool
    token_embedding_parameters: int
    lm_head_weight_parameters: int
    lm_head_bias_parameters: int
    total_vocabulary_parameters: int


@dataclass(frozen=True, slots=True)
class RebalancedModel:
    model: ModelSpec
    parameter_count: int
    target_parameters: int
    target_delta: int
    vocabulary_parameters: int
    vocabulary_share: float


@dataclass(frozen=True, slots=True)
class TokenizerTradeoffPoint:
    algorithm: str
    requested_vocab_size: int
    actual_vocab_size: int
    held_out_tokens: int
    byte_baseline_tokens: int
    token_reduction_vs_bytes: float
    repeatable_artifact_identity: bool
    strict_round_trip: bool
    unknown_tokens: int
    vocabulary_parameters: int
    vocabulary_share_of_stage_target: float
    output_projection_work_ratio_vs_bytes: float
    source_sha: str
    evidence_sha256: str


def vocabulary_cost(
    spec: ModelSpec,
    *,
    vocab_size: int | None = None,
    tied_lm_head: bool | None = None,
) -> VocabularyCost:
    """Count the trainable parameter surface controlled directly by vocabulary size."""
    vocab = spec.vocab_size if vocab_size is None else vocab_size
    tied = spec.tie_word_embeddings if tied_lm_head is None else tied_lm_head
    if vocab <= 0:
        raise VocabularyAllocationError("vocab_size must be positive")
    embedding = vocab * spec.d_model
    head_weight = 0 if tied else embedding
    head_bias = vocab if spec.lm_head_bias else 0
    return VocabularyCost(
        vocab_size=vocab,
        d_model=spec.d_model,
        tied_lm_head=tied,
        token_embedding_parameters=embedding,
        lm_head_weight_parameters=head_weight,
        lm_head_bias_parameters=head_bias,
        total_vocabulary_parameters=embedding + head_weight + head_bias,
    )


def rebalance_d_ff_for_vocabulary(
    spec: ModelSpec,
    *,
    target_parameters: int,
    vocab_size: int,
    tied_lm_head: bool | None = None,
    d_ff_alignment: int = 8,
) -> RebalancedModel:
    """Retarget vocab and spend/recover the difference in d_ff near a parameter target."""
    if target_parameters <= 0:
        raise VocabularyAllocationError("target_parameters must be positive")
    if d_ff_alignment <= 0:
        raise VocabularyAllocationError("d_ff_alignment must be positive")
    tied = spec.tie_word_embeddings if tied_lm_head is None else tied_lm_head
    base = replace(spec, vocab_size=vocab_size, tie_word_embeddings=tied)

    # ModelSpec v1 total is affine in d_ff. MLP bias adds two d_ff biases per layer.
    slope_per_layer = 3 * base.d_model + (2 if base.mlp_bias else 0)
    slope = base.n_layers * slope_per_layer
    constant = base.parameter_count() - slope * base.d_ff
    ideal = (target_parameters - constant) / slope
    if ideal <= 0:
        raise VocabularyAllocationError(
            "vocabulary/non-FFN parameters already exhaust the requested parameter target"
        )

    lower = max(d_ff_alignment, int(ideal // d_ff_alignment) * d_ff_alignment)
    upper = lower + d_ff_alignment
    candidates = sorted({lower, upper})
    scored: list[tuple[tuple[int, bool, int], ModelSpec, int]] = []
    for d_ff in candidates:
        candidate = replace(base, d_ff=d_ff)
        total = candidate.parameter_count()
        # Prefer the closest candidate, then one not over target, then larger FFN capacity.
        key = (abs(total - target_parameters), total > target_parameters, -d_ff)
        scored.append((key, candidate, total))
    _, winner, total = min(scored, key=lambda item: item[0])
    cost = vocabulary_cost(winner)
    return RebalancedModel(
        model=winner,
        parameter_count=total,
        target_parameters=target_parameters,
        target_delta=total - target_parameters,
        vocabulary_parameters=cost.total_vocabulary_parameters,
        vocabulary_share=cost.total_vocabulary_parameters / total,
    )


def tradeoff_points_from_report(
    report: dict[str, Any],
    *,
    spec: ModelSpec,
    stage_target_parameters: int,
) -> list[TokenizerTradeoffPoint]:
    """Convert one real tokenizer report into parameter/token Pareto coordinates."""
    requested = int(report["requested_vocab_size"])
    source_sha = str(report["source"]["source_sha"])
    evidence_sha256 = str(report["evidence_sha256"])
    points: list[TokenizerTradeoffPoint] = []
    for algorithm, result in sorted(report["algorithms"].items()):
        held_out = result["held_out"]
        actual_vocab = int(result["artifact"]["vocab_size"])
        tokens = int(held_out["tokens"])
        byte_tokens = int(held_out["byte_baseline_tokens"])
        cost = vocabulary_cost(spec, vocab_size=actual_vocab)
        output_ratio = (tokens * actual_vocab) / (byte_tokens * 256)
        points.append(
            TokenizerTradeoffPoint(
                algorithm=str(algorithm),
                requested_vocab_size=requested,
                actual_vocab_size=actual_vocab,
                held_out_tokens=tokens,
                byte_baseline_tokens=byte_tokens,
                token_reduction_vs_bytes=float(held_out["token_reduction_vs_bytes"]),
                repeatable_artifact_identity=result["repeatability_status"] == "PASS",
                strict_round_trip=bool(held_out["strict_round_trip_all"]),
                unknown_tokens=int(held_out["unknown_tokens"]),
                vocabulary_parameters=cost.total_vocabulary_parameters,
                vocabulary_share_of_stage_target=(
                    cost.total_vocabulary_parameters / stage_target_parameters
                ),
                output_projection_work_ratio_vs_bytes=output_ratio,
                source_sha=source_sha,
                evidence_sha256=evidence_sha256,
            )
        )
    return points


def pareto_frontier(
    points: Iterable[TokenizerTradeoffPoint],
    *,
    require_repeatable: bool = True,
) -> list[TokenizerTradeoffPoint]:
    """Return non-dominated points minimizing vocabulary parameters and held-out tokens."""
    eligible = [
        point
        for point in points
        if point.strict_round_trip
        and point.unknown_tokens == 0
        and (point.repeatable_artifact_identity or not require_repeatable)
    ]
    frontier: list[TokenizerTradeoffPoint] = []
    for candidate in eligible:
        dominated = False
        for other in eligible:
            if other is candidate:
                continue
            no_worse = (
                other.vocabulary_parameters <= candidate.vocabulary_parameters
                and other.held_out_tokens <= candidate.held_out_tokens
            )
            strictly_better = (
                other.vocabulary_parameters < candidate.vocabulary_parameters
                or other.held_out_tokens < candidate.held_out_tokens
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda point: (point.vocabulary_parameters, point.held_out_tokens, point.algorithm),
    )


def build_matrix_evidence(
    *,
    stage_config_path: str | Path,
    report_paths: Sequence[str | Path],
) -> dict[str, Any]:
    stage = load_stage_config(stage_config_path)
    all_points: list[TokenizerTradeoffPoint] = []
    for path in report_paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        all_points.extend(
            tradeoff_points_from_report(
                report,
                spec=stage.model,
                stage_target_parameters=stage.target_parameters,
            )
        )
    frontier = pareto_frontier(all_points)
    candidates: list[dict[str, Any]] = []
    seen_vocab: set[int] = set()
    for point in frontier:
        if point.actual_vocab_size in seen_vocab:
            continue
        seen_vocab.add(point.actual_vocab_size)
        rebalanced = rebalance_d_ff_for_vocabulary(
            stage.model,
            target_parameters=stage.target_parameters,
            vocab_size=point.actual_vocab_size,
        )
        candidates.append(
            {
                "algorithm": point.algorithm,
                "requested_vocab_size": point.requested_vocab_size,
                "actual_vocab_size": point.actual_vocab_size,
                "model": rebalanced.model.to_dict(),
                "model_identity_sha256": rebalanced.model.identity_sha256(),
                "parameter_count": rebalanced.parameter_count,
                "target_delta": rebalanced.target_delta,
                "vocabulary_parameters": rebalanced.vocabulary_parameters,
                "vocabulary_share": rebalanced.vocabulary_share,
            }
        )
    return {
        "schema": "12-6.vocabulary-pareto-evidence.v1",
        "status": "EXPERIMENTAL_NOT_TOKENIZER_OR_STAGE_FREEZE",
        "stage": stage.stage,
        "stage_model_identity_sha256": stage.model.identity_sha256(),
        "stage_target_parameters": stage.target_parameters,
        "points": [asdict(point) for point in all_points],
        "repeatable_pareto_frontier": [asdict(point) for point in frontier],
        "rebalanced_model_candidates": candidates,
        "truth_boundary": {
            "canonical_s0_unchanged": True,
            "tokenizer_frozen": False,
            "stage_promoted": False,
            "representative_s1_corpus_claimed": False,
            "model_quality_claimed": False,
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix = subparsers.add_parser("matrix", help="aggregate real tokenizer reports")
    matrix.add_argument("--stage-config", required=True)
    matrix.add_argument("--reports", nargs="+", required=True)
    matrix.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command != "matrix":
        raise AssertionError("unreachable command")
    payload = build_matrix_evidence(
        stage_config_path=args.stage_config,
        report_paths=args.reports,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
