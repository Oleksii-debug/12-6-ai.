"""Tokenizer-bound model geometry rebalance for experimental stage candidates.

This module extends the vocabulary-allocation incumbent rather than replacing it.  It
searches depth/head geometry, delegates d_ff retargeting to
``twelve_six.vocabulary.rebalance_d_ff_for_vocabulary``, and binds every resulting
ModelSpec to the exact tokenizer artifact identity used to size the embedding table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .model import InitSpec, ModelSpec, TwelveSixDecoder, canonical_json_sha256, load_stage_config
from .vocabulary import VocabularyAllocationError, rebalance_d_ff_for_vocabulary, vocabulary_cost


TOKENIZER_IDENTITY_SCHEMA = "12-6.tokenizer-artifact-identity.v1"
BOUND_MODELSPEC_SCHEMA = "12-6.bound-modelspec-tokenizer.v1"
SEARCH_SCHEMA = "12-6.model-geometry-search.v1"
STAGE_TABLE_SCHEMA = "12-6.model-rebalance-stage-table.v1"


class ModelRebalanceError(ValueError):
    """Fail-closed model/tokenizer rebalance input error."""


def _validate_sha256(value: str, *, field: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ModelRebalanceError(f"{field} must be a 64-character lowercase hex SHA-256")
    return normalized


def _token_ids_from_tokenizer_json(payload: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    model = payload.get("model")
    if isinstance(model, dict):
        vocab = model.get("vocab")
        if isinstance(vocab, dict):
            ids.extend(int(value) for value in vocab.values())
        elif isinstance(vocab, list):
            for index, item in enumerate(vocab):
                if isinstance(item, list) and len(item) >= 2 and isinstance(item[1], int):
                    ids.append(int(item[1]))
                else:
                    ids.append(index)
    vocab = payload.get("vocab")
    if isinstance(vocab, dict):
        ids.extend(int(value) for value in vocab.values())
    added = payload.get("added_tokens")
    if isinstance(added, list):
        for item in added:
            if isinstance(item, dict) and isinstance(item.get("id"), int):
                ids.append(int(item["id"]))
    return ids


def infer_tokenizer_vocab_size(payload: dict[str, Any]) -> int:
    """Infer embedding cardinality from a serialized tokenizer artifact.

    When token ids are available we use ``max_id + 1`` rather than ``len(vocab)`` so
    added-token ids and sparse/corrupt id surfaces cannot be silently ignored.
    """
    ids = _token_ids_from_tokenizer_json(payload)
    if ids:
        if min(ids) < 0:
            raise ModelRebalanceError("tokenizer artifact contains a negative token id")
        return max(ids) + 1
    for key in ("vocab_size", "actual_vocab_size"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return int(value)
    artifact = payload.get("artifact")
    if isinstance(artifact, dict):
        value = artifact.get("vocab_size")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return int(value)
    raise ModelRebalanceError("cannot infer actual vocabulary size from tokenizer artifact")


@dataclass(frozen=True, slots=True)
class TokenizerArtifactIdentity:
    """Portable identity of the exact tokenizer serialization that owns token ids."""

    vocab_size: int
    tokenizer_json_sha256: str
    source_kind: str
    source_evidence_sha256: str | None = None
    schema: str = TOKENIZER_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TOKENIZER_IDENTITY_SCHEMA:
            raise ModelRebalanceError(f"unsupported tokenizer identity schema: {self.schema}")
        if self.vocab_size <= 0:
            raise ModelRebalanceError("tokenizer vocab_size must be positive")
        object.__setattr__(
            self,
            "tokenizer_json_sha256",
            _validate_sha256(self.tokenizer_json_sha256, field="tokenizer_json_sha256"),
        )
        if self.source_evidence_sha256 is not None:
            object.__setattr__(
                self,
                "source_evidence_sha256",
                _validate_sha256(self.source_evidence_sha256, field="source_evidence_sha256"),
            )
        if not self.source_kind:
            raise ModelRebalanceError("source_kind must be non-empty")

    @classmethod
    def from_artifact(cls, path: str | Path) -> TokenizerArtifactIdentity:
        raw = Path(path).read_bytes()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelRebalanceError("tokenizer artifact must be UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise ModelRebalanceError("tokenizer artifact root must be a JSON object")
        return cls(
            vocab_size=infer_tokenizer_vocab_size(payload),
            tokenizer_json_sha256=hashlib.sha256(raw).hexdigest(),
            source_kind="artifact_bytes",
        )

    @classmethod
    def from_descriptor(cls, path: str | Path) -> TokenizerArtifactIdentity:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != TOKENIZER_IDENTITY_SCHEMA:
            raise ModelRebalanceError("tokenizer identity descriptor schema mismatch")
        return cls(
            vocab_size=int(payload["vocab_size"]),
            tokenizer_json_sha256=str(payload["tokenizer_json_sha256"]),
            source_kind=str(payload["source_kind"]),
            source_evidence_sha256=(
                None
                if payload.get("source_evidence_sha256") is None
                else str(payload["source_evidence_sha256"])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def identity_sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class GeometryConstraints:
    """Explicit search envelope; no architecture dimension changes outside this set."""

    n_layers: tuple[int, ...]
    n_heads: tuple[int, ...]
    head_dims: tuple[int, ...]
    n_kv_heads: tuple[int, ...] = ()
    d_ff_alignment: int = 8
    min_d_ff_ratio: float = 2.0
    max_d_ff_ratio: float = 4.0
    max_embedding_fraction: float = 0.30
    max_target_delta_fraction: float = 0.03
    max_candidates: int = 16

    def __post_init__(self) -> None:
        for name, values in (
            ("n_layers", self.n_layers),
            ("n_heads", self.n_heads),
            ("head_dims", self.head_dims),
        ):
            if not values or any(value <= 0 for value in values):
                raise ModelRebalanceError(f"{name} must contain positive integers")
        if any(value <= 0 for value in self.n_kv_heads):
            raise ModelRebalanceError("n_kv_heads must contain positive integers")
        if self.d_ff_alignment <= 0:
            raise ModelRebalanceError("d_ff_alignment must be positive")
        if not 0 < self.min_d_ff_ratio <= self.max_d_ff_ratio:
            raise ModelRebalanceError("invalid d_ff ratio interval")
        if not 0 < self.max_embedding_fraction < 1:
            raise ModelRebalanceError("max_embedding_fraction must be in (0, 1)")
        if not 0 <= self.max_target_delta_fraction < 1:
            raise ModelRebalanceError("max_target_delta_fraction must be in [0, 1)")
        if self.max_candidates <= 0:
            raise ModelRebalanceError("max_candidates must be positive")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GeometryConstraints:
        return cls(
            n_layers=tuple(int(value) for value in payload["n_layers"]),
            n_heads=tuple(int(value) for value in payload["n_heads"]),
            head_dims=tuple(int(value) for value in payload["head_dims"]),
            n_kv_heads=tuple(int(value) for value in payload.get("n_kv_heads", [])),
            d_ff_alignment=int(payload.get("d_ff_alignment", 8)),
            min_d_ff_ratio=float(payload.get("min_d_ff_ratio", 2.0)),
            max_d_ff_ratio=float(payload.get("max_d_ff_ratio", 4.0)),
            max_embedding_fraction=float(payload.get("max_embedding_fraction", 0.30)),
            max_target_delta_fraction=float(payload.get("max_target_delta_fraction", 0.03)),
            max_candidates=int(payload.get("max_candidates", 16)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("n_layers", "n_heads", "head_dims", "n_kv_heads"):
            payload[key] = list(payload[key])
        return payload


@dataclass(frozen=True, slots=True)
class GeometryCandidate:
    model: ModelSpec
    parameter_count: int
    target_parameters: int
    target_delta: int
    embedding_parameters: int
    embedding_fraction: float
    vocabulary_parameters: int
    vocabulary_fraction: float
    block_parameters: int
    block_fraction: float
    head_valid: bool
    model_spec_identity_sha256: str
    tokenizer_artifact_identity_sha256: str
    tokenizer_json_sha256: str
    bound_modelspec_identity_sha256: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model"] = self.model.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class GeometrySearchResult:
    target_parameters: int
    tokenizer: TokenizerArtifactIdentity
    constraints: GeometryConstraints
    candidates: tuple[GeometryCandidate, ...]
    rejected: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SEARCH_SCHEMA,
            "status": "EXPERIMENTAL_NOT_CANONICAL_PROMOTION",
            "target_parameters": self.target_parameters,
            "tokenizer": self.tokenizer.to_dict(),
            "tokenizer_artifact_identity_sha256": self.tokenizer.identity_sha256(),
            "constraints": self.constraints.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "rejected": dict(sorted(self.rejected.items())),
            "truth_boundary": {
                "canonical_configs_changed": False,
                "tokenizer_frozen": False,
                "model_promoted": False,
            },
        }


def bound_modelspec_identity_sha256(
    model: ModelSpec,
    tokenizer: TokenizerArtifactIdentity,
) -> str:
    """Identity of a ModelSpec inseparably bound to one exact tokenizer JSON hash."""
    return canonical_json_sha256(
        {
            "schema": BOUND_MODELSPEC_SCHEMA,
            "model_spec": model.to_dict(),
            "model_spec_identity_sha256": model.identity_sha256(),
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "tokenizer_json_sha256": tokenizer.tokenizer_json_sha256,
            "tokenizer_artifact_identity_sha256": tokenizer.identity_sha256(),
        }
    )


def _candidate_from_model(
    model: ModelSpec,
    *,
    target_parameters: int,
    tokenizer: TokenizerArtifactIdentity,
) -> GeometryCandidate:
    breakdown = model.parameter_breakdown()
    total = breakdown["total"]
    vocab_cost = vocabulary_cost(model)
    head_valid = (
        model.d_model == model.n_heads * model.head_dim
        and model.n_heads % model.n_kv_heads == 0
        and model.head_dim % 2 == 0
        and model.rope_rotary_dim == model.head_dim
    )
    return GeometryCandidate(
        model=model,
        parameter_count=total,
        target_parameters=target_parameters,
        target_delta=total - target_parameters,
        embedding_parameters=breakdown["token_embedding"],
        embedding_fraction=breakdown["token_embedding"] / total,
        vocabulary_parameters=vocab_cost.total_vocabulary_parameters,
        vocabulary_fraction=vocab_cost.total_vocabulary_parameters / total,
        block_parameters=breakdown["blocks_total"],
        block_fraction=breakdown["blocks_total"] / total,
        head_valid=head_valid,
        model_spec_identity_sha256=model.identity_sha256(),
        tokenizer_artifact_identity_sha256=tokenizer.identity_sha256(),
        tokenizer_json_sha256=tokenizer.tokenizer_json_sha256,
        bound_modelspec_identity_sha256=bound_modelspec_identity_sha256(model, tokenizer),
    )


def search_model_geometry(
    base_spec: ModelSpec,
    *,
    target_parameters: int,
    tokenizer: TokenizerArtifactIdentity,
    constraints: GeometryConstraints,
) -> GeometrySearchResult:
    """Search valid d_model/d_ff geometry while reusing the incumbent d_ff solver."""
    if target_parameters <= 0:
        raise ModelRebalanceError("target_parameters must be positive")
    rejected: Counter[str] = Counter()
    accepted: list[GeometryCandidate] = []
    seen: set[str] = set()

    for n_layers in sorted(set(constraints.n_layers)):
        for n_heads in sorted(set(constraints.n_heads)):
            kv_options = constraints.n_kv_heads or (n_heads,)
            for n_kv_heads in sorted(set(kv_options)):
                for head_dim in sorted(set(constraints.head_dims)):
                    if n_heads % n_kv_heads != 0 or head_dim % 2 != 0:
                        rejected["invalid_head_geometry"] += 1
                        continue
                    d_model = n_heads * head_dim
                    seed = replace(
                        base_spec,
                        vocab_size=tokenizer.vocab_size,
                        d_model=d_model,
                        n_layers=n_layers,
                        n_heads=n_heads,
                        n_kv_heads=n_kv_heads,
                        head_dim=head_dim,
                        rope_rotary_dim=head_dim,
                    )
                    try:
                        rebalanced = rebalance_d_ff_for_vocabulary(
                            seed,
                            target_parameters=target_parameters,
                            vocab_size=tokenizer.vocab_size,
                            d_ff_alignment=constraints.d_ff_alignment,
                        )
                    except VocabularyAllocationError:
                        rejected["non_ffn_budget_exhausted"] += 1
                        continue
                    model = rebalanced.model
                    ratio = model.d_ff / model.d_model
                    if not constraints.min_d_ff_ratio <= ratio <= constraints.max_d_ff_ratio:
                        rejected["d_ff_ratio"] += 1
                        continue
                    candidate = _candidate_from_model(
                        model,
                        target_parameters=target_parameters,
                        tokenizer=tokenizer,
                    )
                    if not candidate.head_valid:
                        rejected["invalid_head_geometry"] += 1
                        continue
                    if candidate.embedding_fraction > constraints.max_embedding_fraction:
                        rejected["embedding_fraction"] += 1
                        continue
                    delta_fraction = abs(candidate.target_delta) / target_parameters
                    if delta_fraction > constraints.max_target_delta_fraction:
                        rejected["target_delta"] += 1
                        continue
                    identity = candidate.model_spec_identity_sha256
                    if identity in seen:
                        rejected["duplicate_modelspec"] += 1
                        continue
                    seen.add(identity)
                    accepted.append(candidate)

    accepted.sort(
        key=lambda candidate: (
            abs(candidate.target_delta),
            candidate.parameter_count > target_parameters,
            -candidate.block_fraction,
            candidate.model.d_model,
            candidate.model.n_layers,
            candidate.model.n_heads,
            candidate.model.d_ff,
        )
    )
    return GeometrySearchResult(
        target_parameters=target_parameters,
        tokenizer=tokenizer,
        constraints=constraints,
        candidates=tuple(accepted[: constraints.max_candidates]),
        rejected=dict(rejected),
    )


def one_training_step_smoke(
    candidate: GeometryCandidate,
    *,
    init_spec: InitSpec | None = None,
    sequence_length: int = 4,
    seed: int = 1234,
) -> dict[str, Any]:
    """Construct the real decoder and execute exactly one CPU optimizer step."""
    if sequence_length < 3:
        raise ModelRebalanceError("sequence_length must be at least 3")
    if sequence_length - 1 > candidate.model.max_seq_len:
        raise ModelRebalanceError("smoke sequence exceeds candidate max_seq_len")
    torch.manual_seed(seed)
    model = TwelveSixDecoder(candidate.model, init_spec)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    token_ids = torch.arange(sequence_length, dtype=torch.long).remainder(candidate.model.vocab_size)
    token_ids = token_ids.unsqueeze(0)
    before = model.token_embedding.weight[0].detach().clone()
    optimizer.zero_grad(set_to_none=True)
    logits = model(token_ids[:, :-1]).logits
    loss = F.cross_entropy(
        logits.reshape(-1, candidate.model.vocab_size),
        token_ids[:, 1:].reshape(-1),
    )
    if not math.isfinite(float(loss.detach())):
        raise RuntimeError("non-finite smoke loss")
    loss.backward()
    optimizer.step()
    changed = not torch.equal(before, model.token_embedding.weight[0].detach())
    if not changed:
        raise RuntimeError("training smoke optimizer step did not change embedding parameter")
    return {
        "status": "PASS",
        "model_spec_identity_sha256": candidate.model_spec_identity_sha256,
        "bound_modelspec_identity_sha256": candidate.bound_modelspec_identity_sha256,
        "parameter_count": candidate.parameter_count,
        "sequence_length": sequence_length,
        "loss": float(loss.detach()),
        "optimizer_steps": 1,
        "parameter_changed": changed,
    }


def build_stage_candidate_table(
    *,
    profiles_path: str | Path,
    tokenizer: TokenizerArtifactIdentity,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    profiles = json.loads(Path(profiles_path).read_text(encoding="utf-8"))
    if profiles.get("schema") != "12-6.model-rebalance-profiles.v1":
        raise ModelRebalanceError("model rebalance profile schema mismatch")
    root = Path(repo_root)
    rows: list[dict[str, Any]] = []
    for profile in profiles["profiles"]:
        base = load_stage_config(root / str(profile["base_stage_config"])).model
        result = search_model_geometry(
            base,
            target_parameters=int(profile["target_parameters"]),
            tokenizer=tokenizer,
            constraints=GeometryConstraints.from_dict(profile["constraints"]),
        )
        if not result.candidates:
            raise ModelRebalanceError(f"no valid candidates for profile {profile['stage']}")
        rows.append(
            {
                "stage": str(profile["stage"]),
                "target_parameters": int(profile["target_parameters"]),
                "base_stage_config": str(profile["base_stage_config"]),
                "candidates": [candidate.to_dict() for candidate in result.candidates],
                "rejected": result.rejected,
            }
        )
    return {
        "schema": STAGE_TABLE_SCHEMA,
        "status": "EXPERIMENTAL_NOT_CANONICAL_PROMOTION",
        "tokenizer": tokenizer.to_dict(),
        "tokenizer_artifact_identity_sha256": tokenizer.identity_sha256(),
        "stages": rows,
        "truth_boundary": {
            "canonical_configs_changed": False,
            "tokenizer_frozen": False,
            "model_promoted": False,
        },
    }


def _comma_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def _add_tokenizer_identity_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tokenizer-artifact")
    group.add_argument("--tokenizer-identity")


def _tokenizer_from_args(args: argparse.Namespace) -> TokenizerArtifactIdentity:
    if args.tokenizer_artifact:
        return TokenizerArtifactIdentity.from_artifact(args.tokenizer_artifact)
    return TokenizerArtifactIdentity.from_descriptor(args.tokenizer_identity)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search", help="search one target parameter budget")
    search.add_argument("--stage-config", required=True)
    _add_tokenizer_identity_args(search)
    search.add_argument("--target-parameters", type=int, required=True)
    search.add_argument("--layers", type=_comma_ints, required=True)
    search.add_argument("--heads", type=_comma_ints, required=True)
    search.add_argument("--kv-heads", type=_comma_ints)
    search.add_argument("--head-dims", type=_comma_ints, required=True)
    search.add_argument("--d-ff-alignment", type=int, default=8)
    search.add_argument("--min-d-ff-ratio", type=float, default=2.0)
    search.add_argument("--max-d-ff-ratio", type=float, default=4.0)
    search.add_argument("--max-embedding-fraction", type=float, default=0.30)
    search.add_argument("--max-target-delta-fraction", type=float, default=0.03)
    search.add_argument("--limit", type=int, default=16)
    search.add_argument("--output", required=True)

    table = subparsers.add_parser("stage-table", help="search 100K/250K/500K/1M/10M profiles")
    table.add_argument("--profiles", required=True)
    table.add_argument("--repo-root", default=".")
    _add_tokenizer_identity_args(table)
    table.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    tokenizer = _tokenizer_from_args(args)
    if args.command == "search":
        base = load_stage_config(args.stage_config).model
        constraints = GeometryConstraints(
            n_layers=args.layers,
            n_heads=args.heads,
            n_kv_heads=() if args.kv_heads is None else args.kv_heads,
            head_dims=args.head_dims,
            d_ff_alignment=args.d_ff_alignment,
            min_d_ff_ratio=args.min_d_ff_ratio,
            max_d_ff_ratio=args.max_d_ff_ratio,
            max_embedding_fraction=args.max_embedding_fraction,
            max_target_delta_fraction=args.max_target_delta_fraction,
            max_candidates=args.limit,
        )
        result = search_model_geometry(
            base,
            target_parameters=args.target_parameters,
            tokenizer=tokenizer,
            constraints=constraints,
        )
        _write_json(args.output, result.to_dict())
        return 0
    if args.command == "stage-table":
        payload = build_stage_candidate_table(
            profiles_path=args.profiles,
            tokenizer=tokenizer,
            repo_root=args.repo_root,
        )
        _write_json(args.output, payload)
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
