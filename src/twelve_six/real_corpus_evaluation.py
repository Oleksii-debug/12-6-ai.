"""Canonical first-party held-out evaluator for the learned 100K-to-1M ladder."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from twelve_six.data.real_corpus_holdout import (
    MODALITIES,
    RealCorpusHoldoutError,
    hash_json,
    require_sha256,
    validate_exclusion_proof,
)

CHECKPOINT_REPORT_SCHEMA = "12-6.real-corpus-checkpoint-evaluation.v1"
LADDER_REPORT_SCHEMA = "12-6.real-corpus-ladder-evaluation.v1"
DASHBOARD_ROW_SCHEMA = "12-6.scaling-dashboard-heldout-quality-row.v1"
AUTHORITY = "LOCAL_FREE_FIRST_PARTY_REAL_CORPUS_QUALITY"
BOOTSTRAP_METHOD = "DETERMINISTIC_DOCUMENT_BOOTSTRAP_SPLITMIX64_PERCENTILE_V1"
MASK64 = (1 << 64) - 1


class RealCorpusEvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvaluationBindings:
    training_corpus_identity_sha256: str
    training_split_identity_sha256: str
    decontamination_report_sha256: str
    training_exclusion_proof: Mapping[str, Any]
    tokenizer_fit_exclusion_proof: Mapping[str, Any]

    def validate(self, *, heldout_identity_sha256: str) -> dict[str, str]:
        for name in (
            "training_corpus_identity_sha256",
            "training_split_identity_sha256",
            "decontamination_report_sha256",
        ):
            require_sha256(name, getattr(self, name))
        training_proof = validate_exclusion_proof(
            self.training_exclusion_proof,
            heldout_identity_sha256=heldout_identity_sha256,
            purpose="MODEL_TRAINING",
        )
        tokenizer_proof = validate_exclusion_proof(
            self.tokenizer_fit_exclusion_proof,
            heldout_identity_sha256=heldout_identity_sha256,
            purpose="TOKENIZER_FIT",
        )
        return {
            "training_exclusion_proof_sha256": training_proof,
            "tokenizer_fit_exclusion_proof_sha256": tokenizer_proof,
        }


@dataclass(frozen=True)
class CheckpointDescriptor:
    label: str
    kind: str
    checkpoint_identity_sha256: str
    model_spec_sha256: str
    initialization_identity_sha256: str
    training_run_identity_sha256: str
    parameter_count: int
    optimized_tokens: int

    def validate(self) -> None:
        if not self.label.strip():
            raise RealCorpusEvaluationError("checkpoint label must be non-empty")
        if self.kind not in {"RANDOM_INIT", "LEARNED"}:
            raise RealCorpusEvaluationError("checkpoint kind must be RANDOM_INIT or LEARNED")
        for name in (
            "checkpoint_identity_sha256",
            "model_spec_sha256",
            "initialization_identity_sha256",
            "training_run_identity_sha256",
        ):
            require_sha256(name, getattr(self, name))
        if isinstance(self.parameter_count, bool) or self.parameter_count <= 0:
            raise RealCorpusEvaluationError("parameter_count must be positive")
        if isinstance(self.optimized_tokens, bool) or self.optimized_tokens < 0:
            raise RealCorpusEvaluationError("optimized_tokens must be non-negative")
        if self.kind == "RANDOM_INIT" and self.optimized_tokens != 0:
            raise RealCorpusEvaluationError("random-init baseline must have zero optimized tokens")
        if self.kind == "LEARNED" and self.optimized_tokens <= 0:
            raise RealCorpusEvaluationError(
                "learned checkpoint must have positive optimized tokens"
            )


def _canonical_tokenizer_identity(tokenizer: object) -> tuple[dict[str, Any], str]:
    identity = getattr(tokenizer, "identity", None)
    if identity is None:
        raise RealCorpusEvaluationError("tokenizer must expose an exact identity")
    if hasattr(identity, "to_dict"):
        value = identity.to_dict()
    elif is_dataclass(identity):
        value = {field.name: getattr(identity, field.name) for field in fields(identity)}
    elif isinstance(identity, Mapping):
        value = dict(identity)
    else:
        raise RealCorpusEvaluationError("unsupported tokenizer identity type")
    if not isinstance(value, dict) or not value:
        raise RealCorpusEvaluationError("tokenizer identity must serialize to a non-empty object")
    vocab_size = getattr(tokenizer, "vocab_size", value.get("vocab_size"))
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 1:
        raise RealCorpusEvaluationError("tokenizer vocab_size must be an integer > 1")
    value["vocab_size"] = vocab_size
    return value, hash_json(value)


def _tensor_digest_update(h: Any, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    h.update(str(value.dtype).encode("utf-8") + b"\0")
    h.update(str(tuple(value.shape)).encode("utf-8") + b"\0")
    h.update(value.view(torch.uint8).numpy().tobytes())


def model_state_sha256(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(b"tensor\0" + name.encode("utf-8") + b"\0")
        _tensor_digest_update(h, tensor)
    for name, module in model.named_modules():
        h.update(b"mode\0" + name.encode("utf-8") + b"\0")
        h.update(b"1" if module.training else b"0")
    return h.hexdigest()


def _state_digest_update(h: Any, value: object) -> None:
    if value is None or isinstance(value, (str, int, bool)):
        h.update(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RealCorpusEvaluationError("trainer state contains non-finite float")
        h.update(value.hex().encode("ascii"))
        return
    if isinstance(value, torch.Tensor):
        h.update(b"T")
        _tensor_digest_update(h, value)
        return
    if is_dataclass(value):
        h.update(b"D")
        for field in fields(value):
            h.update(field.name.encode("utf-8") + b"\0")
            _state_digest_update(h, getattr(value, field.name))
        return
    if isinstance(value, Mapping):
        h.update(b"M")
        keyed = []
        for key, item in value.items():
            key_bytes = json.dumps(
                key, ensure_ascii=False, sort_keys=True, default=str
            ).encode("utf-8")
            keyed.append((key_bytes, item))
        for key_bytes, item in sorted(keyed, key=lambda pair: pair[0]):
            h.update(key_bytes + b"\0")
            _state_digest_update(h, item)
        return
    if isinstance(value, (list, tuple)):
        h.update(b"L" if isinstance(value, list) else b"Q")
        for item in value:
            _state_digest_update(h, item)
        return
    raise RealCorpusEvaluationError(
        f"unsupported trainer-state type: {type(value).__name__}"
    )


def trainer_state_sha256(trainer: object) -> str:
    state_fn = getattr(trainer, "state_dict", None)
    if not callable(state_fn):
        raise RealCorpusEvaluationError("trainer must expose state_dict() for non-mutation proof")
    h = hashlib.sha256()
    _state_digest_update(h, state_fn())
    for name in ("micro_step", "optimizer_step", "tokens_seen"):
        value = getattr(trainer, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RealCorpusEvaluationError(f"trainer {name} must be a non-negative integer")
        h.update(name.encode("ascii") + b"=" + str(value).encode("ascii") + b"\0")
    return h.hexdigest()


def _trainer_snapshot(trainer: object) -> dict[str, Any]:
    return {
        "micro_step": int(getattr(trainer, "micro_step")),
        "optimizer_step": int(getattr(trainer, "optimizer_step")),
        "tokens_seen": int(getattr(trainer, "tokens_seen")),
        "state_sha256": trainer_state_sha256(trainer),
    }


def _model_device(model: torch.nn.Module) -> torch.device:
    devices = {tensor.device for tensor in model.parameters()}
    devices.update(tensor.device for tensor in model.buffers())
    if not devices:
        return torch.device("cpu")
    if len(devices) != 1:
        raise RealCorpusEvaluationError("evaluation requires model tensors on one device")
    return next(iter(devices))


def _actual_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _require_model_contract(
    model: torch.nn.Module, descriptor: CheckpointDescriptor, *, context_tokens: int
) -> None:
    descriptor.validate()
    actual_parameters = _actual_parameter_count(model)
    if actual_parameters != descriptor.parameter_count:
        raise RealCorpusEvaluationError(
            f"parameter count mismatch: {actual_parameters} != {descriptor.parameter_count}"
        )
    spec = getattr(model, "spec", None)
    if spec is not None:
        identity_fn = getattr(spec, "identity_sha256", None)
        if callable(identity_fn) and identity_fn() != descriptor.model_spec_sha256:
            raise RealCorpusEvaluationError("runtime ModelSpec identity mismatch")
        max_seq_len = getattr(spec, "max_seq_len", None)
        if isinstance(max_seq_len, int) and context_tokens > max_seq_len:
            raise RealCorpusEvaluationError("evaluation context exceeds ModelSpec max_seq_len")
        if bool(getattr(model, "training", False)) and float(
            getattr(spec, "attention_dropout", 0.0)
        ) > 0.0:
            raise RealCorpusEvaluationError(
                "dropout-enabled model must already be in deterministic evaluation mode"
            )
    if bool(getattr(model, "training", False)):
        for module in model.modules():
            if isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p > 0.0:
                raise RealCorpusEvaluationError(
                    "dropout-enabled model must already be in deterministic evaluation mode"
                )


def _extract_logits(output: object) -> torch.Tensor:
    logits = getattr(output, "logits", output)
    if not isinstance(logits, torch.Tensor):
        raise RealCorpusEvaluationError("model output must provide Tensor logits")
    return logits


def _safe_perplexity(cross_entropy: float) -> tuple[float | None, str]:
    if not math.isfinite(cross_entropy) or cross_entropy < 0.0:
        return None, "NOT_MEANINGFUL_NONFINITE_OR_NEGATIVE_CROSS_ENTROPY"
    if cross_entropy > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
        return None, "NOT_REPRESENTABLE_OVERFLOW"
    return math.exp(cross_entropy), "MEANINGFUL_ONLY_WITHIN_EXACT_TOKENIZER_IDENTITY"


def _aggregate(documents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not documents:
        raise RealCorpusEvaluationError("cannot aggregate empty held-out document group")
    nll = math.fsum(float(row["nll_nats"]) for row in documents)
    scored_tokens = sum(int(row["scored_tokens"]) for row in documents)
    tokenizer_tokens = sum(int(row["tokenizer_tokens"]) for row in documents)
    source_bytes = sum(int(row["source_bytes"]) for row in documents)
    if scored_tokens <= 0 or tokenizer_tokens <= 0 or source_bytes <= 0:
        raise RealCorpusEvaluationError("held-out aggregate contains no scored content")
    cross_entropy = nll / scored_tokens
    perplexity, perplexity_status = _safe_perplexity(cross_entropy)
    return {
        "documents": len(documents),
        "nll_nats": nll,
        "scored_tokens": scored_tokens,
        "tokenizer_tokens": tokenizer_tokens,
        "source_bytes": source_bytes,
        "cross_entropy_nats_per_token": cross_entropy,
        "perplexity": perplexity,
        "perplexity_status": perplexity_status,
        "bits_per_source_byte": nll / math.log(2.0) / source_bytes,
        "tokens_per_source_byte": tokenizer_tokens / source_bytes,
    }


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    z ^= z >> 31
    return state, z & MASK64


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise RealCorpusEvaluationError("quantile input is empty")
    if len(ordered) == 1:
        return ordered[0]
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_summary(
    documents: Sequence[Mapping[str, Any]],
    *,
    seed_material: str,
    replicates: int,
    confidence: float,
) -> dict[str, Any]:
    if isinstance(replicates, bool) or replicates < 2:
        raise RealCorpusEvaluationError("bootstrap replicates must be >= 2")
    if not 0.0 < confidence < 1.0:
        raise RealCorpusEvaluationError("bootstrap confidence must be between zero and one")
    point = _aggregate(documents)
    metrics = (
        "cross_entropy_nats_per_token",
        "perplexity",
        "bits_per_source_byte",
        "tokens_per_source_byte",
    )
    if len(documents) == 1:
        intervals = {}
        for metric in metrics:
            value = point[metric]
            intervals[metric] = {"lower": value, "median": value, "upper": value}
        return {
            "method": "DEGENERATE_SINGLE_DOCUMENT",
            "replicates": 0,
            "confidence": confidence,
            "seed_sha256": hashlib.sha256(seed_material.encode("utf-8")).hexdigest(),
            "intervals": intervals,
        }
    alpha = (1.0 - confidence) / 2.0
    samples: dict[str, list[float]] = {metric: [] for metric in metrics}
    seed_hash = hashlib.sha256(seed_material.encode("utf-8")).digest()
    base_seed = int.from_bytes(seed_hash[:8], "big")
    for replicate in range(replicates):
        state = (base_seed ^ replicate) & MASK64
        selected = []
        for _ in documents:
            state, value = _splitmix64(state)
            selected.append(documents[value % len(documents)])
        aggregate = _aggregate(selected)
        for metric in metrics:
            value = aggregate[metric]
            if value is not None:
                samples[metric].append(float(value))
    intervals = {}
    for metric in metrics:
        values = samples[metric]
        if not values:
            intervals[metric] = {"lower": None, "median": None, "upper": None}
        else:
            intervals[metric] = {
                "lower": _quantile(values, alpha),
                "median": _quantile(values, 0.5),
                "upper": _quantile(values, 1.0 - alpha),
            }
    return {
        "method": BOOTSTRAP_METHOD,
        "replicates": replicates,
        "confidence": confidence,
        "seed_sha256": hashlib.sha256(seed_material.encode("utf-8")).hexdigest(),
        "intervals": intervals,
    }


def _score_document(
    model: torch.nn.Module,
    tokenizer: object,
    row: Mapping[str, Any],
    *,
    context_tokens: int,
    vocab_size: int,
    device: torch.device,
) -> dict[str, Any]:
    text = row.get("text")
    if not isinstance(text, str) or not text:
        raise RealCorpusEvaluationError("held-out row text is invalid")
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RealCorpusEvaluationError("tokenizer must expose encode()")
    token_ids = list(encode(text, add_bos=False, add_eos=False))
    if len(token_ids) < 2:
        raise RealCorpusEvaluationError("held-out document must encode to at least two tokens")
    if any(isinstance(token, bool) or not isinstance(token, int) for token in token_ids):
        raise RealCorpusEvaluationError("tokenizer returned non-integer token IDs")
    if any(token < 0 or token >= vocab_size for token in token_ids):
        raise RealCorpusEvaluationError("tokenizer returned out-of-vocabulary token ID")
    encoded = text.encode("utf-8")
    if row.get("source_bytes") != len(encoded):
        raise RealCorpusEvaluationError("held-out row source-byte identity mismatch")
    if row.get("content_sha256") != hashlib.sha256(encoded).hexdigest():
        raise RealCorpusEvaluationError("held-out row content identity mismatch")

    nll = 0.0
    scored = 0
    start = 0
    while start < len(token_ids) - 1:
        end = min(start + context_tokens + 1, len(token_ids))
        inputs = torch.tensor([token_ids[start : end - 1]], dtype=torch.long, device=device)
        targets = torch.tensor(token_ids[start + 1 : end], dtype=torch.long, device=device)
        output = model(inputs)
        logits = _extract_logits(output)
        expected_shape = (1, targets.numel(), vocab_size)
        if tuple(logits.shape) != expected_shape:
            raise RealCorpusEvaluationError(
                f"logit shape mismatch: {tuple(logits.shape)} != {expected_shape}"
            )
        loss = F.cross_entropy(logits[0], targets, reduction="sum")
        if not torch.isfinite(loss):
            raise RealCorpusEvaluationError("non-finite held-out NLL")
        nll += float(loss.item())
        scored += int(targets.numel())
        if end == len(token_ids):
            break
        start = end - 1
    return {
        "record_id": str(row["record_id"]),
        "content_sha256": str(row["content_sha256"]),
        "modality": str(row["modality"]),
        "source_family": str(row["source_family"]),
        "source_id": str(row["source_id"]),
        "nll_nats": nll,
        "scored_tokens": scored,
        "tokenizer_tokens": len(token_ids),
        "source_bytes": len(encoded),
    }


def _group_metrics(
    documents: Sequence[Mapping[str, Any]],
    *,
    checkpoint_identity_sha256: str,
    heldout_identity_sha256: str,
    replicates: int,
    confidence: float,
) -> dict[str, Any]:
    def pack(name: str, subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "point": _aggregate(subset),
            "bootstrap": _bootstrap_summary(
                subset,
                seed_material=f"{heldout_identity_sha256}:{checkpoint_identity_sha256}:{name}",
                replicates=replicates,
                confidence=confidence,
            ),
        }

    modalities = {
        modality: pack(
            f"modality:{modality}",
            [row for row in documents if row["modality"] == modality],
        )
        for modality in MODALITIES
    }
    families = sorted({str(row["source_family"]) for row in documents})
    source_families = {
        family: pack(
            f"source_family:{family}",
            [row for row in documents if row["source_family"] == family],
        )
        for family in families
    }
    return {
        "aggregate": pack("aggregate", documents),
        "by_modality": modalities,
        "by_source_family": source_families,
    }


def evaluate_checkpoint(
    model: torch.nn.Module,
    trainer: object,
    tokenizer: object,
    heldout_manifest: Mapping[str, Any],
    heldout_rows: Sequence[Mapping[str, Any]],
    descriptor: CheckpointDescriptor,
    bindings: EvaluationBindings,
    *,
    context_tokens: int,
    bootstrap_replicates: int = 512,
    bootstrap_confidence: float = 0.95,
) -> dict[str, Any]:
    if isinstance(context_tokens, bool) or context_tokens <= 0:
        raise RealCorpusEvaluationError("context_tokens must be positive")
    descriptor.validate()
    heldout_identity = require_sha256(
        "heldout_identity_sha256", str(heldout_manifest.get("heldout_identity_sha256", ""))
    )
    upstream = heldout_manifest.get("upstream")
    if not isinstance(upstream, Mapping):
        raise RealCorpusEvaluationError("held-out upstream identity block missing")
    if (
        upstream.get("decontamination_report_sha256")
        != bindings.decontamination_report_sha256
    ):
        raise RealCorpusEvaluationError(
            "decontamination identity differs between holdout and training"
        )
    proof_hashes = bindings.validate(heldout_identity_sha256=heldout_identity)
    tokenizer_payload, tokenizer_identity = _canonical_tokenizer_identity(tokenizer)
    vocab_size = int(tokenizer_payload["vocab_size"])
    _require_model_contract(model, descriptor, context_tokens=context_tokens)
    if getattr(trainer, "model", model) is not model:
        raise RealCorpusEvaluationError("trainer/model object identity mismatch")
    spec = getattr(model, "spec", None)
    model_vocab = getattr(spec, "vocab_size", vocab_size)
    if model_vocab != vocab_size:
        raise RealCorpusEvaluationError("tokenizer/model vocabulary mismatch")

    before_model = model_state_sha256(model)
    before_trainer = _trainer_snapshot(trainer)
    optimized_before = before_trainer["tokens_seen"]
    if optimized_before != descriptor.optimized_tokens:
        raise RealCorpusEvaluationError(
            "checkpoint descriptor optimized_tokens differs from Trainer tokens_seen"
        )
    device = _model_device(model)
    try:
        with torch.no_grad():
            documents = [
                _score_document(
                    model,
                    tokenizer,
                    row,
                    context_tokens=context_tokens,
                    vocab_size=vocab_size,
                    device=device,
                )
                for row in heldout_rows
            ]
    except RealCorpusHoldoutError as exc:
        raise RealCorpusEvaluationError(str(exc)) from exc
    after_model = model_state_sha256(model)
    after_trainer = _trainer_snapshot(trainer)
    if after_model != before_model:
        raise RealCorpusEvaluationError("held-out evaluation mutated model state")
    if after_trainer != before_trainer:
        raise RealCorpusEvaluationError("held-out evaluation mutated Trainer state")
    optimized_after = after_trainer["tokens_seen"]
    if optimized_after != optimized_before:
        raise RealCorpusEvaluationError("held-out evaluation incremented optimized tokens")
    if {str(row["modality"]) for row in documents} != set(MODALITIES):
        raise RealCorpusEvaluationError("held-out evaluation lacks a required modality")

    metrics = _group_metrics(
        documents,
        checkpoint_identity_sha256=descriptor.checkpoint_identity_sha256,
        heldout_identity_sha256=heldout_identity,
        replicates=bootstrap_replicates,
        confidence=bootstrap_confidence,
    )
    value = {
        "schema_version": CHECKPOINT_REPORT_SCHEMA,
        "authority": AUTHORITY,
        "quality_authority": "FIRST_PARTY_HELDOUT_ONLY",
        "train_loss_used_as_quality_evidence": False,
        "checkpoint": {
            "label": descriptor.label,
            "kind": descriptor.kind,
            "checkpoint_identity_sha256": descriptor.checkpoint_identity_sha256,
            "model_spec_sha256": descriptor.model_spec_sha256,
            "initialization_identity_sha256": descriptor.initialization_identity_sha256,
            "training_run_identity_sha256": descriptor.training_run_identity_sha256,
            "parameter_count": descriptor.parameter_count,
            "optimized_tokens_at_checkpoint": descriptor.optimized_tokens,
        },
        "identities": {
            "tokenizer": tokenizer_payload,
            "tokenizer_identity_sha256": tokenizer_identity,
            "training_corpus_identity_sha256": bindings.training_corpus_identity_sha256,
            "training_split_identity_sha256": bindings.training_split_identity_sha256,
            "heldout_identity_sha256": heldout_identity,
            "evaluation_corpus_identity_sha256": upstream[
                "evaluation_corpus_identity_sha256"
            ],
            "benchmark_registry_sha256": upstream["benchmark_registry_sha256"],
            "decontamination_reference_bundle_sha256": upstream[
                "decontamination_reference_bundle_sha256"
            ],
            "decontamination_report_sha256": bindings.decontamination_report_sha256,
            **proof_hashes,
        },
        "evaluation_contract": {
            "torch_no_grad": True,
            "model_mode_toggled": False,
            "model_state_mutation_allowed": False,
            "trainer_state_mutation_allowed": False,
            "optimized_tokens_delta_required": 0,
            "context_tokens": context_tokens,
            "context_semantics": "DOCUMENT_ISOLATED_ONE_TOKEN_OVERLAP",
            "first_token_semantics": "FIRST_TOKEN_UNSCORED_NO_CANONICAL_BOS",
            "bpb_semantics": (
                "total_scored_token_nll_bits divided by exact frozen UTF-8 source bytes; "
                "first token remains unscored because canonical Base has no BOS"
            ),
            "bootstrap_method": BOOTSTRAP_METHOD,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_confidence": bootstrap_confidence,
        },
        "non_mutation": {
            "model_state_sha256_before": before_model,
            "model_state_sha256_after": after_model,
            "trainer_state_sha256_before": before_trainer["state_sha256"],
            "trainer_state_sha256_after": after_trainer["state_sha256"],
            "trainer_optimizer_step_before": before_trainer["optimizer_step"],
            "trainer_optimizer_step_after": after_trainer["optimizer_step"],
            "optimized_tokens_before": optimized_before,
            "optimized_tokens_after": optimized_after,
            "optimized_tokens_delta": optimized_after - optimized_before,
            "passed": True,
        },
        "metrics": metrics,
        "documents": documents,
    }
    value["checkpoint_report_sha256"] = hash_json(value)
    return value


def _metric_delta(learned: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    output = {}
    for key in (
        "cross_entropy_nats_per_token",
        "bits_per_source_byte",
        "tokens_per_source_byte",
    ):
        learned_value = float(learned[key])
        baseline_value = float(baseline[key])
        output[f"{key}_delta"] = learned_value - baseline_value
    baseline_bpb = float(baseline["bits_per_source_byte"])
    learned_bpb = float(learned["bits_per_source_byte"])
    output["relative_bpb_improvement"] = (
        (baseline_bpb - learned_bpb) / baseline_bpb if baseline_bpb > 0.0 else None
    )
    baseline_ppl = baseline.get("perplexity")
    learned_ppl = learned.get("perplexity")
    output["perplexity_ratio"] = (
        float(learned_ppl) / float(baseline_ppl)
        if learned_ppl is not None and baseline_ppl not in (None, 0.0)
        else None
    )
    return output


def _flatten_group_rows(
    checkpoint: Mapping[str, Any], baseline: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = [("aggregate", "all", checkpoint["metrics"]["aggregate"])]
    groups.extend(
        ("modality", key, value)
        for key, value in sorted(checkpoint["metrics"]["by_modality"].items())
    )
    groups.extend(
        ("source_family", key, value)
        for key, value in sorted(checkpoint["metrics"]["by_source_family"].items())
    )
    baseline_groups: dict[tuple[str, str], Mapping[str, Any]] = {}
    if baseline is not None:
        baseline_groups[("aggregate", "all")] = baseline["metrics"]["aggregate"]
        baseline_groups.update(
            (("modality", key), value)
            for key, value in baseline["metrics"]["by_modality"].items()
        )
        baseline_groups.update(
            (("source_family", key), value)
            for key, value in baseline["metrics"]["by_source_family"].items()
        )
    for dimension, name, group in groups:
        point = group["point"]
        row = {
            "schema_version": DASHBOARD_ROW_SCHEMA,
            "checkpoint_label": checkpoint["checkpoint"]["label"],
            "checkpoint_kind": checkpoint["checkpoint"]["kind"],
            "checkpoint_identity_sha256": checkpoint["checkpoint"][
                "checkpoint_identity_sha256"
            ],
            "model_spec_sha256": checkpoint["checkpoint"]["model_spec_sha256"],
            "parameter_count": checkpoint["checkpoint"]["parameter_count"],
            "optimized_tokens": checkpoint["checkpoint"]["optimized_tokens_at_checkpoint"],
            "tokenizer_identity_sha256": checkpoint["identities"]["tokenizer_identity_sha256"],
            "training_corpus_identity_sha256": checkpoint["identities"][
                "training_corpus_identity_sha256"
            ],
            "training_split_identity_sha256": checkpoint["identities"][
                "training_split_identity_sha256"
            ],
            "heldout_identity_sha256": checkpoint["identities"]["heldout_identity_sha256"],
            "group_dimension": dimension,
            "group_value": name,
            "cross_entropy_nats_per_token": point["cross_entropy_nats_per_token"],
            "perplexity": point["perplexity"],
            "perplexity_status": point["perplexity_status"],
            "bits_per_source_byte": point["bits_per_source_byte"],
            "tokens_per_source_byte": point["tokens_per_source_byte"],
            "confidence_interval": group["bootstrap"],
            "improvement_vs_same_model_random_init": None,
        }
        baseline_group = baseline_groups.get((dimension, name))
        if baseline_group is not None and checkpoint["checkpoint"]["kind"] == "LEARNED":
            row["improvement_vs_same_model_random_init"] = _metric_delta(
                point, baseline_group["point"]
            )
        rows.append(row)
    return rows


def build_ladder_report(checkpoint_reports: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    checkpoints = [dict(value) for value in checkpoint_reports]
    if not checkpoints:
        raise RealCorpusEvaluationError("ladder report requires checkpoint evaluations")
    for checkpoint in checkpoints:
        supplied = require_sha256(
            "checkpoint_report_sha256", str(checkpoint.get("checkpoint_report_sha256", ""))
        )
        unsigned = dict(checkpoint)
        unsigned.pop("checkpoint_report_sha256", None)
        if supplied != hash_json(unsigned):
            raise RealCorpusEvaluationError("checkpoint evaluation self-hash mismatch")
        if checkpoint.get("schema_version") != CHECKPOINT_REPORT_SCHEMA:
            raise RealCorpusEvaluationError("checkpoint evaluation schema mismatch")
        if checkpoint.get("train_loss_used_as_quality_evidence") is not False:
            raise RealCorpusEvaluationError("train loss cannot be quality evidence")
        if checkpoint.get("non_mutation", {}).get("passed") is not True:
            raise RealCorpusEvaluationError("checkpoint lacks non-mutation proof")
        if checkpoint.get("non_mutation", {}).get("optimized_tokens_delta") != 0:
            raise RealCorpusEvaluationError("evaluation changed optimized tokens")

    common = {
        "tokenizer_identity_sha256": checkpoints[0]["identities"]["tokenizer_identity_sha256"],
        "training_corpus_identity_sha256": checkpoints[0]["identities"][
            "training_corpus_identity_sha256"
        ],
        "training_split_identity_sha256": checkpoints[0]["identities"][
            "training_split_identity_sha256"
        ],
        "heldout_identity_sha256": checkpoints[0]["identities"]["heldout_identity_sha256"],
        "decontamination_report_sha256": checkpoints[0]["identities"][
            "decontamination_report_sha256"
        ],
        "training_exclusion_proof_sha256": checkpoints[0]["identities"][
            "training_exclusion_proof_sha256"
        ],
        "tokenizer_fit_exclusion_proof_sha256": checkpoints[0]["identities"][
            "tokenizer_fit_exclusion_proof_sha256"
        ],
    }
    for checkpoint in checkpoints[1:]:
        for key, expected in common.items():
            if checkpoint["identities"].get(key) != expected:
                raise RealCorpusEvaluationError(f"non-comparable checkpoint identity drift: {key}")

    baselines: dict[str, Mapping[str, Any]] = {}
    for checkpoint in checkpoints:
        if checkpoint["checkpoint"]["kind"] == "RANDOM_INIT":
            model_id = str(checkpoint["checkpoint"]["model_spec_sha256"])
            if model_id in baselines:
                raise RealCorpusEvaluationError("multiple random-init baselines for one ModelSpec")
            baselines[model_id] = checkpoint
    learned = [c for c in checkpoints if c["checkpoint"]["kind"] == "LEARNED"]
    if not learned:
        raise RealCorpusEvaluationError("ladder report requires at least one learned checkpoint")
    for checkpoint in learned:
        if checkpoint["checkpoint"]["model_spec_sha256"] not in baselines:
            raise RealCorpusEvaluationError(
                "each learned ModelSpec requires a same-geometry random-init baseline"
            )

    enriched = []
    dashboard_rows = []
    for checkpoint in checkpoints:
        value = dict(checkpoint)
        model_id = str(checkpoint["checkpoint"]["model_spec_sha256"])
        baseline = baselines.get(model_id)
        if checkpoint["checkpoint"]["kind"] == "LEARNED" and baseline is not None:
            value["improvement_vs_same_model_random_init"] = _metric_delta(
                checkpoint["metrics"]["aggregate"]["point"],
                baseline["metrics"]["aggregate"]["point"],
            )
        else:
            value["improvement_vs_same_model_random_init"] = None
        enriched.append(value)
        dashboard_rows.extend(_flatten_group_rows(checkpoint, baseline))

    enriched.sort(
        key=lambda value: (
            int(value["checkpoint"]["parameter_count"]),
            int(value["checkpoint"]["optimized_tokens_at_checkpoint"]),
            str(value["checkpoint"]["label"]),
        )
    )
    dashboard_rows.sort(
        key=lambda value: (
            int(value["parameter_count"]),
            int(value["optimized_tokens"]),
            str(value["checkpoint_label"]),
            str(value["group_dimension"]),
            str(value["group_value"]),
        )
    )
    report = {
        "schema_version": LADDER_REPORT_SCHEMA,
        "authority": AUTHORITY,
        "quality_authority": "FIRST_PARTY_HELDOUT_ONLY",
        "train_loss_used_as_quality_evidence": False,
        "comparable_identity": common,
        "random_init_baseline_policy": "ONE_BASELINE_PER_MODEL_SPEC_IDENTITY",
        "checkpoint_reports": enriched,
        "dashboard_rows": dashboard_rows,
        "dashboard_contract": {
            "schema_version": DASHBOARD_ROW_SCHEMA,
            "row_count": len(dashboard_rows),
            "quality_selection_may_use_train_loss": False,
        },
    }
    report["report_sha256"] = hash_json(report)
    verify_ladder_report(report)
    return report


def _contains_forbidden_train_loss(value: object) -> bool:
    allowed_false_flags = {
        "train_loss_used_as_quality_evidence",
        "quality_selection_may_use_train_loss",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if "train_loss" in key_text:
                if key_text not in allowed_false_flags or item is not False:
                    return True
            if _contains_forbidden_train_loss(item):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_train_loss(item) for item in value)
    return False


def verify_ladder_report(report: Mapping[str, Any]) -> str:
    if report.get("schema_version") != LADDER_REPORT_SCHEMA:
        raise RealCorpusEvaluationError("ladder report schema mismatch")
    supplied = require_sha256("report_sha256", str(report.get("report_sha256", "")))
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    if supplied != hash_json(unsigned):
        raise RealCorpusEvaluationError("ladder report self-hash mismatch")
    if report.get("quality_authority") != "FIRST_PARTY_HELDOUT_ONLY":
        raise RealCorpusEvaluationError("quality authority weakened")
    if report.get("train_loss_used_as_quality_evidence") is not False:
        raise RealCorpusEvaluationError("train loss marked as quality evidence")
    if _contains_forbidden_train_loss(report):
        raise RealCorpusEvaluationError("train-loss field leaked into quality report")
    rows = report.get("dashboard_rows")
    if not isinstance(rows, list) or not rows:
        raise RealCorpusEvaluationError("dashboard rows missing")
    if report.get("dashboard_contract", {}).get("row_count") != len(rows):
        raise RealCorpusEvaluationError("dashboard row count mismatch")
    if (
        report.get("dashboard_contract", {}).get("quality_selection_may_use_train_loss")
        is not False
    ):
        raise RealCorpusEvaluationError("dashboard train-loss authority weakened")
    for row in rows:
        if row.get("schema_version") != DASHBOARD_ROW_SCHEMA:
            raise RealCorpusEvaluationError("dashboard row schema mismatch")
        if row.get("group_dimension") not in {"aggregate", "modality", "source_family"}:
            raise RealCorpusEvaluationError("dashboard grouping dimension invalid")
        require_sha256(
            "checkpoint_identity_sha256", str(row.get("checkpoint_identity_sha256", ""))
        )
    return supplied
