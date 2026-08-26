"""MODEL-197 matched 10M context-transfer experiment.

The scientific variable is the within-document training horizon (256/512/1024).
ModelSpec, initial weights, tokenizer, ordered source documents, optimized causal
pairs, optimizer hyperparameters, and optimizer-update grouping remain fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import resource
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from twelve_six.long_dependency import materialize_suite, score_suite
from twelve_six.milestone100_first_learned import (
    EXPECTED_CORPUS_ID,
    _build_corpus,
    _rows,
    _state_hash,
    _write_json,
)
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.packing import TextRecord, collate_right_trimmed_rows, iter_packed_examples
from twelve_six.tokenization import ByteTokenizer

SCHEMA = "12-6.model197-context-transfer.v1"
PLAN_SCHEMA = "12-6.model197-context-plan.v1"
SUMMARY_SCHEMA = "12-6.model197-context-summary.v1"
AUTHORITY = "LOCAL_FREE_CONTEXT_TRANSFER_EXPERIMENT_NOT_STAGE_PROMOTION"
STAGE_CONFIG = Path("configs/stages/alternatives/s3_10m_scale03_byte_gqa.execution.json")
EXPECTED_MODEL_SHA = "61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998"
EXPECTED_PARAMETERS = 10_000_640
EXPECTED_TRAIN_BYTES = 20_000_775
HORIZONS = (256, 512, 1024)
COMMON_SHORT_HORIZON = 256
SEED = 20260825
LR = 3e-4
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.1
CLIP_NORM = 1.0
DOCS_PER_UPDATE = 4
DEFAULT_TARGET_TOKENS = 262_144
HELDOUT_DOCS_PER_STRATUM = 32
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 197_001
MIXTURE = (
    "uk", "en", "uk", "code", "en",
    "uk", "en", "uk", "code", "uk",
    "en", "uk", "en", "code", "uk",
    "en", "uk", "code", "en", "uk",
)


class Model197Error(RuntimeError):
    pass


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _model(repo: Path) -> tuple[ModelSpec, InitSpec]:
    stage = load_stage_config(repo / STAGE_CONFIG)
    spec, init = stage.model, stage.init
    if spec.identity_sha256() != EXPECTED_MODEL_SHA:
        raise Model197Error("S3 ModelSpec identity drift")
    if spec.parameter_count() != EXPECTED_PARAMETERS:
        raise Model197Error("S3 parameter-count drift")
    if spec.max_seq_len != 1024:
        raise Model197Error("MODEL-197 requires the accepted 1024-capable S3 ModelSpec")
    if any(horizon > spec.max_seq_len for horizon in HORIZONS):
        raise Model197Error("requested context arm exceeds current ModelSpec")
    return spec, init


def _record_target_count(tok: ByteTokenizer, record: TextRecord) -> int:
    return max(0, len(tok.encode(record.text)) - 1)


def _records(corpus: Path, manifest: dict[str, Any], split: str) -> dict[str, list[TextRecord]]:
    result: dict[str, list[TextRecord]] = {"uk": [], "en": [], "code": []}
    for stratum in result:
        result[stratum] = [
            TextRecord(str(row["record_id"]), str(row["text"]), str(row["split"]))
            for row in _rows(corpus, manifest, split, stratum)
        ]
    return result


def _select_training_records(
    by_stratum: dict[str, list[TextRecord]], tok: ByteTokenizer, target_tokens: int
) -> tuple[list[TextRecord], int]:
    if target_tokens <= 0:
        raise Model197Error("target_tokens must be positive")
    cursors = {key: 0 for key in by_stratum}
    selected: list[TextRecord] = []
    optimized = 0
    mixture_index = 0
    while optimized < target_tokens:
        stratum = MIXTURE[mixture_index % len(MIXTURE)]
        mixture_index += 1
        cursor = cursors[stratum]
        if cursor >= len(by_stratum[stratum]):
            raise Model197Error(f"{stratum} corpus exhausted before matched target budget")
        record = by_stratum[stratum][cursor]
        cursors[stratum] += 1
        count = _record_target_count(tok, record)
        if count <= 0:
            continue
        selected.append(record)
        optimized += count
    if optimized >= EXPECTED_TRAIN_BYTES:
        raise Model197Error("selected exposure would reach/recycle the retained training corpus")
    return selected, optimized


def _eval_selection(by_stratum: dict[str, list[TextRecord]]) -> list[TextRecord]:
    selected: list[TextRecord] = []
    for stratum in ("uk", "en", "code"):
        rows = by_stratum[stratum][:HELDOUT_DOCS_PER_STRATUM]
        if len(rows) != HELDOUT_DOCS_PER_STRATUM:
            raise Model197Error(f"insufficient validation documents for {stratum}")
        selected.extend(rows)
    return selected


def _plan_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def prepare(repo: Path, out: Path, target_tokens: int) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    manifest = _build_corpus(repo, out)
    if manifest["corpus_identity_sha256"] != EXPECTED_CORPUS_ID:
        raise Model197Error("DATA-25 identity drift")
    if manifest["by_split"]["train"]["byte_tokens"] != EXPECTED_TRAIN_BYTES:
        raise Model197Error("DATA-25 train byte count drift")
    tok = ByteTokenizer()
    spec, init = _model(repo)
    corpus = out / "corpus-a"
    train = _records(corpus, manifest, "train")
    validation = _records(corpus, manifest, "validation")
    selected, realized_targets = _select_training_records(train, tok, target_tokens)
    heldout = _eval_selection(validation)
    stratum_lookup = {
        record.record_id: stratum
        for stratum, rows in train.items()
        for record in rows
    }
    heldout_lookup = {
        record.record_id: stratum
        for stratum, rows in validation.items()
        for record in rows
    }
    payload: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "authority": AUTHORITY,
        "corpus_identity_sha256": manifest["corpus_identity_sha256"],
        "tokenizer": {
            "version": tok.identity.version,
            "config_sha256": tok.identity.config_sha256,
            "vocab_sha256": tok.identity.vocab_sha256,
            "vocab_size": tok.identity.vocab_size,
        },
        "model_spec": spec.to_dict(),
        "model_spec_sha256": spec.identity_sha256(),
        "init_spec_sha256": init.identity_sha256(),
        "parameter_count": spec.parameter_count(),
        "horizons": list(HORIZONS),
        "common_short_horizon": COMMON_SHORT_HORIZON,
        "seed": SEED,
        "optimizer": {
            "name": "AdamW", "learning_rate": LR, "betas": list(BETAS), "eps": EPS,
            "weight_decay": WEIGHT_DECAY, "gradient_clip_norm": CLIP_NORM,
            "scheduler": "constant",
        },
        "docs_per_optimizer_update": DOCS_PER_UPDATE,
        "requested_optimized_tokens": target_tokens,
        "realized_optimized_tokens": realized_targets,
        "training_documents": [
            {
                "record_id": record.record_id,
                "stratum": stratum_lookup[record.record_id],
                "byte_tokens": len(tok.encode(record.text)),
                "causal_targets": _record_target_count(tok, record),
            }
            for record in selected
        ],
        "heldout_documents": [
            {
                "record_id": record.record_id,
                "stratum": heldout_lookup[record.record_id],
                "byte_tokens": len(tok.encode(record.text)),
            }
            for record in heldout
        ],
        "right_trim": {
            "enabled": True,
            "authority": "PERF-147 accepted semantics-preserving right trim",
            "cross_document": False,
        },
        "long_dependency": {
            "suite": "EVAL-135-LONG-DEPENDENCY-v1",
            "maximum_probe_distance_tokens": 512,
            "truth_boundary": "1024 arm receives no behavioral claim beyond 512 from EVAL-135 v1",
        },
        "paid_compute": False,
    }
    payload["plan_sha256"] = _plan_hash(payload)
    _write_json(out / "model197-plan.json", payload)
    return payload


def _load_plan(out: Path) -> dict[str, Any]:
    plan = json.loads((out / "model197-plan.json").read_text(encoding="utf-8"))
    claimed = plan.pop("plan_sha256")
    actual = _plan_hash(plan)
    plan["plan_sha256"] = claimed
    if claimed != actual:
        raise Model197Error("MODEL-197 plan hash mismatch")
    return plan


def _ordered_records(
    corpus: Path, manifest: dict[str, Any], split: str, ids: Sequence[str]
) -> list[TextRecord]:
    by = _records(corpus, manifest, split)
    lookup = {record.record_id: record for rows in by.values() for record in rows}
    missing = [record_id for record_id in ids if record_id not in lookup]
    if missing:
        raise Model197Error(f"planned records missing from rebuilt corpus: {missing[:3]}")
    return [lookup[record_id] for record_id in ids]


def _trimmed_batch(example: Any) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    rows = collate_right_trimmed_rows((example,), target_mode="labels")
    input_ids = torch.tensor(rows["input_ids"], dtype=torch.long)
    labels = torch.tensor(rows["labels"], dtype=torch.long)
    targets = int(labels[:, 1:].ne(-100).sum().item())
    return input_ids, labels, targets, int(input_ids.numel())


def _summed_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
        reduction="sum",
    )


def _global_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        squared += float(torch.sum(grad * grad).item())
    return math.sqrt(squared)


def _groups(records: Sequence[TextRecord], size: int) -> Iterable[Sequence[TextRecord]]:
    for start in range(0, len(records), size):
        yield records[start : start + size]


def _evaluate_documents(
    model: TwelveSixDecoder, tok: ByteTokenizer, records: Sequence[TextRecord], horizon: int
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    documents: list[dict[str, Any]] = []
    total_nll = 0.0
    total_targets = 0
    try:
        with torch.no_grad():
            for record in records:
                nll = 0.0
                targets = 0
                blocks = 0
                for example in iter_packed_examples(
                    (record,), tok, expected_split=record.split, sequence_length=horizon,
                    cross_document=False,
                ):
                    input_ids, labels, count, _ = _trimmed_batch(example)
                    value = float(_summed_nll(model(input_ids).logits, labels).item())
                    nll += value
                    targets += count
                    blocks += 1
                if targets != _record_target_count(tok, record):
                    raise Model197Error("heldout causal-target coverage drift")
                documents.append({
                    "record_id": record.record_id,
                    "byte_tokens": len(tok.encode(record.text)),
                    "targets": targets,
                    "nll": nll,
                    "bits_per_byte": nll / targets / math.log(2.0),
                    "blocks": blocks,
                })
                total_nll += nll
                total_targets += targets
    finally:
        model.train(was_training)
    return {
        "horizon": horizon,
        "documents": documents,
        "nll": total_nll,
        "targets": total_targets,
        "bits_per_byte": total_nll / total_targets / math.log(2.0),
    }


class _MemoryBackend:
    eos_token_id = None

    def __init__(self, model: TwelveSixDecoder, tok: ByteTokenizer, horizon: int) -> None:
        self.model = model
        self.tok = tok
        self.max_context_tokens = horizon

    def encode(self, text: str) -> list[int]:
        return list(self.tok.encode(text))

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tok.decode(token_ids)

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        if not input_ids or len(input_ids) > self.max_context_tokens:
            raise ValueError("probe prefix outside trained context horizon")
        with torch.no_grad():
            tensor = torch.tensor([list(input_ids)], dtype=torch.long)
            return self.model(tensor).logits[0, -1].detach().cpu().tolist()


def _long_dependency(model: TwelveSixDecoder, tok: ByteTokenizer, horizon: int) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    try:
        backend = _MemoryBackend(model, tok, horizon)
        suite = materialize_suite(backend)
        report = score_suite(backend, suite, model_label=f"MODEL-197-context-{horizon}")
    finally:
        model.train(was_training)
    return {
        "suite_id": report["suite_id"],
        "suite_identity_sha256": report["suite_identity_sha256"],
        "materialized_identity_sha256": report["materialized_identity_sha256"],
        "evaluation": report["evaluation"],
        "by_distance": report["by_distance"],
        "interpretation": report["interpretation"],
        "probe_ceiling_tokens": 512,
    }


def run_arm(repo: Path, out: Path, horizon: int) -> dict[str, Any]:
    if horizon not in HORIZONS:
        raise Model197Error(f"unsupported MODEL-197 horizon: {horizon}")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    plan = _load_plan(out)
    manifest = json.loads((out / "corpus-manifest.json").read_text(encoding="utf-8"))
    if manifest["corpus_identity_sha256"] != plan["corpus_identity_sha256"]:
        raise Model197Error("corpus identity changed after preregistration")
    corpus = out / "corpus-a"
    train_ids = [row["record_id"] for row in plan["training_documents"]]
    heldout_ids = [row["record_id"] for row in plan["heldout_documents"]]
    train_records = _ordered_records(corpus, manifest, "train", train_ids)
    heldout_records = _ordered_records(corpus, manifest, "validation", heldout_ids)
    tok = ByteTokenizer()
    spec, init = _model(repo)
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, init)
    initial_hash = _state_hash(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY
    )
    grad_norms: list[float] = []
    clipped = 0
    nonfinite = 0
    train_nll = 0.0
    train_targets = 0
    raw_positions = 0
    trimmed_positions = 0
    blocks = 0
    start = time.perf_counter()
    for group in _groups(train_records, DOCS_PER_UPDATE):
        group_targets = sum(_record_target_count(tok, record) for record in group)
        if group_targets <= 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        group_nll = 0.0
        observed_targets = 0
        for record in group:
            for example in iter_packed_examples(
                (record,), tok, expected_split="train", sequence_length=horizon,
                cross_document=False,
            ):
                input_ids, labels, targets, positions = _trimmed_batch(example)
                loss_sum = _summed_nll(model(input_ids).logits, labels)
                (loss_sum / group_targets).backward()
                group_nll += float(loss_sum.detach().item())
                observed_targets += targets
                raw_positions += horizon
                trimmed_positions += positions
                blocks += 1
        if observed_targets != group_targets:
            raise Model197Error("context arm changed optimized causal-target coverage")
        grad_norm = _global_grad_norm(model.parameters())
        if not math.isfinite(grad_norm):
            nonfinite += 1
            raise Model197Error("non-finite gradient norm before clipping")
        grad_norms.append(grad_norm)
        if grad_norm > CLIP_NORM:
            clipped += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM, error_if_nonfinite=True)
        optimizer.step()
        train_nll += group_nll
        train_targets += group_targets
    wall = time.perf_counter() - start
    if train_targets != plan["realized_optimized_tokens"]:
        raise Model197Error("arm optimized-token count differs from preregistered plan")
    common_short = _evaluate_documents(model, tok, heldout_records, COMMON_SHORT_HORIZON)
    native = _evaluate_documents(model, tok, heldout_records, horizon)
    dependency = _long_dependency(model, tok, horizon)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "plan_sha256": plan["plan_sha256"],
        "horizon": horizon,
        "model_spec_sha256": spec.identity_sha256(),
        "parameter_count": spec.parameter_count(),
        "initial_state_sha256": initial_hash,
        "final_state_sha256": _state_hash(model),
        "optimized_tokens": train_targets,
        "optimizer_updates": len(grad_norms),
        "training": {
            "bits_per_byte": train_nll / train_targets / math.log(2.0),
            "wall_seconds": wall,
            "optimized_tokens_per_second": train_targets / wall,
            "trimmed_tensor_positions_per_second": trimmed_positions / wall,
        },
        "packing": {
            "blocks": blocks,
            "raw_tensor_positions": raw_positions,
            "trimmed_tensor_positions": trimmed_positions,
            "right_trim_reduction_fraction": 1.0 - trimmed_positions / raw_positions,
            "raw_input_utilization": trimmed_positions / raw_positions,
            "loss_target_per_raw_position": train_targets / raw_positions,
            "loss_target_per_trimmed_position": train_targets / trimmed_positions,
        },
        "gradients": {
            "preclip_global_l2_mean": statistics.fmean(grad_norms),
            "preclip_global_l2_median": statistics.median(grad_norms),
            "preclip_global_l2_p95": _percentile(grad_norms, 0.95),
            "preclip_global_l2_max": max(grad_norms),
            "clip_count": clipped,
            "clip_rate": clipped / len(grad_norms),
            "nonfinite_count": nonfinite,
        },
        "memory": {
            "device": "cpu",
            "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "scope": "CPU_PROCESS_HIGH_WATER; NOT GPU MEMORY EVIDENCE",
        },
        "evaluation": {
            "common_short": common_short,
            "native": native,
            "long_dependency": dependency,
        },
        "truth_boundary": {
            "systems_metrics_are_cpu_specific": True,
            "gpu_retest_required_for_accelerator_throughput_memory": True,
            "rope_math_used_as_capability_evidence": False,
            "1024_dependency_claim_beyond_512": False,
            "single_initialization_seed": SEED,
        },
        "paid_compute": False,
    }
    _write_json(out / f"arm-{horizon}.json", result)
    return result


def _aggregate_rows(rows: Sequence[dict[str, Any]], ids: set[str] | None = None) -> float:
    chosen = [row for row in rows if ids is None or row["record_id"] in ids]
    if not chosen:
        raise Model197Error("bootstrap subset is empty")
    return sum(float(row["nll"]) for row in chosen) / sum(int(row["targets"]) for row in chosen) / math.log(2.0)


def _paired_bootstrap(
    a_rows: Sequence[dict[str, Any]], b_rows: Sequence[dict[str, Any]], *, min_bytes: int = 0
) -> dict[str, Any]:
    a = {row["record_id"]: row for row in a_rows if int(row["byte_tokens"]) > min_bytes}
    b = {row["record_id"]: row for row in b_rows if int(row["byte_tokens"]) > min_bytes}
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"eligible": False, "documents": 0, "min_byte_tokens_exclusive": min_bytes}
    observed = _aggregate_rows([b[i] for i in ids]) - _aggregate_rows([a[i] for i in ids])
    rng = random.Random(BOOTSTRAP_SEED + min_bytes + len(ids))
    deltas: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        a_nll = sum(float(a[i]["nll"]) for i in sample)
        a_tok = sum(int(a[i]["targets"]) for i in sample)
        b_nll = sum(float(b[i]["nll"]) for i in sample)
        b_tok = sum(int(b[i]["targets"]) for i in sample)
        deltas.append((b_nll / b_tok - a_nll / a_tok) / math.log(2.0))
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]
    return {
        "eligible": True,
        "documents": len(ids),
        "min_byte_tokens_exclusive": min_bytes,
        "delta_b_minus_a_bpb": observed,
        "ci95": [lo, hi],
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED + min_bytes + len(ids),
    }


def summarize(out: Path) -> dict[str, Any]:
    plan = _load_plan(out)
    arms = {h: json.loads((out / f"arm-{h}.json").read_text(encoding="utf-8")) for h in HORIZONS}
    invariants = {
        "plan_identity_equal": len({arms[h]["plan_sha256"] for h in HORIZONS}) == 1,
        "model_identity_equal": len({arms[h]["model_spec_sha256"] for h in HORIZONS}) == 1,
        "parameter_count_equal": len({arms[h]["parameter_count"] for h in HORIZONS}) == 1,
        "initial_weights_equal": len({arms[h]["initial_state_sha256"] for h in HORIZONS}) == 1,
        "optimized_tokens_equal": len({arms[h]["optimized_tokens"] for h in HORIZONS}) == 1,
        "optimizer_updates_equal": len({arms[h]["optimizer_updates"] for h in HORIZONS}) == 1,
    }
    if not all(invariants.values()):
        raise Model197Error(f"matched-comparison invariant failed: {invariants}")
    comparisons: dict[str, Any] = {}
    for a, b in ((256, 512), (512, 1024), (256, 1024)):
        comparisons[f"{a}_to_{b}"] = {
            "common_short": _paired_bootstrap(
                arms[a]["evaluation"]["common_short"]["documents"],
                arms[b]["evaluation"]["common_short"]["documents"],
            ),
            "native_all": _paired_bootstrap(
                arms[a]["evaluation"]["native"]["documents"],
                arms[b]["evaluation"]["native"]["documents"],
            ),
            "native_gt256": _paired_bootstrap(
                arms[a]["evaluation"]["native"]["documents"],
                arms[b]["evaluation"]["native"]["documents"], min_bytes=256,
            ),
            "native_gt512": _paired_bootstrap(
                arms[a]["evaluation"]["native"]["documents"],
                arms[b]["evaluation"]["native"]["documents"], min_bytes=512,
            ),
        }
    decision = "INSUFFICIENT_EVIDENCE"
    c512 = comparisons["256_to_512"]["native_gt256"]
    c1024 = comparisons["512_to_1024"]["native_gt512"]
    short512 = comparisons["256_to_512"]["common_short"]
    short1024 = comparisons["512_to_1024"]["common_short"]
    gain512 = c512.get("eligible") and c512["ci95"][1] < 0 and short512["ci95"][1] <= 0
    gain1024 = c1024.get("eligible") and c1024["ci95"][1] < 0 and short1024["ci95"][1] <= 0
    dep512 = arms[512]["evaluation"]["long_dependency"]["interpretation"].get("usable_long_dependency_claim", False)
    if gain512 and dep512:
        decision = "SELECT_512_OVER_256"
        if gain1024:
            decision = "1024_NATIVE_LIKELIHOOD_GAIN_BUT_GT512_DEPENDENCY_UNPROVEN"
    elif gain1024:
        decision = "1024_NATIVE_LIKELIHOOD_GAIN_WITHOUT_512_CHAIN_EVIDENCE"
    else:
        no512 = c512.get("eligible") and c512["ci95"][0] >= 0
        no1024 = c1024.get("eligible") and c1024["ci95"][0] >= 0
        if no512 and no1024:
            decision = "KEEP_256_NO_LONGER_CONTEXT_BPB_GAIN"
    summary = {
        "schema": SUMMARY_SCHEMA,
        "authority": AUTHORITY,
        "plan_sha256": plan["plan_sha256"],
        "invariants": invariants,
        "arms": {
            str(h): {
                "training_bpb": arms[h]["training"]["bits_per_byte"],
                "heldout_common_short_bpb": arms[h]["evaluation"]["common_short"]["bits_per_byte"],
                "heldout_native_bpb": arms[h]["evaluation"]["native"]["bits_per_byte"],
                "packing": arms[h]["packing"],
                "throughput": arms[h]["training"],
                "memory": arms[h]["memory"],
                "gradients": arms[h]["gradients"],
                "long_dependency_interpretation": arms[h]["evaluation"]["long_dependency"]["interpretation"],
            }
            for h in HORIZONS
        },
        "paired_document_bootstrap": comparisons,
        "decision": decision,
        "decision_contract": {
            "longer_context_gain_requires_ci95_upper_below_zero_on_relevant_long_document_subset": True,
            "common_short_non_regression_requires_ci95_upper_at_or_below_zero": True,
            "512_capability_requires_interpretable_positive_EVAL135_signal": True,
            "1024_general_dependency_promotion_blocked_because_EVAL135_v1_stops_at_512": True,
        },
        "truth_boundary": {
            "single_seed": True,
            "cpu_specific_systems_metrics": True,
            "gpu_retest_preregistered": True,
            "gpu_retest_must_repeat_256_512_1024_on_same_accelerator_with_same_plan": True,
            "no_paid_compute": True,
        },
    }
    _write_json(out / "model197-summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="MODEL-197 matched 10M context transfer")
    sub = parser.add_subparsers(dest="command", required=True)
    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--repo", type=Path, default=Path("."))
    p_prepare.add_argument("--out", type=Path, required=True)
    p_prepare.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    p_arm = sub.add_parser("run-arm")
    p_arm.add_argument("--repo", type=Path, default=Path("."))
    p_arm.add_argument("--out", type=Path, required=True)
    p_arm.add_argument("--horizon", type=int, required=True, choices=HORIZONS)
    p_summary = sub.add_parser("summarize")
    p_summary.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.repo.resolve(), args.out.resolve(), args.target_tokens)
    elif args.command == "run-arm":
        result = run_arm(args.repo.resolve(), args.out.resolve(), args.horizon)
    else:
        result = summarize(args.out.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
