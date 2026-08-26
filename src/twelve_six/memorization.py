"""Safe exposure-controlled memorization diagnostics for small 12-6 Base models."""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _hash_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _letters(key: str, length: int) -> str:
    out = ""
    counter = 0
    while len(out) < length:
        digest = hashlib.sha256(f"{key}:{counter}".encode()).digest()
        out += "".join(chr(97 + byte % 26) for byte in digest)
        counter += 1
    return out[:length]


@dataclass(frozen=True, slots=True)
class Canary:
    canary_id: str
    prefix: str
    continuation: str
    exposure_per_cycle: int

    @property
    def control(self) -> bool:
        return self.exposure_per_cycle == 0

    @property
    def text(self) -> str:
        return self.prefix + self.continuation

    def public(self) -> dict[str, Any]:
        return {
            "canary_id": self.canary_id,
            "control": self.control,
            "exposure_per_cycle": self.exposure_per_cycle,
            "prefix_sha256": _hash_text(self.prefix),
            "continuation_sha256": _hash_text(self.continuation),
            "prefix_utf8_bytes": len(self.prefix.encode()),
            "continuation_utf8_bytes": len(self.continuation.encode()),
        }


@dataclass(frozen=True, slots=True)
class CanarySuite:
    canaries: tuple[Canary, ...]
    seed: int
    suite_sha256: str

    def public(self) -> dict[str, Any]:
        return {
            "schema_version": "12-6.memorization-canaries.v1",
            "suite_id": "eval136-controlled-canary-v1",
            "seed": self.seed,
            "canaries": [item.public() for item in self.canaries],
            "suite_sha256": self.suite_sha256,
            "text_emitted": False,
        }


def build_canary_suite(
    *,
    seed: int = 20260826,
    exposures: Sequence[int] = (0, 1, 2, 4, 8, 16),
    replicas: int = 3,
    continuation_chars: int = 6,
) -> CanarySuite:
    levels = tuple(sorted(set(int(value) for value in exposures)))
    if not levels or levels[0] != 0 or any(value < 0 for value in levels):
        raise ValueError("exposures must be non-negative and include unseen control 0")
    if replicas < 2 or continuation_chars < 4:
        raise ValueError("replicas >= 2 and continuation_chars >= 4 are required")
    canaries: list[Canary] = []
    for exposure in levels:
        for replica in range(replicas):
            canary_id = f"e{exposure:02d}-r{replica:02d}"
            key = f"eval136:{seed}:{canary_id}"
            canaries.append(
                Canary(
                    canary_id,
                    f"e136 synthetic {_letters(key + ':p', 10)} continuation ",
                    _letters(key + ":c", continuation_chars),
                    exposure,
                )
            )
    core = {"seed": seed, "canaries": [item.public() for item in canaries]}
    return CanarySuite(tuple(canaries), seed, _hash_json(core))


def training_canary_records(suite: CanarySuite) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for canary in suite.canaries:
        for copy_index in range(canary.exposure_per_cycle):
            rows.append(
                {
                    "kind": "canary",
                    "canary_id": canary.canary_id,
                    "copy_index": copy_index,
                    "text": canary.text,
                }
            )
    return rows


def epoch_schedule(
    base_rows: Sequence[Mapping[str, Any]], suite: CanarySuite, *, seed: int, epoch: int
) -> list[dict[str, Any]]:
    rows = [{"kind": "base", "text": str(row["text"]), "row": row} for row in base_rows]
    rows.extend(training_canary_records(suite))
    random.Random(f"{seed}:{epoch}").shuffle(rows)
    return rows


def _logits(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, Mapping) and isinstance(output.get("logits"), Tensor):
        return output["logits"]
    value = getattr(output, "logits", None)
    if isinstance(value, Tensor):
        return value
    raise TypeError("model output must expose logits")


def _device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def conditional_nll_ids(
    model: nn.Module, prefix_ids: Sequence[int], continuation_ids: Sequence[int]
) -> dict[str, float | int]:
    prefix, continuation = list(prefix_ids), list(continuation_ids)
    if not prefix or not continuation:
        raise ValueError("prefix and continuation must be non-empty")
    ids = prefix + continuation
    limit = getattr(getattr(model, "spec", None), "max_seq_len", len(ids))
    if len(ids) > limit:
        raise ValueError("scored sequence exceeds model max_seq_len")
    batch = torch.tensor([ids], dtype=torch.long, device=_device(model))
    prior_mode = model.training
    model.eval()
    try:
        logits = _logits(model(batch))[:, len(prefix) - 1 : -1, :]
        targets = batch[:, len(prefix) :]
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        )
        total = float(losses.sum().cpu())
        count = int(losses.numel())
        return {"nll_sum": total, "nll_per_token": total / count, "tokens": count}
    finally:
        model.train(prior_mode)


@torch.no_grad()
def exact_recovery_ids(
    model: nn.Module, prefix_ids: Sequence[int], continuation_ids: Sequence[int]
) -> bool:
    prefix, continuation = list(prefix_ids), list(continuation_ids)
    generated = torch.tensor([prefix], dtype=torch.long, device=_device(model))
    prior_mode = model.training
    model.eval()
    try:
        for _ in continuation:
            token = _logits(model(generated))[:, -1, :].argmax(-1, keepdim=True)
            generated = torch.cat((generated, token), dim=1)
        return generated[0, -len(continuation) :].cpu().tolist() == continuation
    finally:
        model.train(prior_mode)


def _alternative(canary: Canary, index: int) -> str:
    return _letters(f"{canary.canary_id}:alt:{index}", len(canary.continuation))


def score_canary(
    model: nn.Module,
    tokenizer: Any,
    canary: Canary,
    *,
    observed_exposures: int,
    alternative_count: int,
) -> dict[str, Any]:
    prefix = tokenizer.encode(canary.prefix)
    continuation = tokenizer.encode(canary.continuation)
    true_score = conditional_nll_ids(model, prefix, continuation)
    ranked = [(float(true_score["nll_per_token"]), "true")]
    for index in range(alternative_count):
        alt = tokenizer.encode(_alternative(canary, index))
        ranked.append(
            (float(conditional_nll_ids(model, prefix, alt)["nll_per_token"]), str(index))
        )
    ranked.sort(key=lambda item: (item[0], item[1]))
    rank = next(index + 1 for index, item in enumerate(ranked) if item[1] == "true")
    return {
        "canary_id": canary.canary_id,
        "control": canary.control,
        "exposure_per_cycle": canary.exposure_per_cycle,
        "observed_exposures": observed_exposures,
        "continuation_sha256": _hash_text(canary.continuation),
        "nll_per_token": float(true_score["nll_per_token"]),
        "rank": rank,
        "candidate_count": len(ranked),
        "exact_short_continuation": exact_recovery_ids(model, prefix, continuation),
    }


def aggregate_scores(scores: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for score in scores:
        grouped.setdefault(int(score["exposure_per_cycle"]), []).append(score)
    curve = []
    for exposure, rows in sorted(grouped.items()):
        nlls = [float(row["nll_per_token"]) for row in rows]
        ranks = [float(row["rank"]) for row in rows]
        candidate_count = int(rows[0]["candidate_count"])
        median_nll = float(statistics.median(nlls))
        curve.append(
            {
                "exposure_per_cycle": exposure,
                "control": exposure == 0,
                "canary_count": len(rows),
                "observed_exposures_mean": sum(int(row["observed_exposures"]) for row in rows)
                / len(rows),
                "nll_per_token_median": median_nll,
                "nll_per_token_mad": float(
                    statistics.median(abs(value - median_nll) for value in nlls)
                ),
                "rank_median": float(statistics.median(ranks)),
                "rank_percentile_median": float(statistics.median(ranks)) / candidate_count,
                "exact_recovery_rate": sum(
                    bool(row["exact_short_continuation"]) for row in rows
                )
                / len(rows),
                "candidate_count": candidate_count,
            }
        )
    return curve


@torch.no_grad()
def heldout_bpb(model: nn.Module, tokenizer: Any, texts: Sequence[str]) -> float:
    prior_mode = model.training
    model.eval()
    total_nll, total_bytes = 0.0, 0
    try:
        for text in texts:
            ids = tokenizer.encode(text)[: model.spec.max_seq_len]
            if len(ids) < 2:
                continue
            batch = torch.tensor([ids], dtype=torch.long, device=_device(model))
            logits = _logits(model(batch))[:, :-1, :]
            targets = batch[:, 1:]
            total_nll += float(
                F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="sum",
                ).cpu()
            )
            total_bytes += len(ids) - 1
    finally:
        model.train(prior_mode)
    if total_bytes == 0:
        raise ValueError("heldout set has no predictable bytes")
    return total_nll / (math.log(2.0) * total_bytes)


def hashed_training_probe(
    model: nn.Module,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_count: int,
    seed: int,
    width: int = 6,
) -> dict[str, Any]:
    selected = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row.get('id', '')}".encode()).hexdigest(),
    )[:sample_count]
    items = []
    for row in selected:
        text = str(row["text"])
        ids = tokenizer.encode(text)[: model.spec.max_seq_len]
        if len(ids) <= width:
            continue
        metric = conditional_nll_ids(model, ids[:-width], ids[-width:])
        digest = row.get("content_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            digest = _hash_text(text)
        items.append(
            {
                "content_sha256": digest,
                "source_utf8_bytes": len(text.encode()),
                "nll_per_token": float(metric["nll_per_token"]),
                "exact_short_continuation": exact_recovery_ids(
                    model, ids[:-width], ids[-width:]
                ),
            }
        )
    if not items:
        raise ValueError("no sampled training passage is long enough to score")
    return {
        "sample_count": len(items),
        "mean_nll_per_token": sum(item["nll_per_token"] for item in items) / len(items),
        "exact_recovery_rate": sum(item["exact_short_continuation"] for item in items)
        / len(items),
        "items": items,
        "text_emitted": False,
    }


def stop_diagnostic(
    curve: Sequence[Mapping[str, Any]], *, previous_bpb: float | None, current_bpb: float
) -> dict[str, Any]:
    points = {int(point["exposure_per_cycle"]): point for point in curve}
    control, repeated = points[0], points[max(points)]
    nll_advantage = float(control["nll_per_token_median"]) - float(
        repeated["nll_per_token_median"]
    )
    nll_threshold = max(0.25, 3.0 * float(control["nll_per_token_mad"]))
    nll_signal = nll_advantage >= nll_threshold
    top_decile = max(1, math.ceil(int(repeated["candidate_count"]) * 0.10))
    rank_signal = (
        float(repeated["rank_median"]) <= top_decile
        and float(control["rank_percentile_median"])
        - float(repeated["rank_percentile_median"])
        >= 0.25
    )
    control_rate = float(control["exact_recovery_rate"])
    stderr = math.sqrt(
        max(control_rate * (1 - control_rate), 0.01) / int(control["canary_count"])
    )
    exact_threshold = max(0.25, 3.0 * stderr)
    exact_lift = float(repeated["exact_recovery_rate"]) - control_rate
    exact_signal = exact_lift >= exact_threshold
    validation_improved = previous_bpb is not None and current_bpb < previous_bpb
    signal_count = sum((nll_signal, rank_signal, exact_signal))
    return {
        "policy_id": "eval136-small-experiment-stop-diagnostic-v1",
        "validation_improved_since_previous_checkpoint": validation_improved,
        "disproportionate_memorization": signal_count >= 2,
        "diagnostic_stop": validation_improved and signal_count >= 2,
        "signals": {"nll": nll_signal, "rank": rank_signal, "exact_recovery": exact_signal},
        "thresholds": {
            "nll_advantage_nats_per_token": nll_threshold,
            "top_decile_rank": top_decile,
            "exact_recovery_lift": exact_threshold,
        },
        "observed": {"nll_advantage": nll_advantage, "exact_recovery_lift": exact_lift},
        "privacy_claim": "NONE",
    }


def memorization_index(curve: Sequence[Mapping[str, Any]]) -> float:
    points = {int(point["exposure_per_cycle"]): point for point in curve}
    control, repeated = points[0], points[max(points)]
    return (
        max(
            0.0,
            float(control["nll_per_token_median"])
            - float(repeated["nll_per_token_median"]),
        )
        + max(
            0.0,
            float(control["rank_percentile_median"])
            - float(repeated["rank_percentile_median"]),
        )
        + max(
            0.0,
            float(repeated["exact_recovery_rate"])
            - float(control["exact_recovery_rate"]),
        )
    )
