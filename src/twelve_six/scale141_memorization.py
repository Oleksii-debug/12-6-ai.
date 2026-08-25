"""Hash-only memorization diagnostic for SCALE-141's unchanged DATA-25 corpus.

SCALE-141 must reconstruct the exact corpus rather than inject new canaries into the
training stream. This probe therefore reuses the safe EVAL-136 non-canary idea:
deterministically sample real project-owned training records, score only a short
continuation, and emit hashes/aggregate metrics rather than source text.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

PROBE_ID = "scale141-hash-only-train-continuation-v1"
WIDTH = 6
SAMPLES_PER_MODALITY = 6


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _content_hash(row: Mapping[str, Any], text: str) -> str:
    for key in ("content_sha256", "text_sha256", "sha256"):
        value = row.get(key)
        if isinstance(value, str) and len(value) == 64:
            return value
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_id(row: Mapping[str, Any]) -> str:
    for key in ("record_id", "id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return _content_hash(row, str(row.get("text", "")))


def _conditional_metrics(
    model: torch.nn.Module,
    token_ids: Sequence[int],
    *,
    width: int,
) -> tuple[float, bool]:
    if len(token_ids) <= width:
        raise ValueError("probe sequence must contain prefix plus continuation")
    prefix_length = len(token_ids) - width
    device = _model_device(model)
    full = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    logits = model(full).logits
    continuation_logits = logits[:, prefix_length - 1 : -1, :]
    continuation_targets = full[:, prefix_length:]
    nll = F.cross_entropy(
        continuation_logits.reshape(-1, continuation_logits.shape[-1]),
        continuation_targets.reshape(-1),
        reduction="mean",
    )

    generated = full[:, :prefix_length]
    for _ in range(width):
        next_token = model(generated).logits[:, -1, :].argmax(-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
    exact = bool(
        torch.equal(
            generated[0, -width:].detach().cpu(),
            continuation_targets[0].detach().cpu(),
        )
    )
    return float(nll.detach().cpu()), exact


@torch.no_grad()
def hashed_training_probe(
    model: torch.nn.Module,
    tokenizer: Any,
    rows_by_modality: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: int,
    context_tokens: int,
    width: int = WIDTH,
    samples_per_modality: int = SAMPLES_PER_MODALITY,
) -> dict[str, Any]:
    if context_tokens > model.spec.max_seq_len:
        raise ValueError("probe context exceeds model max_seq_len")
    if width < 1 or context_tokens <= width:
        raise ValueError("invalid short-continuation probe geometry")

    before_mode = model.training
    before_state = hashlib.sha256(
        b"".join(
            parameter.detach().cpu().contiguous().numpy().tobytes()
            for parameter in model.parameters()
        )
    ).hexdigest()
    model.eval()
    items: list[dict[str, Any]] = []
    try:
        for modality in ("uk", "en", "code"):
            rows = rows_by_modality[modality]
            selected = sorted(
                rows,
                key=lambda row: hashlib.sha256(
                    f"{seed}:{modality}:{_record_id(row)}".encode("utf-8")
                ).hexdigest(),
            )[:samples_per_modality]
            observed = 0
            for row in selected:
                text = str(row["text"])
                ids = tokenizer.encode(text)[:context_tokens]
                if len(ids) <= width:
                    continue
                nll, exact = _conditional_metrics(model, ids, width=width)
                items.append(
                    {
                        "modality": modality,
                        "content_sha256": _content_hash(row, text),
                        "source_utf8_bytes": len(text.encode("utf-8")),
                        "scored_context_tokens": len(ids),
                        "continuation_tokens": width,
                        "nll_per_token": nll,
                        "bits_per_byte": nll / math.log(2.0),
                        "exact_short_continuation": exact,
                    }
                )
                observed += 1
            if observed == 0:
                raise ValueError(f"no {modality} training passage was long enough to score")
    finally:
        model.train(before_mode)

    after_state = hashlib.sha256(
        b"".join(
            parameter.detach().cpu().contiguous().numpy().tobytes()
            for parameter in model.parameters()
        )
    ).hexdigest()
    if after_state != before_state:
        raise RuntimeError("memorization probe mutated model state")

    by_modality: dict[str, Any] = {}
    for modality in ("uk", "en", "code"):
        subset = [item for item in items if item["modality"] == modality]
        by_modality[modality] = {
            "sample_count": len(subset),
            "mean_nll_per_token": sum(item["nll_per_token"] for item in subset) / len(subset),
            "mean_bits_per_byte": sum(item["bits_per_byte"] for item in subset) / len(subset),
            "exact_short_continuation_rate": sum(bool(item["exact_short_continuation"]) for item in subset) / len(subset),
        }

    return {
        "probe_id": PROBE_ID,
        "scope": "HASH_ONLY_PROJECT_OWNED_TRAINING_PASSAGE_DIAGNOSTIC",
        "sample_count": len(items),
        "continuation_tokens": width,
        "context_token_cap": context_tokens,
        "mean_nll_per_token": sum(item["nll_per_token"] for item in items) / len(items),
        "mean_bits_per_byte": sum(item["bits_per_byte"] for item in items) / len(items),
        "exact_short_continuation_rate": sum(bool(item["exact_short_continuation"]) for item in items) / len(items),
        "by_modality": by_modality,
        "items": items,
        "text_emitted": False,
        "model_non_mutation_passed": True,
        "privacy_leakage_claim": "NONE",
        "canary_injection": False,
        "canary_injection_reason": "SCALE-141 preserves the exact reconstructed DATA-25 corpus",
    }
