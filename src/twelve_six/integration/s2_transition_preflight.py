"""Bounded S2 mechanics evidence for the post-S0 engineering transition."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from twelve_six.model import TwelveSixDecoder, count_trainable_parameters, load_stage_config

AUTHORITY = "ENGINEERING_S2_MECHANICS_PREFLIGHT_ONLY_NOT_STAGE_EVIDENCE"
SCOPE = "SYNTHETIC_TOKEN_IDS_ONLY_NOT_S2_DATA_OR_TOKENIZER"


def collect_s2_transition_preflight(
    stage_config: str | Path = "configs/stages/s2_1m.json",
    *,
    seed: int = 20260825,
    sequence_length: int = 16,
) -> dict[str, Any]:
    """Instantiate the current S2 geometry and prove a finite backward pass.

    This deliberately performs no optimizer step and makes no data, tokenizer,
    quality, capability, promotion, or architecture-freeze claim.
    """

    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")

    config = load_stage_config(stage_config)
    if config.stage != "S2":
        raise ValueError(f"expected S2 stage config, got {config.stage!r}")
    if config.target_parameters != 1_000_000:
        raise ValueError("S2 transition preflight expects the 1M target stage")
    if config.canonical_base != "random_init":
        raise ValueError("S2 Base must remain random_init")
    if sequence_length > config.model.max_seq_len:
        raise ValueError("sequence_length exceeds S2 max_seq_len")

    torch.manual_seed(seed)
    model = TwelveSixDecoder(config.model, config.init)
    model.train()

    input_ids = (
        torch.arange(sequence_length, dtype=torch.long).unsqueeze(0) % config.model.vocab_size
    )
    logits = model(input_ids).logits
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, config.model.vocab_size),
        input_ids[:, 1:].reshape(-1),
    )
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    all_gradients_finite = bool(gradients) and all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )
    any_gradient_nonzero = any(bool(torch.count_nonzero(gradient).item()) for gradient in gradients)
    loss_value = float(loss.detach().item())

    if not math.isfinite(loss_value):
        raise RuntimeError("S2 preflight loss is not finite")
    if not all_gradients_finite:
        raise RuntimeError("S2 preflight gradients are not all finite")
    if not any_gradient_nonzero:
        raise RuntimeError("S2 preflight produced no nonzero gradient")

    return {
        "schema": "12-6.s2-transition-preflight.v1",
        "authority": AUTHORITY,
        "scope": SCOPE,
        "stage": config.stage,
        "target_parameters": config.target_parameters,
        "expected_parameters": config.expected_parameters,
        "actual_trainable_parameters": count_trainable_parameters(model),
        "model_identity_sha256": config.model.identity_sha256(),
        "init_identity_sha256": config.init.identity_sha256(),
        "vocab_size": config.model.vocab_size,
        "max_seq_len": config.model.max_seq_len,
        "sequence_length": sequence_length,
        "logits_shape": list(logits.shape),
        "loss": loss_value,
        "all_gradients_finite": all_gradients_finite,
        "any_gradient_nonzero": any_gradient_nonzero,
        "optimizer_steps": 0,
        "tokenizer_selected": False,
        "data_selected": False,
        "architecture_frozen": False,
        "quality_claim": False,
        "promotion_allowed": False,
        "paid_compute": False,
        "canonical_base": config.canonical_base,
    }
