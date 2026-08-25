"""Transparent first-order memory estimates for scale planning.

Defaults follow the storage actually used by the current Trainer on its default fp32 path:
fp32 model parameters, fp32 gradients, and two fp32 AdamW moments with no duplicate master
weight tensor. Runtime RSS remains a separate allocator/process measurement and is deliberately
not folded into this analytical estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import ModelScaleSpec, ParallelPlan

GIB = 1024**3


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    parameter_bytes_per_rank: int
    gradient_bytes_per_rank: int
    optimizer_bytes_per_rank: int
    master_weight_bytes_per_rank: int
    activation_bytes_per_rank: int
    total_bytes_per_rank: int

    @property
    def total_gib_per_rank(self) -> float:
        return self.total_bytes_per_rank / GIB


def estimate_training_memory(
    model: ModelScaleSpec,
    plan: ParallelPlan,
    *,
    parameter_bytes: int = 4,
    gradient_bytes: int = 4,
    optimizer_bytes_per_parameter: int = 8,
    master_weight_bytes: int = 0,
    activation_bytes: int = 4,
    activation_multiplier: float = 8.0,
) -> MemoryEstimate:
    """Estimate per-rank memory using an explicit dense AdamW-like state model.

    The defaults are bound to TRAIN-57 observations of the live Trainer: model parameters and
    gradients are fp32 (4 bytes each), AdamW materializes two fp32 moment tensors after the first
    optimizer update (8 bytes per parameter total), and the Trainer does not maintain a separate
    fp32 master-weight copy. Its bf16 mode is autocast, so persistent model/gradient/Adam storage
    remains fp32 there as well.

    Activation memory is still a coarse ``B*S*H*L`` coefficient rather than allocator telemetry.
    Callers planning a different parameter-storage or activation precision must override the
    explicit coefficients instead of relying on these Trainer-bound defaults.

    EP>1 fails closed because total_parameters alone cannot distinguish replicated dense weights
    from expert-only weights. A future MoE-aware estimator must receive that decomposition.
    """

    model.validate()
    plan.validate()
    if plan.expert_parallel > 1:
        raise ValueError(
            "generic memory estimator does not model MoE dense/expert parameter partition; "
            "use expert_parallel=1 or a future MoE-aware estimator"
        )

    coefficients = (
        parameter_bytes,
        gradient_bytes,
        optimizer_bytes_per_parameter,
        master_weight_bytes,
        activation_bytes,
    )
    if any(value < 0 for value in coefficients):
        raise ValueError("memory byte coefficients must be non-negative")
    if activation_multiplier < 0:
        raise ValueError("activation_multiplier must be non-negative")

    state_factor = plan.model_state_shard_factor
    activation_factor = plan.activation_shard_factor
    params_per_rank = (model.total_parameters + state_factor - 1) // state_factor

    parameter_total = params_per_rank * parameter_bytes
    gradient_total = params_per_rank * gradient_bytes
    optimizer_total = params_per_rank * optimizer_bytes_per_parameter
    master_total = params_per_rank * master_weight_bytes
    raw_activations = (
        model.micro_batch_size
        * model.sequence_length
        * model.hidden_size
        * model.num_layers
        * activation_bytes
        * activation_multiplier
    )
    activation_total = int(raw_activations / activation_factor)
    total = parameter_total + gradient_total + optimizer_total + master_total + activation_total
    return MemoryEstimate(
        parameter_bytes_per_rank=parameter_total,
        gradient_bytes_per_rank=gradient_total,
        optimizer_bytes_per_rank=optimizer_total,
        master_weight_bytes_per_rank=master_total,
        activation_bytes_per_rank=activation_total,
        total_bytes_per_rank=total,
    )
