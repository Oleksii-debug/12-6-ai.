"""Transparent first-order memory estimates for scale planning.

This is an estimator, not measured allocator telemetry. It is deliberately explicit about its
assumptions so later profiling can replace coefficients with measured values. Context-length
planning must not hide the quadratic dense-attention term behind a linear B*S*H*L approximation.
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
    linear_activation_bytes_per_rank: int
    attention_score_equivalent_bytes_per_rank: int
    activation_bytes_per_rank: int
    total_bytes_per_rank: int

    @property
    def total_gib_per_rank(self) -> float:
        return self.total_bytes_per_rank / GIB


def estimate_training_memory(
    model: ModelScaleSpec,
    plan: ParallelPlan,
    *,
    parameter_bytes: int = 2,
    gradient_bytes: int = 2,
    optimizer_bytes_per_parameter: int = 8,
    master_weight_bytes: int = 4,
    activation_bytes: int = 2,
    activation_multiplier: float = 8.0,
    attention_memory_mode: str = "materialized_score_equivalent",
    attention_score_multiplier: float = 1.0,
) -> MemoryEstimate:
    """Estimate per-rank memory using an explicit dense Adam-like state model.

    Defaults approximate bf16/fp16 parameters and gradients, fp32 master weights, and two fp32
    Adam moments. ``linear_activation_bytes_per_rank`` retains the earlier coarse B*S*H*L term.

    Dense attention also has O(S^2) arithmetic and a classical materialized score tensor shaped
    B*L*heads*S*S. By default this estimator adds one such score-tensor *equivalent* to the
    planning total so increasing context cannot look linear by accident. This is deliberately
    conservative for fused SDPA/FlashAttention kernels, which can avoid materializing the full
    score matrix in HBM. It is not measured peak allocator memory and must be replaced by target-
    hardware profiling before a paid-run capacity claim.

    Set ``attention_memory_mode='linear_only'`` only for backward-compatible historical
    comparisons, not for long-context capacity planning.

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
    if attention_score_multiplier < 0:
        raise ValueError("attention_score_multiplier must be non-negative")
    if attention_memory_mode not in {"materialized_score_equivalent", "linear_only"}:
        raise ValueError(
            "attention_memory_mode must be 'materialized_score_equivalent' or 'linear_only'"
        )

    state_factor = plan.model_state_shard_factor
    activation_factor = plan.activation_shard_factor
    params_per_rank = (model.total_parameters + state_factor - 1) // state_factor

    parameter_total = params_per_rank * parameter_bytes
    gradient_total = params_per_rank * gradient_bytes
    optimizer_total = params_per_rank * optimizer_bytes_per_parameter
    master_total = params_per_rank * master_weight_bytes

    raw_linear_activations = (
        model.micro_batch_size
        * model.sequence_length
        * model.hidden_size
        * model.num_layers
        * activation_bytes
        * activation_multiplier
    )
    linear_activation_total = int(raw_linear_activations / activation_factor)

    attention_score_equivalent_total = 0
    if attention_memory_mode == "materialized_score_equivalent":
        raw_attention_scores = (
            model.micro_batch_size
            * model.num_layers
            * model.num_attention_heads
            * model.sequence_length
            * model.sequence_length
            * activation_bytes
            * attention_score_multiplier
        )
        attention_score_equivalent_total = int(raw_attention_scores / activation_factor)

    activation_total = linear_activation_total + attention_score_equivalent_total
    total = parameter_total + gradient_total + optimizer_total + master_total + activation_total
    return MemoryEstimate(
        parameter_bytes_per_rank=parameter_total,
        gradient_bytes_per_rank=gradient_total,
        optimizer_bytes_per_rank=optimizer_total,
        master_weight_bytes_per_rank=master_total,
        linear_activation_bytes_per_rank=linear_activation_total,
        attention_score_equivalent_bytes_per_rank=attention_score_equivalent_total,
        activation_bytes_per_rank=activation_total,
        total_bytes_per_rank=total,
    )