from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/research/r01_model341_compute_envelope_v1.json")
SCHEMA = "12-6.r01-model341-compute-envelope.v1"


class ComputeAccountingError(ValueError):
    """Raised when compute or cost evidence violates a fail-closed invariant."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ComputeAccountingError(f"{name} must be a positive integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComputeAccountingError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ComputeAccountingError(f"{name} must be finite and > 0")
    return result


@dataclass(frozen=True, slots=True)
class DenseDecoderGeometry:
    parameter_count: int
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    context_length: int
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        for name in (
            "parameter_count",
            "vocab_size",
            "d_model",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "head_dim",
            "d_ff",
            "context_length",
        ):
            _positive_int(getattr(self, name), name)
        if self.n_kv_heads > self.n_heads:
            raise ComputeAccountingError("n_kv_heads may not exceed n_heads")
        if self.n_heads * self.head_dim != self.d_model:
            raise ComputeAccountingError("n_heads * head_dim must equal d_model")
        if self.n_heads % self.n_kv_heads:
            raise ComputeAccountingError("n_heads must be divisible by n_kv_heads")
        expected = self.expected_parameter_count
        if self.parameter_count != expected:
            raise ComputeAccountingError(
                f"parameter_count does not match geometry: {self.parameter_count} != {expected}"
            )

    @property
    def embedding_parameters(self) -> int:
        matrices = 1 if self.tie_word_embeddings else 2
        return matrices * self.vocab_size * self.d_model

    @property
    def attention_parameters_per_layer(self) -> int:
        query_width = self.n_heads * self.head_dim
        kv_width = self.n_kv_heads * self.head_dim
        return (
            self.d_model * query_width
            + 2 * self.d_model * kv_width
            + query_width * self.d_model
        )

    @property
    def mlp_parameters_per_layer(self) -> int:
        return 3 * self.d_model * self.d_ff

    @property
    def expected_parameter_count(self) -> int:
        input_embedding = self.vocab_size * self.d_model
        output_head = 0 if self.tie_word_embeddings else input_embedding
        per_layer = (
            self.attention_parameters_per_layer
            + self.mlp_parameters_per_layer
            + 2 * self.d_model
        )
        final_norm = self.d_model
        return input_embedding + output_head + self.n_layers * per_layer + final_norm

    @property
    def non_embedding_parameters(self) -> int:
        return self.parameter_count - self.embedding_parameters

    def six_n_reference_flops_per_token(self) -> int:
        """Return historical 6N planning compute per training token."""
        return 6 * self.parameter_count

    def architecture_aware_flops_per_token(self, *, sequence_length: int) -> int:
        """Estimate training FLOPs/token with an explicit attention-overhead term.

        This uses the DeepSeek scaling-study representation
        M = 6 * N_non_embedding + 12 * L * d_model * sequence_length.
        It remains a planning estimate, not measured accelerator work.
        """
        sequence_length = _positive_int(sequence_length, "sequence_length")
        if sequence_length > self.context_length:
            raise ComputeAccountingError("sequence_length exceeds geometry context_length")
        linear_and_parameter_work = 6 * self.non_embedding_parameters
        attention_work = 12 * self.n_layers * self.d_model * sequence_length
        return linear_and_parameter_work + attention_work


@dataclass(frozen=True, slots=True)
class CampaignEstimate:
    tokens_per_parameter: int
    training_tokens: int
    six_n_reference_total_flops: int
    architecture_aware_total_flops: int
    architecture_to_six_n_ratio: float


def estimate_campaign(
    geometry: DenseDecoderGeometry,
    *,
    tokens_per_parameter: int,
    sequence_length: int,
) -> CampaignEstimate:
    tokens_per_parameter = _positive_int(tokens_per_parameter, "tokens_per_parameter")
    training_tokens = geometry.parameter_count * tokens_per_parameter
    six_n_per_token = geometry.six_n_reference_flops_per_token()
    aware_per_token = geometry.architecture_aware_flops_per_token(
        sequence_length=sequence_length
    )
    return CampaignEstimate(
        tokens_per_parameter=tokens_per_parameter,
        training_tokens=training_tokens,
        six_n_reference_total_flops=six_n_per_token * training_tokens,
        architecture_aware_total_flops=aware_per_token * training_tokens,
        architecture_to_six_n_ratio=aware_per_token / six_n_per_token,
    )


def project_wall_hours(
    *,
    training_tokens: int,
    measured_same_geometry_tokens_per_second: float,
) -> float:
    training_tokens = _positive_int(training_tokens, "training_tokens")
    throughput = _positive_number(
        measured_same_geometry_tokens_per_second,
        "measured_same_geometry_tokens_per_second",
    )
    return training_tokens / throughput / 3600.0


def project_cost_eur(
    *,
    wall_hours: float,
    accelerator_count: int,
    eur_per_accelerator_hour: float,
    same_geometry_measurement: bool,
) -> float:
    if same_geometry_measurement is not True:
        raise ComputeAccountingError(
            "cost projection requires measured same-geometry accelerator throughput"
        )
    wall_hours = _positive_number(wall_hours, "wall_hours")
    accelerator_count = _positive_int(accelerator_count, "accelerator_count")
    rate = _positive_number(eur_per_accelerator_hour, "eur_per_accelerator_hour")
    return wall_hours * accelerator_count * rate


def _geometry_from_config(config: dict[str, Any]) -> DenseDecoderGeometry:
    model = config["model341_geometry"]
    return DenseDecoderGeometry(
        parameter_count=model["parameter_count"],
        vocab_size=model["vocab_size"],
        d_model=model["d_model"],
        n_layers=model["n_layers"],
        n_heads=model["n_heads"],
        n_kv_heads=model["n_kv_heads"],
        head_dim=model["head_dim"],
        d_ff=model["d_ff"],
        context_length=model["context_length"],
        tie_word_embeddings=model["tie_word_embeddings"],
    )


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA:
        raise ComputeAccountingError("unexpected schema_version")
    if config.get("repository") != "Oleksii-debug/12-6-ai.":
        raise ComputeAccountingError("repository drift")
    if config["authority"]["long_training_authorized"] is not False:
        raise ComputeAccountingError("compute envelope must not authorize long training")
    if config["authority"]["paid_compute_authorized"] is not False:
        raise ComputeAccountingError("compute envelope must not authorize paid compute")

    geometry = _geometry_from_config(config)
    sequence_length = config["planning_sequence_length"]
    if sequence_length != geometry.context_length:
        raise ComputeAccountingError("MODEL-341 envelope must use exact context length")

    expected_reference = geometry.six_n_reference_flops_per_token()
    expected_aware = geometry.architecture_aware_flops_per_token(
        sequence_length=sequence_length
    )
    accounting = config["compute_accounting"]
    if accounting["six_n_reference_flops_per_token"] != expected_reference:
        raise ComputeAccountingError("6N per-token reference drift")
    if accounting["architecture_aware_flops_per_token"] != expected_aware:
        raise ComputeAccountingError("architecture-aware per-token estimate drift")

    ratios = config["planned_tokens_per_parameter"]
    if ratios != [10, 20, 40]:
        raise ComputeAccountingError("R01 token-per-parameter measurement grid drift")
    rows = config["model341_campaign_rows"]
    if len(rows) != len(ratios):
        raise ComputeAccountingError("campaign row count drift")
    for ratio, row in zip(ratios, rows, strict=True):
        estimate = estimate_campaign(
            geometry,
            tokens_per_parameter=ratio,
            sequence_length=sequence_length,
        )
        expected = {
            "tokens_per_parameter": estimate.tokens_per_parameter,
            "training_tokens": estimate.training_tokens,
            "six_n_reference_total_flops": estimate.six_n_reference_total_flops,
            "architecture_aware_total_flops": estimate.architecture_aware_total_flops,
        }
        for field, value in expected.items():
            if row[field] != value:
                raise ComputeAccountingError(f"campaign row {ratio} {field} drift")
        if row["projected_wall_hours"] is not None:
            raise ComputeAccountingError("wall hours fabricated without measured throughput")
        if row["projected_cost_eur"] is not None:
            raise ComputeAccountingError("cost fabricated without measured throughput/rate")

    future = config["future_size_targets"]
    for row in future:
        if row["model_spec_frozen"] is not False:
            raise ComputeAccountingError("future ModelSpec falsely frozen")
        if row["architecture_aware_total_flops"] is not None:
            raise ComputeAccountingError("future architecture-aware FLOPs require frozen ModelSpec")
        if row["projected_cost_eur"] is not None:
            raise ComputeAccountingError("future cost fabricated before ModelSpec/throughput")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the R01 MODEL-341 architecture-aware compute envelope."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    geometry = _geometry_from_config(config)
    aware = geometry.architecture_aware_flops_per_token(
        sequence_length=config["planning_sequence_length"]
    )
    reference = geometry.six_n_reference_flops_per_token()
    print(
        json.dumps(
            {
                "schema_version": SCHEMA,
                "planning_valid": True,
                "parameter_count": geometry.parameter_count,
                "six_n_reference_flops_per_token": reference,
                "architecture_aware_flops_per_token": aware,
                "architecture_to_six_n_ratio": aware / reference,
                "paid_compute_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
