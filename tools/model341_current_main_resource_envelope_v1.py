from __future__ import annotations

import hashlib
import json
import math
import platform
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder, count_trainable_parameters

SCHEMA = "12-6.model341.current-main-local-free-resource-probe.v1"
CURRENT_MAIN_SHA = "c95286b118c1c65c21050de4a48a7becee8a81ea"
MODEL_BLOB_SHA1 = "c3879fe0ba9193d5a8176c284e1942f058ef7885"
PYPROJECT_BLOB_SHA1 = "ab7518370c9f4ff13eb007d3bacecb363bb4956f"
EXPECTED_PARAMETER_COUNT = 20_613_440
EXPECTED_MODEL_IDENTITY_SHA256 = (
    "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
)
EXPECTED_INIT_IDENTITY_SHA256 = (
    "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
)
TRUTH_BOUNDARY = {
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "optimizer_updates_executed_on_real_targets": 0,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights": False,
    "external_llm_or_api_used_for_data_or_intelligence": False,
}


def model_spec() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=1024,
        d_model=320,
        n_layers=16,
        n_heads=10,
        n_kv_heads=2,
        head_dim=32,
        d_ff=1080,
        activation="swiglu",
        norm_kind="rmsnorm",
        norm_placement="pre",
        norm_eps=1e-5,
        position_embedding="rope",
        rope_theta=10_000.0,
        rope_rotary_dim=32,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        final_norm=True,
        tie_word_embeddings=True,
        lm_head_bias=False,
    )


def init_spec() -> InitSpec:
    return InitSpec(
        schema_version=1,
        family="normal",
        std=0.02,
        residual_branch_scale="sqrt_2_layers",
    )


def git_blob_sha1(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def _parameter_fingerprint(model: TwelveSixDecoder) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            digest.update(name.encode("utf-8"))
            tensor = parameter.detach().cpu().contiguous()
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _cpu_name() -> str:
    value = platform.processor().strip()
    if value:
        return value
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return "UNKNOWN"


def _finite_positive(value: Any, name: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a JSON number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def validate_source_root(root: Path) -> None:
    model_path = root / "src/twelve_six/model.py"
    pyproject_path = root / "pyproject.toml"
    if git_blob_sha1(model_path) != MODEL_BLOB_SHA1:
        raise ValueError("current-main model.py identity mismatch")
    if git_blob_sha1(pyproject_path) != PYPROJECT_BLOB_SHA1:
        raise ValueError("current-main pyproject.toml identity mismatch")


def run_probe(
    root: Path,
    *,
    warmup_samples: int = 1,
    measured_samples: int = 3,
    intraop_threads: int = 2,
) -> dict[str, Any]:
    if type(warmup_samples) is not int or warmup_samples < 0:
        raise ValueError("warmup_samples must be a non-negative integer")
    if type(measured_samples) is not int or measured_samples <= 0:
        raise ValueError("measured_samples must be a positive integer")
    if type(intraop_threads) is not int or intraop_threads <= 0:
        raise ValueError("intraop_threads must be a positive integer")

    validate_source_root(root)
    torch.set_num_threads(intraop_threads)
    torch.manual_seed(341)

    spec = model_spec()
    init = init_spec()
    if spec.identity_sha256() != EXPECTED_MODEL_IDENTITY_SHA256:
        raise ValueError("MODEL-341 ModelSpec identity drift")
    if init.identity_sha256() != EXPECTED_INIT_IDENTITY_SHA256:
        raise ValueError("MODEL-341 InitSpec identity drift")
    if spec.parameter_count() != EXPECTED_PARAMETER_COUNT:
        raise ValueError("MODEL-341 parameter formula drift")

    model = TwelveSixDecoder(spec, init)
    model.train()
    if count_trainable_parameters(model) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("MODEL-341 instantiated parameter count drift")

    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    before = _parameter_fingerprint(model)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(341)
    input_ids = torch.randint(
        low=0,
        high=spec.vocab_size,
        size=(1, 128),
        generator=generator,
        dtype=torch.long,
    )
    causal_targets = input_ids.numel() - input_ids.shape[0]

    def one_sample() -> tuple[float, float]:
        model.zero_grad(set_to_none=True)
        started = time.perf_counter()
        output = model(input_ids)
        logits = output.logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
        )
        loss.backward()
        elapsed = time.perf_counter() - started
        return elapsed, float(loss.detach())

    for _ in range(warmup_samples):
        one_sample()

    elapsed_samples: list[float] = []
    losses: list[float] = []
    for _ in range(measured_samples):
        elapsed, loss = one_sample()
        elapsed_samples.append(elapsed)
        losses.append(loss)

    after = _parameter_fingerprint(model)
    median_seconds = statistics.median(elapsed_samples)
    throughput = causal_targets / median_seconds
    gradient_bytes = sum(
        p.grad.numel() * p.grad.element_size()
        for p in model.parameters()
        if p.grad is not None
    )
    hwm_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    hwm_mib = hwm_raw / 1024.0 if sys.platform != "darwin" else hwm_raw / 1024.0**2

    return {
        "schema": SCHEMA,
        "source": {
            "current_main_sha": CURRENT_MAIN_SHA,
            "model_blob_sha1": MODEL_BLOB_SHA1,
            "pyproject_blob_sha1": PYPROJECT_BLOB_SHA1,
        },
        "model": {
            "model_identity_sha256": spec.identity_sha256(),
            "init_identity_sha256": init.identity_sha256(),
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "vocab_size": spec.vocab_size,
            "sequence_length": 128,
            "micro_batch_size": 1,
            "precision": "fp32",
            "seed": 341,
            "causal_targets_per_microbatch": causal_targets,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu": _cpu_name(),
            "cuda_available": torch.cuda.is_available(),
            "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
        },
        "measurement": {
            "warmup_samples": warmup_samples,
            "measured_samples": measured_samples,
            "elapsed_seconds": elapsed_samples,
            "median_forward_loss_backward_seconds": median_seconds,
            "median_causal_targets_per_second": throughput,
            "synthetic_loss_median": statistics.median(losses),
            "parameter_bytes": parameter_bytes,
            "gradient_bytes": gradient_bytes,
            "process_hwm_mib_approx": hwm_mib,
            "parameter_fingerprint_before_sha256": before,
            "parameter_fingerprint_after_sha256": after,
            "parameter_fingerprint_unchanged": before == after,
            "optimizer_object_created": False,
            "optimizer_updates": 0,
            "model_updates": 0,
        },
        "planning": {
            "scope": "forward+causal_ce+backward_only",
            "target_positions_example": 20_000_000,
            "mechanics_only_lower_bound_seconds_example": 20_000_000 / throughput,
            "mechanics_only_lower_bound_hours_example": 20_000_000 / throughput / 3600.0,
            "cross_host_extrapolation_allowed": False,
            "excluded": [
                "optimizer.step",
                "checkpoint_io",
                "evaluation",
                "packing_data_io",
                "real_pilot_overhead",
            ],
        },
        "truth_boundary": dict(TRUTH_BOUNDARY),
    }


def validate_probe(report: dict[str, Any]) -> None:
    if report.get("schema") != SCHEMA:
        raise ValueError("schema mismatch")
    source = report.get("source")
    if type(source) is not dict:
        raise ValueError("source must be an object")
    expected_source = {
        "current_main_sha": CURRENT_MAIN_SHA,
        "model_blob_sha1": MODEL_BLOB_SHA1,
        "pyproject_blob_sha1": PYPROJECT_BLOB_SHA1,
    }
    if source != expected_source:
        raise ValueError("source root mismatch")

    model = report.get("model")
    if type(model) is not dict:
        raise ValueError("model must be an object")
    if model.get("model_identity_sha256") != EXPECTED_MODEL_IDENTITY_SHA256:
        raise ValueError("model identity mismatch")
    if model.get("init_identity_sha256") != EXPECTED_INIT_IDENTITY_SHA256:
        raise ValueError("init identity mismatch")
    if type(model.get("parameter_count")) is not int:
        raise ValueError("parameter_count type mismatch")
    if model["parameter_count"] != EXPECTED_PARAMETER_COUNT:
        raise ValueError("parameter_count mismatch")

    measurement = report.get("measurement")
    if type(measurement) is not dict:
        raise ValueError("measurement must be an object")
    seconds = measurement.get("median_forward_loss_backward_seconds")
    throughput = measurement.get("median_causal_targets_per_second")
    seconds_value = _finite_positive(seconds, "median seconds")
    throughput_value = _finite_positive(throughput, "median throughput")
    expected_throughput = model["causal_targets_per_microbatch"] / seconds_value
    if not math.isclose(throughput_value, expected_throughput, rel_tol=1e-9):
        raise ValueError("throughput arithmetic mismatch")
    if measurement.get("parameter_fingerprint_unchanged") is not True:
        raise ValueError("parameter fingerprint changed")
    if measurement.get("optimizer_object_created") is not False:
        raise ValueError("optimizer object was created")
    for field in ("optimizer_updates", "model_updates"):
        if type(measurement.get(field)) is not int or measurement[field] != 0:
            raise ValueError(f"{field} must be exact integer zero")

    if report.get("truth_boundary") != TRUTH_BOUNDARY:
        raise ValueError("truth boundary mismatch")
    planning = report.get("planning")
    if type(planning) is not dict:
        raise ValueError("planning must be an object")
    if planning.get("cross_host_extrapolation_allowed") is not False:
        raise ValueError("cross-host extrapolation must be false")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report = run_probe(root)
    validate_probe(report)
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
