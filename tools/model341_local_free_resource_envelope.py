from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "12-6.model341.local-free-cpu-resource-envelope.v4"
PROBE_SCHEMA = "12-6.model341.local-free-cpu-resource-envelope.probe.v1"
MEASURED_CARRIER_HEAD = "82c43005bb5db153482ae5b20a31d59240faaebc"
CURRENT_CARRIER_HEAD = "61aa37b340565dd1ba791adc16ce430f9b17fbaf"
CURRENT_CARRIER_PR = 802
MODEL_SPEC_SHA256 = "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
INIT_SPEC_SHA256 = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
EXPECTED_PARAMETER_COUNT = 20_613_440
EXPECTED_BLOBS = {
    "configs/candidates/model341_20m_candidate_a.json": "69e3cbd5f5c83c9d3d529a2a6376db3055979c40",
    "src/twelve_six/__init__.py": "2ce6d14cbdfd23573abf1c47d0a770e5a719de46",
    "src/twelve_six/attention_perf.py": "2323a709bf8f94047ffddc2de707b57d6edb8efd",
    "src/twelve_six/model.py": "d0823aa666883ddb5c445a438730043ef8b50ff1",
}
EXPECTED_TRUTH_BOUNDARY = {
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
EXPECTED_EXCLUDED = [
    "optimizer.step",
    "checkpoint_io",
    "evaluation",
    "packing_data_io",
    "real_pilot_overhead",
]
EXPECTED_LEGACY = {
    "schema": "12-6.model341.local-free-cpu-resource-envelope.v1",
    "report_identity_sha256": "4b8b3965ece6301629eba2f4b90c38bde3334bccb14aa8d08bfee33ad3a9567f",
    "report_file_sha256": "6158a8c763143b1fe34e1484b4f0f471d8a3f85ab936873e8373eda1ab64e0c2",
    "measurement_validator_harness_sha256": "bba3d47b70d9a6abf594493abd97da29ee6c2ff000b5363e07ab611e5c31e35a",
    "focused_adversarial_tests_sha256": "149d5c5c96f92c015d4d01ba7d177cf270755611f712e3942c26c2a29afb6310",
    "focused_adversarial_tests_passed": 7,
}
TOP_LEVEL_KEYS = {
    "schema",
    "status",
    "measurement_origin",
    "model_authority",
    "runtime",
    "measurement",
    "planning",
    "legacy_package",
    "rebind",
    "truth_boundary",
    "report_identity_sha256",
}


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_blob_sha1(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def _exact_keys(payload: dict[str, Any], expected: set[str], name: str) -> None:
    if type(payload) is not dict:
        raise ValueError(f"{name} must be an object")
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{name} keys mismatch: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )


def _strict_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact JSON integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be an exact JSON boolean")
    return value


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if type(value) not in {int, float} or type(value) is bool:
        raise ValueError(f"{name} must be a finite JSON number")
    as_float = float(value)
    if not math.isfinite(as_float):
        raise ValueError(f"{name} must be finite")
    if positive and as_float <= 0:
        raise ValueError(f"{name} must be positive")
    return as_float


def _sha256(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be a 64-hex SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a 64-hex SHA-256") from exc
    return value


def _sha1(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 40:
        raise ValueError(f"{name} must be a 40-hex Git object SHA-1")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a 40-hex Git object SHA-1") from exc
    return value


def _validate_truth_boundary(boundary: dict[str, Any]) -> None:
    _exact_keys(boundary, set(EXPECTED_TRUTH_BOUNDARY), "truth_boundary")
    for key, expected in EXPECTED_TRUTH_BOUNDARY.items():
        actual = boundary[key]
        if type(expected) is bool:
            _strict_bool(actual, f"truth_boundary.{key}")
        else:
            _strict_int(actual, f"truth_boundary.{key}", minimum=0)
        if actual != expected:
            raise ValueError(f"truth boundary widened: {key}")


def validate_report(report: dict[str, Any]) -> None:
    _exact_keys(report, TOP_LEVEL_KEYS, "report")
    if report["schema"] != SCHEMA:
        raise ValueError("resource-envelope schema mismatch")
    if report["status"] != "MEASURED_MECHANICS_REBOUND_TO_CURRENT_CARRIER":
        raise ValueError("resource-envelope status mismatch")

    claimed_identity = _sha256(report["report_identity_sha256"], "report_identity_sha256")
    payload = copy.deepcopy(report)
    payload.pop("report_identity_sha256")
    if canonical_json_sha256(payload) != claimed_identity:
        raise ValueError("report identity mismatch")

    origin = report["measurement_origin"]
    _exact_keys(
        origin,
        {
            "issue",
            "measured_carrier_head",
            "current_carrier_pr",
            "current_carrier_head",
            "execution_bearing_blobs",
        },
        "measurement_origin",
    )
    if _strict_int(origin["issue"], "measurement_origin.issue", minimum=1) != 1262:
        raise ValueError("measurement origin issue mismatch")
    if origin["measured_carrier_head"] != MEASURED_CARRIER_HEAD:
        raise ValueError("measured carrier head mismatch")
    if _strict_int(
        origin["current_carrier_pr"], "measurement_origin.current_carrier_pr", minimum=1
    ) != CURRENT_CARRIER_PR:
        raise ValueError("current carrier PR mismatch")
    if origin["current_carrier_head"] != CURRENT_CARRIER_HEAD:
        raise ValueError("current carrier head mismatch")
    blobs = origin["execution_bearing_blobs"]
    _exact_keys(blobs, set(EXPECTED_BLOBS), "execution_bearing_blobs")
    for path, expected_sha in EXPECTED_BLOBS.items():
        if _sha1(blobs[path], f"execution_bearing_blobs.{path}") != expected_sha:
            raise ValueError(f"execution-bearing blob drift: {path}")

    authority = report["model_authority"]
    _exact_keys(
        authority,
        {
            "model_spec_sha256",
            "init_spec_sha256",
            "parameter_count",
            "canonical_base",
            "sequence_length",
            "micro_batch_size",
            "precision",
            "seed",
            "vocab_size",
            "causal_targets_per_microbatch",
        },
        "model_authority",
    )
    if _sha256(
        authority["model_spec_sha256"], "model_authority.model_spec_sha256"
    ) != MODEL_SPEC_SHA256:
        raise ValueError("ModelSpec identity mismatch")
    if _sha256(
        authority["init_spec_sha256"], "model_authority.init_spec_sha256"
    ) != INIT_SPEC_SHA256:
        raise ValueError("InitSpec identity mismatch")
    expected_authority = {
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "canonical_base": "random_init",
        "sequence_length": 128,
        "micro_batch_size": 1,
        "precision": "fp32",
        "seed": 341,
        "vocab_size": 256,
        "causal_targets_per_microbatch": 127,
    }
    for key, expected in expected_authority.items():
        actual = authority[key]
        if type(expected) is int:
            _strict_int(actual, f"model_authority.{key}", minimum=1)
        elif type(actual) is not str:
            raise ValueError(f"model_authority.{key} must be a string")
        if actual != expected:
            raise ValueError(f"model authority mismatch: {key}")

    runtime = report["runtime"]
    _exact_keys(
        runtime,
        {
            "python",
            "torch",
            "cpu",
            "cuda_available",
            "intraop_threads",
            "interop_threads",
        },
        "runtime",
    )
    for key in ("python", "torch", "cpu"):
        if type(runtime[key]) is not str or not runtime[key]:
            raise ValueError(f"runtime.{key} must be a non-empty string")
    if _strict_bool(runtime["cuda_available"], "runtime.cuda_available"):
        raise ValueError("sealed LOCAL_FREE CPU evidence must not claim CUDA")
    _strict_int(runtime["intraop_threads"], "runtime.intraop_threads", minimum=1)
    _strict_int(runtime["interop_threads"], "runtime.interop_threads", minimum=1)

    measurement = report["measurement"]
    _exact_keys(
        measurement,
        {
            "warmup_samples",
            "measured_samples",
            "raw_measured_seconds",
            "raw_samples_retained",
            "median_forward_loss_backward_ms",
            "median_causal_targets_per_second",
            "synthetic_loss",
            "parameter_bytes",
            "gradient_bytes",
            "process_hwm_mib_approx",
            "parameter_fingerprint_unchanged",
            "optimizer_object_created",
            "optimizer_updates",
            "model_updates",
        },
        "measurement",
    )
    if _strict_int(
        measurement["warmup_samples"], "measurement.warmup_samples", minimum=0
    ) != 3:
        raise ValueError("warmup sample count mismatch")
    if _strict_int(
        measurement["measured_samples"], "measurement.measured_samples", minimum=1
    ) != 7:
        raise ValueError("measured sample count mismatch")
    if type(measurement["raw_measured_seconds"]) is not list:
        raise ValueError("raw_measured_seconds must be an array")
    raw = [
        _finite_number(value, f"raw_measured_seconds[{index}]", positive=True)
        for index, value in enumerate(measurement["raw_measured_seconds"])
    ]
    retained = _strict_bool(
        measurement["raw_samples_retained"], "measurement.raw_samples_retained"
    )
    if retained != bool(raw):
        raise ValueError("raw sample retention flag mismatch")
    if raw and len(raw) != measurement["measured_samples"]:
        raise ValueError("raw sample count mismatch")

    median_ms = _finite_number(
        measurement["median_forward_loss_backward_ms"],
        "measurement.median_forward_loss_backward_ms",
        positive=True,
    )
    throughput = _finite_number(
        measurement["median_causal_targets_per_second"],
        "measurement.median_causal_targets_per_second",
        positive=True,
    )
    _finite_number(measurement["synthetic_loss"], "measurement.synthetic_loss", positive=True)
    _finite_number(
        measurement["process_hwm_mib_approx"],
        "measurement.process_hwm_mib_approx",
        positive=True,
    )
    if _strict_int(
        measurement["parameter_bytes"], "measurement.parameter_bytes", minimum=1
    ) != 82_453_760:
        raise ValueError("parameter byte count mismatch")
    if _strict_int(
        measurement["gradient_bytes"], "measurement.gradient_bytes", minimum=1
    ) != 82_453_760:
        raise ValueError("gradient byte count mismatch")
    if not _strict_bool(
        measurement["parameter_fingerprint_unchanged"],
        "measurement.parameter_fingerprint_unchanged",
    ):
        raise ValueError("parameter fingerprint changed")
    if _strict_bool(
        measurement["optimizer_object_created"], "measurement.optimizer_object_created"
    ):
        raise ValueError("optimizer object must not exist in mechanics evidence")
    if _strict_int(
        measurement["optimizer_updates"], "measurement.optimizer_updates", minimum=0
    ) != 0:
        raise ValueError("optimizer updates must remain zero")
    if _strict_int(measurement["model_updates"], "measurement.model_updates", minimum=0) != 0:
        raise ValueError("model updates must remain zero")
    if raw:
        raw_median_ms = statistics.median(raw) * 1000.0
        if not math.isclose(median_ms, raw_median_ms, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("raw sample median mismatch")
        raw_throughput = 127.0 / statistics.median(raw)
        if not math.isclose(throughput, raw_throughput, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("raw sample throughput mismatch")

    planning = report["planning"]
    _exact_keys(
        planning,
        {
            "target_positions",
            "mechanics_only_lower_bound_seconds",
            "mechanics_only_lower_bound_hours",
            "scope",
            "excluded",
            "cross_host_extrapolation_allowed",
        },
        "planning",
    )
    targets = _strict_int(planning["target_positions"], "planning.target_positions", minimum=1)
    if targets != 20_000_000:
        raise ValueError("planning target mismatch")
    seconds = _finite_number(
        planning["mechanics_only_lower_bound_seconds"],
        "planning.mechanics_only_lower_bound_seconds",
        positive=True,
    )
    hours = _finite_number(
        planning["mechanics_only_lower_bound_hours"],
        "planning.mechanics_only_lower_bound_hours",
        positive=True,
    )
    expected_seconds = targets / throughput
    if not math.isclose(seconds, expected_seconds, rel_tol=1e-15, abs_tol=1e-12):
        raise ValueError("planning seconds arithmetic mismatch")
    if not math.isclose(hours, seconds / 3600.0, rel_tol=1e-15, abs_tol=1e-12):
        raise ValueError("planning hours arithmetic mismatch")
    if planning["scope"] != "forward+causal_ce+backward_only":
        raise ValueError("planning scope widened")
    if planning["excluded"] != EXPECTED_EXCLUDED:
        raise ValueError("planning exclusions mismatch")
    if _strict_bool(
        planning["cross_host_extrapolation_allowed"],
        "planning.cross_host_extrapolation_allowed",
    ):
        raise ValueError("cross-host extrapolation must remain forbidden")

    legacy = report["legacy_package"]
    _exact_keys(legacy, set(EXPECTED_LEGACY), "legacy_package")
    for key, expected in EXPECTED_LEGACY.items():
        actual = legacy[key]
        if key.endswith("sha256"):
            _sha256(actual, f"legacy_package.{key}")
        elif type(expected) is int:
            _strict_int(actual, f"legacy_package.{key}", minimum=0)
        elif type(actual) is not str:
            raise ValueError(f"legacy_package.{key} must be a string")
        if actual != expected:
            raise ValueError(f"legacy package mismatch: {key}")

    rebind = report["rebind"]
    _exact_keys(
        rebind,
        {
            "measured_to_current_compare_status",
            "ahead_by",
            "execution_bearing_blobs_unchanged",
            "changed_surface_families",
            "scientific_authority",
            "authorizes_training",
        },
        "rebind",
    )
    if rebind["measured_to_current_compare_status"] != "ahead":
        raise ValueError("carrier compare status mismatch")
    if _strict_int(rebind["ahead_by"], "rebind.ahead_by", minimum=0) != 16:
        raise ValueError("carrier compare commit count mismatch")
    if not _strict_bool(
        rebind["execution_bearing_blobs_unchanged"],
        "rebind.execution_bearing_blobs_unchanged",
    ):
        raise ValueError("execution-bearing bytes must be unchanged")
    if rebind["changed_surface_families"] != ["inference", "evaluation"]:
        raise ValueError("carrier delta classification mismatch")
    if rebind["scientific_authority"] != "mechanics-only":
        raise ValueError("resource envelope must remain mechanics-only")
    if _strict_bool(rebind["authorizes_training"], "rebind.authorizes_training"):
        raise ValueError("resource envelope must not authorize training")

    _validate_truth_boundary(report["truth_boundary"])


def validate_repo_execution_blobs(repo_root: Path) -> None:
    for relative_path, expected_sha in EXPECTED_BLOBS.items():
        path = repo_root / relative_path
        if not path.is_file():
            raise ValueError(f"missing execution-bearing file: {relative_path}")
        actual_sha = git_blob_sha1(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"execution-bearing blob mismatch for {relative_path}: "
                f"{actual_sha} != {expected_sha}"
            )


def _parameter_fingerprint(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def run_probe(
    repo_root: Path,
    *,
    warmup_samples: int = 3,
    measured_samples: int = 7,
    intraop_threads: int = 4,
    interop_threads: int = 1,
) -> dict[str, Any]:
    _strict_int(warmup_samples, "warmup_samples", minimum=0)
    _strict_int(measured_samples, "measured_samples", minimum=1)
    _strict_int(intraop_threads, "intraop_threads", minimum=1)
    _strict_int(interop_threads, "interop_threads", minimum=1)
    validate_repo_execution_blobs(repo_root)

    src_dir = str(repo_root / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    import torch
    import torch.nn.functional as F
    from twelve_six import TwelveSixDecoder, count_trainable_parameters, load_stage_config

    torch.set_num_threads(intraop_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        pass

    config = load_stage_config(repo_root / "configs/candidates/model341_20m_candidate_a.json")
    if config.model.identity_sha256() != MODEL_SPEC_SHA256:
        raise ValueError("fresh probe ModelSpec identity mismatch")
    if config.init.identity_sha256() != INIT_SPEC_SHA256:
        raise ValueError("fresh probe InitSpec identity mismatch")

    torch.manual_seed(341)
    model = TwelveSixDecoder(config.model, config.init)
    model.train()
    if count_trainable_parameters(model) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("fresh probe parameter count mismatch")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(341)
    input_ids = torch.randint(
        0,
        config.model.vocab_size,
        (1, 128),
        generator=generator,
        dtype=torch.long,
    )

    before = _parameter_fingerprint(model)
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )

    def one_pass() -> tuple[float, float]:
        model.zero_grad(set_to_none=True)
        started = time.perf_counter()
        output = model(input_ids)
        logits = output.logits
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            input_ids[:, 1:].reshape(-1),
        )
        loss.backward()
        elapsed = time.perf_counter() - started
        return elapsed, float(loss.detach().cpu())

    for _ in range(warmup_samples):
        one_pass()

    samples: list[float] = []
    losses: list[float] = []
    for _ in range(measured_samples):
        elapsed, loss = one_pass()
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("fresh probe produced invalid elapsed time")
        if not math.isfinite(loss):
            raise ValueError("fresh probe produced non-finite loss")
        samples.append(elapsed)
        losses.append(loss)

    after = _parameter_fingerprint(model)
    gradient_bytes = sum(
        parameter.grad.numel() * parameter.grad.element_size()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    if before != after:
        raise ValueError("fresh probe mutated model parameters")

    median_seconds = statistics.median(samples)
    payload: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "source_carrier_head": CURRENT_CARRIER_HEAD,
        "current_checkout_execution_blobs": {
            path: git_blob_sha1(repo_root / path) for path in sorted(EXPECTED_BLOBS)
        },
        "model_spec_sha256": config.model.identity_sha256(),
        "init_spec_sha256": config.init.identity_sha256(),
        "parameter_count": count_trainable_parameters(model),
        "seed": 341,
        "sequence_length": 128,
        "micro_batch_size": 1,
        "causal_targets_per_microbatch": 127,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu": platform.processor() or platform.machine(),
            "cuda_available": bool(torch.cuda.is_available()),
            "intraop_threads": int(torch.get_num_threads()),
            "interop_threads": int(torch.get_num_interop_threads()),
        },
        "warmup_samples": warmup_samples,
        "measured_samples": measured_samples,
        "raw_measured_seconds": samples,
        "loss_samples": losses,
        "median_forward_loss_backward_seconds": median_seconds,
        "median_causal_targets_per_second": 127.0 / median_seconds,
        "parameter_bytes": parameter_bytes,
        "gradient_bytes": gradient_bytes,
        "parameter_fingerprint_before": before,
        "parameter_fingerprint_after": after,
        "parameter_fingerprint_unchanged": before == after,
        "optimizer_object_created": False,
        "optimizer_updates": 0,
        "model_updates": 0,
        "truth_boundary": copy.deepcopy(EXPECTED_TRUTH_BOUNDARY),
    }
    payload["probe_identity_sha256"] = canonical_json_sha256(payload)
    return payload


def load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if type(payload) is not dict:
        raise ValueError("report root must be an object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/model341_local_free_resource_envelope_v4.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-warmups", type=int, default=3)
    parser.add_argument("--probe-samples", type=int, default=7)
    parser.add_argument("--probe-intraop-threads", type=int, default=4)
    parser.add_argument("--probe-interop-threads", type=int, default=1)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    report = load_report(repo_root / args.report)
    validate_report(report)
    validate_repo_execution_blobs(repo_root)

    if args.probe:
        probe = run_probe(
            repo_root,
            warmup_samples=args.probe_warmups,
            measured_samples=args.probe_samples,
            intraop_threads=args.probe_intraop_threads,
            interop_threads=args.probe_interop_threads,
        )
        print(json.dumps(probe, sort_keys=True, indent=2, allow_nan=False))
    else:
        print(
            json.dumps(
                {
                    "validation": "PASS",
                    "schema": report["schema"],
                    "report_identity_sha256": report["report_identity_sha256"],
                    "current_carrier_head": CURRENT_CARRIER_HEAD,
                    "scientific_authority": "mechanics-only",
                    "authorizes_training": False,
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
