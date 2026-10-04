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
MEASUREMENT_AUTHORITY_SCHEMA = (
    "12-6.model341.local-free-cpu-measurement-authority.v1"
)
MEASUREMENT_AUTHORITY_SOURCE_ISSUE = 1262
MEASUREMENT_AUTHORITY_SOURCE_COMMENT_ID = 5634725801
MEASUREMENT_AUTHORITY_SHA256 = (
    "5090abe87bdad694274c4a183117e41a35869ab852301048a81f244e39c89965"
)
EXPECTED_BLOBS = {
    "configs/candidates/model341_20m_candidate_a.json": (
        "69e3cbd5f5c83c9d3d529a2a6376db3055979c40"
    ),
    "src/twelve_six/__init__.py": "2ce6d14cbdfd23573abf1c47d0a770e5a719de46",
    "src/twelve_six/attention_perf.py": (
        "2323a709bf8f94047ffddc2de707b57d6edb8efd"
    ),
    "src/twelve_six/model.py": "d0823aa666883ddb5c445a438730043ef8b50ff1",
}
EXPECTED_MODEL_AUTHORITY = {
    "model_spec_sha256": MODEL_SPEC_SHA256,
    "init_spec_sha256": INIT_SPEC_SHA256,
    "parameter_count": EXPECTED_PARAMETER_COUNT,
    "canonical_base": "random_init",
    "sequence_length": 128,
    "micro_batch_size": 1,
    "precision": "fp32",
    "seed": 341,
    "vocab_size": 256,
    "causal_targets_per_microbatch": 127,
}
EXPECTED_RUNTIME = {
    "python": "3.13.5",
    "torch": "2.10.0+cpu",
    "cpu": "Intel Xeon Platinum 8370C",
    "cuda_available": False,
    "intraop_threads": 4,
    "interop_threads": 1,
}
EXPECTED_MEASUREMENT = {
    "warmup_samples": 3,
    "measured_samples": 7,
    "raw_measured_seconds": [],
    "raw_samples_retained": False,
    "median_forward_loss_backward_ms": 208.360758,
    "median_causal_targets_per_second": 609.5197637935258,
    "synthetic_loss": 5.628034591674805,
    "parameter_bytes": 82_453_760,
    "gradient_bytes": 82_453_760,
    "process_hwm_mib_approx": 478.56,
    "parameter_fingerprint_unchanged": True,
    "optimizer_object_created": False,
    "optimizer_updates": 0,
    "model_updates": 0,
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
EXPECTED_LEGACY = {
    "schema": "12-6.model341.local-free-cpu-resource-envelope.v1",
    "report_identity_sha256": (
        "4b8b3965ece6301629eba2f4b90c38bde3334bccb14aa8d08bfee33ad3a9567f"
    ),
    "report_file_sha256": (
        "6158a8c763143b1fe34e1484b4f0f471d8a3f85ab936873e8373eda1ab64e0c2"
    ),
    "measurement_validator_harness_sha256": (
        "bba3d47b70d9a6abf594493abd97da29ee6c2ff000b5363e07ab611e5c31e35a"
    ),
    "focused_adversarial_tests_sha256": (
        "149d5c5c96f92c015d4d01ba7d177cf270755611f712e3942c26c2a29afb6310"
    ),
    "focused_adversarial_tests_passed": 7,
}
EXPECTED_REBIND = {
    "measured_to_current_compare_status": "ahead",
    "ahead_by": 16,
    "execution_bearing_blobs_unchanged": True,
    "changed_surface_families": ["inference", "evaluation"],
    "scientific_authority": "mechanics-only",
    "authorizes_training": False,
}
EXPECTED_EXCLUDED = [
    "optimizer.step",
    "checkpoint_io",
    "evaluation",
    "packing_data_io",
    "real_pilot_overhead",
]
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


def _exact_keys(payload: Any, expected: set[str], name: str) -> None:
    if type(payload) is not dict:
        raise ValueError(f"{name} must be an object")
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{name} keys mismatch: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )


def _exact_value(actual: Any, expected: Any, name: str) -> None:
    if type(actual) is not type(expected):
        raise ValueError(f"{name} type mismatch")
    if actual != expected:
        raise ValueError(f"{name} mismatch")


def _finite_positive(value: Any, name: str) -> float:
    if type(value) not in {int, float} or type(value) is bool:
        raise ValueError(f"{name} must be a finite JSON number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _measurement_authority_payload(report: dict[str, Any]) -> dict[str, Any]:
    authority = report["model_authority"]
    runtime = report["runtime"]
    measurement = report["measurement"]
    return {
        "authority_schema": MEASUREMENT_AUTHORITY_SCHEMA,
        "source_issue": MEASUREMENT_AUTHORITY_SOURCE_ISSUE,
        "source_comment_id": MEASUREMENT_AUTHORITY_SOURCE_COMMENT_ID,
        "model_spec_sha256": authority["model_spec_sha256"],
        "init_spec_sha256": authority["init_spec_sha256"],
        "parameter_count": authority["parameter_count"],
        "seed": authority["seed"],
        "sequence_length": authority["sequence_length"],
        "micro_batch_size": authority["micro_batch_size"],
        "causal_targets_per_microbatch": authority[
            "causal_targets_per_microbatch"
        ],
        "precision": authority["precision"],
        "runtime": {key: runtime[key] for key in EXPECTED_RUNTIME},
        "measurement": {
            key: measurement[key]
            for key in EXPECTED_MEASUREMENT
            if key != "raw_measured_seconds"
        },
    }


def validate_report(report: dict[str, Any]) -> None:
    _exact_keys(report, TOP_LEVEL_KEYS, "report")
    _exact_value(report["schema"], SCHEMA, "schema")
    _exact_value(
        report["status"],
        "MEASURED_MECHANICS_REBOUND_TO_CURRENT_CARRIER",
        "status",
    )
    claimed = report["report_identity_sha256"]
    if type(claimed) is not str or len(claimed) != 64:
        raise ValueError("report_identity_sha256 must be 64 hex characters")
    payload = copy.deepcopy(report)
    payload.pop("report_identity_sha256")
    if canonical_json_sha256(payload) != claimed:
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
    _exact_value(origin["issue"], 1262, "measurement_origin.issue")
    _exact_value(
        origin["measured_carrier_head"],
        MEASURED_CARRIER_HEAD,
        "measured_carrier_head",
    )
    _exact_value(
        origin["current_carrier_pr"], CURRENT_CARRIER_PR, "current_carrier_pr"
    )
    _exact_value(
        origin["current_carrier_head"],
        CURRENT_CARRIER_HEAD,
        "current_carrier_head",
    )
    blobs = origin["execution_bearing_blobs"]
    _exact_keys(blobs, set(EXPECTED_BLOBS), "execution_bearing_blobs")
    for path, expected_sha in EXPECTED_BLOBS.items():
        _exact_value(
            blobs[path], expected_sha, f"execution_bearing_blobs.{path}"
        )

    authority = report["model_authority"]
    _exact_keys(authority, set(EXPECTED_MODEL_AUTHORITY), "model_authority")
    for key, expected in EXPECTED_MODEL_AUTHORITY.items():
        _exact_value(authority[key], expected, f"model_authority.{key}")

    runtime = report["runtime"]
    _exact_keys(runtime, set(EXPECTED_RUNTIME), "runtime")
    for key, expected in EXPECTED_RUNTIME.items():
        _exact_value(runtime[key], expected, f"runtime.{key}")

    measurement = report["measurement"]
    _exact_keys(measurement, set(EXPECTED_MEASUREMENT), "measurement")
    if type(measurement["raw_measured_seconds"]) is not list:
        raise ValueError("raw_measured_seconds must be an array")
    if measurement["raw_measured_seconds"]:
        raise ValueError(
            "raw samples were not retained by the prepublished authority"
        )
    median_ms = _finite_positive(
        measurement["median_forward_loss_backward_ms"],
        "measurement.median_forward_loss_backward_ms",
    )
    throughput = _finite_positive(
        measurement["median_causal_targets_per_second"],
        "measurement.median_causal_targets_per_second",
    )
    expected_throughput = authority["causal_targets_per_microbatch"] / (
        median_ms / 1000.0
    )
    if not math.isclose(
        throughput,
        expected_throughput,
        rel_tol=1e-15,
        abs_tol=1e-12,
    ):
        raise ValueError("median/throughput arithmetic mismatch")
    authority_sha256 = canonical_json_sha256(
        _measurement_authority_payload(report)
    )
    if authority_sha256 != MEASUREMENT_AUTHORITY_SHA256:
        raise ValueError(
            "measurement authority tuple mismatch: historical telemetry must "
            f"match prepublished issue #{MEASUREMENT_AUTHORITY_SOURCE_ISSUE} "
            f"comment #{MEASUREMENT_AUTHORITY_SOURCE_COMMENT_ID}"
        )
    for key, expected in EXPECTED_MEASUREMENT.items():
        _exact_value(measurement[key], expected, f"measurement.{key}")

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
    _exact_value(
        planning["target_positions"], 20_000_000, "planning.target_positions"
    )
    seconds = _finite_positive(
        planning["mechanics_only_lower_bound_seconds"], "planning.seconds"
    )
    hours = _finite_positive(
        planning["mechanics_only_lower_bound_hours"], "planning.hours"
    )
    if not math.isclose(
        seconds,
        20_000_000 / throughput,
        rel_tol=1e-15,
        abs_tol=1e-12,
    ):
        raise ValueError("planning seconds arithmetic mismatch")
    if not math.isclose(
        hours,
        seconds / 3600.0,
        rel_tol=1e-15,
        abs_tol=1e-12,
    ):
        raise ValueError("planning hours arithmetic mismatch")
    _exact_value(
        planning["scope"],
        "forward+causal_ce+backward_only",
        "planning.scope",
    )
    _exact_value(planning["excluded"], EXPECTED_EXCLUDED, "planning.excluded")
    _exact_value(
        planning["cross_host_extrapolation_allowed"],
        False,
        "planning.cross_host_extrapolation_allowed",
    )

    legacy = report["legacy_package"]
    _exact_keys(legacy, set(EXPECTED_LEGACY), "legacy_package")
    for key, expected in EXPECTED_LEGACY.items():
        _exact_value(legacy[key], expected, f"legacy_package.{key}")

    rebind = report["rebind"]
    _exact_keys(rebind, set(EXPECTED_REBIND), "rebind")
    for key, expected in EXPECTED_REBIND.items():
        _exact_value(rebind[key], expected, f"rebind.{key}")

    truth = report["truth_boundary"]
    _exact_keys(truth, set(EXPECTED_TRUTH_BOUNDARY), "truth_boundary")
    for key, expected in EXPECTED_TRUTH_BOUNDARY.items():
        _exact_value(truth[key], expected, f"truth_boundary.{key}")


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
        digest.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
        )
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
    for value, name, allow_zero in (
        (warmup_samples, "warmup_samples", True),
        (measured_samples, "measured_samples", False),
        (intraop_threads, "intraop_threads", False),
        (interop_threads, "interop_threads", False),
    ):
        minimum = 0 if allow_zero else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} has invalid type/value")

    validate_repo_execution_blobs(repo_root)
    src_dir = str(repo_root / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    import torch
    import torch.nn.functional as F

    from twelve_six import (
        TwelveSixDecoder,
        count_trainable_parameters,
        load_stage_config,
    )

    torch.set_num_threads(intraop_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        pass

    config = load_stage_config(
        repo_root / "configs/candidates/model341_20m_candidate_a.json"
    )
    if config.model.identity_sha256() != MODEL_SPEC_SHA256:
        raise ValueError("fresh probe model identity mismatch")
    if config.init.identity_sha256() != INIT_SPEC_SHA256:
        raise ValueError("fresh probe init identity mismatch")

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
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )

    def one_pass() -> tuple[float, float]:
        model.zero_grad(set_to_none=True)
        started = time.perf_counter()
        logits = model(input_ids).logits
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            input_ids[:, 1:].reshape(-1),
        )
        loss.backward()
        return time.perf_counter() - started, float(loss.detach().cpu())

    for _ in range(warmup_samples):
        one_pass()

    samples: list[float] = []
    losses: list[float] = []
    for _ in range(measured_samples):
        elapsed, loss = one_pass()
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("fresh probe produced invalid timing")
        if not math.isfinite(loss):
            raise ValueError("fresh probe produced invalid loss")
        samples.append(elapsed)
        losses.append(loss)

    after = _parameter_fingerprint(model)
    if before != after:
        raise ValueError("fresh probe mutated model parameters")
    gradient_bytes = sum(
        parameter.grad.numel() * parameter.grad.element_size()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
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
        "parameter_fingerprint_unchanged": True,
        "optimizer_object_created": False,
        "optimizer_updates": 0,
        "model_updates": 0,
        "truth_boundary": copy.deepcopy(EXPECTED_TRUTH_BOUNDARY),
    }
    payload["probe_identity_sha256"] = canonical_json_sha256(payload)
    return payload


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
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
        result = run_probe(
            repo_root,
            warmup_samples=args.probe_warmups,
            measured_samples=args.probe_samples,
            intraop_threads=args.probe_intraop_threads,
            interop_threads=args.probe_interop_threads,
        )
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    else:
        print(
            json.dumps(
                {
                    "validation": "PASS",
                    "schema": report["schema"],
                    "report_identity_sha256": report["report_identity_sha256"],
                    "current_carrier_head": CURRENT_CARRIER_HEAD,
                    "measurement_authority_source_issue": (
                        MEASUREMENT_AUTHORITY_SOURCE_ISSUE
                    ),
                    "measurement_authority_source_comment_id": (
                        MEASUREMENT_AUTHORITY_SOURCE_COMMENT_ID
                    ),
                    "measurement_authority_sha256": MEASUREMENT_AUTHORITY_SHA256,
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
