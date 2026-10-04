from __future__ import annotations

import ctypes
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import twelve_six.model as model_module
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder, count_trainable_parameters

try:
    import resource as _resource
except ModuleNotFoundError:  # Windows
    _resource = None

SCHEMA = "12-6.model341.current-main-local-free-resource-probe.v2"
EXECUTION_ROOT_MAIN_SHA = "7b3df41c10a826183fab0b04ae85a90cdf0ce351"
MODEL_BLOB_SHA1 = "c3879fe0ba9193d5a8176c284e1942f058ef7885"
PYPROJECT_BLOB_SHA1 = "5fabd477e49aff807b1553502e80308a13e7b3b3"
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
    "probe_uses_real_corpus_or_final_test_payload": False,
    "probe_uses_deterministic_synthetic_token_ids_only": True,
    "external_model_or_api_called_by_probe": False,
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
    content = path.read_bytes().replace(b"\r\n", b"\n")
    if b"\r" in content:
        raise ValueError("source text contains unsupported bare CR")
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


def _windows_peak_working_set_mib() -> float | None:
    if sys.platform != "win32":
        return None

    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.argtypes = []
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL
        handle = get_current_process()
        ok = get_process_memory_info(
            handle,
            ctypes.byref(counters),
            ctypes.sizeof(counters),
        )
        if not ok:
            return None
        return float(counters.PeakWorkingSetSize) / 1024.0**2
    except (AttributeError, OSError, ValueError):
        return None


def _process_hwm_mib_approx() -> tuple[float | None, str]:
    if sys.platform == "win32":
        windows_value = _windows_peak_working_set_mib()
        if windows_value is None:
            return None, "unavailable"
        return windows_value, "windows_peak_working_set"

    if _resource is None:
        return None, "unavailable"
    if sys.platform != "darwin" and not sys.platform.startswith("linux"):
        return None, "unavailable"

    try:
        hwm_raw = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError, ValueError):
        return None, "unavailable"
    if sys.platform == "darwin":
        return float(hwm_raw) / 1024.0**2, "ru_maxrss_bytes"
    return float(hwm_raw) / 1024.0, "ru_maxrss_kib"


def _finite_positive(value: Any, name: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a JSON number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _require_exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} keys mismatch: missing={missing}, extra={extra}")


def validate_source_root(root: Path) -> None:
    model_path = root / "src/twelve_six/model.py"
    pyproject_path = root / "pyproject.toml"
    runtime_model_file = getattr(model_module, "__file__", None)
    if runtime_model_file is None or Path(runtime_model_file).resolve() != model_path.resolve():
        raise ValueError("runtime model import path mismatch")
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
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("probe requires torch.float32 default dtype")
    if torch.get_default_device().type != "cpu":
        raise ValueError("probe requires CPU default device")
    previous_threads = torch.get_num_threads()

    try:
        torch.set_num_threads(intraop_threads)
        with (
            torch.random.fork_rng(devices=[]),
            torch.enable_grad(),
            torch.autocast("cpu", enabled=False),
        ):
            torch.default_generator.manual_seed(341)

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
            hwm_mib, hwm_source = _process_hwm_mib_approx()

            return {
                "schema": SCHEMA,
                "source": {
                    "execution_root_main_sha": EXECUTION_ROOT_MAIN_SHA,
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
                    "torch": str(torch.__version__),
                    "cpu": _cpu_name(),
                    "platform": sys.platform,
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
                    "synthetic_loss_samples": losses,
                    "synthetic_loss_median": statistics.median(losses),
                    "parameter_bytes": parameter_bytes,
                    "gradient_bytes": gradient_bytes,
                    "process_hwm_mib_approx": hwm_mib,
                    "process_hwm_source": hwm_source,
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
    finally:
        torch.set_num_threads(previous_threads)


def validate_probe(report: dict[str, Any]) -> None:
    if type(report) is not dict:
        raise ValueError("report must be an object")
    if report.get("schema") != SCHEMA:
        raise ValueError("schema mismatch")
    _require_exact_keys(
        report,
        {"schema", "source", "model", "runtime", "measurement", "planning", "truth_boundary"},
        "report",
    )
    source = report.get("source")
    if type(source) is not dict:
        raise ValueError("source must be an object")
    expected_source = {
        "execution_root_main_sha": EXECUTION_ROOT_MAIN_SHA,
        "model_blob_sha1": MODEL_BLOB_SHA1,
        "pyproject_blob_sha1": PYPROJECT_BLOB_SHA1,
    }
    if source != expected_source:
        raise ValueError("source root mismatch")

    model = report.get("model")
    if type(model) is not dict:
        raise ValueError("model must be an object")
    _require_exact_keys(
        model,
        {
            "model_identity_sha256",
            "init_identity_sha256",
            "parameter_count",
            "vocab_size",
            "sequence_length",
            "micro_batch_size",
            "precision",
            "seed",
            "causal_targets_per_microbatch",
        },
        "model",
    )
    if model.get("model_identity_sha256") != EXPECTED_MODEL_IDENTITY_SHA256:
        raise ValueError("model identity mismatch")
    if model.get("init_identity_sha256") != EXPECTED_INIT_IDENTITY_SHA256:
        raise ValueError("init identity mismatch")
    if type(model.get("parameter_count")) is not int:
        raise ValueError("parameter_count type mismatch")
    if model["parameter_count"] != EXPECTED_PARAMETER_COUNT:
        raise ValueError("parameter_count mismatch")
    expected_model_report = {
        "vocab_size": 256,
        "sequence_length": 128,
        "micro_batch_size": 1,
        "precision": "fp32",
        "seed": 341,
        "causal_targets_per_microbatch": 127,
    }
    for field, expected in expected_model_report.items():
        if model.get(field) != expected or (
            type(expected) is int and type(model.get(field)) is not int
        ):
            raise ValueError(f"model report {field} mismatch")

    runtime = report.get("runtime")
    if type(runtime) is not dict:
        raise ValueError("runtime must be an object")
    _require_exact_keys(
        runtime,
        {
            "python",
            "torch",
            "cpu",
            "platform",
            "cuda_available",
            "intraop_threads",
            "interop_threads",
        },
        "runtime",
    )
    for field in ("python", "torch", "cpu", "platform"):
        if type(runtime.get(field)) is not str or not runtime[field].strip():
            raise ValueError(f"runtime {field} must be a non-empty string")
    if type(runtime.get("cuda_available")) is not bool:
        raise ValueError("runtime cuda_available must be a boolean")
    for field in ("intraop_threads", "interop_threads"):
        if type(runtime.get(field)) is not int or runtime[field] <= 0:
            raise ValueError(f"runtime {field} must be an exact positive integer")

    measurement = report.get("measurement")
    if type(measurement) is not dict:
        raise ValueError("measurement must be an object")
    _require_exact_keys(
        measurement,
        {
            "warmup_samples",
            "measured_samples",
            "elapsed_seconds",
            "median_forward_loss_backward_seconds",
            "median_causal_targets_per_second",
            "synthetic_loss_samples",
            "synthetic_loss_median",
            "parameter_bytes",
            "gradient_bytes",
            "process_hwm_mib_approx",
            "process_hwm_source",
            "parameter_fingerprint_before_sha256",
            "parameter_fingerprint_after_sha256",
            "parameter_fingerprint_unchanged",
            "optimizer_object_created",
            "optimizer_updates",
            "model_updates",
        },
        "measurement",
    )
    warmup_samples = measurement.get("warmup_samples")
    if type(warmup_samples) is not int or warmup_samples < 0:
        raise ValueError("warmup_samples must be an exact non-negative integer")
    measured_samples = measurement.get("measured_samples")
    if type(measured_samples) is not int or measured_samples <= 0:
        raise ValueError("measured_samples must be an exact positive integer")

    elapsed_samples = measurement.get("elapsed_seconds")
    if type(elapsed_samples) is not list or len(elapsed_samples) != measured_samples:
        raise ValueError("elapsed sample count mismatch")
    elapsed_values = [
        _finite_positive(value, f"elapsed_seconds[{index}]")
        for index, value in enumerate(elapsed_samples)
    ]

    seconds = measurement.get("median_forward_loss_backward_seconds")
    throughput = measurement.get("median_causal_targets_per_second")
    seconds_value = _finite_positive(seconds, "median seconds")
    throughput_value = _finite_positive(throughput, "median throughput")
    expected_median_seconds = statistics.median(elapsed_values)
    if not math.isclose(seconds_value, expected_median_seconds, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("median seconds do not match elapsed samples")
    expected_throughput = model["causal_targets_per_microbatch"] / seconds_value
    if not math.isclose(throughput_value, expected_throughput, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("throughput arithmetic mismatch")

    loss_samples = measurement.get("synthetic_loss_samples")
    if type(loss_samples) is not list or len(loss_samples) != measured_samples:
        raise ValueError("synthetic loss sample count mismatch")
    loss_values = [
        _finite_positive(value, f"synthetic_loss_samples[{index}]")
        for index, value in enumerate(loss_samples)
    ]
    loss_median = _finite_positive(
        measurement.get("synthetic_loss_median"), "synthetic loss median"
    )
    if not math.isclose(
        loss_median,
        statistics.median(loss_values),
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError("synthetic loss median does not match samples")

    expected_fp32_bytes = EXPECTED_PARAMETER_COUNT * 4
    for field in ("parameter_bytes", "gradient_bytes"):
        value = measurement.get(field)
        if type(value) is not int:
            raise ValueError(f"{field} must be an exact positive integer")
        if value != expected_fp32_bytes:
            raise ValueError(f"{field} mismatch for exact fp32 model")

    before_fingerprint = measurement.get("parameter_fingerprint_before_sha256")
    after_fingerprint = measurement.get("parameter_fingerprint_after_sha256")
    for name, value in (
        ("parameter_fingerprint_before_sha256", before_fingerprint),
        ("parameter_fingerprint_after_sha256", after_fingerprint),
    ):
        if type(value) is not str or len(value) != 64:
            raise ValueError(f"{name} must be a SHA-256 hex digest")
        if any(character not in "0123456789abcdef" for character in value.lower()):
            raise ValueError(f"{name} must be a SHA-256 hex digest")
    if before_fingerprint != after_fingerprint:
        raise ValueError("parameter fingerprint changed")
    if measurement.get("parameter_fingerprint_unchanged") is not True:
        raise ValueError("parameter fingerprint changed")
    if measurement.get("optimizer_object_created") is not False:
        raise ValueError("optimizer object was created")
    for field in ("optimizer_updates", "model_updates"):
        if type(measurement.get(field)) is not int or measurement[field] != 0:
            raise ValueError(f"{field} must be exact integer zero")

    hwm = measurement.get("process_hwm_mib_approx")
    hwm_source = measurement.get("process_hwm_source")
    runtime_platform = runtime["platform"]
    if hwm_source == "unavailable":
        if hwm is not None:
            raise ValueError("unavailable process HWM must be null")
    elif hwm_source == "windows_peak_working_set":
        if runtime_platform != "win32":
            raise ValueError("Windows process HWM source requires win32 platform")
        _finite_positive(hwm, "process HWM")
    elif hwm_source == "ru_maxrss_bytes":
        if runtime_platform != "darwin":
            raise ValueError("ru_maxrss_bytes requires darwin platform")
        _finite_positive(hwm, "process HWM")
    elif hwm_source == "ru_maxrss_kib":
        if not runtime_platform.startswith("linux"):
            raise ValueError("ru_maxrss_kib platform mismatch")
        _finite_positive(hwm, "process HWM")
    else:
        raise ValueError("process HWM source mismatch")

    truth_boundary = report.get("truth_boundary")
    if type(truth_boundary) is not dict:
        raise ValueError("truth_boundary must be an object")
    _require_exact_keys(truth_boundary, set(TRUTH_BOUNDARY), "truth_boundary")
    for field, expected in TRUTH_BOUNDARY.items():
        actual = truth_boundary[field]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"truth boundary {field} mismatch")

    planning = report.get("planning")
    if type(planning) is not dict:
        raise ValueError("planning must be an object")
    _require_exact_keys(
        planning,
        {
            "scope",
            "target_positions_example",
            "mechanics_only_lower_bound_seconds_example",
            "mechanics_only_lower_bound_hours_example",
            "cross_host_extrapolation_allowed",
            "excluded",
        },
        "planning",
    )
    if planning.get("scope") != "forward+causal_ce+backward_only":
        raise ValueError("planning scope mismatch")
    target_positions = planning.get("target_positions_example")
    if type(target_positions) is not int or target_positions != 20_000_000:
        raise ValueError("planning target position example mismatch")
    lower_bound_seconds = _finite_positive(
        planning.get("mechanics_only_lower_bound_seconds_example"),
        "mechanics lower-bound seconds",
    )
    lower_bound_hours = _finite_positive(
        planning.get("mechanics_only_lower_bound_hours_example"),
        "mechanics lower-bound hours",
    )
    expected_lower_bound_seconds = target_positions / throughput_value
    if not math.isclose(
        lower_bound_seconds,
        expected_lower_bound_seconds,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError("planning lower-bound seconds mismatch")
    if not math.isclose(
        lower_bound_hours,
        lower_bound_seconds / 3600.0,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError("planning lower-bound hours mismatch")
    expected_excluded = [
        "optimizer.step",
        "checkpoint_io",
        "evaluation",
        "packing_data_io",
        "real_pilot_overhead",
    ]
    if planning.get("excluded") != expected_excluded:
        raise ValueError("planning exclusions mismatch")
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
