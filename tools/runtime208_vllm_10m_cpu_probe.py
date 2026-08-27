from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from tools.runtime208_vllm_learned_10m import (
    EXPECTED_MODEL_SPEC_SHA256,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_VLLM_CPU_WHEEL_SHA256,
    EXPECTED_VLLM_IMPORT_VERSION,
    Runtime208Error,
    expected_model_spec,
    installed_runtime_identity,
    validate_cpu_runtime_identity,
)
from twelve_six.inference.transformers_llama import llama_config_dict

SCHEMA = "12-6.runtime208-vllm-10m-cpu-config-probe.v1"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _wheel_sha(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip().split()[0]
    except (OSError, UnicodeDecodeError, IndexError) as exc:
        raise Runtime208Error("invalid exact-vLLM wheel SHA evidence") from exc
    if value != EXPECTED_VLLM_CPU_WHEEL_SHA256:
        raise Runtime208Error("exact-vLLM CPU wheel SHA mismatch")
    return value


def collect(wheel_sha_file: Path) -> dict[str, Any]:
    import vllm
    from vllm import ModelRegistry
    from vllm.engine.arg_utils import EngineArgs

    spec = expected_model_spec()
    config = llama_config_dict(spec)
    runtime = installed_runtime_identity(require_cuda=False)
    wheel_sha256 = _wheel_sha(wheel_sha_file)
    validate_cpu_runtime_identity(runtime, wheel_sha256)

    if vllm.__version__ != EXPECTED_VLLM_IMPORT_VERSION:
        raise Runtime208Error("vLLM import version drifted during exact 10M CPU config probe")
    supported = set(ModelRegistry.get_supported_archs())
    if "LlamaForCausalLM" not in supported:
        raise Runtime208Error("maintained LlamaForCausalLM is not registered in vLLM")

    with tempfile.TemporaryDirectory(prefix="runtime208-10m-config-") as temp_name:
        root = Path(temp_name)
        (root / "config.json").write_text(
            json.dumps(config, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        args = EngineArgs(
            model=str(root),
            skip_tokenizer_init=True,
            trust_remote_code=False,
            dtype="float32",
            max_model_len=spec.max_seq_len,
            max_logprobs=-1,
            logprobs_mode="raw_logits",
        )
        model_config = args.create_model_config()

    architecture = getattr(model_config, "architecture", None)
    max_model_len = getattr(model_config, "max_model_len", None)
    if architecture != "LlamaForCausalLM":
        raise Runtime208Error(f"vLLM resolved unexpected 10M architecture: {architecture!r}")
    if max_model_len != 1024:
        raise Runtime208Error(f"vLLM resolved unexpected 10M context: {max_model_len!r}")

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PREPARED_REQUIRES_TERMINAL_GREEN_SCALE141_ARTIFACT",
        "model": {
            "model_spec_sha256": EXPECTED_MODEL_SPEC_SHA256,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "model_spec": spec.to_dict(),
            "standard_llama_config_sha256": _hash(config),
        },
        "vllm": {
            "implementation": "BUILTIN_LLAMA",
            "custom_model_implemented": False,
            "architecture_registered": True,
            "resolved_architecture": architecture,
            "max_model_len": max_model_len,
            "skip_tokenizer_init": True,
            "trust_remote_code": False,
            "logprobs_mode": "raw_logits",
            "max_logprobs": -1,
        },
        "cpu_runtime_package_identity": {
            **runtime,
            "verified_vllm_wheel_sha256": wheel_sha256,
        },
        "checkpoint_bound": False,
        "learned_checkpoint_reason": "REQUIRES_TERMINAL_GREEN_SCALE141_RETAINED_BEST",
        "logits_generation_parity_executed": False,
        "foreign_pretrained_weights_used": False,
        "paid_compute": False,
    }
    result["identity_sha256"] = _hash(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-wheel-sha-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = collect(args.vllm_wheel_sha_file)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
