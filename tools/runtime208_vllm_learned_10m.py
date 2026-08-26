from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SCHEMA = "12-6.runtime208-vllm-learned-10m.v1"
SOURCE_GATE_SCHEMA = "12-6.runtime208-source-gate.v1"
RUNTIME_PACKAGE_SCHEMA = "12-6.vllm-runtime-package-identity.v1"
SOURCE_REPOSITORY = "Oleksii-debug/12-6-ai."
SOURCE_HEAD_SHA = "e055893808c3fa0f9c5deb1ab83203b82aabbd63"
SOURCE_WORKFLOW_RUN_ID = 32938501819
SOURCE_WORKFLOW_ID = 342449937
SOURCE_WORKFLOW_NAME = "SCALE-141 10M Learned Continuation"
SOURCE_ARTIFACT_NAME = "scale141-10m-learned-fallback"
RETAINED_ROLE = "best"
EXPECTED_MODEL_SPEC_SHA256 = "61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998"
EXPECTED_PARAMETER_COUNT = 10_000_640
EXPECTED_TOKENIZER_VERSION = "s0-byte-v1"
EXPECTED_TOKENIZER_CONFIG_SHA256 = (
    "b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1"
)
EXPECTED_TOKENIZER_VOCAB_SHA256 = (
    "905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571"
)
EXPECTED_COMMON_EVAL_ID = "7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113"
EXPECTED_PYTHON = "3.11.16"
EXPECTED_VLLM_IMPORT_VERSION = "0.27.1"
EXPECTED_VLLM_CPU_DIST_VERSION = "0.27.1+cpu"
EXPECTED_TORCH_CPU_VERSION = "2.13.0+cpu"
EXPECTED_TRANSFORMERS_VERSION = "5.15.1"
EXPECTED_SAFETENSORS_VERSION = "0.8.0"
EXPECTED_VLLM_CPU_WHEEL_SHA256 = (
    "36f0e7b2031233ff09e521716723b0e05ab62054c9a9a05d873af43052140f33"
)


class Runtime208Error(RuntimeError):
    pass


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_value(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Runtime208Error(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Runtime208Error(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def expected_model_spec():
    from twelve_six.model import ModelSpec

    spec = ModelSpec.from_dict(
        {
            "schema_version": 1,
            "vocab_size": 256,
            "max_seq_len": 1024,
            "d_model": 256,
            "n_layers": 12,
            "n_heads": 8,
            "n_kv_heads": 2,
            "head_dim": 32,
            "d_ff": 864,
            "activation": "swiglu",
            "norm_kind": "rmsnorm",
            "norm_placement": "pre",
            "norm_eps": 1e-5,
            "position_embedding": "rope",
            "rope_theta": 10000.0,
            "rope_rotary_dim": 32,
            "attention_bias": False,
            "mlp_bias": False,
            "attention_dropout": 0.0,
            "final_norm": True,
            "tie_word_embeddings": True,
            "lm_head_bias": False,
        }
    )
    if spec.parameter_count() != EXPECTED_PARAMETER_COUNT:
        raise Runtime208Error("10M parameter-count constant drift")
    if spec.identity_sha256() != EXPECTED_MODEL_SPEC_SHA256:
        raise Runtime208Error("10M ModelSpec constant drift")
    return spec


def validate_source_run(run: dict[str, Any]) -> dict[str, Any]:
    exact = {
        "id": SOURCE_WORKFLOW_RUN_ID,
        "workflow_id": SOURCE_WORKFLOW_ID,
        "name": SOURCE_WORKFLOW_NAME,
        "head_sha": SOURCE_HEAD_SHA,
    }
    for key, expected in exact.items():
        if run.get(key) != expected:
            raise Runtime208Error(
                f"SCALE-141 producer {key} mismatch: expected {expected!r}, got {run.get(key)!r}"
            )

    status = run.get("status")
    conclusion = run.get("conclusion")
    if status == "completed" and conclusion == "success":
        state = "SOURCE_TERMINAL_SUCCESS"
        ready = True
    elif status == "completed":
        state = "BLOCKED_SOURCE_NOT_SUCCESS"
        ready = False
    else:
        state = "BLOCKED_SOURCE_NOT_TERMINAL"
        ready = False

    payload = {
        "schema": SOURCE_GATE_SCHEMA,
        "state": state,
        "ready_for_preparation": ready,
        "repository": SOURCE_REPOSITORY,
        "producer": {
            "workflow_run_id": SOURCE_WORKFLOW_RUN_ID,
            "workflow_id": SOURCE_WORKFLOW_ID,
            "workflow_name": SOURCE_WORKFLOW_NAME,
            "head_sha": SOURCE_HEAD_SHA,
            "status": status,
            "conclusion": conclusion,
        },
        "required_artifact_name": SOURCE_ARTIFACT_NAME,
        "foreign_pretrained_weights_used": False,
        "paid_compute": False,
    }
    payload["identity_sha256"] = _sha256_value(payload)
    return payload


def validate_artifact_metadata(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("name") != SOURCE_ARTIFACT_NAME:
        raise Runtime208Error("SCALE-141 artifact name mismatch")
    if artifact.get("expired") is not False:
        raise Runtime208Error("SCALE-141 artifact is expired or expiration state is unknown")
    artifact_id = artifact.get("id")
    if not isinstance(artifact_id, int) or isinstance(artifact_id, bool) or artifact_id <= 0:
        raise Runtime208Error("SCALE-141 artifact ID is invalid")
    digest = artifact.get("digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:") or len(digest) != 71:
        raise Runtime208Error("SCALE-141 artifact digest is not an exact SHA-256 identity")
    workflow_run = artifact.get("workflow_run")
    if not isinstance(workflow_run, dict):
        raise Runtime208Error("SCALE-141 artifact is missing workflow-run provenance")
    if workflow_run.get("id") != SOURCE_WORKFLOW_RUN_ID:
        raise Runtime208Error("SCALE-141 artifact workflow-run ID mismatch")
    if workflow_run.get("head_sha") != SOURCE_HEAD_SHA:
        raise Runtime208Error("SCALE-141 artifact head SHA mismatch")
    return {
        "artifact_id": artifact_id,
        "artifact_name": SOURCE_ARTIFACT_NAME,
        "artifact_digest": digest,
        "workflow_run_id": SOURCE_WORKFLOW_RUN_ID,
        "head_sha": SOURCE_HEAD_SHA,
    }


def _evidence_root(artifact_root: Path) -> Path:
    candidates = (artifact_root, artifact_root / "scale141-evidence")
    for candidate in candidates:
        if (candidate / "fresh-verification.json").is_file() and (
            candidate / "retained" / "index.json"
        ).is_file():
            return candidate
    raise Runtime208Error("downloaded SCALE-141 artifact has no retained fresh-verification payload")


def validate_retained_checkpoint(artifact_root: Path) -> dict[str, Any]:
    root = _evidence_root(artifact_root)
    fresh = _read_json(root / "fresh-verification.json")
    retained = _read_json(root / "retained" / "index.json")

    if fresh.get("source_sha") != SOURCE_HEAD_SHA:
        raise Runtime208Error("fresh-verification source SHA mismatch")
    fresh_status = fresh.get("fresh_verification")
    if not isinstance(fresh_status, dict) or fresh_status.get("status") != "PASS":
        raise Runtime208Error("SCALE-141 fresh verification is not PASS")
    common_eval = fresh.get("ladder_common_evaluation")
    if not isinstance(common_eval, dict):
        raise Runtime208Error("SCALE-141 fresh verification lacks ladder-common evaluation")
    common_identity = common_eval.get("identity")
    if not isinstance(common_identity, dict) or (
        common_identity.get("identity_sha256") != EXPECTED_COMMON_EVAL_ID
    ):
        raise Runtime208Error("SCALE-141 M150 common evaluation identity mismatch")

    if retained.get("source_sha") != SOURCE_HEAD_SHA:
        raise Runtime208Error("retained-index source SHA mismatch")
    roles = retained.get("roles")
    if not isinstance(roles, dict) or not isinstance(roles.get(RETAINED_ROLE), dict):
        raise Runtime208Error("retained-index has no best checkpoint")
    role = roles[RETAINED_ROLE]
    if role.get("fresh_verification") != "PASS":
        raise Runtime208Error("retained best checkpoint is not fresh-verification PASS")

    evidence = fresh.get("evidence")
    if not isinstance(evidence, dict) or not isinstance(evidence.get(RETAINED_ROLE), dict):
        raise Runtime208Error("fresh verification has no best-checkpoint evidence")
    fresh_best = evidence[RETAINED_ROLE]
    checkpoint_id = role.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or len(checkpoint_id) != 64:
        raise Runtime208Error("retained best checkpoint ID is invalid")
    if fresh_best.get("checkpoint_id") != checkpoint_id:
        raise Runtime208Error("retained best checkpoint ID differs from fresh verification")
    if fresh_best.get("target_optimized_tokens") != role.get("target_optimized_tokens"):
        raise Runtime208Error("retained best optimized-token target mismatch")

    checkpoint = root / "retained" / RETAINED_ROLE
    if not checkpoint.is_dir():
        raise Runtime208Error("retained best checkpoint directory is missing")
    return {
        "evidence_root": root,
        "checkpoint": checkpoint,
        "checkpoint_id": checkpoint_id,
        "target_optimized_tokens": role.get("target_optimized_tokens"),
        "fresh_verification_identity_sha256": fresh.get("identity_sha256"),
        "common_evaluation_identity_sha256": EXPECTED_COMMON_EVAL_ID,
    }


def _read_verified_wheel_sha(path: Path) -> str:
    try:
        first = path.read_text(encoding="utf-8").strip().split()[0]
    except (OSError, UnicodeDecodeError, IndexError) as exc:
        raise Runtime208Error("invalid vLLM wheel SHA evidence") from exc
    if first != EXPECTED_VLLM_CPU_WHEEL_SHA256:
        raise Runtime208Error("vLLM CPU wheel SHA-256 mismatch")
    return first


def installed_runtime_identity(*, require_cuda: bool) -> dict[str, Any]:
    import torch
    import vllm

    cuda_available = bool(torch.cuda.is_available())
    if require_cuda and not cuda_available:
        raise Runtime208Error("runtime package contract requested CUDA but no CUDA GPU is visible")
    value: dict[str, Any] = {
        "schema": RUNTIME_PACKAGE_SCHEMA,
        "python": platform.python_version(),
        "vllm_import_version": vllm.__version__,
        "vllm_distribution_version": metadata.version("vllm"),
        "torch": torch.__version__,
        "transformers": metadata.version("transformers"),
        "safetensors": metadata.version("safetensors"),
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
    }
    if cuda_available:
        value["cuda_device_name"] = torch.cuda.get_device_name(0)
        value["cuda_device_capability"] = list(torch.cuda.get_device_capability(0))
    value["identity_sha256"] = _sha256_value(value)
    return value


def validate_cpu_runtime_identity(identity: dict[str, Any], wheel_sha256: str) -> None:
    expected = {
        "python": EXPECTED_PYTHON,
        "vllm_import_version": EXPECTED_VLLM_IMPORT_VERSION,
        "vllm_distribution_version": EXPECTED_VLLM_CPU_DIST_VERSION,
        "torch": EXPECTED_TORCH_CPU_VERSION,
        "transformers": EXPECTED_TRANSFORMERS_VERSION,
        "safetensors": EXPECTED_SAFETENSORS_VERSION,
        "cuda_available": False,
        "torch_cuda_version": None,
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise Runtime208Error(
                f"CPU vLLM purpose identity mismatch for {key}: "
                f"expected {value!r}, got {identity.get(key)!r}"
            )
    if wheel_sha256 != EXPECTED_VLLM_CPU_WHEEL_SHA256:
        raise Runtime208Error("CPU vLLM purpose wheel identity mismatch")


def prepare_10m(args: argparse.Namespace) -> dict[str, Any]:
    from tools.prepare_vllm_learned_checkpoint import prepare as prepare_learned
    from twelve_six.inference.vllm_native_llama import probe_vllm_import_and_config

    artifact_metadata = validate_artifact_metadata(_read_json(args.artifact_json))
    retained = validate_retained_checkpoint(args.artifact_root.resolve())
    expected_model_spec()

    wheel_sha256 = _read_verified_wheel_sha(args.vllm_wheel_sha_file)
    runtime_identity = installed_runtime_identity(require_cuda=False)
    validate_cpu_runtime_identity(runtime_identity, wheel_sha256)

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise Runtime208Error(f"output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    logical_interop_root = args.output_root / "interop"

    prepared = prepare_learned(
        SimpleNamespace(
            checkpoint=retained["checkpoint"],
            output_root=logical_interop_root,
            expected_checkpoint_id=retained["checkpoint_id"],
            expected_model_spec_sha256=EXPECTED_MODEL_SPEC_SHA256,
            expected_parameter_count=EXPECTED_PARAMETER_COUNT,
            expected_tokenizer_config_sha256=EXPECTED_TOKENIZER_CONFIG_SHA256,
            expected_tokenizer_vocab_sha256=EXPECTED_TOKENIZER_VOCAB_SHA256,
            source_repository=SOURCE_REPOSITORY,
            source_artifact_id=artifact_metadata["artifact_id"],
            source_artifact_name=SOURCE_ARTIFACT_NAME,
            source_artifact_digest=artifact_metadata["artifact_digest"],
            source_artifact_head_sha=SOURCE_HEAD_SHA,
            expected_vllm_version=EXPECTED_VLLM_IMPORT_VERSION,
            expected_vllm_dist_version=None,
        )
    )
    learned = prepared["learned_checkpoint"]
    if learned.get("parameter_count") != EXPECTED_PARAMETER_COUNT:
        raise Runtime208Error("prepared 10M checkpoint parameter count mismatch")
    if learned.get("model_spec_sha256") != EXPECTED_MODEL_SPEC_SHA256:
        raise Runtime208Error("prepared 10M ModelSpec mismatch")
    if prepared["tokenizer"].get("id") != EXPECTED_TOKENIZER_VERSION:
        raise Runtime208Error("prepared 10M tokenizer version mismatch")

    probe = probe_vllm_import_and_config(
        output_root / "interop" / "vllm-model",
        expected_vllm_version=EXPECTED_VLLM_IMPORT_VERSION,
    )
    if probe.architecture != "LlamaForCausalLM" or not probe.llama_registered:
        raise Runtime208Error("maintained vLLM Llama registration was not proven")
    if probe.max_model_len != 1024:
        raise Runtime208Error("vLLM ModelConfig context length mismatch for learned 10M")

    gpu_command = prepared["gpu_parity_command"] + (
        " --runtime-package-contract-json runtime208-gpu-package-identity.json"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PREPARED_NOT_GPU_EXECUTED",
        "producer": {
            "repository": SOURCE_REPOSITORY,
            "head_sha": SOURCE_HEAD_SHA,
            "workflow_run_id": SOURCE_WORKFLOW_RUN_ID,
            "workflow_id": SOURCE_WORKFLOW_ID,
            "artifact": artifact_metadata,
            "retained_role": RETAINED_ROLE,
            "fresh_verification_identity_sha256": retained[
                "fresh_verification_identity_sha256"
            ],
            "common_evaluation_identity_sha256": EXPECTED_COMMON_EVAL_ID,
        },
        "checkpoint": {
            "checkpoint_id": retained["checkpoint_id"],
            "target_optimized_tokens": retained["target_optimized_tokens"],
            "model_spec_sha256": EXPECTED_MODEL_SPEC_SHA256,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
        },
        "tokenizer": prepared["tokenizer"],
        "standard_llama_export": prepared["standard_llama_export"],
        "vllm_binding": {
            "implementation": "BUILTIN_LLAMA",
            "custom_vllm_model_implemented": False,
            "cpu_import_config_probe": probe.to_dict(),
        },
        "cpu_vllm_purpose_runtime": {
            **runtime_identity,
            "verified_wheel_sha256": wheel_sha256,
        },
        "gpu_parity": {
            "status": "NOT_EXECUTED",
            "required_device": "COMPATIBLE_CUDA_GPU",
            "full_raw_logits": "REQUIRED",
            "greedy_token_ids": "REQUIRED_EXACT",
            "decoded_continuations": "REQUIRED_EXACT",
            "prompts": ["UA", "EN", "code", "max_context_tokens_minus_one"],
            "atol": 1e-5,
            "rtol": 1e-5,
            "runtime_package_contract_required": True,
            "command": gpu_command,
        },
        "foreign_pretrained_weights_used": False,
        "paid_compute": False,
    }
    result["identity_sha256"] = _sha256_value(result)
    _write_json(output_root / "runtime208.json", result)
    return result


def source_gate(args: argparse.Namespace) -> dict[str, Any]:
    result = validate_source_run(_read_json(args.run_json))
    _write_json(args.output, result)
    return result


def write_runtime_package_contract(args: argparse.Namespace) -> dict[str, Any]:
    result = installed_runtime_identity(require_cuda=args.require_cuda)
    _write_json(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gate = subparsers.add_parser("source-gate")
    gate.add_argument("--run-json", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)

    prep = subparsers.add_parser("prepare")
    prep.add_argument("--artifact-root", type=Path, required=True)
    prep.add_argument("--artifact-json", type=Path, required=True)
    prep.add_argument("--vllm-wheel-sha-file", type=Path, required=True)
    prep.add_argument("--output-root", type=Path, required=True)

    package = subparsers.add_parser("runtime-package-contract")
    package.add_argument("--require-cuda", action="store_true")
    package.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "source-gate":
            result = source_gate(args)
        elif args.command == "prepare":
            result = prepare_10m(args)
        else:
            result = write_runtime_package_contract(args)
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
