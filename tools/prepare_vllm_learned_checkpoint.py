from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

from twelve_six.checkpoint import export_hf_directory
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.inference.transformers_llama import llama_config_dict
from twelve_six.inference.vllm_native_llama import (
    materialize_vllm_llama_directory,
    verify_vllm_llama_directory,
)

SCHEMA = "12-6.vllm-native-llama-learned-preparation.v1"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_equal(name: str, actual: object, expected: object | None) -> None:
    if expected is not None and actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _copy_checkpoint(source: Path, target: Path) -> Path:
    if target.exists():
        raise FileExistsError(f"checkpoint copy target already exists: {target}")
    shutil.copytree(source, target, symlinks=False)
    return target


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source_checkpoint = args.checkpoint.resolve()
    reference = load_first_party_backend(source_checkpoint)
    diagnostics = reference.diagnostics()
    identity = reference.manifest["identity"]
    spec = reference.model.spec

    step = identity.get("step")
    tokens_seen = identity.get("tokens_seen")
    if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
        raise ValueError("learned vLLM preparation requires a checkpoint with step > 0")
    if not isinstance(tokens_seen, int) or isinstance(tokens_seen, bool) or tokens_seen <= 0:
        raise ValueError("learned vLLM preparation requires a checkpoint with tokens_seen > 0")

    _require_equal("checkpoint_id", diagnostics["checkpoint_id"], args.expected_checkpoint_id)
    _require_equal(
        "model_spec_sha256", diagnostics["model_spec_sha256"], args.expected_model_spec_sha256
    )
    _require_equal(
        "parameter_count",
        diagnostics["parameter_count"],
        getattr(args, "expected_parameter_count", None),
    )
    _require_equal(
        "tokenizer_config_sha256",
        diagnostics["tokenizer_config_sha256"],
        args.expected_tokenizer_config_sha256,
    )
    _require_equal(
        "tokenizer_vocab_sha256",
        diagnostics["tokenizer_vocab_sha256"],
        args.expected_tokenizer_vocab_sha256,
    )

    logical_output_root = args.output_root
    output_root = logical_output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    output_root.mkdir(parents=True)

    checkpoint = _copy_checkpoint(source_checkpoint, output_root / "source-checkpoint")
    hf_export = export_hf_directory(
        checkpoint,
        output_root / "hf-export",
        hf_config=llama_config_dict(spec),
    )
    model_dir = materialize_vllm_llama_directory(hf_export, output_root / "vllm-model")
    binding = verify_vllm_llama_directory(model_dir)

    _require_equal("runtime checkpoint_id", binding["checkpoint_id"], diagnostics["checkpoint_id"])
    _require_equal(
        "runtime ModelSpec", binding["model_spec_sha256"], diagnostics["model_spec_sha256"]
    )
    if binding["source_model_spec"] != spec.to_dict():
        raise ValueError("runtime ModelSpec payload differs from learned checkpoint")

    logical_checkpoint = logical_output_root / "source-checkpoint"
    logical_hf_export = logical_output_root / "hf-export"
    logical_model_dir = logical_output_root / "vllm-model"
    logical_gpu_output = logical_output_root / "gpu-parity.json"
    parity_command = [
        "python",
        "tools/validate_vllm_learned_parity.py",
        "--checkpoint",
        str(logical_checkpoint),
        "--model-dir",
        str(logical_model_dir),
        "--source-artifact-id",
        str(args.source_artifact_id),
        "--source-artifact-digest",
        args.source_artifact_digest,
        "--source-artifact-head-sha",
        args.source_artifact_head_sha,
        "--max-new-tokens",
        "8",
        "--atol",
        "1e-5",
        "--rtol",
        "1e-5",
        "--dtype",
        "float32",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.5",
    ]
    expected_vllm_version = getattr(args, "expected_vllm_version", None)
    if expected_vllm_version is not None:
        parity_command.extend(["--expected-vllm-version", expected_vllm_version])
    expected_vllm_dist_version = getattr(args, "expected_vllm_dist_version", None)
    if expected_vllm_dist_version is not None:
        parity_command.extend(["--expected-vllm-dist-version", expected_vllm_dist_version])
    parity_command.extend(["--output", str(logical_gpu_output)])

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "learned_checkpoint": {
            **diagnostics,
            "model_spec": spec.to_dict(),
        },
        "learned_source_artifact": {
            "repository": args.source_repository,
            "artifact_id": args.source_artifact_id,
            "artifact_name": args.source_artifact_name,
            "artifact_digest": args.source_artifact_digest,
            "head_sha": args.source_artifact_head_sha,
        },
        "standard_llama_export": {
            "source_export": binding["source_export"],
            "runtime_config_sha256": binding["target"]["config_sha256"],
            "runtime_weights_sha256": binding["target"]["weights_sha256"],
            "architecture": binding["target"]["architecture"],
            "model_type": binding["target"]["model_type"],
            "qk_rope_basis_conversion": "INCUMBENT_D07_EXACT",
        },
        "tokenizer": {
            "owner": "12-6_CANONICAL_OUTSIDE_VLLM",
            "id": reference.tokenizer.identity.version,
            "config_sha256": reference.tokenizer.identity.config_sha256,
            "vocab_sha256": reference.tokenizer.identity.vocab_sha256,
            "vocab_size": reference.tokenizer.vocab_size,
            "vllm_tokenizer_initialized": False,
            "chat_template_used": False,
        },
        "execution_contract": binding["execution_contract"],
        "paths": {
            "checkpoint": str(logical_checkpoint),
            "hf_export": str(logical_hf_export),
            "vllm_model": str(logical_model_dir),
        },
        "gpu_parity_command": shlex.join(parity_command),
        "pretrained_foreign_weights_used": False,
        "paid_compute": False,
    }
    payload["evidence_sha256"] = _sha256_bytes(_canonical_json_bytes(payload))
    (output_root / "prepared.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind a verified learned 12-6 checkpoint to the incumbent D05 -> standard-Llama "
            "-> native-vLLM path without creating another model adapter."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-id", required=True)
    parser.add_argument("--expected-model-spec-sha256", required=True)
    parser.add_argument("--expected-parameter-count", type=int)
    parser.add_argument("--expected-tokenizer-config-sha256", required=True)
    parser.add_argument("--expected-tokenizer-vocab-sha256", required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-artifact-id", type=int, required=True)
    parser.add_argument("--source-artifact-name", required=True)
    parser.add_argument("--source-artifact-digest", required=True)
    parser.add_argument("--source-artifact-head-sha", required=True)
    parser.add_argument("--expected-vllm-version")
    parser.add_argument("--expected-vllm-dist-version")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = prepare(args)
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
