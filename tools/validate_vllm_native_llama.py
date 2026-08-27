from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from twelve_six.inference.vllm_native_llama import (
    collect_vllm_runtime_parity,
    materialize_vllm_llama_directory,
    probe_vllm_import_and_config,
    verify_vllm_llama_directory,
)
from twelve_six.model import ModelSpec

DEFAULT_VLLM_VERSION = "0.27.1"
DEFAULT_PROBES = (
    "Hello from 12-6.",
    "Привіт від 12-6.",
    "def square(x): return x * x",
)


def _write_json(payload: object, output: Path | None) -> None:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    if output is None:
        print(encoded)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded + "\n", encoding="utf-8")
    print(output)


def _context_probe(model_dir: Path) -> str:
    provenance = verify_vllm_llama_directory(model_dir)
    spec = ModelSpec.from_dict(dict(provenance["source_model_spec"]))
    if spec.max_seq_len < 2:
        raise ValueError("ModelSpec max_seq_len must be at least 2 for context parity")
    return "x" * (spec.max_seq_len - 1)


def _materialize(args: argparse.Namespace) -> int:
    target = materialize_vllm_llama_directory(args.source_export, args.output_dir)
    _write_json(verify_vllm_llama_directory(target), args.report)
    return 0


def _probe(args: argparse.Namespace) -> int:
    result = probe_vllm_import_and_config(
        args.model_dir,
        expected_vllm_version=args.expected_vllm_version,
    )
    _write_json(result.to_dict(), args.output)
    return 0


def _parity(args: argparse.Namespace) -> int:
    prompts = list(args.prompt or DEFAULT_PROBES)
    boundary = _context_probe(args.model_dir)
    if boundary not in prompts:
        prompts.append(boundary)

    evidence = collect_vllm_runtime_parity(
        args.checkpoint,
        args.model_dir,
        prompts,
        max_new_tokens=args.max_new_tokens,
        atol=args.atol,
        rtol=args.rtol,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    _write_json(evidence, args.output)
    return 0 if evidence["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize and validate 12-6 exports through vLLM's built-in "
            "LlamaForCausalLM implementation."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser(
        "materialize",
        help="convert one verified D05 HF-style export to standard Llama bytes",
    )
    materialize.add_argument("--source-export", type=Path, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    materialize.add_argument("--report", type=Path)
    materialize.set_defaults(func=_materialize)

    probe = subparsers.add_parser(
        "probe",
        help="import installed vLLM and construct its Llama ModelConfig",
    )
    probe.add_argument("--model-dir", type=Path, required=True)
    probe.add_argument(
        "--expected-vllm-version",
        default=DEFAULT_VLLM_VERSION,
        help="exact installed vLLM version required for this validation run",
    )
    probe.add_argument("--output", type=Path)
    probe.set_defaults(func=_probe)

    parity = subparsers.add_parser(
        "parity",
        help="execute full raw-logit/greedy/decode/context parity against first-party",
    )
    parity.add_argument("--checkpoint", type=Path, required=True)
    parity.add_argument("--model-dir", type=Path, required=True)
    parity.add_argument(
        "--prompt",
        action="append",
        help=(
            "parity prompt; repeat for multiple probes. When omitted, English, "
            "Ukrainian and code probes are used."
        ),
    )
    parity.add_argument("--max-new-tokens", type=int, default=8)
    parity.add_argument("--atol", type=float, default=1e-5)
    parity.add_argument("--rtol", type=float, default=1e-5)
    parity.add_argument("--dtype", default="float32")
    parity.add_argument("--tensor-parallel-size", type=int, default=1)
    parity.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parity.add_argument("--output", type=Path, required=True)
    parity.set_defaults(func=_parity)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
