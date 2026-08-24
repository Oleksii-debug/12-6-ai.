"""Runtime parity evidence for transactional D05 HF-style exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from safetensors.torch import load as load_safetensors_bytes

from twelve_six.checkpoint import verify_hf_directory
from twelve_six.model import TwelveSixDecoder

from .first_party import FirstPartyInferenceBackend, load_first_party_backend
from .parity import compare_backends

EVIDENCE_SCHEMA = "12-6.hf-style-export-runtime-parity.v1"
EXPORTED_WEIGHTS_NAME = "model.safetensors"


def _canonical_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_hf_style_export_backend(
    checkpoint_dir: str | Path,
    export_dir: str | Path,
) -> FirstPartyInferenceBackend:
    """Load an HF-style export only when its consumed bytes match D05 verification.

    D05's v2 verifier validates an exact export snapshot and returns the bound weight
    hash/checkpoint identity. Candidate weight bytes are then read for execution and
    independently hashed before decode. If the path changes after verification, the
    consumed bytes must still match the verifier's cryptographic identity or loading
    fails closed.

    Architecture/tokenizer semantics come from the existing canonical first-party
    checkpoint loader; this module does not duplicate D01 or D04 logic.
    """

    checkpoint_path = Path(checkpoint_dir)
    export_path = Path(export_dir)
    reference = load_first_party_backend(checkpoint_path)
    attestation = verify_hf_directory(export_path)

    if attestation.get("checkpoint_id") != reference.manifest.get("checkpoint_id"):
        raise ValueError("HF-style export checkpoint_id does not match reference checkpoint")
    expected_weight_hash = reference.manifest["files"]["weights.safetensors"]["sha256"]
    if attestation.get("model_safetensors_sha256") != expected_weight_hash:
        raise ValueError("HF-style export weight identity does not match reference checkpoint")

    candidate_bytes = (export_path / EXPORTED_WEIGHTS_NAME).read_bytes()
    if _sha256_bytes(candidate_bytes) != attestation["model_safetensors_sha256"]:
        raise ValueError("consumed HF-style export bytes changed after verification")
    try:
        state = load_safetensors_bytes(candidate_bytes)
    except Exception as exc:
        raise ValueError("verified HF-style model.safetensors cannot be decoded") from exc

    model = TwelveSixDecoder(reference.model.spec)
    try:
        model.load_state_dict(state, strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("HF-style export weights are incompatible with canonical D01 model") from exc
    model.eval()
    return FirstPartyInferenceBackend(
        model,
        reference.tokenizer,
        manifest=reference.manifest,
        checkpoint_path=export_path,
    )


def collect_hf_export_parity_evidence(
    checkpoint_dir: str | Path,
    export_dir: str | Path,
    prompts: tuple[str, ...] | list[str],
    *,
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    """Compare canonical checkpoint inference to exported bytes at zero tolerance."""

    if not prompts:
        raise ValueError("at least one parity prompt is required")
    if any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError("parity prompts must be non-empty strings")

    checkpoint_path = Path(checkpoint_dir)
    export_path = Path(export_dir)
    reference = load_first_party_backend(checkpoint_path)
    candidate = load_hf_style_export_backend(checkpoint_path, export_path)
    attestation = verify_hf_directory(export_path)

    report = compare_backends(
        reference,
        candidate,
        tuple(prompts),
        max_new_tokens=max_new_tokens,
        atol=0.0,
        rtol=0.0,
    )
    payload: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "passed": report.passed,
        "checkpoint_id": reference.manifest["checkpoint_id"],
        "source_git_sha": reference.manifest["identity"]["git_sha"],
        "model_spec_sha256": reference.manifest["identity"]["model_spec_hash"],
        "tokenizer_config_sha256": reference.manifest["identity"]["tokenizer_hash"],
        "tokenizer_vocab_sha256": reference.manifest["identity"]["tokenizer_vocab_hash"],
        "source_manifest_sha256": attestation["source_manifest_sha256"],
        "reference_weights_sha256": reference.manifest["files"]["weights.safetensors"][
            "sha256"
        ],
        "candidate_weights_sha256": attestation["model_safetensors_sha256"],
        "candidate_config_sha256": attestation["config_sha256"],
        "prompt_sha256": [
            hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts
        ],
        "parity": report.to_dict(),
        "compatibility": {
            "layout": "HF_STYLE_SAFETENSORS_DIRECTORY",
            "runtime_logit_token_decode_parity": "PASS" if report.passed else "FAIL",
            "tolerance": "EXACT_ZERO",
            "transformers_architecture": "NOT_CLAIMED",
        },
        "truth_boundary": {
            "native_12_6_runtime_only": True,
            "transformers_runtime_claimed": False,
            "vllm_runtime_claimed": False,
            "gguf_llamacpp_claimed": False,
            "promotion_authority": False,
        },
    }
    payload["evidence_sha256"] = _canonical_hash(payload)
    return payload


def assert_hf_export_parity(
    checkpoint_dir: str | Path,
    export_dir: str | Path,
    prompts: tuple[str, ...] | list[str],
    *,
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    """Return evidence only when exact runtime parity passes."""

    evidence = collect_hf_export_parity_evidence(
        checkpoint_dir,
        export_dir,
        prompts,
        max_new_tokens=max_new_tokens,
    )
    if not evidence["passed"]:
        raise RuntimeError(
            f"HF-style export runtime parity failed: {evidence['parity']['failures']}"
        )
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m twelve_six.inference.hf_export_parity",
        description=(
            "Compare verified D05 HF-style exported bytes with canonical first-party "
            "checkpoint inference at exact zero tolerance."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export", dest="export_dir", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evidence = assert_hf_export_parity(
            args.checkpoint,
            args.export_dir,
            tuple(args.prompt),
            max_new_tokens=args.max_new_tokens,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    serialized = json.dumps(
        evidence,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    if args.json:
        print(serialized)
    else:
        parity = evidence["parity"]
        print(
            "hf-export-parity: PASS "
            f"checkpoint_id={evidence['checkpoint_id']} "
            f"prompts={parity['prompts_compared']} steps={parity['steps_compared']} "
            f"max_abs_error={parity['max_abs_error']:.12g} "
            f"max_rel_error={parity['max_rel_error']:.12g} "
            f"evidence_sha256={evidence['evidence_sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
