"""Runtime parity evidence for D05 HF-style exports using canonical D07 inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from safetensors.torch import load as load_safetensors_bytes

from twelve_six.checkpoint.hf_export import verify_hf_directory
from twelve_six.model import TwelveSixDecoder

from .first_party import FirstPartyInferenceBackend, load_first_party_backend
from .parity import compare_backends

EVIDENCE_SCHEMA = "12-6.hf-style-export-runtime-parity.v1"


def _canonical_hash(value: dict[str, Any]) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_hf_style_export_backend(
    checkpoint_dir: str | Path,
    export_dir: str | Path,
) -> FirstPartyInferenceBackend:
    """Load exported weight bytes only after source-bound D05 verification.

    Architecture/tokenizer semantics are not reimplemented here. The canonical
    first-party loader verifies the D01 ModelSpec + D04 tokenizer + D05 source
    checkpoint first. The export verifier then proves that the candidate manifest
    and weight bytes are exact copies of that same verified checkpoint. Only those
    already-verified candidate bytes are decoded into a fresh D01 model.
    """

    checkpoint_path = Path(checkpoint_dir)
    export_path = Path(export_dir)
    reference = load_first_party_backend(checkpoint_path)
    verified_export = verify_hf_directory(checkpoint_path, export_path)

    try:
        state = load_safetensors_bytes(verified_export.weights_bytes)
    except Exception as exc:
        raise ValueError("verified exported model.safetensors cannot be decoded") from exc

    model = TwelveSixDecoder(reference.model.spec)
    try:
        model.load_state_dict(state, strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("exported weights are incompatible with canonical D01 model") from exc
    model.eval()

    return FirstPartyInferenceBackend(
        model,
        reference.tokenizer,
        manifest=verified_export.source_manifest,
        checkpoint_path=export_path,
    )


def collect_hf_export_parity_evidence(
    checkpoint_dir: str | Path,
    export_dir: str | Path,
    prompts: tuple[str, ...] | list[str],
    *,
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    """Compare canonical checkpoint inference against exported bytes at zero tolerance."""

    if not prompts:
        raise ValueError("at least one parity prompt is required")
    if any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError("parity prompts must be non-empty strings")

    checkpoint_path = Path(checkpoint_dir)
    export_path = Path(export_dir)
    reference = load_first_party_backend(checkpoint_path)
    candidate = load_hf_style_export_backend(checkpoint_path, export_path)
    verified_export = verify_hf_directory(checkpoint_path, export_path)

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
        "source_manifest_sha256": verified_export.attestation["source_manifest_sha256"],
        "reference_weights_sha256": verified_export.parity_request[
            "reference_weights_sha256"
        ],
        "candidate_weights_sha256": verified_export.parity_request[
            "candidate_weights_sha256"
        ],
        "candidate_config_sha256": verified_export.attestation["config_sha256"],
        "prompt_sha256": [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts],
        "parity": report.to_dict(),
        "compatibility": {
            "layout": "HF_STYLE_SAFETENSORS_DIRECTORY",
            "runtime_logit_token_decode_parity": "PASS" if report.passed else "FAIL",
            "tolerance": "EXACT_ZERO",
            "transformers_architecture": "NOT_CLAIMED",
        },
        "promotion_authority": False,
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
        failures = evidence["parity"]["failures"]
        raise RuntimeError(f"HF-style export runtime parity failed: {failures}")
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m twelve_six.inference.hf_export_parity",
        description=(
            "Verify a D05 HF-style export and compare it with canonical first-party "
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

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(evidence, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(evidence, ensure_ascii=False, sort_keys=True, allow_nan=False))
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
