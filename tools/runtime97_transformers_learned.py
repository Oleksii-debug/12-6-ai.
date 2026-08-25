"""Artifact-bound learned-checkpoint parity for the incumbent Transformers/Llama bridge.

This is an evidence harness, not another model adapter. It consumes the existing
verified HF-style export and standard-Llama materialization seams, then executes
maintained Transformers LlamaForCausalLM from the resulting local directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import LlamaForCausalLM

from twelve_six.checkpoint import export_hf_directory
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.inference.llama_runtime_export import (
    materialize_standard_llama_directory,
    verify_standard_llama_directory,
)
from twelve_six.inference.transformers_llama import llama_config_dict
from twelve_six.inference.transformers_llama_runtime import (
    LOGIT_ATOL,
    LOGIT_RTOL,
    TransformersLlamaRuntime,
)

SCHEMA = "12-6.runtime97-transformers-learned-parity.v1"
EXPECTED_TRANSFORMERS_VERSION = "5.15.1"
EXPECTED_PARAMETER_COUNT = 992_896
EXPECTED_MODEL_SPEC_SHA256 = "18284b303eb31cef5191ddb3ed4ddba5ce51789aadf4b14cc90d4226c5c527b5"
EXPECTED_LEARNED_SOURCE_SHA = "003e268655b672df9df00afb8a32dbec4db5d2e1"
EXPECTED_TRAINING_EVIDENCE_SCHEMA = "12-6.s2-1m-executable-preflight.v1"
EXPECTED_ARTIFACT_ID = 9_558_539_580
EXPECTED_ARTIFACT_DIGEST = (
    "sha256:aa57a7ebd96929fa8fd7eb28b7232de84f00c1062e9550b1237bd931eb60068a"
)
EXPECTED_D08_SOURCE_SHA = "10da7d4c1d2fe5a229659143877ad82d3739bed9"
EXPECTED_D08_PROFILE_SHA256 = "d54843c76befcd0e8a1703f8d34c822fdbf3085351d1d357661b91265148a039"
EXPECTED_D08_OVERLAY_SHA256 = "68f73eaac9e3a9418e85e54a3170d1378c51af0161138d0f4f8667c595deb0b0"

PROMPTS: tuple[tuple[str, str], ...] = (
    ("uk", "Україна розвиває власні мовні моделі."),
    ("en", "A small learned language model should preserve runtime parity."),
    ("code", "def add(a, b):\n    return a + b\n"),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def _canonical_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(payload)


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
    return _sha256(value.numpy().tobytes(order="C"))


def _logit_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.detach().float().cpu()
    candidate = candidate.detach().float().cpu()
    absolute = (reference - candidate).abs()
    denominator = reference.abs().clamp_min(torch.finfo(reference.dtype).tiny)
    return {
        "reference_sha256": _tensor_sha256(reference),
        "candidate_sha256": _tensor_sha256(candidate),
        "max_abs_error": float(absolute.max().item()),
        "max_rel_error": float((absolute / denominator).max().item()),
        "allclose": bool(
            torch.allclose(reference, candidate, atol=LOGIT_ATOL, rtol=LOGIT_RTOL)
        ),
    }


def _reference_logits(reference: Any, ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(reference.next_token_logits(ids), dtype=torch.float32)


def _load_training_evidence(path: Path, checkpoint_id: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("SCALE-02 training evidence must be a JSON object")
    if value.get("schema") != EXPECTED_TRAINING_EVIDENCE_SCHEMA:
        raise ValueError("unexpected SCALE-02 training evidence schema")
    if value.get("candidate_sha") != EXPECTED_LEARNED_SOURCE_SHA:
        raise ValueError("learned evidence source SHA drifted")
    model = value.get("model")
    training = value.get("training")
    checkpoint = value.get("checkpoint")
    claims = value.get("claims")
    if not all(isinstance(item, dict) for item in (model, training, checkpoint, claims)):
        raise TypeError("learned evidence is missing required mappings")
    if model.get("parameter_count") != EXPECTED_PARAMETER_COUNT:
        raise ValueError("learned evidence parameter count drifted")
    if model.get("model_spec_sha256") != EXPECTED_MODEL_SPEC_SHA256:
        raise ValueError("learned evidence ModelSpec drifted")
    if training.get("optimizer_steps") != 4 or training.get("optimized_tokens") != 490:
        raise ValueError("learned evidence training trajectory drifted")
    delta = training.get("weight_delta")
    if not isinstance(delta, dict) or int(delta.get("changed_parameter_elements", 0)) <= 0:
        raise ValueError("learned evidence does not prove updated weights")
    if checkpoint.get("checkpoint_id") != checkpoint_id:
        raise ValueError("training evidence checkpoint_id does not match consumed checkpoint")
    if checkpoint.get("step") != 2 or checkpoint.get("tokens_seen") != 254:
        raise ValueError("retained learned checkpoint step/tokens drifted")
    if claims.get("foreign_pretrained_weights_used") is not False:
        raise ValueError("learned source does not preserve foreign-weight boundary")
    if claims.get("paid_compute_used") is not False:
        raise ValueError("learned source does not preserve LOCAL_FREE boundary")
    return value


def _prompt_evidence(
    reference: Any,
    candidate: TransformersLlamaRuntime,
    *,
    category: str,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    ids = candidate.encode(prompt)
    if ids != list(prompt.encode("utf-8")) or candidate.decode(ids) != prompt:
        raise RuntimeError(f"canonical byte-tokenizer mapping failed for {category}")

    reference_ids = list(ids)
    candidate_ids = list(ids)
    steps: list[dict[str, Any]] = []
    for index in range(min(max_new_tokens, candidate.max_context_tokens - len(ids))):
        reference_logits = _reference_logits(reference, reference_ids)
        candidate_logits = candidate.next_token_logits_tensor(candidate_ids)
        metrics = _logit_metrics(reference_logits, candidate_logits)
        reference_token = int(torch.argmax(reference_logits).item())
        candidate_token = int(torch.argmax(candidate_logits).item())
        metrics.update(
            {
                "index": index,
                "reference_argmax": reference_token,
                "candidate_argmax": candidate_token,
                "greedy_token_exact": reference_token == candidate_token,
            }
        )
        steps.append(metrics)
        reference_ids.append(reference_token)
        candidate_ids.append(candidate_token)

    continuation_ref = reference_ids[len(ids) :]
    continuation_candidate = candidate_ids[len(ids) :]
    decoded_ref = reference.decode(continuation_ref)
    decoded_candidate = candidate.decode(continuation_candidate)
    return {
        "category": category,
        "prompt": prompt,
        "prompt_sha256": _sha256(prompt.encode("utf-8")),
        "input_token_ids": ids,
        "input_token_count": len(ids),
        "reference_continuation_ids": continuation_ref,
        "candidate_continuation_ids": continuation_candidate,
        "greedy_tokens_exact": continuation_ref == continuation_candidate,
        "reference_decoded_continuation": decoded_ref,
        "candidate_decoded_continuation": decoded_candidate,
        "decoded_continuation_exact": decoded_ref == decoded_candidate,
        "steps": steps,
    }


def collect_evidence(
    *,
    repo_root: Path,
    checkpoint_dir: Path,
    training_evidence_path: Path,
    output_dir: Path,
    bridge_source_sha: str,
    source_artifact_id: int,
    source_artifact_digest: str,
    d08_source_sha: str,
    d08_profile_sha256: str,
    d08_overlay_sha256: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    if _git_head(repo_root) != bridge_source_sha:
        raise ValueError("bridge_source_sha does not match checkout HEAD")
    if source_artifact_id != EXPECTED_ARTIFACT_ID:
        raise ValueError("unexpected learned source artifact id")
    if source_artifact_digest != EXPECTED_ARTIFACT_DIGEST:
        raise ValueError("unexpected learned source artifact digest")
    if d08_source_sha != EXPECTED_D08_SOURCE_SHA:
        raise ValueError("unexpected D08 purpose-runtime source SHA")
    if d08_profile_sha256 != EXPECTED_D08_PROFILE_SHA256:
        raise ValueError("unexpected D08 Transformers profile semantic hash")
    if d08_overlay_sha256 != EXPECTED_D08_OVERLAY_SHA256:
        raise ValueError("unexpected D08 Transformers overlay lock hash")
    if importlib.metadata.version("transformers") != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError("RUNTIME-97 is not executing the exact D08 Transformers version")

    output_dir.mkdir(parents=True, exist_ok=False)
    reference = load_first_party_backend(checkpoint_dir)
    diagnostics = reference.diagnostics()
    if diagnostics["parameter_count"] != EXPECTED_PARAMETER_COUNT:
        raise ValueError("consumed checkpoint is not the learned ~1M incumbent")
    if diagnostics["model_spec_sha256"] != EXPECTED_MODEL_SPEC_SHA256:
        raise ValueError("consumed learned checkpoint ModelSpec drifted")
    if diagnostics["git_sha"] != EXPECTED_LEARNED_SOURCE_SHA:
        raise ValueError("consumed learned checkpoint source SHA drifted")
    if int(diagnostics["step"]) <= 0 or int(diagnostics["tokens_seen"]) <= 0:
        raise ValueError("consumed checkpoint is not learned")

    training_evidence = _load_training_evidence(
        training_evidence_path, diagnostics["checkpoint_id"]
    )

    canonical_export = export_hf_directory(
        checkpoint_dir,
        output_dir / "canonical-hf-export",
        hf_config=llama_config_dict(reference.model.spec),
    )
    runtime_dir = materialize_standard_llama_directory(
        canonical_export,
        output_dir / "llama-runtime",
    )
    runtime_provenance = verify_standard_llama_directory(runtime_dir)
    if runtime_provenance.get("rope_transform") != (
        "PAIRWISE_INTERLEAVED_TO_LLAMA_HALF_SPLIT"
    ):
        raise RuntimeError("standard Llama export did not apply the incumbent Q/K RoPE basis conversion")
    if runtime_provenance.get("source_checkpoint_id") != diagnostics["checkpoint_id"]:
        raise RuntimeError("standard Llama export is not bound to consumed learned checkpoint")
    if runtime_provenance.get("parameter_count") != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("standard Llama export parameter count drifted")

    target = LlamaForCausalLM.from_pretrained(
        runtime_dir,
        local_files_only=True,
    ).eval()
    candidate = TransformersLlamaRuntime(
        model=target,
        tokenizer=reference.tokenizer,
        spec=reference.model.spec,
        checkpoint_id=diagnostics["checkpoint_id"],
        source_manifest_sha256=runtime_provenance["source_manifest_sha256"],
        weights_sha256=runtime_provenance["runtime_weights_sha256"],
        config_sha256=runtime_provenance["runtime_config_sha256"],
        plan_sha256="standard-directory-consumes-incumbent-conversion",
    )

    prompt_records = [
        _prompt_evidence(
            reference,
            candidate,
            category=category,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        for category, prompt in PROMPTS
    ]

    boundary_ids = [
        (index * 17 + 3) % candidate.spec.vocab_size
        for index in range(candidate.max_context_tokens)
    ]
    boundary_reference = _reference_logits(reference, boundary_ids)
    boundary_candidate = candidate.next_token_logits_tensor(boundary_ids)
    boundary_metrics = _logit_metrics(boundary_reference, boundary_candidate)

    over_context = boundary_ids + [0]
    reference_rejected = candidate_rejected = False
    try:
        reference.next_token_logits(over_context)
    except ValueError:
        reference_rejected = True
    try:
        candidate.next_token_logits(over_context)
    except ValueError:
        candidate_rejected = True

    all_steps = [step for prompt in prompt_records for step in prompt["steps"]]
    max_abs = max(
        [float(step["max_abs_error"]) for step in all_steps]
        + [float(boundary_metrics["max_abs_error"])]
    )
    max_rel = max(
        [float(step["max_rel_error"]) for step in all_steps]
        + [float(boundary_metrics["max_rel_error"])]
    )
    failures: list[str] = []
    if not all(bool(step["allclose"]) for step in all_steps):
        failures.append("prompt_logit_parity")
    if not all(bool(step["greedy_token_exact"]) for step in all_steps):
        failures.append("greedy_token_parity")
    if not all(bool(prompt["greedy_tokens_exact"]) for prompt in prompt_records):
        failures.append("greedy_sequence_parity")
    if not all(bool(prompt["decoded_continuation_exact"]) for prompt in prompt_records):
        failures.append("decoded_continuation_parity")
    if not boundary_metrics["allclose"]:
        failures.append("context_boundary_logit_parity")
    if not reference_rejected or not candidate_rejected:
        failures.append("context_overflow_rejection")

    evidence: dict[str, Any] = {
        "schema": SCHEMA,
        "passed": not failures,
        "bridge": {
            "source_sha": bridge_source_sha,
            "conversion_seam": "twelve_six.inference.transformers_llama",
            "standard_directory_seam": "twelve_six.inference.llama_runtime_export",
            "new_transformers_adapter_added": False,
        },
        "learned_source": {
            "artifact_id": source_artifact_id,
            "artifact_digest": source_artifact_digest,
            "artifact_name": "scale02-s2-1m-executable-evidence",
            "workflow_run_id": 32_835_548_366,
            "source_git_sha": diagnostics["git_sha"],
            "checkpoint_id": diagnostics["checkpoint_id"],
            "model_spec_sha256": diagnostics["model_spec_sha256"],
            "parameter_count": diagnostics["parameter_count"],
            "checkpoint_step": diagnostics["step"],
            "checkpoint_tokens_seen": diagnostics["tokens_seen"],
            "training_evidence_sha256": _sha256_file(training_evidence_path),
            "training_evidence_semantic_sha256": training_evidence["evidence_sha256"],
        },
        "d08_runtime": {
            "profile_id": "linux-x86_64-transformers-interop",
            "source_sha": d08_source_sha,
            "profile_sha256": d08_profile_sha256,
            "overlay_lock_sha256": d08_overlay_sha256,
            "python": importlib.metadata.version("pip") and __import__("platform").python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "device": "cpu",
            "dtype": "float32",
        },
        "execution": {
            "architecture": target.__class__.__name__,
            "class_module": target.__class__.__module__,
            "load_method": "LlamaForCausalLM.from_pretrained(local_directory, local_files_only=True)",
            "canonical_export_checkpoint_id": diagnostics["checkpoint_id"],
            "runtime_weights_sha256": runtime_provenance["runtime_weights_sha256"],
            "runtime_config_sha256": runtime_provenance["runtime_config_sha256"],
            "qk_rope_basis_conversion": runtime_provenance["rope_transform"],
            "foreign_pretrained_weights": runtime_provenance["foreign_pretrained_weights"],
            "model_downloaded": runtime_provenance["model_downloaded"],
        },
        "acceptance": {
            "logit_atol": LOGIT_ATOL,
            "logit_rtol": LOGIT_RTOL,
            "tolerance_changed_for_runtime97": False,
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "logit_steps_compared": len(all_steps) + 1,
        },
        "prompts": prompt_records,
        "context": {
            "max_context_tokens": candidate.max_context_tokens,
            "boundary_input_sha256": _sha256(bytes(boundary_ids)),
            "boundary_logits": boundary_metrics,
            "over_context_reference_rejected": reference_rejected,
            "over_context_transformers_bridge_rejected": candidate_rejected,
        },
        "failures": failures,
        "truth_boundary": {
            "pretrained_foreign_weights_used": False,
            "hub_model_download_used": False,
            "instruction_or_alignment_claim": False,
            "quality_or_capability_claim": False,
            "paid_compute_used": False,
            "production_stage_promotion": False,
        },
    }
    evidence["evidence_sha256"] = _canonical_hash(evidence)
    evidence_path = output_dir / "runtime97-transformers-learned-parity.json"
    evidence_path.write_text(
        json.dumps(evidence, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(f"learned Transformers parity failed: {failures}")
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--training-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bridge-source-sha", required=True)
    parser.add_argument("--source-artifact-id", type=int, required=True)
    parser.add_argument("--source-artifact-digest", required=True)
    parser.add_argument("--d08-source-sha", required=True)
    parser.add_argument("--d08-profile-sha256", required=True)
    parser.add_argument("--d08-overlay-sha256", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence = collect_evidence(
        repo_root=args.repo_root.resolve(),
        checkpoint_dir=args.checkpoint_dir.resolve(),
        training_evidence_path=args.training_evidence.resolve(),
        output_dir=args.output_dir.resolve(),
        bridge_source_sha=args.bridge_source_sha,
        source_artifact_id=args.source_artifact_id,
        source_artifact_digest=args.source_artifact_digest,
        d08_source_sha=args.d08_source_sha,
        d08_profile_sha256=args.d08_profile_sha256,
        d08_overlay_sha256=args.d08_overlay_sha256,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
