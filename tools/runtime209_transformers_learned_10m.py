"""Exact learned-10M evidence harness for the incumbent 12-6 -> Llama bridge.

This module deliberately does not add an exporter or model adapter. It consumes
RUNTIME-97/#135's verified canonical HF export, standard-Llama materialization,
and Transformers runtime seams, binds them to a retained SCALE-141 learned 10M
checkpoint, and records fail-closed local-only parity evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import LlamaForCausalLM

from twelve_six.checkpoint import export_hf_directory
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.inference.llama_runtime_export import (
    materialize_standard_llama_directory,
    verify_standard_llama_directory,
)
from twelve_six.inference.transformers_llama import (
    build_llama_interop_plan,
    llama_config_dict,
    rope_pairwise_to_llama_permutation,
)
from twelve_six.inference.transformers_llama_runtime import (
    LOGIT_ATOL,
    LOGIT_RTOL,
    TransformersLlamaRuntime,
)

SCHEMA = "12-6.runtime209-transformers-learned-10m-parity.v1"
PREPARED_SCHEMA = "12-6.runtime209-transformers-learned-10m-prepared.v1"
EXPECTED_TRANSFORMERS_VERSION = "5.15.1"
EXPECTED_PARAMETER_COUNT = 10_000_640
EXPECTED_MODEL_SPEC_SHA256 = "61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998"
EXPECTED_TOKENIZER_VERSION = "s0-byte-v1"
EXPECTED_TOKENIZER_CONFIG_SHA256 = "b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1"
EXPECTED_TOKENIZER_VOCAB_SHA256 = "905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571"
EXPECTED_CORPUS_SHA256 = "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
EXPECTED_FRESH_SCHEMA = "12-6.scale141-fresh-verification.v2"
EXPECTED_RETAINED_SCHEMA = "12-6.scale141-retained-checkpoints.v1"
EXPECTED_ARTIFACT_NAME = "scale141-10m-learned-fallback"
EXPECTED_D08_SOURCE_SHA = "10da7d4c1d2fe5a229659143877ad82d3739bed9"
EXPECTED_D08_PROFILE_SHA256 = "d54843c76befcd0e8a1703f8d34c822fdbf3085351d1d357661b91265148a039"
EXPECTED_D08_OVERLAY_SHA256 = "68f73eaac9e3a9418e85e54a3170d1378c51af0161138d0f4f8667c595deb0b0"
EXPECTED_BOOTSTRAP_SOURCE_SHA = "c127216ecf9722bea1964c7488cb7ff0f8cdebe4"
EXPECTED_BOOTSTRAP_MANIFEST_SCHEMA = "12-6.execution-environment-manifest.v1"
EXPECTED_D08_CANONICAL_MANIFEST_SHA256 = "283ca83571e527babda700e0c66ed03fb1c2aa4674bee0dba2272f64f344e1bf"
EXPECTED_D08_CANONICAL_INDEX_SHA256 = "5de40d40012123ccf654b3e29d9cd47df814978e4155ca9dde232b61e9cd6341"
EXPECTED_CAPABILITIES = {"runtime", "tests", "lint", "transformers"}
CHECKPOINT_ROLE = "best"

PROMPTS: tuple[tuple[str, str], ...] = (
    ("uk", "Україна розвиває власні мовні моделі."),
    ("en", "A learned ten million parameter model should preserve exact runtime semantics."),
    ("code", "def add(a, b):\n    return a + b\n"),
)


class Runtime209Error(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def _canonical_hash(value: Mapping[str, Any]) -> str:
    data = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(data)


def _json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Runtime209Error(f"{label} is not readable UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Runtime209Error(f"{label} must be a JSON object")
    return value


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return _sha256(value.numpy().tobytes(order="C"))


def _logit_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.detach().to(device="cpu", dtype=torch.float32).contiguous()
    candidate = candidate.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if reference.shape != candidate.shape:
        return {
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "shape_exact": False,
            "allclose": False,
        }
    absolute = (reference - candidate).abs()
    denominator = reference.abs().clamp_min(torch.finfo(reference.dtype).tiny)
    return {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "shape_exact": True,
        "reference_sha256": _tensor_sha256(reference),
        "candidate_sha256": _tensor_sha256(candidate),
        "max_abs_error": float(absolute.max().item()),
        "max_rel_error": float((absolute / denominator).max().item()),
        "allclose": bool(
            torch.allclose(reference, candidate, atol=LOGIT_ATOL, rtol=LOGIT_RTOL)
        ),
    }


def _lower_packages(packages: Mapping[str, Any]) -> dict[str, str]:
    return {str(name).lower(): str(version) for name, version in packages.items()}


def _validate_bootstrap_manifest(path: Path, source_sha: str) -> dict[str, Any]:
    if source_sha != EXPECTED_BOOTSTRAP_SOURCE_SHA:
        raise Runtime209Error("universal bootstrap source SHA drifted")
    manifest = _json(path, label="universal bootstrap manifest")
    if manifest.get("schema") != EXPECTED_BOOTSTRAP_MANIFEST_SCHEMA:
        raise Runtime209Error("universal bootstrap manifest schema mismatch")
    plan = manifest.get("plan")
    if not isinstance(plan, dict):
        raise Runtime209Error("universal bootstrap manifest is missing execution plan")
    if set(plan.get("capabilities", [])) != EXPECTED_CAPABILITIES:
        raise Runtime209Error("universal bootstrap capability set mismatch")
    if plan.get("cuda_packages_present") is not False:
        raise Runtime209Error("CPU Transformers purpose runtime inherited CUDA packages")
    authority = plan.get("d08_authority")
    if not isinstance(authority, dict):
        raise Runtime209Error("universal bootstrap plan is missing D08 authority")
    if authority.get("canonical_manifest_sha256") != EXPECTED_D08_CANONICAL_MANIFEST_SHA256:
        raise Runtime209Error("universal bootstrap D08 manifest authority drifted")
    if authority.get("canonical_index_sha256") != EXPECTED_D08_CANONICAL_INDEX_SHA256:
        raise Runtime209Error("universal bootstrap D08 index authority drifted")
    overlays = [
        row for row in plan.get("locks", [])
        if isinstance(row, dict) and row.get("role") == "transformers_overlay"
    ]
    if len(overlays) != 1 or overlays[0].get("sha256") != EXPECTED_D08_OVERLAY_SHA256:
        raise Runtime209Error("universal bootstrap did not consume exact Transformers overlay")
    preflight = manifest.get("preflight")
    if not isinstance(preflight, dict) or preflight.get("status") != "PASS":
        raise Runtime209Error("universal bootstrap preflight did not pass")
    cuda = preflight.get("cuda")
    if not isinstance(cuda, dict) or cuda.get("hardware_claim") is not False:
        raise Runtime209Error("CPU purpose runtime made a CUDA hardware claim")
    packages = manifest.get("packages")
    if not isinstance(packages, dict):
        raise Runtime209Error("universal bootstrap package inventory is missing")
    lowered = _lower_packages(packages)
    if lowered.get("transformers") != EXPECTED_TRANSFORMERS_VERSION:
        raise Runtime209Error("universal bootstrap Transformers version drifted")
    if not lowered.get("torch", "").startswith("2.13.0"):
        raise Runtime209Error("universal bootstrap torch version drifted")
    return manifest


def _validate_artifact_binding(
    *,
    artifact_metadata_path: Path,
    run_metadata_path: Path,
    source_artifact_id: int,
    source_artifact_digest: str,
    source_run_id: int,
    source_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = _json(artifact_metadata_path, label="SCALE-141 artifact metadata")
    run = _json(run_metadata_path, label="SCALE-141 workflow-run metadata")
    if artifact.get("id") != source_artifact_id:
        raise Runtime209Error("SCALE-141 artifact id mismatch")
    if artifact.get("name") != EXPECTED_ARTIFACT_NAME:
        raise Runtime209Error("SCALE-141 artifact name mismatch")
    if artifact.get("digest") != source_artifact_digest:
        raise Runtime209Error("SCALE-141 artifact digest mismatch")
    if artifact.get("expired") is not False:
        raise Runtime209Error("SCALE-141 artifact is expired")
    workflow_run = artifact.get("workflow_run")
    if not isinstance(workflow_run, dict):
        raise Runtime209Error("SCALE-141 artifact is missing workflow-run provenance")
    if workflow_run.get("id") != source_run_id:
        raise Runtime209Error("SCALE-141 artifact workflow-run id mismatch")
    if workflow_run.get("head_sha") != source_sha:
        raise Runtime209Error("SCALE-141 artifact workflow-run source SHA mismatch")
    if run.get("id") != source_run_id or run.get("head_sha") != source_sha:
        raise Runtime209Error("SCALE-141 workflow-run identity mismatch")
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise Runtime209Error("SCALE-141 source run is not terminal SUCCESS")
    return artifact, run


def _validate_retained_source(
    *,
    checkpoint_dir: Path,
    fresh_path: Path,
    retained_index_path: Path,
    source_sha: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fresh = _json(fresh_path, label="SCALE-141 fresh verification")
    retained = _json(retained_index_path, label="SCALE-141 retained index")
    if fresh.get("schema") != EXPECTED_FRESH_SCHEMA:
        raise Runtime209Error("SCALE-141 fresh-verification schema mismatch")
    verification = fresh.get("fresh_verification")
    if not isinstance(verification, dict) or verification.get("status") != "PASS":
        raise Runtime209Error("SCALE-141 fresh verification is not PASS")
    required_fresh = (
        "checkpoint_load",
        "first_party_logits",
        "evaluation_non_mutation",
        "checkpoint_identity",
        "generation",
        "reproducibility_manifest_validation",
        "best_and_final_retained",
    )
    if any(verification.get(field) is not True for field in required_fresh):
        raise Runtime209Error("SCALE-141 fresh-verification gates are incomplete")
    if fresh.get("source_sha") != source_sha:
        raise Runtime209Error("SCALE-141 fresh-verification source SHA mismatch")
    if fresh.get("corpus_identity_sha256") != EXPECTED_CORPUS_SHA256:
        raise Runtime209Error("SCALE-141 fresh-verification corpus identity mismatch")
    if retained.get("schema") != EXPECTED_RETAINED_SCHEMA or retained.get("source_sha") != source_sha:
        raise Runtime209Error("SCALE-141 retained-checkpoint index identity mismatch")
    roles = retained.get("roles")
    if not isinstance(roles, dict) or not isinstance(roles.get(CHECKPOINT_ROLE), dict):
        raise Runtime209Error("SCALE-141 retained best checkpoint is missing")
    retained_best = roles[CHECKPOINT_ROLE]
    if retained_best.get("fresh_verification") != "PASS":
        raise Runtime209Error("retained best checkpoint lacks fresh-verification PASS")
    evidence = fresh.get("evidence")
    if not isinstance(evidence, dict) or not isinstance(evidence.get(CHECKPOINT_ROLE), dict):
        raise Runtime209Error("SCALE-141 best-checkpoint fresh evidence is missing")
    fresh_best = evidence[CHECKPOINT_ROLE]
    if fresh_best.get("checkpoint_id") != retained_best.get("checkpoint_id"):
        raise Runtime209Error("retained best checkpoint id differs from fresh verification")
    if fresh_best.get("target_optimized_tokens") != retained_best.get("target_optimized_tokens"):
        raise Runtime209Error("retained best optimized-token target differs from fresh verification")
    if not checkpoint_dir.is_dir():
        raise Runtime209Error("retained best checkpoint directory is missing")
    identity = fresh_best.get("checkpoint_identity")
    if not isinstance(identity, dict):
        raise Runtime209Error("SCALE-141 best checkpoint identity is missing")
    if identity.get("git_sha") != source_sha:
        raise Runtime209Error("learned 10M checkpoint source SHA mismatch")
    if identity.get("model_spec_hash") != EXPECTED_MODEL_SPEC_SHA256:
        raise Runtime209Error("learned 10M checkpoint ModelSpec mismatch")
    if identity.get("parameter_count") != EXPECTED_PARAMETER_COUNT:
        raise Runtime209Error("learned 10M checkpoint parameter count mismatch")
    if identity.get("tokenizer_hash") != EXPECTED_TOKENIZER_CONFIG_SHA256:
        raise Runtime209Error("learned 10M checkpoint tokenizer config identity mismatch")
    if identity.get("tokenizer_vocab_hash") != EXPECTED_TOKENIZER_VOCAB_SHA256:
        raise Runtime209Error("learned 10M checkpoint tokenizer vocab identity mismatch")
    if identity.get("dataset_manifest_hash") != EXPECTED_CORPUS_SHA256:
        raise Runtime209Error("learned 10M checkpoint corpus identity mismatch")
    if int(identity.get("step", 0)) <= 0 or int(identity.get("tokens_seen", 0)) <= 0:
        raise Runtime209Error("consumed 10M checkpoint is not learned")
    if int(identity["tokens_seen"]) != int(fresh_best["target_optimized_tokens"]):
        raise Runtime209Error("checkpoint optimized-token identity mismatch")
    return fresh, retained, fresh_best


def _full_logits(reference: Any, target: Any, ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = torch.tensor([list(ids)], dtype=torch.long)
    with torch.inference_mode():
        reference_logits = reference.model(tensor).logits[0].detach().float().cpu()
        candidate_logits = target(input_ids=tensor, use_cache=False).logits[0].detach().float().cpu()
    return reference_logits, candidate_logits


def _prompt_evidence(
    reference: Any,
    candidate: TransformersLlamaRuntime,
    target: Any,
    *,
    category: str,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    ids = candidate.encode(prompt)
    if ids != list(prompt.encode("utf-8")) or candidate.decode(ids) != prompt:
        raise Runtime209Error(f"canonical byte tokenizer failed for {category}")
    full_reference, full_candidate = _full_logits(reference, target, ids)
    full_metrics = _logit_metrics(full_reference, full_candidate)

    reference_ids = list(ids)
    candidate_ids = list(ids)
    steps: list[dict[str, Any]] = []
    for index in range(min(max_new_tokens, candidate.max_context_tokens - len(ids))):
        ref_logits = torch.tensor(reference.next_token_logits(reference_ids), dtype=torch.float32)
        cand_logits = candidate.next_token_logits_tensor(candidate_ids)
        metrics = _logit_metrics(ref_logits, cand_logits)
        reference_token = int(torch.argmax(ref_logits).item())
        candidate_token = int(torch.argmax(cand_logits).item())
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

    reference_continuation = reference_ids[len(ids):]
    candidate_continuation = candidate_ids[len(ids):]
    reference_text = reference.decode(reference_continuation)
    candidate_text = candidate.decode(candidate_continuation)
    return {
        "category": category,
        "prompt": prompt,
        "prompt_sha256": _sha256(prompt.encode("utf-8")),
        "input_token_ids": ids,
        "input_token_count": len(ids),
        "full_prompt_logits": full_metrics,
        "reference_continuation_ids": reference_continuation,
        "candidate_continuation_ids": candidate_continuation,
        "greedy_tokens_exact": reference_continuation == candidate_continuation,
        "reference_decoded_continuation": reference_text,
        "candidate_decoded_continuation": candidate_text,
        "decoded_continuation_exact": reference_text == candidate_text,
        "generation_steps": steps,
    }


def _qk_learned_weight_evidence(reference: Any, target: Any) -> dict[str, Any]:
    spec = reference.model.spec
    source = reference.model.state_dict()
    target_state = target.state_dict()
    q_permutation = torch.tensor(
        rope_pairwise_to_llama_permutation(heads=spec.n_heads, head_dim=spec.head_dim),
        dtype=torch.long,
    )
    k_permutation = torch.tensor(
        rope_pairwise_to_llama_permutation(heads=spec.n_kv_heads, head_dim=spec.head_dim),
        dtype=torch.long,
    )
    layers: list[dict[str, Any]] = []
    for layer in range(spec.n_layers):
        source_q = source[f"blocks.{layer}.attn.q_proj.weight"].detach().cpu()
        source_k = source[f"blocks.{layer}.attn.k_proj.weight"].detach().cpu()
        expected_q = source_q.index_select(0, q_permutation)
        expected_k = source_k.index_select(0, k_permutation)
        actual_q = target_state[f"model.layers.{layer}.self_attn.q_proj.weight"].detach().cpu()
        actual_k = target_state[f"model.layers.{layer}.self_attn.k_proj.weight"].detach().cpu()
        q_exact = bool(torch.equal(expected_q, actual_q))
        k_exact = bool(torch.equal(expected_k, actual_k))
        layers.append(
            {
                "layer": layer,
                "q_exact": q_exact,
                "k_exact": k_exact,
                "source_q_sha256": _tensor_sha256(source_q),
                "source_k_sha256": _tensor_sha256(source_k),
                "expected_llama_q_sha256": _tensor_sha256(expected_q),
                "expected_llama_k_sha256": _tensor_sha256(expected_k),
                "actual_llama_q_sha256": _tensor_sha256(actual_q),
                "actual_llama_k_sha256": _tensor_sha256(actual_k),
            }
        )
    return {
        "transform": "PAIRWISE_INTERLEAVED_TO_LLAMA_HALF_SPLIT",
        "q_permutation_rows": int(q_permutation.numel()),
        "k_permutation_rows": int(k_permutation.numel()),
        "layers": layers,
        "all_q_exact": all(row["q_exact"] for row in layers),
        "all_k_exact": all(row["k_exact"] for row in layers),
    }


def collect_evidence(
    *,
    repo_root: Path,
    checkpoint_dir: Path,
    fresh_verification_path: Path,
    retained_index_path: Path,
    artifact_metadata_path: Path,
    run_metadata_path: Path,
    bootstrap_manifest_path: Path,
    output_dir: Path,
    bridge_source_sha: str,
    source_artifact_id: int,
    source_artifact_digest: str,
    source_run_id: int,
    source_sha: str,
    d08_source_sha: str,
    d08_profile_sha256: str,
    d08_overlay_sha256: str,
    bootstrap_source_sha: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    if _git_head(repo_root) != bridge_source_sha:
        raise Runtime209Error("bridge source SHA does not match checkout HEAD")
    if d08_source_sha != EXPECTED_D08_SOURCE_SHA:
        raise Runtime209Error("D08 purpose-runtime source SHA drifted")
    if d08_profile_sha256 != EXPECTED_D08_PROFILE_SHA256:
        raise Runtime209Error("D08 Transformers purpose profile hash drifted")
    if d08_overlay_sha256 != EXPECTED_D08_OVERLAY_SHA256:
        raise Runtime209Error("D08 Transformers overlay lock hash drifted")
    if importlib.metadata.version("transformers") != EXPECTED_TRANSFORMERS_VERSION:
        raise Runtime209Error("RUNTIME-209 is not executing exact Transformers 5.15.1")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise Runtime209Error("Transformers execution must run with hub access disabled")

    bootstrap_manifest = _validate_bootstrap_manifest(
        bootstrap_manifest_path, bootstrap_source_sha
    )
    artifact_metadata, run_metadata = _validate_artifact_binding(
        artifact_metadata_path=artifact_metadata_path,
        run_metadata_path=run_metadata_path,
        source_artifact_id=source_artifact_id,
        source_artifact_digest=source_artifact_digest,
        source_run_id=source_run_id,
        source_sha=source_sha,
    )
    fresh, retained, fresh_best = _validate_retained_source(
        checkpoint_dir=checkpoint_dir,
        fresh_path=fresh_verification_path,
        retained_index_path=retained_index_path,
        source_sha=source_sha,
    )

    reference = load_first_party_backend(checkpoint_dir)
    diagnostics = reference.diagnostics()
    required_diagnostics = {
        "git_sha": source_sha,
        "model_spec_sha256": EXPECTED_MODEL_SPEC_SHA256,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "tokenizer_version": EXPECTED_TOKENIZER_VERSION,
        "tokenizer_config_sha256": EXPECTED_TOKENIZER_CONFIG_SHA256,
        "tokenizer_vocab_sha256": EXPECTED_TOKENIZER_VOCAB_SHA256,
        "dataset_manifest_sha256": EXPECTED_CORPUS_SHA256,
    }
    for field, expected in required_diagnostics.items():
        if diagnostics.get(field) != expected:
            raise Runtime209Error(f"first-party checkpoint diagnostic mismatch: {field}")
    if diagnostics.get("checkpoint_id") != fresh_best.get("checkpoint_id"):
        raise Runtime209Error("first-party checkpoint id differs from SCALE-141 fresh verification")
    if diagnostics.get("run_manifest_sha256") != fresh.get("run_manifest_identity_sha256"):
        raise Runtime209Error("first-party run-manifest identity differs from SCALE-141 verification")
    if int(diagnostics.get("step", 0)) <= 0 or int(diagnostics.get("tokens_seen", 0)) <= 0:
        raise Runtime209Error("first-party source checkpoint is not learned")

    plan = build_llama_interop_plan(reference.model.spec)
    if plan.source_model_spec_sha256 != EXPECTED_MODEL_SPEC_SHA256:
        raise Runtime209Error("incumbent Llama plan did not bind exact learned 10M ModelSpec")
    if plan.source_parameter_count != EXPECTED_PARAMETER_COUNT:
        raise Runtime209Error("incumbent Llama plan parameter count drifted")
    if plan.target_architecture != "LlamaForCausalLM":
        raise Runtime209Error("incumbent Llama plan target architecture drifted")
    if plan.rope_transform != "PAIRWISE_INTERLEAVED_TO_LLAMA_HALF_SPLIT":
        raise Runtime209Error("incumbent Llama plan RoPE basis conversion drifted")

    output_dir.mkdir(parents=True, exist_ok=False)
    canonical_export = export_hf_directory(
        checkpoint_dir,
        output_dir / "canonical-hf-export",
        hf_config=llama_config_dict(reference.model.spec),
    )
    runtime_dir = materialize_standard_llama_directory(
        canonical_export,
        output_dir / "standard-llama",
    )
    runtime_provenance = verify_standard_llama_directory(runtime_dir)
    if runtime_provenance.get("source_checkpoint_id") != diagnostics["checkpoint_id"]:
        raise Runtime209Error("standard-Llama directory is not bound to learned checkpoint")
    if runtime_provenance.get("model_spec_sha256") != EXPECTED_MODEL_SPEC_SHA256:
        raise Runtime209Error("standard-Llama directory ModelSpec drifted")
    if runtime_provenance.get("parameter_count") != EXPECTED_PARAMETER_COUNT:
        raise Runtime209Error("standard-Llama directory parameter count drifted")
    if runtime_provenance.get("rope_transform") != plan.rope_transform:
        raise Runtime209Error("standard-Llama directory RoPE transform drifted")
    if runtime_provenance.get("foreign_pretrained_weights") is not False:
        raise Runtime209Error("standard-Llama directory introduced foreign weights")
    if runtime_provenance.get("model_downloaded") is not False:
        raise Runtime209Error("standard-Llama directory provenance claims a model download")

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
        plan_sha256=plan.identity_sha256(),
    )

    qk_evidence = _qk_learned_weight_evidence(reference, target)
    prompt_records = [
        _prompt_evidence(
            reference,
            candidate,
            target,
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
    boundary_reference, boundary_candidate = _full_logits(reference, target, boundary_ids)
    boundary_metrics = _logit_metrics(boundary_reference, boundary_candidate)
    over_context = boundary_ids + [0]
    reference_rejected = False
    candidate_rejected = False
    try:
        reference.next_token_logits(over_context)
    except ValueError:
        reference_rejected = True
    try:
        candidate.next_token_logits(over_context)
    except ValueError:
        candidate_rejected = True

    failures: list[str] = []
    if not qk_evidence["all_q_exact"] or not qk_evidence["all_k_exact"]:
        failures.append("learned_qk_rope_weight_conversion")
    if not all(row["full_prompt_logits"].get("allclose") for row in prompt_records):
        failures.append("full_prompt_logits")
    all_steps = [step for row in prompt_records for step in row["generation_steps"]]
    if not all(step.get("allclose") for step in all_steps):
        failures.append("generation_step_logits")
    if not all(step.get("greedy_token_exact") for step in all_steps):
        failures.append("greedy_next_token")
    if not all(row["greedy_tokens_exact"] for row in prompt_records):
        failures.append("short_generation_token_ids")
    if not all(row["decoded_continuation_exact"] for row in prompt_records):
        failures.append("short_generation_decoded")
    if not boundary_metrics.get("allclose"):
        failures.append("full_context_logits")
    if not reference_rejected or not candidate_rejected:
        failures.append("context_overflow_rejection")

    metric_records = [row["full_prompt_logits"] for row in prompt_records]
    metric_records.extend(all_steps)
    metric_records.append(boundary_metrics)
    valid_abs = [float(row["max_abs_error"]) for row in metric_records if "max_abs_error" in row]
    valid_rel = [float(row["max_rel_error"]) for row in metric_records if "max_rel_error" in row]

    evidence: dict[str, Any] = {
        "schema": SCHEMA,
        "passed": not failures,
        "bridge": {
            "source_sha": bridge_source_sha,
            "reused_runtime97_harness_lineage": True,
            "conversion_seam": "twelve_six.inference.transformers_llama",
            "standard_directory_seam": "twelve_six.inference.llama_runtime_export",
            "second_exporter_added": False,
            "new_transformers_model_adapter_added": False,
            "plan_sha256": plan.identity_sha256(),
        },
        "learned_source": {
            "artifact_id": source_artifact_id,
            "artifact_name": artifact_metadata["name"],
            "artifact_digest": source_artifact_digest,
            "workflow_run_id": source_run_id,
            "source_git_sha": source_sha,
            "checkpoint_role": CHECKPOINT_ROLE,
            "checkpoint_id": diagnostics["checkpoint_id"],
            "checkpoint_step": diagnostics["step"],
            "checkpoint_tokens_seen": diagnostics["tokens_seen"],
            "model_spec_sha256": diagnostics["model_spec_sha256"],
            "parameter_count": diagnostics["parameter_count"],
            "corpus_identity_sha256": diagnostics["dataset_manifest_sha256"],
            "run_manifest_sha256": diagnostics["run_manifest_sha256"],
            "fresh_verification_sha256": _sha256_file(fresh_verification_path),
            "fresh_verification_identity_sha256": fresh.get("identity_sha256"),
            "retained_index_sha256": _sha256_file(retained_index_path),
            "source_run_conclusion": run_metadata["conclusion"],
        },
        "tokenizer_identity": {
            "version": diagnostics["tokenizer_version"],
            "config_sha256": diagnostics["tokenizer_config_sha256"],
            "vocab_sha256": diagnostics["tokenizer_vocab_sha256"],
            "vocab_size": diagnostics["vocab_size"],
            "special_token_semantics_invented": False,
        },
        "runtime": {
            "d08_source_sha": d08_source_sha,
            "d08_profile_id": "linux-x86_64-transformers-interop",
            "d08_profile_sha256": d08_profile_sha256,
            "d08_overlay_lock_sha256": d08_overlay_sha256,
            "universal_bootstrap_source_sha": bootstrap_source_sha,
            "universal_bootstrap_manifest_sha256": _sha256_file(bootstrap_manifest_path),
            "universal_bootstrap_identity_sha256": bootstrap_manifest.get("identity_sha256"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "device": "cpu",
            "dtype": "float32",
            "offline_environment": {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            },
        },
        "standard_llama_directory": {
            "architecture": target.__class__.__name__,
            "class_module": target.__class__.__module__,
            "load_method": "LlamaForCausalLM.from_pretrained(local_directory, local_files_only=True)",
            "runtime_weights_sha256": runtime_provenance["runtime_weights_sha256"],
            "runtime_config_sha256": runtime_provenance["runtime_config_sha256"],
            "source_manifest_sha256": runtime_provenance["source_manifest_sha256"],
            "qk_rope_basis_conversion": runtime_provenance["rope_transform"],
            "foreign_pretrained_weights": runtime_provenance["foreign_pretrained_weights"],
            "model_downloaded": runtime_provenance["model_downloaded"],
        },
        "learned_qk_rope_weights": qk_evidence,
        "prompts": prompt_records,
        "context_boundary": {
            "max_context_tokens": candidate.max_context_tokens,
            "boundary_input_sha256": _sha256(bytes(boundary_ids)),
            "full_logits": boundary_metrics,
            "reference_over_context_rejected": reference_rejected,
            "transformers_bridge_over_context_rejected": candidate_rejected,
        },
        "acceptance": {
            "logit_atol": LOGIT_ATOL,
            "logit_rtol": LOGIT_RTOL,
            "tolerance_changed_for_runtime209": False,
            "max_abs_error": max(valid_abs) if valid_abs else None,
            "max_rel_error": max(valid_rel) if valid_rel else None,
            "full_logit_tensors_compared": len(prompt_records) + 1,
            "generation_next_logit_vectors_compared": len(all_steps),
        },
        "failures": failures,
        "truth_boundary": {
            "foreign_pretrained_weights_used": False,
            "hub_model_download_used": False,
            "paid_compute_used": False,
            "instruction_or_alignment_claim": False,
            "quality_or_capability_claim": False,
            "production_stage_promotion": False,
            "source_substitution_allowed": False,
        },
    }
    evidence["evidence_sha256"] = _canonical_hash(evidence)
    evidence_path = output_dir / "runtime209-transformers-learned-10m-parity.json"
    evidence_path.write_text(
        json.dumps(evidence, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise Runtime209Error(f"learned 10M Transformers parity failed: {failures}")
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--fresh-verification", type=Path, required=True)
    parser.add_argument("--retained-index", type=Path, required=True)
    parser.add_argument("--artifact-metadata", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--bootstrap-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bridge-source-sha", required=True)
    parser.add_argument("--source-artifact-id", type=int, required=True)
    parser.add_argument("--source-artifact-digest", required=True)
    parser.add_argument("--source-run-id", type=int, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--d08-source-sha", required=True)
    parser.add_argument("--d08-profile-sha256", required=True)
    parser.add_argument("--d08-overlay-sha256", required=True)
    parser.add_argument("--bootstrap-source-sha", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence = collect_evidence(
        repo_root=args.repo_root.resolve(),
        checkpoint_dir=args.checkpoint_dir.resolve(),
        fresh_verification_path=args.fresh_verification.resolve(),
        retained_index_path=args.retained_index.resolve(),
        artifact_metadata_path=args.artifact_metadata.resolve(),
        run_metadata_path=args.run_metadata.resolve(),
        bootstrap_manifest_path=args.bootstrap_manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        bridge_source_sha=args.bridge_source_sha,
        source_artifact_id=args.source_artifact_id,
        source_artifact_digest=args.source_artifact_digest,
        source_run_id=args.source_run_id,
        source_sha=args.source_sha,
        d08_source_sha=args.d08_source_sha,
        d08_profile_sha256=args.d08_profile_sha256,
        d08_overlay_sha256=args.d08_overlay_sha256,
        bootstrap_source_sha=args.bootstrap_source_sha,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(evidence, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
