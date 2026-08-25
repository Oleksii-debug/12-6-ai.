"""Locked Transformers 5.15 execution and parity evidence for 12-6 Llama exports."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load as load_safetensors_bytes

from twelve_six.checkpoint import verify_hf_directory
from twelve_six.checkpoint.hf_export import (
    EXPORTED_CONFIG_NAME,
    EXPORTED_SOURCE_MANIFEST_NAME,
    EXPORTED_WEIGHTS_NAME,
)
from twelve_six.model import ModelSpec, RotaryEmbedding, apply_rope

from .first_party import FirstPartyInferenceBackend, load_first_party_backend
from .transformers_llama import (
    build_llama_interop_plan,
    convert_state_dict_to_llama,
    llama_config_dict,
    rope_pairwise_to_llama_permutation,
)

TRANSFORMERS_VERSION = "5.15.0"
EVIDENCE_SCHEMA = "12-6.transformers-llama-runtime-parity.v1"
# Independent but semantically equivalent kernels may round differently in fp32.
# This is a provisional acceptance ceiling; runtime evidence reports observed error
# so the bound can be tightened to the smallest justified stable value.
LOGIT_ATOL = 1e-5
LOGIT_RTOL = 1e-5


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(value: dict[str, Any]) -> str:
    return _sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def _json_object(data: bytes, *, artifact: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{artifact} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{artifact} must contain a JSON object")
    return value


def _transformers_types():
    version = importlib.metadata.version("transformers")
    if version != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Transformers runtime version mismatch: expected {TRANSFORMERS_VERSION}, got {version}"
        )
    from transformers import LlamaConfig, LlamaForCausalLM  # noqa: PLC0415
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb  # noqa: PLC0415

    return LlamaConfig, LlamaForCausalLM, apply_rotary_pos_emb


@dataclass(slots=True)
class TransformersLlamaRuntime:
    model: Any
    tokenizer: Any
    spec: ModelSpec
    checkpoint_id: str
    source_manifest_sha256: str
    weights_sha256: str
    config_sha256: str
    plan_sha256: str

    @property
    def max_context_tokens(self) -> int:
        return self.spec.max_seq_len

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(token_ids, errors="replace")

    def _validate_ids(self, input_ids: Sequence[int]) -> list[int]:
        ids = list(input_ids)
        if not ids:
            raise ValueError("input_ids must be non-empty")
        if len(ids) > self.spec.max_seq_len:
            raise ValueError("input_ids exceed canonical model context")
        for token_id in ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError("input token IDs must be integers")
            if not 0 <= token_id < self.spec.vocab_size:
                raise ValueError("input token ID is outside canonical vocabulary")
        return ids

    @torch.no_grad()
    def next_token_logits_tensor(self, input_ids: Sequence[int]) -> torch.Tensor:
        ids = self._validate_ids(input_ids)
        tensor = torch.tensor([ids], dtype=torch.long, device=next(self.model.parameters()).device)
        logits = self.model(input_ids=tensor, use_cache=False).logits[0, -1]
        return logits.detach().float().cpu()

    def next_token_logits(self, input_ids: Sequence[int]) -> list[float]:
        return self.next_token_logits_tensor(input_ids).tolist()

    @torch.no_grad()
    def greedy_ids(self, input_ids: Sequence[int], *, max_new_tokens: int) -> list[int]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        generated = self._validate_ids(input_ids)
        steps = min(max_new_tokens, self.spec.max_seq_len - len(generated))
        for _ in range(steps):
            token = int(torch.argmax(self.next_token_logits_tensor(generated)).item())
            generated.append(token)
        return generated


def load_transformers_llama_runtime(
    checkpoint_dir: str | Path,
    export_dir: str | Path,
) -> tuple[FirstPartyInferenceBackend, TransformersLlamaRuntime]:
    """Instantiate LlamaForCausalLM from the exact verified #95 export bytes.

    No pretrained-model API is called. The exported source tensors retain canonical
    12-6 names; they are decoded from the exact consumed safetensors bytes and
    converted in memory using the incumbent D07 mapping before strict loading.
    """

    checkpoint = Path(checkpoint_dir)
    export = Path(export_dir)
    reference = load_first_party_backend(checkpoint)
    attestation = verify_hf_directory(export)

    weights_bytes = (export / EXPORTED_WEIGHTS_NAME).read_bytes()
    config_bytes = (export / EXPORTED_CONFIG_NAME).read_bytes()
    source_manifest_bytes = (export / EXPORTED_SOURCE_MANIFEST_NAME).read_bytes()
    weights_sha = _sha256(weights_bytes)
    config_sha = _sha256(config_bytes)
    source_manifest_sha = _sha256(source_manifest_bytes)

    if weights_sha != attestation.get("model_safetensors_sha256"):
        raise ValueError("consumed model.safetensors bytes changed after export verification")
    if config_sha != attestation.get("config_sha256"):
        raise ValueError("consumed config.json bytes changed after export verification")
    if source_manifest_sha != attestation.get("source_manifest_sha256"):
        raise ValueError("consumed source manifest bytes changed after export verification")
    if attestation.get("checkpoint_id") != reference.manifest.get("checkpoint_id"):
        raise ValueError("export checkpoint_id does not match canonical reference")

    source_manifest = _json_object(source_manifest_bytes, artifact=EXPORTED_SOURCE_MANIFEST_NAME)
    if source_manifest != reference.manifest:
        raise ValueError("exported source manifest differs from verified checkpoint manifest")
    identity = source_manifest.get("identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("model_spec"), dict):
        raise ValueError("exported source manifest is missing ModelSpec")
    spec = ModelSpec.from_dict(identity["model_spec"])
    if spec.identity_sha256() != identity.get("model_spec_hash"):
        raise ValueError("exported ModelSpec identity mismatch")
    if weights_sha != reference.manifest["files"]["weights.safetensors"]["sha256"]:
        raise ValueError("executed export weights do not match canonical checkpoint weights")

    config_payload = _json_object(config_bytes, artifact=EXPORTED_CONFIG_NAME)
    expected_config = llama_config_dict(spec)
    if config_payload != expected_config:
        raise ValueError("executed export config does not match exact D07 Llama config mapping")

    try:
        source_state = load_safetensors_bytes(weights_bytes)
    except Exception as exc:
        raise ValueError("verified exported safetensors bytes cannot be decoded") from exc
    converted = convert_state_dict_to_llama(spec, source_state)
    plan = build_llama_interop_plan(spec)

    LlamaConfig, LlamaForCausalLM, _ = _transformers_types()
    config = LlamaConfig.from_dict(config_payload)
    model = LlamaForCausalLM(config)
    incompatible = model.load_state_dict(converted, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "Transformers Llama strict tensor mapping failed: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.eval()

    return reference, TransformersLlamaRuntime(
        model=model,
        tokenizer=reference.tokenizer,
        spec=spec,
        checkpoint_id=reference.manifest["checkpoint_id"],
        source_manifest_sha256=source_manifest_sha,
        weights_sha256=weights_sha,
        config_sha256=config_sha,
        plan_sha256=plan.identity_sha256(),
    )


def _logit_error(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float, bool]:
    absolute = (reference - candidate).abs()
    max_abs = float(absolute.max().item())
    denominator = reference.abs().clamp_min(torch.finfo(reference.dtype).tiny)
    max_rel = float((absolute / denominator).max().item())
    passed = bool(torch.allclose(reference, candidate, atol=LOGIT_ATOL, rtol=LOGIT_RTOL))
    return max_abs, max_rel, passed


def _reference_logits(reference: FirstPartyInferenceBackend, ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(reference.next_token_logits(ids), dtype=torch.float32)


def _rope_runtime_probe(runtime: TransformersLlamaRuntime) -> dict[str, Any]:
    _, _, hf_apply_rope = _transformers_types()
    spec = runtime.spec
    torch.manual_seed(2408)
    source = torch.randn(1, spec.n_heads, 7, spec.head_dim, dtype=torch.float32)
    canonical_rope = RotaryEmbedding(spec.rope_rotary_dim, spec.rope_theta)
    cos, sin = canonical_rope.cos_sin(7, device=source.device, dtype=source.dtype)
    canonical = apply_rope(source, cos, sin, spec.rope_rotary_dim)

    permutation = torch.tensor(
        rope_pairwise_to_llama_permutation(heads=1, head_dim=spec.head_dim),
        dtype=torch.long,
    )
    llama_basis = source.index_select(-1, permutation)
    position_ids = torch.arange(7, dtype=torch.long).unsqueeze(0)
    hf_cos, hf_sin = runtime.model.model.rotary_emb(llama_basis, position_ids)
    hf_rotated, _ = hf_apply_rope(llama_basis, llama_basis, hf_cos, hf_sin)
    expected = canonical.index_select(-1, permutation)
    max_abs = float((hf_rotated - expected).abs().max().item())
    return {
        "exact": bool(torch.equal(hf_rotated, expected)),
        "max_abs_error": max_abs,
        "sequence_length": 7,
    }


def collect_transformers_llama_parity_evidence(
    checkpoint_dir: str | Path,
    export_dir: str | Path,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 4,
) -> dict[str, Any]:
    """Execute canonical and Transformers paths and return machine-checkable evidence."""

    if not prompts or any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError("prompts must contain non-empty strings")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    reference, candidate = load_transformers_llama_runtime(checkpoint_dir, export_dir)
    max_abs = 0.0
    max_rel = 0.0
    steps_compared = 0
    prompt_records: list[dict[str, Any]] = []
    failures: list[str] = []

    for prompt in prompts:
        ids = candidate.encode(prompt)
        byte_ids = list(prompt.encode("utf-8"))
        tokenizer_exact = ids == byte_ids and candidate.decode(ids) == prompt
        if not tokenizer_exact:
            failures.append("tokenizer_mapping")

        reference_ids = list(ids)
        candidate_ids = list(ids)
        steps = min(max_new_tokens, candidate.max_context_tokens - len(ids))
        for _ in range(steps):
            reference_logits = _reference_logits(reference, reference_ids)
            candidate_logits = candidate.next_token_logits_tensor(candidate_ids)
            abs_error, rel_error, logits_ok = _logit_error(reference_logits, candidate_logits)
            max_abs = max(max_abs, abs_error)
            max_rel = max(max_rel, rel_error)
            steps_compared += 1
            if not logits_ok:
                failures.append("logit_parity")
            reference_token = int(torch.argmax(reference_logits).item())
            candidate_token = int(torch.argmax(candidate_logits).item())
            if reference_token != candidate_token:
                failures.append("greedy_token_parity")
            reference_ids.append(reference_token)
            candidate_ids.append(candidate_token)

        greedy_exact = reference_ids == candidate_ids
        decode_exact = reference.decode(reference_ids) == candidate.decode(candidate_ids)
        if not greedy_exact:
            failures.append("greedy_sequence_parity")
        if not decode_exact:
            failures.append("decode_parity")
        prompt_records.append(
            {
                "prompt_sha256": _sha256(prompt.encode("utf-8")),
                "input_tokens": len(ids),
                "tokenizer_exact_utf8_bytes": tokenizer_exact,
                "generated_tokens": len(candidate_ids) - len(ids),
                "greedy_exact": greedy_exact,
                "decode_exact": decode_exact,
            }
        )

    boundary_ids = [index % candidate.spec.vocab_size for index in range(candidate.max_context_tokens)]
    boundary_reference = _reference_logits(reference, boundary_ids)
    boundary_candidate = candidate.next_token_logits_tensor(boundary_ids)
    boundary_abs, boundary_rel, boundary_ok = _logit_error(boundary_reference, boundary_candidate)
    max_abs = max(max_abs, boundary_abs)
    max_rel = max(max_rel, boundary_rel)
    if not boundary_ok:
        failures.append("context_boundary_logit_parity")
    boundary_generation_exact = (
        reference.model.generate(
            torch.tensor([boundary_ids], dtype=torch.long),
            max_new_tokens=1,
            do_sample=False,
        )[0].tolist()
        == candidate.greedy_ids(boundary_ids, max_new_tokens=1)
        == boundary_ids
    )
    if not boundary_generation_exact:
        failures.append("context_boundary_generation")

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
    if not (reference_rejected and candidate_rejected):
        failures.append("context_overflow_rejection")

    rope = _rope_runtime_probe(candidate)
    if not rope["exact"]:
        failures.append("rope_runtime_exactness")

    failures = sorted(set(failures))
    evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "passed": not failures,
        "transformers_version": importlib.metadata.version("transformers"),
        "architecture": "LlamaForCausalLM",
        "checkpoint_id": candidate.checkpoint_id,
        "source_git_sha": reference.manifest["identity"]["git_sha"],
        "model_spec_sha256": candidate.spec.identity_sha256(),
        "tokenizer_config_sha256": reference.tokenizer.identity.config_sha256,
        "tokenizer_vocab_sha256": reference.tokenizer.identity.vocab_sha256,
        "source_manifest_sha256": candidate.source_manifest_sha256,
        "executed_weights_sha256": candidate.weights_sha256,
        "executed_config_sha256": candidate.config_sha256,
        "interop_plan_sha256": candidate.plan_sha256,
        "tensor_mapping": {
            "strict_load": "PASS",
            "source_tensor_count": len(reference.model.state_dict()),
            "target_tensor_count": len(candidate.model.state_dict()),
            "qk_rope_basis_conversion": "PASS" if rope["exact"] else "FAIL",
        },
        "rope": rope,
        "logits": {
            "dtype": "float32",
            "atol": LOGIT_ATOL,
            "rtol": LOGIT_RTOL,
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "steps_compared": steps_compared + 1,
            "exact_zero_observed": max_abs == 0.0,
        },
        "prompts": prompt_records,
        "context": {
            "max_context_tokens": candidate.max_context_tokens,
            "boundary_logit_parity": boundary_ok,
            "boundary_generation_emits_zero_tokens": boundary_generation_exact,
            "over_context_reference_rejected": reference_rejected,
            "over_context_transformers_rejected": candidate_rejected,
        },
        "failures": failures,
        "truth_boundary": {
            "pretrained_weights_used": False,
            "pretrained_model_api_used": False,
            "hub_model_download_used": False,
            "chat_template_used": False,
            "tested_runtime": "CPU_FP32",
            "larger_modelspecs_runtime_tested": False,
        },
    }
    evidence["evidence_sha256"] = _canonical_hash(evidence)
    return evidence


def assert_transformers_llama_parity(
    checkpoint_dir: str | Path,
    export_dir: str | Path,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 4,
) -> dict[str, Any]:
    evidence = collect_transformers_llama_parity_evidence(
        checkpoint_dir,
        export_dir,
        prompts,
        max_new_tokens=max_new_tokens,
    )
    if not evidence["passed"]:
        raise RuntimeError(f"Transformers Llama parity failed: {evidence['failures']}")
    return evidence
