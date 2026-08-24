"""Machine-readable evidence for verified first-party Base inference.

This module observes existing D01/D04/D05/D07 contracts. It does not implement
model architecture, tokenization, checkpoint serialization, or sampling policy.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import GenerationConfig, GenerationResult, InferenceBackend
from .generation import generate
from .sampling import greedy_token

INFERENCE_EVIDENCE_SCHEMA = "12-6.s0-first-party-inference-evidence.v1"
AUTHORITY = "LOCAL_FREE_OR_FREE_HOSTED_CPU_EVIDENCE_NOT_PROMOTION"
_SHA256_HEX = frozenset("0123456789abcdef")


class InferenceEvidenceError(ValueError):
    """Raised when first-party inference evidence is incomplete or inconsistent."""


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_logits(values: Sequence[float]) -> str:
    digest = hashlib.sha256()
    for index, value in enumerate(values):
        number = float(value)
        if not math.isfinite(number):
            raise InferenceEvidenceError(f"non-finite logit at index {index}")
        digest.update(struct.pack(">d", number))
    return digest.hexdigest()


def _result_dict(result: GenerationResult) -> dict[str, Any]:
    return {
        "prompt_token_ids": list(result.prompt_token_ids),
        "generated_token_ids": list(result.generated_token_ids),
        "text": result.text,
        "text_sha256": _sha256_text(result.text),
        "stop_reason": result.stop_reason,
    }


def _require_exact_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(ch not in _SHA256_HEX for ch in value)
    ):
        raise InferenceEvidenceError(f"{field} must be an exact lowercase SHA-256")
    return value


def _require_candidate_diagnostics(candidate: InferenceBackend) -> dict[str, Any]:
    diagnostics_fn = getattr(candidate, "diagnostics", None)
    if not callable(diagnostics_fn):
        raise InferenceEvidenceError("candidate backend must expose diagnostics()")
    diagnostics = diagnostics_fn()
    if not isinstance(diagnostics, Mapping):
        raise InferenceEvidenceError("candidate diagnostics must be a mapping")
    payload = dict(diagnostics)
    required = {
        "backend",
        "checkpoint_id",
        "git_sha",
        "model_spec_sha256",
        "parameter_count",
        "vocab_size",
        "max_context_tokens",
        "tokenizer_version",
        "tokenizer_config_sha256",
        "tokenizer_vocab_sha256",
        "step",
        "tokens_seen",
        "device",
    }
    missing = required - payload.keys()
    if missing:
        raise InferenceEvidenceError(
            f"candidate diagnostics missing required fields: {sorted(missing)}"
        )
    if payload["backend"] != "first_party_torch":
        raise InferenceEvidenceError("candidate backend is not canonical first_party_torch")
    for field in (
        "checkpoint_id",
        "model_spec_sha256",
        "tokenizer_config_sha256",
        "tokenizer_vocab_sha256",
    ):
        _require_exact_sha256(payload[field], f"diagnostics.{field}")
    git_sha = payload["git_sha"]
    if (
        not isinstance(git_sha, str)
        or len(git_sha) not in {40, 64}
        or git_sha != git_sha.lower()
        or any(ch not in _SHA256_HEX for ch in git_sha)
    ):
        raise InferenceEvidenceError("diagnostics.git_sha must be a full lowercase Git SHA")
    if payload["parameter_count"] != 10_140:
        raise InferenceEvidenceError("S0 first-party evidence requires 10,140 parameters")
    if payload["vocab_size"] != 256:
        raise InferenceEvidenceError("S0 first-party evidence requires vocab_size=256")
    if payload["max_context_tokens"] != 128:
        raise InferenceEvidenceError("S0 first-party evidence requires context=128")
    if not isinstance(payload["step"], int) or payload["step"] <= 0:
        raise InferenceEvidenceError("candidate checkpoint must contain trained optimizer steps")
    if not isinstance(payload["tokens_seen"], int) or payload["tokens_seen"] <= 0:
        raise InferenceEvidenceError("candidate checkpoint must contain optimized tokens")
    return payload


def _exact_parity_trace(
    reference: InferenceBackend,
    candidate: InferenceBackend,
    prompt: str,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    reference_prompt = reference.encode(prompt)
    candidate_prompt = candidate.encode(prompt)
    if reference_prompt != candidate_prompt:
        raise InferenceEvidenceError("reference/candidate prompt tokenization diverged")
    if not reference_prompt:
        raise InferenceEvidenceError("prompt encoded to zero tokens")
    if reference.max_context_tokens != candidate.max_context_tokens:
        raise InferenceEvidenceError("reference/candidate context limits diverged")
    if reference.eos_token_id != candidate.eos_token_id:
        raise InferenceEvidenceError("reference/candidate EOS contracts diverged")
    if len(reference_prompt) > reference.max_context_tokens:
        raise InferenceEvidenceError("parity prompt exceeds context limit")

    generated: list[int] = []
    steps: list[dict[str, Any]] = []
    for step_index in range(max_new_tokens):
        input_ids = (*reference_prompt, *generated)
        if len(input_ids) >= reference.max_context_tokens:
            break
        reference_logits = [float(value) for value in reference.next_token_logits(input_ids)]
        candidate_logits = [float(value) for value in candidate.next_token_logits(input_ids)]
        if len(reference_logits) != len(candidate_logits):
            raise InferenceEvidenceError("reference/candidate logit vector sizes diverged")
        if reference_logits != candidate_logits:
            raise InferenceEvidenceError(
                f"reference/candidate logits diverged at step {step_index}"
            )
        reference_token = greedy_token(reference_logits)
        candidate_token = greedy_token(candidate_logits)
        if reference_token != candidate_token:
            raise InferenceEvidenceError(
                f"reference/candidate greedy token diverged at step {step_index}"
            )
        generated.append(reference_token)
        reference_decoded = reference.decode(generated)
        candidate_decoded = candidate.decode(generated)
        if reference_decoded != candidate_decoded:
            raise InferenceEvidenceError(
                f"reference/candidate decode diverged at step {step_index}"
            )
        reference_logits_sha = _sha256_logits(reference_logits)
        candidate_logits_sha = _sha256_logits(candidate_logits)
        steps.append(
            {
                "step_index": step_index,
                "input_token_ids": list(input_ids),
                "reference_logits_sha256": reference_logits_sha,
                "candidate_logits_sha256": candidate_logits_sha,
                "logits_exact_equal": reference_logits_sha == candidate_logits_sha,
                "greedy_token_id": reference_token,
                "decoded_prefix": reference_decoded,
                "decoded_prefix_sha256": _sha256_text(reference_decoded),
            }
        )
        if reference.eos_token_id is not None and reference_token == reference.eos_token_id:
            break

    return {
        "prompt": prompt,
        "prompt_token_ids": reference_prompt,
        "steps": steps,
        "steps_compared": len(steps),
        "all_logits_exact": all(step["logits_exact_equal"] for step in steps),
        "generated_token_ids": generated,
        "decoded_text": reference.decode(generated),
        "decoded_text_sha256": _sha256_text(reference.decode(generated)),
    }


def collect_first_party_inference_evidence(
    reference: InferenceBackend,
    candidate: InferenceBackend,
    *,
    prompts: Sequence[str] = ("12-6", "Base"),
    seed: int = 17,
    max_new_tokens: int = 6,
) -> dict[str, Any]:
    """Collect strict trained-checkpoint generation and parity evidence.

    ``reference`` is the in-memory D01 model adapter immediately after training.
    ``candidate`` is the same weights reloaded through the verified D05 -> D07
    first-party checkpoint adapter.
    """

    if not prompts or any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise InferenceEvidenceError("prompts must contain non-empty strings")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise InferenceEvidenceError("seed must be a non-negative integer")
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise InferenceEvidenceError("max_new_tokens must be a positive integer")
    if reference.max_context_tokens != candidate.max_context_tokens:
        raise InferenceEvidenceError("reference/candidate context limits diverged")
    if reference.eos_token_id != candidate.eos_token_id:
        raise InferenceEvidenceError("reference/candidate EOS contracts diverged")

    diagnostics = _require_candidate_diagnostics(candidate)
    prompt_reports: list[dict[str, Any]] = []
    greedy_config = GenerationConfig(max_new_tokens=max_new_tokens, sample=False, seed=seed)
    sampled_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        sample=True,
        temperature=0.8,
        top_k=32,
        top_p=0.9,
        seed=seed,
    )

    for prompt in prompts:
        trace = _exact_parity_trace(
            reference,
            candidate,
            prompt,
            max_new_tokens=max_new_tokens,
        )
        greedy_reference = generate(reference, prompt, greedy_config)
        greedy_candidate = generate(candidate, prompt, greedy_config)
        if greedy_reference != greedy_candidate:
            raise InferenceEvidenceError("direct/reloaded greedy generation diverged")

        sampled_reference = generate(reference, prompt, sampled_config)
        sampled_first = generate(candidate, prompt, sampled_config)
        sampled_second = generate(candidate, prompt, sampled_config)
        if sampled_reference != sampled_first or sampled_first != sampled_second:
            raise InferenceEvidenceError(
                "seeded sampling is not repeatable across direct/reloaded execution"
            )
        prompt_reports.append(
            {
                "parity_trace": trace,
                "greedy": {
                    "reference": _result_dict(greedy_reference),
                    "candidate": _result_dict(greedy_candidate),
                    "exact_equal": True,
                },
                "seeded_sampling": {
                    "seed": seed,
                    "temperature": sampled_config.temperature,
                    "top_k": sampled_config.top_k,
                    "top_p": sampled_config.top_p,
                    "reference": _result_dict(sampled_reference),
                    "candidate_first": _result_dict(sampled_first),
                    "candidate_second": _result_dict(sampled_second),
                    "direct_vs_reloaded_exact": True,
                    "repeatable": True,
                },
            }
        )

    stop_prompt = prompts[0]
    stop_prompt_ids = candidate.encode(stop_prompt)
    first_token = greedy_token(candidate.next_token_logits(stop_prompt_ids))
    stop_token_result = generate(
        candidate,
        stop_prompt,
        GenerationConfig(max_new_tokens=max_new_tokens, stop_token_ids=(first_token,)),
    )
    if stop_token_result.stop_reason != "stop_token" or stop_token_result.generated_token_ids != (
        first_token,
    ):
        raise InferenceEvidenceError("stop-token semantics diverged")

    first_token_text = candidate.decode((first_token,))
    if not first_token_text:
        raise InferenceEvidenceError("first generated token decoded to empty text")
    stop_string_result = generate(
        candidate,
        stop_prompt,
        GenerationConfig(max_new_tokens=max_new_tokens, stop_strings=(first_token_text,)),
    )
    if stop_string_result.stop_reason != "stop_string" or stop_string_result.text != "":
        raise InferenceEvidenceError("stop-string strip semantics diverged")

    context_prompt = "A" * candidate.max_context_tokens
    if len(candidate.encode(context_prompt)) != candidate.max_context_tokens:
        raise InferenceEvidenceError(
            "canonical context probe no longer maps one ASCII byte to one token"
        )
    context_result = generate(
        candidate,
        context_prompt,
        GenerationConfig(max_new_tokens=1),
    )
    if context_result.stop_reason != "context_limit" or context_result.generated_token_ids:
        raise InferenceEvidenceError("context-limit stop semantics diverged")
    over_context_rejected = False
    try:
        generate(
            candidate,
            context_prompt + "A",
            GenerationConfig(max_new_tokens=1),
        )
    except ValueError:
        over_context_rejected = True
    if not over_context_rejected:
        raise InferenceEvidenceError("over-context prompt was not rejected")

    evidence: dict[str, Any] = {
        "schema_version": INFERENCE_EVIDENCE_SCHEMA,
        "authority": AUTHORITY,
        "backend_diagnostics": diagnostics,
        "contract": {
            "max_context_tokens": candidate.max_context_tokens,
            "eos_token_id": candidate.eos_token_id,
            "raw_base_completion_semantics": True,
        },
        "prompts": prompt_reports,
        "stop_semantics": {
            "stop_token_id": first_token,
            "stop_token_result": _result_dict(stop_token_result),
            "stop_string_sha256": _sha256_text(first_token_text),
            "stop_string_result": _result_dict(stop_string_result),
            "strip_stop_string_verified": True,
        },
        "context_semantics": {
            "exact_context_prompt_tokens": candidate.max_context_tokens,
            "at_limit_result": _result_dict(context_result),
            "over_context_rejected": True,
        },
        "claims": {
            "foreign_pretrained_weights_used": False,
            "instruction_or_alignment_behavior_added": False,
            "paid_compute_authorized_or_used": False,
            "candidate_or_stable_promotion": False,
            "windows_nvda_live_runtime_tested": False,
        },
    }
    evidence["evidence_sha256"] = _canonical_hash(evidence)
    validate_first_party_inference_evidence(evidence)
    return evidence


def validate_first_party_inference_evidence(payload: Mapping[str, Any]) -> None:
    """Fail closed on tampered or semantically incomplete evidence."""

    if not isinstance(payload, Mapping):
        raise InferenceEvidenceError("evidence must be a mapping")
    data = copy.deepcopy(dict(payload))
    supplied_hash = data.pop("evidence_sha256", None)
    _require_exact_sha256(supplied_hash, "evidence_sha256")
    if _canonical_hash(data) != supplied_hash:
        raise InferenceEvidenceError("inference evidence self-hash mismatch")
    if data.get("schema_version") != INFERENCE_EVIDENCE_SCHEMA:
        raise InferenceEvidenceError("unsupported inference evidence schema")
    if data.get("authority") != AUTHORITY:
        raise InferenceEvidenceError("inference evidence authority boundary changed")

    diagnostics = data.get("backend_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise InferenceEvidenceError("backend_diagnostics must be a mapping")
    if diagnostics.get("backend") != "first_party_torch":
        raise InferenceEvidenceError("evidence backend is not first_party_torch")
    for field in (
        "checkpoint_id",
        "model_spec_sha256",
        "tokenizer_config_sha256",
        "tokenizer_vocab_sha256",
    ):
        _require_exact_sha256(diagnostics.get(field), f"backend_diagnostics.{field}")
    if diagnostics.get("parameter_count") != 10_140:
        raise InferenceEvidenceError("evidence parameter count drifted")
    if diagnostics.get("vocab_size") != 256:
        raise InferenceEvidenceError("evidence vocabulary drifted")
    if diagnostics.get("max_context_tokens") != 128:
        raise InferenceEvidenceError("evidence context limit drifted")

    prompts = data.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise InferenceEvidenceError("evidence requires prompt reports")
    for report in prompts:
        if not isinstance(report, Mapping):
            raise InferenceEvidenceError("prompt report must be a mapping")
        trace = report.get("parity_trace")
        if not isinstance(trace, Mapping) or not trace.get("all_logits_exact"):
            raise InferenceEvidenceError("prompt parity trace is not exact")
        steps = trace.get("steps")
        if not isinstance(steps, list) or not steps:
            raise InferenceEvidenceError("prompt parity trace contains no compared steps")
        for step in steps:
            if not isinstance(step, Mapping) or not step.get("logits_exact_equal"):
                raise InferenceEvidenceError("successful logit comparison is missing")
            left = _require_exact_sha256(
                step.get("reference_logits_sha256"), "reference_logits_sha256"
            )
            right = _require_exact_sha256(
                step.get("candidate_logits_sha256"), "candidate_logits_sha256"
            )
            if left != right:
                raise InferenceEvidenceError("logit comparison digests differ")
            _require_exact_sha256(
                step.get("decoded_prefix_sha256"), "decoded_prefix_sha256"
            )
        greedy = report.get("greedy")
        if not isinstance(greedy, Mapping) or not greedy.get("exact_equal"):
            raise InferenceEvidenceError("greedy parity is not exact")
        if greedy.get("reference") != greedy.get("candidate"):
            raise InferenceEvidenceError("greedy result payloads differ")
        sampled = report.get("seeded_sampling")
        if (
            not isinstance(sampled, Mapping)
            or not sampled.get("repeatable")
            or not sampled.get("direct_vs_reloaded_exact")
        ):
            raise InferenceEvidenceError("seeded sampling proof is incomplete")
        if not (
            sampled.get("reference")
            == sampled.get("candidate_first")
            == sampled.get("candidate_second")
        ):
            raise InferenceEvidenceError("seeded sampling result payloads differ")

    stop = data.get("stop_semantics")
    if not isinstance(stop, Mapping) or not stop.get("strip_stop_string_verified"):
        raise InferenceEvidenceError("stop semantics proof is incomplete")
    stop_token_result = stop.get("stop_token_result")
    stop_string_result = stop.get("stop_string_result")
    if (
        not isinstance(stop_token_result, Mapping)
        or stop_token_result.get("stop_reason") != "stop_token"
    ):
        raise InferenceEvidenceError("stop-token evidence is invalid")
    if (
        not isinstance(stop_string_result, Mapping)
        or stop_string_result.get("stop_reason") != "stop_string"
        or stop_string_result.get("text") != ""
    ):
        raise InferenceEvidenceError("stop-string evidence is invalid")

    context = data.get("context_semantics")
    if not isinstance(context, Mapping) or not context.get("over_context_rejected"):
        raise InferenceEvidenceError("over-context rejection evidence is missing")
    at_limit = context.get("at_limit_result")
    if (
        not isinstance(at_limit, Mapping)
        or at_limit.get("stop_reason") != "context_limit"
        or at_limit.get("generated_token_ids") != []
    ):
        raise InferenceEvidenceError("exact-context boundary evidence is invalid")

    claims = data.get("claims")
    if not isinstance(claims, Mapping):
        raise InferenceEvidenceError("claims must be a mapping")
    forbidden_true = (
        "foreign_pretrained_weights_used",
        "instruction_or_alignment_behavior_added",
        "paid_compute_authorized_or_used",
        "candidate_or_stable_promotion",
        "windows_nvda_live_runtime_tested",
    )
    if any(claims.get(field) is not False for field in forbidden_true):
        raise InferenceEvidenceError("evidence truth boundary was widened")
