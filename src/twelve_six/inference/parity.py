"""Deterministic backend parity evidence for canonical and converted inference paths."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any

from .contracts import InferenceBackend
from .loader import load_backend
from .sampling import greedy_token

PARITY_SCHEMA = "12-6.inference-parity.v1"


@dataclass(frozen=True, slots=True)
class ParityFailure:
    prompt_index: int
    step_index: int | None
    kind: str
    detail: str


@dataclass(frozen=True, slots=True)
class ParityReport:
    prompts_compared: int
    steps_compared: int
    max_abs_error: float
    max_rel_error: float
    max_new_tokens: int
    atol: float
    rtol: float
    failures: tuple[ParityFailure, ...]

    @property
    def passed(self) -> bool:
        """Require both zero failures and non-vacuous numerical comparison."""

        return (
            not self.failures
            and self.prompts_compared > 0
            and self.steps_compared > 0
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": PARITY_SCHEMA,
            "passed": self.passed,
            "prompts_compared": self.prompts_compared,
            "steps_compared": self.steps_compared,
            "max_new_tokens": self.max_new_tokens,
            "atol": self.atol,
            "rtol": self.rtol,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "failures": [asdict(failure) for failure in self.failures],
        }


def _validated_tolerance(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not bool or coerced text")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return converted


def _validated_max_new_tokens(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("max_new_tokens must be a positive integer")
    if value <= 0:
        raise ValueError("max_new_tokens must be > 0 for parity evidence")
    return value


def _validated_prompts(prompts: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(prompts, (list, tuple)):
        raise TypeError("prompts must be a list or tuple of strings")
    if not prompts:
        raise ValueError("at least one prompt is required")
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, str):
            raise TypeError(f"prompt at index {index} must be a string")
    return tuple(prompts)


def _backend_contract_failure(
    backend: InferenceBackend,
    *,
    role: str,
) -> ParityFailure | None:
    context = backend.max_context_tokens
    if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
        return ParityFailure(
            -1,
            None,
            f"invalid_{role}_context_window",
            f"{role} max_context_tokens must be a positive integer",
        )

    eos_token_id = backend.eos_token_id
    if eos_token_id is not None and (
        not isinstance(eos_token_id, int)
        or isinstance(eos_token_id, bool)
        or eos_token_id < 0
    ):
        return ParityFailure(
            -1,
            None,
            f"invalid_{role}_eos_token",
            f"{role} eos_token_id must be None or a non-negative integer",
        )
    return None


def _contract_failure(
    reference: InferenceBackend,
    candidate: InferenceBackend,
) -> ParityFailure | None:
    reference_failure = _backend_contract_failure(reference, role="reference")
    if reference_failure is not None:
        return reference_failure
    candidate_failure = _backend_contract_failure(candidate, role="candidate")
    if candidate_failure is not None:
        return candidate_failure

    if reference.max_context_tokens != candidate.max_context_tokens:
        detail = (
            f"reference={reference.max_context_tokens} "
            f"candidate={candidate.max_context_tokens}"
        )
        return ParityFailure(-1, None, "context_window_mismatch", detail)
    if reference.eos_token_id != candidate.eos_token_id:
        detail = f"reference={reference.eos_token_id} candidate={candidate.eos_token_id}"
        return ParityFailure(-1, None, "eos_token_mismatch", detail)
    return None


def _validated_token_ids(
    token_ids: Any,
    *,
    role: str,
) -> tuple[int, ...] | ParityFailure:
    if not isinstance(token_ids, list):
        return ParityFailure(
            -1,
            None,
            f"invalid_{role}_encoded_prompt",
            f"{role} encode() must return list[int]",
        )
    for token_index, token_id in enumerate(token_ids):
        if (
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
        ):
            return ParityFailure(
                -1,
                None,
                f"invalid_{role}_encoded_prompt",
                f"{role} token at index {token_index} must be a non-negative integer",
            )
    return tuple(token_ids)


def _compare_logits(
    reference_logits: list[float],
    candidate_logits: list[float],
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, float, float, str | None]:
    if len(reference_logits) != len(candidate_logits):
        detail = (
            f"logit size mismatch: reference={len(reference_logits)} "
            f"candidate={len(candidate_logits)}"
        )
        return False, 0.0, 0.0, detail
    if not reference_logits:
        return False, 0.0, 0.0, "logit vectors must not be empty"

    max_abs_error = 0.0
    max_rel_error = 0.0
    for index, (reference_value, candidate_value) in enumerate(
        zip(reference_logits, candidate_logits, strict=True)
    ):
        if math.isnan(reference_value) or math.isnan(candidate_value):
            return False, max_abs_error, max_rel_error, f"NaN logit at index {index}"
        if reference_value == candidate_value:
            continue
        if not math.isfinite(reference_value) or not math.isfinite(candidate_value):
            detail = f"non-matching infinite logit at index {index}"
            return False, max_abs_error, max_rel_error, detail

        abs_error = abs(reference_value - candidate_value)
        rel_error = abs_error / max(abs(reference_value), 1e-30)
        max_abs_error = max(max_abs_error, abs_error)
        max_rel_error = max(max_rel_error, rel_error)
        if abs_error > atol + rtol * abs(reference_value):
            detail = f"logit tolerance exceeded at index {index}: abs_error={abs_error:.12g}"
            return False, max_abs_error, max_rel_error, detail

    return True, max_abs_error, max_rel_error, None


def _runtime_vocab_failure(
    reference: InferenceBackend,
    candidate: InferenceBackend,
    *,
    input_ids: tuple[int, ...],
    vocab_size: int,
    prompt_index: int,
    step_index: int,
) -> ParityFailure | None:
    for token_index, token_id in enumerate(input_ids):
        if token_id >= vocab_size:
            return ParityFailure(
                prompt_index,
                step_index,
                "input_token_out_of_vocab",
                (
                    f"input token at index {token_index} is {token_id}; "
                    f"logit vocabulary size is {vocab_size}"
                ),
            )

    eos_token_id = reference.eos_token_id
    if eos_token_id is not None and eos_token_id >= vocab_size:
        return ParityFailure(
            prompt_index,
            step_index,
            "eos_token_out_of_vocab",
            f"shared eos_token_id={eos_token_id} is outside vocabulary size {vocab_size}",
        )
    if candidate.eos_token_id != eos_token_id:
        return ParityFailure(
            prompt_index,
            step_index,
            "eos_token_mismatch",
            f"reference={eos_token_id} candidate={candidate.eos_token_id}",
        )
    return None


def compare_backends(
    reference: InferenceBackend,
    candidate: InferenceBackend,
    prompts: list[str] | tuple[str, ...],
    *,
    max_new_tokens: int = 8,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> ParityReport:
    prompts = _validated_prompts(prompts)
    max_new_tokens = _validated_max_new_tokens(max_new_tokens)
    atol = _validated_tolerance("atol", atol)
    rtol = _validated_tolerance("rtol", rtol)

    contract_failure = _contract_failure(reference, candidate)
    if contract_failure is not None:
        return ParityReport(
            prompts_compared=0,
            steps_compared=0,
            max_abs_error=0.0,
            max_rel_error=0.0,
            max_new_tokens=max_new_tokens,
            atol=atol,
            rtol=rtol,
            failures=(contract_failure,),
        )

    failures: list[ParityFailure] = []
    steps_compared = 0
    max_abs_error = 0.0
    max_rel_error = 0.0

    for prompt_index, prompt in enumerate(prompts):
        reference_encoded = _validated_token_ids(reference.encode(prompt), role="reference")
        if isinstance(reference_encoded, ParityFailure):
            failures.append(
                ParityFailure(
                    prompt_index,
                    None,
                    reference_encoded.kind,
                    reference_encoded.detail,
                )
            )
            continue
        candidate_encoded = _validated_token_ids(candidate.encode(prompt), role="candidate")
        if isinstance(candidate_encoded, ParityFailure):
            failures.append(
                ParityFailure(
                    prompt_index,
                    None,
                    candidate_encoded.kind,
                    candidate_encoded.detail,
                )
            )
            continue

        if reference_encoded != candidate_encoded:
            failures.append(
                ParityFailure(
                    prompt_index,
                    None,
                    "encoded_prompt_mismatch",
                    "reference and candidate token IDs differ",
                )
            )
            continue
        if not reference_encoded:
            failures.append(
                ParityFailure(prompt_index, None, "empty_prompt", "prompt encoded to zero tokens")
            )
            continue
        if len(reference_encoded) > reference.max_context_tokens:
            detail = (
                f"prompt_tokens={len(reference_encoded)} "
                f"context={reference.max_context_tokens}"
            )
            failures.append(ParityFailure(prompt_index, None, "prompt_over_context", detail))
            continue

        generated: list[int] = []
        prompt_failed = False
        prompt_steps_before = steps_compared
        for step_index in range(max_new_tokens):
            input_ids = (*reference_encoded, *generated)
            if len(input_ids) >= reference.max_context_tokens:
                break

            reference_logits = [float(value) for value in reference.next_token_logits(input_ids)]
            candidate_logits = [float(value) for value in candidate.next_token_logits(input_ids)]
            steps_compared += 1
            ok, step_abs, step_rel, detail = _compare_logits(
                reference_logits,
                candidate_logits,
                atol=atol,
                rtol=rtol,
            )
            max_abs_error = max(max_abs_error, step_abs)
            max_rel_error = max(max_rel_error, step_rel)
            if not ok:
                failures.append(
                    ParityFailure(
                        prompt_index,
                        step_index,
                        "logit_mismatch",
                        detail or "logit mismatch",
                    )
                )
                prompt_failed = True
                break

            runtime_failure = _runtime_vocab_failure(
                reference,
                candidate,
                input_ids=input_ids,
                vocab_size=len(reference_logits),
                prompt_index=prompt_index,
                step_index=step_index,
            )
            if runtime_failure is not None:
                failures.append(runtime_failure)
                prompt_failed = True
                break

            reference_token = greedy_token(reference_logits)
            candidate_token = greedy_token(candidate_logits)
            if reference_token != candidate_token:
                detail = f"reference={reference_token} candidate={candidate_token}"
                failures.append(
                    ParityFailure(prompt_index, step_index, "greedy_token_mismatch", detail)
                )
                prompt_failed = True
                break

            generated.append(reference_token)
            if reference.eos_token_id is not None and reference_token == reference.eos_token_id:
                break

        if prompt_failed:
            continue
        if steps_compared == prompt_steps_before:
            failures.append(
                ParityFailure(
                    prompt_index,
                    None,
                    "no_logit_steps",
                    (
                        "prompt left no context capacity for a numerical parity step; "
                        "parity evidence cannot pass vacuously"
                    ),
                )
            )
            continue
        if reference.decode(generated) != candidate.decode(generated):
            failures.append(
                ParityFailure(
                    prompt_index,
                    None,
                    "decoded_text_mismatch",
                    "reference and candidate decode outputs differ",
                )
            )

    return ParityReport(
        prompts_compared=len(prompts),
        steps_compared=steps_compared,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        max_new_tokens=max_new_tokens,
        atol=atol,
        rtol=rtol,
        failures=tuple(failures),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m twelve_six.inference.parity",
        description="Compare canonical and alternative 12-6 inference backends.",
    )
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-backend-loader", required=True, metavar="MODULE:CALLABLE")
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-backend-loader", required=True, metavar="MODULE:CALLABLE")
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        reference = load_backend(args.reference_backend_loader, args.reference_checkpoint)
        candidate = load_backend(args.candidate_backend_loader, args.candidate_checkpoint)
        report = compare_backends(
            reference,
            candidate,
            tuple(args.prompt),
            max_new_tokens=args.max_new_tokens,
            atol=args.atol,
            rtol=args.rtol,
        )
    except (ImportError, AttributeError, FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False))
    else:
        verdict = "PASS" if report.passed else "FAIL"
        print(
            f"parity: {verdict} prompts={report.prompts_compared} "
            f"steps={report.steps_compared} max_new_tokens={report.max_new_tokens} "
            f"atol={report.atol:.12g} rtol={report.rtol:.12g} "
            f"max_abs_error={report.max_abs_error:.12g} "
            f"max_rel_error={report.max_rel_error:.12g}"
        )
        for failure in report.failures:
            print(
                "failure: "
                f"prompt_index={failure.prompt_index} step_index={failure.step_index} "
                f"kind={failure.kind} detail={failure.detail}"
            )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
