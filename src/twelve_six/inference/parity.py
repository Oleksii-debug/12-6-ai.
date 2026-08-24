"""Deterministic backend parity evidence for canonical and converted inference paths."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
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
        return not self.failures

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


def _validated_tolerance(atol: float, rtol: float) -> tuple[float, float]:
    values: list[float] = []
    for name, value in (("atol", atol), ("rtol", rtol)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real number")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{name} must be finite and >= 0")
        values.append(number)
    return values[0], values[1]


def _validated_max_new_tokens(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("max_new_tokens must be an integer")
    if value < 0:
        raise ValueError("max_new_tokens must be >= 0")
    return value


def _backend_contract_failure(
    reference: InferenceBackend,
    candidate: InferenceBackend,
) -> ParityFailure | None:
    for side, backend in (("reference", reference), ("candidate", candidate)):
        context = backend.max_context_tokens
        if not isinstance(context, int) or isinstance(context, bool) or context < 1:
            return ParityFailure(
                -1,
                None,
                f"invalid_{side}_context_window",
                f"{side} max_context_tokens must be a positive integer",
            )
        eos = backend.eos_token_id
        if eos is not None and (
            not isinstance(eos, int) or isinstance(eos, bool) or eos < 0
        ):
            return ParityFailure(
                -1,
                None,
                f"invalid_{side}_eos_token",
                f"{side} eos_token_id must be a non-negative integer or None",
            )

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


def _validated_prompt_tokens(
    value: Any,
    *,
    side: str,
    prompt_index: int,
) -> tuple[list[int] | None, ParityFailure | None]:
    if not isinstance(value, list):
        return None, ParityFailure(
            prompt_index,
            None,
            f"invalid_{side}_prompt_tokens",
            f"{side} encode must return list[int]",
        )
    if not value:
        return None, ParityFailure(
            prompt_index,
            None,
            f"empty_{side}_prompt",
            f"{side} prompt encoded to zero tokens",
        )
    if any(
        not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
        for token_id in value
    ):
        return None, ParityFailure(
            prompt_index,
            None,
            f"invalid_{side}_prompt_tokens",
            f"{side} encode returned a non-integer or negative token ID",
        )
    return value, None


def _validated_logits(
    raw_logits: Any,
    *,
    side: str,
    prompt_index: int,
    step_index: int,
) -> tuple[list[float] | None, ParityFailure | None]:
    try:
        raw_values = list(raw_logits)
    except (TypeError, ValueError, OverflowError):
        return None, ParityFailure(
            prompt_index,
            step_index,
            f"invalid_{side}_logits",
            f"{side} logits are not an iterable numeric sequence",
        )
    if not raw_values:
        return None, ParityFailure(
            prompt_index,
            step_index,
            f"invalid_{side}_logits",
            f"{side} logits must not be empty",
        )

    values: list[float] = []
    finite_count = 0
    for index, value in enumerate(raw_values):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None, ParityFailure(
                prompt_index,
                step_index,
                f"invalid_{side}_logits",
                f"{side} logit at index {index} is not a real number",
            )
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None, ParityFailure(
                prompt_index,
                step_index,
                f"invalid_{side}_logits",
                f"{side} logit at index {index} cannot be represented as float",
            )
        if math.isnan(number) or number == math.inf:
            return None, ParityFailure(
                prompt_index,
                step_index,
                f"invalid_{side}_logits",
                f"{side} logit at index {index} is NaN or +inf",
            )
        if math.isfinite(number):
            finite_count += 1
        values.append(number)

    if finite_count == 0:
        return None, ParityFailure(
            prompt_index,
            step_index,
            f"invalid_{side}_logits",
            f"{side} logits contain no finite candidate",
        )
    return values, None


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

    max_abs_error = 0.0
    max_rel_error = 0.0
    for index, (reference_value, candidate_value) in enumerate(
        zip(reference_logits, candidate_logits, strict=True)
    ):
        # +inf and NaN are rejected before this helper; matching -inf is the
        # only non-finite equality allowed and represents a masked candidate.
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


def _backend_call_failure(
    *,
    side: str,
    operation: str,
    prompt_index: int,
    step_index: int | None,
    exc: Exception,
) -> ParityFailure:
    # Exception messages can echo prompts or filesystem paths. Evidence keeps
    # only the exception class, which is enough to fail closed without leaking input.
    return ParityFailure(
        prompt_index,
        step_index,
        f"{side}_{operation}_error",
        f"{side} {operation} raised {type(exc).__name__}",
    )


def compare_backends(
    reference: InferenceBackend,
    candidate: InferenceBackend,
    prompts: list[str] | tuple[str, ...],
    *,
    max_new_tokens: int = 8,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> ParityReport:
    if not isinstance(prompts, (list, tuple)) or not prompts:
        raise ValueError("at least one prompt is required")
    if any(not isinstance(prompt, str) for prompt in prompts):
        raise TypeError("prompts must contain only strings")
    max_new_tokens = _validated_max_new_tokens(max_new_tokens)
    atol, rtol = _validated_tolerance(atol, rtol)

    contract_failure = _backend_contract_failure(reference, candidate)
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
        try:
            raw_reference_prompt = reference.encode(prompt)
        except Exception as exc:  # noqa: BLE001 - a backend exception is parity failure evidence
            failures.append(
                _backend_call_failure(
                    side="reference",
                    operation="encode",
                    prompt_index=prompt_index,
                    step_index=None,
                    exc=exc,
                )
            )
            continue
        try:
            raw_candidate_prompt = candidate.encode(prompt)
        except Exception as exc:  # noqa: BLE001 - a backend exception is parity failure evidence
            failures.append(
                _backend_call_failure(
                    side="candidate",
                    operation="encode",
                    prompt_index=prompt_index,
                    step_index=None,
                    exc=exc,
                )
            )
            continue

        reference_prompt, prompt_failure = _validated_prompt_tokens(
            raw_reference_prompt,
            side="reference",
            prompt_index=prompt_index,
        )
        if prompt_failure is not None:
            failures.append(prompt_failure)
            continue
        candidate_prompt, prompt_failure = _validated_prompt_tokens(
            raw_candidate_prompt,
            side="candidate",
            prompt_index=prompt_index,
        )
        if prompt_failure is not None:
            failures.append(prompt_failure)
            continue
        assert reference_prompt is not None and candidate_prompt is not None

        if reference_prompt != candidate_prompt:
            failures.append(
                ParityFailure(
                    prompt_index,
                    None,
                    "encoded_prompt_mismatch",
                    "reference and candidate token IDs differ",
                )
            )
            continue
        if len(reference_prompt) > reference.max_context_tokens:
            detail = (
                f"prompt_tokens={len(reference_prompt)} "
                f"context={reference.max_context_tokens}"
            )
            failures.append(ParityFailure(prompt_index, None, "prompt_over_context", detail))
            continue

        generated: list[int] = []
        prompt_failed = False
        for step_index in range(max_new_tokens):
            input_ids = (*reference_prompt, *generated)
            if len(input_ids) >= reference.max_context_tokens:
                break

            try:
                raw_reference_logits = reference.next_token_logits(input_ids)
            except Exception as exc:  # noqa: BLE001 - backend failure must not become parity PASS
                failures.append(
                    _backend_call_failure(
                        side="reference",
                        operation="next_token_logits",
                        prompt_index=prompt_index,
                        step_index=step_index,
                        exc=exc,
                    )
                )
                prompt_failed = True
                break
            try:
                raw_candidate_logits = candidate.next_token_logits(input_ids)
            except Exception as exc:  # noqa: BLE001 - backend failure must not become parity PASS
                failures.append(
                    _backend_call_failure(
                        side="candidate",
                        operation="next_token_logits",
                        prompt_index=prompt_index,
                        step_index=step_index,
                        exc=exc,
                    )
                )
                prompt_failed = True
                break

            reference_logits, logit_failure = _validated_logits(
                raw_reference_logits,
                side="reference",
                prompt_index=prompt_index,
                step_index=step_index,
            )
            if logit_failure is not None:
                failures.append(logit_failure)
                prompt_failed = True
                break
            candidate_logits, logit_failure = _validated_logits(
                raw_candidate_logits,
                side="candidate",
                prompt_index=prompt_index,
                step_index=step_index,
            )
            if logit_failure is not None:
                failures.append(logit_failure)
                prompt_failed = True
                break
            assert reference_logits is not None and candidate_logits is not None

            if len(reference_logits) != len(candidate_logits):
                failures.append(
                    ParityFailure(
                        prompt_index,
                        step_index,
                        "logit_mismatch",
                        (
                            "logit size mismatch: "
                            f"reference={len(reference_logits)} candidate={len(candidate_logits)}"
                        ),
                    )
                )
                prompt_failed = True
                break

            vocab_size = len(reference_logits)
            invalid_input = next((token_id for token_id in input_ids if token_id >= vocab_size), None)
            if invalid_input is not None:
                failures.append(
                    ParityFailure(
                        prompt_index,
                        step_index,
                        "input_token_out_of_range",
                        f"input token ID {invalid_input} is outside logit vocabulary {vocab_size}",
                    )
                )
                prompt_failed = True
                break
            eos = reference.eos_token_id
            if eos is not None and eos >= vocab_size:
                failures.append(
                    ParityFailure(
                        prompt_index,
                        step_index,
                        "eos_token_out_of_range",
                        f"eos token ID {eos} is outside logit vocabulary {vocab_size}",
                    )
                )
                prompt_failed = True
                break

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
            if eos is not None and reference_token == eos:
                break

        if prompt_failed:
            continue
        try:
            reference_text = reference.decode(generated)
        except Exception as exc:  # noqa: BLE001 - backend failure must not become parity PASS
            failures.append(
                _backend_call_failure(
                    side="reference",
                    operation="decode",
                    prompt_index=prompt_index,
                    step_index=None,
                    exc=exc,
                )
            )
            continue
        try:
            candidate_text = candidate.decode(generated)
        except Exception as exc:  # noqa: BLE001 - backend failure must not become parity PASS
            failures.append(
                _backend_call_failure(
                    side="candidate",
                    operation="decode",
                    prompt_index=prompt_index,
                    step_index=None,
                    exc=exc,
                )
            )
            continue
        if not isinstance(reference_text, str) or not isinstance(candidate_text, str):
            failures.append(
                ParityFailure(
                    prompt_index,
                    None,
                    "invalid_decoded_text",
                    "reference and candidate decode outputs must both be strings",
                )
            )
            continue
        if reference_text != candidate_text:
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
