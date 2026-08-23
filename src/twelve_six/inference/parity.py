from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .contracts import InferenceBackend
from .loader import load_backend
from .sampling import greedy_token


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
    failures: tuple[ParityFailure, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "prompts_compared": self.prompts_compared,
            "steps_compared": self.steps_compared,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "failures": [asdict(failure) for failure in self.failures],
        }


def _validated_tolerance(atol: float, rtol: float) -> tuple[float, float]:
    atol = float(atol)
    rtol = float(rtol)
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and >= 0")
    if not math.isfinite(rtol) or rtol < 0:
        raise ValueError("rtol must be finite and >= 0")
    return atol, rtol


def _contract_failure(
    reference: InferenceBackend, candidate: InferenceBackend
) -> ParityFailure | None:
    if reference.max_context_tokens != candidate.max_context_tokens:
        return ParityFailure(
            prompt_index=-1,
            step_index=None,
            kind="context_window_mismatch",
            detail=(
                f"reference={reference.max_context_tokens} "
                f"candidate={candidate.max_context_tokens}"
            ),
        )
    if reference.eos_token_id != candidate.eos_token_id:
        return ParityFailure(
            prompt_index=-1,
            step_index=None,
            kind="eos_token_mismatch",
            detail=f"reference={reference.eos_token_id} candidate={candidate.eos_token_id}",
        )
    return None


def _compare_logits(
    reference_logits: list[float],
    candidate_logits: list[float],
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, float, float, str | None]:
    if len(reference_logits) != len(candidate_logits):
        return (
            False,
            0.0,
            0.0,
            f"logit size mismatch: reference={len(reference_logits)} "
            f"candidate={len(candidate_logits)}",
        )

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
            return (
                False,
                max_abs_error,
                max_rel_error,
                f"non-matching infinite logit at index {index}",
            )

        abs_error = abs(reference_value - candidate_value)
        rel_error = abs_error / max(abs(reference_value), 1e-30)
        max_abs_error = max(max_abs_error, abs_error)
        max_rel_error = max(max_rel_error, rel_error)
        if abs_error > atol + rtol * abs(reference_value):
            return (
                False,
                max_abs_error,
                max_rel_error,
                f"logit tolerance exceeded at index {index}: "
                f"abs_error={abs_error:.12g}",
            )

    return True, max_abs_error, max_rel_error, None


def compare_backends(
    reference: InferenceBackend,
    candidate: InferenceBackend,
    prompts: list[str] | tuple[str, ...],
    *,
    max_new_tokens: int = 8,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> ParityReport:
    if not prompts:
        raise ValueError("at least one prompt is required")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    atol, rtol = _validated_tolerance(atol, rtol)

    contract_failure = _contract_failure(reference, candidate)
    if contract_failure is not None:
        return ParityReport(0, 0, 0.0, 0.0, (contract_failure,))

    failures: list[ParityFailure] = []
    steps_compared = 0
    max_abs_error = 0.0
    max_rel_error = 0.0

    for prompt_index, prompt in enumerate(prompts):
        reference_prompt = reference.encode(prompt)
        candidate_prompt = candidate.encode(prompt)
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
        if not reference_prompt:
            failures.append(
                ParityFailure(prompt_index, None, "empty_prompt", "prompt encoded to zero tokens")
            )
            continue
        if len(reference_prompt) > reference.max_context_tokens:
            failures.append(
                ParityFailure(
                    prompt_index,
                    None,
                    "prompt_over_context",
                    f"prompt_tokens={len(reference_prompt)} "
                    f"context={reference.max_context_tokens}",
                )
            )
            continue

        generated: list[int] = []
        prompt_failed = False
        for step_index in range(max_new_tokens):
            input_ids = (*reference_prompt, *generated)
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

            reference_token = greedy_token(reference_logits)
            candidate_token = greedy_token(candidate_logits)
            if reference_token != candidate_token:
                failures.append(
                    ParityFailure(
                        prompt_index,
                        step_index,
                        "greedy_token_mismatch",
                        f"reference={reference_token} candidate={candidate_token}",
                    )
                )
                prompt_failed = True
                break

            generated.append(reference_token)
            if reference.eos_token_id is not None and reference_token == reference.eos_token_id:
                break

        if prompt_failed:
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
    except (ImportError, AttributeError, FileNotFoundError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False))
    else:
        verdict = "PASS" if report.passed else "FAIL"
        print(
            f"parity: {verdict} prompts={report.prompts_compared} "
            f"steps={report.steps_compared} max_abs_error={report.max_abs_error:.12g} "
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
