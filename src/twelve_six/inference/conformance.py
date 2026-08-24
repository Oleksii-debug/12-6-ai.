"""Single-backend conformance gate for the D07 inference protocol.

Parity compares two implementations. This module answers the earlier question: does
one candidate backend satisfy the minimum deterministic/mechanical contract well enough
to be meaningfully compared at all?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .contracts import GenerationConfig, InferenceBackend
from .generation import generate
from .loader import load_backend
from .sampling import greedy_token

SCHEMA_VERSION = "12-6.inference-backend-conformance.v1"
AUTHORITY = "BACKEND_INTERFACE_CONFORMANCE_NOT_PARITY_OR_PROMOTION"
_DEFAULT_LOADER = "twelve_six.inference.first_party:load_first_party_backend"
_DEFAULT_PROMPTS = ("12-6", "Base", "Україна")
_CHECK_NAMES = (
    "structural_contract",
    "deterministic_encode",
    "deterministic_decode",
    "deterministic_logits",
    "finite_logits",
    "stable_vocab_width",
    "token_ids_in_vocab",
    "eos_in_vocab",
    "generation_path",
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _require_token_ids(value: object, *, name: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must return list[int]")
    if not value:
        raise ValueError(f"{name} returned zero tokens")
    for index, token_id in enumerate(value):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"{name}[{index}] must be an integer token ID")
        if token_id < 0:
            raise ValueError(f"{name}[{index}] must be non-negative")
    return value


def _require_logits(value: object, *, name: str) -> list[float]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must return a one-dimensional numeric sequence")
    logits: list[float] = []
    if len(value) == 0:
        raise ValueError(f"{name} returned an empty vocabulary")
    for index, raw in enumerate(value):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"{name}[{index}] must be a real numeric logit")
        converted = float(raw)
        if not math.isfinite(converted):
            raise ValueError(f"{name}[{index}] must be finite")
        logits.append(converted)
    return logits


def _validate_repeat_atol(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("repeat_atol must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError("repeat_atol must be finite and >= 0")
    return converted


def _backend_type(backend: InferenceBackend) -> str:
    cls = type(backend)
    return f"{cls.__module__}.{cls.__qualname__}"


def _logit_fingerprint(logits: Sequence[float]) -> str:
    return _canonical_sha256([float(value).hex() for value in logits])


def _repeat_delta(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        raise ValueError("repeated next_token_logits calls changed vocabulary width")
    return max((abs(left - right) for left, right in zip(first, second, strict=True)), default=0.0)


def _probe_backend(
    backend: InferenceBackend,
    prompt: str,
    *,
    repeat_atol: float,
    expected_vocab_size: int | None,
) -> tuple[dict[str, object], int]:
    encoded_first = _require_token_ids(backend.encode(prompt), name="encode")
    encoded_second = _require_token_ids(backend.encode(prompt), name="encode")
    if encoded_first != encoded_second:
        raise ValueError("backend encode() is not deterministic for a repeated prompt")
    if len(encoded_first) > backend.max_context_tokens:
        raise ValueError("probe prompt exceeds backend max_context_tokens")

    decoded_first = backend.decode(encoded_first)
    decoded_second = backend.decode(encoded_first)
    if not isinstance(decoded_first, str) or not isinstance(decoded_second, str):
        raise TypeError("backend decode() must return str")
    if decoded_first != decoded_second:
        raise ValueError("backend decode() is not deterministic for repeated token IDs")

    logits_first = _require_logits(
        backend.next_token_logits(encoded_first),
        name="next_token_logits",
    )
    logits_second = _require_logits(
        backend.next_token_logits(encoded_first),
        name="next_token_logits",
    )
    max_repeat_delta = _repeat_delta(logits_first, logits_second)
    if max_repeat_delta > repeat_atol:
        raise ValueError(
            "backend next_token_logits() is not repeatable within tolerance: "
            f"max_abs_delta={max_repeat_delta} repeat_atol={repeat_atol}"
        )

    vocab_size = len(logits_first)
    if expected_vocab_size is not None and vocab_size != expected_vocab_size:
        raise ValueError(
            "backend next_token_logits() vocabulary width changed across probes: "
            f"expected={expected_vocab_size} observed={vocab_size}"
        )
    for token_id in encoded_first:
        if token_id >= vocab_size:
            raise ValueError(
                f"backend encode() returned token ID {token_id} outside inferred vocabulary "
                f"[0, {vocab_size})"
            )

    greedy_id = greedy_token(logits_first)
    if isinstance(greedy_id, bool) or not isinstance(greedy_id, int):
        raise TypeError("canonical greedy sampler returned a non-integer token ID")
    if not 0 <= greedy_id < vocab_size:
        raise ValueError("canonical greedy sampler returned an out-of-vocabulary token")
    greedy_text = backend.decode([greedy_id])
    if not isinstance(greedy_text, str):
        raise TypeError("backend decode() must return str for generated tokens")

    return (
        {
            "prompt_utf8_sha256": _text_sha256(prompt),
            "prompt_token_count": len(encoded_first),
            "prompt_token_ids_sha256": _canonical_sha256(encoded_first),
            "decoded_prompt_utf8_sha256": _text_sha256(decoded_first),
            "vocab_size": vocab_size,
            "logits_float64_hex_sha256": _logit_fingerprint(logits_first),
            "repeat_max_abs_delta": max_repeat_delta,
            "greedy_token_id": greedy_id,
            "greedy_text_utf8_sha256": _text_sha256(greedy_text),
        },
        vocab_size,
    )


def run_backend_conformance(
    backend: InferenceBackend,
    prompts: Sequence[str] = _DEFAULT_PROMPTS,
    *,
    repeat_atol: float = 0.0,
) -> dict[str, Any]:
    """Validate one backend before parity or serving evidence is attempted."""
    repeat_atol = _validate_repeat_atol(repeat_atol)
    max_context_tokens = _require_positive_int(
        backend.max_context_tokens,
        name="backend max_context_tokens",
    )
    eos_token_id = backend.eos_token_id
    if eos_token_id is not None:
        if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
            raise TypeError("backend eos_token_id must be int or None")
        if eos_token_id < 0:
            raise ValueError("backend eos_token_id must be non-negative when set")

    prompt_values = tuple(prompts)
    if not prompt_values:
        raise ValueError("at least one conformance prompt is required")
    if any(not isinstance(prompt, str) for prompt in prompt_values):
        raise TypeError("conformance prompts must be strings")
    if any(prompt == "" for prompt in prompt_values):
        raise ValueError("conformance prompts must not be empty")

    probes: list[dict[str, object]] = []
    inferred_vocab_size: int | None = None
    generation_probe_index: int | None = None
    for index, prompt in enumerate(prompt_values):
        probe, vocab_size = _probe_backend(
            backend,
            prompt,
            repeat_atol=repeat_atol,
            expected_vocab_size=inferred_vocab_size,
        )
        inferred_vocab_size = vocab_size
        probes.append(probe)
        if generation_probe_index is None and probe["prompt_token_count"] < max_context_tokens:
            generation_probe_index = index

    assert inferred_vocab_size is not None
    if eos_token_id is not None and eos_token_id >= inferred_vocab_size:
        raise ValueError(
            "backend eos_token_id is outside inferred logits vocabulary: "
            f"eos={eos_token_id} vocab={inferred_vocab_size}"
        )

    generation: dict[str, object]
    if generation_probe_index is None:
        generation = {
            "executed": False,
            "reason": "all_probes_fill_context",
        }
    else:
        generation_result = generate(
            backend,
            prompt_values[generation_probe_index],
            GenerationConfig(max_new_tokens=1, sample=False),
        )
        if len(generation_result.generated_token_ids) != 1:
            raise ValueError("canonical generation path did not produce exactly one probe token")
        generated_id = generation_result.generated_token_ids[0]
        if not 0 <= generated_id < inferred_vocab_size:
            raise ValueError("canonical generation path produced out-of-vocabulary token")
        generation = {
            "executed": True,
            "probe_index": generation_probe_index,
            "generated_token_id": generated_id,
            "text_utf8_sha256": _text_sha256(generation_result.text),
            "stop_reason": generation_result.stop_reason,
        }

    checks = {name: True for name in _CHECK_NAMES}
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "backend_type": _backend_type(backend),
        "max_context_tokens": max_context_tokens,
        "eos_token_id": eos_token_id,
        "inferred_vocab_size": inferred_vocab_size,
        "repeat_atol": repeat_atol,
        "probe_count": len(probes),
        "probes": probes,
        "generation_probe": generation,
        "checks": checks,
        "conformance_pass": True,
        "parity_proven": False,
        "checkpoint_identity_proven": False,
        "promotion_authority": False,
    }
    report["report_sha256"] = _canonical_sha256(report)
    validate_conformance_report(report)
    return report


def validate_conformance_report(report: dict[str, Any]) -> None:
    """Validate a serialized conformance report without trusting copied PASS fields."""
    expected_keys = {
        "schema_version",
        "authority",
        "backend_type",
        "max_context_tokens",
        "eos_token_id",
        "inferred_vocab_size",
        "repeat_atol",
        "probe_count",
        "probes",
        "generation_probe",
        "checks",
        "conformance_pass",
        "parity_proven",
        "checkpoint_identity_proven",
        "promotion_authority",
        "report_sha256",
    }
    if set(report) != expected_keys:
        raise ValueError("conformance report top-level schema mismatch")
    if report["schema_version"] != SCHEMA_VERSION or report["authority"] != AUTHORITY:
        raise ValueError("conformance report authority/schema mismatch")
    _require_positive_int(report["max_context_tokens"], name="report max_context_tokens")
    _require_positive_int(report["inferred_vocab_size"], name="report inferred_vocab_size")
    repeat_atol = _validate_repeat_atol(report["repeat_atol"])
    if repeat_atol != report["repeat_atol"]:
        raise ValueError("conformance report repeat_atol must be canonical")
    probes = report["probes"]
    if not isinstance(probes, list) or not probes:
        raise ValueError("conformance report must contain probe evidence")
    if report["probe_count"] != len(probes):
        raise ValueError("conformance report probe_count mismatch")
    checks = report["checks"]
    if not isinstance(checks, dict) or tuple(checks) != _CHECK_NAMES:
        raise ValueError("conformance report check schema mismatch")
    if not all(value is True for value in checks.values()):
        raise ValueError("conformance report contains a non-passing check")
    if report["conformance_pass"] is not True:
        raise ValueError("conformance report does not pass")
    if report["parity_proven"] is not False:
        raise ValueError("single-backend conformance may not claim parity")
    if report["checkpoint_identity_proven"] is not False:
        raise ValueError("generic conformance may not claim checkpoint identity")
    if report["promotion_authority"] is not False:
        raise ValueError("conformance may not grant promotion authority")
    expected_hash = report["report_sha256"]
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("conformance report hash is invalid")
    unhashed = dict(report)
    unhashed.pop("report_sha256")
    if expected_hash != _canonical_sha256(unhashed):
        raise ValueError("conformance report hash mismatch")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate one D07 InferenceBackend before parity/serving evidence",
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--backend-loader", default=_DEFAULT_LOADER)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--repeat-atol", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    backend = load_backend(args.backend_loader, args.checkpoint)
    report = run_backend_conformance(
        backend,
        tuple(args.prompts) if args.prompts else _DEFAULT_PROMPTS,
        repeat_atol=args.repeat_atol,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "PASS "
            f"backend={report['backend_type']} "
            f"vocab={report['inferred_vocab_size']} "
            f"context={report['max_context_tokens']} "
            f"probes={report['probe_count']} "
            f"report_sha256={report['report_sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
