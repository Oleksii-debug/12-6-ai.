#!/usr/bin/env python3
"""Bind or verify the learned-20M canonical byte-tokenizer decision authority."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from twelve_six.tokenization.decision_authority import (
    TokenizerDecisionError,
    bind_byte_baseline_decision,
    verify_byte_baseline_decision,
)


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non_finite_json_constant:{value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non_finite_json_number:{value}")
    # A lexically nonzero external number must not silently become zero.
    # Preserve genuine positive/negative JSON zero, including 0e-9999.
    significand = value.split("e", 1)[0].split("E", 1)[0]
    if parsed == 0.0 and any(digit in "123456789" for digit in significand):
        raise ValueError(f"nonzero_json_number_underflowed_to_zero:{value}")
    return parsed


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
            parse_float=_parse_finite_float,
        )
    except RecursionError as exc:
        raise ValueError("tokenizer input JSON nesting limit exceeded") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _serialize_report(value: dict[str, Any]) -> str:
    """One strict finite JSON contract for both file and stdout reports."""
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("tokenizer report is not strict finite JSON") from exc


def _write(path: Path, value: dict[str, Any]) -> None:
    """Publish only a new complete report; never replace existing authority."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    payload = _serialize_report(value).encode("utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            if handle.write(payload) != len(payload):
                raise OSError("incomplete tokenizer report staging write")
            handle.flush()
            os.fsync(handle.fileno())
        # Same-directory hard link atomically fails if the target already exists.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _emit_input_error(exc: Exception) -> None:
    print(
        json.dumps(
            {"contract_valid": False, "error": str(exc)},
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced-selection", type=Path, required=True)
    parser.add_argument("--split-application", type=Path, required=True)
    parser.add_argument("--expected-selection-identity-sha256", required=True)
    parser.add_argument("--expected-application-identity-sha256", required=True)
    parser.add_argument("--expected-retained-inventory-identity-sha256", required=True)
    parser.add_argument("--expected-decontamination-authority-sha256", required=True)
    parser.add_argument("--expected-dedup-authority-sha256", required=True)
    parser.add_argument("--expected-balance-policy-identity-sha256", required=True)
    parser.add_argument("--expected-balance-result-identity-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-report", type=Path)
    return parser.parse_args()


def _kwargs(args: argparse.Namespace) -> dict[str, str]:
    return {
        "expected_selection_identity_sha256": args.expected_selection_identity_sha256,
        "expected_application_identity_sha256": (
            args.expected_application_identity_sha256
        ),
        "expected_retained_inventory_identity_sha256": (
            args.expected_retained_inventory_identity_sha256
        ),
        "expected_decontamination_authority_sha256": (
            args.expected_decontamination_authority_sha256
        ),
        "expected_dedup_authority_sha256": args.expected_dedup_authority_sha256,
        "expected_balance_policy_identity_sha256": (
            args.expected_balance_policy_identity_sha256
        ),
        "expected_balance_result_identity_sha256": (
            args.expected_balance_result_identity_sha256
        ),
    }


def main() -> int:
    args = parse_args()
    try:
        selection = _load(args.balanced_selection)
        application = _load(args.split_application)
        verified_report = (
            _load(args.verify_report) if args.verify_report is not None else None
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _emit_input_error(exc)
        return 2

    try:
        if verified_report is not None:
            report = verified_report
            verify_byte_baseline_decision(report, selection, application, **_kwargs(args))
        else:
            report = bind_byte_baseline_decision(selection, application, **_kwargs(args))
    except TokenizerDecisionError as exc:
        _emit_input_error(exc)
        return 2

    if args.output is not None:
        try:
            _write(args.output, report)
        except (OSError, ValueError) as exc:
            _emit_input_error(exc)
            return 2
    else:
        try:
            serialized = _serialize_report(report)
        except ValueError as exc:
            _emit_input_error(exc)
            return 2
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
