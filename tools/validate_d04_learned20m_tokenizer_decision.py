#!/usr/bin/env python3
"""Bind or verify the learned-20M canonical byte-tokenizer decision authority."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from twelve_six.tokenization.decision_authority import (
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
    return parsed


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_pairs_without_duplicates,
        parse_constant=_reject_constant,
        parse_float=_parse_finite_float,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


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

    if verified_report is not None:
        report = verified_report
        verify_byte_baseline_decision(report, selection, application, **_kwargs(args))
    else:
        report = bind_byte_baseline_decision(selection, application, **_kwargs(args))

    if args.output is not None:
        _write(args.output, report)
    else:
        print(
            json.dumps(
                report,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
