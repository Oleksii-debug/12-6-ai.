#!/usr/bin/env python3
"""Validate the frozen LEARN-345 learned-20M recipe authority."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from twelve_six.learned20m_recipe import (
    bind_terminal_authorities,
    blocked_template,
    validate_policy,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs/research/r01_learned20m_recipe_authority_v1.json"
MAX_AUTHORITY_JSON_BYTES = 8 * 1024 * 1024


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate object member")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number is not finite")
    return parsed


def _load_json(path: Path) -> Any:
    try:
        # A local authority file is untrusted until its identity and schema pass.
        # Bound the raw read before JSON parsing to avoid memory exhaustion.
        with path.open("rb") as source:
            raw = source.read(MAX_AUTHORITY_JSON_BYTES + 1)
        if len(raw) > MAX_AUTHORITY_JSON_BYTES:
            raise ValueError("authority JSON exceeds 8 MiB input limit")
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_parse_finite_float,
        )
    except RecursionError as exc:
        # Limit only untrusted JSON decoding; preserve genuine validator errors.
        raise ValueError("JSON nesting exceeds decoder limit") from exc


def _print_input_failure(label: str, exc: BaseException) -> int:
    print(
        json.dumps(
            {"status": "FAIL", "error": f"invalid {label}: {exc}"},
            sort_keys=True,
        )
    )
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--bindings",
        type=Path,
        default=None,
        help="Optional terminal authority bindings JSON. Omit for checked-in blocked template.",
    )
    parser.add_argument(
        "--trusted-authorities",
        type=Path,
        default=None,
        help=(
            "Out-of-packet role-bound tokenizer/D04/D05/D06 authority JSON. "
            "Required whenever --bindings is supplied."
        ),
    )
    parser.add_argument(
        "--expected-trusted-authorities-identity-sha256",
        default=None,
        help=(
            "Externally pinned SHA-256 identity of the trusted-authorities document. "
            "Required whenever --bindings is supplied and must not be derived from "
            "either JSON input by this tool."
        ),
    )
    args = parser.parse_args()

    try:
        policy = _load_json(args.policy)
    except (OSError, UnicodeError, ValueError) as exc:
        return _print_input_failure("policy JSON", exc)
    try:
        validate_policy(policy)
    except (TypeError, ValueError) as exc:
        return _print_input_failure("policy authority", exc)
    if args.bindings is None:
        if args.trusted_authorities is not None:
            return _print_input_failure(
                "authority arguments", ValueError("--trusted-authorities requires --bindings")
            )
        if args.expected_trusted_authorities_identity_sha256 is not None:
            return _print_input_failure(
                "authority arguments",
                ValueError("--expected-trusted-authorities-identity-sha256 requires --bindings"),
            )
        result = blocked_template(policy)
    else:
        if args.trusted_authorities is None:
            return _print_input_failure(
                "authority arguments",
                ValueError("--trusted-authorities is required with --bindings"),
            )
        if args.expected_trusted_authorities_identity_sha256 is None:
            return _print_input_failure(
                "authority arguments",
                ValueError(
                    "--expected-trusted-authorities-identity-sha256 is required with --bindings"
                ),
            )
        try:
            bindings = _load_json(args.bindings)
        except (OSError, UnicodeError, ValueError) as exc:
            return _print_input_failure("bindings JSON", exc)
        try:
            trusted_authorities = _load_json(args.trusted_authorities)
        except (OSError, UnicodeError, ValueError) as exc:
            return _print_input_failure("trusted-authorities JSON", exc)
        try:
            result = bind_terminal_authorities(
                policy,
                bindings,
                trusted_authorities=trusted_authorities,
                expected_trusted_authorities_identity_sha256=(
                    args.expected_trusted_authorities_identity_sha256
                ),
            )
        except (TypeError, ValueError) as exc:
            return _print_input_failure("terminal authority bindings", exc)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
