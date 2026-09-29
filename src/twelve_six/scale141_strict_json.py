"""Standards-strict JSON decoding for SCALE-141 recovery authority."""

from __future__ import annotations

import json
import math
from typing import Any


class Scale141StrictJsonError(ValueError):
    """Raised when trust-bearing SCALE-141 JSON is ambiguous or non-standard."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise Scale141StrictJsonError(f"duplicate JSON object member: {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> Any:
    raise Scale141StrictJsonError(f"non-finite JSON constant is not allowed: {value}")


def _finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise Scale141StrictJsonError("invalid JSON number") from exc
    if not math.isfinite(parsed):
        raise Scale141StrictJsonError("non-finite JSON number is not allowed")
    return parsed


def strict_json_loads(text: str) -> Any:
    """Decode one unambiguous standards-strict JSON value."""

    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except Scale141StrictJsonError:
        raise
    except json.JSONDecodeError as exc:
        raise Scale141StrictJsonError("invalid JSON") from exc
