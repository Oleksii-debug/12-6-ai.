"""Text-free, read-only diagnostic for incumbent V3 callable attestation failures.

This is NOT an alternate attester. Even when structural fields agree, a marshal
digest mismatch remains a hard failure in the incumbent verifier. The report
only distinguishes useful fault-investigation cases; it grants no data credit.
"""

from __future__ import annotations

import hashlib
import marshal
from types import CodeType
from typing import Any

_CODE_FIELDS = (
    "co_argcount",
    "co_posonlyargcount",
    "co_kwonlyargcount",
    "co_nlocals",
    "co_stacksize",
    "co_flags",
    "co_code",
    "co_consts",
    "co_names",
    "co_varnames",
    "co_freevars",
    "co_cellvars",
    "co_filename",
    "co_name",
    "co_qualname",
    "co_firstlineno",
    "co_linetable",
    "co_exceptiontable",
)
_MAX_NODES = 10_000
_MAX_DEPTH = 32
_MAX_DIFFERENCES = 24


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def compare_code_objects(live: CodeType, canonical: CodeType) -> dict[str, Any]:
    """Report bounded, non-payload code-object differences for independent triage.

    A structural match NEVER overrides the canonical marshal-based attestation.
    Both inputs must be code objects from a separately authenticated source.
    This function neither imports, executes, patches nor authorizes those inputs.
    """
    if type(live) is not CodeType or type(canonical) is not CodeType:
        raise TypeError("both attestation inputs must be exact Python code objects")

    differences: list[str] = []
    visited = 0
    limited = False

    def visit(left: Any, right: Any, path: str, depth: int) -> None:
        nonlocal visited, limited
        if limited:
            return
        visited += 1
        if visited > _MAX_NODES or depth > _MAX_DEPTH:
            limited = True
            return
        if len(differences) >= _MAX_DIFFERENCES:
            limited = True
            return
        if type(left) is not type(right):
            differences.append(f"{path}:type")
        elif type(left) is CodeType:
            for field in _CODE_FIELDS:
                visit(getattr(left, field), getattr(right, field), f"{path}.{field}", depth + 1)
                if limited:
                    break
        elif type(left) is tuple:
            if len(left) != len(right):
                differences.append(f"{path}:length")
            else:
                for index, (first, second) in enumerate(zip(left, right)):
                    visit(first, second, f"{path}[{index}]", depth + 1)
                    if limited:
                        break
        elif type(left) is frozenset:
            if len(left) > _MAX_NODES or len(right) > _MAX_NODES:
                limited = True
                return
            # Nested frozensets and code objects can trigger expensive recursive
            # hashing or marshal traversal beyond our depth/node budget.
            # Diagnose only known scalar members, otherwise fail closed.
            scalar = {type(None), type(Ellipsis), bool, int, float, complex, str, bytes}
            if any(type(item) not in scalar for item in left | right):
                limited = True
                return
            # Hash each scalar separately so outer marshal alias/reference flags
            # do not become the comparison criterion. No constant is printed.
            try:
                left_items = sorted((type(item).__name__, _sha256(marshal.dumps(item)))
                                    for item in left)
                right_items = sorted((type(item).__name__, _sha256(marshal.dumps(item)))
                                     for item in right)
            except (TypeError, ValueError):
                differences.append(f"{path}:unsupported-frozenset")
            else:
                if left_items != right_items:
                    differences.append(path)
        elif type(left) in {float, complex}:
            # Value equality misses signed zero; NaN is unequal even to itself.
            # Match the actual encoded scalar without printing its contents.
            if marshal.dumps(left) != marshal.dumps(right):
                differences.append(path)
        elif type(left) in {type(None), type(Ellipsis), bool, int, str, bytes}:
            if left != right:
                differences.append(path)
        else:
            # Unknown objects are NOT presumed equivalent.
            differences.append(f"{path}:unsupported-type")

    # Enforce structural depth/node limits BEFORE serializing either code tree.
    # Otherwise deeply nested inputs raise from marshal instead of returning a
    # bounded, explicitly incomplete diagnostic.
    visit(live, canonical, "code", 0)
    live_digest: str | None = None
    canonical_digest: str | None = None
    if not limited:
        try:
            live_digest = _sha256(marshal.dumps(live))
            canonical_digest = _sha256(marshal.dumps(canonical))
        except (ValueError, RecursionError, OverflowError):
            limited = True
            live_digest = canonical_digest = None
    marshal_equal = None if limited else live_digest == canonical_digest
    if limited:
        classification = "INCOMPLETE_DIAGNOSTIC"
    elif differences:
        classification = "STRUCTURAL_CODE_MISMATCH"
    elif not marshal_equal:
        classification = "SERIALIZATION_MISMATCH_UNRESOLVED"
    else:
        classification = "NO_CODE_MISMATCH_OBSERVED"
    return {
        "schema": "12-6.d03-v3-code-object-diagnostic.v1",
        "classification": classification,
        "marshal_equal": marshal_equal,
        "live_marshal_sha256": live_digest,
        "canonical_marshal_sha256": canonical_digest,
        "structural_fields_equal": not limited and not differences,
        "different_field_paths": differences,
        "diagnostic_limited": limited,
        "visited_nodes": visited,
        "attestation_override_allowed": False,
        "canonical_corpus_credit": 0,
        "training_authorized": False,
    }
