"""Authenticate the externally pinned learned-20M readiness trust bundle."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from typing import Any

from twelve_six.learned20m_readiness import trusted_readiness_inputs
from twelve_six.preoptimizer_authority import validate_preoptimizer_authorities

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_V1_KEYS = {
    "schema_version",
    "scientific_authorities",
    "verified_authorization_refs",
}
_V2_KEYS = _V1_KEYS | {"portable_execution"}
_V3_KEYS = _V2_KEYS | {"preoptimizer_authorities"}


def _readiness_projection(bindings: Any) -> dict[str, Any] | None:
    """Project v1/v2/v3 bundles onto the incumbent readiness trust-input schema."""
    if not isinstance(bindings, dict):
        return None
    keys = set(bindings)
    version = bindings.get("schema_version")
    if version == 1 and keys == _V1_KEYS:
        return bindings
    if version == 2 and keys == _V2_KEYS and isinstance(bindings.get("portable_execution"), dict):
        return {
            "schema_version": 1,
            "scientific_authorities": bindings.get("scientific_authorities"),
            "verified_authorization_refs": bindings.get("verified_authorization_refs"),
        }
    if (
        version == 3
        and keys == _V3_KEYS
        and isinstance(bindings.get("portable_execution"), dict)
        and isinstance(bindings.get("preoptimizer_authorities"), dict)
    ):
        return {
            "schema_version": 1,
            "scientific_authorities": bindings.get("scientific_authorities"),
            "verified_authorization_refs": bindings.get("verified_authorization_refs"),
        }
    return None


def trusted_readiness_bundle_sha256(bindings: Any) -> str | None:
    """Return the canonical identity of a structurally valid trusted bundle.

    Version 2 extends the externally pinned root with ``portable_execution``.
    Version 3 additionally carries the independently sourced pre-optimizer D10,
    loss-bearing-content, tokenizer-decision and measured-resource projection.
    None of those objects are trusted merely because they exist: the full bundle
    bytes must still match an independently supplied expected SHA-256.
    """
    readiness_projection = _readiness_projection(bindings)
    if readiness_projection is None or trusted_readiness_inputs(readiness_projection) is None:
        return None
    if isinstance(bindings, dict) and bindings.get("schema_version") == 3:
        if validate_preoptimizer_authorities(bindings.get("preoptimizer_authorities")):
            return None
    try:
        payload = json.dumps(
            bindings,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        return None
    return hashlib.sha256(payload).hexdigest()


def authenticated_trusted_launch_bundle(
    bindings: Any,
    *,
    expected_identity_sha256: Any,
) -> tuple[set[str], set[str], dict[str, Any] | None, dict[str, Any] | None] | None:
    """Resolve readiness, execution and pre-optimizer inputs under one external root."""
    if (
        not isinstance(expected_identity_sha256, str)
        or _SHA256_RE.fullmatch(expected_identity_sha256) is None
    ):
        return None

    observed = trusted_readiness_bundle_sha256(bindings)
    if observed is None or not hmac.compare_digest(observed, expected_identity_sha256):
        return None

    readiness_projection = _readiness_projection(bindings)
    if readiness_projection is None:
        return None
    resolved = trusted_readiness_inputs(readiness_projection)
    if resolved is None:
        return None
    scientific, refs = resolved

    portable_execution = None
    preoptimizer_authorities = None
    if isinstance(bindings, dict) and bindings.get("schema_version") in {2, 3}:
        value = bindings.get("portable_execution")
        if not isinstance(value, dict):
            return None
        portable_execution = copy.deepcopy(value)
    if isinstance(bindings, dict) and bindings.get("schema_version") == 3:
        value = bindings.get("preoptimizer_authorities")
        if not isinstance(value, dict) or validate_preoptimizer_authorities(value):
            return None
        preoptimizer_authorities = copy.deepcopy(value)

    return scientific, refs, portable_execution, preoptimizer_authorities


def authenticated_trusted_readiness_bundle(
    bindings: Any,
    *,
    expected_identity_sha256: Any,
) -> tuple[set[str], set[str], dict[str, Any] | None] | None:
    """Compatibility resolver for readiness + portable-execution consumers."""
    resolved = authenticated_trusted_launch_bundle(
        bindings,
        expected_identity_sha256=expected_identity_sha256,
    )
    if resolved is None:
        return None
    scientific, refs, portable_execution, _ = resolved
    return scientific, refs, portable_execution


def authenticated_trusted_readiness_inputs(
    bindings: Any,
    *,
    expected_identity_sha256: Any,
) -> tuple[set[str], set[str]] | None:
    """Compatibility resolver for readiness-only consumers of the same root."""
    resolved = authenticated_trusted_launch_bundle(
        bindings,
        expected_identity_sha256=expected_identity_sha256,
    )
    if resolved is None:
        return None
    scientific, refs, _, _ = resolved
    return scientific, refs
