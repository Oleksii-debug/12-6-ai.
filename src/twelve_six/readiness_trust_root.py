"""Authenticate the externally pinned learned-20M readiness trust bundle."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from typing import Any

from twelve_six.learned20m_readiness import trusted_readiness_inputs

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def trusted_readiness_bundle_sha256(bindings: Any) -> str | None:
    """Return the canonical identity of a structurally valid trusted bundle."""
    if trusted_readiness_inputs(bindings) is None:
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


def authenticated_trusted_readiness_inputs(
    bindings: Any,
    *,
    expected_identity_sha256: Any,
) -> tuple[set[str], set[str]] | None:
    """Resolve trust inputs only after an independent expected root matches.

    The expected identity is deliberately a separate input. Callers must obtain
    it from an authority surface independent of both the candidate packet and
    the trusted-bundle bytes; this function never derives an expectation for
    the caller.
    """
    if (
        not isinstance(expected_identity_sha256, str)
        or _SHA256_RE.fullmatch(expected_identity_sha256) is None
    ):
        return None

    observed = trusted_readiness_bundle_sha256(bindings)
    if observed is None or not hmac.compare_digest(observed, expected_identity_sha256):
        return None
    return trusted_readiness_inputs(bindings)
