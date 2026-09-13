"""Helpers for fail-closed license prose validation.

License *identity* must be established separately from exact bytes (for example
Git blob SHA-1 and SHA-256).  These helpers address only a narrower problem:
prose grants in an immutable license are commonly line-wrapped, so literal
substring matching against raw text can reject byte-identical licenses for
formatting-only reasons.

Whitespace normalization here is deliberately conservative: runs of Unicode
whitespace become one ASCII space.  Case and punctuation are preserved, so a
changed or missing legal phrase still fails.
"""

from __future__ import annotations

from collections.abc import Iterable


class LicensePhraseError(ValueError):
    """Raised when required license prose is absent after whitespace folding."""


def normalize_license_prose(text: str) -> str:
    """Return *text* with whitespace runs folded to single spaces.

    This function must never be used to compute or replace an immutable license
    identity.  Hash checks belong on the original bytes before prose matching.
    """

    if not isinstance(text, str):
        raise TypeError("license text must be str")
    return " ".join(text.split())


def require_license_phrases(text: str, phrases: Iterable[str]) -> None:
    """Require every non-empty phrase in license prose after whitespace folding.

    The same whitespace folding is applied to the required phrase so callers
    may store a phrase naturally without depending on the upstream line-wrap.
    Matching remains case- and punctuation-sensitive by design.
    """

    normalized_text = normalize_license_prose(text)
    for phrase in phrases:
        if not isinstance(phrase, str):
            raise TypeError("license phrase must be str")
        normalized_phrase = normalize_license_prose(phrase)
        if not normalized_phrase:
            raise ValueError("license phrase must not be empty")
        if normalized_phrase not in normalized_text:
            raise LicensePhraseError(
                f"required license phrase missing after whitespace normalization: {phrase}"
            )
