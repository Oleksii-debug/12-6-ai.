"""Deterministic high-confidence PII/secret filtering for pretraining corpora.

This module deliberately does not claim universal PII detection. It implements a
bounded, interpretable pattern/validator policy and emits evidence that never
contains matched secret values or text previews.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

from .corpus_foundation import PolicyHookEvidence

PRIVACY_POLICY_VERSION = "12-6.pii-secrets-policy.v1"
PRIVACY_MANIFEST_SCHEMA = "12-6.pii-secrets-scan-manifest.v1"
PRIVACY_TOOL_REF = "twelve_six.data.privacy_filter@1"
COVERAGE_CLAIM = (
    "high-confidence-patterns-only; zero detections does not imply zero PII/secrets"
)

Action = Literal["ALLOW", "REDACT", "QUARANTINE", "EXCLUDE"]
Status = Literal["PASS", "REVIEW_REQUIRED", "REJECT"]
Modality = Literal["natural", "code"]

_ACTION_RANK: dict[Action, int] = {
    "ALLOW": 0,
    "REDACT": 1,
    "QUARANTINE": 2,
    "EXCLUDE": 3,
}

_DETECTOR_ACTIONS: dict[str, Action] = {
    "email": "REDACT",
    "phone": "REDACT",
    "us_ssn": "REDACT",
    "payment_card_luhn": "REDACT",
    "iban": "REDACT",
    "private_key": "EXCLUDE",
    "aws_access_key_id": "EXCLUDE",
    "aws_secret_access_key": "EXCLUDE",
    "github_token": "EXCLUDE",
    "slack_token": "EXCLUDE",
    "stripe_live_secret": "EXCLUDE",
    "google_api_key": "EXCLUDE",
    "bearer_jwt": "EXCLUDE",
    "azure_storage_account_key": "EXCLUDE",
    "generic_password_assignment": "QUARANTINE",
    "generic_secret_assignment": "QUARANTINE",
}

_REDACTION_PLACEHOLDERS = {
    "email": "[EMAIL_REDACTED]",
    "phone": "[PHONE_REDACTED]",
    "us_ssn": "[SSN_REDACTED]",
    "payment_card_luhn": "[CARD_REDACTED]",
    "iban": "[IBAN_REDACTED]",
}

_EMAIL_RE = re.compile(
    r"(?<![\w.+-])[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,63}(?![\w.-])",
    re.IGNORECASE,
)
_PHONE_CANDIDATE_RE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[\s().-]*)?(?:\(?\d{2,4}\)?[\s.-]*){2,4}\d{2,4}(?!\w)"
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----",
    re.IGNORECASE,
)
_AWS_ACCESS_RE = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_AWS_SECRET_RE = re.compile(
    r"(?i)\b(?:aws_)?secret_access_key\b\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?"
)
_GITHUB_TOKEN_RE = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{50,255})\b"
)
_SLACK_TOKEN_RE = re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")
_STRIPE_LIVE_RE = re.compile(r"\b(?:sk|rk)_live_[A-Za-z0-9]{16,}\b")
_GOOGLE_API_RE = re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")
_BEARER_JWT_RE = re.compile(
    r"(?i)\bBearer\s+(eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})\b"
)
_AZURE_ACCOUNT_KEY_RE = re.compile(
    r"(?i)\bAccountKey\s*=\s*([A-Za-z0-9+/]{40,}={0,2})"
)
_GENERIC_ASSIGNMENT_RE = re.compile(
    r"(?im)\b(?P<name>password|passwd|pwd|client_secret|api_secret|secret|token|api_key)\b"
    r"\s*[:=]\s*(?P<quote>['\"]?)(?P<value>[^\s'\";,}]{8,256})(?P=quote)"
)
_SSN_RE = re.compile(r"(?<!\d)(\d{3})-(\d{2})-(\d{4})(?!\d)")
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_IBAN_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}(?![A-Z0-9])")

_FALSE_GENERIC_VALUES = frozenset(
    {
        "changeme",
        "change_me",
        "password",
        "example",
        "example123",
        "dummy",
        "placeholder",
        "notsecret",
        "not_secret",
        "your_password",
        "your_secret",
        "your_token",
        "your_api_key",
        "xxxxxxxx",
        "********",
        "<secret>",
        "<password>",
    }
)


class PrivacyFilterError(ValueError):
    """Raised when privacy-filter inputs or evidence are unsafe/inconsistent."""


@dataclass(frozen=True)
class PrivacyFinding:
    detector_id: str
    action: Action
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.detector_id not in _DETECTOR_ACTIONS:
            raise PrivacyFilterError(f"unknown detector: {self.detector_id}")
        if self.action != _DETECTOR_ACTIONS[self.detector_id]:
            raise PrivacyFilterError("finding action does not match policy")
        if self.start < 0 or self.end <= self.start:
            raise PrivacyFilterError("invalid finding span")


@dataclass(frozen=True)
class PrivacyRecordResult:
    record_id: str
    source_id: str
    source_version: str
    modality: Modality
    action: Action
    status: Status
    input_sha256: str
    output_sha256: str | None
    detector_counts: Mapping[str, int]
    redaction_count: int
    policy_sha256: str
    sanitized_text: str | None

    @property
    def train_eligible_after_privacy(self) -> bool:
        return self.status == "PASS" and self.sanitized_text is not None

    def evidence_record(self) -> dict[str, Any]:
        """Sanitized evidence only; no source text, previews, or matched values."""
        return {
            "record_id": self.record_id,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "modality": self.modality,
            "action": self.action,
            "status": self.status,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "detector_counts": dict(sorted(self.detector_counts.items())),
            "redaction_count": self.redaction_count,
            "policy_sha256": self.policy_sha256,
        }

    def evidence_sha256(self) -> str:
        return _sha256_json(self.evidence_record())

    def policy_hook_evidence(
        self,
        *,
        executed_at: str,
        tool_ref: str = PRIVACY_TOOL_REF,
    ) -> PolicyHookEvidence:
        if not executed_at.strip() or not tool_ref.strip():
            raise PrivacyFilterError("executed_at and tool_ref must be non-empty")
        return PolicyHookEvidence(
            hook_id="pii",
            status=self.status,
            policy_version=PRIVACY_POLICY_VERSION,
            tool_ref=tool_ref,
            executed_at=executed_at,
            evidence_sha256=self.evidence_sha256(),
        )


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def privacy_policy_manifest() -> dict[str, Any]:
    core = {
        "schema_version": PRIVACY_POLICY_VERSION,
        "coverage_claim": COVERAGE_CLAIM,
        "detector_actions": dict(sorted(_DETECTOR_ACTIONS.items())),
        "redaction_placeholders": dict(sorted(_REDACTION_PLACEHOLDERS.items())),
        "evidence_policy": {
            "matched_values": "never persisted",
            "text_previews": "never persisted",
            "document_hashes": "sha256 allowed",
            "sanitized_output_hashes": "sha256 allowed",
        },
    }
    return {**core, "policy_sha256": _sha256_json(core)}


def _valid_phone(candidate: str) -> bool:
    digits = re.sub(r"\D", "", candidate)
    if not 10 <= len(digits) <= 15:
        return False
    stripped = candidate.strip()
    separators = sum(ch in " -.()" for ch in stripped)
    if not stripped.startswith("+") and separators < 2:
        return False
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
        return False
    if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", stripped):
        return False
    return True


def _valid_ssn(match: re.Match[str]) -> bool:
    area, group, serial = match.groups()
    if area == "000" or area == "666" or 900 <= int(area) <= 999:
        return False
    return group != "00" and serial != "0000"


def _luhn_valid(digits: str) -> bool:
    if len(set(digits)) == 1:
        return False
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _valid_iban(candidate: str) -> bool:
    compact = candidate.replace(" ", "")
    if not 15 <= len(compact) <= 34 or not compact.isalnum():
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(ord(ch) - 55) if ch.isalpha() else ch for ch in rearranged)
    remainder = 0
    for char in numeric:
        remainder = (remainder * 10 + int(char)) % 97
    return remainder == 1


def _looks_like_real_generic_secret(name: str, value: str) -> bool:
    lowered = value.casefold()
    if lowered in _FALSE_GENERIC_VALUES:
        return False
    if lowered.startswith(("${", "{{", "env(", "os.getenv", "process.env")):
        return False
    if value.startswith("<") and value.endswith(">"):
        return False
    if len(value) < 8:
        return False
    classes = sum(
        bool(test(value))
        for test in (
            lambda v: re.search(r"[a-z]", v),
            lambda v: re.search(r"[A-Z]", v),
            lambda v: re.search(r"\d", v),
            lambda v: re.search(r"[^A-Za-z0-9_]", v),
        )
    )
    # Passwords can be human-like; generic token/secret names require stronger shape.
    if name in {"password", "passwd", "pwd"}:
        return classes >= 2 and len(value) >= 10
    return (classes >= 2 and len(value) >= 12) or len(value) >= 20


def _append(matches: list[PrivacyFinding], detector: str, start: int, end: int) -> None:
    matches.append(PrivacyFinding(detector, _DETECTOR_ACTIONS[detector], start, end))


def detect_privacy_findings(text: str) -> tuple[PrivacyFinding, ...]:
    """Return findings without storing or returning matched values."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    matches: list[PrivacyFinding] = []

    for match in _EMAIL_RE.finditer(text):
        _append(matches, "email", match.start(), match.end())
    for match in _PHONE_CANDIDATE_RE.finditer(text):
        if _valid_phone(match.group(0)):
            _append(matches, "phone", match.start(), match.end())
    for match in _SSN_RE.finditer(text):
        if _valid_ssn(match):
            _append(matches, "us_ssn", match.start(), match.end())
    for match in _CARD_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_valid(digits):
            _append(matches, "payment_card_luhn", match.start(), match.end())
    for match in _IBAN_RE.finditer(text):
        if _valid_iban(match.group(0)):
            _append(matches, "iban", match.start(), match.end())

    for detector, regex in (
        ("private_key", _PRIVATE_KEY_RE),
        ("aws_access_key_id", _AWS_ACCESS_RE),
        ("aws_secret_access_key", _AWS_SECRET_RE),
        ("github_token", _GITHUB_TOKEN_RE),
        ("slack_token", _SLACK_TOKEN_RE),
        ("stripe_live_secret", _STRIPE_LIVE_RE),
        ("google_api_key", _GOOGLE_API_RE),
        ("bearer_jwt", _BEARER_JWT_RE),
        ("azure_storage_account_key", _AZURE_ACCOUNT_KEY_RE),
    ):
        for match in regex.finditer(text):
            _append(matches, detector, match.start(), match.end())

    for match in _GENERIC_ASSIGNMENT_RE.finditer(text):
        name = match.group("name").casefold()
        value = match.group("value")
        if not _looks_like_real_generic_secret(name, value):
            continue
        if any(
            existing.action == "EXCLUDE"
            and match.start() < existing.end
            and existing.start < match.end()
            for existing in matches
        ):
            continue
        detector = (
            "generic_password_assignment"
            if name in {"password", "passwd", "pwd"}
            else "generic_secret_assignment"
        )
        _append(matches, detector, match.start(), match.end())

    unique = {(m.detector_id, m.start, m.end): m for m in matches}
    return tuple(sorted(unique.values(), key=lambda m: (m.start, m.end, m.detector_id)))


def _redact_text(text: str, findings: Iterable[PrivacyFinding]) -> tuple[str, int]:
    candidates = [f for f in findings if f.action == "REDACT"]
    # Prefer the more specific detector when spans overlap; process left-to-right.
    detector_priority = {
        "payment_card_luhn": 4,
        "us_ssn": 4,
        "iban": 4,
        "email": 3,
        "phone": 2,
    }
    ordered = sorted(
        candidates,
        key=lambda f: (f.start, -detector_priority.get(f.detector_id, 0), -(f.end - f.start)),
    )
    selected: list[PrivacyFinding] = []
    for finding in ordered:
        if selected and finding.start < selected[-1].end:
            continue
        selected.append(finding)
    if not selected:
        return text, 0
    pieces: list[str] = []
    cursor = 0
    for finding in selected:
        pieces.append(text[cursor:finding.start])
        pieces.append(_REDACTION_PLACEHOLDERS[finding.detector_id])
        cursor = finding.end
    pieces.append(text[cursor:])
    return "".join(pieces), len(selected)


def scan_record(
    *,
    record_id: str,
    source_id: str,
    source_version: str,
    modality: Modality,
    text: str,
) -> PrivacyRecordResult:
    for name, value in (
        ("record_id", record_id),
        ("source_id", source_id),
        ("source_version", source_version),
    ):
        if not isinstance(value, str) or not value.strip():
            raise PrivacyFilterError(f"{name} must be non-empty")
    if modality not in {"natural", "code"}:
        raise PrivacyFilterError("modality must be natural or code")
    if not isinstance(text, str):
        raise TypeError("text must be str")

    policy_sha256 = privacy_policy_manifest()["policy_sha256"]
    findings = detect_privacy_findings(text)
    counts = Counter(f.detector_id for f in findings)
    action: Action = max(
        (f.action for f in findings),
        default="ALLOW",
        key=lambda item: _ACTION_RANK[item],
    )
    input_sha256 = _sha256_text(text)

    if action == "EXCLUDE":
        status: Status = "REJECT"
        sanitized = None
        output_sha256 = None
        redaction_count = 0
    elif action == "QUARANTINE":
        status = "REVIEW_REQUIRED"
        sanitized = None
        output_sha256 = None
        redaction_count = 0
    else:
        status = "PASS"
        sanitized, redaction_count = _redact_text(text, findings)
        output_sha256 = _sha256_text(sanitized)
        action = "REDACT" if redaction_count else "ALLOW"

    return PrivacyRecordResult(
        record_id=record_id.strip(),
        source_id=source_id.strip(),
        source_version=source_version.strip(),
        modality=modality,
        action=action,
        status=status,
        input_sha256=input_sha256,
        output_sha256=output_sha256,
        detector_counts=dict(sorted(counts.items())),
        redaction_count=redaction_count,
        policy_sha256=policy_sha256,
        sanitized_text=sanitized,
    )


def build_scan_manifest(
    results: Iterable[PrivacyRecordResult],
    *,
    input_content_sha256: str,
    source_registry_sha256: str,
) -> dict[str, Any]:
    for field, value in (
        ("input_content_sha256", input_content_sha256),
        ("source_registry_sha256", source_registry_sha256),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise PrivacyFilterError(f"{field} must be lowercase SHA-256")
    rows = list(results)
    expected_policy = privacy_policy_manifest()["policy_sha256"]
    if any(row.policy_sha256 != expected_policy for row in rows):
        raise PrivacyFilterError("privacy policy identity drift within scan")

    detector_counts: Counter[str] = Counter({name: 0 for name in _DETECTOR_ACTIONS})
    action_counts: Counter[str] = Counter({name: 0 for name in _ACTION_RANK})
    modality_counts: dict[str, Counter[str]] = {
        modality: Counter(
            {
                **{name: 0 for name in _ACTION_RANK},
                **{f"detector:{name}": 0 for name in _DETECTOR_ACTIONS},
            }
        )
        for modality in ("natural", "code")
    }
    for row in rows:
        detector_counts.update(row.detector_counts)
        action_counts[row.action] += 1
        modality_counts[row.modality][row.action] += 1
        for detector, count in row.detector_counts.items():
            modality_counts[row.modality][f"detector:{detector}"] += count

    core = {
        "schema_version": PRIVACY_MANIFEST_SCHEMA,
        "privacy_policy_sha256": expected_policy,
        "input_content_sha256": input_content_sha256,
        "source_registry_sha256": source_registry_sha256,
        "coverage_claim": COVERAGE_CLAIM,
        "records_total": len(rows),
        "records_train_eligible_after_privacy": sum(
            row.train_eligible_after_privacy for row in rows
        ),
        "action_counts": dict(sorted(action_counts.items())),
        "detector_counts": dict(sorted(detector_counts.items())),
        "by_modality": {
            key: dict(sorted(value.items())) for key, value in modality_counts.items()
        },
        "records": [row.evidence_record() for row in rows],
    }
    return {**core, "manifest_sha256": _sha256_json(core)}


def assert_no_secret_values_in_manifest(manifest: Mapping[str, Any]) -> None:
    """Structural guard: evidence schema must not grow text/preview/value fields."""
    forbidden = {"text", "raw_text", "sanitized_text", "preview", "match", "value", "secret"}

    def walk(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = str(key).casefold()
                if (
                    normalized in forbidden
                    or normalized.endswith("_value")
                    or normalized.endswith("_preview")
                ):
                    raise PrivacyFilterError(
                        "privacy evidence contains a forbidden value-bearing field: "
                        + ".".join((*path, str(key)))
                    )
                walk(child, (*path, str(key)))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, (*path, str(index)))

    walk(manifest)
