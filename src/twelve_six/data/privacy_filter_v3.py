"""Deterministic privacy/secret filter V3.

LOCAL_FREE successor to DATA-33. Findings contain only detector id and span;
public evidence never contains source text, previews, matched values, or match hashes.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

POLICY_VERSION = "12-6.privacy-filter-v3.frozen-20260826"
MANIFEST_SCHEMA = "12-6.privacy-filter-v3.scan-manifest.v1"
COVERAGE_CLAIM = "deterministic high-confidence patterns; zero findings is not proof of zero PII/secrets"

Action = Literal["ALLOW", "REDACT", "QUARANTINE", "EXCLUDE"]
_ACTION_RANK = {"ALLOW": 0, "REDACT": 1, "QUARANTINE": 2, "EXCLUDE": 3}

DETECTOR_ACTIONS: dict[str, Action] = {
    "api_token": "EXCLUDE",
    "cloud_credential": "EXCLUDE",
    "authorization_header": "EXCLUDE",
    "credential_url": "EXCLUDE",
    "private_filesystem_path": "REDACT",
    "email": "REDACT",
    "phone": "REDACT",
    "sensitive_id": "REDACT",
    "ssh_material": "EXCLUDE",
    "database_url": "QUARANTINE",
    "environment_secret_assignment": "QUARANTINE",
}

PLACEHOLDERS = frozenset({
    "changeme", "change_me", "example", "example123", "dummy", "placeholder",
    "notsecret", "not_secret", "redacted", "masked", "your_token", "your_api_key",
    "your_secret", "your_password", "token_here", "secret_here", "password_here",
    "xxxxxxxx", "********", "<secret>", "<token>", "<password>", "<redacted>",
})
GENERIC_USER_SEGMENTS = frozenset({"user", "users", "username", "name", "example", "demo", "test", "rootfs"})
EXAMPLE_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "example.com", "example.org", "example.net", "db.example.com"})

EMAIL_RE = re.compile(
    r"(?<![\w.+-])[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,63}(?![\w.-])",
    re.I,
)
PHONE_RE = re.compile(r"(?<![\w.])(?:\+\d{1,3}[\s().-]*)?(?:\(?\d{2,4}\)?[\s.-]*){2,4}\d{2,4}(?!\w)")
SSN_RE = re.compile(r"(?<!\d)(\d{3})-(\d{2})-(\d{4})(?!\d)")
LABELED_ID_RE = re.compile(
    r"(?im)\b(?:passport(?:[_ -]?(?:no|number))?|national[_ -]?id|tax[_ -]?id|taxpayer[_ -]?id|"
    r"id[_ -]?number|rnokpp|РНОКПП|ідентифікаційний\s+номер)\b\s*[:=#-]?\s*"
    r"(?P<value>[A-ZА-ЯІЇЄ0-9][A-ZА-ЯІЇЄ0-9 -]{5,23})"
)

API_TOKEN_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,255}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,255}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,255}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,255}\b"),
    re.compile(r"\bnpm_[A-Za-z0-9]{20,255}\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,255}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,255}\b"),
    re.compile(r"\b(?:sk|rk)_live_[A-Za-z0-9]{16,255}\b"),
)
CLOUD_PATTERNS = (
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(r"(?i)\b(?:aws_)?secret_access_key\b\s*[:=]\s*['\"]?(?P<value>[A-Za-z0-9/+=]{40})['\"]?"),
    re.compile(r"(?i)\bAccountKey\s*=\s*(?P<value>[A-Za-z0-9+/]{40,}={0,2})"),
)
AUTH_HEADER_RE = re.compile(
    r"(?im)^\s*(?:Authorization|Proxy-Authorization)\s*:\s*"
    r"(?P<scheme>Bearer|Basic|Token|ApiKey|API-Key)\s+(?P<value>[^\s,;]{8,1024})"
)
CREDENTIAL_URL_RE = re.compile(
    r"(?i)\b(?:https?|ftp|git\+https|ssh|sftp)://(?P<user>[^\s/:@]{1,128}):(?P<pwd>[^\s/@]{4,256})@(?P<host>[^\s/:?#]+)"
)
UNIX_HOME_RE = re.compile(r"(?<![\w.])(?P<path>/(?:home|Users)/(?P<user>[^/\s]{1,64})/(?:[^\s\x00]{1,512}))")
ROOT_PATH_RE = re.compile(r"(?<![\w.])(?P<path>/root/(?:[^\s\x00]{1,512}))")
WINDOWS_HOME_RE = re.compile(r"(?i)(?<![\w])(?P<path>[A-Z]:\\Users\\(?P<user>[^\\\s]{1,64})\\[^\r\n\t\x00]{1,512})")
SSH_PRIVATE_RE = re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC |DSA |PGP )?PRIVATE KEY-----", re.I)
SSH_PUBLIC_RE = re.compile(r"(?m)^\s*(?:ssh-rsa|ssh-ed25519|ecdsa-sha2-nistp\d+)\s+[A-Za-z0-9+/]{40,}={0,3}(?:\s+\S+)?\s*$")
DB_URL_RE = re.compile(
    r"(?i)\b(?P<scheme>postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis(?:s)?|mssql|sqlserver)://"
    r"(?P<authority>[^\s/'\"<>]{1,512})"
)
SENSITIVE_ENV_RE = re.compile(
    r"(?im)^\s*(?:export\s+|set\s+)?(?P<name>[A-Z][A-Z0-9_]{2,80})\s*=\s*(?P<quote>['\"]?)(?P<value>[^\r\n'\"]{1,2048})(?P=quote)\s*$"
)
GENERIC_SECRET_ASSIGN_RE = re.compile(
    r"(?im)\b(?P<name>password|passwd|pwd|client_secret|api_secret|secret|token|api_key|access_token)\b"
    r"\s*[:=]\s*(?P<quote>['\"]?)(?P<value>[^\s'\";,}]{8,256})(?P=quote)"
)
SENSITIVE_ENV_NAMES = re.compile(
    r"(?i)(?:^|_)(?:PASSWORD|PASSWD|PWD|SECRET|TOKEN|API_KEY|ACCESS_KEY|PRIVATE_KEY|CLIENT_SECRET|DATABASE_URL|DB_URL|CONNECTION_STRING|AUTH)(?:$|_)"
)


@dataclass(frozen=True)
class Finding:
    detector_id: str
    action: Action
    start: int
    end: int


@dataclass(frozen=True)
class HashSafeScanResult:
    input_sha256: str
    input_bytes: int
    action: Action
    detector_counts: Mapping[str, int]

    def evidence(self) -> dict[str, Any]:
        return {
            "input_sha256": self.input_sha256,
            "input_bytes": self.input_bytes,
            "action": self.action,
            "detector_counts": dict(sorted(self.detector_counts.items())),
        }


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def policy_manifest() -> dict[str, Any]:
    core = {
        "schema_version": POLICY_VERSION,
        "coverage_claim": COVERAGE_CLAIM,
        "detector_actions": dict(sorted(DETECTOR_ACTIONS.items())),
        "evidence_policy": {
            "source_text": "never persisted",
            "text_preview": "never persisted",
            "matched_values": "never persisted",
            "matched_value_hashes": "never persisted",
            "whole_input_sha256": "allowed",
            "detector_counts": "allowed",
        },
    }
    return {**core, "policy_sha256": hashlib.sha256(_canonical_json_bytes(core)).hexdigest()}


def _placeholder(value: str) -> bool:
    v = value.strip().strip("'\"").casefold()
    if v in PLACEHOLDERS:
        return True
    return (
        v.startswith(("${", "{{", "$", "%", "process.env", "os.getenv", "env("))
        or (v.startswith("<") and v.endswith(">"))
    )


def _secretish(value: str, *, min_len: int = 8) -> bool:
    value = value.strip().strip("'\"")
    if len(value) < min_len or _placeholder(value):
        return False
    classes = sum(bool(re.search(p, value)) for p in (r"[a-z]", r"[A-Z]", r"\d", r"[^A-Za-z0-9_]"))
    return classes >= 2 or len(value) >= 20


def _valid_phone(candidate: str) -> bool:
    digits = re.sub(r"\D", "", candidate)
    if not 10 <= len(digits) <= 15:
        return False
    stripped = candidate.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
        return False
    if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", stripped):
        return False
    separators = sum(ch in " -.()" for ch in stripped)
    return stripped.startswith("+") or separators >= 2


def _valid_ssn(match: re.Match[str]) -> bool:
    area, group, serial = match.groups()
    return area not in {"000", "666"} and not 900 <= int(area) <= 999 and group != "00" and serial != "0000"


def _append(out: list[Finding], detector: str, start: int, end: int) -> None:
    out.append(Finding(detector, DETECTOR_ACTIONS[detector], start, end))


def _overlaps(out: Iterable[Finding], start: int, end: int, detectors: set[str] | None = None) -> bool:
    return any(start < f.end and f.start < end and (detectors is None or f.detector_id in detectors) for f in out)


def detect(text: str) -> tuple[Finding, ...]:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    out: list[Finding] = []

    for regex in API_TOKEN_PATTERNS:
        for m in regex.finditer(text):
            _append(out, "api_token", m.start(), m.end())
    for regex in CLOUD_PATTERNS:
        for m in regex.finditer(text):
            if "value" in m.groupdict() and not _secretish(m.group("value"), min_len=20):
                continue
            _append(out, "cloud_credential", m.start(), m.end())

    for m in AUTH_HEADER_RE.finditer(text):
        if _secretish(m.group("value"), min_len=8):
            _append(out, "authorization_header", m.start(), m.end())

    for m in CREDENTIAL_URL_RE.finditer(text):
        if not _placeholder(m.group("pwd")):
            _append(out, "credential_url", m.start(), m.end())

    for m in UNIX_HOME_RE.finditer(text):
        if m.group("user").casefold() not in GENERIC_USER_SEGMENTS:
            _append(out, "private_filesystem_path", m.start("path"), m.end("path"))
    for m in ROOT_PATH_RE.finditer(text):
        _append(out, "private_filesystem_path", m.start("path"), m.end("path"))
    for m in WINDOWS_HOME_RE.finditer(text):
        if m.group("user").casefold() not in GENERIC_USER_SEGMENTS:
            _append(out, "private_filesystem_path", m.start("path"), m.end("path"))

    for m in EMAIL_RE.finditer(text):
        _append(out, "email", m.start(), m.end())
    for m in PHONE_RE.finditer(text):
        if _valid_phone(m.group(0)):
            _append(out, "phone", m.start(), m.end())
    for m in SSN_RE.finditer(text):
        if _valid_ssn(m):
            _append(out, "sensitive_id", m.start(), m.end())
    for m in LABELED_ID_RE.finditer(text):
        value = re.sub(r"[ -]", "", m.group("value"))
        if 6 <= len(value) <= 20 and not _placeholder(value) and len(set(value)) > 1:
            _append(out, "sensitive_id", m.start(), m.end())

    for regex in (SSH_PRIVATE_RE, SSH_PUBLIC_RE):
        for m in regex.finditer(text):
            _append(out, "ssh_material", m.start(), m.end())

    for m in DB_URL_RE.finditer(text):
        authority = m.group("authority")
        hostpart = authority.rsplit("@", 1)[-1]
        host = hostpart.rsplit(":", 1)[0].strip("[]").casefold()
        has_userinfo = "@" in authority and ":" in authority.split("@", 1)[0]
        if has_userinfo or host not in EXAMPLE_HOSTS:
            _append(out, "database_url", m.start(), m.end())

    for m in SENSITIVE_ENV_RE.finditer(text):
        name = m.group("name")
        value = m.group("value").strip()
        if SENSITIVE_ENV_NAMES.search(name) and _secretish(value, min_len=8):
            _append(out, "environment_secret_assignment", m.start(), m.end())

    for m in GENERIC_SECRET_ASSIGN_RE.finditer(text):
        if _secretish(m.group("value"), min_len=8):
            if not _overlaps(out, m.start(), m.end(), {"api_token", "cloud_credential", "authorization_header", "environment_secret_assignment"}):
                _append(out, "environment_secret_assignment", m.start(), m.end())

    unique = {(f.detector_id, f.start, f.end): f for f in out}
    return tuple(sorted(unique.values(), key=lambda f: (f.start, f.end, f.detector_id)))


def hash_safe_scan(data: bytes | str) -> HashSafeScanResult:
    if isinstance(data, str):
        raw = data.encode("utf-8")
        text = data
    elif isinstance(data, bytes):
        raw = data
        text = data.decode("utf-8", errors="replace")
    else:
        raise TypeError("data must be bytes or str")
    findings = detect(text)
    counts = Counter(f.detector_id for f in findings)
    action = max((f.action for f in findings), default="ALLOW", key=lambda a: _ACTION_RANK[a])
    return HashSafeScanResult(
        input_sha256=hashlib.sha256(raw).hexdigest(),
        input_bytes=len(raw),
        action=action,
        detector_counts=dict(sorted(counts.items())),
    )


def build_manifest(results: Iterable[HashSafeScanResult], *, inventory_sha256: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", inventory_sha256):
        raise ValueError("inventory_sha256 must be lowercase SHA-256")
    rows = list(results)
    totals = Counter({name: 0 for name in DETECTOR_ACTIONS})
    actions = Counter({name: 0 for name in _ACTION_RANK})
    for row in rows:
        totals.update(row.detector_counts)
        actions[row.action] += 1
    core = {
        "schema_version": MANIFEST_SCHEMA,
        "privacy_policy_sha256": policy_manifest()["policy_sha256"],
        "inventory_sha256": inventory_sha256,
        "coverage_claim": COVERAGE_CLAIM,
        "records_total": len(rows),
        "total_input_bytes": sum(r.input_bytes for r in rows),
        "action_counts": dict(sorted(actions.items())),
        "detector_counts": dict(sorted(totals.items())),
        "records": [r.evidence() for r in rows],
        "public_evidence_policy": {
            "source_text_retained": False,
            "text_previews_retained": False,
            "matched_values_retained": False,
            "matched_value_hashes_retained": False,
        },
    }
    return {**core, "manifest_sha256": hashlib.sha256(_canonical_json_bytes(core)).hexdigest()}


def assert_hash_safe_evidence(value: Any) -> None:
    forbidden_keys = {"text", "raw_text", "preview", "match", "matched_value", "value", "secret_value", "match_hash"}

    def walk(v: Any) -> None:
        if isinstance(v, Mapping):
            for k, child in v.items():
                if str(k).casefold() in forbidden_keys:
                    raise ValueError(f"forbidden evidence key: {k}")
                walk(child)
        elif isinstance(v, (list, tuple)):
            for child in v:
                walk(child)

    walk(value)
