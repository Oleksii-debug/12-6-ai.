"""Deterministic, interpretable document-quality filtering for D03 corpora.

This module is deliberately quality-only. It does not grant source rights, decide
PII/copyright policy, perform language admission, or claim semantic cleanliness.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

QUALITY_POLICY_SCHEMA = "12-6.document-quality-policy.v1"
QUALITY_DECISION_SCHEMA = "12-6.document-quality-decision.v1"
QUALITY_RUN_SCHEMA = "12-6.document-quality-run.v1"
QUALITY_CALIBRATION_SCHEMA = "12-6.document-quality-calibration.v1"

Mode = Literal["uk", "en", "code"]

_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_NATURAL_WORD_RE = re.compile(r"[^\W\d_]+(?:['’ʼ-][^\W\d_]+)*", re.UNICODE)
_CODE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CODE_KEYWORD_RE = re.compile(
    r"\b(?:def|class|return|if|else|elif|for|while|try|except|import|from|"
    r"function|const|let|var|interface|type|public|private|static|void|int|"
    r"SELECT|FROM|WHERE|GROUP|ORDER|JOIN|CREATE|INSERT|UPDATE|DELETE|"
    r"echo|fi|then|do|done|case|esac)\b"
)
_TEMPLATE_MARKERS = (
    "privacy policy",
    "terms of use",
    "terms and conditions",
    "cookie policy",
    "accept cookies",
    "all rights reserved",
    "sign in",
    "log in",
    "subscribe",
    "newsletter",
    "skip to content",
    "політика конфіденційності",
    "умови використання",
    "файли cookie",
    "прийняти cookie",
    "усі права захищено",
    "увійти",
    "підписатися",
    "перейти до вмісту",
)
_BOILERPLATE_MARKERS = (
    "javascript is required",
    "enable javascript",
    "read more",
    "click here",
    "share this",
    "follow us",
    "contact us",
    "advertisement",
    "sponsored",
    "©",
    "powered by",
    "завантажити більше",
    "читати далі",
    "поділитися",
    "реклама",
    "зв'язатися з нами",
)


class DocumentQualityError(ValueError):
    """Raised when quality policy/input/evidence is malformed."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise DocumentQualityError(f"{field} must be lowercase SHA-256 hex")
    return value


@dataclass(frozen=True)
class ModeThresholds:
    min_chars: int
    max_chars: int
    max_symbol_ratio: float
    max_repeated_line_ratio: float
    max_url_char_ratio: float
    max_template_line_ratio: float
    max_boilerplate_line_ratio: float
    min_distinct_token_ratio: float
    max_dominant_token_ratio: float
    diversity_min_tokens: int
    max_other_script_letter_ratio: float
    min_code_structure_score: int = 0

    def __post_init__(self) -> None:
        if self.min_chars <= 0 or self.max_chars <= self.min_chars:
            raise DocumentQualityError("invalid character thresholds")
        for name in (
            "max_symbol_ratio",
            "max_repeated_line_ratio",
            "max_url_char_ratio",
            "max_template_line_ratio",
            "max_boilerplate_line_ratio",
            "min_distinct_token_ratio",
            "max_dominant_token_ratio",
            "max_other_script_letter_ratio",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise DocumentQualityError(f"{name} must be in [0,1]")
        if self.diversity_min_tokens < 2:
            raise DocumentQualityError("diversity_min_tokens must be >= 2")
        if self.min_code_structure_score < 0:
            raise DocumentQualityError("min_code_structure_score must be non-negative")


@dataclass(frozen=True)
class QualityPolicy:
    policy_id: str
    uk: ModeThresholds
    en: ModeThresholds
    code: ModeThresholds
    reject_replacement_character: bool = True
    reject_surrogates: bool = True
    reject_disallowed_controls: bool = True
    score_weights_version: str = "interpretable-penalty-v1"
    schema_version: str = QUALITY_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise DocumentQualityError("policy_id must be non-empty")
        if self.schema_version != QUALITY_POLICY_SCHEMA:
            raise DocumentQualityError("unsupported quality policy schema")

    def thresholds_for(self, mode: Mode) -> ModeThresholds:
        return {"uk": self.uk, "en": self.en, "code": self.code}[mode]

    def manifest(self) -> dict[str, Any]:
        core = asdict(self)
        return {**core, "policy_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def default_quality_policy() -> QualityPolicy:
    """Conservative v1 thresholds calibrated for false-removal avoidance."""
    natural = dict(
        min_chars=60,
        max_chars=250_000,
        max_symbol_ratio=0.40,
        max_repeated_line_ratio=0.60,
        max_url_char_ratio=0.25,
        max_template_line_ratio=0.50,
        max_boilerplate_line_ratio=0.50,
        min_distinct_token_ratio=0.20,
        max_dominant_token_ratio=0.22,
        diversity_min_tokens=30,
        max_other_script_letter_ratio=0.20,
    )
    return QualityPolicy(
        policy_id="d03-lightweight-uk-en-code-v1",
        uk=ModeThresholds(**natural),
        en=ModeThresholds(**natural),
        code=ModeThresholds(
            min_chars=30,
            max_chars=400_000,
            max_symbol_ratio=0.78,
            max_repeated_line_ratio=0.75,
            max_url_char_ratio=0.45,
            max_template_line_ratio=0.70,
            max_boilerplate_line_ratio=0.70,
            min_distinct_token_ratio=0.10,
            max_dominant_token_ratio=0.38,
            diversity_min_tokens=20,
            max_other_script_letter_ratio=1.0,
            min_code_structure_score=2,
        ),
    )


@dataclass(frozen=True)
class QualityFeatures:
    chars: int
    utf8_bytes: int
    nonspace_chars: int
    lines: int
    nonempty_lines: int
    replacement_characters: int
    surrogate_codepoints: int
    disallowed_controls: int
    latin_letters: int
    cyrillic_letters: int
    other_script_letters: int
    other_script_letter_ratio: float
    punctuation_symbol_chars: int
    symbol_ratio: float
    repeated_lines: int
    repeated_line_ratio: float
    url_count: int
    url_chars: int
    url_char_ratio: float
    template_lines: int
    template_line_ratio: float
    boilerplate_lines: int
    boilerplate_line_ratio: float
    token_count: int
    unique_token_count: int
    distinct_token_ratio: float
    dominant_token_ratio: float
    code_identifiers: int
    code_keyword_hits: int
    indented_lines: int
    code_structure_score: int


@dataclass(frozen=True)
class QualityDecision:
    record_id: str
    mode: Mode
    accepted: bool
    score: int
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    edge_margin: float
    features: QualityFeatures
    schema_version: str = QUALITY_DECISION_SCHEMA

    def manifest(self) -> dict[str, Any]:
        core = asdict(self)
        return {**core, "decision_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def _script_counts(text: str) -> tuple[int, int, int]:
    latin = cyrillic = other = 0
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if "LATIN" in name:
            latin += 1
        elif "CYRILLIC" in name:
            cyrillic += 1
        else:
            other += 1
    return latin, cyrillic, other


def _line_density(lines: list[str], markers: tuple[str, ...]) -> tuple[int, float]:
    if not lines:
        return 0, 0.0
    hits = 0
    for line in lines:
        folded = line.casefold()
        if any(marker in folded for marker in markers):
            hits += 1
    return hits, hits / len(lines)


def _repetition(lines: list[str]) -> tuple[int, float]:
    normalized = [" ".join(line.casefold().split()) for line in lines if line.strip()]
    if len(normalized) < 2:
        return 0, 0.0
    counts = Counter(normalized)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    return duplicates, duplicates / len(normalized)


def _tokens(text: str, mode: Mode) -> list[str]:
    pattern = _CODE_TOKEN_RE if mode == "code" else _NATURAL_WORD_RE
    return [match.group(0).casefold() for match in pattern.finditer(text)]


def _code_structure(text: str, tokens: list[str]) -> tuple[int, int, int, int]:
    lines = text.splitlines()
    indented = sum(1 for line in lines if line.startswith((" ", "\t")) and line.strip())
    keyword_hits = len(_CODE_KEYWORD_RE.findall(text))
    identifiers = len(tokens)
    delimiter_hits = sum(text.count(char) for char in "{}[]();:=<>")
    comment_hits = sum(text.count(marker) for marker in ("#", "//", "/*", "<!--"))
    score = 0
    if len([line for line in lines if line.strip()]) >= 2:
        score += 1
    if indented:
        score += 1
    if keyword_hits:
        score += 1
    if identifiers >= 3:
        score += 1
    if delimiter_hits >= 2:
        score += 1
    if comment_hits:
        score += 1
    return identifiers, keyword_hits, indented, score


def extract_quality_features(text: str, mode: Mode) -> QualityFeatures:
    if mode not in {"uk", "en", "code"}:
        raise DocumentQualityError(f"unsupported quality mode: {mode!r}")
    if not isinstance(text, str):
        raise TypeError("text must be str")

    replacements = text.count("\ufffd")
    surrogates = sum(0xD800 <= ord(char) <= 0xDFFF for char in text)
    controls = sum(
        unicodedata.category(char) == "Cc" and char not in "\n\t" for char in text
    )
    try:
        utf8_bytes = len(text.encode("utf-8", errors="strict"))
    except UnicodeEncodeError:
        utf8_bytes = -1

    nonspace = [char for char in text if not char.isspace()]
    symbol_count = sum(
        unicodedata.category(char)[0] in {"P", "S"} for char in nonspace
    )
    symbol_ratio = symbol_count / len(nonspace) if nonspace else 0.0

    latin, cyrillic, other = _script_counts(text)
    letters = latin + cyrillic + other
    other_ratio = other / letters if letters else 0.0

    raw_lines = text.splitlines() or [text]
    nonempty_lines = [line for line in raw_lines if line.strip()]
    repeated, repeated_ratio = _repetition(nonempty_lines)
    template_lines, template_ratio = _line_density(nonempty_lines, _TEMPLATE_MARKERS)
    boilerplate_lines, boilerplate_ratio = _line_density(
        nonempty_lines, _BOILERPLATE_MARKERS
    )

    url_matches = list(_URL_RE.finditer(text))
    url_chars = sum(match.end() - match.start() for match in url_matches)
    url_ratio = url_chars / max(len(text), 1)

    tokens = _tokens(text, mode)
    token_counts = Counter(tokens)
    unique = len(token_counts)
    distinct = unique / len(tokens) if tokens else 0.0
    dominant = max(token_counts.values(), default=0) / len(tokens) if tokens else 0.0

    identifiers = keyword_hits = indented = code_score = 0
    if mode == "code":
        identifiers, keyword_hits, indented, code_score = _code_structure(text, tokens)

    return QualityFeatures(
        chars=len(text),
        utf8_bytes=utf8_bytes,
        nonspace_chars=len(nonspace),
        lines=len(raw_lines),
        nonempty_lines=len(nonempty_lines),
        replacement_characters=replacements,
        surrogate_codepoints=surrogates,
        disallowed_controls=controls,
        latin_letters=latin,
        cyrillic_letters=cyrillic,
        other_script_letters=other,
        other_script_letter_ratio=other_ratio,
        punctuation_symbol_chars=symbol_count,
        symbol_ratio=symbol_ratio,
        repeated_lines=repeated,
        repeated_line_ratio=repeated_ratio,
        url_count=len(url_matches),
        url_chars=url_chars,
        url_char_ratio=url_ratio,
        template_lines=template_lines,
        template_line_ratio=template_ratio,
        boilerplate_lines=boilerplate_lines,
        boilerplate_line_ratio=boilerplate_ratio,
        token_count=len(tokens),
        unique_token_count=unique,
        distinct_token_ratio=distinct,
        dominant_token_ratio=dominant,
        code_identifiers=identifiers,
        code_keyword_hits=keyword_hits,
        indented_lines=indented,
        code_structure_score=code_score,
    )


def _threshold_margins(
    features: QualityFeatures, thresholds: ModeThresholds, mode: Mode
) -> list[float]:
    margins = [
        (features.chars - thresholds.min_chars) / max(thresholds.min_chars, 1),
        (thresholds.max_chars - features.chars) / max(thresholds.max_chars, 1),
        (thresholds.max_symbol_ratio - features.symbol_ratio)
        / max(thresholds.max_symbol_ratio, 1e-9),
        (thresholds.max_repeated_line_ratio - features.repeated_line_ratio)
        / max(thresholds.max_repeated_line_ratio, 1e-9),
        (thresholds.max_url_char_ratio - features.url_char_ratio)
        / max(thresholds.max_url_char_ratio, 1e-9),
        (thresholds.max_template_line_ratio - features.template_line_ratio)
        / max(thresholds.max_template_line_ratio, 1e-9),
        (thresholds.max_boilerplate_line_ratio - features.boilerplate_line_ratio)
        / max(thresholds.max_boilerplate_line_ratio, 1e-9),
        (thresholds.max_other_script_letter_ratio - features.other_script_letter_ratio)
        / max(thresholds.max_other_script_letter_ratio, 1e-9),
    ]
    if features.token_count >= thresholds.diversity_min_tokens:
        margins.extend(
            [
                (features.distinct_token_ratio - thresholds.min_distinct_token_ratio)
                / max(thresholds.min_distinct_token_ratio, 1e-9),
                (thresholds.max_dominant_token_ratio - features.dominant_token_ratio)
                / max(thresholds.max_dominant_token_ratio, 1e-9),
            ]
        )
    if mode == "code":
        margins.append(
            (features.code_structure_score - thresholds.min_code_structure_score)
            / max(thresholds.min_code_structure_score, 1)
        )
    return margins


def assess_document(
    record_id: str,
    text: str,
    mode: Mode,
    *,
    policy: QualityPolicy | None = None,
) -> QualityDecision:
    """Assess document quality without consulting rights, PII, copyright, or held-out state."""
    if not isinstance(record_id, str) or not record_id.strip():
        raise DocumentQualityError("record_id must be non-empty")
    policy = policy or default_quality_policy()
    thresholds = policy.thresholds_for(mode)
    features = extract_quality_features(text, mode)

    reasons: list[str] = []
    warnings: list[str] = []
    if policy.reject_replacement_character and features.replacement_characters:
        reasons.append("invalid_replacement_character")
    if policy.reject_surrogates and features.surrogate_codepoints:
        reasons.append("invalid_surrogate_codepoint")
    if features.utf8_bytes < 0:
        reasons.append("invalid_utf8_scalar_sequence")
    if policy.reject_disallowed_controls and features.disallowed_controls:
        reasons.append("disallowed_control_character")
    if features.chars < thresholds.min_chars:
        reasons.append("too_short")
    if features.chars > thresholds.max_chars:
        reasons.append("too_long")
    if features.nonspace_chars == 0:
        reasons.append("empty_or_whitespace")
    if features.symbol_ratio > thresholds.max_symbol_ratio:
        reasons.append("high_symbol_ratio")
    if features.repeated_line_ratio > thresholds.max_repeated_line_ratio:
        reasons.append("high_line_repetition")
    if features.url_char_ratio > thresholds.max_url_char_ratio:
        reasons.append("high_url_density")
    if features.template_line_ratio > thresholds.max_template_line_ratio:
        reasons.append("high_template_density")
    if features.boilerplate_line_ratio > thresholds.max_boilerplate_line_ratio:
        reasons.append("high_boilerplate_density")
    if features.other_script_letter_ratio > thresholds.max_other_script_letter_ratio:
        reasons.append("unexpected_script_density")
    if features.token_count >= thresholds.diversity_min_tokens:
        if features.distinct_token_ratio < thresholds.min_distinct_token_ratio:
            reasons.append("low_token_diversity")
        if features.dominant_token_ratio > thresholds.max_dominant_token_ratio:
            reasons.append("dominant_token_repetition")
    if mode == "code" and features.code_structure_score < thresholds.min_code_structure_score:
        reasons.append("insufficient_code_structure")

    if not reasons:
        if features.url_char_ratio > thresholds.max_url_char_ratio * 0.70:
            warnings.append("near_url_density_limit")
        if features.repeated_line_ratio > thresholds.max_repeated_line_ratio * 0.70:
            warnings.append("near_line_repetition_limit")
        if features.symbol_ratio > thresholds.max_symbol_ratio * 0.80:
            warnings.append("near_symbol_ratio_limit")
        if features.template_line_ratio > thresholds.max_template_line_ratio * 0.70:
            warnings.append("near_template_density_limit")
        if (
            features.token_count >= thresholds.diversity_min_tokens
            and features.distinct_token_ratio < thresholds.min_distinct_token_ratio * 1.30
        ):
            warnings.append("near_diversity_limit")

    margins = _threshold_margins(features, thresholds, mode)
    invalid = any(
        reason.startswith("invalid_") or reason == "disallowed_control_character"
        for reason in reasons
    )
    if invalid:
        margins.append(-1.0)
    edge_margin = min(margins) if margins else 0.0

    score = max(0, 100 - 25 * len(reasons) - 5 * len(warnings))
    return QualityDecision(
        record_id=record_id,
        mode=mode,
        accepted=not reasons,
        score=score,
        reasons=tuple(sorted(set(reasons))),
        warnings=tuple(sorted(set(warnings))),
        edge_margin=round(edge_margin, 6),
        features=features,
    )


def _excerpt(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def run_quality_filter(
    records: Sequence[Mapping[str, Any]],
    *,
    input_manifest_sha256: str,
    policy: QualityPolicy | None = None,
    edge_samples_per_class: int = 3,
) -> dict[str, Any]:
    """Run deterministic quality-only filtering and emit manifest-bound evidence."""
    input_hash = _require_sha256(input_manifest_sha256, "input_manifest_sha256")
    if edge_samples_per_class < 0:
        raise DocumentQualityError("edge_samples_per_class must be non-negative")
    policy = policy or default_quality_policy()
    policy_manifest = policy.manifest()

    decisions: list[tuple[QualityDecision, str]] = []
    for row in sorted(records, key=lambda item: str(item.get("id", ""))):
        record_id = row.get("id")
        text = row.get("text")
        mode = row.get("mode")
        if (
            not isinstance(record_id, str)
            or not isinstance(text, str)
            or mode not in {"uk", "en", "code"}
        ):
            raise DocumentQualityError(
                "each record requires string id/text and mode uk|en|code"
            )
        decisions.append((assess_document(record_id, text, mode, policy=policy), text))

    accepted = [(decision, text) for decision, text in decisions if decision.accepted]
    rejected = [(decision, text) for decision, text in decisions if not decision.accepted]

    by_mode: dict[str, dict[str, int]] = {}
    for mode in ("uk", "en", "code"):
        mode_items = [decision for decision, _ in decisions if decision.mode == mode]
        by_mode[mode] = {
            "input": len(mode_items),
            "accepted": sum(item.accepted for item in mode_items),
            "rejected": sum(not item.accepted for item in mode_items),
        }

    reason_counts = Counter(reason for decision, _ in rejected for reason in decision.reasons)

    def choose_edges(items: list[tuple[QualityDecision, str]]) -> list[dict[str, Any]]:
        chosen = sorted(
            items,
            key=lambda pair: (abs(pair[0].edge_margin), pair[0].record_id),
        )[:edge_samples_per_class]
        return [
            {
                "id": decision.record_id,
                "mode": decision.mode,
                "accepted": decision.accepted,
                "score": decision.score,
                "edge_margin": decision.edge_margin,
                "reasons": list(decision.reasons),
                "warnings": list(decision.warnings),
                "excerpt": _excerpt(text),
            }
            for decision, text in chosen
        ]

    decision_core = [
        {
            "id": decision.record_id,
            "mode": decision.mode,
            "accepted": decision.accepted,
            "score": decision.score,
            "reasons": list(decision.reasons),
            "warnings": list(decision.warnings),
            "edge_margin": decision.edge_margin,
            "features": asdict(decision.features),
        }
        for decision, _ in decisions
    ]
    core = {
        "schema_version": QUALITY_RUN_SCHEMA,
        "input_manifest_sha256": input_hash,
        "quality_policy_sha256": policy_manifest["policy_sha256"],
        "quality_policy_id": policy.policy_id,
        "quality_scope": "DOCUMENT_QUALITY_ONLY_NO_RIGHTS_PII_COPYRIGHT_OR_LID_AUTHORITY",
        "input_documents": len(decisions),
        "accepted_documents": len(accepted),
        "rejected_documents": len(rejected),
        "acceptance_rate": round(len(accepted) / len(decisions), 6) if decisions else 0.0,
        "by_mode": by_mode,
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "decisions_sha256": _sha256_bytes(_canonical_json_bytes(decision_core)),
        "edge_samples": {
            "accepted_near_boundary": choose_edges(accepted),
            "rejected_near_boundary": choose_edges(rejected),
        },
    }
    return {**core, "run_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def evaluate_calibration(
    rows: Sequence[Mapping[str, Any]],
    *,
    calibration_manifest_sha256: str,
    policy: QualityPolicy | None = None,
) -> dict[str, Any]:
    """Evaluate explicit project-owned ACCEPT/REJECT labels."""
    calibration_hash = _require_sha256(
        calibration_manifest_sha256, "calibration_manifest_sha256"
    )
    policy = policy or default_quality_policy()
    false_accepts: list[str] = []
    false_rejects: list[str] = []
    by_mode: dict[str, Counter[str]] = {
        mode: Counter() for mode in ("uk", "en", "code")
    }

    for row in rows:
        label = row.get("label")
        if label not in {"ACCEPT", "REJECT"}:
            raise DocumentQualityError("calibration label must be ACCEPT or REJECT")
        decision = assess_document(
            str(row.get("id")),
            str(row.get("text")),
            row.get("mode"),  # type: ignore[arg-type]
            policy=policy,
        )
        predicted = "ACCEPT" if decision.accepted else "REJECT"
        by_mode[decision.mode]["total"] += 1
        by_mode[decision.mode]["correct"] += predicted == label
        if label == "REJECT" and predicted == "ACCEPT":
            false_accepts.append(decision.record_id)
        elif label == "ACCEPT" and predicted == "REJECT":
            false_rejects.append(decision.record_id)

    total = len(rows)
    correct = total - len(false_accepts) - len(false_rejects)
    core = {
        "schema_version": QUALITY_CALIBRATION_SCHEMA,
        "calibration_manifest_sha256": calibration_hash,
        "quality_policy_sha256": policy.manifest()["policy_sha256"],
        "samples": total,
        "correct": correct,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "false_accepts": sorted(false_accepts),
        "false_rejects": sorted(false_rejects),
        "false_accept_rate": round(len(false_accepts) / total, 6) if total else 0.0,
        "false_reject_rate": round(len(false_rejects) / total, 6) if total else 0.0,
        "by_mode": {
            mode: {
                "samples": counts["total"],
                "correct": counts["correct"],
                "accuracy": round(counts["correct"] / counts["total"], 6)
                if counts["total"]
                else 0.0,
            }
            for mode, counts in by_mode.items()
        },
        "authority": "PROJECT_OWNED_CALIBRATION_ONLY_NOT_GENERAL_CORPUS_QUALITY_PROOF",
    }
    return {
        **core,
        "calibration_run_sha256": _sha256_bytes(_canonical_json_bytes(core)),
    }


def to_d03_quality_hook(
    decision: QualityDecision,
    *,
    policy: QualityPolicy | None = None,
    executed_at: str,
    tool_ref: str = "twelve_six.data.document_quality",
) -> Any:
    """Adapt a decision to the incumbent D03 quality hook without touching other hooks."""
    from .corpus_foundation import PolicyHookEvidence

    if not isinstance(executed_at, str) or not executed_at.strip():
        raise DocumentQualityError("executed_at must be a non-empty evidence timestamp")
    policy = policy or default_quality_policy()
    return PolicyHookEvidence(
        hook_id="document_quality",
        status="PASS" if decision.accepted else "REJECT",
        policy_version=policy.manifest()["policy_sha256"],
        tool_ref=tool_ref,
        executed_at=executed_at,
        evidence_sha256=decision.manifest()["decision_sha256"],
    )
