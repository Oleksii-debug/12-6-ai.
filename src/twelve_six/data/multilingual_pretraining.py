"""Multilingual Base-pretraining admission, mixture, and token-cost contracts.

This extends the incumbent D03/D09 provenance foundation and D10/D04
mixture/tokenizer contracts. It does not replace them or mutate canonical S0 bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Literal, cast

from twelve_six.data.external_sources import RIGHTS_APPROVED
from twelve_six.data.pipeline import normalize_text
from twelve_six.packing.scale_contracts import MixturePlan, MixtureSource, RestartCursor

MULTILINGUAL_SCHEMA = "12-6.multilingual-pretraining-v1"
_SELECTION_WEIGHTS = {"uk": 45, "en": 35, "code": 20}
_FORBIDDEN_TRAIN_SPLITS = frozenset(
    {"validation", "val", "evaluation", "eval", "test", "heldout", "benchmark"}
)
_FORBIDDEN_SOURCE_PURPOSES = frozenset(
    {"benchmark", "evaluation", "validation", "test", "heldout_test"}
)
_UK_SPECIFIC = frozenset("іїєґІЇЄҐ")
_UK_LEXICAL = frozenset(
    {
        "і", "й", "та", "але", "або", "що", "щоб", "для", "від", "до",
        "після", "перед", "цей", "ця", "це", "ці", "який", "яка", "яке",
        "які", "україна", "український", "мови", "мова", "дані", "модель",
    }
)
_EN_LEXICAL = frozenset(
    {
        "the", "and", "or", "but", "for", "from", "with", "this", "that",
        "these", "those", "is", "are", "was", "were", "data", "model",
        "language", "training",
    }
)
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’ʼ-][^\W\d_]+)*", re.UNICODE)


class MultilingualDataError(ValueError):
    """Fail-closed multilingual pretraining contract error."""


@dataclass(frozen=True)
class ScriptProfile:
    codepoints: int
    utf8_bytes: int
    latin_letters: int
    cyrillic_letters: int
    ukrainian_specific_letters: int
    digits: int
    whitespace: int
    punctuation_symbols: int
    other: int
    replacement_characters: int

    @property
    def alphabetic_letters(self) -> int:
        return self.latin_letters + self.cyrillic_letters


@dataclass(frozen=True)
class LanguageEvidence:
    label: Literal["uk", "en", "code", "mixed", "und"]
    confidence: float
    script: ScriptProfile
    ukrainian_lexical_hits: int
    english_lexical_hits: int
    reason: str


@dataclass(frozen=True)
class PretrainingRecord:
    record_id: str
    source_id: str
    source_version: str
    source_manifest_sha256: str
    split: str
    source_purpose: str
    modality: Literal["natural", "code"]
    text: str
    language_hint: str | None = None
    external: bool = False
    rights_status: str | None = None
    allows_model_training: bool | None = None
    project_authored_synthetic: bool = False

    def __post_init__(self) -> None:
        fields = ("record_id", "source_id", "source_version", "split", "source_purpose")
        for field_name in fields:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise MultilingualDataError(f"{field_name} must be non-empty")
        _require_sha256(self.source_manifest_sha256, "source_manifest_sha256")
        if self.modality not in {"natural", "code"}:
            raise MultilingualDataError("modality must be natural or code")
        if not isinstance(self.text, str):
            raise TypeError("text must be str")
        if self.external and self.project_authored_synthetic:
            raise MultilingualDataError(
                "external and project_authored_synthetic are mutually exclusive"
            )


@dataclass(frozen=True)
class AdmittedRecord:
    record_id: str
    source_id: str
    source_version: str
    source_manifest_sha256: str
    split: str
    modality: str
    language: str
    normalized_text: str
    normalized_sha256: str
    language_evidence: LanguageEvidence


@dataclass(frozen=True)
class MixtureStratum:
    name: Literal["uk", "en", "code"]
    manifest_sha256: str
    weight_units: int

    def __post_init__(self) -> None:
        if self.name not in _SELECTION_WEIGHTS:
            raise MultilingualDataError("mixture stratum must be uk, en, or code")
        _require_sha256(self.manifest_sha256, "manifest_sha256")
        if self.weight_units <= 0:
            raise MultilingualDataError("weight_units must be positive")


@dataclass(frozen=True)
class TokenizerCost:
    name: str
    vocab_size: int
    observed_tokens: int
    byte_baseline_tokens: int
    token_reduction_vs_bytes: float
    d_model: int
    tied_vocabulary_parameters: int
    vocabulary_parameters_vs_byte: int


def _require_sha256(value: str, field: str) -> None:
    valid = isinstance(value, str) and len(value) == 64 and value == value.lower()
    if not valid or any(char not in "0123456789abcdef" for char in value):
        raise MultilingualDataError(f"{field} must be lowercase SHA-256")


def _normalize_code_layout(text: str) -> str:
    """Keep indentation/newlines while retaining the incumbent NFKC policy."""
    normalized = unicodedata.normalize(
        "NFKC", text.replace("\r\n", "\n").replace("\r", "\n")
    )
    return normalized.strip("\n")


def strict_normalize_utf8(
    text: str,
    *,
    preserve_layout: bool = False,
) -> tuple[str, ScriptProfile]:
    """Validate Unicode scalars and normalize without corrupting code indentation."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    if "\ufffd" in text:
        raise MultilingualDataError("replacement character U+FFFD is forbidden")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in text):
        raise MultilingualDataError("surrogate code points are forbidden")
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise MultilingualDataError("text is not strict UTF-8 encodable") from exc

    normalized = _normalize_code_layout(text) if preserve_layout else normalize_text(text)
    if not normalized:
        raise MultilingualDataError("normalized text is empty")
    normalized.encode("utf-8", errors="strict")
    return normalized, script_profile(normalized)


def script_profile(text: str) -> ScriptProfile:
    latin = cyrillic = uk_specific = digits = whitespace = punctuation = other = 0
    replacements = 0
    for char in text:
        if char == "\ufffd":
            replacements += 1
        name = unicodedata.name(char, "")
        category = unicodedata.category(char)
        if char.isalpha() and "LATIN" in name:
            latin += 1
        elif char.isalpha() and "CYRILLIC" in name:
            cyrillic += 1
            if char in _UK_SPECIFIC:
                uk_specific += 1
        elif char.isdigit():
            digits += 1
        elif char.isspace():
            whitespace += 1
        elif category[0] in {"P", "S"}:
            punctuation += 1
        else:
            other += 1
    return ScriptProfile(
        len(text), len(text.encode("utf-8")), latin, cyrillic, uk_specific,
        digits, whitespace, punctuation, other, replacements,
    )


def detect_language(
    text: str,
    *,
    modality: Literal["natural", "code"] = "natural",
    language_hint: str | None = None,
) -> LanguageEvidence:
    normalized, profile = strict_normalize_utf8(
        text, preserve_layout=modality == "code"
    )
    if modality == "code":
        return LanguageEvidence("code", 1.0, profile, 0, 0, "explicit-source-modality")

    words = [
        word.casefold().replace("’", "'").replace("ʼ", "'")
        for word in _WORD_RE.findall(normalized)
    ]
    uk_hits = sum(word in _UK_LEXICAL for word in words)
    en_hits = sum(word in _EN_LEXICAL for word in words)
    alpha = max(profile.alphabetic_letters, 1)
    latin_ratio = profile.latin_letters / alpha
    cyrillic_ratio = profile.cyrillic_letters / alpha

    if profile.latin_letters >= 20 and profile.cyrillic_letters >= 20:
        evidence = LanguageEvidence(
            "mixed", 0.0, profile, uk_hits, en_hits, "mixed-latin-cyrillic"
        )
    elif profile.cyrillic_letters >= 20 and (
        profile.ukrainian_specific_letters > 0 or uk_hits >= 2
    ):
        confidence = min(
            1.0,
            0.55
            + 0.35 * cyrillic_ratio
            + 0.05 * min(profile.ukrainian_specific_letters, 2)
            + 0.025 * min(uk_hits, 2),
        )
        evidence = LanguageEvidence(
            "uk", confidence, profile, uk_hits, en_hits, "uk-script-lexical"
        )
    elif profile.latin_letters >= 20 and latin_ratio >= 0.9 and en_hits >= 1:
        confidence = min(1.0, 0.6 + 0.3 * latin_ratio + 0.025 * min(en_hits, 4))
        evidence = LanguageEvidence(
            "en", confidence, profile, uk_hits, en_hits, "en-script-lexical"
        )
    else:
        evidence = LanguageEvidence(
            "und", 0.0, profile, uk_hits, en_hits, "insufficient-evidence"
        )

    if language_hint in {"uk", "en"} and evidence.label not in {language_hint, "und"}:
        raise MultilingualDataError(
            f"language hint {language_hint!r} conflicts with detected {evidence.label!r}"
        )
    return evidence


def admit_for_pretraining(
    record: PretrainingRecord,
    *,
    reserved_normalized_sha256: frozenset[str] = frozenset(),
) -> AdmittedRecord:
    """Admit only after provenance, split, encoding, LID, and contamination gates."""
    split = record.split.casefold()
    purpose = record.source_purpose.casefold()
    if split in _FORBIDDEN_TRAIN_SPLITS:
        raise MultilingualDataError(f"split {record.split!r} cannot enter pretraining")
    if purpose in _FORBIDDEN_SOURCE_PURPOSES:
        raise MultilingualDataError(f"source purpose {record.source_purpose!r} is held out")
    if record.external:
        if record.rights_status != RIGHTS_APPROVED or record.allows_model_training is not True:
            raise MultilingualDataError(
                "external source is not explicitly approved for model training"
            )
    elif not record.project_authored_synthetic:
        raise MultilingualDataError(
            "non-external records require explicit project-authored provenance"
        )

    normalized, _ = strict_normalize_utf8(
        record.text, preserve_layout=record.modality == "code"
    )
    fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    if fingerprint in reserved_normalized_sha256:
        raise MultilingualDataError(
            "record overlaps a reserved validation/evaluation fingerprint"
        )

    evidence = detect_language(
        normalized, modality=record.modality, language_hint=record.language_hint
    )
    if record.modality == "natural" and evidence.label not in {"uk", "en"}:
        raise MultilingualDataError(
            f"natural-language record is not confidently uk/en: {evidence.label}"
        )
    if len(normalized) < 20:
        raise MultilingualDataError("record is too short for multilingual pretraining admission")

    return AdmittedRecord(
        record.record_id,
        record.source_id,
        record.source_version,
        record.source_manifest_sha256,
        record.split,
        record.modality,
        evidence.label,
        normalized,
        fingerprint,
        evidence,
    )


def assert_no_cross_split_overlap(
    train: tuple[AdmittedRecord, ...], held_out_sha256: frozenset[str]
) -> None:
    overlap = sorted(
        record.normalized_sha256
        for record in train
        if record.normalized_sha256 in held_out_sha256
    )
    if overlap:
        raise MultilingualDataError(
            f"training records overlap held-out fingerprints: {overlap[:3]}"
        )


def build_token_budget_mixture(
    strata: tuple[MixtureStratum, ...],
    *,
    tokenizer_config_sha256: str,
    tokenizer_vocab_sha256: str,
    packing_config_sha256: str,
    seed: int,
    num_shards: int,
) -> MixturePlan:
    """Build the incumbent D10 deterministic mixture plan from multilingual strata."""
    if {stratum.name for stratum in strata} != {"uk", "en", "code"}:
        raise MultilingualDataError("mixture requires exactly uk, en and code strata")
    return MixturePlan(
        plan_id="uk-en-code-token-budget-v1",
        tokenizer_config_sha256=tokenizer_config_sha256,
        tokenizer_vocab_sha256=tokenizer_vocab_sha256,
        packing_config_sha256=packing_config_sha256,
        sources=tuple(
            MixtureSource(stratum.name, stratum.manifest_sha256, stratum.weight_units)
            for stratum in strata
        ),
        seed=seed,
        num_shards=num_shards,
    )


def default_token_budget_strata(
    manifest_by_name: dict[str, str],
) -> tuple[MixtureStratum, ...]:
    if set(manifest_by_name) != set(_SELECTION_WEIGHTS):
        raise MultilingualDataError("manifest_by_name must contain exactly uk, en and code")
    result = []
    for raw_name, weight in _SELECTION_WEIGHTS.items():
        name = cast(Literal["uk", "en", "code"], raw_name)
        result.append(MixtureStratum(name, manifest_by_name[name], weight))
    return tuple(result)


def replay_schedule(
    plan: MixturePlan,
    *,
    samples: int,
    cursor: RestartCursor | None = None,
) -> tuple[Counter[str], RestartCursor]:
    if samples < 0:
        raise MultilingualDataError("samples must be non-negative")
    current = RestartCursor.initial(plan) if cursor is None else cursor
    current.require_compatible(plan)
    counts: Counter[str] = Counter()
    for _ in range(samples):
        source, _offset = current.next_source_and_offset(plan)
        counts[source] += 1
        current = current.advance(
            plan, source_name=source, emitted_sequences=1, emitted_loss_tokens=1
        )
    return counts, current


def tokenizer_cost(
    *,
    name: str,
    vocab_size: int,
    observed_tokens: int,
    byte_baseline_tokens: int,
    d_model: int,
) -> TokenizerCost:
    if min(vocab_size, observed_tokens, byte_baseline_tokens, d_model) <= 0:
        raise MultilingualDataError("tokenizer cost inputs must be positive")
    vocab_params = vocab_size * d_model
    return TokenizerCost(
        name,
        vocab_size,
        observed_tokens,
        byte_baseline_tokens,
        1.0 - observed_tokens / byte_baseline_tokens,
        d_model,
        vocab_params,
        vocab_params - 256 * d_model,
    )


def corpus_requirements(tokens_per_parameter: int = 20) -> dict[str, dict[str, int]]:
    """Planning floors, not quality guarantees: training tokens per model parameter."""
    if tokens_per_parameter <= 0:
        raise MultilingualDataError("tokens_per_parameter must be positive")
    stages = {"1M": 1_000_000, "10M": 10_000_000, "100M": 100_000_000}
    requirements: dict[str, dict[str, int]] = {}
    for stage, parameters in stages.items():
        total = parameters * tokens_per_parameter
        requirements[stage] = {
            "total_train_tokens": total,
            "uk_train_tokens": total * 45 // 100,
            "en_train_tokens": total * 35 // 100,
            "code_train_tokens": total * 20 // 100,
            "held_out_tokens_min": max(100_000, total // 200),
        }
    return requirements


def manifest_payload(
    *,
    admitted: tuple[AdmittedRecord, ...],
    mixture_plan: MixturePlan,
    tokenizer_costs: tuple[TokenizerCost, ...],
) -> dict[str, object]:
    language_counts = Counter(record.language for record in admitted)
    payload: dict[str, object] = {
        "schema": MULTILINGUAL_SCHEMA,
        "base_pretraining_only": True,
        "instruction_tuning": False,
        "canonical_s0_mutated": False,
        "languages": dict(sorted(language_counts.items())),
        "mixture_plan_sha256": mixture_plan.sha256,
        "tokenizer_costs": [asdict(cost) for cost in tokenizer_costs],
        "corpus_requirements": corpus_requirements(),
    }
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return payload
