"""Rights-aware bounded intake for reviewed external text sources.

This is a thin acquisition/extraction front-end over the incumbent D03/DATA-10
normalization, language validation and exact-dedup seams. Candidate discovery is
not canonical corpus promotion: only ELIGIBLE sources are fetched, and canonical
external-source snapshots remain governed by ``external_sources.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from twelve_six.data.corpus_foundation import SQLiteExactDedupIndex
from twelve_six.data.external_sources import (
    RIGHTS_APPROVED,
    RIGHTS_REJECTED,
    RIGHTS_REVIEW_REQUIRED,
    RightsDecision,
)
from twelve_six.data.multilingual_pretraining import (
    LanguageEvidence,
    MultilingualDataError,
    detect_language,
    strict_normalize_utf8,
)
from twelve_six.data.scalable_ingestion import DATATROVE_VERSION

CANDIDATE_SCHEMA = "12-6.external-source-candidates.v1"
INTAKE_MANIFEST_SCHEMA = "12-6.external-source-intake-manifest.v1"
ELIGIBLE = "ELIGIBLE"
BLOCKED_BY_RIGHTS = "BLOCKED_BY_RIGHTS"
_ALLOWED_ELIGIBILITY = frozenset({ELIGIBLE, BLOCKED_BY_RIGHTS})
_ALLOWED_ADAPTERS = frozenset({"html_text", "plain_text"})
_ALLOWED_LANGUAGES = frozenset({"uk", "en"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHARSET_RE = re.compile(
    br"""charset\s*=\s*["']?\s*([A-Za-z0-9._-]+)""", re.IGNORECASE
)
_UK_REVIEWED_MIN_CYRILLIC_RATIO = 0.85
_UK_REVIEWED_MIN_LEXICAL_HITS = 2
_UK_REVIEWED_MIN_SPECIFIC_LETTERS = 2


class SourceIntakeError(ValueError):
    """Raised when candidate intake metadata or bounded acquisition is unsafe."""


@dataclass(frozen=True)
class CandidateSource:
    source_id: str
    source_version: str
    language: str
    provider: str
    source_url: str
    source_kind: str
    purpose: str
    adapter: str | None
    acquisition_urls: tuple[str, ...]
    expected_language: str
    rights: RightsDecision
    rights_basis: tuple[str, ...]
    allows_commercial_use: bool
    eligibility_status: str
    block_reason: str | None

    def __post_init__(self) -> None:
        for field in (
            "source_id",
            "source_version",
            "language",
            "provider",
            "source_url",
            "source_kind",
            "purpose",
            "expected_language",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise SourceIntakeError(f"{field} must be a non-empty string")
        if self.language not in _ALLOWED_LANGUAGES:
            raise SourceIntakeError(f"{self.source_id}: unsupported language {self.language!r}")
        if self.expected_language != self.language:
            raise SourceIntakeError(
                f"{self.source_id}: expected_language must equal declared language"
            )
        if self.eligibility_status not in _ALLOWED_ELIGIBILITY:
            raise SourceIntakeError(
                f"{self.source_id}: unsupported eligibility status {self.eligibility_status!r}"
            )
        if type(self.allows_commercial_use) is not bool:
            raise SourceIntakeError("allows_commercial_use must be an exact boolean")
        if not self.rights_basis:
            raise SourceIntakeError(f"{self.source_id}: rights basis must not be empty")

        if self.eligibility_status == ELIGIBLE:
            if self.adapter not in _ALLOWED_ADAPTERS:
                raise SourceIntakeError(
                    f"{self.source_id}: eligible source requires a supported adapter"
                )
            if not self.acquisition_urls:
                raise SourceIntakeError(
                    f"{self.source_id}: eligible source requires bounded acquisition URLs"
                )
            if (
                self.rights.status != RIGHTS_APPROVED
                or self.rights.allows_model_training is not True
            ):
                raise SourceIntakeError(
                    f"{self.source_id}: ELIGIBLE requires explicit model-training approval"
                )
            self.rights.__post_init__()
        else:
            if self.acquisition_urls:
                raise SourceIntakeError(
                    f"{self.source_id}: rights-blocked source must not be fetchable"
                )
            if self.adapter is not None:
                raise SourceIntakeError(
                    f"{self.source_id}: rights-blocked source must not have an adapter"
                )
            if not self.block_reason:
                raise SourceIntakeError(
                    f"{self.source_id}: rights-blocked source requires a blocker"
                )
            if self.rights.status == RIGHTS_APPROVED:
                raise SourceIntakeError(
                    f"{self.source_id}: approved rights cannot be marked blocked-by-rights"
                )

    @property
    def source_identity_sha256(self) -> str:
        payload = {
            "source_id": self.source_id,
            "source_version": self.source_version,
            "source_url": self.source_url,
            "rights": asdict(self.rights),
            "rights_basis": list(self.rights_basis),
            "eligibility_status": self.eligibility_status,
        }
        return _sha256_json(payload)


@dataclass(frozen=True)
class DownloadedBytes:
    payload: bytes
    content_type: str | None = None


Fetcher = Callable[[str, int], DownloadedBytes]


class _TextHTMLParser(HTMLParser):
    _BLOCKS = frozenset(
        {
            "address",
            "article",
            "aside",
            "blockquote",
            "br",
            "dd",
            "div",
            "dl",
            "dt",
            "figcaption",
            "footer",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "header",
            "li",
            "main",
            "nav",
            "p",
            "pre",
            "section",
            "table",
            "td",
            "th",
            "tr",
        }
    )
    _SKIP = frozenset({"script", "style", "noscript", "svg"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.casefold()
        if lowered in self._SKIP:
            self._skip_depth += 1
        elif self._skip_depth == 0 and lowered in self._BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered in self._SKIP:
            if self._skip_depth:
                self._skip_depth -= 1
        elif self._skip_depth == 0 and lowered in self._BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        lines = []
        for line in "".join(self.parts).splitlines():
            collapsed = " ".join(line.split())
            if collapsed:
                lines.append(collapsed)
        return "\n".join(lines)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceIntakeError(f"{field} must be an object")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise SourceIntakeError(f"{field} must be an array")
    return value


def _candidate_from_mapping(raw: Mapping[str, Any]) -> CandidateSource:
    rights_raw = _require_mapping(raw.get("rights"), "rights")
    basis = tuple(str(item) for item in _require_list(rights_raw.get("basis"), "rights.basis"))
    rights = RightsDecision(
        status=rights_raw.get("status"),
        license_id=rights_raw.get("license_id"),
        terms_url=rights_raw.get("terms_url"),
        allows_model_training=rights_raw.get("allows_model_training"),
        allows_derivatives=rights_raw.get("allows_derivatives"),
        allows_redistribution=rights_raw.get("allows_redistribution"),
        policy_ref=rights_raw.get("policy_ref"),
        reviewed_at=rights_raw.get("reviewed_at"),
        reviewer_ref=rights_raw.get("reviewer_ref"),
    )
    acquisition_urls = tuple(
        str(item) for item in _require_list(raw.get("acquisition_urls"), "acquisition_urls")
    )
    return CandidateSource(
        source_id=raw.get("source_id"),
        source_version=raw.get("source_version"),
        language=raw.get("language"),
        provider=raw.get("provider"),
        source_url=raw.get("source_url"),
        source_kind=raw.get("source_kind"),
        purpose=raw.get("purpose"),
        adapter=raw.get("adapter"),
        acquisition_urls=acquisition_urls,
        expected_language=raw.get("expected_language"),
        rights=rights,
        rights_basis=basis,
        allows_commercial_use=rights_raw.get("allows_commercial_use"),
        eligibility_status=raw.get("eligibility_status"),
        block_reason=raw.get("block_reason"),
    )


def validate_candidate_registry(registry: Mapping[str, Any]) -> tuple[CandidateSource, ...]:
    """Validate canonical candidate identity and the fail-closed rights/fetch boundary."""
    if registry.get("schema_version") != CANDIDATE_SCHEMA:
        raise SourceIntakeError("unsupported candidate registry schema")
    if registry.get("authority_boundary") != (
        "CANDIDATE_DISCOVERY_AND_RIGHTS_REVIEW_NOT_CANONICAL_CORPUS_APPROVAL"
    ):
        raise SourceIntakeError("candidate registry authority boundary changed")
    raw_sources = _require_list(registry.get("sources"), "sources")
    sources = tuple(
        _candidate_from_mapping(_require_mapping(item, "source")) for item in raw_sources
    )
    keys = [(item.source_id, item.source_version) for item in sources]
    if len(keys) != len(set(keys)):
        raise SourceIntakeError("duplicate source_id/source_version")

    core = dict(registry)
    claimed = core.pop("registry_identity_sha256", None)
    expected = _sha256_json(core)
    if claimed != expected:
        raise SourceIntakeError("candidate registry identity mismatch")
    return sources


def load_candidate_registry(
    path: str | Path,
) -> tuple[Mapping[str, Any], tuple[CandidateSource, ...]]:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(registry, Mapping):
        raise SourceIntakeError("candidate registry root must be an object")
    return registry, validate_candidate_registry(registry)


def bounded_http_fetch(url: str, max_bytes: int) -> DownloadedBytes:
    """Fetch one explicitly allowlisted candidate object with a hard byte ceiling."""
    if max_bytes <= 0:
        raise SourceIntakeError("max_bytes must be positive")
    request = Request(
        url,
        headers={
            "User-Agent": "12-6-data-intake/1.0 (+rights-aware bounded research)",
            "Accept-Encoding": "identity",
        },
    )
    with urlopen(request, timeout=30) as response:
        length = response.headers.get("Content-Length")
        if length is not None:
            try:
                declared = int(length)
            except ValueError as exc:
                raise SourceIntakeError("invalid Content-Length") from exc
            if declared > max_bytes:
                raise SourceIntakeError(
                    f"remote object declares {declared} bytes > max_bytes={max_bytes}"
                )
        payload = response.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise SourceIntakeError(f"download exceeded max_bytes={max_bytes}")
        return DownloadedBytes(payload, response.headers.get("Content-Type"))


def _decode_payload(downloaded: DownloadedBytes) -> tuple[str, str]:
    payload = downloaded.payload
    declared: str | None = None
    match = _CHARSET_RE.search(payload[:8192])
    if match:
        declared = match.group(1).decode("ascii", errors="ignore").casefold()
    if declared is None and downloaded.content_type:
        header_match = re.search(
            r"charset\s*=\s*([A-Za-z0-9._-]+)",
            downloaded.content_type,
            flags=re.IGNORECASE,
        )
        if header_match:
            declared = header_match.group(1).casefold()

    aliases = {
        "windows-1251": "cp1251",
        "win-1251": "cp1251",
        "1251": "cp1251",
        "utf8": "utf-8",
    }
    candidates = []
    for encoding in (declared, "utf-8", "cp1251"):
        if encoding:
            normalized = aliases.get(encoding, encoding)
            if normalized not in candidates:
                candidates.append(normalized)
    for encoding in candidates:
        try:
            return payload.decode(encoding, errors="strict"), encoding
        except (UnicodeDecodeError, LookupError):
            continue
    raise SourceIntakeError("payload cannot be decoded as declared/UTF-8/CP1251")


def extract_text(downloaded: DownloadedBytes, adapter: str) -> tuple[str, str]:
    decoded, encoding = _decode_payload(downloaded)
    if adapter == "plain_text":
        return decoded, encoding
    if adapter == "html_text":
        parser = _TextHTMLParser()
        parser.feed(decoded)
        parser.close()
        return parser.text(), encoding
    raise SourceIntakeError(f"unsupported adapter: {adapter}")


def _record_id(source: CandidateSource, acquisition_url: str) -> str:
    identity = f"{source.source_id}\0{source.source_version}\0{acquisition_url}".encode()
    return "ext-" + hashlib.sha256(identity).hexdigest()[:32]


def _validated_source_language(text: str, source: CandidateSource) -> LanguageEvidence:
    """Validate source LID, with one conservative reviewed-source Ukrainian override.

    The global multilingual detector intentionally treats any sizeable Latin+Cyrillic
    mixture as ``mixed``. Long official Ukrainian legal text can cross the absolute
    Latin threshold through Roman amendment identifiers while remaining overwhelmingly
    Ukrainian. Only an already rights-approved, explicitly Ukrainian intake source may
    reinterpret that result, and only with dominant Cyrillic plus independent Ukrainian
    lexical and orthographic evidence. Balanced mixed text remains rejected.
    """

    evidence = detect_language(text, modality="natural", language_hint=None)
    if evidence.label == source.expected_language:
        return evidence

    if source.expected_language == "uk" and evidence.label == "mixed":
        alpha = max(evidence.script.alphabetic_letters, 1)
        cyrillic_ratio = evidence.script.cyrillic_letters / alpha
        reviewed_uk = (
            source.eligibility_status == ELIGIBLE
            and source.rights.status == RIGHTS_APPROVED
            and source.rights.allows_model_training is True
        )
        strong_uk_evidence = (
            evidence.ukrainian_lexical_hits >= _UK_REVIEWED_MIN_LEXICAL_HITS
            and evidence.script.ukrainian_specific_letters
            >= _UK_REVIEWED_MIN_SPECIFIC_LETTERS
        )
        if (
            reviewed_uk
            and cyrillic_ratio >= _UK_REVIEWED_MIN_CYRILLIC_RATIO
            and strong_uk_evidence
        ):
            confidence = min(
                1.0,
                0.55
                + 0.35 * cyrillic_ratio
                + 0.05 * min(evidence.script.ukrainian_specific_letters, 2)
                + 0.025 * min(evidence.ukrainian_lexical_hits, 2),
            )
            return LanguageEvidence(
                "uk",
                confidence,
                evidence.script,
                evidence.ukrainian_lexical_hits,
                evidence.english_lexical_hits,
                "uk-reviewed-source-dominant-cyrillic",
            )

    raise MultilingualDataError(
        f"expected {source.expected_language}, detected {evidence.label}"
    )


def run_bounded_intake(
    registry: Mapping[str, Any],
    output_dir: str | Path,
    *,
    fetcher: Fetcher = bounded_http_fetch,
    max_download_bytes: int = 2_000_000,
    max_normalized_chars: int = 50_000,
) -> dict[str, Any]:
    """Process only reviewed eligible bytes through incumbent normalization/LID/dedup."""
    if max_download_bytes <= 0 or max_normalized_chars <= 0:
        raise SourceIntakeError("bounded limits must be positive")
    sources = validate_candidate_registry(registry)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    text_dir = output / "accepted_text"
    text_dir.mkdir(exist_ok=True)
    dedup_path = output / "exact-dedup.sqlite3"
    records_path = output / "records.jsonl"

    source_counts = {
        "candidate": len(sources),
        "eligible": sum(item.eligibility_status == ELIGIBLE for item in sources),
        "blocked_by_rights": sum(
            item.eligibility_status == BLOCKED_BY_RIGHTS for item in sources
        ),
        "rights_rejected": sum(item.rights.status == RIGHTS_REJECTED for item in sources),
        "rights_review_required": sum(
            item.rights.status == RIGHTS_REVIEW_REQUIRED for item in sources
        ),
    }
    blocked_sources = [
        {
            "source_id": item.source_id,
            "source_version": item.source_version,
            "rights_status": item.rights.status,
            "license_id": item.rights.license_id,
            "failure_reason": item.block_reason,
        }
        for item in sources
        if item.eligibility_status == BLOCKED_BY_RIGHTS
    ]

    attempted = accepted = rejected = duplicates = 0
    raw_downloaded_bytes = accepted_utf8_bytes = 0
    accepted_by_language = {"uk": 0, "en": 0}
    accepted_bytes_by_language = {"uk": 0, "en": 0}
    record_results: list[dict[str, Any]] = []

    with SQLiteExactDedupIndex(dedup_path) as dedup, records_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as records_file:
        for source in sources:
            if source.eligibility_status != ELIGIBLE:
                continue
            source.rights.__post_init__()
            if source.rights.status != RIGHTS_APPROVED or not source.rights.allows_model_training:
                raise SourceIntakeError(
                    f"{source.source_id}: fail-closed rights gate changed after validation"
                )
            for acquisition_url in source.acquisition_urls:
                attempted += 1
                record_id = _record_id(source, acquisition_url)
                try:
                    downloaded = fetcher(acquisition_url, max_download_bytes)
                    if len(downloaded.payload) > max_download_bytes:
                        raise SourceIntakeError("fetcher violated max_download_bytes contract")
                    raw_downloaded_bytes += len(downloaded.payload)
                    extracted, encoding = extract_text(downloaded, source.adapter or "")
                    bounded = extracted[:max_normalized_chars]
                    normalized, _profile = strict_normalize_utf8(bounded)
                    evidence = _validated_source_language(normalized, source)
                    normalized_sha = _sha256_bytes(normalized.encode("utf-8"))
                    if dedup.seen_or_add(normalized_sha):
                        duplicates += 1
                        rejected += 1
                        record_results.append(
                            {
                                "record_id": record_id,
                                "source_id": source.source_id,
                                "status": "REJECTED",
                                "failure_reason": "EXACT_DUPLICATE_NORMALIZED_SHA256",
                                "normalized_sha256": normalized_sha,
                            }
                        )
                        continue

                    raw_sha = _sha256_bytes(downloaded.payload)
                    normalized_bytes = normalized.encode("utf-8")
                    record = {
                        "id": record_id,
                        "source_id": source.source_id,
                        "source_version": source.source_version,
                        "source_identity_sha256": source.source_identity_sha256,
                        "acquisition_url": acquisition_url,
                        "raw_sha256": raw_sha,
                        "raw_bytes": len(downloaded.payload),
                        "decoded_encoding": encoding,
                        "content_sha256": normalized_sha,
                        "normalized_utf8_bytes": len(normalized_bytes),
                        "language": evidence.label,
                        "language_confidence": evidence.confidence,
                        "language_reason": evidence.reason,
                        "rights_status": source.rights.status,
                        "license_id": source.rights.license_id,
                        "allows_model_training": source.rights.allows_model_training,
                        "text_path": f"accepted_text/{record_id}.txt",
                    }
                    (text_dir / f"{record_id}.txt").write_text(
                        normalized + "\n", encoding="utf-8", newline="\n"
                    )
                    records_file.write(
                        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    record_results.append({**record, "status": "ACCEPTED"})
                    accepted += 1
                    accepted_utf8_bytes += len(normalized_bytes)
                    accepted_by_language[evidence.label] += 1
                    accepted_bytes_by_language[evidence.label] += len(normalized_bytes)
                except (OSError, ValueError) as exc:
                    rejected += 1
                    record_results.append(
                        {
                            "record_id": record_id,
                            "source_id": source.source_id,
                            "source_version": source.source_version,
                            "acquisition_url": acquisition_url,
                            "status": "REJECTED",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
        dedup.commit()

    datatrove_handoff = {
        "runtime_version": DATATROVE_VERSION,
        "input_format": "jsonl",
        "input_path": "records.jsonl",
        "exact_dedup_engine": "incumbent_SQLiteExactDedupIndex",
        "exact_dedup_completed": True,
        "near_dedup_engine": "datatrove_minhash",
        "near_dedup_status": "NOT_RUN_BOUNDED_SAMPLE",
        "canonical_snapshot_promotion": "NOT_PERFORMED",
    }
    manifest_core = {
        "schema_version": INTAKE_MANIFEST_SCHEMA,
        "candidate_registry_identity_sha256": registry["registry_identity_sha256"],
        "authority_boundary": (
            "REAL_BOUNDED_SAMPLE_NOT_CANONICAL_CORPUS_FREEZE_OR_SOURCE_SNAPSHOT_PROMOTION"
        ),
        "limits": {
            "max_download_bytes_per_object": max_download_bytes,
            "max_normalized_chars_per_object": max_normalized_chars,
        },
        "source_counts": source_counts,
        "record_counts": {
            "attempted": attempted,
            "accepted": accepted,
            "rejected": rejected,
            "exact_duplicates": duplicates,
        },
        "byte_counts": {
            "raw_downloaded_bytes": raw_downloaded_bytes,
            "accepted_normalized_utf8_bytes": accepted_utf8_bytes,
            "accepted_normalized_utf8_bytes_by_language": accepted_bytes_by_language,
        },
        "accepted_records_by_language": accepted_by_language,
        "blocked_sources": blocked_sources,
        "records": record_results,
        "datatrove_handoff": datatrove_handoff,
    }
    manifest = {**manifest_core, "manifest_sha256": _sha256_json(manifest_core)}
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
