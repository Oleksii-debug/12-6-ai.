"""Bounded, fail-closed Ukrainian external-source ingestion for DATA-21."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from .corpus_foundation import SQLiteExactDedupIndex
from .pipeline import language_id, normalize_text

CANDIDATE_SCHEMA = "12-6.ukrainian-source-candidates.v1"
SAMPLE_MANIFEST_SCHEMA = "12-6.ukrainian-bounded-sample.v1"
DISPOSITION_ACCEPTED = "ACCEPTED"
DISPOSITION_REJECTED = "REJECTED"
DISPOSITION_BLOCKED = "BLOCKED_BY_RIGHTS"
_ALLOWED_DISPOSITIONS = frozenset(
    {DISPOSITION_ACCEPTED, DISPOSITION_REJECTED, DISPOSITION_BLOCKED}
)


class UkrainianSourceError(ValueError):
    """Raised when rights, provenance, download, or Ukrainian-text invariants fail."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UkrainianSourceError(f"{field} must be a non-empty string")
    return value.strip()


def _require_sha256(value: Any, field: str) -> str:
    text = _require_text(value, field)
    if len(text) != 64 or text != text.lower():
        raise UkrainianSourceError(f"{field} must be lowercase SHA-256")
    if any(ch not in "0123456789abcdef" for ch in text):
        raise UkrainianSourceError(f"{field} must be lowercase SHA-256")
    return text


@dataclass(frozen=True)
class SampleObject:
    path: str
    raw_size_bytes: int
    raw_sha256: str
    upstream_git_blob_sha1: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SampleObject":
        path = _require_text(value.get("path"), "sample_object.path")
        size = value.get("raw_size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise UkrainianSourceError("sample_object.raw_size_bytes must be positive")
        raw_sha = _require_sha256(value.get("raw_sha256"), "sample_object.raw_sha256")
        blob = _require_text(value.get("upstream_git_blob_sha1"), "upstream_git_blob_sha1")
        if len(blob) != 40 or any(ch not in "0123456789abcdef" for ch in blob):
            raise UkrainianSourceError("upstream_git_blob_sha1 must be lowercase SHA-1")
        if path.startswith("/") or ".." in Path(path).parts:
            raise UkrainianSourceError("sample_object.path must be repository-relative")
        return cls(path=path, raw_size_bytes=size, raw_sha256=raw_sha, upstream_git_blob_sha1=blob)


@dataclass(frozen=True)
class CandidateSource:
    source_id: str
    provider: str
    source_url: str
    upstream_version: str
    language: str
    disposition: str
    license_id: str
    training_rights_established: bool
    allows_model_training: bool | None
    benchmark_material: bool
    auto_ingest: bool
    rights_evidence_urls: tuple[str, ...]
    provenance_evidence_urls: tuple[str, ...]
    decision_reason: str
    repository_slug: str | None
    sample_objects: tuple[SampleObject, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CandidateSource":
        disposition = _require_text(value.get("disposition"), "disposition")
        if disposition not in _ALLOWED_DISPOSITIONS:
            raise UkrainianSourceError(f"unsupported disposition: {disposition}")
        rights = value.get("training_rights_established")
        if type(rights) is not bool:
            raise UkrainianSourceError("training_rights_established must be boolean")
        allows = value.get("allows_model_training")
        if allows is not None and type(allows) is not bool:
            raise UkrainianSourceError("allows_model_training must be boolean or null")
        benchmark = value.get("benchmark_material")
        auto = value.get("auto_ingest")
        if type(benchmark) is not bool or type(auto) is not bool:
            raise UkrainianSourceError("benchmark_material/auto_ingest must be boolean")
        evidence = value.get("rights_evidence_urls")
        provenance = value.get("provenance_evidence_urls")
        if not isinstance(evidence, list) or not evidence:
            raise UkrainianSourceError("rights_evidence_urls must be non-empty")
        if not isinstance(provenance, list) or not provenance:
            raise UkrainianSourceError("provenance_evidence_urls must be non-empty")
        samples_raw = value.get("sample_objects", [])
        if not isinstance(samples_raw, list):
            raise UkrainianSourceError("sample_objects must be an array")
        samples = tuple(SampleObject.from_mapping(item) for item in samples_raw)
        candidate = cls(
            source_id=_require_text(value.get("source_id"), "source_id"),
            provider=_require_text(value.get("provider"), "provider"),
            source_url=_require_text(value.get("source_url"), "source_url"),
            upstream_version=_require_text(value.get("upstream_version"), "upstream_version"),
            language=_require_text(value.get("language"), "language"),
            disposition=disposition,
            license_id=_require_text(value.get("license_id"), "license_id"),
            training_rights_established=rights,
            allows_model_training=allows,
            benchmark_material=benchmark,
            auto_ingest=auto,
            rights_evidence_urls=tuple(_require_text(x, "rights_evidence_url") for x in evidence),
            provenance_evidence_urls=tuple(
                _require_text(x, "provenance_evidence_url") for x in provenance
            ),
            decision_reason=_require_text(value.get("decision_reason"), "decision_reason"),
            repository_slug=value.get("repository_slug"),
            sample_objects=samples,
        )
        candidate._validate_policy_state()
        return candidate

    def _validate_policy_state(self) -> None:
        if self.language != "uk":
            raise UkrainianSourceError(f"{self.source_id}: DATA-21 candidate must be Ukrainian")
        if self.disposition == DISPOSITION_ACCEPTED:
            if not self.training_rights_established or self.allows_model_training is not True:
                raise UkrainianSourceError(
                    f"{self.source_id}: accepted source requires established model-training rights"
                )
            if self.benchmark_material or not self.auto_ingest:
                raise UkrainianSourceError(
                    f"{self.source_id}: accepted source cannot be benchmark and must be auto_ingest"
                )
            if self.license_id.upper() == "NOASSERTION":
                raise UkrainianSourceError(f"{self.source_id}: accepted license cannot be NOASSERTION")
            if not self.repository_slug or not self.sample_objects:
                raise UkrainianSourceError(
                    f"{self.source_id}: accepted bounded source requires repository and sample objects"
                )
        else:
            if self.auto_ingest:
                raise UkrainianSourceError(
                    f"{self.source_id}: non-accepted candidates must remain fail-closed"
                )

    def assert_fetchable(self) -> None:
        if self.disposition != DISPOSITION_ACCEPTED:
            raise UkrainianSourceError(
                f"{self.source_id}: source is {self.disposition}; downloader must not run"
            )
        self._validate_policy_state()


def load_candidate_registry(path: str | Path) -> tuple[CandidateSource, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != CANDIDATE_SCHEMA:
        raise UkrainianSourceError("unsupported Ukrainian candidate registry schema")
    raw = payload.get("candidates")
    if not isinstance(raw, list) or not raw:
        raise UkrainianSourceError("candidate registry must be non-empty")
    candidates = tuple(CandidateSource.from_mapping(item) for item in raw)
    ids = [item.source_id for item in candidates]
    if len(ids) != len(set(ids)):
        raise UkrainianSourceError("duplicate source_id in candidate registry")
    core = {"schema_version": CANDIDATE_SCHEMA, "candidates": raw}
    expected = _sha256_bytes(_canonical_json_bytes(core))
    if payload.get("registry_identity_sha256") != expected:
        raise UkrainianSourceError("candidate registry identity mismatch")
    return candidates


def candidate_counts(candidates: Sequence[CandidateSource]) -> dict[str, int]:
    counter = Counter(item.disposition for item in candidates)
    return {
        "candidate": len(candidates),
        "accepted": counter[DISPOSITION_ACCEPTED],
        "rejected": counter[DISPOSITION_REJECTED],
        "blocked_by_rights": counter[DISPOSITION_BLOCKED],
    }


def ukrainian_script_metrics(text: str) -> dict[str, int | float | str]:
    alpha = latin = cyrillic = ukrainian_specific = 0
    for char in text:
        if not char.isalpha():
            continue
        alpha += 1
        name = unicodedata.name(char, "")
        if "CYRILLIC" in name:
            cyrillic += 1
            if char.casefold() in {"і", "ї", "є", "ґ"}:
                ukrainian_specific += 1
        elif "LATIN" in name:
            latin += 1
    return {
        "language": language_id(text),
        "alpha_chars": alpha,
        "latin_alpha_chars": latin,
        "cyrillic_alpha_chars": cyrillic,
        "cyrillic_alpha_ratio": (cyrillic / alpha if alpha else 0.0),
        "ukrainian_specific_chars": ukrainian_specific,
    }


def validate_ukrainian_text(text: str) -> dict[str, int | float | str]:
    metrics = ukrainian_script_metrics(text)
    if len(text) < 60:
        raise UkrainianSourceError("document too short for bounded Ukrainian sample")
    if metrics["language"] != "uk":
        raise UkrainianSourceError(f"language/script validation failed: {metrics}")
    if float(metrics["cyrillic_alpha_ratio"]) < 0.85:
        raise UkrainianSourceError(f"Cyrillic ratio below 0.85: {metrics}")
    if int(metrics["ukrainian_specific_chars"]) < 1:
        raise UkrainianSourceError(f"no Ukrainian-specific Cyrillic evidence: {metrics}")
    return metrics


Fetcher = Callable[[CandidateSource, SampleObject], bytes]


def github_raw_fetch(candidate: CandidateSource, obj: SampleObject) -> bytes:
    candidate.assert_fetchable()
    commit = candidate.upstream_version
    if not commit.startswith("git:") or len(commit) != 44:
        raise UkrainianSourceError("accepted GitHub candidate must pin git:<40-hex-commit>")
    revision = commit.removeprefix("git:")
    url = (
        f"https://raw.githubusercontent.com/{candidate.repository_slug}/{revision}/"
        + quote(obj.path, safe="/")
    )
    request = Request(url, headers={"User-Agent": "12-6-data21-bounded-source/1"})
    with urlopen(request, timeout=30) as response:
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > obj.raw_size_bytes:
            raise UkrainianSourceError("upstream object exceeds pinned size before read")
        payload = response.read(obj.raw_size_bytes + 1)
    if len(payload) != obj.raw_size_bytes:
        raise UkrainianSourceError(
            f"{obj.path}: size mismatch expected={obj.raw_size_bytes} got={len(payload)}"
        )
    if _sha256_bytes(payload) != obj.raw_sha256:
        raise UkrainianSourceError(f"{obj.path}: SHA-256 mismatch")
    git_sha = hashlib.sha1(
        f"blob {len(payload)}\0".encode("ascii") + payload, usedforsecurity=False
    ).hexdigest()
    if git_sha != obj.upstream_git_blob_sha1:
        raise UkrainianSourceError(f"{obj.path}: Git blob identity mismatch")
    return payload


def ingest_bounded_candidate(
    candidate: CandidateSource,
    output_jsonl: str | Path,
    manifest_path: str | Path,
    dedup_db: str | Path,
    *,
    fetcher: Fetcher = github_raw_fetch,
) -> dict[str, Any]:
    """Download only pinned bounded objects and stage normalized exact-unique JSONL."""
    candidate.assert_fetchable()  # rights gate occurs before the first fetch
    output = Path(output_jsonl)
    manifest_file = Path(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    receipts: list[dict[str, Any]] = []
    with SQLiteExactDedupIndex(dedup_db) as dedup, output.open("wb") as writer:
        for obj in candidate.sample_objects:
            counters["candidate"] += 1
            raw = fetcher(candidate, obj)
            if len(raw) != obj.raw_size_bytes or _sha256_bytes(raw) != obj.raw_sha256:
                raise UkrainianSourceError(f"{obj.path}: fetcher returned unpinned bytes")
            try:
                decoded = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                counters["rejected"] += 1
                raise UkrainianSourceError(f"{obj.path}: not valid UTF-8") from exc
            normalized = normalize_text(decoded)
            try:
                metrics = validate_ukrainian_text(normalized)
            except UkrainianSourceError:
                counters["rejected"] += 1
                raise
            content_sha = _sha256_bytes(normalized.encode("utf-8"))
            if dedup.seen_or_add(content_sha):
                counters["rejected"] += 1
                counters["exact_duplicate"] += 1
                continue
            document_id = _sha256_bytes(
                (
                    f"{candidate.source_id}\0{candidate.upstream_version}\0"
                    f"{obj.path}\0{content_sha}"
                ).encode("utf-8")
            )
            record = {
                "document_id": document_id,
                "text": normalized,
                "language_hint": "uk",
                "content_sha256": content_sha,
                "source_path": obj.path,
                "upstream_git_blob_sha1": obj.upstream_git_blob_sha1,
            }
            writer.write(_canonical_json_bytes(record))
            counters["accepted"] += 1
            receipts.append(
                {
                    "path": obj.path,
                    "raw_size_bytes": len(raw),
                    "raw_sha256": _sha256_bytes(raw),
                    "upstream_git_blob_sha1": obj.upstream_git_blob_sha1,
                    "normalized_size_bytes": len(normalized.encode("utf-8")),
                    "normalized_sha256": content_sha,
                    **metrics,
                }
            )
        dedup.commit()
    sample_bytes = output.read_bytes()
    core = {
        "schema_version": SAMPLE_MANIFEST_SCHEMA,
        "source_id": candidate.source_id,
        "upstream_version": candidate.upstream_version,
        "rights_disposition": candidate.disposition,
        "license_id": candidate.license_id,
        "counts": {
            "candidate": counters["candidate"],
            "accepted": counters["accepted"],
            "rejected": counters["rejected"],
            "blocked_by_rights": 0,
        },
        "download_receipts": receipts,
        "sample_jsonl": {
            "path": output.as_posix(),
            "size_bytes": len(sample_bytes),
            "sha256": _sha256_bytes(sample_bytes),
        },
        "normalization": "twelve_six.data.pipeline.normalize_text/NFKC-whitespace-v1",
        "language_validation": "pipeline.language_id + Cyrillic>=0.85 + Ukrainian-specific-char>=1",
        "dedup_staging": "SQLiteExactDedupIndex(content_sha256)",
    }
    manifest = {**core, "manifest_sha256": _sha256_bytes(_canonical_json_bytes(core))}
    manifest_file.write_bytes(_canonical_json_bytes(manifest))
    return manifest


def _main() -> int:
    parser = argparse.ArgumentParser(description="Bounded Ukrainian source ingestion")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dedup-db", type=Path, required=True)
    args = parser.parse_args()
    candidates = load_candidate_registry(args.registry)
    match = next((item for item in candidates if item.source_id == args.source_id), None)
    if match is None:
        raise SystemExit(f"unknown source_id: {args.source_id}")
    manifest = ingest_bounded_candidate(
        match, args.output_jsonl, args.manifest, args.dedup_db
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
