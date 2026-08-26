"""Deterministic staging for immutable Wikipedia XML dump snapshots.

This module deliberately stops at provenance-preserving raw-wikitext materialization.
Downstream quality, privacy, deduplication, evaluation decontamination, parsing,
tokenization, packing, and training authorization remain separate gates.
"""

from __future__ import annotations

import bz2
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DUMP_DATE = re.compile(r"^\d{8}$")
_LANGUAGE = re.compile(r"^[a-z][a-z0-9-]{0,15}$")


class WikimediaIngestionError(ValueError):
    """Raised when a Wikipedia snapshot or materialization is unsafe or inconsistent."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(element: ET.Element, name: str) -> ET.Element | None:
    for item in element:
        if _local_name(item.tag) == name:
            return item
    return None


def _child_text(element: ET.Element, name: str) -> str | None:
    item = _child(element, name)
    return None if item is None else item.text


def _positive_limit(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WikimediaIngestionError(f"{name} must be a positive integer or None")
    return value


@dataclass(frozen=True)
class WikipediaDumpPlan:
    """Authority for staging one exact Wikipedia pages-articles XML/BZ2 snapshot."""

    source_id: str
    language: str
    dump_date: str
    dump_url: str
    dump_filename: str
    snapshot_sha256: str
    rights_authority_id: str
    license_id: str = "CC-BY-SA-4.0"
    namespace: int = 0
    exclude_redirects: bool = True

    def __post_init__(self) -> None:
        for field in ("source_id", "rights_authority_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise WikimediaIngestionError(f"{field} must be a non-empty string")
        if not _LANGUAGE.fullmatch(self.language):
            raise WikimediaIngestionError("language must be a canonical Wikipedia language code")
        if not _DUMP_DATE.fullmatch(self.dump_date):
            raise WikimediaIngestionError("dump_date must be YYYYMMDD")
        if not self.dump_filename.endswith(".xml.bz2"):
            raise WikimediaIngestionError("dump_filename must end in .xml.bz2")
        if not _HEX64.fullmatch(self.snapshot_sha256):
            raise WikimediaIngestionError("snapshot_sha256 must be lowercase SHA-256 hex")
        if self.license_id != "CC-BY-SA-4.0":
            raise WikimediaIngestionError(
                "Wikipedia text staging is pinned to CC-BY-SA-4.0 until rights policy is revalidated"
            )
        if isinstance(self.namespace, bool) or not isinstance(self.namespace, int):
            raise WikimediaIngestionError("namespace must be an integer")

        parsed = urlsplit(self.dump_url)
        if parsed.scheme != "https" or parsed.netloc != "dumps.wikimedia.org":
            raise WikimediaIngestionError("dump_url must use https://dumps.wikimedia.org")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise WikimediaIngestionError("dump_url must not contain credentials/query/fragment")
        project = f"{self.language}wiki"
        expected_path = f"/{project}/{self.dump_date}/{self.dump_filename}"
        if parsed.path != expected_path:
            raise WikimediaIngestionError(
                f"dump_url path must exactly match immutable dump identity {expected_path}"
            )

    def manifest(self) -> dict[str, Any]:
        core = {
            "schema_version": "12-6.wikipedia-dump-plan.v1",
            **asdict(self),
            "project": f"{self.language}wiki",
            "content_state": "RAW_WIKITEXT_STAGING_ONLY",
            "training_authorized": False,
            "required_downstream_gates": [
                "WIKITEXT_TO_TEXT_NORMALIZATION",
                "QUALITY_LANGUAGE_PRIVACY",
                "GLOBAL_EXACT_NEAR_DEDUP",
                "EVALUATION_DECONTAMINATION",
                "CLUSTER_SAFE_SPLIT",
                "TOKENIZE_PACK_UNIQUE_LOSS_LEDGER",
            ],
        }
        return {
            **core,
            "plan_sha256": hashlib.sha256(_canonical_json_bytes(core)).hexdigest(),
        }


@dataclass(frozen=True)
class WikimediaMaterializationResult:
    """Identity and accounting for one bounded raw-wikitext staging artifact."""

    plan_sha256: str
    snapshot_sha256: str
    record_count: int
    text_utf8_bytes: int
    output_sha256: str
    inventory_sha256: str
    training_authorized: bool = False


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a local snapshot without loading it into memory."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise WikimediaIngestionError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dump_snapshot(path: str | Path, plan: WikipediaDumpPlan) -> None:
    """Fail closed unless local bytes match the immutable dump authority."""

    actual = sha256_file(path)
    if actual != plan.snapshot_sha256:
        raise WikimediaIngestionError(
            f"snapshot SHA-256 mismatch: expected {plan.snapshot_sha256}, got {actual}"
        )


def iter_wikimedia_revision_records(
    path: str | Path,
    plan: WikipediaDumpPlan,
    *,
    max_records: int | None = None,
    max_text_bytes: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream canonical main-namespace revision records from an XML/BZ2 dump.

    Call :func:`verify_dump_snapshot` first unless the caller has already bound the
    local snapshot bytes. The emitted text remains raw wikitext and is staging-only.
    """

    max_records = _positive_limit("max_records", max_records)
    max_text_bytes = _positive_limit("max_text_bytes", max_text_bytes)
    emitted = 0
    emitted_bytes = 0
    project = f"{plan.language}wiki"

    with bz2.open(Path(path), "rb") as stream:
        for _event, element in ET.iterparse(stream, events=("end",)):
            if _local_name(element.tag) != "page":
                continue
            try:
                namespace_text = _child_text(element, "ns")
                if namespace_text is None:
                    raise WikimediaIngestionError("page is missing namespace")
                try:
                    namespace = int(namespace_text)
                except ValueError as exc:
                    raise WikimediaIngestionError("page namespace must be an integer") from exc
                if namespace != plan.namespace:
                    continue
                if plan.exclude_redirects and _child(element, "redirect") is not None:
                    continue

                title = _child_text(element, "title")
                page_id_text = _child_text(element, "id")
                revision = _child(element, "revision")
                if title is None or page_id_text is None or revision is None:
                    raise WikimediaIngestionError("page is missing title/id/revision")
                revision_id_text = _child_text(revision, "id")
                timestamp = _child_text(revision, "timestamp")
                text_element = _child(revision, "text")
                if revision_id_text is None or timestamp is None or text_element is None:
                    raise WikimediaIngestionError(
                        f"page {page_id_text} is missing revision id/timestamp/text"
                    )

                try:
                    page_id = int(page_id_text)
                    revision_id = int(revision_id_text)
                except ValueError as exc:
                    raise WikimediaIngestionError("page/revision ids must be integers") from exc
                if page_id <= 0 or revision_id <= 0:
                    raise WikimediaIngestionError("page/revision ids must be positive")

                text = text_element.text or ""
                text_payload = text.encode("utf-8")
                text_bytes = len(text_payload)
                if max_text_bytes is not None and emitted_bytes + text_bytes > max_text_bytes:
                    break

                upstream_sha1 = _child_text(revision, "sha1")
                record = {
                    "schema_version": "12-6.wikipedia-raw-record.v1",
                    "source_id": plan.source_id,
                    "project": project,
                    "language": plan.language,
                    "dump_date": plan.dump_date,
                    "page_id": page_id,
                    "revision_id": revision_id,
                    "title": title,
                    "namespace": plan.namespace,
                    "timestamp": timestamp,
                    "upstream_sha1": upstream_sha1,
                    "text": text,
                    "text_sha256": hashlib.sha256(text_payload).hexdigest(),
                    "attribution_url": (
                        f"https://{plan.language}.wikipedia.org/w/index.php?oldid={revision_id}"
                    ),
                    "license_id": plan.license_id,
                    "rights_authority_id": plan.rights_authority_id,
                    "content_state": "RAW_WIKITEXT_STAGING_ONLY",
                    "training_authorized": False,
                }
                yield record
                emitted += 1
                emitted_bytes += text_bytes
                if max_records is not None and emitted >= max_records:
                    break
            finally:
                element.clear()


def materialize_wikimedia_jsonl(
    snapshot_path: str | Path,
    output_path: str | Path,
    plan: WikipediaDumpPlan,
    *,
    max_records: int | None = None,
    max_text_bytes: int | None = None,
) -> WikimediaMaterializationResult:
    """Verify, stream, and atomically materialize a bounded canonical JSONL artifact."""

    max_records = _positive_limit("max_records", max_records)
    max_text_bytes = _positive_limit("max_text_bytes", max_text_bytes)
    snapshot = Path(snapshot_path)
    destination = Path(output_path)
    if snapshot.resolve() == destination.resolve():
        raise WikimediaIngestionError("output_path must differ from snapshot_path")
    verify_dump_snapshot(snapshot, plan)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    output_digest = hashlib.sha256()
    inventory_digest = hashlib.sha256()
    record_count = 0
    text_utf8_bytes = 0

    try:
        with temporary.open("wb") as handle:
            for record in iter_wikimedia_revision_records(
                snapshot,
                plan,
                max_records=max_records,
                max_text_bytes=max_text_bytes,
            ):
                payload = _canonical_json_bytes(record)
                handle.write(payload)
                output_digest.update(payload)
                inventory_digest.update(
                    _canonical_json_bytes(
                        {
                            "page_id": record["page_id"],
                            "revision_id": record["revision_id"],
                            "text_sha256": record["text_sha256"],
                        }
                    )
                )
                record_count += 1
                text_utf8_bytes += len(record["text"].encode("utf-8"))
            if record_count == 0:
                raise WikimediaIngestionError("materialization produced zero eligible records")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return WikimediaMaterializationResult(
        plan_sha256=plan.manifest()["plan_sha256"],
        snapshot_sha256=plan.snapshot_sha256,
        record_count=record_count,
        text_utf8_bytes=text_utf8_bytes,
        output_sha256=output_digest.hexdigest(),
        inventory_sha256=inventory_digest.hexdigest(),
        training_authorized=False,
    )
