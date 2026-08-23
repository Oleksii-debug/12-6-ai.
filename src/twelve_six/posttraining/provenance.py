"""Deterministic manifest hashing for post-training datasets and candidates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from .contracts import Split


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible data with deterministic ordering and encoding."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    record_id: str
    record_sha256: str
    split: Split
    provenance_sha256: str

    def __post_init__(self) -> None:
        for name, digest in (
            ("record_sha256", self.record_sha256),
            ("provenance_sha256", self.provenance_sha256),
        ):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"{name} must be a lowercase 64-hex digest")
        if not self.record_id.strip():
            raise ValueError("record_id must be non-empty")


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """A compact, deterministic dataset manifest.

    Entries are sorted by record_id before hashing, so input ordering cannot alter
    the manifest identity. Duplicate record IDs are rejected.
    """

    dataset_id: str
    entries: tuple[ManifestEntry, ...]
    format_version: str = "posttraining-manifest-v1"

    @classmethod
    def from_entries(
        cls,
        dataset_id: str,
        entries: Iterable[ManifestEntry],
    ) -> DatasetManifest:
        ordered = tuple(sorted(entries, key=lambda item: item.record_id))
        return cls(dataset_id=dataset_id, entries=ordered)

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must be non-empty")
        ids = [entry.record_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("dataset manifest contains duplicate record_id values")

    @property
    def sha256(self) -> str:
        return canonical_sha256(
            {
                "dataset_id": self.dataset_id,
                "format_version": self.format_version,
                "entries": [asdict(entry) for entry in self.entries],
            }
        )
