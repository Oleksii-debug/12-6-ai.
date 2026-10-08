"""Plan 2 S15: verify real public-domain book snapshots, without training promotion.

This is a thin physical-source audit using the existing Plan-2 file publication
authority; it is NOT a second dedup, privacy, tokenizer or training pipeline.
The book catalog is immutable by Git blob; changes require a versioned catalog.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2.public-domain-physical-audit.v1"
CATALOG = "configs/data/plan2_public_domain_books_v1.json"
CATALOG_BLOB = "3336231ed17550c9aa1fe27b3bc27295970b79d9"
OUTPUT = "public-domain-physical-manifest.json"
PREFIX = "data/external/snapshots/plan2-public-domain-books-v1/"
HEX40 = re.compile(r"[a-f0-9]{40}\Z")
HEX64 = re.compile(r"[a-f0-9]{64}\Z")
LICENSE_BLOB = "8d062dda262bcdc42d45b861bd796117feb6d0fe"


class BookCohortDenied(ValueError):
    """Unqualified snapshot, drifted provenance, or unauthorized release."""


def need(condition: bool, reason: str) -> None:
    if not condition:
        raise BookCohortDenied(reason)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def read_checked(root: Path, name: str) -> bytes:
    """Never follow a source symlink, absolute path or dot-dot component."""
    need(type(name) is str and bool(name) and "\\" not in name,
         "invalid source pathname")
    relative = Path(name)
    need(not relative.is_absolute() and all(p not in {".", "..", ""}
                                         for p in relative.parts),
         "escaped or ambiguous physical source path")
    current = root
    for part in relative.parts:
        current = current / part
        need(not current.is_symlink(), "source or parent is symlink")
    need(current.is_file() and current.resolve().is_relative_to(root.resolve()),
         "missing or escaped source")
    need(current.stat().st_size <= 2_000_000, "unbounded physical source")
    return current.read_bytes()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, value in pairs:
        need(key not in data, "duplicate JSON catalog key")
        data[key] = value
    return data


def inspect(root: Path) -> dict[str, Any]:
    raw = read_checked(root, CATALOG)
    need(git_blob(raw) == CATALOG_BLOB, "pinned source catalog Git blob changed")
    try:
        catalog = json.loads(raw.decode("utf-8", "strict"),
                             object_pairs_hook=_unique_pairs)
    except (UnicodeError, ValueError) as exc:
        raise BookCohortDenied("invalid catalog JSON") from exc
    need(type(catalog) is dict and
         catalog.get("schema_version") ==
         "12-6.plan2.public-domain-book-cohort.v1" and
         catalog.get("release_status") == "PHYSICAL_SNAPSHOT_ONLY_NOT_TRAINING_READY" and
         catalog.get("book_count") == 3 and
         catalog.get("training_authorized") is False and
         catalog.get("tokenizer_fit_authorized") is False and
         catalog.get("production_release_authorized") is False and
         catalog.get("project_gutenberg_markup_removed") is True,
         "source catalog authority was elevated or changed")
    sources = catalog.get("sources")
    need(type(sources) is list and len(sources) == 3,
         "physical book sources are incomplete")
    book_ids: set[str] = set()
    families: set[str] = set()
    upstreams: set[tuple[str, str]] = set()
    records: list[dict[str, Any]] = []
    total = 0
    for source in sources:
        need(type(source) is dict, "invalid book metadata")
        source_id, family = source.get("source_id"), source.get("source_family")
        author, title = source.get("author"), source.get("title")
        revision = source.get("source_revision")
        path = source.get("snapshot_path")
        origin = source.get("source_repository")
        source_file = source.get("source_file")
        need(all(type(v) is str and v for v in (
            source_id, family, author, title, revision, path, origin, source_file
        )), "book provenance incomplete")
        need(source_id not in book_ids and family not in families,
             "duplicate book identity or falsely independent family")
        need((origin, source_file) not in upstreams,
             "duplicate upstream source presented as independent")
        book_ids.add(source_id)
        families.add(family)
        upstreams.add((origin, source_file))
        need(HEX40.fullmatch(revision) is not None and
             HEX40.fullmatch(source.get("source_git_blob_sha1", "")) is not None and
             source.get("license_git_blob_sha1") == LICENSE_BLOB and
             source.get("upstream_source_url") ==
             f"https://github.com/{origin}/blob/{revision}/{source_file}",
             "unverifiable upstream repository, revision or license")
        death = source.get("author_death_year")
        need(type(death) is int and death <= 1900,
             "author-date public-domain basis not established")
        need(path.startswith(PREFIX) and
             re.fullmatch(r"[a-z0-9-]+\.en\.txt", path[len(PREFIX):])
             is not None, "book snapshot path not versioned")
        payload = read_checked(root, path)
        need(type(source.get("normalized_bytes")) is int and
             len(payload) == source["normalized_bytes"] and
             100_000 <= len(payload) < 2_000_000 and
             HEX64.fullmatch(source.get("normalized_sha256", "")) is not None and
             sha(payload) == source["normalized_sha256"],
             "physical book SHA-256 or length mismatch")
        try:
            decoded = payload.decode("utf-8", "strict")
        except UnicodeError as exc:
            raise BookCohortDenied("invalid UTF-8 source") from exc
        need(decoded.endswith("\n") and "\r" not in decoded and
             "\ufeff" not in decoded and "\x00" not in decoded and
             "project gutenberg" not in decoded.lower(),
             "not normalized or Project Gutenberg markup retained")
        total += len(payload)
        records.append({
            "source_id": source_id,
            "source_family": family,
            "author": author,
            "title": title,
            "snapshot_path": path,
            "snapshot_sha256": sha(payload),
            "snapshot_bytes": len(payload),
            "source_git_blob_sha1": source["source_git_blob_sha1"],
            "source_revision": revision,
        })
    need(total >= 1_000_000, "incomplete bounded real-text cohort")
    need(len(book_ids) == len(families) == len(upstreams) == 3,
         "independent source lineages missing")
    core = {
        "schema_version": SCHEMA,
        "source_catalog_git_blob": CATALOG_BLOB,
        "physical_source_families": len(families),
        "physical_source_bytes": total,
        "books": sorted(records, key=lambda item: item["source_id"]),
        "decision": "REAL_BOOK_SNAPSHOTS_VERIFIED_CANDIDATE_ONLY",
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "production_release_authorized": False,
        "paid_compute_used": False,
    }
    return {**core, "manifest_sha256": sha(canonical(core))}


def stage(root: Path, out: Path) -> dict[str, Any]:
    need(not any(p.is_symlink() for p in (out, *out.parents)),
         "symlink output directory")
    result = inspect(root)
    out.mkdir(parents=True, exist_ok=True)
    target = out / OUTPUT
    expected = canonical(result)
    if target.exists() or target.is_symlink():
        need(not target.is_symlink() and _read_destination(target) == expected,
             "immutable book audit drift on restart")
    else:
        _atomic_write(out, target, expected)
    need(_read_destination(target) == expected, "physical book audit readback drift")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan2 S15 public-domain real books")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    opts = parser.parse_args()
    result = stage(opts.root, opts.out_dir)
    print(json.dumps({
        "decision": result["decision"],
        "manifest_sha256": result["manifest_sha256"],
        "physical_source_bytes": result["physical_source_bytes"],
        "training_corpus_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
