"""Plan 2 / Section 4: policy-pinned deterministic candidate normalization.

This is a non-authorizing derivative of the verified Plan-2 Section-3 cohort.
The incumbent DATA324 source normalization and Section-3 acquisition authority
remain authoritative. No training, tokenizer fit or source admission occurs.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

from tools.plan2_physical_materialization_v1 import (
    Plan2MaterializationError,
    _atomic_write,
    _canonical,
    _json,
    _read_destination,
    _sha,
    stage_candidate_cohort,
)

SCHEMA = "12-6.plan2-normalized-candidate.v1"
POLICY = {
    "schema_version": "12-6.plan2-normalization-policy.v1",
    "input": "verified-section3-normalized-utf8-strict",
    "unicode": "NFC",
    "newlines": "CRLF-CR-to-LF",
    "bom": "strip-one-leading-U+FEFF",
    "record_boundary": "nonempty-paragraphs-after-strip",
    "identification": "conservative-script-evidence-v1",
    "supported_languages": ["en", "uk"],
    "modality": "text",
}
POLICY_SHA256 = _sha(_canonical(POLICY))
FILES = frozenset({"normalized.utf8", "manifest.json"})


class Plan2NormalizationError(ValueError):
    """Source candidate, policy, payload or manifest failed normalization gates."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Plan2NormalizationError(message)


def _language(text: str) -> dict[str, Any]:
    alpha = [c for c in text if c.isalpha()]
    cyrl = sum("\u0400" <= c <= "\u052f" for c in alpha)
    latin = sum(("A" <= c <= "Z") or ("a" <= c <= "z") for c in alpha)
    uk_markers = sum(c in "іїєґІЇЄҐ" for c in alpha)
    length = len(alpha)
    language = "unknown"
    # Script ratios are evidence, NOT a calibrated statistical probability.
    if length >= 6 and cyrl / length >= 0.85 and uk_markers:
        language = "uk"
    elif length >= 6 and latin / length >= 0.95:
        language = "en"
    confidence = round(max(cyrl, latin) / length, 4) if length else 0.0
    return {
        "language": language,
        "supported": language != "unknown",
        "modality": "text",
        "identification_method": POLICY["identification"],
        "script_share_not_calibrated": confidence,
        "alphabetic_count": length,
        "cyrillic_count": cyrl,
        "latin_count": latin,
        "ukrainian_marker_count": uk_markers,
    }


def normalize_verified_candidate(payload: bytes, source_manifest: dict[str, Any]
                                 ) -> tuple[bytes, dict[str, Any]]:
    """Normalize verified UTF-8 and derive immutable record spans/provenance.

    Offsets are Unicode codepoint offsets into the published normalized text.
    Unsupported/mixed language remains explicitly unknown, never silently uk/en.
    """
    _require(type(payload) is bytes, "payload must be exact bytes")
    _require(type(source_manifest) is dict and
             source_manifest.get("schema_version") ==
             "12-6.plan2-physical-cohort.v1", "missing Section-3 authority")
    _require(source_manifest.get("source_level_candidate_only") is True and
             source_manifest.get("training_corpus_authorized") is False and
             source_manifest.get("tokenizer_fit_authorized") is False and
             source_manifest.get("evaluation_authorized") is False,
             "unauthorized predecessor authority")
    _require(source_manifest.get("normalized", {}).get("sha256") == _sha(payload) and
             source_manifest.get("normalized", {}).get("bytes") == len(payload),
             "predecessor normalized bytes changed")
    _require(source_manifest.get("source_member_count") == 1 and
             type(source_manifest.get("source_id")) is str,
             "predecessor source identity unavailable")
    core = dict(source_manifest)
    original_identity = core.pop("manifest_sha256", None)
    _require(type(original_identity) is str and
             original_identity == _sha(_canonical(core)),
             "predecessor manifest checksum drift")
    try:
        text = payload.decode("utf-8", "strict")
    except UnicodeError as exc:
        raise Plan2NormalizationError("invalid strict UTF-8") from exc
    _require("\x00" not in text and
             not any(0xD800 <= ord(c) <= 0xDFFF for c in text),
             "invalid embedded Unicode/control boundary")
    text = text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    # One canonical blank line between independently auditable records.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    normalized = "\n\n".join(paragraphs)
    output = normalized.encode("utf-8", "strict")
    records: list[dict[str, Any]] = []
    offset = 0
    for index, paragraph in enumerate(paragraphs):
        start = offset
        end = start + len(paragraph)
        evidence = _language(paragraph)
        records.append({
            "record_id": f"r{index:08d}",
            "source_id": source_manifest["source_id"],
            "start": start,
            "end": end,
            "text_sha256": _sha(paragraph.encode("utf-8")),
            **evidence,
        })
        offset = end + 2
    _require(len(records) > 0, "empty normalized candidate")
    manifest = {
        "schema_version": SCHEMA,
        "policy": POLICY,
        "policy_sha256": POLICY_SHA256,
        "source_id": source_manifest["source_id"],
        "predecessor_manifest_sha256": original_identity,
        "predecessor_normalized_sha256": _sha(payload),
        "normalized_sha256": _sha(output),
        "normalized_bytes": len(output),
        "record_count": len(records),
        "records": records,
        "source_level_candidate_only": True,
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "evaluation_authorized": False,
        "paid_compute_used": False,
    }
    manifest["manifest_sha256"] = _sha(_canonical(manifest))
    return output, manifest


def verify_normalized_candidate(destination: Path) -> tuple[bytes, dict[str, Any]]:
    _require(destination.is_dir() and not destination.is_symlink(),
             "normalized destination is not a regular directory")
    _require({p.name for p in destination.iterdir()} == FILES,
             "normalized cohort membership drift")
    for name in FILES:
        _require(not (destination / name).is_symlink(), "symlinked candidate member")
    output = _read_destination(destination / "normalized.utf8")
    manifest_bytes = _read_destination(destination / "manifest.json")
    try:
        manifest = _json(manifest_bytes)
    except Plan2MaterializationError as exc:
        raise Plan2NormalizationError("invalid canonical manifest") from exc
    _require(manifest_bytes == _canonical(manifest), "noncanonical manifest JSON")
    _require(manifest.get("schema_version") == SCHEMA and
             manifest.get("policy") == POLICY and
             manifest.get("policy_sha256") == POLICY_SHA256,
             "normalization policy drift")
    digest = manifest.get("manifest_sha256")
    core = dict(manifest)
    core.pop("manifest_sha256", None)
    _require(digest == _sha(_canonical(core)), "normalization manifest checksum drift")
    _require(_sha(output) == manifest.get("normalized_sha256") and
             len(output) == manifest.get("normalized_bytes"),
             "normalized member checksum drift")
    try:
        decoded = output.decode("utf-8", "strict")
    except UnicodeError as exc:
        raise Plan2NormalizationError("normalized UTF-8 invalid on restart") from exc
    rows = manifest.get("records")
    _require(type(rows) is list and len(rows) == manifest.get("record_count") and
             bool(rows), "record inventory drift")
    prev_end = -2
    for index, row in enumerate(rows):
        _require(type(row) is dict and
                 row.get("record_id") == f"r{index:08d}" and
                 row.get("source_id") == manifest.get("source_id") and
                 type(row.get("start")) is int and type(row.get("end")) is int,
                 "record provenance drift")
        start, end = row["start"], row["end"]
        _require(start == prev_end + 2 and start < end <= len(decoded),
                 "record offsets overlap/gap")
        paragraph = decoded[start:end]
        _require(row.get("text_sha256") == _sha(paragraph.encode("utf-8")) and
                 all(row.get(k) == v for k, v in _language(paragraph).items()),
                 "record hash/language provenance drift")
        prev_end = end
    _require(prev_end == len(decoded), "trailing unaudited text")
    _require(decoded == "\n\n".join(decoded[
        r["start"]:r["end"]] for r in rows), "record boundary drift")
    _require(manifest.get("source_level_candidate_only") is True and
             all(manifest.get(k) is False for k in (
                 "training_corpus_authorized", "tokenizer_fit_authorized",
                 "evaluation_authorized", "paid_compute_used")),
             "candidate authority promotion")
    return output, manifest


def stage_normalized_candidate(root: Path, destination: Path) -> dict[str, Any]:
    """Derive from freshly verified Section-3 physical bytes; atomic and replay-safe."""
    with tempfile.TemporaryDirectory(prefix="plan2-s4-incumbent-") as staging:
        predecessor = stage_candidate_cohort(root, Path(staging))
        source = (Path(staging) / "normalized.utf8").read_bytes()
        normalized, manifest = normalize_verified_candidate(source, predecessor)
    _require(not destination.is_symlink() and
             not any(p.is_symlink() for p in destination.parents),
             "destination symlink")
    destination.mkdir(parents=True, exist_ok=True)
    _require(destination.is_dir() and not destination.is_symlink(),
             "unsafe destination directory")
    members = {
        "normalized.utf8": normalized,
        "manifest.json": _canonical(manifest),
    }
    _require({p.name for p in destination.iterdir()} <= FILES,
             "unknown candidate output member")
    # Reuse Section-3 exclusive publication semantics (atomic hardlink + fsync).
    # The manifest is published last; partial members may resume iff exact.
    published = destination / "manifest.json"
    if published.exists() or published.is_symlink():
        _require(_read_destination(published) == members["manifest.json"],
                 "immutable normalized manifest drift")
        _require({p.name for p in destination.iterdir()} == FILES,
                 "immutable normalized membership gap")
    for name in ("normalized.utf8", "manifest.json"):
        path = destination / name
        if path.exists() or path.is_symlink():
            _require(_read_destination(path) == members[name],
                     "normalized member drift; refuse overwrite")
        else:
            _require(not published.exists(), "published normalized member missing")
            _atomic_write(destination, path, members[name])
    _, verified = verify_normalized_candidate(destination)
    _require(verified == manifest, "normalized restart verification mismatch")
    return verified


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Plan2 S4 LOCAL_FREE normalized candidate")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = stage_normalized_candidate(args.root, args.out_dir)
    print(json.dumps({
        "decision": "PASS_NORMALIZED_SOURCE_CANDIDATE_ONLY",
        "manifest_sha256": report["manifest_sha256"],
        "policy_sha256": report["policy_sha256"],
        "record_count": report["record_count"],
        "tokenizer_fit_authorized": False,
        "training_corpus_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
