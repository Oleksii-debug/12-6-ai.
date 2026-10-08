"""Plan 2 Section 4: immutable evidence over the incumbent DATA324 normalization.

This adapter neither fetches nor re-admits sources. DATA324 remains the sole
physical text-normalization implementation; Plan-2 Section 3 remains the sole
candidate materialization authority. Unknown languages are never promoted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tools.materialize_data324_kubernetes_ua import language_evidence, normalize_markdown_uk
from tools.plan2_physical_materialization_v1 import (
    Plan2MaterializationError,
    _atomic_write,
    _preflight,
    _read_destination,
    stage_candidate_cohort,
)

POLICY_FILE = "configs/data/plan2_normalization_policy_v1.json"
POLICY_SCHEMA = "12-6.plan2-normalization-policy.v1"
MANIFEST_SCHEMA = "12-6.plan2-normalization-manifest.v1"
SUPPORTED_POLICY = {
    "schema_version": POLICY_SCHEMA,
    "policy_id": "data324-uk-markdown-nfkc-lf.v1",
    "implementation": "tools/materialize_data324_kubernetes_ua.py:normalize_markdown_uk",
    "encoding": "UTF-8 strict",
    "unicode": "NFKC",
    "line_endings": "LF",
    "record_boundary": "nonempty-normalized-line",
    "language": "incumbent-UA-threshold-else-und",
    "modality": "verified-strict-readable-text-only",
    "downstream_binding": "policy_sha256+raw_sha256+normalized_sha256",
}


class NormalizationEvidenceError(ValueError):
    """Invalid normalization, incompatible downstream binding or unsafe output."""


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            + "\n").encode("utf-8")


def _policy(root: Path) -> tuple[dict[str, str], str]:
    source = root / POLICY_FILE
    if source.is_symlink() or not source.is_file():
        raise NormalizationEvidenceError("missing or symlinked normalization policy")
    try:
        policy = json.loads(source.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise NormalizationEvidenceError("invalid normalization policy") from exc
    if not isinstance(policy, dict) or policy != SUPPORTED_POLICY:
        raise NormalizationEvidenceError("unsupported normalization policy")
    return policy, _sha(_canonical(policy))


def inspect_raw(raw: bytes) -> tuple[bytes, dict[str, Any], list[dict[str, Any]]]:
    """Inspect LOCAL_FREE fixture bytes; never grant them corpus admission."""
    if not isinstance(raw, bytes) or not raw or b"\x00" in raw:
        raise NormalizationEvidenceError("empty or binary input")
    try:
        normalized = normalize_markdown_uk(raw).encode("utf-8", "strict")
    except (UnicodeError, RuntimeError) as exc:
        raise NormalizationEvidenceError("invalid encoding or record structure") from exc
    if not normalized.strip():
        raise NormalizationEvidenceError("no normalized text")
    text = normalized.decode("utf-8", "strict")
    # Reuse the exact incumbent Ukrainian gate; failures stay UNKNOWN rather
    # than claiming support for arbitrary Latin/Cyrillic or mixed-language text.
    try:
        evidence = language_evidence(text)
        language, confidence = "uk", min(
            0.99, float(evidence["cyrillic_share_of_alpha"]))
    except RuntimeError:
        alpha = sum(ch.isalpha() for ch in text)
        cyr = sum("\u0400" <= ch <= "\u04ff" for ch in text)
        evidence = {"decision": "UNKNOWN", "alphabetic_chars": alpha,
                    "cyrillic_chars": cyr,
                    "rule": "incumbent UA test did not pass"}
        language, confidence = "und", 0.0
    if any(ord(ch) < 32 and ch not in "\n\t" for ch in text):
        raise NormalizationEvidenceError("unsupported control character")
    classification: dict[str, Any] = {
        "language": language,
        "language_confidence_heuristic": confidence,
        "language_evidence": evidence,
        "modality": "text",
        "modality_evidence": "strict UTF-8, no NUL or unsupported controls",
    }
    rows: list[dict[str, Any]] = []
    offset = 0
    for index, record in enumerate(normalized.splitlines(keepends=True)):
        if not record.strip():
            raise NormalizationEvidenceError("empty normalized record")
        rows.append({"index": index, "start_byte": offset,
                     "end_byte": offset + len(record),
                     "sha256": _sha(record)})
        offset += len(record)
    if not rows or offset != len(normalized):
        raise NormalizationEvidenceError("record boundaries are incomplete")
    return normalized, classification, rows


def assert_downstream_binding(receipt: dict[str, Any], binding: dict[str, Any]) -> None:
    """Reject stale consumers after raw, policy or normalized identity changes."""
    keys = ("policy_sha256", "raw_sha256", "normalized_sha256", "manifest_sha256")
    if (not isinstance(binding, dict) or set(binding) != set(keys) or
            any(binding[key] != receipt.get(key) for key in keys)):
        raise NormalizationEvidenceError("stale normalization downstream binding")


def stage_normalization(root: Path, destination: Path) -> dict[str, Any]:
    """Verify S3's canonical cohort; publish only an immutable metadata receipt."""
    if any(part.is_symlink() for part in (destination, *destination.parents)):
        raise NormalizationEvidenceError("symlink destination")
    root = root.resolve()
    policy, policy_sha = _policy(root)
    # S3's current accepted-main authority verifies raw bytes, provenance,
    # rights, hash integrity, source identity and physical staging.
    try:
        preflight, raw, incumbent_normalized = _preflight(root, None, None)
        cohort = stage_candidate_cohort(root, destination / "cohort")
    except (Plan2MaterializationError, OSError) as exc:
        raise NormalizationEvidenceError("upstream cohort validation failed") from exc
    normalized, classification, records = inspect_raw(raw)
    if normalized != incumbent_normalized or cohort != preflight:
        raise NormalizationEvidenceError("incumbent normalization drift")
    if classification["language"] != "uk":
        raise NormalizationEvidenceError("current Ukrainian candidate is unknown")
    result: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "policy_version": policy["policy_id"],
        "policy_sha256": policy_sha,
        "source_id": cohort["source_id"],
        "upstream_materialization_sha256": cohort["manifest_sha256"],
        "raw_sha256": _sha(raw),
        "normalized_sha256": _sha(normalized),
        "normalized_bytes": len(normalized),
        "record_count": len(records),
        "records": records,
        "classification": classification,
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "evaluation_authorized": False,
    }
    result["manifest_sha256"] = _sha(_canonical(result))
    expected = _canonical(result)
    if destination.is_symlink() or not destination.is_dir():
        raise NormalizationEvidenceError("invalid normalization destination")
    artifact = destination / "normalization-manifest.json"
    if artifact.exists() or artifact.is_symlink():
        if _read_destination(artifact) != expected:
            raise NormalizationEvidenceError("immutable normalization policy/input drift")
    else:
        _atomic_write(destination, artifact, expected)
    if _read_destination(artifact) != expected:
        raise NormalizationEvidenceError("normalization readback mismatch")
    assert_downstream_binding(result, {key: result[key] for key in (
        "policy_sha256", "raw_sha256", "normalized_sha256", "manifest_sha256")})
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Plan-2 immutable normalization evidence")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = stage_normalization(args.root, args.out_dir)
    print(json.dumps({"decision": "PASS_SOURCE_CANDIDATE_ONLY",
                      "manifest_sha256": result["manifest_sha256"],
                      "record_count": result["record_count"],
                      "language": result["classification"]["language"],
                      "training_corpus_authorized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
