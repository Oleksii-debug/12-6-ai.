"""Plan 2 / Section 3: resumable physical witness over canonical DATA324 storage.

This reproduces an immutable, hash-verified LOCAL_FREE candidate cohort from
preexisting accepted-main raw and normalized snapshots. The rights catalog is
the incumbent Plan-2 Section-2 authority; this module never grants corpus,
tokenizer-fit, evaluation or model-training authority.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from tools.plan2_source_admissibility_v1 import build_catalog, verify_catalog
from tools.plan2_source_inventory_v1 import build_inventory
from tools.validate_data324_current_main_port import validate_snapshot

SCHEMA = "12-6.plan2-physical-cohort.v1"
SEED = "configs/data/plan2_source_inventory_v1.json"
RIGHTS = "configs/data/plan2_rights_admissibility_v1.json"
DATA324_MANIFEST = "data/external/snapshots/data324-kubernetes-ua-v1/manifest.json"
SOURCE_ID = "uk.kubernetes.docs.what-is-kubernetes"
FILES = ("raw.snapshot", "normalized.utf8", "manifest.json")
PARTIAL_PREFIX = ".plan2-partial-"


class Plan2MaterializationError(ValueError):
    """Unverifiable, incomplete, or unsafe physical cohort."""


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise Plan2MaterializationError(reason)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) +
            "\n").encode("utf-8")


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    doc: dict[str, Any] = {}
    for key, value in pairs:
        _need(key not in doc, "duplicate JSON key")
        doc[key] = value
    return doc


def _json(data: bytes) -> dict[str, Any]:
    try:
        result = json.loads(data.decode("utf-8", "strict"), object_pairs_hook=_object_pairs)
    except (ValueError, UnicodeError) as exc:
        raise Plan2MaterializationError("invalid exact JSON authority") from exc
    _need(isinstance(result, dict), "JSON authority must be object")
    return result


def _read_source(root: Path, relative: str) -> bytes:
    _need(isinstance(relative, str) and relative and "\\" not in relative,
          "invalid source path")
    candidate = Path(relative)
    _need(not candidate.is_absolute() and ".." not in candidate.parts,
          "source path traversal")
    current = root
    for part in candidate.parts:
        _need(part not in (".", "..", ""), "source path ambiguous")
        current = current / part
        _need(not current.is_symlink(), "source symlink")
    _need(current.is_file() and current.resolve().is_relative_to(root.resolve()),
          "missing or escaped source")
    return current.read_bytes()


def _preflight(root: Path, inventory_seed: Mapping[str, Any] | None,
               rights_seed: Mapping[str, Any] | None) -> tuple[dict[str, Any], bytes, bytes]:
    try:
        old = validate_snapshot(root)
    except (ValueError, RuntimeError, OSError, TypeError, KeyError) as exc:
        raise Plan2MaterializationError("incumbent DATA324 snapshot failed verification") from exc
    _need(old.get("decision") == "PASS_SOURCE_LEVEL_CURRENT_MAIN_PORT",
          "DATA324 did not pass physical qualification")
    inventory_seed = inventory_seed if inventory_seed is not None else _json(_read_source(root, SEED))
    rights_seed = rights_seed if rights_seed is not None else _json(_read_source(root, RIGHTS))
    _need(isinstance(inventory_seed, Mapping) and
          set(inventory_seed) == {"schema_version", "revision", "sources"} and
          inventory_seed["schema_version"] == "12-6.plan2-source-registry-seed.v1",
          "source seed changed")
    inv = build_inventory(inventory_seed["revision"], inventory_seed["sources"])
    _need(len(inv["sources"]) == 1 and inv["sources"][0]["source_id"] == SOURCE_ID and
          inv["sources"][0]["status"] == "candidate", "unreviewed cohort source")
    _need(isinstance(rights_seed, Mapping) and
          set(rights_seed) == {"schema_version", "grants"} and
          rights_seed["schema_version"] == "12-6.plan2-rights-candidate-seed.v1" and
          isinstance(rights_seed["grants"], list) and len(rights_seed["grants"]) == 1,
          "rights seed incomplete")
    grant = rights_seed["grants"][0]
    _need(isinstance(grant, Mapping) and grant.get("source_id") == SOURCE_ID and
          grant.get("license_id") == "CC-BY-4.0" and
          grant.get("allowed_uses") == ["source_candidate"] and
          grant.get("revoked") is False, "rights cannot be promoted at this stage")
    evidence = _read_source(root, grant["evidence_path"])
    evidence_bytes = {grant["evidence_path"]: evidence}
    cat = build_catalog(inv, rights_seed["grants"], evidence_bytes)
    verify_catalog(inv, cat, evidence_bytes)
    _need(cat["training_corpus_authorized"] is False, "candidate misrepresented")
    manifest = _json(_read_source(root, DATA324_MANIFEST))
    _need(type(manifest.get("objects")) is list and len(manifest["objects"]) == 1,
          "source cohort has gaps or duplicate members")
    obj = manifest["objects"][0]
    src = inv["sources"][0]
    _need(obj["source_id"] == SOURCE_ID and
          src["source_family"] == manifest["source_family"] and
          src["provenance"]["upstream_revision"] == manifest["upstream_revision"] and
          src["provenance"]["content_sha256"] == obj["normalized_sha256"] and
          src["location"] == "repo://" + obj["normalized_snapshot_path"],
          "source/physical identity disagreement")
    raw = _read_source(root, obj["raw_snapshot_path"])
    normalized = _read_source(root, obj["normalized_snapshot_path"])
    _need(_sha(raw) == obj["raw_sha256"] and len(raw) == obj["raw_bytes"],
          "raw snapshot checksum mismatch")
    _need(_sha(normalized) == obj["normalized_sha256"] and
          len(normalized) == obj["normalized_utf8_bytes"], "normalized checksum mismatch")
    result = {
        "schema_version": SCHEMA,
        "source_id": SOURCE_ID,
        "source_family": src["source_family"],
        "source_inventory_sha256": inv["inventory_sha256"],
        "rights_catalog_sha256": cat["catalog_sha256"],
        "upstream_revision": manifest["upstream_revision"],
        "source_member_count": 1,
        "raw": {"path": "raw.snapshot", "source_path": obj["raw_snapshot_path"],
                "sha256": obj["raw_sha256"], "bytes": len(raw)},
        "normalized": {"path": "normalized.utf8",
                       "source_path": obj["normalized_snapshot_path"],
                       "sha256": obj["normalized_sha256"], "bytes": len(normalized)},
        "rights_evidence_sha256": grant["evidence_sha256"],
        "source_level_candidate_only": True,
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "evaluation_authorized": False,
        "paid_compute_used": False,
    }
    result["manifest_sha256"] = _sha(_canonical(result))
    return result, raw, normalized


def _read_destination(path: Path) -> bytes:
    _need(not path.is_symlink() and path.is_file(), "staged member missing or symlink")
    return path.read_bytes()


def _sync_dir(root: Path) -> None:
    if hasattr(os, "O_DIRECTORY"):
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _atomic_write(directory: Path, target: Path, content: bytes) -> None:
    _need(not target.exists() and not target.is_symlink(), "refusing overwrite")
    name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=directory,
                                         prefix=PARTIAL_PREFIX, delete=False) as fp:
            name = fp.name
            fp.write(content)
            fp.flush()
            os.fsync(fp.fileno())
        _need(not target.exists() and not target.is_symlink(), "concurrent overwrite")
        os.replace(name, target)
        name = None
        _sync_dir(directory)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def stage_candidate_cohort(root: Path, destination: Path, *,
                           inventory_seed: Mapping[str, Any] | None = None,
                           rights_seed: Mapping[str, Any] | None = None,
                           purpose: str = "evidence_only") -> dict[str, Any]:
    """Verify, materialize and exact-readback a current source-level candidate.

    Incomplete pre-manifest copies resume. An already published manifest is
    immutable: missing/corrupt bytes fail closed rather than silently repair.
    No network, no training, no source promotion, no alternative corpus engine.
    """
    _need(purpose == "evidence_only", "no training/release materialization authority")
    root = root.resolve()
    _need(not destination.is_symlink(), "destination symlink")
    expected, raw, normalized = _preflight(root, inventory_seed, rights_seed)
    if destination.exists():
        _need(destination.is_dir(), "destination must be a directory")
    else:
        destination.mkdir(parents=True)
    _need(not destination.is_symlink(), "destination was replaced by a symlink")
    # The staging dir is exclusively Plan-2-owned. Unexpected entries are errors.
    allowed = set(FILES)
    for path in destination.iterdir():
        if path.name.startswith(PARTIAL_PREFIX):
            _need(path.is_file() and not path.is_symlink(),
                  "unsafe interrupted staging file")
            path.unlink()
        else:
            _need(path.name in allowed, "unexpected physical cohort member")
    manifest_path = destination / "manifest.json"
    expected_data = _canonical(expected)
    if manifest_path.exists() or manifest_path.is_symlink():
        _need(_read_destination(manifest_path) == expected_data,
              "published cohort identity drift")
        _need(set(p.name for p in destination.iterdir()) == set(FILES),
              "published cohort membership gap")
    for name, content in (("raw.snapshot", raw), ("normalized.utf8", normalized)):
        path = destination / name
        if path.exists() or path.is_symlink():
            _need(_read_destination(path) == content,
                  "existing physical member mismatch")
        else:
            _need(not manifest_path.exists(), "immutable manifest lost a physical member")
            _atomic_write(destination, path, content)
        _need(_sha(_read_destination(path)) == expected[
            "raw" if name == "raw.snapshot" else "normalized"]["sha256"],
              "post-write hash readback failed")
    if not manifest_path.exists():
        _need(set(p.name for p in destination.iterdir()) ==
              {"raw.snapshot", "normalized.utf8"}, "pre-publication membership gap")
        _atomic_write(destination, manifest_path, expected_data)
    _need(_read_destination(manifest_path) == expected_data,
          "manifest round-trip failed")
    _need(_json(expected_data) == expected, "manifest restart/readback drift")
    return expected
