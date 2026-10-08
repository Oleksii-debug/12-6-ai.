"""Plan 2 Section 15: reproducible LOCAL_FREE audit, never a production release.

Reuses the accepted S3/S9/S10/S12/S13/S14 authorities. Both clean builds
verify actual candidate bytes and synthetic downstream fixtures independently.
In particular this tool cannot convert source_candidate rights into training
permission, or an isolated synthetic tokenizer into production authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from tools import plan2_cluster_split_v1 as split
from tools import plan2_corpus_mixture_v1 as mixture
from tools import plan2_deterministic_packing_v1 as packing
from tools import plan2_exposure_ledger_v1 as exposure
from tools import plan2_physical_materialization_v1 as physical
from tools import plan2_tokenizer_fit_freeze_v1 as fit

SCHEMA = "12-6.plan2-final-audit-local-free.v1"
OUTPUT = "plan2-section15-audit.json"


class QualificationDenied(ValueError):
    """Unqualified, non-reproducible or unsafe release assertion."""


def _require(value: bool, reason: str) -> None:
    if not value:
        raise QualificationDenied(reason)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _inventory(base: Path) -> dict[str, str]:
    """Exact byte inventory; reject symlinks even within aborted stage trees."""
    members: dict[str, str] = {}
    for path in sorted(base.rglob("*")):
        _require(not path.is_symlink(), "symlink in qualification artifact tree")
        _require(path.is_file() or path.is_dir(), "unexpected artifact member")
        if path.is_file():
            relative = path.relative_to(base).as_posix()
            _require(relative not in members, "duplicate artifact member")
            members[relative] = _digest(path.read_bytes())
    _require(bool(members), "empty qualification artifact tree")
    return members


def _read_gate(path: Path) -> dict[str, Any]:
    """Verify a staged authority, not just an untrusted 64-character hash."""
    _require(path.is_file() and not path.is_symlink(), "physical gate member missing")
    raw = physical._read_destination(path)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise QualificationDenied("invalid physical gate JSON") from exc
    _require(type(value) is dict and raw == _canonical(value),
             "noncanonical physical gate JSON")
    digest = value.get("manifest_sha256")
    unsigned = {k: v for k, v in value.items() if k != "manifest_sha256"}
    _require(type(digest) is str and digest == _digest(_canonical(unsigned)),
             "physical gate manifest hash mismatch")
    _require(value.get("training_corpus_authorized") is False,
             "physical gate incorrectly authorizes training")
    for capability in ("tokenizer_fit_authorized", "evaluation_authorized",
                       "generated_auto_reentry_authorized", "raw_text_emitted",
                       "real_final_test_material_accessed"):
        if capability in value:
            _require(value[capability] is False,
                     "physical gate incorrectly authorizes " + capability)
    return value


def _audit_physical_gates(candidate: Path, mixture_hash: str) -> dict[str, str]:
    """Bind S5-S8 readbacks into the S9 physical candidate, fail closed."""
    firewall_dir = candidate / "firewall"
    near_dir = firewall_dir / "near"
    exact_dir = near_dir / "exact"
    privacy_dir = exact_dir / "privacy"
    stages = {
        "privacy": _read_gate(privacy_dir / "privacy-manifest.json"),
        "exact_dedup": _read_gate(exact_dir / "exact-dedup-manifest.json"),
        "near_dedup": _read_gate(near_dir / "near-dedup-manifest.json"),
        "reserved_eval": _read_gate(
            firewall_dir / "reserved-eval-firewall-manifest.json"
        ),
    }
    rights = stages["privacy"]["manifest_sha256"]
    _require(any(source.get("privacy_manifest_sha256") == rights
                 for source in stages["exact_dedup"]["sources"]),
             "S5-S6 privacy provenance disconnected")
    _require(stages["near_dedup"]["upstream_exact_manifest_sha256"] ==
             stages["exact_dedup"]["manifest_sha256"],
             "S6-S7 exact dedup lineage disconnected")
    _require(stages["reserved_eval"]["physical_s7_manifest_sha256"] ==
             stages["near_dedup"]["manifest_sha256"],
             "S7-S8 decontamination lineage disconnected")
    mixture_raw = physical._read_destination(
        candidate / "corpus-mixture-manifest.json"
    )
    try:
        mixture_doc = json.loads(mixture_raw.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise QualificationDenied("invalid physical mixture JSON") from exc
    _require(type(mixture_doc) is dict
             and mixture_raw == _canonical(mixture_doc),
             "noncanonical physical mixture JSON")
    mixture_core = {
        k: v for k, v in mixture_doc.items() if k != "dataset_candidate_sha256"
    }
    _require(
        mixture_doc.get("dataset_candidate_sha256") == _digest(_canonical(mixture_core))
        and mixture_doc["dataset_candidate_sha256"] == mixture_hash
        and mixture_doc["training_corpus_authorized"] is False
        and mixture_doc["physical_s8_manifest_sha256"] ==
        stages["reserved_eval"]["manifest_sha256"],
        "S8-S9 mixture lineage disconnected",
    )
    return {key: value["manifest_sha256"] for key, value in stages.items()}


def _rebuild(root: Path, destination: Path) -> dict[str, Any]:
    """Reuse all extant component authorities, without creating a second pipeline."""
    cohort = physical.stage_candidate_cohort(root, destination / "source")
    current = mixture.stage_mixture(root, destination / "physical-candidate")
    _require(cohort["source_level_candidate_only"] is True
             and cohort["training_corpus_authorized"] is False,
             "source candidate unexpectedly promoted")
    _require(current["training_corpus_authorized"] is False,
             "physical mixture unexpectedly grants training")

    # A changed source-family count must refuse before reading stale S9 hashes.
    # Never mislabel an unrelated integrity error as a known corpus limitation.
    families = current["contributions"]["family"]
    _require(type(families) is dict and len(families) == 1,
             "physical source-family count changed; release requires requalification")

    gates = _audit_physical_gates(
        destination / "physical-candidate", current["dataset_candidate_sha256"]
    )
    try:
        split.stage_candidate(root, destination / "physical-split")
    except split.Plan2SplitError as exc:
        causes: list[str] = []
        reason: BaseException | None = exc
        while reason is not None:
            causes.append(str(reason))
            reason = reason.__cause__
        _require(
            "insufficient independent document families" in causes
            or "three-way cluster-safe split has empty partition" in causes,
            "unexpected physical split failure; not a family-count refusal",
        )
        physical_split = "DENIED_SINGLE_SOURCE_FAMILY"
    else:
        raise QualificationDenied(
            "single-family physical candidate unexpectedly passed safe split"
        )

    tokenizer = fit.stage(root, destination / "synthetic-tokenizer")
    packed = packing.stage(root, destination / "synthetic-shards")
    ledger = exposure.stage(root, destination / "synthetic-shards",
                            destination / "synthetic-ledger")
    _require(tokenizer["production_release_authorized"] is False
             and tokenizer["physical_tokenizer_fit_authorized"] is False
             and packed["production_release_authorized"] is False
             and ledger["production_release_authorized"] is False
             and ledger["training_corpus_authorized"] is False,
             "synthetic downstream authority escalation")
    _require(packed["manifest_sha256"] == ledger["packing_manifest_sha256"],
             "packing-to-exposure identity mismatch")
    _require(tokenizer["manifest_sha256"] == packed["tokenizer_manifest_sha256"],
             "tokenizer-to-packing identity mismatch")
    _require(packed["target_count"] == ledger["target_count"],
             "target exposure mismatch")

    return {
        "physical_source_manifest_sha256": cohort["manifest_sha256"],
        "physical_mixture_manifest_sha256": current["dataset_candidate_sha256"],
        "physical_gate_manifest_sha256": gates,
        "physical_split": physical_split,
        "physical_source_family_count": len(families),
        "synthetic_tokenizer_manifest_sha256": tokenizer["manifest_sha256"],
        "synthetic_packing_manifest_sha256": packed["manifest_sha256"],
        "synthetic_exposure_manifest_sha256": ledger["manifest_sha256"],
        "synthetic_target_count": ledger["target_count"],
        "member_sha256": _inventory(destination),
    }


def audit(root: Path) -> dict[str, Any]:
    """Compare two independently materialized complete LOCAL_FREE artifact trees."""
    root = root.resolve(strict=True)
    _require(root.is_dir(), "missing repository root")
    with tempfile.TemporaryDirectory(prefix="plan2-s15-first-") as a:
        with tempfile.TemporaryDirectory(prefix="plan2-s15-clean-") as b:
            first = _rebuild(root, Path(a))
            second = _rebuild(root, Path(b))
            _require(first == second, "clean rebuild hash or inventory mismatch")
    core = {
        "schema_version": SCHEMA,
        "decision": "COMPONENT_AUDIT_ONLY_NOT_TERMINAL",
        "plan": 2,
        "section": 15,
        "reproducible_clean_builds": 2,
        "local_free": True,
        "paid_compute_used": False,
        "physical_corpus_training_authorized": False,
        "real_production_tokenizer_present": False,
        "real_production_shards_present": False,
        "production_release_authorized": False,
        "terminal_done": False,
        "blocking_gates": [
            "training_rights_require_independent_authorization",
            "physical_corpus_requires_multiple_safe_source_families",
            "production_tokenizer_fit_and_compatibility_not_yet_established",
            "production_packing_shards_exposure_not_yet_materialized",
            "plan9_production_handoff_missing",
        ],
        "evidence": first,
    }
    return {**core, "audit_sha256": _digest(_canonical(core))}


def stage(root: Path, destination: Path) -> dict[str, Any]:
    _require(not any(p.is_symlink() for p in (destination, *destination.parents)),
             "symlink output path")
    report = audit(root)
    destination.mkdir(parents=True, exist_ok=True)
    _require(destination.is_dir() and not destination.is_symlink(),
             "invalid output directory")
    target = destination / OUTPUT
    raw = _canonical(report)
    if target.exists() or target.is_symlink():
        _require(not target.is_symlink() and target.is_file()
                 and target.read_bytes() == raw,
                 "immutable audit report changed")
    else:
        physical._atomic_write(destination, target, raw)
    _require(target.read_bytes() == raw, "audit readback mismatch")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan2 S15 non-release audit")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = stage(args.root, args.out_dir)
    print(json.dumps({"decision": report["decision"],
                      "audit_sha256": report["audit_sha256"],
                      "production_release_authorized": False,
                      "terminal_done": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
