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


def _rebuild(root: Path, destination: Path) -> dict[str, Any]:
    """Reuse all extant component authorities, without creating a second pipeline."""
    cohort = physical.stage_candidate_cohort(root, destination / "source")
    current = mixture.stage_mixture(root, destination / "physical-candidate")
    _require(cohort["source_level_candidate_only"] is True
             and cohort["training_corpus_authorized"] is False,
             "source candidate unexpectedly promoted")
    _require(current["training_corpus_authorized"] is False,
             "physical mixture unexpectedly grants training")

    # The sole accepted current corpus consists of ONE source/document family.
    # Family-safe train/validation/test cannot honestly be produced from it.
    try:
        split.stage_candidate(root, destination / "physical-split")
    except split.Plan2SplitError:
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
        "physical_split": physical_split,
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
