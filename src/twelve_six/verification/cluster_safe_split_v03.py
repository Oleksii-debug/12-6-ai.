"""Independent cluster-safe split verifier for VERIFY-309 (LOCAL_FREE only)."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BOUNDARIES = ("train", "validation", "reserved")
SCHEMA_VERSION = "12-6.verify309-cluster-safe-split.v1"
FIXTURE_SCHEMA_VERSION = "12-6.verify309-adversarial-fixtures.v1"


class SplitSafetyEvidenceError(ValueError):
    """Raised when split-safety evidence is incomplete or internally inconsistent."""


def _canonical_json_sha256(value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SplitSafetyEvidenceError(f"{field} must be non-empty text")
    return value.strip()


def _require_sha256(value: Any, field: str) -> str:
    text = _require_text(value, field)
    if (
        len(text) != 64
        or text != text.lower()
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise SplitSafetyEvidenceError(f"{field} must be lowercase SHA-256 hex")
    return text


def _optional_non_negative_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SplitSafetyEvidenceError(f"{field} must be a non-negative integer or null")
    return value


def _first_present(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


@dataclass(frozen=True, slots=True)
class BoundaryRecord:
    """Minimal public split evidence. No raw corpus text is required."""

    boundary: str
    record_id: str
    source_id: str
    source_family: str
    content_sha256: str
    cluster_id: str
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.boundary not in BOUNDARIES:
            raise SplitSafetyEvidenceError(f"unsupported boundary: {self.boundary}")
        for field in ("record_id", "source_id", "source_family", "cluster_id"):
            _require_text(getattr(self, field), field)
        _require_sha256(self.content_sha256, "content_sha256")
        _optional_non_negative_int(self.size_bytes, "size_bytes")

    @classmethod
    def from_mapping(cls, boundary: str, record: Mapping[str, Any]) -> "BoundaryRecord":
        if not isinstance(record, Mapping):
            raise SplitSafetyEvidenceError("manifest record must be an object")
        record_id = _first_present(record, ("record_id", "id"))
        source_family = _first_present(record, ("source_family", "family"))
        cluster_id = _first_present(
            record,
            ("near_duplicate_cluster_id", "dedup_cluster_id", "cluster_id"),
        )
        size_bytes = _first_present(
            record,
            ("normalized_bytes", "size_bytes", "bytes"),
        )
        content_sha256 = _first_present(
            record,
            ("content_sha256", "normalized_sha256"),
        )
        return cls(
            boundary=boundary,
            record_id=_require_text(record_id, "record_id"),
            source_id=_require_text(record.get("source_id"), "source_id"),
            source_family=_require_text(source_family, "source_family"),
            content_sha256=_require_sha256(content_sha256, "content_sha256"),
            cluster_id=_require_text(cluster_id, "near_duplicate_cluster_id"),
            size_bytes=_optional_non_negative_int(size_bytes, "size_bytes"),
        )

    def identity_mapping(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary,
            "record_id": self.record_id,
            "source_id": self.source_id,
            "source_family": self.source_family,
            "content_sha256": self.content_sha256,
            "cluster_id": self.cluster_id,
            "size_bytes": self.size_bytes,
        }


def _group_crossings(
    records: Sequence[BoundaryRecord],
    *,
    key_name: str,
    value_getter: Any,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[BoundaryRecord]] = defaultdict(list)
    for record in records:
        grouped[value_getter(record)].append(record)
    result: list[dict[str, Any]] = []
    for value in sorted(grouped):
        members = grouped[value]
        boundaries = sorted({item.boundary for item in members})
        if len(boundaries) < 2:
            continue
        result.append(
            {
                key_name: value,
                "boundaries": boundaries,
                "record_ids": sorted({item.record_id for item in members}),
                "source_ids": sorted({item.source_id for item in members}),
                "source_families": sorted({item.source_family for item in members}),
            }
        )
    return result


def _record_id_crossings(records: Sequence[BoundaryRecord]) -> list[dict[str, Any]]:
    return _group_crossings(
        records,
        key_name="record_id",
        value_getter=lambda record: record.record_id,
    )


def _exact_content_crossings(records: Sequence[BoundaryRecord]) -> list[dict[str, Any]]:
    return _group_crossings(
        records,
        key_name="content_sha256",
        value_getter=lambda record: record.content_sha256,
    )


def _cluster_crossings(records: Sequence[BoundaryRecord]) -> list[dict[str, Any]]:
    return _group_crossings(
        records,
        key_name="cluster_id",
        value_getter=lambda record: record.cluster_id,
    )


def _source_family_distribution(records: Sequence[BoundaryRecord]) -> dict[str, Any]:
    by_boundary: dict[str, dict[str, dict[str, int]]] = {}
    for boundary in BOUNDARIES:
        family_counts: dict[str, dict[str, int]] = {}
        boundary_records = [record for record in records if record.boundary == boundary]
        families = sorted({record.source_family for record in boundary_records})
        for family in families:
            members = [record for record in boundary_records if record.source_family == family]
            known_bytes = sum(item.size_bytes or 0 for item in members)
            unknown_bytes = sum(item.size_bytes is None for item in members)
            family_counts[family] = {
                "records": len(members),
                "known_bytes": known_bytes,
                "unknown_byte_records": unknown_bytes,
            }
        by_boundary[boundary] = family_counts
    return by_boundary


def audit_records(records: Iterable[BoundaryRecord]) -> dict[str, Any]:
    """Audit exact, near-cluster, and record-identity crossings across all boundaries."""

    materialized = list(records)
    if not materialized:
        raise SplitSafetyEvidenceError("at least one split record is required")

    seen_boundary_ids: set[tuple[str, str]] = set()
    for record in materialized:
        key = (record.boundary, record.record_id)
        if key in seen_boundary_ids:
            raise SplitSafetyEvidenceError(
                f"duplicate record id inside {record.boundary}: {record.record_id}"
            )
        seen_boundary_ids.add(key)

    record_crossings = _record_id_crossings(materialized)
    exact_crossings = _exact_content_crossings(materialized)
    cluster_crossings = _cluster_crossings(materialized)
    distribution = _source_family_distribution(materialized)
    canonical_assignment = [
        record.identity_mapping()
        for record in sorted(materialized, key=lambda item: (item.boundary, item.record_id))
    ]
    counts = {
        boundary: sum(record.boundary == boundary for record in materialized)
        for boundary in BOUNDARIES
    }
    passed = not (record_crossings or exact_crossings or cluster_crossings)
    core = {
        "schema_version": SCHEMA_VERSION,
        "verdict": "PASS_SPLIT_SAFE" if passed else "FAIL_SPLIT_SAFETY",
        "records": len(materialized),
        "boundary_record_counts": counts,
        "record_identity_cross_boundary_count": len(record_crossings),
        "exact_content_cross_boundary_count": len(exact_crossings),
        "cluster_cross_boundary_count": len(cluster_crossings),
        "record_identity_crossings": record_crossings,
        "exact_content_crossings": exact_crossings,
        "cluster_crossings": cluster_crossings,
        "source_family_distribution": distribution,
        "canonical_assignment_sha256": _canonical_json_sha256(
            {"assignments": canonical_assignment}
        ),
        "source_family_distribution_sha256": _canonical_json_sha256(
            {"source_family_distribution": distribution}
        ),
    }
    return {**core, "report_sha256": _canonical_json_sha256(core)}


def _load_jsonl(path: Path, boundary: str) -> list[BoundaryRecord]:
    if not path.is_file():
        raise SplitSafetyEvidenceError(f"required manifest missing: {path.as_posix()}")
    records: list[BoundaryRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SplitSafetyEvidenceError(
                f"invalid JSON at {path.as_posix()}:{line_number}"
            ) from exc
        records.append(BoundaryRecord.from_mapping(boundary, item))
    return records


def audit_artifact_root(root: str | Path) -> dict[str, Any]:
    """Audit the exact three public split/reservation manifests in one clean build root."""

    base = Path(root)
    records: list[BoundaryRecord] = []
    records.extend(_load_jsonl(base / "manifests/train.jsonl", "train"))
    records.extend(
        _load_jsonl(base / "manifests/selection-validation.jsonl", "validation")
    )
    records.extend(
        _load_jsonl(base / "manifests/final-test-reservation.jsonl", "reserved")
    )
    report = audit_records(records)
    return {"root": base.as_posix(), **report}


def compare_clean_roots(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    """Require deterministic split assignment and source-family distribution across clean roots."""

    fields = (
        "canonical_assignment_sha256",
        "source_family_distribution_sha256",
        "records",
        "record_identity_cross_boundary_count",
        "exact_content_cross_boundary_count",
        "cluster_cross_boundary_count",
    )
    mismatches = [field for field in fields if first.get(field) != second.get(field)]
    passed = not mismatches
    core = {
        "schema_version": "12-6.verify309-clean-root-comparison.v1",
        "verdict": (
            "PASS_DETERMINISTIC_ASSIGNMENT"
            if passed
            else "FAIL_NONDETERMINISTIC_ASSIGNMENT"
        ),
        "mismatched_fields": mismatches,
        "root_a_assignment_sha256": first.get("canonical_assignment_sha256"),
        "root_b_assignment_sha256": second.get("canonical_assignment_sha256"),
        "root_a_family_distribution_sha256": first.get(
            "source_family_distribution_sha256"
        ),
        "root_b_family_distribution_sha256": second.get(
            "source_family_distribution_sha256"
        ),
    }
    return {**core, "comparison_sha256": _canonical_json_sha256(core)}


def _fixture_record(
    boundary: str,
    record_id: str,
    source_id: str,
    source_family: str,
    content: str,
    cluster_id: str,
) -> BoundaryRecord:
    return BoundaryRecord(
        boundary=boundary,
        record_id=record_id,
        source_id=source_id,
        source_family=source_family,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        cluster_id=cluster_id,
        size_bytes=len(content.encode("utf-8")),
    )


def _clean_fixture() -> list[BoundaryRecord]:
    return [
        _fixture_record("train", "t-1", "src-uk-a", "uk.family.a", "alpha", "c-1"),
        _fixture_record("train", "t-2", "src-en-a", "en.family.a", "beta", "c-2"),
        _fixture_record(
            "validation",
            "v-1",
            "src-uk-b",
            "uk.family.b",
            "gamma",
            "c-3",
        ),
        _fixture_record(
            "reserved",
            "r-1",
            "eval-authority",
            "reserved.eval",
            "delta",
            "c-4",
        ),
    ]


def adversarial_self_test() -> dict[str, Any]:
    """Run independent adversarial cluster fixtures without corpus payloads or paid compute."""

    checks: dict[str, bool] = {}

    clean = _clean_fixture()
    clean_report = audit_records(clean)
    checks["clean_fixture_passes"] = clean_report["verdict"] == "PASS_SPLIT_SAFE"

    reversed_report = audit_records(reversed(clean))
    checks["input_order_deterministic"] = (
        clean_report["canonical_assignment_sha256"]
        == reversed_report["canonical_assignment_sha256"]
        and clean_report["source_family_distribution_sha256"]
        == reversed_report["source_family_distribution_sha256"]
    )

    exact_alias = list(clean)
    exact_alias.append(
        BoundaryRecord(
            boundary="validation",
            record_id="v-exact-alias",
            source_id="src-en-b",
            source_family="en.family.b",
            content_sha256=clean[0].content_sha256,
            cluster_id=clean[0].cluster_id,
            size_bytes=clean[0].size_bytes,
        )
    )
    exact_report = audit_records(exact_alias)
    checks["exact_alias_cross_boundary_rejected"] = (
        exact_report["verdict"] == "FAIL_SPLIT_SAFETY"
        and exact_report["exact_content_cross_boundary_count"] == 1
    )

    near_reserved = list(clean)
    near_reserved.append(
        _fixture_record(
            "reserved",
            "r-near",
            "eval-authority-2",
            "reserved.eval.two",
            "epsilon different content",
            clean[1].cluster_id,
        )
    )
    near_reserved_report = audit_records(near_reserved)
    checks["near_cluster_train_reserved_rejected"] = (
        near_reserved_report["verdict"] == "FAIL_SPLIT_SAFETY"
        and near_reserved_report["cluster_cross_boundary_count"] == 1
    )

    cross_family = list(clean)
    cross_family.append(
        _fixture_record(
            "validation",
            "v-cross-family-near",
            "src-code-a",
            "code.family.a",
            "zeta different modality content",
            clean[0].cluster_id,
        )
    )
    cross_family_report = audit_records(cross_family)
    checks["near_cluster_train_validation_cross_family_rejected"] = (
        cross_family_report["verdict"] == "FAIL_SPLIT_SAFETY"
        and cross_family_report["cluster_cross_boundary_count"] == 1
    )

    reused_id = list(clean)
    reused_id.append(
        _fixture_record(
            "reserved",
            clean[0].record_id,
            "eval-authority-3",
            "reserved.eval.three",
            "eta unique content",
            "c-unique",
        )
    )
    reused_report = audit_records(reused_id)
    checks["record_identity_reuse_rejected"] = (
        reused_report["verdict"] == "FAIL_SPLIT_SAFETY"
        and reused_report["record_identity_cross_boundary_count"] == 1
    )

    passed = all(checks.values())
    return {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "verdict": "PASS_ADVERSARIAL_FIXTURES" if passed else "FAIL_ADVERSARIAL_FIXTURES",
        "checks": checks,
        "fixture_count": len(checks),
    }


def _emit(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-a", type=Path)
    parser.add_argument("--root-b", type=Path)
    parser.add_argument("--self-test-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    fixtures = adversarial_self_test()
    if args.self_test_only:
        _emit(fixtures)
        return 0 if fixtures["verdict"] == "PASS_ADVERSARIAL_FIXTURES" else 1

    if args.root_a is None or args.root_b is None:
        _emit(
            {
                "schema_version": SCHEMA_VERSION,
                "verdict": "BLOCKED_TWO_CLEAN_ROOTS_REQUIRED",
                "reason": "both --root-a and --root-b are required",
                "adversarial_fixtures": fixtures,
            }
        )
        return 2

    try:
        root_a = audit_artifact_root(args.root_a)
        root_b = audit_artifact_root(args.root_b)
    except SplitSafetyEvidenceError as exc:
        _emit(
            {
                "schema_version": SCHEMA_VERSION,
                "verdict": "BLOCKED_INSUFFICIENT_SPLIT_EVIDENCE",
                "reason": str(exc),
                "adversarial_fixtures": fixtures,
            }
        )
        return 2

    deterministic = compare_clean_roots(root_a, root_b)
    passed = (
        fixtures["verdict"] == "PASS_ADVERSARIAL_FIXTURES"
        and root_a["verdict"] == "PASS_SPLIT_SAFE"
        and root_b["verdict"] == "PASS_SPLIT_SAFE"
        and deterministic["verdict"] == "PASS_DETERMINISTIC_ASSIGNMENT"
    )
    _emit(
        {
            "schema_version": SCHEMA_VERSION,
            "verdict": "PASS_CLUSTER_SAFE_SPLIT" if passed else "FAIL_CLUSTER_SAFE_SPLIT",
            "adversarial_fixtures": fixtures,
            "root_a": root_a,
            "root_b": root_b,
            "determinism": deterministic,
        }
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
