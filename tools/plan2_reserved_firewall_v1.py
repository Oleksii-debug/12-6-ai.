"""Plan 2 Section 8: source-level reserved evaluation firewall.

A fail-closed adapter of incumbent S5/S6/S7 and DATA-232 authorities. The
caller pins the reservation identity independently before any training packing.
No training/evaluation authorization or raw-text artifact is produced.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from twelve_six.data.decontamination_authority_v2 import build_report, verify_report
from tools import plan2_exact_dedup_v1 as exact
from tools import plan2_near_dedup_v1 as near
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2-reserved-evaluation-firewall.v1"
ROLES = {"train_candidate", "selection_validation", "final_test"}
RESERVED = {"selection_validation", "final_test"}


class FirewallError(ValueError):
    """Unverified reservation, contaminated upstream, or immutable receipt drift."""


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise FirewallError(message)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def reservation_identity(reservations: Mapping[str, Mapping[str, Any]]) -> str:
    """Compute identity for an *independently pinned* source-role assignment.

    This hash is not a grant of rights. A caller must supply the expected hash
    from a separate approved reservation authority.
    """
    _need(isinstance(reservations, Mapping) and bool(reservations),
          "empty reservations")
    roles: set[str] = set()
    normalized: dict[str, dict[str, str]] = {}
    for source, item in sorted(reservations.items()):
        _need(type(source) is str and bool(source) and
              isinstance(item, Mapping) and
              set(item) == {"role", "origin", "normalization_manifest_sha256",
                            "privacy_manifest_sha256"},
              "source reservation schema drift")
        role, origin = item["role"], item["origin"]
        _need(role in ROLES and origin in {"lawful_source", "teacher_generated"},
              "unrecognized role or origin")
        _need(not (role == "train_candidate" and origin == "teacher_generated"),
              "generated teacher material cannot automatically reenter training")
        for field in ("normalization_manifest_sha256", "privacy_manifest_sha256"):
            digest = item[field]
            _need(type(digest) is str and len(digest) == 64 and
                  all(char in "0123456789abcdef" for char in digest),
                  "reservation upstream identity invalid")
        roles.add(role)
        normalized[source] = dict(item)
    _need(roles == ROLES, "train and both independently reserved roles required")
    return _sha(_canonical(normalized))


def inspect_firewall(
    cohorts: Sequence[tuple[Mapping[str, Any], bytes, Mapping[str, Any]]],
    reservations: Mapping[str, Mapping[str, Any]],
    *,
    expected_reservation_sha256: str,
) -> dict[str, Any]:
    """Rerun upstream proofs, reserve whole sources, then invoke DATA-232.

    All S5-clean records are compared, including S6/S7-suppressed duplicates.
    Only the train-side S7 survivors outside DATA-232's excluded clusters may
    appear in the output. This is an *unpromoted candidate*, not train authority.
    """
    actual_reservation = reservation_identity(reservations)
    _need(actual_reservation == expected_reservation_sha256,
          "reservation not independently pinned")
    _need(type(cohorts) in (list, tuple) and bool(cohorts), "missing cohorts")
    sources: dict[str, tuple[Mapping[str, Any], bytes, Mapping[str, Any]]] = {}
    for normalized, payload, privacy in cohorts:
        _need(isinstance(normalized, Mapping) and type(payload) is bytes and
              isinstance(privacy, Mapping), "invalid physical cohort")
        source = normalized.get("source_id")
        _need(type(source) is str and source not in sources, "duplicate source")
        _need(source in reservations, "unreserved source encountered")
        pin = reservations[source]
        _need(normalized.get("manifest_sha256") ==
              pin["normalization_manifest_sha256"] and
              privacy.get("manifest_sha256") == pin["privacy_manifest_sha256"],
              "reservation not bound to actual upstream manifests")
        sources[source] = (normalized, payload, privacy)
    _need(set(sources) == set(reservations), "reservation source coverage gap")
    try:
        exact_receipt = exact.inspect_exact(tuple(sources.values()))
        train = tuple(sources[s] for s in sorted(sources)
                      if reservations[s]["role"] == "train_candidate")
        held = tuple(sources[s] for s in sorted(sources)
                     if reservations[s]["role"] in RESERVED)
        near_train = near.inspect_near(train)
        near_held = near.inspect_near(held)
    except (ValueError, KeyError, TypeError) as exc:
        raise FirewallError("upstream S5/S6/S7 verification failed") from exc

    admissible = {r["record_id"] for r in exact_receipt["record_inventory"]}
    training: list[dict[str, str]] = []
    evaluation: list[dict[str, str]] = []
    for source in sorted(sources):
        normalized, payload, _receipt = sources[source]
        for part in normalized["records"]:
            rid = f"{source}:r{part['index']:08d}"
            if rid not in admissible:
                continue
            raw = payload[part["start_byte"]:part["end_byte"]]
            _need(_sha(raw) == part["sha256"], "physical record identity drift")
            row = {
                "record_id": rid, "source_id": source, "source_family": source,
                "modality": "text", "text": raw.decode("utf-8", "strict"),
            }
            (training if reservations[source]["role"] == "train_candidate"
             else evaluation).append(row)
    _need(training and evaluation, "nonempty training and evaluation required")
    authorities: list[dict[str, str]] = []
    for source in sorted(sources):
        if reservations[source]["role"] not in RESERVED:
            continue
        normalized, payload, _receipt = sources[source]
        git_blob = hashlib.sha1(
            b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
        ).hexdigest()
        authorities.append({
            "authority_id": source,
            "identity_sha256": normalized["normalized_sha256"],
            "role": reservations[source]["role"],
            "source_sha": git_blob,
        })
    def role_id(role: str) -> str:
        return _sha(_canonical([
            {"source_id": source, "normalized_sha256":
             sources[source][0]["normalized_sha256"]}
            for source in sorted(sources) if reservations[source]["role"] == role
        ]))
    try:
        report = build_report(
            training, evaluation,
            training_corpus_identity=near_train["manifest_sha256"],
            selection_validation_identity=role_id("selection_validation"),
            final_test_identity=role_id("final_test"),
            authorities={"authorities": authorities},
        )
        verify_report(report)
    except (ValueError, KeyError, TypeError) as exc:
        raise FirewallError("incumbent DATA-232 execution failed") from exc
    id_by_hash = {_sha(r["record_id"].encode("utf-8")): r["record_id"]
                  for r in training}
    blocked = set()
    for item in report["excluded_records"]:
        h = item["record_id_sha256"]
        _need(h in id_by_hash, "DATA-232 returned foreign exclusion")
        blocked.add(id_by_hash[h])
    retained = sorted(set(near_train["retained_record_ids"]) - blocked)
    _need(bool(retained), "all training candidates contaminated: fail closed")
    _need(not set(retained) & set(near_held["retained_record_ids"]),
          "evaluation identity in training")
    core = {
        "schema_version": SCHEMA,
        "reservation_identity_sha256": actual_reservation,
        "upstream_exact_manifest_sha256": exact_receipt["manifest_sha256"],
        "train_near_manifest_sha256": near_train["manifest_sha256"],
        "reserved_near_manifest_sha256": near_held["manifest_sha256"],
        "data232_report_sha256": report["report_sha256"],
        "data232_report": report,
        "reserved_source_ids": sorted(s for s in sources
                                      if reservations[s]["role"] in RESERVED),
        "input_train_record_count": len(training),
        "excluded_data232_record_count": len(blocked),
        "retained_record_ids": retained,
        "retained_record_count": len(retained),
        "teacher_generated_train_allowed": False,
        "evaluation_sources_excluded_before_packing": True,
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "evaluation_authorized": False,
        "raw_text_emitted": False,
    }
    return {**core, "manifest_sha256": _sha(_canonical(core))}


def stage_firewall(
    destination: Path, receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Immutable restart-safe text-free handoff (not a rights admission)."""
    _need(isinstance(receipt, Mapping) and
          receipt.get("schema_version") == SCHEMA, "firewall schema drift")
    core = {k: v for k, v in receipt.items() if k != "manifest_sha256"}
    _need(receipt.get("manifest_sha256") == _sha(_canonical(core)) and
          receipt.get("raw_text_emitted") is False and
          receipt.get("training_corpus_authorized") is False,
          "firewall receipt identity/authority drift")
    _need(not any(p.is_symlink() for p in (destination, *destination.parents)),
          "symlink destination")
    destination.mkdir(parents=True, exist_ok=True)
    _need(destination.is_dir() and not destination.is_symlink(),
          "invalid firewall destination")
    target = destination / "reserved-firewall-manifest.json"
    expected = _canonical(dict(receipt))
    if target.exists() or target.is_symlink():
        _need(_read_destination(target) == expected,
              "immutable firewall publication mismatch")
    else:
        _atomic_write(destination, target, expected)
    _need(_read_destination(target) == expected, "firewall readback drift")
    return dict(receipt)
