"""Fail-closed authority facade for expanded D03 global dedup V9.

The original V9 implementation is retained verbatim in the private implementation
module so this repair does not fork matcher semantics.  This facade closes the
independent-audit trust-boundary findings before delegating to that implementation:

* Rada matcher rows must be the exact semantic parse of the authenticated JSONL;
* the injected matcher must be the exact terminal #824/V3 semantic closure;
* the reconstructed full V8 graph must reproduce the sealed nested V3 report before
  any expanded matching; and
* every selected V8 survivor is cross-bound to the sealed stable-origin/object,
  normalized-payload and comparison/provenance projection before use.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from twelve_six.data import _expanded_global_dedup_v9_impl as _impl

# Preserve the incumbent implementation/API surface.  Hardened definitions below
# intentionally override only the trust-boundary entry points.
for _name in dir(_impl):
    if not _name.startswith("__") and _name not in {"validate_rada_rows", "run_expanded_dedup"}:
        globals()[_name] = getattr(_impl, _name)

# Static aliases for the authority primitives used by this facade.  The loop above
# keeps the incumbent import surface intact; these aliases keep Ruff/F821 honest.
ExpandedDedupError = _impl.ExpandedDedupError
V8_NESTED_V3_SHA256 = _impl.V8_NESTED_V3_SHA256
_canonical = _impl._canonical
_require = _impl._require
_sha256 = _impl._sha256

_LEGACY_VALIDATE_RADA_ROWS = _impl.validate_rada_rows
_LEGACY_RUN_EXPANDED_DEDUP = _impl.run_expanded_dedup

# Exact Git blob identities from terminal V7 head
# d3333ec1b4a508df232a5aefccd6686adda745fb.  Together these files are the
# complete non-stdlib semantic closure used by cross_source_capacity_audit_v3.
_EXPECTED_MATCHER_BLOBS = {
    "twelve_six.data.cross_source_capacity_audit_v3": "11490b1803e0aa2266d8ac0053676efcfb0f91ba",
    "twelve_six.data.cross_source_capacity_audit": "84cdf00b2d468d2709a542ac3ee2ea372aae5716",
    "twelve_six.data._data232_decontamination_matching": "dab5da98dfc43133aa8f3c2e3c78c809252b741b",
}
_V8_SURVIVOR_BINDING_FIELDS = (
    "source_family",
    "modality",
    "declared_capacity_bytes",
    "verified_raw_sha256",
    "normalized_sha256",
    "stable_origin_id_sha256",
    "stable_object_id_sha256",
)


def _git_blob_sha1(raw: bytes) -> str:
    prefix = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(prefix + raw).hexdigest()


def _module_source_blob(module: ModuleType) -> str:
    source = getattr(module, "__file__", None)
    _require(isinstance(source, str) and source, f"matcher module has no source: {module.__name__}")
    path = Path(source)
    _require(path.is_file() and not path.is_symlink(), f"matcher module source is not a regular file: {path}")
    return _git_blob_sha1(path.read_bytes())


def _verify_matcher_semantic_closure(
    matcher_audit: Callable[[Mapping[str, Any], Mapping[str, bytes]], Mapping[str, Any]],
    matcher_verify: Callable[[Mapping[str, Any]], None],
) -> None:
    """Bind executable matcher callbacks to the exact terminal #824/V3 closure."""

    audit_module_name = getattr(matcher_audit, "__module__", None)
    verify_module_name = getattr(matcher_verify, "__module__", None)
    _require(
        audit_module_name == "twelve_six.data.cross_source_capacity_audit_v3"
        and verify_module_name == audit_module_name,
        "expanded matcher callbacks are not terminal V3",
    )
    v3 = sys.modules.get(str(audit_module_name))
    _require(isinstance(v3, ModuleType), "terminal V3 matcher module is not loaded")
    _require(getattr(v3, "audit_payloads", None) is matcher_audit, "matcher audit callback was replaced")
    _require(getattr(v3, "verify_report", None) is matcher_verify, "matcher verifier callback was replaced")

    v1 = getattr(v3, "v1", None)
    _require(
        isinstance(v1, ModuleType) and v1.__name__ == "twelve_six.data.cross_source_capacity_audit",
        "terminal V3 base matcher dependency drift",
    )
    dependency_names = {
        getattr(getattr(v1, name, None), "__module__", None)
        for name in ("normalize_for_contamination", "code_skeleton_tokens")
    }
    _require(
        dependency_names == {"twelve_six.data._data232_decontamination_matching"},
        "terminal V3 DATA-232 matcher dependency drift",
    )

    for module_name, expected_blob in _EXPECTED_MATCHER_BLOBS.items():
        module = sys.modules.get(module_name)
        _require(isinstance(module, ModuleType), f"matcher dependency is not loaded: {module_name}")
        _require(
            _module_source_blob(module) == expected_blob,
            f"matcher implementation authority drift: {module_name}",
        )


def _parse_authenticated_rada_jsonl(raw_jsonl: bytes) -> list[dict[str, Any]]:
    _require(isinstance(raw_jsonl, bytes) and raw_jsonl, "Rada output JSONL is empty")
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(raw_jsonl.splitlines(), 1):
        _require(bool(raw_line.strip()), f"blank authenticated Rada JSONL line: {line_no}")
        try:
            text = raw_line.decode("utf-8", errors="strict")
            value = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExpandedDedupError(f"invalid authenticated Rada JSONL row {line_no}: {exc}") from exc
        _require(isinstance(value, dict), f"authenticated Rada JSONL row {line_no} must be object")
        rows.append(value)
    _require(bool(rows), "authenticated Rada JSONL contains no rows")
    return rows


def validate_rada_rows(
    rows: Sequence[Mapping[str, Any]], raw_jsonl: bytes, report: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return only rows proven to be the exact semantic parse of authenticated bytes."""

    parsed = _parse_authenticated_rada_jsonl(raw_jsonl)
    _require(
        isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)),
        "Rada rows must be a sequence",
    )
    supplied: list[dict[str, Any]] = []
    for row in rows:
        _require(isinstance(row, Mapping), "supplied Rada row must be an object")
        supplied.append(dict(row))
    _require(
        supplied == parsed,
        "supplied Rada rows do not match authenticated output JSONL",
    )
    # Reuse all incumbent row/hash/byte/privacy/partial-unit invariants, but run them
    # over the rows parsed from the authenticated bytes rather than caller objects.
    _LEGACY_VALIDATE_RADA_ROWS(parsed, raw_jsonl, report)
    return parsed


def _index_source_rows(rows: Any, *, label: str) -> dict[str, Mapping[str, Any]]:
    _require(isinstance(rows, list) and rows, f"{label} source rows missing")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), f"{label} source row must be object")
        source_id = row.get("source_id")
        _require(
            isinstance(source_id, str) and source_id and source_id not in result,
            f"{label} source id invalid/duplicate",
        )
        result[source_id] = row
    return result


def _validate_reconstructed_v8_against_preflight(
    inventory: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    survivor_authority: Mapping[str, Any],
    preflight_report: Mapping[str, Any],
) -> None:
    """Cross-bind reconstructed semantic metadata to sealed V8 authority."""

    _require(
        preflight_report.get("report_sha256") == V8_NESTED_V3_SHA256,
        "reconstructed V8 matcher report identity drift",
    )
    reconstructed = _index_source_rows(inventory.get("sources"), label="reconstructed V8")
    preflight = _index_source_rows(preflight_report.get("sources"), label="preflight V8")
    survivor_rows = survivor_authority.get("survivors")
    _require(isinstance(survivor_rows, list) and survivor_rows, "V8 survivor rows missing")

    for survivor in survivor_rows:
        _require(isinstance(survivor, Mapping), "V8 survivor row must be object")
        source_id = survivor.get("source_id")
        _require(isinstance(source_id, str) and source_id, "V8 survivor source id missing")
        source = reconstructed.get(source_id)
        projection = preflight.get(source_id)
        payload = payloads.get(source_id)
        _require(source is not None and projection is not None, f"V8 survivor unavailable: {source_id}")
        _require(isinstance(payload, bytes), f"V8 survivor payload unavailable: {source_id}")

        for field in _V8_SURVIVOR_BINDING_FIELDS:
            _require(
                survivor.get(field) == projection.get(field),
                f"V8 survivor {field} is not bound to reconstructed matcher semantics: {source_id}",
            )

        _require(
            projection.get("verified_raw_bytes") == len(payload)
            and projection.get("verified_raw_sha256") == _sha256(payload),
            f"V8 preflight raw payload drift: {source_id}",
        )
        stable_origin = source.get("stable_origin_id")
        stable_object = source.get("stable_object_id")
        _require(
            isinstance(stable_origin, str)
            and _sha256(stable_origin.encode("utf-8")) == projection.get("stable_origin_id_sha256"),
            f"V8 stable origin authority drift: {source_id}",
        )
        _require(
            isinstance(stable_object, str)
            and _sha256(stable_object.encode("utf-8")) == projection.get("stable_object_id_sha256"),
            f"V8 stable object authority drift: {source_id}",
        )

        comparison_policy = projection.get("comparison_policy")
        if comparison_policy == "DATA232_GENERIC_FROM_RAW":
            _require(
                source.get("comparison_normalization") is None,
                f"V8 generic comparison metadata substituted: {source_id}",
            )
            _require(
                projection.get("comparison_payload_bytes") == len(payload)
                and projection.get("comparison_payload_sha256") == _sha256(payload),
                f"V8 generic comparison projection drift: {source_id}",
            )
        else:
            _require(
                source.get("comparison_normalization") == comparison_policy,
                f"V8 comparison policy substituted: {source_id}",
            )
            _require(
                source.get("expected_comparison_bytes") == projection.get("comparison_payload_bytes")
                and source.get("expected_comparison_sha256")
                == projection.get("comparison_payload_sha256"),
                f"V8 comparison payload authority drift: {source_id}",
            )


def _restrict_lineage_to_survivors(
    inventory: Mapping[str, Any], survivor_authority: Mapping[str, Any]
) -> dict[str, Any]:
    prepared = copy.deepcopy(dict(inventory))
    survivor_rows = survivor_authority.get("survivors")
    _require(isinstance(survivor_rows, list) and survivor_rows, "V8 survivor rows missing")
    survivor_ids = {str(row["source_id"]) for row in survivor_rows if isinstance(row, Mapping)}
    _require(len(survivor_ids) == len(survivor_rows), "V8 survivor source ids invalid/duplicate")

    edges = prepared.get("lineage_edges", [])
    _require(isinstance(edges, list), "reconstructed V8 lineage_edges must be a list")
    retained: list[dict[str, Any]] = []
    for edge in edges:
        _require(isinstance(edge, Mapping), "reconstructed V8 lineage edge must be object")
        left = edge.get("left_source_id")
        right = edge.get("right_source_id")
        _require(isinstance(left, str) and isinstance(right, str), "V8 lineage edge endpoints malformed")
        if left in survivor_ids and right in survivor_ids:
            retained.append(copy.deepcopy(dict(edge)))
    prepared["lineage_edges"] = retained
    return prepared


def run_expanded_dedup(
    *,
    matcher_audit: Callable[[Mapping[str, Any], Mapping[str, bytes]], Mapping[str, Any]],
    matcher_verify: Callable[[Mapping[str, Any]], None],
    reconstructed_v8_inventory: Mapping[str, Any],
    reconstructed_v8_payloads: Mapping[str, bytes],
    v8_survivor_authority: Mapping[str, Any],
    data526_evidence: Mapping[str, Any],
    data526_record_inventory: Mapping[str, Any],
    rada_language_report: Mapping[str, Any],
    rada_quality_privacy_report: Mapping[str, Any],
    expected_rada_report_sha256: str,
    rada_rows: Sequence[Mapping[str, Any]],
    rada_raw_jsonl: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hardened V9 entry point; no expanded matcher call occurs before authority proof."""

    verified_rada_rows = validate_rada_rows(
        rada_rows,
        rada_raw_jsonl,
        rada_quality_privacy_report,
    )
    _verify_matcher_semantic_closure(matcher_audit, matcher_verify)

    # Reconstruct the exact sealed V8 semantic result before adding any Rada unit.
    # This binds stable IDs, normalization, comparison metadata, thresholds and all
    # lineage-derived matches to the already sealed nested V3 authority.
    preflight = matcher_audit(
        copy.deepcopy(dict(reconstructed_v8_inventory)),
        dict(reconstructed_v8_payloads),
    )
    _require(isinstance(preflight, Mapping), "V8 semantic preflight returned non-object report")
    matcher_verify(preflight)
    _validate_reconstructed_v8_against_preflight(
        reconstructed_v8_inventory,
        reconstructed_v8_payloads,
        v8_survivor_authority,
        preflight,
    )

    prepared_inventory = _restrict_lineage_to_survivors(
        reconstructed_v8_inventory,
        v8_survivor_authority,
    )
    report, survivors = _LEGACY_RUN_EXPANDED_DEDUP(
        matcher_audit=matcher_audit,
        matcher_verify=matcher_verify,
        reconstructed_v8_inventory=prepared_inventory,
        reconstructed_v8_payloads=reconstructed_v8_payloads,
        v8_survivor_authority=v8_survivor_authority,
        data526_evidence=data526_evidence,
        data526_record_inventory=data526_record_inventory,
        rada_language_report=rada_language_report,
        rada_quality_privacy_report=rada_quality_privacy_report,
        expected_rada_report_sha256=expected_rada_report_sha256,
        rada_rows=verified_rada_rows,
        rada_raw_jsonl=rada_raw_jsonl,
    )

    # Durable outer evidence now states the exact semantic authorities enforced by
    # this facade; this does not grant any additional corpus or training credit.
    report = copy.deepcopy(report)
    report["matcher_execution_authority"] = {
        "terminal_v7_head_sha": "d3333ec1b4a508df232a5aefccd6686adda745fb",
        "nested_v3_report_sha256": V8_NESTED_V3_SHA256,
        "v3_git_blob_sha1": _EXPECTED_MATCHER_BLOBS[
            "twelve_six.data.cross_source_capacity_audit_v3"
        ],
        "v1_git_blob_sha1": _EXPECTED_MATCHER_BLOBS[
            "twelve_six.data.cross_source_capacity_audit"
        ],
        "data232_matching_git_blob_sha1": _EXPECTED_MATCHER_BLOBS[
            "twelve_six.data._data232_decontamination_matching"
        ],
        "authenticated_rada_rows_only": True,
        "sealed_v8_semantic_preflight_required": True,
    }
    core = dict(report)
    core.pop("report_sha256", None)
    report["report_sha256"] = _sha256(_canonical(core))
    return report, survivors
