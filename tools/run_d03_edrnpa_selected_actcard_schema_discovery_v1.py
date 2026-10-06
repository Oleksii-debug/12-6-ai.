#!/usr/bin/env python3
"""Discover text-free act-card schema for the exact selected EDRNPA candidate.

This is a zero-credit physical discovery tool. It replays the exact pinned
candidate materializer, binds the already-merged conservative source policy,
then inspects only XML structure for the 176 selected source ordinals. No
source text or metadata values are persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tools.probe_d03_edrnpa_open_data_v1 as probe
import tools.verify_d03_edrnpa_rights_provenance_source_policy_v1 as source_policy

SCHEMA = "12-6.d03-edrnpa-selected-actcard-schema-discovery.v1"
RECEIPT_SCHEMA = "12-6.d03-edrnpa-selected-actcard-schema-receipt.v1"
EXPECTED_SOURCE_BYTES = 611_865_397
EXPECTED_SELECTED_OBJECTS = 176
EXPECTED_SELECTED_BYTES = 4_000_000
EXPECTED_INVENTORY_IDENTITY = (
    "380939a9ef8f67dcfb55912c010941f08d35b868267116d927603e988e710cca"
)
EXPECTED_REPORT_IDENTITY = (
    "f5668ad4ddb3ce7ed0cd0130377f803ddbcebfb5b547c6757176efa8043122a6"
)
MAX_FIELD_NAMES = 512
MAX_FIELD_NAME_CHARS = 256
MAX_SIGNATURES_PER_FIELD = 32


class DiscoveryError(RuntimeError):
    """Fail-closed discovery error."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _identity(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(dict(value))).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiscoveryError(message)


def _selected_ordinals(rows: list[dict[str, Any]]) -> set[int]:
    result: set[int] = set()
    for row in rows:
        path = row.get("source_path")
        _require(isinstance(path, str), "selected source_path missing")
        marker = "#document:"
        _require(marker in path, "selected source_path ordinal missing")
        raw = path.rsplit(marker, 1)[1]
        _require(raw.isascii() and raw.isdigit(), "selected ordinal is not decimal")
        ordinal = int(raw)
        _require(ordinal > 0 and ordinal not in result, "selected ordinal invalid/duplicate")
        result.add(ordinal)
    _require(len(result) == EXPECTED_SELECTED_OBJECTS, "selected ordinal count drift")
    return result


def _safe_field_name(value: Any) -> str:
    _require(isinstance(value, str), "item name must be text")
    value = unicodedata.normalize("NFC", value)
    _require(bool(value) and len(value) <= MAX_FIELD_NAME_CHARS, "item name size drift")
    _require(
        all(ch >= " " and ch not in {"\x7f", "\x00"} for ch in value),
        "item name contains control character",
    )
    return value


def _discover_schema(
    *,
    nested: Any,
    info: Any,
    selected_ordinals: set[int],
) -> dict[str, Any]:
    field_occurrences: defaultdict[str, int] = defaultdict(int)
    field_documents: defaultdict[str, set[int]] = defaultdict(set)
    field_signatures: defaultdict[str, set[str]] = defaultdict(set)
    richtext_names: defaultdict[str, int] = defaultdict(int)
    selected_documents_seen: set[int] = set()

    stack: list[str] = []
    ordinal = 0
    active_selected = False
    active_item_name: str | None = None
    active_item_child_tags: list[str] = []
    active_item_has_richtext = False

    with nested.open(info) as raw_stream:
        guarded = probe._GuardedXMLReader(raw_stream, expected_size=info.file_size)
        try:
            iterator = ET.iterparse(guarded, events=("start", "end"))
            for event, elem in iterator:
                tag = probe._local_tag(elem.tag)
                if event == "start":
                    stack.append(tag)
                    if tag == "document":
                        ordinal += 1
                        active_selected = ordinal in selected_ordinals
                        if active_selected:
                            selected_documents_seen.add(ordinal)
                    elif active_selected and tag == "item":
                        _require(
                            stack == ["rna", "database", "document", "item"],
                            "selected item path drift",
                        )
                        _require(set(elem.attrib) == {"name"}, "selected item attributes drift")
                        active_item_name = _safe_field_name(elem.attrib["name"])
                        active_item_child_tags = []
                        active_item_has_richtext = False
                    elif active_selected and active_item_name is not None:
                        if len(stack) >= 5:
                            active_item_child_tags.append("/".join(stack[4:]))
                        if tag == "richtext":
                            active_item_has_richtext = True
                    continue

                _require(bool(stack) and stack[-1] == tag, "XML stack drift")
                if active_selected and tag == "item":
                    _require(active_item_name is not None, "selected item state missing")
                    name = active_item_name
                    field_occurrences[name] += 1
                    field_documents[name].add(ordinal)
                    signature = "|".join(sorted(set(active_item_child_tags))) or "<empty>"
                    _require(
                        len(field_signatures[name]) < MAX_SIGNATURES_PER_FIELD
                        or signature in field_signatures[name],
                        f"too many structural signatures for field {name!r}",
                    )
                    field_signatures[name].add(signature)
                    if active_item_has_richtext:
                        richtext_names[name] += 1
                    active_item_name = None
                    active_item_child_tags = []
                    active_item_has_richtext = False
                    elem.clear()
                elif tag == "document":
                    active_selected = False
                    elem.clear()
                elif active_selected:
                    elem.clear()
                stack.pop()
        except ET.ParseError as exc:
            raise DiscoveryError("EDRNPA metadata discovery XML parse failure") from exc

        _require(guarded.bytes_seen == info.file_size, "nested XML not fully consumed")
        _require(
            guarded.removed_control_bytes >= 0,
            "invalid guarded XML control-removal count",
        )

    _require(not stack, "unterminated XML stack")
    _require(
        selected_documents_seen == selected_ordinals,
        "not all selected ordinals were observed in XML",
    )
    _require(
        0 < len(field_occurrences) <= MAX_FIELD_NAMES,
        "selected item-name cardinality outside safety bound",
    )

    fields = []
    for name in sorted(field_occurrences):
        fields.append(
            {
                "item_name": name,
                "selected_document_count": len(field_documents[name]),
                "occurrence_count": field_occurrences[name],
                "richtext_occurrence_count": richtext_names.get(name, 0),
                "structural_signatures": sorted(field_signatures[name]),
            }
        )
    return {
        "selected_documents_seen": len(selected_documents_seen),
        "xml10_control_bytes_removed": guarded.removed_control_bytes,
        "xml10_control_removal_identity_sha256": guarded.removal_identity_sha256,
        "unique_item_names": len(fields),
        "richtext_item_names": [
            {"item_name": name, "occurrence_count": count}
            for name, count in sorted(richtext_names.items())
        ],
        "fields": fields,
    }


def execute(source_path: Path, *, execution_head_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(
        len(execution_head_sha) == 40
        and execution_head_sha == execution_head_sha.lower()
        and all(ch in "0123456789abcdef" for ch in execution_head_sha),
        "execution head must be lowercase 40-hex",
    )

    policy = source_policy.validate_policy(source_policy.load_policy(), root=Path.cwd())
    _require(
        policy["admission_policy"]["record_level_provenance_classification_required"] is True,
        "source policy no longer requires record-level classification",
    )
    _require(
        policy["admission_policy"]["payload_source_admission_executed"] is False,
        "source policy unexpectedly admits payload",
    )

    with tempfile.TemporaryDirectory(prefix="edrnpa-schema-discovery-") as tmp:
        root = Path(tmp)
        snapshot = root / "verified-source.zip"
        source_bytes, source_md5, source_sha = probe._hash_file(
            source_path, snapshot_path=snapshot
        )
        _require(source_bytes == EXPECTED_SOURCE_BYTES, "source byte count drift")
        _require(source_md5 == probe.RESOURCE_MD5, "source MD5 drift")
        _require(source_sha == probe.RESOURCE_SHA256, "source SHA-256 drift")

        rows, report = probe.materialize_archive_path(
            snapshot,
            expected_md5=probe.RESOURCE_MD5,
            expected_sha256=probe.RESOURCE_SHA256,
            byte_cap=probe.MAX_SELECTED_BYTES,
            work_dir=root / "materializer",
        )
        _require(len(rows) == EXPECTED_SELECTED_OBJECTS, "selected object count drift")
        _require(
            sum(int(row["normalized_bytes"]) for row in rows) == EXPECTED_SELECTED_BYTES,
            "selected byte count drift",
        )
        _require(
            report["selection"]["inventory_identity_sha256"] == EXPECTED_INVENTORY_IDENTITY,
            "candidate inventory identity drift",
        )
        _require(
            report["report_identity_sha256"] == EXPECTED_REPORT_IDENTITY,
            "candidate report identity drift",
        )
        _require(
            all(row["training_eligible"] is False for row in rows),
            "candidate training eligibility widened",
        )
        _require(
            all(row["evaluation_eligible"] is False for row in rows),
            "candidate evaluation eligibility widened",
        )
        selected_ordinals = _selected_ordinals(rows)
        normalization = report.get("normalization")
        _require(isinstance(normalization, Mapping), "candidate normalization missing")
        expected_control_count = normalization.get("xml10_control_bytes_removed")
        expected_control_identity = normalization.get(
            "xml10_control_removal_identity_sha256"
        )
        _require(
            isinstance(expected_control_count, int)
            and not isinstance(expected_control_count, bool)
            and expected_control_count >= 0,
            "candidate XML control-removal count invalid",
        )
        _require(
            isinstance(expected_control_identity, str)
            and len(expected_control_identity) == 64
            and expected_control_identity == expected_control_identity.lower()
            and all(ch in "0123456789abcdef" for ch in expected_control_identity),
            "candidate XML control-removal identity invalid",
        )

        nested_path = root / "schema-nested.zip"
        nested_name, nested_size = probe._extract_nested_zip(snapshot, nested_path)
        nested, xml_info, xml_path = probe._nested_xml_info(nested_path)
        try:
            structure = _discover_schema(
                nested=nested,
                info=xml_info,
                selected_ordinals=selected_ordinals,
            )
        finally:
            nested.close()
        _require(
            structure["xml10_control_bytes_removed"] == expected_control_count,
            "schema replay XML control-removal count drift",
        )
        _require(
            structure["xml10_control_removal_identity_sha256"]
            == expected_control_identity,
            "schema replay XML control-removal identity drift",
        )

    core = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": execution_head_sha,
        "source_binding": {
            "dataset_id": probe.DATASET_ID,
            "resource_id": probe.RESOURCE_ID,
            "resource_updated": probe.RESOURCE_UPDATED,
            "source_bytes": source_bytes,
            "source_md5": source_md5,
            "source_sha256": source_sha,
            "family_id": probe.FAMILY_ID,
            "nested_zip_member": nested_name,
            "nested_zip_bytes": nested_size,
            "nested_xml_member": xml_path,
            "nested_xml_bytes": xml_info.file_size,
        },
        "candidate_binding": {
            "selected_objects": EXPECTED_SELECTED_OBJECTS,
            "selected_bytes": EXPECTED_SELECTED_BYTES,
            "inventory_identity_sha256": EXPECTED_INVENTORY_IDENTITY,
            "report_identity_sha256": EXPECTED_REPORT_IDENTITY,
        },
        "policy_binding": {
            "schema_version": policy["schema_version"],
            "decision_class": policy["admission_policy"]["decision_class"],
            "record_level_act_card_binding_required": policy["admission_policy"][
                "requires_record_level_act_card_binding"
            ],
            "record_level_provenance_classification_required": policy["admission_policy"][
                "record_level_provenance_classification_required"
            ],
        },
        "selected_act_card_structure": structure,
        "content_boundary": {
            "legal_text_persisted": False,
            "metadata_values_persisted": False,
            "only_item_names_and_structural_counts_persisted": True,
        },
        "truth_boundary": {
            "record_level_payload_provenance_executed": False,
            "payload_source_admission_executed": False,
            "admitted_payload_records": 0,
            "admitted_payload_bytes": 0,
            "canonical_capacity_credit_bytes": 0,
            "family_count_credit_added": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "model_training_executed": False,
            "learned_weights_created": False,
            "final_test_accessed": False,
            "paid_compute_used": False,
        },
    }
    discovery = {**core, "discovery_identity_sha256": _identity(core)}

    receipt_core = {
        "schema_version": RECEIPT_SCHEMA,
        "execution_head_sha": execution_head_sha,
        "source_sha256": source_sha,
        "candidate_inventory_identity_sha256": EXPECTED_INVENTORY_IDENTITY,
        "candidate_report_identity_sha256": EXPECTED_REPORT_IDENTITY,
        "discovery_identity_sha256": discovery["discovery_identity_sha256"],
        "selected_documents_seen": structure["selected_documents_seen"],
        "unique_item_names": structure["unique_item_names"],
        "metadata_values_persisted": False,
        "payload_source_admission_executed": False,
        "canonical_capacity_credit_bytes": 0,
        "training_authorized_bytes": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "final_test_accessed": False,
        "paid_compute_used": False,
    }
    receipt = {**receipt_core, "receipt_identity_sha256": _identity(receipt_core)}
    return discovery, receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--execution-head-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        discovery, receipt = execute(
            args.source_zip,
            execution_head_sha=args.execution_head_sha,
        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "selected-actcard-schema.json").write_bytes(_canonical(discovery))
        (args.output_dir / "execution-receipt.json").write_bytes(_canonical(receipt))
    except (DiscoveryError, source_policy.EdrnpaSourcePolicyError, probe.ProbeError, OSError) as exc:
        print(f"BLOCKED: {exc}")
        return 2

    print(
        "EDRNPA_SELECTED_ACTCARD_SCHEMA_DISCOVERY="
        + discovery["discovery_identity_sha256"]
    )
    print(
        "SELECTED_DOCUMENTS_SEEN="
        + str(discovery["selected_act_card_structure"]["selected_documents_seen"])
    )
    print(
        "UNIQUE_ITEM_NAMES="
        + str(discovery["selected_act_card_structure"]["unique_item_names"])
    )
    print("PAYLOAD_SOURCE_ADMISSION_EXECUTED=false")
    print("CANONICAL_CAPACITY_CREDIT_BYTES=0")
    print("TOKENIZER_FIT_AUTHORIZED=false")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
