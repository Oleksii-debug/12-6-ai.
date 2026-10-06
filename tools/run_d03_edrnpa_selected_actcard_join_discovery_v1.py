"""Discover the exact selected EDRNPA text-to-act-card join without admitting data.

This zero-credit execution tool reconstructs the exact pinned 176-row text
candidate, extracts its reestr_kod values only in memory, then scans the
25.1 act-card XML from the same verified outer source. Persisted evidence
contains only structural field names and aggregate join cardinalities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import tools.probe_d03_edrnpa_open_data_v1 as probe
import tools.run_d03_edrnpa_selected_actcard_schema_discovery_v1 as parent_discovery
import tools.verify_d03_edrnpa_rights_provenance_source_policy_v1 as source_policy

SCHEMA = "12-6.d03-edrnpa-selected-actcard-join-discovery.v1"
RECEIPT_SCHEMA = "12-6.d03-edrnpa-selected-actcard-join-receipt.v1"
CARD_ZIP_MEMBER = "25.1-edrnpa_cards.zip"
TEXT_CODE_FIELD = "reestr_kod"
EXPECTED_SOURCE_BYTES = 611_865_397
EXPECTED_SELECTED_OBJECTS = 176
EXPECTED_SELECTED_BYTES = 4_000_000
EXPECTED_INVENTORY_IDENTITY = (
    "380939a9ef8f67dcfb55912c010941f08d35b868267116d927603e988e710cca"
)
EXPECTED_REPORT_IDENTITY = (
    "f5668ad4ddb3ce7ed0cd0130377f803ddbcebfb5b547c6757176efa8043122a6"
)
EXPECTED_SELECTED_PAYLOAD_IDENTITY = (
    "9b1c7b65a71f36ebd3c99273abf3e9f4c3c1885c96432d65761275e37e0b4435"
)
PARENT_DISCOVERY_HEAD = "66c812a5345e707f59424b2562502943ddf7a33f"
PARENT_DISCOVERY_RUN = 37456392095
PARENT_DISCOVERY_ARTIFACT = 11410515967
PARENT_DISCOVERY_ARTIFACT_ZIP_SHA256 = (
    "e32d40b3d9df1435e9976f77c416c50c596e9a9885a350bc28409cf6900887cb"
)
PARENT_DISCOVERY_IDENTITY = (
    "d8f2ca73a95a37f996cae60098f39a6f576d30362d3c380a4d4331eaaaf0a9e0"
)
MAX_ITEMS_PER_CARD = 2_048
MAX_ITEM_VALUE_CHARS = 1_000_000
MAX_FIELD_NAMES = 512
MAX_SIGNATURES_PER_FIELD = 64


class JoinDiscoveryError(RuntimeError):
    """Fail-closed selected text-to-act-card discovery error."""


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


def _selected_payload_identity(rows: list[dict[str, Any]]) -> str:
    """Reproduce the exact historical physical selected-payload identity."""

    hasher = hashlib.sha256()
    for row in rows:
        hasher.update(probe.cjson(row))
    return hasher.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise JoinDiscoveryError(message)


def _safe_value(
    value: str,
    *,
    label: str,
    require_nonempty: bool = True,
) -> str:
    normalized = unicodedata.normalize("NFC", value.strip())
    if require_nonempty:
        _require(bool(normalized), f"{label} must be non-empty")
    _require(len(normalized) <= MAX_ITEM_VALUE_CHARS, f"{label} exceeds safety bound")
    _require(
        all(ch >= " " and ch not in {"\x7f", "\x00"} for ch in normalized),
        f"{label} contains control character",
    )
    return normalized


def _extract_named_nested_zip(
    source_path: Path,
    target_path: Path,
    *,
    member_name: str,
) -> tuple[str, int]:
    try:
        outer = zipfile.ZipFile(source_path)
    except zipfile.BadZipFile as exc:
        raise JoinDiscoveryError("invalid outer EDRNPA ZIP") from exc
    with outer:
        infos = outer.infolist()
        _require(
            len(infos) <= probe.MAX_OUTER_MEMBERS,
            "outer ZIP member-count safety limit exceeded",
        )
        target_info: zipfile.ZipInfo | None = None
        for info in infos:
            try:
                safe_path = probe._validate_zip_member(
                    info,
                    max_size=probe.MAX_NESTED_ZIP_BYTES,
                )
            except probe.ProbeError as exc:
                raise JoinDiscoveryError(str(exc)) from exc
            if info.is_dir():
                continue
            if safe_path == member_name:
                _require(target_info is None, f"duplicate nested ZIP member: {member_name}")
                target_info = info
        _require(target_info is not None, f"pinned nested ZIP member missing: {member_name}")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        copied = 0
        with outer.open(target_info) as src, target_path.open("xb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                _require(
                    copied <= probe.MAX_NESTED_ZIP_BYTES,
                    f"nested ZIP exceeds byte safety limit: {member_name}",
                )
                dst.write(chunk)
        _require(copied == target_info.file_size, "nested ZIP byte count mismatch")
    return member_name, copied


def _single_xml_info(
    nested_path: Path,
    *,
    label: str,
) -> tuple[zipfile.ZipFile, zipfile.ZipInfo, str]:
    try:
        nested = zipfile.ZipFile(nested_path)
    except zipfile.BadZipFile as exc:
        raise JoinDiscoveryError(f"invalid {label} nested ZIP") from exc
    infos = nested.infolist()
    if len(infos) > probe.MAX_NESTED_MEMBERS:
        nested.close()
        raise JoinDiscoveryError(f"{label} nested ZIP member-count safety limit exceeded")
    eligible: list[tuple[zipfile.ZipInfo, str]] = []
    try:
        for info in infos:
            try:
                safe_path = probe._validate_zip_member(
                    info,
                    max_size=probe.MAX_NESTED_XML_BYTES,
                )
            except probe.ProbeError as exc:
                raise JoinDiscoveryError(str(exc)) from exc
            if info.is_dir():
                continue
            if PurePosixPath(safe_path).suffix.casefold() == ".xml":
                eligible.append((info, safe_path))
        _require(len(eligible) == 1, f"{label} nested ZIP must contain exactly one XML")
    except Exception:
        nested.close()
        raise
    info, path = eligible[0]
    return nested, info, path


def _collect_selected_text_codes(
    *,
    nested: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    selected_ordinals: set[int],
) -> dict[int, str]:
    codes: dict[int, str] = {}
    stack: list[str] = []
    ordinal = 0
    active_selected = False
    active_code_item = False
    active_code_parts: list[str] = []
    code_item_count = 0

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
                        code_item_count = 0
                    elif active_selected and tag == "item":
                        _require(
                            stack == ["rna", "database", "document", "item"],
                            "selected text item path drift",
                        )
                        _require(
                            set(elem.attrib) == {"name"},
                            "selected text item attributes drift",
                        )
                        name = parent_discovery._safe_field_name(elem.attrib["name"])
                        active_code_item = name == TEXT_CODE_FIELD
                        if active_code_item:
                            code_item_count += 1
                            _require(
                                code_item_count == 1,
                                "selected text document has duplicate reestr_kod items",
                            )
                            active_code_parts = []
                    continue

                _require(bool(stack) and stack[-1] == tag, "selected text XML stack drift")
                if active_selected and active_code_item and tag == "text":
                    active_code_parts.append("".join(elem.itertext()))
                if active_selected and tag == "item":
                    if active_code_item:
                        code = _safe_value(
                            "".join(active_code_parts),
                            label="selected text reestr_kod",
                        )
                        codes[ordinal] = code
                    active_code_item = False
                    active_code_parts = []
                    elem.clear()
                elif tag == "document":
                    if active_selected:
                        _require(
                            ordinal in codes,
                            "selected text document is missing reestr_kod",
                        )
                    active_selected = False
                    elem.clear()
                elif active_selected and not active_code_item:
                    elem.clear()
                stack.pop()
        except ET.ParseError as exc:
            raise JoinDiscoveryError("selected text XML parse failure") from exc

        _require(guarded.bytes_seen == info.file_size, "selected text XML not fully consumed")

    _require(not stack, "unterminated selected text XML stack")
    _require(
        set(codes) == selected_ordinals,
        "selected text reestr_kod coverage drift",
    )
    return codes


def _item_signature(elem: ET.Element) -> tuple[str, bool]:
    paths: set[str] = set()
    has_richtext = False

    def walk(node: ET.Element, prefix: tuple[str, ...]) -> None:
        nonlocal has_richtext
        for child in list(node):
            tag = probe._local_tag(child.tag)
            path = (*prefix, tag)
            paths.add("/".join(path))
            if tag == "richtext":
                has_richtext = True
            walk(child, path)

    walk(elem, ())
    return "|".join(sorted(paths)) or "<empty>", has_richtext


def _discover_card_join(
    *,
    nested: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    selected_codes_by_ordinal: Mapping[int, str],
) -> dict[str, Any]:
    code_to_text_ordinals: defaultdict[str, list[int]] = defaultdict(list)
    for ordinal, code in selected_codes_by_ordinal.items():
        code_to_text_ordinals[code].append(ordinal)
    selected_codes = set(code_to_text_ordinals)

    matches: defaultdict[str, list[tuple[int, str]]] = defaultdict(list)
    matched_card_documents: set[int] = set()
    field_occurrences: defaultdict[str, int] = defaultdict(int)
    field_documents: defaultdict[str, set[int]] = defaultdict(set)
    field_signatures: defaultdict[str, set[str]] = defaultdict(set)
    richtext_occurrences: defaultdict[str, int] = defaultdict(int)

    stack: list[str] = []
    card_ordinal = 0
    current_items: list[tuple[str, str, bool, str]] = []

    with nested.open(info) as raw_stream:
        guarded = probe._GuardedXMLReader(raw_stream, expected_size=info.file_size)
        try:
            iterator = ET.iterparse(guarded, events=("start", "end"))
            for event, elem in iterator:
                tag = probe._local_tag(elem.tag)
                if event == "start":
                    stack.append(tag)
                    if tag == "document":
                        card_ordinal += 1
                        current_items = []
                    elif tag == "item":
                        _require(
                            stack == ["rna", "database", "document", "item"],
                            "act-card item path drift",
                        )
                        _require(
                            set(elem.attrib) == {"name"},
                            "act-card item attributes drift",
                        )
                    continue

                _require(bool(stack) and stack[-1] == tag, "act-card XML stack drift")
                if tag == "item":
                    name = parent_discovery._safe_field_name(elem.attrib["name"])
                    value = _safe_value(
                        "".join(elem.itertext()),
                        label="act-card item value",
                        require_nonempty=False,
                    )
                    signature, has_richtext = _item_signature(elem)
                    current_items.append((name, signature, has_richtext, value))
                    _require(
                        len(current_items) <= MAX_ITEMS_PER_CARD,
                        "act-card item-count safety limit exceeded",
                    )
                    elem.clear()
                elif tag == "document":
                    matched = [
                        (value, name)
                        for name, _signature, _richtext, value in current_items
                        if value in selected_codes
                    ]
                    if matched:
                        matched_card_documents.add(card_ordinal)
                        for code, name in matched:
                            matches[code].append((card_ordinal, name))
                        for name, signature, has_richtext, _value in current_items:
                            field_occurrences[name] += 1
                            field_documents[name].add(card_ordinal)
                            _require(
                                len(field_signatures[name]) < MAX_SIGNATURES_PER_FIELD
                                or signature in field_signatures[name],
                                f"too many structural signatures for act-card field {name!r}",
                            )
                            field_signatures[name].add(signature)
                            if has_richtext:
                                richtext_occurrences[name] += 1
                    current_items = []
                    elem.clear()
                stack.pop()
        except ET.ParseError as exc:
            raise JoinDiscoveryError("act-card XML parse failure") from exc

        _require(guarded.bytes_seen == info.file_size, "act-card XML not fully consumed")

    _require(not stack, "unterminated act-card XML stack")
    _require(
        len(field_occurrences) <= MAX_FIELD_NAMES,
        "matched act-card field cardinality exceeds safety bound",
    )

    duplicate_text_code_groups = sum(
        1 for ordinals in code_to_text_ordinals.values() if len(ordinals) != 1
    )
    unmatched_codes = [code for code in selected_codes if not matches.get(code)]
    ambiguous_codes = [
        code
        for code in selected_codes
        if matches.get(code) and len(matches[code]) != 1
    ]
    uniquely_matched_codes = [
        code
        for code in selected_codes
        if len(matches.get(code, [])) == 1
    ]
    unique_join_card_documents = {
        matches[code][0][0] for code in uniquely_matched_codes
    }
    join_fields: defaultdict[str, int] = defaultdict(int)
    for code in uniquely_matched_codes:
        _document, field = matches[code][0]
        join_fields[field] += 1

    if duplicate_text_code_groups:
        status = "BLOCKED_DUPLICATE_SELECTED_TEXT_CODE"
    elif unmatched_codes:
        status = "BLOCKED_UNMATCHED_SELECTED_CODES"
    elif ambiguous_codes:
        status = "BLOCKED_AMBIGUOUS_CARD_MATCH"
    elif len(unique_join_card_documents) != len(selected_codes_by_ordinal):
        status = "BLOCKED_NON_BIJECTIVE_CARD_JOIN"
    else:
        status = "COMPLETE_UNIQUE_SELECTED_TEXT_TO_CARD_JOIN"

    fields = [
        {
            "item_name": name,
            "matched_card_document_count": len(field_documents[name]),
            "occurrence_count": field_occurrences[name],
            "richtext_occurrence_count": richtext_occurrences.get(name, 0),
            "structural_signatures": sorted(field_signatures[name]),
        }
        for name in sorted(field_occurrences)
    ]

    return {
        "join_status": status,
        "selected_text_documents": len(selected_codes_by_ordinal),
        "selected_unique_codes": len(selected_codes),
        "duplicate_selected_code_groups": duplicate_text_code_groups,
        "matched_selected_unique_codes": len(selected_codes) - len(unmatched_codes),
        "unmatched_selected_unique_codes": len(unmatched_codes),
        "ambiguous_selected_unique_codes": len(ambiguous_codes),
        "uniquely_matched_selected_codes": len(uniquely_matched_codes),
        "matched_card_documents": len(matched_card_documents),
        "unique_join_card_documents": len(unique_join_card_documents),
        "card_documents_seen": card_ordinal,
        "join_field_names": [
            {"item_name": name, "unique_selected_code_matches": count}
            for name, count in sorted(join_fields.items())
        ],
        "matched_card_structure": {
            "unique_item_names": len(fields),
            "fields": fields,
        },
        "xml10_control_bytes_removed": guarded.removed_control_bytes,
        "xml10_control_removal_identity_sha256": guarded.removal_identity_sha256,
    }


def execute(
    source_path: Path,
    *,
    execution_head_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(
        len(execution_head_sha) == 40
        and execution_head_sha == execution_head_sha.lower()
        and all(ch in "0123456789abcdef" for ch in execution_head_sha),
        "execution head must be lowercase 40-hex",
    )

    policy = source_policy.validate_policy(source_policy.load_policy(), root=Path.cwd())
    _require(
        policy["admission_policy"]["requires_record_level_act_card_binding"] is True,
        "source policy no longer requires act-card binding",
    )
    _require(
        policy["admission_policy"]["payload_source_admission_executed"] is False,
        "source policy unexpectedly admits payload",
    )
    _require(
        policy["admission_policy"]["requires_selected_payload_identity"] is True,
        "source policy no longer requires selected-payload identity",
    )
    _require(
        policy["candidate_binding"]["selected_payload_identity_sha256"]
        == EXPECTED_SELECTED_PAYLOAD_IDENTITY,
        "source-policy selected-payload identity drift",
    )

    with tempfile.TemporaryDirectory(prefix="edrnpa-actcard-join-") as tmp:
        root = Path(tmp)
        snapshot = root / "verified-source.zip"
        source_bytes, source_md5, source_sha = probe._hash_file(
            source_path,
            snapshot_path=snapshot,
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
            report["selection"]["inventory_identity_sha256"]
            == EXPECTED_INVENTORY_IDENTITY,
            "candidate inventory identity drift",
        )
        _require(
            report["report_identity_sha256"] == EXPECTED_REPORT_IDENTITY,
            "candidate report identity drift",
        )
        selected_payload_identity = _selected_payload_identity(rows)
        _require(
            selected_payload_identity == EXPECTED_SELECTED_PAYLOAD_IDENTITY,
            "candidate selected-payload identity drift",
        )
        _require(
            all(row["training_eligible"] is False for row in rows),
            "candidate training eligibility widened",
        )
        _require(
            all(row["evaluation_eligible"] is False for row in rows),
            "candidate evaluation eligibility widened",
        )
        selected_ordinals = parent_discovery._selected_ordinals(rows)

        text_zip_path = root / "text.zip"
        text_member, text_zip_bytes = _extract_named_nested_zip(
            snapshot,
            text_zip_path,
            member_name=probe.NESTED_TEXT_ZIP,
        )
        text_nested, text_xml_info, text_xml_path = _single_xml_info(
            text_zip_path,
            label="text",
        )
        try:
            selected_codes = _collect_selected_text_codes(
                nested=text_nested,
                info=text_xml_info,
                selected_ordinals=selected_ordinals,
            )
        finally:
            text_nested.close()

        card_zip_path = root / "cards.zip"
        card_member, card_zip_bytes = _extract_named_nested_zip(
            snapshot,
            card_zip_path,
            member_name=CARD_ZIP_MEMBER,
        )
        card_nested, card_xml_info, card_xml_path = _single_xml_info(
            card_zip_path,
            label="act-card",
        )
        try:
            join = _discover_card_join(
                nested=card_nested,
                info=card_xml_info,
                selected_codes_by_ordinal=selected_codes,
            )
        finally:
            card_nested.close()

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
            "text_nested_zip_member": text_member,
            "text_nested_zip_bytes": text_zip_bytes,
            "text_nested_xml_member": text_xml_path,
            "text_nested_xml_bytes": text_xml_info.file_size,
            "card_nested_zip_member": card_member,
            "card_nested_zip_bytes": card_zip_bytes,
            "card_nested_xml_member": card_xml_path,
            "card_nested_xml_bytes": card_xml_info.file_size,
        },
        "candidate_binding": {
            "selected_objects": EXPECTED_SELECTED_OBJECTS,
            "selected_bytes": EXPECTED_SELECTED_BYTES,
            "inventory_identity_sha256": EXPECTED_INVENTORY_IDENTITY,
            "selected_payload_identity_sha256": selected_payload_identity,
            "report_identity_sha256": EXPECTED_REPORT_IDENTITY,
        },
        "parent_discovery_binding": {
            "execution_head_sha": PARENT_DISCOVERY_HEAD,
            "workflow_run": PARENT_DISCOVERY_RUN,
            "artifact_id": PARENT_DISCOVERY_ARTIFACT,
            "artifact_zip_sha256": PARENT_DISCOVERY_ARTIFACT_ZIP_SHA256,
            "discovery_identity_sha256": PARENT_DISCOVERY_IDENTITY,
            "observed_text_item_names": ["TEXT_ED", TEXT_CODE_FIELD],
        },
        "policy_binding": {
            "schema_version": policy["schema_version"],
            "decision_class": policy["admission_policy"]["decision_class"],
            "record_level_act_card_binding_required": policy["admission_policy"][
                "requires_record_level_act_card_binding"
            ],
            "record_level_provenance_classification_required": policy[
                "admission_policy"
            ]["record_level_provenance_classification_required"],
        },
        "selected_text_to_act_card_join": join,
        "content_boundary": {
            "legal_text_persisted": False,
            "selected_reestr_kod_values_persisted": False,
            "act_card_metadata_values_persisted": False,
            "only_field_names_structures_and_aggregate_counts_persisted": True,
        },
        "truth_boundary": {
            "record_level_payload_provenance_executed": False,
            "official_normative_act_classification_executed": False,
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
        "candidate_selected_payload_identity_sha256": selected_payload_identity,
        "candidate_report_identity_sha256": EXPECTED_REPORT_IDENTITY,
        "parent_discovery_identity_sha256": PARENT_DISCOVERY_IDENTITY,
        "discovery_identity_sha256": discovery["discovery_identity_sha256"],
        "join_status": join["join_status"],
        "selected_text_documents": join["selected_text_documents"],
        "matched_selected_unique_codes": join["matched_selected_unique_codes"],
        "unmatched_selected_unique_codes": join["unmatched_selected_unique_codes"],
        "ambiguous_selected_unique_codes": join["ambiguous_selected_unique_codes"],
        "selected_reestr_kod_values_persisted": False,
        "act_card_metadata_values_persisted": False,
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
        (args.output_dir / "selected-actcard-join-discovery.json").write_bytes(
            _canonical(discovery)
        )
        (args.output_dir / "execution-receipt.json").write_bytes(_canonical(receipt))
    except (
        JoinDiscoveryError,
        source_policy.EdrnpaSourcePolicyError,
        probe.ProbeError,
        OSError,
    ) as exc:
        print(f"BLOCKED: {exc}")
        return 2

    join = discovery["selected_text_to_act_card_join"]
    print("EDRNPA_SELECTED_ACTCARD_JOIN_DISCOVERY=" + discovery["discovery_identity_sha256"])
    print("JOIN_STATUS=" + join["join_status"])
    print("SELECTED_TEXT_DOCUMENTS=" + str(join["selected_text_documents"]))
    print("MATCHED_SELECTED_UNIQUE_CODES=" + str(join["matched_selected_unique_codes"]))
    print("UNMATCHED_SELECTED_UNIQUE_CODES=" + str(join["unmatched_selected_unique_codes"]))
    print("AMBIGUOUS_SELECTED_UNIQUE_CODES=" + str(join["ambiguous_selected_unique_codes"]))
    print("PAYLOAD_SOURCE_ADMISSION_EXECUTED=false")
    print("CANONICAL_CAPACITY_CREDIT_BYTES=0")
    print("TOKENIZER_FIT_AUTHORIZED=false")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
