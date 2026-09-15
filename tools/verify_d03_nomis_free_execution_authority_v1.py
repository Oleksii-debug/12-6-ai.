#!/usr/bin/env python3
"""Fail-closed independent authority checks for SWARM-2065 physical clean successor."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

BASE = "0a594f91ceb61586fdf5d7062e4eae2a53ed90f7"
V7_HEAD = "d3333ec1b4a508df232a5aefccd6686adda745fb"
V7_TREE = "f6bb58379e9e249583480c246b844b673be38b4c"
P623_HEAD = "70d6ccc87396d129d00771bbf0b6b29bc673bfc4"
P623_TREE = "e10c66fe59d88997a7de40de4775257da896e28a"
BLOCK_ID = "ua.verba.nomis1864.bounded24"
BLOCK_FAM = "ua.verba.public-domain.nomis1864"
BLOCK_SHA = "1eb91dbd631898c6a2efe274b700a5be0deaca243c0a9d5d30994ddadcf43598"
BLOCK_BYTES = 1659
QUAR = "e9f29dd9f710fac057550e5cd671b7f412720e1ceb36568565a79909f11cf5b6"
V7_REPORT = "80997ac88b9d604afaf652807cda2a2d9fd0f6cb75754460ae6f4aa7af6e0267"
V7_V3 = "c33e0d06a469473aac191e9b5bf7baec23322cd3e9200f0caab2633c921afd84"
BULK_REPORT = "80f3a8f20dfb82825e6c89ac1f76f2f41233296b8a426a1cf466b0fcc0892985"
SRC_SCHEMA = "12-6.d03-nomis-free-clean-successor-report.v1"
SURV_SCHEMA = "12-6.next100-065f-post-dedup-survivors.v1"
EV_SCHEMA = "12-6.d03-nomis-free-data526-successor-evidence.v1"
INV_SCHEMA = "12-6.data526-record-inventory.v1"
SELECTION_RULE = "largest_declared_capacity_then_lexicographically_smallest_source_id"
SOURCE_WORKER = "SWARM-2065-NOMIS-FREE-CLEAN-SUCCESSOR-V1"
DATA526_WORKER = "SWARM-2065-NOMIS-FREE-DATA526-SUCCESSOR-V1"

CLUSTER = sorted(
    [
        "data-bulk-code1:pallets/werkzeug:src/werkzeug/__init__.py",
        "data-bulk-code1:pallets/werkzeug:src/werkzeug/routing/__init__.py",
        "data-bulk-code1:pallets/werkzeug:src/werkzeug/wrappers/__init__.py",
    ]
)
SELECTED = "data-bulk-code1:pallets/werkzeug:src/werkzeug/routing/__init__.py"

BASE_BLOBS = {
    "tools/run_next100_065f_global_dedup_v8.py": "3a4df4b3cf6381893a58641dde479d9fd7fbe46d",
    "tools/derive_next100_065f_v8_survivors.py": "ae91d60e5d62466c69394abb1c4b27d2e49f40e3",
    "src/twelve_six/data/external_llm_provenance_quarantine_v1.py": "5615c2e4732d14cbd5dc418bb6277e52afd579e6",
    "configs/data/d03_external_llm_provenance_quarantine_v1.json": "c746f130d3100f90b233c80ffed6f877fa6eabcf",
    "tools/materialize_data_bulk_code1_permissive_python_bundle.py": "eda12f74cc8fef35f7de007a930f2bea254059e4",
    "configs/data/data_bulk_code1_permissive_python_bundle_v1.json": "05d2f5e6a83d4a8cf8159f422bd9a66a9dd3f393",
    "evidence/data_bulk_code1/permissive_python_bundle_v1_terminal.json": "b2b9b361dbf0e6c883dca4d7f006812c69cdd4e0",
    "configs/data/next100_065f_global_dedup_v8.json": "51d72c0448c4197807780e38b3cbbb94be36ba96",
    "tools/materialize_data526_record_inventory_v1.py": "9b0c5b9df66df468e7b6a4a317c2cd82412ee659",
}
BEHAVIOR = (
    "tools/run_d03_nomis_free_clean_successor_v1.py",
    "tools/materialize_d03_nomis_free_data526_successor_v1.py",
    "tools/run_d03_nomis_free_v7_data_only_v1.py",
    "tools/verify_d03_nomis_free_execution_authority_v1.py",
    ".github/workflows/d03-nomis-free-clean-successor-v1.yml",
)
P623 = {
    "tools/materialize_data526_records_from_v7.py": "190d8d5d7727230aefb04cc1464ba42e023b32eb",
    "configs/data/data526_record_materialization_v5.json": "ccc6ed614955739439e4ea09423a0ecbe2f9d3b9",
}
P623_CONFIG = "configs/data/data526_record_materialization_v5.json"

SOURCE_TRUTH_BOUNDARY = {
    "source_object_authority_only": True,
    "clean_historical_record_materialization_required": True,
    "data526_clean_record_graph_materialized": False,
    "decontamination_executed_for_successor": False,
    "post_composition_quality_privacy_passed": False,
    "balance_release_claimed": False,
    "split_pack_complete": False,
    "tokenizer_fit_authorized": False,
    "authorized_training_exposure": 0,
    "model_training_executed": False,
    "optimizer_updates": 0,
    "learned_weights_created": False,
    "final_test_payload_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights": False,
}
SOURCE_BLOCKERS = [
    "fresh_clean_historical_data526_record_materialization",
    "fresh_clean_data526_record_graph_composition",
    "reserved_evaluation_decontamination",
    "post_composition_quality_privacy",
    "balance_and_family_caps",
    "cluster_safe_split",
    "deterministic_tokenizer_and_packing",
    "post_pack_unique_loss_ledger",
]
SURVIVOR_TRUTH_BOUNDARY = {
    "source_object_authority_only": True,
    "training_record_inventory_materialized": False,
    "evaluation_decontamination_passed": False,
    "tokenizer_fit_authorized": False,
    "authorized_training_exposure": 0,
    "model_training_executed": False,
    "final_test_payload_read": False,
    "paid_compute_used": False,
}
DATA526_TRUTH_BOUNDARY = {
    "clean_data526_record_graph_materialized": True,
    "corpus_released": False,
    "decontamination_executed_for_successor": False,
    "post_composition_quality_privacy_passed": False,
    "balance_release_claimed": False,
    "split_pack_complete": False,
    "tokenizer_fit_authorized": False,
    "authorized_training_exposure": 0,
    "model_training_executed": False,
    "optimizer_updates": 0,
    "learned_weights_created": False,
    "final_test_payload_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights": False,
    "raw_payloads_committed_to_repository": False,
    "raw_payloads_uploaded_as_public_evidence": False,
}
DATA526_BLOCKERS = [
    "reserved_evaluation_decontamination",
    "post_composition_quality_privacy",
    "balance_and_family_caps",
    "cluster_safe_split",
    "deterministic_tokenizer_and_packing",
    "post_pack_unique_loss_ledger",
]


class AuthorityError(RuntimeError):
    pass


def req(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


def canon(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def blob(value: bytes) -> str:
    return hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuthorityError(f"cannot read JSON {path}: {exc}") from exc
    req(isinstance(value, dict), f"JSON root invalid: {path}")
    return value


def _require_exact_json(actual: Any, expected: Any, label: str) -> None:
    """Require closed-world JSON equality, preserving bool-vs-int distinctions."""
    req(canon(actual) == canon(expected), f"{label} drift")


def git(root: Path, *args: str, text: bool = True):
    env = dict(os.environ)
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=text,
        env=env,
    ).stdout


def gtext(root: Path, *args: str) -> str:
    return str(git(root, *args)).strip()


def gbytes(root: Path, *args: str) -> bytes:
    return bytes(git(root, *args, text=False))


def no_replace(root: Path) -> None:
    req(
        gtext(root, "for-each-ref", "--format=%(refname)", "refs/replace") == "",
        f"replace refs present: {root}",
    )


def path_at_head(root: Path, head: str, rel: str) -> None:
    try:
        worktree = (root / rel).read_bytes()
        committed = gbytes(root, "show", f"{head}:{rel}")
    except Exception as exc:
        raise AuthorityError(f"cannot bind {rel}: {exc}") from exc
    req(worktree == committed, f"worktree drift: {rel}")


def verify_product_checkout(root: Path, expected: str) -> dict[str, Any]:
    req(
        len(expected) == 40 and all(c in "0123456789abcdef" for c in expected),
        "invalid expected Product head",
    )
    no_replace(root)
    head = gtext(root, "rev-parse", "HEAD")
    req(head == expected, "Product HEAD drift")
    for rel in BEHAVIOR:
        path_at_head(root, head, rel)
    for rel, expected_blob in BASE_BLOBS.items():
        path_at_head(root, head, rel)
        req(
            blob((root / rel).read_bytes()) == expected_blob,
            f"base-main binding drift: {rel}",
        )
    return {
        "product_head_sha": head,
        "base_main_sha": BASE,
        "base_main_bound_blob_count": len(BASE_BLOBS),
    }


def _checkout(
    root: Path,
    head: str,
    tree: str,
    label: str,
    blobs: Mapping[str, str] | None = None,
) -> dict[str, str]:
    no_replace(root)
    req(gtext(root, "rev-parse", "HEAD") == head, f"{label} HEAD drift")
    req(gtext(root, "rev-parse", "HEAD^{tree}") == tree, f"{label} tree drift")
    req(
        gtext(root, "status", "--porcelain=v1", "--untracked-files=all") == "",
        f"{label} worktree dirty",
    )
    for rel, expected_blob in (blobs or {}).items():
        path_at_head(root, head, rel)
        req(
            blob((root / rel).read_bytes()) == expected_blob,
            f"{label} blob drift: {rel}",
        )
    return {"head_sha": head, "tree_sha": tree}


def verify_v7_checkout(root: Path):
    return _checkout(root, V7_HEAD, V7_TREE, "terminal V7")


def verify_pr623_checkout(root: Path):
    return _checkout(root, P623_HEAD, P623_TREE, "PR623", P623)


def selfhash(value: Mapping[str, Any], key: str) -> str:
    body = dict(value)
    body.pop(key, None)
    return sha(canon(body))


def srcrows(dedup: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = dedup.get("sources")
    req(isinstance(rows, list), "source rows missing")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        req(
            isinstance(row, Mapping) and isinstance(row.get("source_id"), str),
            "invalid source row",
        )
        source_id = str(row["source_id"])
        req(source_id not in result, "duplicate source id")
        result[source_id] = row
    return result


def clusters(dedup: Mapping[str, Any]) -> list[list[str]]:
    terminal = dedup.get("terminal_candidates")
    req(
        isinstance(terminal, Mapping)
        and isinstance(terminal.get("duplicate_clusters"), list),
        "clusters missing",
    )
    result: list[list[str]] = []
    for raw_cluster in terminal["duplicate_clusters"]:
        req(
            isinstance(raw_cluster, Sequence)
            and not isinstance(raw_cluster, (str, bytes)),
            "invalid cluster",
        )
        cluster = sorted(map(str, raw_cluster))
        req(
            len(cluster) >= 2 and len(cluster) == len(set(cluster)),
            "invalid cluster members",
        )
        result.append(cluster)
    return sorted(result)


def _expected_survivor_core(
    report: Mapping[str, Any], rows: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    dropped = set(CLUSTER) - {SELECTED}
    expected_ids = sorted(set(rows) - dropped)
    survivor_rows = [
        {
            "source_id": source_id,
            "source_family": rows[source_id]["source_family"],
            "modality": rows[source_id]["modality"],
            "declared_capacity_bytes": rows[source_id]["declared_capacity_bytes"],
            "verified_raw_sha256": rows[source_id]["verified_raw_sha256"],
            "normalized_sha256": rows[source_id]["normalized_sha256"],
            "stable_origin_id_sha256": rows[source_id]["stable_origin_id_sha256"],
            "stable_object_id_sha256": rows[source_id]["stable_object_id_sha256"],
        }
        for source_id in expected_ids
    ]
    survivor_capacity = sum(int(row["declared_capacity_bytes"]) for row in survivor_rows)
    cap = max(int(rows[source_id]["declared_capacity_bytes"]) for source_id in CLUSTER)
    tied = sorted(
        source_id
        for source_id in CLUSTER
        if int(rows[source_id]["declared_capacity_bytes"]) == cap
    )
    req(tied[0] == SELECTED, "independent survivor selection oracle drift")
    modality_counts = {
        modality: {
            "source_object_count": sum(
                1 for row in survivor_rows if row["modality"] == modality
            ),
            "declared_capacity_bytes": sum(
                int(row["declared_capacity_bytes"])
                for row in survivor_rows
                if row["modality"] == modality
            ),
        }
        for modality in ("uk", "en", "code")
    }
    dedup = report["dedup_v3"]
    return {
        "schema_version": SURV_SCHEMA,
        "selection_rule": SELECTION_RULE,
        "v8_report_sha256": report["report_sha256"],
        "nested_v3_report_sha256": dedup["report_sha256"],
        "pre_dedup_source_object_count": len(rows),
        "post_dedup_survivor_source_object_count": len(survivor_rows),
        "pre_dedup_declared_capacity_bytes": report["source_vector"][
            "source_capacity_bytes_before_global_dedup"
        ],
        "post_dedup_declared_capacity_bytes": survivor_capacity,
        "duplicate_discount_bytes": report["source_vector"]["duplicate_discount_bytes"],
        "duplicate_cluster_count": 1,
        "duplicate_clusters": [
            {
                "member_source_ids": CLUSTER,
                "selected_source_id": SELECTED,
                "selected_declared_capacity_bytes": cap,
            }
        ],
        "survivors": survivor_rows,
        "by_modality": modality_counts,
        "truth_boundary": SURVIVOR_TRUTH_BOUNDARY,
    }


def verify_source_authority(
    *,
    current_root: Path,
    expected_product_head: str,
    report: Mapping[str, Any],
    survivor: Mapping[str, Any],
) -> dict[str, Any]:
    product = verify_product_checkout(current_root, expected_product_head)
    req(report.get("schema_version") == SRC_SCHEMA, "source schema drift")
    req(report.get("worker_id") == SOURCE_WORKER, "source worker drift")
    req(report.get("execution_profile") == "LOCAL_FREE", "source execution profile drift")
    req(report.get("base_main_sha") == BASE, "base drift")
    req(
        report.get("report_sha256") == selfhash(report, "report_sha256"),
        "source self hash drift",
    )
    req(report.get("raw_text_emitted") is False, "raw text leak")
    _require_exact_json(
        report.get("truth_boundary"), SOURCE_TRUTH_BOUNDARY, "source truth boundary"
    )
    _require_exact_json(
        report.get("remaining_blockers"), SOURCE_BLOCKERS, "source remaining blockers"
    )

    incumbent = report.get("incumbent_bindings")
    req(isinstance(incumbent, Mapping), "incumbent bindings missing")
    expected_incumbent = {
        "v8_tool_git_blob_sha1": BASE_BLOBS["tools/run_next100_065f_global_dedup_v8.py"],
        "survivor_tool_git_blob_sha1": BASE_BLOBS[
            "tools/derive_next100_065f_v8_survivors.py"
        ],
        "quarantine_module_git_blob_sha1": BASE_BLOBS[
            "src/twelve_six/data/external_llm_provenance_quarantine_v1.py"
        ],
        "quarantine_config_git_blob_sha1": BASE_BLOBS[
            "configs/data/d03_external_llm_provenance_quarantine_v1.json"
        ],
        "historical_v7_head_sha": V7_HEAD,
        "historical_v7_report_sha256": V7_REPORT,
        "historical_v7_nested_v3_sha256": V7_V3,
        "bulk_terminal_report_identity_sha256": BULK_REPORT,
    }
    _require_exact_json(dict(incumbent), expected_incumbent, "incumbent binding")

    deauth = report.get("deauthorization")
    req(isinstance(deauth, Mapping), "deauth missing")
    req(
        deauth.get("quarantine_identity_sha256") == QUAR
        and deauth.get("blocked_source_id") == BLOCK_ID
        and deauth.get("blocked_family") == BLOCK_FAM
        and deauth.get("blocked_payload_sha256") == BLOCK_SHA
        and deauth.get("blocked_payload_bytes") == BLOCK_BYTES,
        "deauth identity drift",
    )
    req(
        deauth.get("removed_before_new_global_dedup") is True
        and deauth.get("pre_source_object_count") == 35
        and deauth.get("post_source_object_count") == 34,
        "deauth transition drift",
    )

    historical = report.get("clean_historical")
    req(isinstance(historical, Mapping), "clean historical missing")
    historical_vector = historical.get("source_vector")
    req(isinstance(historical_vector, Mapping), "historical vector missing")
    req(
        historical_vector.get("source_object_count") == 34
        and historical_vector.get("source_capacity_bytes_before_global_dedup") == 2_213_956
        and historical_vector.get("source_family_counts") == {"uk": 3, "en": 5, "code": 6},
        "historical vector drift",
    )
    historical_dedup = historical.get("dedup_v3")
    req(isinstance(historical_dedup, Mapping), "historical v3 missing")
    historical_rows = srcrows(historical_dedup)
    req(
        len(historical_rows) == 34 and BLOCK_ID not in historical_rows,
        "historical source cut drift",
    )
    req(
        all(
            row.get("source_family") != BLOCK_FAM
            and row.get("verified_raw_sha256") != BLOCK_SHA
            for row in historical_rows.values()
        ),
        "historical prohibited root survived",
    )

    vector = report.get("source_vector")
    req(isinstance(vector, Mapping), "vector missing")
    req(
        vector.get("source_object_count") == 263
        and vector.get("source_family_counts") == {"uk": 3, "en": 5, "code": 12}
        and vector.get("source_capacity_bytes_before_global_dedup") == 6_093_965
        and vector.get("conservative_unique_capacity_bytes_after_global_dedup") == 6_093_662
        and vector.get("duplicate_discount_bytes") == 303
        and vector.get("duplicate_cluster_count") == 1,
        "composed vector drift",
    )
    dedup = report.get("dedup_v3")
    req(isinstance(dedup, Mapping), "composed v3 missing")
    rows = srcrows(dedup)
    req(len(rows) == 263 and BLOCK_ID not in rows, "composed source cut drift")
    req(
        all(
            row.get("source_family") != BLOCK_FAM
            and row.get("verified_raw_sha256") != BLOCK_SHA
            for row in rows.values()
        ),
        "composed prohibited root survived",
    )
    req(clusters(dedup) == [CLUSTER], "exact Werkzeug cluster drift")

    expected_core = _expected_survivor_core(report, rows)
    expected_survivor = dict(expected_core)
    expected_survivor["survivor_authority_sha256"] = sha(canon(expected_core))
    _require_exact_json(survivor, expected_survivor, "complete survivor authority")

    return {
        **product,
        "source_report_sha256": report["report_sha256"],
        "survivor_authority_sha256": survivor["survivor_authority_sha256"],
        "duplicate_cluster_selected_source_id": SELECTED,
    }


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line.decode("utf-8"))
        except Exception as exc:
            raise AuthorityError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
        req(isinstance(value, dict), "record not object")
        records.append(value)
    req(
        raw == b"".join(canon(record) + b"\n" for record in records),
        f"noncanonical JSONL: {path}",
    )
    return records, raw


def inventory(records: list[dict[str, Any]]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        req(
            set(record) == {"record_id", "source_id", "family", "modality", "normalized_payload"},
            "record schema drift",
        )
        record_id = record["record_id"]
        req(
            all(isinstance(record[key], str) and record[key] for key in record),
            "record text field invalid",
        )
        req(record_id not in seen, "duplicate record id")
        seen.add(record_id)
        raw = record["normalized_payload"].encode("utf-8")
        items.append(
            {
                "record_id": record_id,
                "source_id": record["source_id"],
                "family": record["family"],
                "modality": record["modality"],
                "payload_sha256": sha(raw),
                "payload_bytes": len(raw),
            }
        )
    items.sort(key=lambda item: item["record_id"])
    payload_projection = [
        {
            "record_id": item["record_id"],
            "payload_sha256": item["payload_sha256"],
            "payload_bytes": item["payload_bytes"],
        }
        for item in items
    ]
    return {
        "schema_version": INV_SCHEMA,
        "record_count": len(items),
        "total_payload_bytes": sum(item["payload_bytes"] for item in items),
        "record_inventory_digest_sha256": sha(canon(items)),
        "payload_inventory_digest_sha256": sha(canon(payload_projection)),
        "records": items,
    }


def reject(records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        req(
            record.get("source_id") != BLOCK_ID
            and record.get("record_id") != BLOCK_ID
            and record.get("family") != BLOCK_FAM,
            "blocked identity survived DATA526",
        )
        payload = record.get("normalized_payload")
        req(
            isinstance(payload, str) and sha(payload.encode("utf-8")) != BLOCK_SHA,
            "blocked payload survived DATA526",
        )


def _items_by_record_id(
    physical_inventory: Mapping[str, Any], label: str
) -> dict[str, Mapping[str, Any]]:
    rows = physical_inventory.get("records")
    req(isinstance(rows, list), f"{label} inventory records missing")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        req(
            isinstance(row, Mapping) and isinstance(row.get("record_id"), str),
            f"{label} inventory row invalid",
        )
        record_id = str(row["record_id"])
        req(record_id not in result, f"{label} duplicate record id")
        result[record_id] = row
    return result


def _require_item(
    actual: Mapping[str, Any],
    *,
    record_id: str,
    source_id: str,
    family: str,
    modality: str,
    payload_sha256: str,
    payload_bytes: int | None,
    label: str,
) -> None:
    expected = {
        "record_id": record_id,
        "source_id": source_id,
        "family": family,
        "modality": modality,
        "payload_sha256": payload_sha256,
    }
    for key, value in expected.items():
        req(actual.get(key) == value, f"{label} {key} drift: {record_id}")
    if payload_bytes is not None:
        req(
            type(actual.get("payload_bytes")) is int
            and actual.get("payload_bytes") == payload_bytes,
            f"{label} payload bytes drift: {record_id}",
        )


def _verify_historical_content_identity(
    historical_inventory: Mapping[str, Any],
    historical_rows: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    items = _items_by_record_id(historical_inventory, "historical")
    data213 = config["data213_normalized_artifact"]["sources"]
    kmu = config["kmu_authority"]["sources"]
    cpython = config["cpython_authority"]
    cpython_id = str(cpython["source_id"])
    consumed: set[str] = set()

    for source_id, source in historical_rows.items():
        family = str(source["source_family"])
        modality = str(source["modality"])
        if source_id in data213:
            spec = data213[source_id]
            record = items.get(source_id)
            req(isinstance(record, Mapping), f"missing DATA213 record: {source_id}")
            _require_item(
                record,
                record_id=source_id,
                source_id=source_id,
                family=family,
                modality=modality,
                payload_sha256=str(spec["payload_sha256"]),
                payload_bytes=int(spec["payload_bytes"]),
                label="DATA213 record",
            )
            consumed.add(source_id)
        elif source_id in kmu:
            spec = kmu[source_id]
            record = items.get(source_id)
            req(isinstance(record, Mapping), f"missing KMU record: {source_id}")
            _require_item(
                record,
                record_id=source_id,
                source_id=source_id,
                family=family,
                modality=modality,
                payload_sha256=str(spec["payload_sha256"]),
                payload_bytes=int(spec["payload_bytes"]),
                label="KMU record",
            )
            consumed.add(source_id)
        elif source_id == cpython_id:
            accepted = list(cpython["accepted_normalized_sha256"])
            cpython_bytes = 0
            for index, payload_hash in enumerate(accepted, 1):
                record_id = f"{source_id}#accepted-{index:02d}-{payload_hash[:12]}"
                record = items.get(record_id)
                req(
                    isinstance(record, Mapping),
                    f"missing CPython accepted record: {record_id}",
                )
                _require_item(
                    record,
                    record_id=record_id,
                    source_id=source_id,
                    family=family,
                    modality=modality,
                    payload_sha256=str(payload_hash),
                    payload_bytes=None,
                    label="CPython record",
                )
                req(
                    type(record.get("payload_bytes")) is int
                    and int(record["payload_bytes"]) > 0,
                    f"CPython payload bytes invalid: {record_id}",
                )
                cpython_bytes += int(record["payload_bytes"])
                consumed.add(record_id)
            req(
                len(accepted) == int(cpython["accepted_chunk_count"])
                and cpython_bytes == int(cpython["accepted_capacity_bytes"]),
                "CPython accepted identity aggregate drift",
            )
        else:
            record = items.get(source_id)
            req(isinstance(record, Mapping), f"missing direct record: {source_id}")
            req(
                type(source.get("comparison_payload_bytes")) is int
                and source.get("comparison_payload_bytes") == source.get("declared_capacity_bytes"),
                f"direct comparison byte authority drift: {source_id}",
            )
            _require_item(
                record,
                record_id=source_id,
                source_id=source_id,
                family=family,
                modality=modality,
                payload_sha256=str(source["comparison_payload_sha256"]),
                payload_bytes=int(source["comparison_payload_bytes"]),
                label="direct record",
            )
            consumed.add(source_id)

    req(consumed == set(items), "historical record identity has unexpected/missing records")


def _verify_final_content_identity(
    final_inventory: Mapping[str, Any],
    historical_inventory: Mapping[str, Any],
    composed_rows: Mapping[str, Mapping[str, Any]],
    survivor: Mapping[str, Any],
) -> None:
    historical_items = _items_by_record_id(historical_inventory, "historical")
    survivor_rows = survivor.get("survivors")
    req(isinstance(survivor_rows, list), "survivor rows missing")
    survivor_ids = {
        str(row["source_id"]) for row in survivor_rows if isinstance(row, Mapping)
    }
    historical_source_ids = {
        str(row["source_id"]) for row in historical_items.values()
    }
    req(
        historical_source_ids <= survivor_ids,
        "historical source unexpectedly absent from survivor authority",
    )

    expected: dict[str, Mapping[str, Any] | dict[str, Any]] = dict(historical_items)
    for source_id in sorted(survivor_ids - historical_source_ids):
        source = composed_rows.get(source_id)
        req(isinstance(source, Mapping), f"missing composed source row: {source_id}")
        expected[source_id] = {
            "record_id": source_id,
            "source_id": source_id,
            "family": source["source_family"],
            "modality": source["modality"],
            "payload_sha256": source["verified_raw_sha256"],
            "payload_bytes": source["declared_capacity_bytes"],
        }
    _require_exact_json(
        list(final_inventory["records"]),
        [expected[record_id] for record_id in sorted(expected)],
        "complete DATA526 per-record content projection",
    )


def verify_data526_authority(
    *,
    current_root: Path,
    expected_product_head: str,
    source_report: Mapping[str, Any],
    survivor: Mapping[str, Any],
    historical_records_path: Path,
    records_path: Path,
    inventory: Mapping[str, Any],
    evidence: Mapping[str, Any],
    pr623_root: Path | None = None,
) -> dict[str, Any]:
    source_summary = verify_source_authority(
        current_root=current_root,
        expected_product_head=expected_product_head,
        report=source_report,
        survivor=survivor,
    )
    req(pr623_root is not None, "PR623 checkout is required for retained authority")
    verify_pr623_checkout(pr623_root)
    config = _read_json(pr623_root / P623_CONFIG)

    historical_records, historical_raw = read_jsonl(historical_records_path)
    records, records_raw = read_jsonl(records_path)
    historical_inventory = globals()["inventory"](historical_records)
    final_inventory = globals()["inventory"](records)
    req(
        historical_inventory["record_count"] == 47
        and historical_inventory["total_payload_bytes"] == 2_213_956,
        "historical DATA526 aggregate drift",
    )
    req(
        final_inventory["record_count"] == 274
        and final_inventory["total_payload_bytes"] == 6_093_662,
        "DATA526 aggregate drift",
    )
    _require_exact_json(inventory, final_inventory, "retained inventory physical rehash")
    reject(historical_records)
    reject(records)

    source_dedup = source_report["dedup_v3"]
    composed_rows = srcrows(source_dedup)
    historical_dedup = source_report["clean_historical"]["dedup_v3"]
    historical_rows = srcrows(historical_dedup)
    historical_ids = {str(record["source_id"]) for record in historical_records}
    req(historical_ids == set(historical_rows), "physical historical source set drift")

    _verify_historical_content_identity(historical_inventory, historical_rows, config)
    _verify_final_content_identity(
        final_inventory, historical_inventory, composed_rows, survivor
    )

    final_ids = {str(record["source_id"]) for record in records}
    survivor_ids = {
        str(row["source_id"])
        for row in survivor["survivors"]
        if isinstance(row, Mapping)
    }
    req(final_ids == survivor_ids, "DATA526 source set != survivor set")
    req(
        SELECTED in final_ids
        and all(source_id == SELECTED or source_id not in final_ids for source_id in CLUSTER),
        "physical Werkzeug survivor drift",
    )

    req(
        evidence.get("schema_version") == EV_SCHEMA
        and evidence.get("worker_id") == DATA526_WORKER
        and evidence.get("execution_profile") == "LOCAL_FREE"
        and evidence.get("execution_head_sha") == expected_product_head,
        "evidence execution binding drift",
    )
    req(
        evidence.get("source_report_sha256") == source_report.get("report_sha256")
        and evidence.get("survivor_authority_sha256")
        == survivor.get("survivor_authority_sha256"),
        "evidence upstream binding drift",
    )
    req(
        evidence.get("bulk_report_identity_sha256") == BULK_REPORT,
        "bulk report identity drift",
    )
    historical_materializer = evidence.get("historical_materializer")
    req(
        isinstance(historical_materializer, Mapping)
        and historical_materializer.get("head_sha") == P623_HEAD
        and historical_materializer.get("tool_git_blob_sha1")
        == P623["tools/materialize_data526_records_from_v7.py"]
        and historical_materializer.get("config_git_blob_sha1") == P623[P623_CONFIG]
        and historical_materializer.get("transform_contract_reused_without_modification")
        is True,
        "PR623 evidence binding drift",
    )

    evidence_historical = evidence.get("clean_historical")
    evidence_final = evidence.get("clean_data526")
    req(
        isinstance(evidence_historical, Mapping)
        and isinstance(evidence_final, Mapping),
        "clean evidence sections missing",
    )
    req(
        evidence_historical.get("record_count") == 47
        and evidence_historical.get("total_payload_bytes") == 2_213_956
        and evidence_historical.get("record_payload_jsonl_sha256") == sha(historical_raw)
        and evidence_historical.get("record_inventory_digest_sha256")
        == historical_inventory["record_inventory_digest_sha256"]
        and evidence_historical.get("payload_inventory_digest_sha256")
        == historical_inventory["payload_inventory_digest_sha256"],
        "historical evidence rehash drift",
    )
    req(
        evidence_final.get("record_count") == 274
        and evidence_final.get("source_object_count") == 261
        and evidence_final.get("total_payload_bytes") == 6_093_662
        and evidence_final.get("record_payload_jsonl_sha256") == sha(records_raw)
        and evidence_final.get("record_inventory_digest_sha256")
        == final_inventory["record_inventory_digest_sha256"]
        and evidence_final.get("payload_inventory_digest_sha256")
        == final_inventory["payload_inventory_digest_sha256"],
        "DATA526 evidence rehash drift",
    )
    req(
        evidence.get("raw_text_emitted_to_durable_evidence") is False,
        "durable raw text leak",
    )
    _require_exact_json(
        evidence.get("truth_boundary"), DATA526_TRUTH_BOUNDARY, "DATA526 truth boundary"
    )
    _require_exact_json(
        evidence.get("remaining_blockers"), DATA526_BLOCKERS, "DATA526 remaining blockers"
    )
    req(
        evidence.get("evidence_identity_sha256")
        == selfhash(evidence, "evidence_identity_sha256"),
        "evidence self hash drift",
    )
    return {
        **source_summary,
        "data526_evidence_identity_sha256": evidence["evidence_identity_sha256"],
        "record_count": 274,
        "payload_bytes": 6_093_662,
    }
