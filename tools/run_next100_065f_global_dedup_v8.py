#!/usr/bin/env python3
"""Current-main successor global dedup over terminal V7 plus DATA-BULK-CODE-1.

This intentionally reuses the incumbent NEXT100-065 V3/V7 matcher from its exact
terminal execution head. It does not fork or rewrite matching semantics. The exact
35-object V7 graph is reconstructed first, then the 229 exact source files admitted by
merged PR #818 are materialized and appended before one global audit is executed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "12-6.next100-065f-global-dedup.v8"
REPORT_SCHEMA = "12-6.next100-065f-global-dedup-report.v8"
WORKER_ID = "NEXT100-065F-CURRENT-MAIN-GLOBAL-DEDUP-V8"
EXPECTED_MAIN = "4fa4839f2882984e7bf148dc38ce315d51b60957"
EXPECTED_V7_HEAD = "d3333ec1b4a508df232a5aefccd6686adda745fb"
EXPECTED_V7_REPORT = "80997ac88b9d604afaf652807cda2a2d9fd0f6cb75754460ae6f4aa7af6e0267"
EXPECTED_V7_DEDUP = "c33e0d06a469473aac191e9b5bf7baec23322cd3e9200f0caab2633c921afd84"
EXPECTED_BULK_REPORT = "80f3a8f20dfb82825e6c89ac1f76f2f41233296b8a426a1cf466b0fcc0892985"
EXPECTED_BULK_CONTRACT = "7fd2228208f928859ebe68e947a72c977cda6952035a654d12923ce3a19a7dd6"
EXPECTED_BULK_BYTES = 3_880_009
EXPECTED_BULK_FILES = 229
EXPECTED_BASE_BYTES = 2_215_615
EXPECTED_BASE_OBJECTS = 35
EXPECTED_COMPOSED_BYTES = 6_095_624
EXPECTED_COMPOSED_OBJECTS = 264
EXPECTED_FAMILIES = {"uk": 4, "en": 5, "code": 12}
EXPECTED_BULK_FAMILIES = [
    "github:pallets/flask",
    "github:pallets/click",
    "github:pallets/jinja",
    "github:pallets/werkzeug",
    "github:agronholm/anyio",
    "github:pytest-dev/pytest",
]


class V8Error(RuntimeError):
    """Fail-closed V8 execution error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V8Error(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    core = dict(value)
    core.pop("report_sha256", None)
    return _sha256(_canonical_bytes(core))


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V8Error(f"cannot read V8 config: {exc}") from exc
    _validate_config(config)
    return config


def _validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == SCHEMA, "V8 schema drift")
    _require(config.get("worker_id") == WORKER_ID, "V8 worker drift")
    _require(config.get("execution_profile") == "LOCAL_FREE", "LOCAL_FREE boundary weakened")
    main = config.get("main_authority", {})
    _require(main.get("sha") == EXPECTED_MAIN, "main authority drift")
    _require(main.get("merged_pr") == 818, "main merged PR drift")

    base = config.get("baseline_v7", {})
    _require(base.get("head_sha") == EXPECTED_V7_HEAD, "V7 head drift")
    _require(base.get("artifact_id") == 9635595510, "V7 artifact drift")
    _require(
        base.get("artifact_zip_sha256")
        == "cca6921a2093d4e033976b23b0af180e9dc1945b624b82e218780f8d20bafd18",
        "V7 artifact ZIP identity drift",
    )
    _require(
        base.get("artifact_report_file_sha256")
        == "a917a5f240e2eda0015fd09564b29430ca8c95a9916a03f734a4463fe458c08f",
        "V7 report file identity drift",
    )
    _require(base.get("report_sha256") == EXPECTED_V7_REPORT, "V7 report identity drift")
    _require(base.get("dedup_v3_report_sha256") == EXPECTED_V7_DEDUP, "V7 dedup identity drift")
    _require(base.get("source_object_count") == EXPECTED_BASE_OBJECTS, "V7 object count drift")
    _require(
        base.get("source_capacity_bytes_after_global_dedup") == EXPECTED_BASE_BYTES,
        "V7 capacity drift",
    )
    _require(base.get("source_family_counts") == {"uk": 4, "en": 5, "code": 6}, "V7 family drift")

    bulk = config.get("data_bulk_code1", {})
    _require(bulk.get("issue") == 635 and bulk.get("merged_pr") == 818, "bulk authority drift")
    _require(
        bulk.get("execution_head_sha") == "a045c602bfcead862f7852924fb78a5d78c992d6",
        "bulk execution head drift",
    )
    _require(bulk.get("workflow_run_id") == 34155446113, "bulk workflow run drift")
    _require(bulk.get("artifact_id") == 10030825509, "bulk artifact id drift")
    _require(
        bulk.get("artifact_digest")
        == "sha256:9febb6e4e900df63c58b1cb0ed2a7003df56d6e9bd593e4b4adbaa54496010ac",
        "bulk artifact digest drift",
    )
    _require(bulk.get("contract_identity_sha256") == EXPECTED_BULK_CONTRACT, "bulk contract drift")
    _require(bulk.get("report_identity_sha256") == EXPECTED_BULK_REPORT, "bulk report drift")
    _require(bulk.get("eligible_file_count") == EXPECTED_BULK_FILES, "bulk file count drift")
    _require(bulk.get("eligible_utf8_bytes") == EXPECTED_BULK_BYTES, "bulk byte count drift")
    _require(bulk.get("source_family_ids") == EXPECTED_BULK_FAMILIES, "bulk family vector drift")

    composed = config.get("expected_composed_input", {})
    _require(composed.get("source_object_count") == EXPECTED_COMPOSED_OBJECTS, "composed object drift")
    _require(
        composed.get("source_capacity_bytes_before_global_dedup") == EXPECTED_COMPOSED_BYTES,
        "composed capacity drift",
    )
    _require(composed.get("source_family_counts") == EXPECTED_FAMILIES, "composed family drift")

    boundary = config.get("claim_boundary", {})
    false_keys = (
        "corpus_released",
        "decontamination_reused_as_pass_for_new_bytes",
        "post_composition_quality_privacy_pass_claimed",
        "balance_release_claimed",
        "split_pack_complete",
        "postpack_unique_loss_ledger_complete",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "final_test_payload_read",
        "paid_compute_used",
    )
    for key in false_keys:
        _require(boundary.get(key) is False, f"claim boundary weakened: {key}")
    _require(boundary.get("authorized_training_exposure") == 0, "training exposure must remain zero")


def _load_module_from_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_v7(v7_root: Path) -> Any:
    source_root = v7_root / "src"
    _require(source_root.is_dir(), f"missing V7 source root: {source_root}")
    source_text = str(source_root.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return importlib.import_module("twelve_six.data.cross_source_capacity_audit_v7")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V8Error(f"cannot read JSON {path}: {exc}") from exc


def _capture_terminal_v7(v7_root: Path, config: Mapping[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, bytes]]:
    v7 = _load_v7(v7_root)
    paths = {
        "base": "configs/data/next100_065_cross_source_dedup_v3.json",
        "v4": "configs/data/next100_065b_cross_source_dedup_v4.json",
        "v5": "configs/data/next100_065c_cross_source_dedup_v5.json",
        "v6": "configs/data/next100_065d_cross_source_dedup_v6.json",
        "v7": "configs/data/next100_065e_cross_source_dedup_v7.json",
    }
    docs = {key: _read_json(v7_root / rel) for key, rel in paths.items()}
    captured: dict[str, Any] = {}
    original = v7.v6.v3.audit_payloads

    def capture(inventory: Mapping[str, Any], payloads: Mapping[str, bytes]) -> dict[str, Any]:
        captured["inventory"] = copy.deepcopy(dict(inventory))
        captured["payloads"] = dict(payloads)
        return original(inventory, payloads)

    v7.v6.v3.audit_payloads = capture
    try:
        baseline_report = v7.audit_live(docs["base"], docs["v4"], docs["v5"], docs["v6"], docs["v7"])
    finally:
        v7.v6.v3.audit_payloads = original

    _require(baseline_report.get("report_sha256") == config["baseline_v7"]["report_sha256"], "reproduced V7 report identity drift")
    _require(
        baseline_report.get("dedup_v3", {}).get("report_sha256")
        == config["baseline_v7"]["dedup_v3_report_sha256"],
        "reproduced V7 dedup identity drift",
    )
    _require("inventory" in captured and "payloads" in captured, "failed to capture terminal V7 payload graph")
    _require(len(captured["inventory"].get("sources", [])) == EXPECTED_BASE_OBJECTS, "captured V7 source count drift")
    _require(len(captured["payloads"]) == EXPECTED_BASE_OBJECTS, "captured V7 payload count drift")
    return v7, baseline_report, captured["inventory"], captured["payloads"]


def _validate_bulk_terminal_evidence(current_root: Path, config: Mapping[str, Any]) -> None:
    evidence = _read_json(current_root / "evidence/data_bulk_code1/permissive_python_bundle_v1_terminal.json")
    bulk = config["data_bulk_code1"]
    _require(evidence.get("contract_identity_sha256") == bulk["contract_identity_sha256"], "terminal bulk contract drift")
    _require(evidence.get("execution_head_sha") == bulk["execution_head_sha"], "terminal bulk head drift")
    _require(evidence.get("workflow_run_id") == bulk["workflow_run_id"], "terminal bulk run drift")
    _require(evidence.get("workflow_conclusion") == "success", "terminal bulk workflow non-success")
    artifact = evidence.get("artifact", {})
    _require(artifact.get("id") == bulk["artifact_id"], "terminal bulk artifact drift")
    _require(artifact.get("digest") == bulk["artifact_digest"], "terminal bulk artifact digest drift")
    _require(
        artifact.get("retained_report_identity_sha256") == bulk["report_identity_sha256"],
        "terminal bulk retained report drift",
    )
    aggregate = evidence.get("aggregate", {})
    _require(aggregate.get("eligible_file_count") == EXPECTED_BULK_FILES, "terminal bulk file count drift")
    _require(aggregate.get("eligible_utf8_bytes") == EXPECTED_BULK_BYTES, "terminal bulk byte count drift")
    _require(aggregate.get("source_family_count") == len(EXPECTED_BULK_FAMILIES), "terminal bulk family count drift")
    _require(evidence.get("claim_boundary", {}).get("automatic_canonical_capacity_credit") is False, "bulk evidence auto-credit weakened")
    _require(evidence.get("claim_boundary", {}).get("authorized_training_exposure") == 0, "bulk evidence training exposure drift")


def _source_id(repository: str, path: str) -> str:
    return f"data-bulk-code1:{repository}:{path}"


def _materialize_bulk(current_root: Path, workspace: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bytes]]:
    _validate_bulk_terminal_evidence(current_root, config)
    materializer = _load_module_from_file(
        "_next100_065f_data_bulk_materializer",
        current_root / "tools/materialize_data_bulk_code1_permissive_python_bundle.py",
    )
    contract = materializer.load_contract(current_root / "configs/data/data_bulk_code1_permissive_python_bundle_v1.json")
    report = materializer.materialize(contract, workspace)
    _require(report.get("report_identity_sha256") == EXPECTED_BULK_REPORT, "fresh bulk report identity drift")
    _require(report.get("eligible_file_count") == EXPECTED_BULK_FILES, "fresh bulk file count drift")
    _require(report.get("eligible_utf8_bytes") == EXPECTED_BULK_BYTES, "fresh bulk byte count drift")
    _require(report.get("source_family_count") == len(EXPECTED_BULK_FAMILIES), "fresh bulk family count drift")
    _require(report.get("security", {}).get("credential_scan") == "PASS_NO_HIGH_CONFIDENCE_HITS", "fresh bulk credential scan drift")

    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    families: list[str] = []
    for source in report["sources"]:
        repository = str(source["repository"])
        family = str(source["family_id"])
        families.append(family)
        repo_dir = workspace / repository.replace("/", "__")
        commit = str(source["commit"])
        for item in source["files"]:
            path = str(item["path"])
            raw = (repo_dir / path).read_bytes()
            _require(len(raw) == int(item["utf8_bytes"]), f"bulk payload byte drift: {repository}:{path}")
            _require(_sha256(raw) == item["sha256"], f"bulk payload hash drift: {repository}:{path}")
            sid = _source_id(repository, path)
            _require(sid not in payloads, f"duplicate bulk source id: {sid}")
            rows.append(
                {
                    "source_id": sid,
                    "source_family": family,
                    "stable_origin_id": family,
                    "stable_object_id": f"sha256:{item['sha256']}",
                    "modality": "code",
                    "evidence_status": "DEDICATED_TERMINAL",
                    "authority_ref": "DATA-BULK-CODE-1 PR#818 run 34155446113",
                    "declared_capacity_bytes": len(raw),
                    "expected_raw_bytes": len(raw),
                    "expected_raw_sha256": item["sha256"],
                    "acquisition_url": f"https://raw.githubusercontent.com/{repository}/{commit}/{path}",
                    "origin_key": f"github:{repository}:{commit}:{path}",
                }
            )
            payloads[sid] = raw
    _require(families == EXPECTED_BULK_FAMILIES, f"fresh bulk family ordering drift: {families}")
    _require(len(rows) == EXPECTED_BULK_FILES and len(payloads) == EXPECTED_BULK_FILES, "fresh bulk object cardinality drift")
    _require(sum(len(raw) for raw in payloads.values()) == EXPECTED_BULK_BYTES, "fresh bulk payload byte arithmetic drift")
    return report, rows, payloads


def _family_counts(dedup_report: Mapping[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for modality in ("uk", "en", "code"):
        result[modality] = len(
            {
                row["source_family"]
                for row in dedup_report.get("sources", [])
                if row.get("modality") == modality
            }
        )
    return result


def run_audit(current_root: Path, v7_root: Path, workspace: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    v7, baseline_report, inventory, payloads = _capture_terminal_v7(v7_root, config)
    bulk_report, bulk_rows, bulk_payloads = _materialize_bulk(current_root, workspace, config)

    combined_inventory = copy.deepcopy(inventory)
    existing_ids = {row.get("source_id") for row in combined_inventory.get("sources", [])}
    _require(not (existing_ids & set(bulk_payloads)), "bulk source id collides with terminal V7 graph")
    combined_inventory["sources"] = [*combined_inventory["sources"], *bulk_rows]
    combined_inventory["final_refresh_required"] = False
    combined_inventory["terminal_refresh_cutoff_utc"] = "2026-09-07T19:35:07Z"
    combined_inventory["terminal_refresh_rule"] = (
        "V8 composes exact terminal V7 payloads with merged DATA-BULK-CODE-1 source-intake payloads; "
        "all capacity remains pre-decontamination/pre-pack and grants zero training exposure."
    )
    combined_payloads = dict(payloads)
    combined_payloads.update(bulk_payloads)

    dedup = v7.v6.v3.audit_payloads(combined_inventory, combined_payloads)
    v7.v6.v3.verify_report(dedup)
    _require(dedup.get("source_count") == EXPECTED_COMPOSED_OBJECTS, "V8 source count drift")
    _require(_family_counts(dedup) == EXPECTED_FAMILIES, f"V8 family vector drift: {_family_counts(dedup)}")
    terminal = dedup.get("terminal_candidates", {})
    _require(
        terminal.get("declared_capacity_bytes_before") == EXPECTED_COMPOSED_BYTES,
        "V8 composed pre-dedup capacity drift",
    )
    post = int(terminal.get("conservative_unique_capacity_bytes_after", -1))
    _require(0 < post <= EXPECTED_COMPOSED_BYTES, "V8 post-dedup capacity invalid")

    code_terminal = terminal.get("by_modality", {}).get("code", {})
    expected_code_before = 276_466 + EXPECTED_BULK_BYTES
    _require(
        code_terminal.get("declared_capacity_bytes_before") == expected_code_before,
        "V8 code pre-dedup capacity drift",
    )
    _require(
        terminal.get("by_modality", {}).get("uk", {}).get("declared_capacity_bytes_before") == 100_856,
        "V8 UK capacity drift",
    )
    _require(
        terminal.get("by_modality", {}).get("en", {}).get("declared_capacity_bytes_before") == 1_838_293,
        "V8 EN capacity drift",
    )

    core: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "worker_id": WORKER_ID,
        "execution_profile": "LOCAL_FREE",
        "base_main_sha": EXPECTED_MAIN,
        "baseline_v7": {
            "head_sha": EXPECTED_V7_HEAD,
            "report_sha256": baseline_report["report_sha256"],
            "dedup_v3_report_sha256": baseline_report["dedup_v3"]["report_sha256"],
            "source_object_count": EXPECTED_BASE_OBJECTS,
            "unique_capacity_bytes": EXPECTED_BASE_BYTES,
        },
        "data_bulk_code1": {
            "report_identity_sha256": bulk_report["report_identity_sha256"],
            "eligible_file_count": bulk_report["eligible_file_count"],
            "eligible_utf8_bytes": bulk_report["eligible_utf8_bytes"],
            "source_family_count": bulk_report["source_family_count"],
            "credential_scan": bulk_report["security"]["credential_scan"],
        },
        "source_vector": {
            "source_object_count": dedup["source_count"],
            "source_family_counts": _family_counts(dedup),
            "source_capacity_bytes_before_global_dedup": terminal["declared_capacity_bytes_before"],
            "conservative_unique_capacity_bytes_after_global_dedup": post,
            "duplicate_discount_bytes": terminal["duplicate_discount_bytes"],
            "duplicate_cluster_count": terminal["duplicate_cluster_count"],
            "effective_independent_origin_count": terminal["effective_independent_origin_count"],
            "by_modality": {
                modality: {
                    "source_count": terminal["by_modality"][modality]["source_count"],
                    "source_family_count": terminal["by_modality"][modality]["declared_source_family_count"],
                    "capacity_bytes_before_global_dedup": terminal["by_modality"][modality]["declared_capacity_bytes_before"],
                    "conservative_unique_capacity_bytes_after_global_dedup": terminal["by_modality"][modality]["conservative_unique_capacity_bytes_after"],
                    "duplicate_discount_bytes": terminal["by_modality"][modality]["duplicate_discount_bytes"],
                }
                for modality in ("uk", "en", "code")
            },
        },
        "dedup_v3": dedup,
        "raw_text_emitted": False,
        "claim_boundary": copy.deepcopy(config["claim_boundary"]),
        "remaining_blockers": [
            "reserved_evaluation_decontamination_must_be_rerun_for_the_new_229_objects",
            "post_composition_quality_privacy_revalidation",
            "balance_and_family_caps",
            "cluster_safe_split_and_deterministic_sharding",
            "two_clean_corpus_builds",
            "deterministic_tokenizer_and_packing_authority",
            "post_pack_unique_loss_ledger",
        ],
    }
    core["report_sha256"] = _self_hash(core)
    return core


def verify_report(config: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    _validate_config(config)
    _require(report.get("schema_version") == REPORT_SCHEMA, "V8 report schema drift")
    _require(report.get("worker_id") == WORKER_ID, "V8 report worker drift")
    _require(report.get("execution_profile") == "LOCAL_FREE", "V8 report execution profile drift")
    _require(report.get("base_main_sha") == EXPECTED_MAIN, "V8 report main binding drift")
    _require(report.get("report_sha256") == _self_hash(report), "V8 report self-hash mismatch")
    _require(report.get("raw_text_emitted") is False, "V8 durable report must be text-free")
    base = report.get("baseline_v7", {})
    _require(base.get("report_sha256") == EXPECTED_V7_REPORT, "V8 report V7 identity drift")
    _require(base.get("dedup_v3_report_sha256") == EXPECTED_V7_DEDUP, "V8 report V7 dedup drift")
    bulk = report.get("data_bulk_code1", {})
    _require(bulk.get("report_identity_sha256") == EXPECTED_BULK_REPORT, "V8 report bulk identity drift")
    _require(bulk.get("eligible_file_count") == EXPECTED_BULK_FILES, "V8 report bulk file drift")
    _require(bulk.get("eligible_utf8_bytes") == EXPECTED_BULK_BYTES, "V8 report bulk byte drift")
    _require(bulk.get("credential_scan") == "PASS_NO_HIGH_CONFIDENCE_HITS", "V8 report credential scan drift")
    vector = report.get("source_vector", {})
    _require(vector.get("source_object_count") == EXPECTED_COMPOSED_OBJECTS, "V8 report object drift")
    _require(vector.get("source_family_counts") == EXPECTED_FAMILIES, "V8 report family drift")
    _require(
        vector.get("source_capacity_bytes_before_global_dedup") == EXPECTED_COMPOSED_BYTES,
        "V8 report pre-dedup capacity drift",
    )
    post = vector.get("conservative_unique_capacity_bytes_after_global_dedup")
    _require(isinstance(post, int) and 0 < post <= EXPECTED_COMPOSED_BYTES, "V8 report post-dedup capacity invalid")
    boundary = report.get("claim_boundary", {})
    _require(boundary == config["claim_boundary"], "V8 report claim boundary drift")
    _require(boundary.get("authorized_training_exposure") == 0, "V8 report fabricated training exposure")
    _require(boundary.get("tokenizer_fit_authorized") is False, "V8 report fabricated tokenizer authority")
    _require(boundary.get("model_training_executed") is False, "V8 report fabricated training execution")
    _require(boundary.get("final_test_payload_read") is False, "V8 report final-test boundary weakened")
    _require(boundary.get("paid_compute_used") is False, "V8 report paid-compute boundary weakened")


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(report) + b"\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check-config", "run", "verify"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/next100_065f_global_dedup_v8.json"),
    )
    parser.add_argument("--current-root", type=Path, default=Path("."))
    parser.add_argument("--v7-root", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.command == "check-config":
        _validate_bulk_terminal_evidence(args.current_root, config)
        print("PASS_CONFIG_AND_TERMINAL_BULK_BINDING")
        return 0
    if args.command == "run":
        _require(args.v7_root is not None, "--v7-root is required")
        _require(args.workspace is not None, "--workspace is required")
        _require(args.report is not None, "--report is required")
        args.workspace.mkdir(parents=True, exist_ok=False)
        report = run_audit(args.current_root, args.v7_root, args.workspace, config)
        verify_report(config, report)
        _write_report(args.report, report)
        print(f"report_sha256={report['report_sha256']}")
        print(
            "post_global_dedup_unique_bytes="
            f"{report['source_vector']['conservative_unique_capacity_bytes_after_global_dedup']}"
        )
        print(f"duplicate_discount_bytes={report['source_vector']['duplicate_discount_bytes']}")
        return 0
    _require(args.report is not None, "--report is required")
    report = _read_json(args.report)
    verify_report(config, report)
    print("PASS_V8_REPORT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())