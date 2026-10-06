"""Execute exact current-Rada post-dedup survivors through current DATA-232.

Execution-only carrier. It reconstructs the exact #2818 source graph and comparison
payloads, binds a text-free current-Rada inventory to the full survivor projection,
and invokes the incumbent reserved-evaluation DATA-232 adapter. It grants no corpus,
tokenizer, training, final-test-outcome, paid-compute, or scale authority.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "tools", ROOT / "src"):
    value = str(location)
    if value not in sys.path:
        sys.path.insert(0, value)

import run_d03_rada_current_global_dedup_v1 as parent

PARENT_EXECUTION_HEAD = "a4663e87b010b190343caf1d42784f5dc7984601"
PARENT_RUNNER_BLOB = "1f7109ae2efca9a97ea49ab5c29b8f095657489c"
CURRENT_RESERVED_BLOB = "e5c555e3cd27844e98d4ae91af0b746e427f36c9"
CURRENT_MATCHER_MODULE = "twelve_six.data._data232_decontamination_matching"
CURRENT_AUTHORITY_MODULE = "twelve_six.data.decontamination_authority_v2"
CURRENT_RESERVED_MODULE = "twelve_six.data.current_reserved_decontamination_v1"
CURRENT_MATCHER_BLOB = "afa70511f82dc81d9c9f85e3d0b67eba343004f9"
CURRENT_AUTHORITY_BLOB = "3ca8f21945c02f692c130a015e036673fa24e7af"
EVAL303_RESOLVER_BLOB = "659cb17fdba903d9c50c1dfc5becf046f523e5ee"
EVAL233_RESOLVER_BLOB = "71dd98204c588bd0bb67b7a7b3ddc5ae8aa87c00"
EVAL647_FUTURE_BLOB = "5516577a0720150a7ec12c1bf8898972968e6970"
EVAL647_RESERVED_BLOB = "ce33771c9fb4a6cc421e2f8e1f6f232c119bec71"

PARENT_ARTIFACT_ID = 11389910019
PARENT_ARTIFACT_ZIP_SHA256 = "63f9e1bf5713989a155429c8862196cacaff3207a7e0fe97586f7dd2c98881f9"
PARENT_REPORT_FILE_SHA256 = "9a38961a5dc5bc7ef93ba046fdfb4f22ccddc175b8b028c875316b3f80324c1f"
PARENT_RADA_AUTHORITY_FILE_SHA256 = "ebdb68e689625e2f46e8a44a02b9d38ffafd9e79a04dbdc49bf6f12b5dcfb6c9"
PARENT_EVIDENCE_FILE_SHA256 = "43a9621ab11fd82232e8251ab26aae4126cf0b4854878af1e97184d587740caa"
PARENT_TWO_CLEAN_FILE_SHA256 = "748adad71a730f7daf18fa52c9e78e3c249ae683e6f3d9cb865b1bbfc9365aa9"

EXPECTED_REPORT_SHA256 = "64e687ae431804862003d5839b9a90c715838e794daad907200f0abfc73b4333"
EXPECTED_FULL_SELECTION_SHA256 = "601398c39769dabb930997f25506eaaf71bd8960aa82d8af9c697d1cc4075e22"
EXPECTED_RADA_SLICE_SHA256 = "f103a3f18216519bd9228e586bd673d49f73cace03b3b0278f2dd0a33383bffb"
EXPECTED_PARENT_EVIDENCE_SHA256 = "a3f7e396cb13a6b107aaa0eb330edcbdc61bead6fa12eb68542ff5c3a3ed9301"
EXPECTED_PARENT_TWO_CLEAN_SHA256 = "f24f4b2d23bee6cb1273a4680297d942aa59030dab50d5c5de7a4d659ff99a5e"

EXPECTED_COMBINED_OBJECTS = 101_995
EXPECTED_COMBINED_DECLARED_BYTES = 198_398_557
EXPECTED_SURVIVORS = 98_861
EXPECTED_SURVIVOR_DECLARED_BYTES = 192_861_011
EXPECTED_RADA_SURVIVORS = 98_601
EXPECTED_RADA_SURVIVOR_DECLARED_BYTES = 186_855_914

INVENTORY_SCHEMA = "12-6.d03-rada-current-postdedup-inventory.v1"
PREPARE_SCHEMA = "12-6.d03-rada-current-data232-prepare.v1"
RESULT_SCHEMA = "12-6.d03-rada-current-data232-execution-result.v1"

RETAINED_ARTIFACT_SOURCE_SHA256 = "9f937b3283dd2d9508a0a92442fc007e7e6d373a17d3d50fa428e324cb2fa290"
RETAINED_ARTIFACT_SOURCE_BYTES = 46_774_786
PROBE_SHA256 = "d2e1b233124551d91df7e7f9f41f38073843807c121adc73914264b04bef19a0"
QUALIFICATION_SHA256 = "78bcb860c2edaf09a3658524efd39b0ce5487508fe16656dfb96bcae11ae6341"
PIN_SHA256 = "3eb281bc0f00a91bee523183017bda2a308f87358eec9ade15419aaf8202ea2a"
PROBE_ENTRY_ID = "1fcc222a959d1dfc24e2b23b71a5412b1050e22a5004cbd36f0dc79998898cc0"
QUALIFICATION_ID = "ced7b9370925ba0aa17952efd2cc5ff917601d44fa301b8449ba135644e5a8b3"
PIN_ID = "dcda0321145c03160cef435bc3d1ef5ac668c3415bc013750ed38cd2d891561e"

EXPECTED_CURRENT_JSONL_SHA256 = parent.EXPECTED_CURRENT_JSONL_SHA256
EXPECTED_CURRENT_JSONL_BYTES = parent.EXPECTED_CURRENT_JSONL_BYTES


class CurrentRadaData232Error(RuntimeError):
    """Fail-closed execution carrier error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CurrentRadaData232Error(message)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def self_hashed(core: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = copy.deepcopy(dict(core))
    value[field] = sha256(canonical(core))
    return value


def verify_self_hash(
    value: Mapping[str, Any],
    field: str,
    expected: str | None = None,
    *,
    label: str,
) -> str:
    require(type(value) is dict, f"{label}: root must be exact object")
    claimed = value.get(field)
    require(
        type(claimed) is str
        and len(claimed) == 64
        and all(ch in "0123456789abcdef" for ch in claimed),
        f"{label}: invalid {field}",
    )
    core = copy.deepcopy(dict(value))
    core.pop(field, None)
    require(sha256(canonical(core)) == claimed, f"{label}: self-hash mismatch")
    if expected is not None:
        require(claimed == expected, f"{label}: identity drift")
    return claimed


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CurrentRadaData232Error(f"cannot load JSON: {path}") from exc
    require(type(value) is dict, f"JSON root must be object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                require(type(row) is dict, f"{path}:{line_number}: row must be object")
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CurrentRadaData232Error(f"cannot load JSONL: {path}") from exc
    require(bool(rows), f"JSONL is empty: {path}")
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(dict(value)) + b"\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical(dict(row)) + b"\n" for row in rows))


def git(*args: str, cwd: Path = ROOT) -> str:
    proc = subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or str(proc.returncode)
        raise CurrentRadaData232Error(f"git {' '.join(args)} failed: {detail}")
    return proc.stdout.strip()


def bind_execution_head(expected_execution_head: str) -> None:
    require(git("rev-parse", "HEAD") == expected_execution_head, "execution HEAD drift")
    require(
        git("merge-base", PARENT_EXECUTION_HEAD, "HEAD") == PARENT_EXECUTION_HEAD,
        "exact #2818 head is not ancestor",
    )
    expected_blobs = {
        "tools/run_d03_rada_current_global_dedup_v1.py": PARENT_RUNNER_BLOB,
        "src/twelve_six/data/current_reserved_decontamination_v1.py": CURRENT_RESERVED_BLOB,
        "src/twelve_six/data/eval303_selection_payload_resolver_v1.py": EVAL303_RESOLVER_BLOB,
        "src/twelve_six/data/eval233_final_test_resolver_v1.py": EVAL233_RESOLVER_BLOB,
        "src/twelve_six/data/eval647_future_training_exclusion_v1.py": EVAL647_FUTURE_BLOB,
        "src/twelve_six/data/eval647_reserved_decontamination_v1.py": EVAL647_RESERVED_BLOB,
    }
    for path, expected in expected_blobs.items():
        require(
            git("rev-parse", f"{PARENT_EXECUTION_HEAD}:{path}") == expected,
            f"parent blob drift: {path}",
        )
        require(
            git("rev-parse", f"HEAD:{path}") == expected,
            f"child mutated incumbent authority: {path}",
        )


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reset_current_data232_namespace(v7_root: Path) -> None:
    """Drop historical DATA-232 imports before activating the current matcher."""
    import twelve_six.data as data_pkg

    historical_data = (v7_root / "src" / "twelve_six" / "data").resolve(strict=True)
    current_data = (ROOT / "src" / "twelve_six" / "data").resolve(strict=True)
    package_paths = [Path(value).resolve(strict=True) for value in data_pkg.__path__]
    require(historical_data not in package_paths, "historical V7 data package path leaked")
    require(current_data in package_paths, "current DATA-232 package path missing")

    for module_name in (
        CURRENT_RESERVED_MODULE,
        CURRENT_AUTHORITY_MODULE,
        CURRENT_MATCHER_MODULE,
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        raw_path = getattr(module, "__file__", None)
        require(type(raw_path) is str and bool(raw_path), f"{module_name}: module path missing")
        module_path = Path(raw_path).resolve(strict=True)
        require(
            module_path.is_relative_to(current_data)
            or module_path.is_relative_to(historical_data),
            f"{module_name}: unexpected module root",
        )
        sys.modules.pop(module_name, None)
        attribute = module_name.rsplit(".", 1)[1]
        if getattr(data_pkg, attribute, None) is module:
            delattr(data_pkg, attribute)
    importlib.invalidate_caches()


def _verify_current_data232_namespace() -> None:
    expected = {
        CURRENT_MATCHER_MODULE: (
            ROOT / "src/twelve_six/data/_data232_decontamination_matching.py",
            CURRENT_MATCHER_BLOB,
        ),
        CURRENT_AUTHORITY_MODULE: (
            ROOT / "src/twelve_six/data/decontamination_authority_v2.py",
            CURRENT_AUTHORITY_BLOB,
        ),
        CURRENT_RESERVED_MODULE: (
            ROOT / "src/twelve_six/data/current_reserved_decontamination_v1.py",
            CURRENT_RESERVED_BLOB,
        ),
    }
    for module_name, (expected_path, expected_blob) in expected.items():
        module = sys.modules.get(module_name)
        require(module is not None, f"{module_name}: current module not loaded")
        raw_path = getattr(module, "__file__", None)
        require(type(raw_path) is str and bool(raw_path), f"{module_name}: current path missing")
        require(
            Path(raw_path).resolve(strict=True) == expected_path.resolve(strict=True),
            f"{module_name}: current module provenance drift",
        )
        require(
            git("hash-object", str(expected_path)) == expected_blob,
            f"{module_name}: current module blob drift",
        )


def replay_current_rada(args: argparse.Namespace) -> None:
    """Recreate the exact Q/P-accepted current-Rada candidate from retained bytes."""
    bind_execution_head(args.expected_execution_head)
    norm = _load_module(
        ROOT / "tools/normalize_d03_rada_bulk_html.py",
        "rada_norm_g6141",
    )
    qp = _load_module(
        args.historical_qp_root / "tools/filter_d03_rada_bulk_quality_privacy.py",
        "rada_qp_g6141",
    )
    norm_cfg = load_json(ROOT / "configs/data/d03_rada_bulk_normalization_v1.json")
    qp_cfg = load_json(
        args.historical_qp_root / "configs/data/d03_rada_bulk_quality_privacy_v1.json"
    )

    source = args.source_zip.read_bytes()
    require(len(source) == RETAINED_ARTIFACT_SOURCE_BYTES, "Rada source archive byte drift")
    require(sha256(source) == RETAINED_ARTIFACT_SOURCE_SHA256, "Rada source archive SHA drift")

    probe_a_raw = args.probe_a.read_bytes()
    probe_b_raw = args.probe_b.read_bytes()
    require(sha256(probe_a_raw) == PROBE_SHA256, "probe A identity drift")
    require(sha256(probe_b_raw) == PROBE_SHA256, "probe B identity drift")
    probe_a = json.loads(probe_a_raw.decode("utf-8"))
    probe_b = json.loads(probe_b_raw.decode("utf-8"))
    require(type(probe_a) is dict and probe_a == probe_b, "retained clean probes diverged")

    qual_raw = args.qualification.read_bytes()
    pin_raw = args.pin.read_bytes()
    require(sha256(qual_raw) == QUALIFICATION_SHA256, "qualification transport drift")
    require(sha256(pin_raw) == PIN_SHA256, "pin transport drift")
    qualification = json.loads(qual_raw.decode("utf-8"))
    pin = json.loads(pin_raw.decode("utf-8"))
    require(
        qualification.get("evidence_identity_sha256") == QUALIFICATION_ID,
        "qualification identity drift",
    )
    require(pin.get("pin_identity_sha256") == PIN_ID, "successor pin identity drift")
    require(
        qualification.get("normalization_reexecution_required") is True
        and qualification.get("quality_privacy_reexecution_required") is True,
        "physical replay boundary drift",
    )
    require(pin.get("purpose") == "NORMALIZATION_INPUT_ONLY", "successor pin purpose widened")
    inventory = probe_a.get("inventory")
    require(type(inventory) is dict, "probe inventory missing")
    require(inventory.get("entry_identity_sha256") == PROBE_ENTRY_ID, "probe entry identity drift")
    require(inventory.get("canonical_entry_count") == 3055, "probe entry count drift")
    require(inventory.get("canonical_raw_bytes") == 353_891_824, "probe raw bytes drift")

    projected_probe = copy.deepcopy(probe_a)
    projected_probe["safe_result"] = norm.PINNED_PROBE_RESULT
    projected_probe["gates"]["exact_archive_identity"] = norm.PINNED_PROBE_GATE
    projected_probe_sha = sha256(norm._serialized_probe_report_bytes(projected_probe))
    normalized_jsonl, manifest = norm._materialize_normalized_records_unbound(
        source,
        projected_probe,
        norm_cfg,
        probe_report_sha256=projected_probe_sha,
    )
    require(manifest.get("safe_result") == norm.UNBOUND_SAFE_RESULT, "normalization widened authority")

    projected_manifest = copy.deepcopy(manifest)
    projected_manifest["safe_result"] = qp.PARENT_SAFE_RESULT
    projected_manifest.pop("manifest_identity_sha256", None)
    projected_manifest["manifest_identity_sha256"] = sha256(qp._canonical_bytes(projected_manifest))
    norm_data = projected_manifest["normalization"]
    parent_probe = projected_manifest["parent_probe"]
    binding = {
        "pr": 2478,
        "head_sha": "9fa66e11faf3710d6ddf75854381eb879a37d1ba",
        "branch": "swarm/2477-rada-laws-fresh-snapshot-v2",
        "execution_head_sha": "9fa66e11faf3710d6ddf75854381eb879a37d1ba",
        "execution_run_id": 37136885886,
        "execution_evidence_identity_sha256": QUALIFICATION_ID,
        "manifest_schema": qp.PARENT_MANIFEST_SCHEMA,
        "manifest_worker_id": qp.PARENT_WORKER_ID,
        "source_family": qp.SOURCE_FAMILY,
        "safe_result": qp.PARENT_SAFE_RESULT,
        "manifest_identity_sha256": projected_manifest["manifest_identity_sha256"],
        "manifest_transport_sha256": sha256(qp._canonical_bytes(projected_manifest)),
        "jsonl_sha256": norm_data["jsonl_sha256"],
        "record_count": norm_data["record_count"],
        "nonempty_record_count": norm_data["nonempty_record_count"],
        "normalized_bytes_observed_not_credited": norm_data["normalized_bytes_observed_not_credited"],
        "normalized_record_inventory_sha256": norm_data["normalized_record_inventory_sha256"],
        "source_encoding_counts": norm_data["source_encoding_counts"],
        "pinned_probe_report_sha256": parent_probe["probe_report_sha256"],
        "archive_sha256": parent_probe["archive_sha256"],
        "entry_identity_sha256": parent_probe["entry_identity_sha256"],
    }
    projected_qp_cfg = copy.deepcopy(qp_cfg)
    projected_qp_cfg["parent_normalization"] = binding
    qp.EXPECTED_PARENT_BINDING = binding
    accepted_jsonl, qp_report = qp._materialize_quality_privacy_candidate_for_test(
        normalized_jsonl,
        projected_manifest,
        projected_qp_cfg,
        parent_manifest_sha256=binding["manifest_transport_sha256"],
    )
    require(len(accepted_jsonl) == EXPECTED_CURRENT_JSONL_BYTES, "current-Rada JSONL byte drift")
    require(sha256(accepted_jsonl) == EXPECTED_CURRENT_JSONL_SHA256, "current-Rada JSONL identity drift")
    args.output_candidate.parent.mkdir(parents=True, exist_ok=True)
    args.output_candidate.write_bytes(accepted_jsonl)

    summary_core = {
        "schema_version": "12-6.d03-rada-current-mechanics-replay-g6141.v1",
        "source_archive_sha256": RETAINED_ARTIFACT_SOURCE_SHA256,
        "candidate_jsonl_sha256": EXPECTED_CURRENT_JSONL_SHA256,
        "candidate_jsonl_file_bytes": EXPECTED_CURRENT_JSONL_BYTES,
        "normalized_record_count": norm_data["record_count"],
        "normalized_payload_bytes_observed_not_credited": norm_data[
            "normalized_bytes_observed_not_credited"
        ],
        "accepted_chunk_count": qp_report["filter_result"]["accepted_chunk_count"],
        "accepted_payload_bytes_observed_not_credited": qp_report["filter_result"][
            "accepted_bytes_observed_not_credited"
        ],
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }
    write_json(args.output_summary, self_hashed(summary_core, "replay_identity_sha256"))


def verify_parent_files(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    report_raw = args.parent_report.read_bytes()
    rada_raw = args.parent_rada_authority.read_bytes()
    evidence_raw = args.parent_evidence.read_bytes()
    two_clean_raw = args.parent_two_clean.read_bytes()
    require(sha256(report_raw) == PARENT_REPORT_FILE_SHA256, "parent report transport drift")
    require(
        sha256(rada_raw) == PARENT_RADA_AUTHORITY_FILE_SHA256,
        "parent Rada authority transport drift",
    )
    require(
        sha256(evidence_raw) == PARENT_EVIDENCE_FILE_SHA256,
        "parent execution evidence transport drift",
    )
    require(
        sha256(two_clean_raw) == PARENT_TWO_CLEAN_FILE_SHA256,
        "parent two-clean transport drift",
    )
    report = json.loads(report_raw.decode("utf-8"))
    rada_authority = json.loads(rada_raw.decode("utf-8"))
    evidence = json.loads(evidence_raw.decode("utf-8"))
    two_clean = json.loads(two_clean_raw.decode("utf-8"))
    require(all(type(v) is dict for v in (report, rada_authority, evidence, two_clean)), "parent JSON root drift")
    require(report.get("report_sha256") == EXPECTED_REPORT_SHA256, "parent report identity drift")
    verify_self_hash(
        rada_authority,
        "survivor_authority_sha256",
        EXPECTED_RADA_SLICE_SHA256,
        label="parent Rada survivor authority",
    )
    verify_self_hash(
        evidence,
        "evidence_identity_sha256",
        EXPECTED_PARENT_EVIDENCE_SHA256,
        label="parent execution evidence",
    )
    verify_self_hash(
        two_clean,
        "two_clean_identity_sha256",
        EXPECTED_PARENT_TWO_CLEAN_SHA256,
        label="parent two-clean evidence",
    )
    require(
        evidence.get("matcher_report_sha256") == EXPECTED_REPORT_SHA256
        and evidence.get("survivor_authority_sha256") == EXPECTED_RADA_SLICE_SHA256,
        "parent evidence cross-binding drift",
    )
    require(
        two_clean.get("matcher_report_sha256") == EXPECTED_REPORT_SHA256
        and two_clean.get("survivor_authority_sha256") == EXPECTED_RADA_SLICE_SHA256,
        "parent two-clean cross-binding drift",
    )
    return report, rada_authority


def prepare(args: argparse.Namespace) -> None:
    bind_execution_head(args.expected_execution_head)
    parent.verify_product_parent(args.expected_execution_head)
    report, rada_authority = verify_parent_files(args)

    helper = parent.load_helper(args.nbu_helper_root)
    require(
        git("rev-parse", "HEAD", cwd=args.v7_root) == parent.V7_HEAD,
        "historical V7 HEAD drift",
    )
    current_sources, current_payloads, projection_receipt = parent.validate_current_projection(
        args.candidate_jsonl
    )
    config = parent.load_json(
        args.nbu_helper_root / "configs/data/next100_065f_global_dedup_v8.json"
    )
    matcher, base_inventory, base_payloads, removal = parent.reconstruct_with_bounded_transport_retry(
        helper,
        v7_root=args.v7_root,
        bulk_workspace=args.bulk_workspace,
        config=config,
    )
    helper._verify_removal_proof(removal)
    inventory, payloads, replacement_proof = parent.compose_current_graph(
        helper,
        base_inventory,
        base_payloads,
        current_sources,
        current_payloads,
    )
    require(len(payloads) == EXPECTED_COMBINED_OBJECTS, "combined payload count drift")
    require(
        parent.declared_capacity(inventory, payloads, label="current graph")
        == EXPECTED_COMBINED_DECLARED_BYTES,
        "combined declared capacity drift",
    )

    matcher.verify_report(report)
    by_id = parent.verify_report_current_rada(report, current_sources, current_payloads)
    require(len(by_id) == EXPECTED_COMBINED_OBJECTS, "parent report source coverage drift")
    computed_rada_authority = parent.survivor_authority(
        helper,
        report,
        by_id,
        set(current_payloads),
        replacement_proof,
    )
    require(
        canonical(computed_rada_authority) == canonical(rada_authority),
        "reconstructed current-Rada survivor authority differs from #2818 artifact",
    )

    selection = helper.v9_semantics._derive_survivors(report)
    selection_id = verify_self_hash(
        selection,
        "survivor_authority_sha256",
        EXPECTED_FULL_SELECTION_SHA256,
        label="full survivor selection",
    )
    require(
        selection.get("post_dedup_survivor_source_object_count") == EXPECTED_SURVIVORS,
        "full survivor count drift",
    )
    require(
        selection.get("post_dedup_declared_capacity_bytes") == EXPECTED_SURVIVOR_DECLARED_BYTES,
        "full survivor declared bytes drift",
    )
    require(
        rada_authority.get("selection_projection_sha256") == selection_id,
        "Rada slice does not bind full selection projection",
    )
    require(
        rada_authority.get("current_rada_survivor_source_object_count") == EXPECTED_RADA_SURVIVORS,
        "Rada survivor count drift",
    )
    require(
        rada_authority.get("current_rada_survivor_declared_capacity_bytes")
        == EXPECTED_RADA_SURVIVOR_DECLARED_BYTES,
        "Rada survivor declared bytes drift",
    )

    rows = inventory.get("sources")
    require(type(rows) is list and len(rows) == EXPECTED_COMBINED_OBJECTS, "graph source vector drift")
    graph_by_id = {
        row.get("source_id"): row
        for row in rows
        if type(row) is dict and type(row.get("source_id")) is str
    }
    require(len(graph_by_id) == len(rows), "graph source IDs invalid")
    require(set(graph_by_id) == set(payloads) == set(by_id), "graph/report/payload coverage drift")

    comparison_payloads: dict[str, bytes] = {}
    for source_id, source in graph_by_id.items():
        raw = payloads[source_id]
        observed = by_id[source_id]
        require(observed.get("source_family") == source.get("source_family"), f"{source_id}: source family drift")
        require(observed.get("modality") == source.get("modality"), f"{source_id}: modality drift")
        require(
            observed.get("declared_capacity_bytes") == source.get("declared_capacity_bytes"),
            f"{source_id}: declared capacity drift",
        )
        require(
            observed.get("verified_raw_bytes") == len(raw)
            and observed.get("verified_raw_sha256") == sha256(raw),
            f"{source_id}: raw payload/report drift",
        )
        stable_origin = source.get("stable_origin_id")
        stable_object = source.get("stable_object_id")
        require(type(stable_origin) is str and type(stable_object) is str, f"{source_id}: stable IDs missing")
        require(
            observed.get("stable_origin_id_sha256") == sha256(stable_origin.encode("utf-8"))
            and observed.get("stable_object_id_sha256") == sha256(stable_object.encode("utf-8")),
            f"{source_id}: stable ID binding drift",
        )
        comparison = matcher._comparison_payload(source, raw)
        if comparison is None:
            comparison = raw
            comparison_policy = "DATA232_GENERIC_FROM_RAW"
        else:
            comparison_policy = source.get("comparison_normalization")
        require(type(comparison) is bytes and bool(comparison), f"{source_id}: comparison payload invalid")
        require(
            observed.get("comparison_policy") == comparison_policy
            and observed.get("comparison_payload_bytes") == len(comparison)
            and observed.get("comparison_payload_sha256") == sha256(comparison),
            f"{source_id}: incumbent comparison-payload reproduction drift",
        )
        comparison_payloads[source_id] = comparison

    survivor_ids = selection.get("survivor_source_ids")
    require(
        type(survivor_ids) is list
        and len(survivor_ids) == EXPECTED_SURVIVORS
        and len(set(survivor_ids)) == EXPECTED_SURVIVORS,
        "full survivor IDs drift",
    )
    survivor_set = set(survivor_ids)
    require(survivor_set <= set(graph_by_id), "selection references unknown source")

    retained_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, str]] = []
    retained_declared = 0
    retained_comparison = 0
    for source_id in sorted(survivor_set):
        source = graph_by_id[source_id]
        observed = by_id[source_id]
        comparison = comparison_payloads[source_id]
        try:
            text = comparison.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise CurrentRadaData232Error(
                f"{source_id}: retained comparison payload is not strict UTF-8"
            ) from exc
        family = source.get("source_family")
        modality = source.get("modality")
        declared = source.get("declared_capacity_bytes")
        require(type(family) is str and family, f"{source_id}: source family missing")
        require(type(modality) is str and modality, f"{source_id}: modality missing")
        require(type(declared) is int and declared > 0, f"{source_id}: declared bytes invalid")
        retained_declared += declared
        retained_comparison += len(comparison)
        retained_rows.append(
            {
                "source_id": source_id,
                "source_family": family,
                "modality": modality,
                "declared_capacity_bytes": declared,
                "stable_origin_id_sha256": observed["stable_origin_id_sha256"],
                "stable_object_id_sha256": observed["stable_object_id_sha256"],
                "verified_raw_bytes": observed["verified_raw_bytes"],
                "verified_raw_sha256": observed["verified_raw_sha256"],
                "comparison_policy": observed["comparison_policy"],
                "comparison_payload_bytes": len(comparison),
                "comparison_payload_sha256": sha256(comparison),
            }
        )
        training_rows.append(
            {
                "record_id": source_id,
                "source_id": source_id,
                "source_family": family,
                "modality": modality,
                "text": text,
            }
        )
    require(retained_declared == EXPECTED_SURVIVOR_DECLARED_BYTES, "retained declared bytes drift")

    # Historical graph reconstruction intentionally leaves exact V7 matcher modules
    # cached in sys.modules. Reset only the DATA-232 namespace before current replay
    # so current authority code cannot bind to historical matcher functions.
    _reset_current_data232_namespace(args.v7_root)
    from twelve_six.data import current_reserved_decontamination_v1 as reserved
    _verify_current_data232_namespace()

    inventory_core = {
        "schema_version": INVENTORY_SCHEMA,
        "parent_execution_head_sha": PARENT_EXECUTION_HEAD,
        "parent_artifact_id": PARENT_ARTIFACT_ID,
        "parent_artifact_zip_sha256": PARENT_ARTIFACT_ZIP_SHA256,
        "matcher_report_sha256": EXPECTED_REPORT_SHA256,
        "full_selection_projection_schema": selection.get("schema_version"),
        "full_selection_projection_sha256": selection_id,
        "current_rada_slice_authority_sha256": EXPECTED_RADA_SLICE_SHA256,
        "current_rada_projection_receipt_identity_sha256": projection_receipt.get(
            "receipt_identity_sha256"
        ),
        "retained_source_count": len(retained_rows),
        "retained_declared_capacity_bytes": retained_declared,
        "retained_comparison_payload_bytes": retained_comparison,
        "retained_sources": retained_rows,
        "raw_text_persisted": False,
        "source_admission_authority_granted": False,
        "canonical_capacity_credited": 0,
        "authorized_training_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }
    inventory_evidence = self_hashed(inventory_core, "inventory_identity_sha256")
    inventory_id = inventory_evidence["inventory_identity_sha256"]

    matcher_projection = reserved._record_projection(training_rows)
    handoff_core = {
        "schema_version": reserved.TRAINING_HANDOFF_SCHEMA,
        "postdedup_inventory_identity_sha256": inventory_id,
        "input_survivor_authority_sha256": selection_id,
        "retained_source_count": len(training_rows),
        "matcher_input_projection": matcher_projection,
        "matcher_input_projection_sha256": sha256(canonical(matcher_projection)),
        "raw_text_persisted_in_evidence": False,
        "final_test_payload_accessed": False,
        "final_test_outcomes_accessed": False,
        "authorized_training_exposure": 0,
    }
    handoff = self_hashed(handoff_core, "handoff_identity_sha256")

    prepare_core = {
        "schema_version": PREPARE_SCHEMA,
        "execution_head_sha": args.expected_execution_head,
        "parent_execution_head_sha": PARENT_EXECUTION_HEAD,
        "parent_artifact_id": PARENT_ARTIFACT_ID,
        "parent_artifact_zip_sha256": PARENT_ARTIFACT_ZIP_SHA256,
        "parent_matcher_report_sha256": EXPECTED_REPORT_SHA256,
        "full_selection_projection_sha256": selection_id,
        "current_rada_slice_authority_sha256": EXPECTED_RADA_SLICE_SHA256,
        "postdedup_inventory_identity_sha256": inventory_id,
        "training_handoff_identity_sha256": handoff["handoff_identity_sha256"],
        "retained_source_count": len(training_rows),
        "retained_declared_capacity_bytes": retained_declared,
        "retained_comparison_payload_bytes": retained_comparison,
        "training_records_projection_sha256": sha256(canonical(matcher_projection)),
        "raw_training_text_ephemeral_only": True,
        "durable_evidence_hash_only": True,
        "canonical_capacity_credited": 0,
        "authorized_training_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "next_gate": "CURRENT_RESERVED_EVALUATION_DATA232_EXECUTION",
    }
    prepare_evidence = self_hashed(prepare_core, "prepare_identity_sha256")

    write_json(args.output_inventory, inventory_evidence)
    write_json(args.output_handoff, handoff)
    write_json(args.output_prepare_evidence, prepare_evidence)
    write_jsonl(args.output_training_records, training_rows)


def execute_data232(args: argparse.Namespace) -> None:
    bind_execution_head(args.expected_execution_head)
    # This command runs in its own fresh process and performs no historical graph
    # reconstruction, so loading the current DATA-232 adapter here is intentional.
    from twelve_six.data import current_reserved_decontamination_v1 as reserved
    inventory = load_json(args.inventory_json)
    handoff = load_json(args.handoff_json)
    training = load_jsonl(args.training_records_jsonl)
    evaluation = load_jsonl(args.evaluation_records_jsonl)
    binding = load_json(args.reserved_binding_json)

    inventory_id = verify_self_hash(
        inventory,
        "inventory_identity_sha256",
        args.expected_inventory_identity_sha256,
        label="postdedup inventory",
    )
    require(
        inventory.get("schema_version") == INVENTORY_SCHEMA,
        "postdedup inventory schema drift",
    )
    require(
        inventory.get("full_selection_projection_sha256") == EXPECTED_FULL_SELECTION_SHA256,
        "inventory full selection identity drift",
    )
    require(
        inventory.get("current_rada_slice_authority_sha256") == EXPECTED_RADA_SLICE_SHA256,
        "inventory Rada slice identity drift",
    )
    require(inventory.get("raw_text_persisted") is False, "inventory persisted raw text")
    require(inventory.get("authorized_training_exposure") == 0, "inventory widened training authority")
    require(inventory.get("tokenizer_fit_authorized") is False, "inventory widened tokenizer authority")

    handoff_id = verify_self_hash(
        handoff,
        "handoff_identity_sha256",
        args.expected_handoff_identity_sha256,
        label="training handoff",
    )
    require(
        handoff.get("postdedup_inventory_identity_sha256") == inventory_id,
        "handoff inventory identity drift",
    )
    require(
        handoff.get("input_survivor_authority_sha256") == EXPECTED_FULL_SELECTION_SHA256,
        "handoff survivor authority drift",
    )

    report, evidence = reserved.execute_reserved_decontamination(
        training,
        evaluation,
        training_handoff_evidence=handoff,
        reserved_payload_binding=binding,
        expected_inventory_identity_sha256=inventory_id,
        expected_survivor_authority_sha256=EXPECTED_FULL_SELECTION_SHA256,
        expected_training_handoff_identity_sha256=handoff_id,
        expected_reserved_binding_identity_sha256=args.expected_reserved_binding_identity_sha256,
        expected_selection_validation_identity_sha256=args.expected_selection_validation_identity_sha256,
        expected_final_test_identity_sha256=args.expected_final_test_identity_sha256,
        quarantine_cross_source_families=True,
    )
    reserved.verify_execution_evidence(evidence, report)
    write_json(args.output_report, report)
    write_json(args.output_execution_evidence, evidence)
    require(evidence.get("authorized_training_exposure") == 0, "DATA-232 widened training exposure")
    require(evidence.get("tokenizer_fit_authorized") is False, "DATA-232 widened tokenizer authority")
    require(evidence.get("training_executed") is False, "DATA-232 executed training")
    require(evidence.get("final_test_outcomes_read") is False, "DATA-232 read final-test outcomes")
    require(evidence.get("paid_compute_used") is False, "DATA-232 used paid compute")

    result_core = {
        "schema_version": RESULT_SCHEMA,
        "execution_head_sha": args.expected_execution_head,
        "parent_execution_head_sha": PARENT_EXECUTION_HEAD,
        "postdedup_inventory_identity_sha256": inventory_id,
        "full_selection_projection_sha256": EXPECTED_FULL_SELECTION_SHA256,
        "current_rada_slice_authority_sha256": EXPECTED_RADA_SLICE_SHA256,
        "training_handoff_identity_sha256": handoff_id,
        "reserved_payload_binding_identity_sha256": binding["binding_identity_sha256"],
        "selection_validation_identity_sha256": args.expected_selection_validation_identity_sha256,
        "final_test_identity_sha256": args.expected_final_test_identity_sha256,
        "data232_report_sha256": report["report_sha256"],
        "data232_execution_identity_sha256": evidence["execution_identity_sha256"],
        "status": evidence["status"],
        "counts": evidence["counts"],
        "final_test_payload_accessed_for_decontamination": True,
        "final_test_outcomes_read": False,
        "durable_evidence_hash_only": True,
        "model_architecture_or_hyperparameters_selected": False,
        "canonical_capacity_credited": 0,
        "authorized_optimized_target_exposure": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_training_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "paid_compute_used": False,
        "scale_promotion_authorized": False,
        "next_gate": "CURRENT_RADA_POST_DATA232_QUALITY_PRIVACY",
    }
    write_json(args.output_result, self_hashed(result_core, "result_identity_sha256"))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)

    replay = sub.add_parser("replay-current-rada")
    replay.add_argument("--expected-execution-head", required=True)
    replay.add_argument("--historical-qp-root", type=Path, required=True)
    replay.add_argument("--source-zip", type=Path, required=True)
    replay.add_argument("--probe-a", type=Path, required=True)
    replay.add_argument("--probe-b", type=Path, required=True)
    replay.add_argument("--qualification", type=Path, required=True)
    replay.add_argument("--pin", type=Path, required=True)
    replay.add_argument("--output-candidate", type=Path, required=True)
    replay.add_argument("--output-summary", type=Path, required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--expected-execution-head", required=True)
    prep.add_argument("--candidate-jsonl", type=Path, required=True)
    prep.add_argument("--nbu-helper-root", type=Path, required=True)
    prep.add_argument("--v7-root", type=Path, required=True)
    prep.add_argument("--bulk-workspace", type=Path, required=True)
    prep.add_argument("--parent-report", type=Path, required=True)
    prep.add_argument("--parent-rada-authority", type=Path, required=True)
    prep.add_argument("--parent-evidence", type=Path, required=True)
    prep.add_argument("--parent-two-clean", type=Path, required=True)
    prep.add_argument("--output-inventory", type=Path, required=True)
    prep.add_argument("--output-handoff", type=Path, required=True)
    prep.add_argument("--output-prepare-evidence", type=Path, required=True)
    prep.add_argument("--output-training-records", type=Path, required=True)

    execute = sub.add_parser("execute-data232")
    execute.add_argument("--expected-execution-head", required=True)
    execute.add_argument("--inventory-json", type=Path, required=True)
    execute.add_argument("--handoff-json", type=Path, required=True)
    execute.add_argument("--training-records-jsonl", type=Path, required=True)
    execute.add_argument("--evaluation-records-jsonl", type=Path, required=True)
    execute.add_argument("--reserved-binding-json", type=Path, required=True)
    execute.add_argument("--expected-inventory-identity-sha256", required=True)
    execute.add_argument("--expected-handoff-identity-sha256", required=True)
    execute.add_argument("--expected-reserved-binding-identity-sha256", required=True)
    execute.add_argument("--expected-selection-validation-identity-sha256", required=True)
    execute.add_argument("--expected-final-test-identity-sha256", required=True)
    execute.add_argument("--output-report", type=Path, required=True)
    execute.add_argument("--output-execution-evidence", type=Path, required=True)
    execute.add_argument("--output-result", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "replay-current-rada":
        replay_current_rada(args)
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "execute-data232":
        execute_data232(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
