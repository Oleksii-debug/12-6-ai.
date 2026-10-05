#!/usr/bin/env python3
"""Reproduce the exact current Rada normalize/QP candidate from retained bytes.

This is an execution-only materializer. It creates an ephemeral accepted JSONL
for the global-dedup carrier and a text-free replay summary. It grants no
production Q/P, corpus, tokenizer-fit, training, or capacity authority.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path
from typing import Any


SOURCE_ARCHIVE_BYTES = 46_774_786
SOURCE_ARCHIVE_SHA256 = (
    "9f937b3283dd2d9508a0a92442fc007e7e6d373a17d3d50fa428e324cb2fa290"
)
PROBE_TRANSPORT_SHA256 = (
    "d2e1b233124551d91df7e7f9f41f38073843807c121adc73914264b04bef19a0"
)
QUALIFICATION_TRANSPORT_SHA256 = (
    "78bcb860c2edaf09a3658524efd39b0ce5487508fe16656dfb96bcae11ae6341"
)
PIN_TRANSPORT_SHA256 = (
    "3eb281bc0f00a91bee523183017bda2a308f87358eec9ade15419aaf8202ea2a"
)
ENTRY_IDENTITY_SHA256 = (
    "1fcc222a959d1dfc24e2b23b71a5412b1050e22a5004cbd36f0dc79998898cc0"
)
QUALIFICATION_IDENTITY_SHA256 = (
    "ced7b9370925ba0aa17952efd2cc5ff917601d44fa301b8449ba135644e5a8b3"
)
PIN_IDENTITY_SHA256 = (
    "dcda0321145c03160cef435bc3d1ef5ac668c3415bc013750ed38cd2d891561e"
)
NORMALIZER_GIT_BLOB_SHA1 = "55ef5f5e7f09042489fa2642d099aedffaddb740"
NORMALIZATION_CONFIG_GIT_BLOB_SHA1 = (
    "3e0ca093762242f18fadb21002946b4b733afea3"
)
QP_TOOL_GIT_BLOB_SHA1 = "69491874e60a1b846d683fff3c10214a869f18ff"
QP_CONFIG_GIT_BLOB_SHA1 = "e1798398a87893c5561afc6d9cd4f3d8fdce6d8b"
EXPECTED_NORMALIZED_RECORDS = 3_055
EXPECTED_NORMALIZED_PAYLOAD_BYTES = 211_492_947
EXPECTED_NORMALIZED_JSONL_BYTES = 214_699_202
EXPECTED_NORMALIZED_JSONL_SHA256 = (
    "da13b3cc036b19e44e212f3816a66ed261a743c41ed8545076c35eeb706ad511"
)
EXPECTED_ACCEPTED_CHUNKS = 101_733
EXPECTED_ACCEPTED_PAYLOAD_BYTES = 192_393_157
EXPECTED_ACCEPTED_JSONL_BYTES = 224_897_989
EXPECTED_ACCEPTED_JSONL_SHA256 = (
    "8d1343708b3ce1747d32c3b551d6fb8ab7123be5b37f74fff3c457b6439159a7"
)
EXPECTED_ACCEPTED_INVENTORY_SHA256 = (
    "468a3132819527e1ce2b3aa375a55a15d55b7af53e66bf06ae1e5a69daf73e7a"
)
EXPECTED_EXACT_DUPLICATE_HASHES = 765


class RadaCurrentReplayMaterializationError(RuntimeError):
    """Raised when exact current Rada replay materialization fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RadaCurrentReplayMaterializationError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    prefix = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(prefix + raw).hexdigest()  # noqa: S324 - Git identity


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise RadaCurrentReplayMaterializationError(
            f"cannot stat {label}: {path}"
        ) from exc
    _require(stat.S_ISREG(metadata.st_mode), f"{label} is not regular file: {path}")
    try:
        with path.open("rb") as handle:
            descriptor = os.fstat(handle.fileno())
            _require(
                descriptor.st_dev == metadata.st_dev
                and descriptor.st_ino == metadata.st_ino,
                f"{label} path changed before descriptor lock",
            )
            raw = handle.read()
            final_descriptor = os.fstat(handle.fileno())
    except OSError as exc:
        raise RadaCurrentReplayMaterializationError(
            f"cannot read {label}: {path}"
        ) from exc
    _require(
        descriptor.st_dev == final_descriptor.st_dev
        and descriptor.st_ino == final_descriptor.st_ino,
        f"{label} descriptor identity changed",
    )
    return raw


def _load_json_with_transport(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
) -> dict[str, Any]:
    raw = _read_regular_bytes(path, label=label)
    _require(_sha256(raw) == expected_sha256, f"{label} transport identity drift")
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RadaCurrentReplayMaterializationError(
            f"{label} is not UTF-8 JSON"
        ) from exc
    _require(type(value) is dict, f"{label} root must be exact object")
    return value


def _load_module(path: Path, name: str, *, expected_blob: str) -> Any:
    raw = _read_regular_bytes(path, label=f"module {name}")
    _require(
        _git_blob_sha1(raw) == expected_blob,
        f"module Git blob drift: {name}",
    )
    spec = importlib.util.spec_from_file_location(name, path)
    _require(
        spec is not None and spec.loader is not None,
        f"cannot load exact module: {name}",
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RadaCurrentReplayMaterializationError(
            f"cannot execute exact module: {name}: {exc}"
        ) from exc
    return module


def _load_config(path: Path, *, expected_blob: str, label: str) -> dict[str, Any]:
    raw = _read_regular_bytes(path, label=label)
    _require(
        _git_blob_sha1(raw) == expected_blob,
        f"{label} Git blob drift",
    )
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RadaCurrentReplayMaterializationError(
            f"{label} is not UTF-8 JSON"
        ) from exc
    _require(type(value) is dict, f"{label} root must be exact object")
    return value


def _write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(
        not path.exists() and not path.is_symlink(),
        f"refusing to overwrite output: {path}",
    )
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise RadaCurrentReplayMaterializationError(
            f"cannot publish output: {path}"
        ) from exc


def materialize(
    *,
    repository_root: Path,
    historical_qp_root: Path,
    retained_root: Path,
    candidate_output: Path,
    summary_output: Path,
) -> dict[str, Any]:
    normalizer = _load_module(
        repository_root / "tools/normalize_d03_rada_bulk_html.py",
        "rada_current_exact_normalizer",
        expected_blob=NORMALIZER_GIT_BLOB_SHA1,
    )
    qp = _load_module(
        historical_qp_root / "tools/filter_d03_rada_bulk_quality_privacy.py",
        "rada_current_exact_qp",
        expected_blob=QP_TOOL_GIT_BLOB_SHA1,
    )
    normalization_config = _load_config(
        repository_root / "configs/data/d03_rada_bulk_normalization_v1.json",
        expected_blob=NORMALIZATION_CONFIG_GIT_BLOB_SHA1,
        label="normalization config",
    )
    qp_config = _load_config(
        historical_qp_root / "configs/data/d03_rada_bulk_quality_privacy_v1.json",
        expected_blob=QP_CONFIG_GIT_BLOB_SHA1,
        label="historical Q/P config",
    )

    source = _read_regular_bytes(
        retained_root / "rada-source-a.zip",
        label="retained Rada source ZIP",
    )
    _require(len(source) == SOURCE_ARCHIVE_BYTES, "source archive byte drift")
    _require(_sha256(source) == SOURCE_ARCHIVE_SHA256, "source archive SHA drift")

    probe_a = _load_json_with_transport(
        retained_root / "rada-probe-a.json",
        label="retained probe A",
        expected_sha256=PROBE_TRANSPORT_SHA256,
    )
    probe_b = _load_json_with_transport(
        retained_root / "rada-probe-b.json",
        label="retained probe B",
        expected_sha256=PROBE_TRANSPORT_SHA256,
    )
    _require(probe_a == probe_b, "retained clean probes differ")
    qualification = _load_json_with_transport(
        retained_root / "rada-fresh-snapshot-qualification-v2.json",
        label="retained qualification",
        expected_sha256=QUALIFICATION_TRANSPORT_SHA256,
    )
    pin = _load_json_with_transport(
        retained_root / "rada-successor-pin-v2.json",
        label="retained successor pin",
        expected_sha256=PIN_TRANSPORT_SHA256,
    )
    _require(
        qualification.get("evidence_identity_sha256")
        == QUALIFICATION_IDENTITY_SHA256,
        "qualification identity drift",
    )
    _require(
        qualification.get("normalization_reexecution_required") is True,
        "normalization replay boundary drift",
    )
    _require(
        qualification.get("quality_privacy_reexecution_required") is True,
        "Q/P replay boundary drift",
    )
    _require(
        pin.get("pin_identity_sha256") == PIN_IDENTITY_SHA256,
        "successor pin identity drift",
    )
    _require(pin.get("purpose") == "NORMALIZATION_INPUT_ONLY", "pin purpose widened")
    inventory = probe_a.get("inventory")
    _require(type(inventory) is dict, "retained probe inventory missing")
    _require(
        inventory.get("entry_identity_sha256") == ENTRY_IDENTITY_SHA256,
        "entry identity drift",
    )
    _require(inventory.get("canonical_entry_count") == 3_055, "entry count drift")
    _require(
        inventory.get("canonical_raw_bytes") == 353_891_824,
        "canonical raw byte total drift",
    )

    projected_probe = copy.deepcopy(probe_a)
    projected_probe["safe_result"] = normalizer.PINNED_PROBE_RESULT
    projected_probe["gates"]["exact_archive_identity"] = normalizer.PINNED_PROBE_GATE
    projected_probe_sha = _sha256(
        normalizer._serialized_probe_report_bytes(projected_probe)
    )
    normalized_jsonl, manifest = normalizer._materialize_normalized_records_unbound(
        source,
        projected_probe,
        normalization_config,
        probe_report_sha256=projected_probe_sha,
    )
    _require(
        manifest.get("safe_result") == normalizer.UNBOUND_SAFE_RESULT,
        "normalization projection widened authority",
    )

    projected_manifest = copy.deepcopy(manifest)
    projected_manifest["safe_result"] = qp.PARENT_SAFE_RESULT
    projected_manifest.pop("manifest_identity_sha256", None)
    projected_manifest["manifest_identity_sha256"] = _sha256(
        qp._canonical_bytes(projected_manifest)
    )
    normalization = projected_manifest.get("normalization")
    parent_probe = projected_manifest.get("parent_probe")
    _require(type(normalization) is dict, "normalization manifest section missing")
    _require(type(parent_probe) is dict, "parent probe section missing")
    binding = {
        "pr": 2478,
        "head_sha": "9fa66e11faf3710d6ddf75854381eb879a37d1ba",
        "branch": "swarm/2477-rada-laws-fresh-snapshot-v2",
        "execution_head_sha": "9fa66e11faf3710d6ddf75854381eb879a37d1ba",
        "execution_run_id": 37136885886,
        "execution_evidence_identity_sha256": QUALIFICATION_IDENTITY_SHA256,
        "manifest_schema": qp.PARENT_MANIFEST_SCHEMA,
        "manifest_worker_id": qp.PARENT_WORKER_ID,
        "source_family": qp.SOURCE_FAMILY,
        "safe_result": qp.PARENT_SAFE_RESULT,
        "manifest_identity_sha256": projected_manifest[
            "manifest_identity_sha256"
        ],
        "manifest_transport_sha256": _sha256(
            qp._canonical_bytes(projected_manifest)
        ),
        "jsonl_sha256": normalization["jsonl_sha256"],
        "record_count": normalization["record_count"],
        "nonempty_record_count": normalization["nonempty_record_count"],
        "normalized_bytes_observed_not_credited": normalization[
            "normalized_bytes_observed_not_credited"
        ],
        "normalized_record_inventory_sha256": normalization[
            "normalized_record_inventory_sha256"
        ],
        "source_encoding_counts": normalization["source_encoding_counts"],
        "pinned_probe_report_sha256": parent_probe["probe_report_sha256"],
        "archive_sha256": parent_probe["archive_sha256"],
        "entry_identity_sha256": parent_probe["entry_identity_sha256"],
    }
    projected_qp_config = copy.deepcopy(qp_config)
    projected_qp_config["parent_normalization"] = binding
    qp.EXPECTED_PARENT_BINDING = binding
    accepted_jsonl, qp_report = qp._materialize_quality_privacy_candidate_for_test(
        normalized_jsonl,
        projected_manifest,
        projected_qp_config,
        parent_manifest_sha256=binding["manifest_transport_sha256"],
    )

    filter_result = qp_report.get("filter_result")
    qp_input = qp_report.get("input")
    _require(type(filter_result) is dict, "Q/P filter result missing")
    _require(type(qp_input) is dict, "Q/P input section missing")
    _require(
        normalization.get("record_count") == EXPECTED_NORMALIZED_RECORDS
        and normalization.get("nonempty_record_count")
        == EXPECTED_NORMALIZED_RECORDS,
        "normalized record count drift",
    )
    _require(
        normalization.get("normalized_bytes_observed_not_credited")
        == EXPECTED_NORMALIZED_PAYLOAD_BYTES,
        "normalized payload bytes drift",
    )
    _require(
        len(normalized_jsonl) == EXPECTED_NORMALIZED_JSONL_BYTES
        and _sha256(normalized_jsonl) == EXPECTED_NORMALIZED_JSONL_SHA256,
        "normalized JSONL identity drift",
    )
    _require(
        filter_result.get("accepted_chunk_count") == EXPECTED_ACCEPTED_CHUNKS,
        "accepted chunk count drift",
    )
    _require(
        filter_result.get("accepted_bytes_observed_not_credited")
        == EXPECTED_ACCEPTED_PAYLOAD_BYTES,
        "accepted payload byte total drift",
    )
    _require(
        len(accepted_jsonl) == EXPECTED_ACCEPTED_JSONL_BYTES
        and _sha256(accepted_jsonl) == EXPECTED_ACCEPTED_JSONL_SHA256,
        "accepted JSONL identity drift",
    )
    _require(
        filter_result.get("accepted_inventory_sha256")
        == EXPECTED_ACCEPTED_INVENTORY_SHA256,
        "accepted inventory identity drift",
    )
    _require(
        filter_result.get("exact_duplicate_accepted_hashes_observed_not_removed")
        == EXPECTED_EXACT_DUPLICATE_HASHES,
        "accepted duplicate observation drift",
    )
    _require(
        qp_input.get("zero_chunk_parent_count") == 0,
        "zero-chunk parent count drift",
    )

    summary_core: dict[str, Any] = {
        "schema_version": "12-6.d03-rada-current-replay-materialization.v1",
        "status": "PASS_EXACT_REPLAY_ZERO_CREDIT",
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_archive_bytes": SOURCE_ARCHIVE_BYTES,
        "normalized_record_count": EXPECTED_NORMALIZED_RECORDS,
        "normalized_payload_utf8_bytes": EXPECTED_NORMALIZED_PAYLOAD_BYTES,
        "normalized_jsonl_file_bytes": EXPECTED_NORMALIZED_JSONL_BYTES,
        "normalized_jsonl_sha256": EXPECTED_NORMALIZED_JSONL_SHA256,
        "accepted_chunk_count": EXPECTED_ACCEPTED_CHUNKS,
        "accepted_payload_utf8_bytes": EXPECTED_ACCEPTED_PAYLOAD_BYTES,
        "accepted_jsonl_file_bytes": EXPECTED_ACCEPTED_JSONL_BYTES,
        "accepted_jsonl_sha256": EXPECTED_ACCEPTED_JSONL_SHA256,
        "accepted_inventory_sha256": EXPECTED_ACCEPTED_INVENTORY_SHA256,
        "exact_duplicate_accepted_hashes_observed_not_removed": (
            EXPECTED_EXACT_DUPLICATE_HASHES
        ),
        "candidate_output_is_ephemeral": True,
        "raw_text_emitted_to_summary": False,
        "production_qp_authority_established": False,
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }
    summary = {
        **summary_core,
        "summary_identity_sha256": _sha256(
            json.dumps(
                summary_core,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ),
    }
    summary_payload = (
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    _require(
        candidate_output.resolve(strict=False)
        != summary_output.resolve(strict=False),
        "candidate and summary outputs must be distinct",
    )
    _write_create_only(candidate_output, accepted_jsonl)
    try:
        _write_create_only(summary_output, summary_payload)
    except Exception:
        candidate_output.unlink(missing_ok=True)
        raise
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--historical-qp-root", type=Path, required=True)
    parser.add_argument("--retained-root", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = materialize(
            repository_root=args.repository_root,
            historical_qp_root=args.historical_qp_root,
            retained_root=args.retained_root,
            candidate_output=args.candidate_output,
            summary_output=args.summary_output,
        )
    except (RadaCurrentReplayMaterializationError, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print("D03_RADA_CURRENT_REPLAY_MATERIALIZATION=PASS_ZERO_CREDIT")
    print("SUMMARY_IDENTITY_SHA256=" + summary["summary_identity_sha256"])
    print("CANDIDATE_SHA256=" + EXPECTED_ACCEPTED_JSONL_SHA256)
    print("CANONICAL_CAPACITY_CREDITED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
